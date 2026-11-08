"""Nạp PaySim và tạo đặc trưng lịch sử giao dịch (không dùng nhãn, không dùng số dư)."""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_FILENAMES, FEATURES, HISTORY_FEATURES, PROJECT_ROOT, REQUIRED_COLUMNS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Băm SHA-256 theo luồng để không nạp cả file lớn vào RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_dataset(explicit: Path | None = None) -> Path:
    """Tìm file PaySim: tham số --data, biến môi trường PAYSIM_CSV, rồi các thư mục quen thuộc của dự án."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PAYSIM_CSV"):
        candidates.append(Path(os.environ["PAYSIM_CSV"]))
    for folder in [PROJECT_ROOT / "datasets" / "paysim", PROJECT_ROOT / "datasets",
                   PROJECT_ROOT.parent / "datasets" / "paysim", PROJECT_ROOT.parent / "datasets",
                   PROJECT_ROOT / "Tabular-Foudation" / "datasets" / "paysim",
                   PROJECT_ROOT / "Tabular-Foudation" / "datasets",
                   PROJECT_ROOT / "code_paysim" / "data", Path.cwd() / "datasets" / "paysim",
                   Path.cwd() / "datasets", Path.cwd()]:
        candidates.extend(folder / name for name in DATASET_FILENAMES)
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():  # chạy trên Kaggle: dataset được gắn vào /kaggle/input
        for name in DATASET_FILENAMES:
            candidates.extend(kaggle.glob(f"**/{name}"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Không tìm thấy PaySim CSV. Dùng --data <đường dẫn> hoặc đặt "
        f"{DATASET_FILENAMES[0]} vào datasets/ (hoặc Tabular-Foudation/datasets/)."
    )


def make_prior_activity(data: pd.DataFrame, account_column: str, prefix: str) -> pd.DataFrame:
    """Với mỗi (tài khoản, step): số giao dịch và tổng tiền của tài khoản đó ở các step NHỎ HƠN step hiện tại.

    Gộp theo từng step rồi trừ chính step hiện tại, vì PaySim không cho biết thứ tự bên trong một giờ.
    Nhờ vậy mọi giao dịch cùng step không nhìn thấy nhau.
    """
    per_step = (
        data.groupby([account_column, "step"], observed=True, sort=False)
        .agg(attempt_count=("amount", "size"), attempted_amount_sum=("amount", "sum"))
        .reset_index()
        .sort_values([account_column, "step"], kind="mergesort")
    )
    groups = per_step.groupby(account_column, observed=True, sort=False)
    per_step[f"{prefix}_prior_attempt_count"] = groups["attempt_count"].cumsum() - per_step["attempt_count"]
    per_step[f"{prefix}_prior_attempted_amount_sum"] = (
        groups["attempted_amount_sum"].cumsum() - per_step["attempted_amount_sum"]
    )
    return per_step[[account_column, "step", f"{prefix}_prior_attempt_count",
                     f"{prefix}_prior_attempted_amount_sum"]]


def build_transfer_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Từ bảng giao dịch thô (đã có cột _source_row) tạo bảng TRANSFER kèm 4 đặc trưng lịch sử."""
    sender = make_prior_activity(raw, "nameOrig", "sender")        # lịch sử tính trên MỌI loại giao dịch
    recipient = make_prior_activity(raw, "nameDest", "recipient")
    targets = raw.loc[raw["type"].eq("TRANSFER")].copy()           # mục tiêu dự đoán chỉ là TRANSFER
    targets = targets.merge(sender, on=["nameOrig", "step"], how="left", validate="many_to_one")
    targets = targets.merge(recipient, on=["nameDest", "step"], how="left", validate="many_to_one")
    targets[HISTORY_FEATURES] = targets[HISTORY_FEATURES].fillna(0)
    return targets.drop(columns=["type", "nameOrig", "nameDest"]).reset_index(drop=True)


def load_paysim(path: Path):
    """Đọc CSV, kiểm tra, tạo đặc trưng. Trả về (frame, X, y, meta)."""
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(REQUIRED_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"Thiếu cột PaySim bắt buộc: {missing}")
    sha = sha256_file(path)
    raw = pd.read_csv(
        path, usecols=REQUIRED_COLUMNS,  # không nạp các cột số dư và isFlaggedFraud
        dtype={"step": "int16", "type": "category", "amount": "float64",
               "nameOrig": "category", "nameDest": "category", "isFraud": "int8"},
    )
    source_rows = len(raw)
    raw.insert(0, "_source_row", np.arange(source_rows, dtype=np.int64))
    if raw[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Có giá trị thiếu trong các cột phân tích.")
    if not set(raw["isFraud"].unique()).issubset({0, 1}):
        raise ValueError("isFraud cần được mã hóa 0/1.")
    if not np.isfinite(raw[["step", "amount"]].to_numpy(dtype=float)).all():
        raise ValueError("step/amount có giá trị không hữu hạn.")
    frame = build_transfer_frame(raw)
    del raw
    gc.collect()
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame["isFraud"].to_numpy(dtype=np.int8)
    if frame.empty or np.unique(y).size != 2:
        raise ValueError("Dữ liệu TRANSFER cần có cả giao dịch hợp lệ và gian lận.")
    if not np.isfinite(X).all():
        raise ValueError("Đặc trưng có giá trị không hữu hạn.")
    meta = {"dataset_path": str(path), "dataset_sha256": sha, "source_rows": source_rows,
            "transfer_rows": int(len(frame))}
    return frame, X, y, meta
