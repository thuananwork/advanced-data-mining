"""Nạp dữ liệu. Mọi bộ dữ liệu trả về (X, y, groups, info). groups gom các dòng trùng nhau
để không bao giờ có cùng một vector đặc trưng ở cả train và test (lỗi của Banknote)."""
from __future__ import annotations

import hashlib
import io
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .paths import DATASETS


@dataclass
class DatasetSpec:
    name: str
    file: str
    drop: list = field(default_factory=list)
    target: str = "target"
    zero_as_missing: list = field(default_factory=list)


SPECS = {
    "breast_cancer": DatasetSpec("breast_cancer", "breast_cancer/breast_cancer_wisconsin.csv", ["row_id"]),
    "banknote": DatasetSpec("banknote", "banknote/banknote_authentication.csv", ["row_id"]),
    "wine": DatasetSpec("wine", "wine/wine_quality_white_binary.csv", ["row_id", "quality"]),
    "magic": DatasetSpec("magic", "magic/magic_gamma_telescope.csv", ["row_id", "class"]),
    "pima": DatasetSpec("pima", "pima/pima_diabetes.csv", [], "target",
                        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]),
}
PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
             "DiabetesPedigreeFunction", "Age", "target"]
PIMA_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
]


def _download_pima(path):
    """Tải Pima: thử OpenML (qua scikit-learn), rồi các URL dự phòng. Giữ nguyên số 0 gốc."""
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name="diabetes", version=1, as_frame=True)
        f = d.frame.copy()
        f.columns = PIMA_COLS
        f["target"] = (f["target"].astype(str) == "tested_positive").astype(int)
    except Exception as e1:  # noqa: BLE001
        f = None
        for url in PIMA_URLS:
            try:
                raw = urllib.request.urlopen(url, timeout=60).read()
                f = pd.read_csv(io.BytesIO(raw), header=None)
                if str(f.iloc[0, 0]).lower().startswith("preg"):
                    f = pd.read_csv(io.BytesIO(raw))
                f.columns = PIMA_COLS
                break
            except Exception:  # noqa: BLE001
                continue
        if f is None:
            raise FileNotFoundError(
                f"Không tải được Pima ({e1!r}). Tải tay file CSV 768 dòng, 9 cột "
                f"{PIMA_COLS} và lưu thành {path}") from e1
    assert f.shape == (768, 9), f.shape
    f.to_csv(path, index=False)


def load_dataset(name: str):
    spec = SPECS[name]
    path = DATASETS / spec.file
    if not path.exists():
        if name == "pima":
            DATASETS.mkdir(parents=True, exist_ok=True)
            _download_pima(path)
        else:
            raise FileNotFoundError(f"Thiếu {path}")
    frame = pd.read_csv(path)
    y = frame[spec.target].to_numpy(dtype=np.int64)
    X = frame.drop(columns=[c for c in spec.drop + [spec.target] if c in frame.columns])
    for c in spec.zero_as_missing:           # Pima: số 0 nghĩa là thiếu
        X.loc[X[c] == 0, c] = np.nan
    cols = list(X.columns)
    Xv = X.to_numpy(dtype=np.float32)
    keys = [hashlib.md5(np.round(np.nan_to_num(r, nan=-9e9), 6).tobytes()).hexdigest() for r in Xv]
    groups = pd.factorize(pd.Series(keys))[0]
    info = {"name": name, "n": len(y), "n_features": Xv.shape[1], "features": cols,
            "positive_rate": float(y.mean()), "n_duplicate_rows": int(len(y) - len(set(keys))),
            "n_missing": int(np.isnan(Xv).sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return Xv, y, groups, info
