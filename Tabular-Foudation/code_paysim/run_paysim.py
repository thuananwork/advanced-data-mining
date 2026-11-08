"""Chạy toàn bộ thí nghiệm PaySim bằng một lệnh.

Ví dụ:
  python run_paysim.py --smoke --skip-tabicl         # thử nhanh, không cần TabICLv2
  python run_paysim.py --data D:\\path\\PS_...log.csv  # chạy đầy đủ 48 lượt (cần tabicl + mạng lần đầu)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paysim import config as C
from paysim.data import find_dataset, load_paysim
from paysim.experiment import blend_ensemble, run_grid, summarize, tune_lightgbm
from paysim.report import write_bundle
from paysim.splits import build_train_subsets, check_no_leakage, temporal_split


def parse_args():
    p = argparse.ArgumentParser(description="PaySim: LR, CatBoost, LightGBM, TabICLv2 và ensemble")
    p.add_argument("--data", type=Path, help="đường dẫn PaySim CSV (mặc định: tự tìm)")
    p.add_argument("--out", type=Path, default=C.Settings().out_dir, help="thư mục ghi kết quả")
    p.add_argument("--smoke", action="store_true", help="chạy thử: 1 seed, 2 cỡ train, 5.000 dòng đánh giá")
    p.add_argument("--skip-tabicl", action="store_true", help="bỏ TabICLv2 (và ensemble)")
    p.add_argument("--resume", action="store_true", help="bỏ qua các lượt đã chạy xong")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--train-sizes", type=int, nargs="+")
    p.add_argument("--max-eval-rows", type=int, help="0 = dùng toàn bộ giai đoạn đánh giá")
    return p.parse_args()


def main():
    a = parse_args()
    s = C.Settings(out_dir=a.out, resume=a.resume)
    if a.smoke:
        s.smoke()
    if a.seeds: s.seeds = a.seeds
    if a.train_sizes: s.train_sizes = a.train_sizes
    if a.max_eval_rows is not None: s.max_eval_rows = a.max_eval_rows or None
    if a.skip_tabicl: s.models = [m for m in s.models if m != "TabICLv2"]
    out = s.out_dir; out.mkdir(parents=True, exist_ok=True)

    path = find_dataset(a.data)
    print("Dữ liệu:", path)
    frame, X, y, meta = load_paysim(path)
    print(f"TRANSFER: {len(frame):,} dòng, gian lận {int(y.sum()):,} ({y.mean():.2%})")
    steps = frame["step"].to_numpy()
    pool, val_idx, test_idx, split = temporal_split(steps, y, s.max_eval_rows, meta["dataset_sha256"])
    subsets = build_train_subsets(pool, y, s.seeds, s.train_sizes)
    check_no_leakage(subsets, val_idx, test_idx, pool, steps, split)
    print("Kiểm tra rò rỉ: OK")
    meta = {**meta, **split}
    (out / "split_config.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (out / "train_subsets.json").write_text(json.dumps(subsets), encoding="utf-8")

    lgbm_params, tuning_info = C.lightgbm_params(), None
    if "LightGBM" in s.models:
        tuning, lgbm_params, tuning_info = tune_lightgbm(X, y, subsets, val_idx, s)
        tuning.to_csv(out / "lightgbm_tuning.csv", index=False)
        print("LightGBM chọn:", {k: lgbm_params[k] for k in ("num_leaves", "min_child_samples")})
    cfg = {"features": C.FEATURES, "seeds": s.seeds, "train_sizes": s.train_sizes, "models": s.models,
           "logistic_params": C.LOGISTIC_PARAMS, "catboost_params": C.CATBOOST_PARAMS,
           "lightgbm_params": lgbm_params, "tabicl_params": C.tabicl_params(),
           "lightgbm_tuning": tuning_info, "excluded_fields": ["isFraud", "isFlaggedFraud", "balance columns", "account IDs"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    runs = run_grid(X, y, subsets, val_idx, test_idx, lgbm_params, meta, s, out / "runs.csv", out / "predictions.csv")
    paired, summary = summarize(runs, s.models)
    if not summary.empty:
        summary.to_csv(out / "summary.csv", index=False)
        print(summary[["model", "train_size", "test_average_precision_mean", "test_roc_auc_mean", "test_f1_mean"]].round(4).to_string(index=False))
    if set(C.ENSEMBLE_MODELS) <= set(s.models) and not paired.empty:
        ens, es, alpha = blend_ensemble(paired, out / "predictions.csv")
        ens.to_csv(out / "ensemble_summary.csv", index=False)
        es.to_csv(out / "ensemble_metrics_summary.csv", index=False)
        print(f"Ensemble: alpha TabICLv2 = {alpha:.2f}")
        print(es[["train_size", "test_average_precision_mean", "test_f1_mean"]].round(4).to_string(index=False))
    zpath, ok, bad = write_bundle(out, meta, s)
    print(f"Hoàn tất: {ok}/{s.expected_runs} thành công, {bad} lỗi. Kết quả: {out}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
