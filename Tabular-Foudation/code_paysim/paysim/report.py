"""Ghi gói báo cáo (README, manifest, ZIP) giống bundle Kaggle; không đưa CSV PaySim gốc vào."""
from __future__ import annotations

import json
import time
import zipfile

import pandas as pd

from .data import sha256_file

ARTIFACTS = ["runs.csv", "predictions.csv", "summary.csv", "ensemble_summary.csv",
             "ensemble_metrics_summary.csv", "lightgbm_tuning.csv", "config.json",
             "split_config.json", "train_subsets.json"]


def write_bundle(out, meta, settings):
    runs = pd.read_csv(out / "runs.csv")
    ok, bad = int(runs["status"].eq("ok").sum()), int(runs["status"].ne("ok").sum())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    files = [out / n for n in ARTIFACTS if (out / n).is_file()]
    payload = {
        "experiment": "PaySim LightGBM challenger and LightGBM+TabICLv2 validation-selected blend",
        "generated_utc": now, "dataset_sha256": meta["dataset_sha256"],
        "completed_runs": ok, "failed_runs": bad, "expected_runs": settings.expected_runs,
        "included_artifacts": [p.name for p in files], "raw_dataset_included": False,
        "interpretation_caveats": [
            "PaySim là dữ liệu mô phỏng, không phải bằng chứng về ngân hàng thật.",
            "Validation/test là mẫu theo thời gian; tần suất gian lận khác nhau giữa các giai đoạn.",
            "Tham số LightGBM, trọng số ensemble và ngưỡng chọn trên validation; test chỉ để đánh giá cuối.",
            "Nếu completed_runs < expected_runs thì bộ kết quả chưa đầy đủ.",
        ]}
    if (out / "summary.csv").is_file():
        payload["single_model_summary"] = json.loads(pd.read_csv(out / "summary.csv").to_json(orient="records"))
    if (out / "ensemble_metrics_summary.csv").is_file():
        payload["ensemble_summary"] = json.loads(pd.read_csv(out / "ensemble_metrics_summary.csv").to_json(orient="records"))
    (out / "report_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    extra = [out / "report_summary.json"]
    manifest = {"bundle": "paysim_fraud_report_bundle.zip", "created_utc": now,
                "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files + extra]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zpath = out / "paysim_fraud_report_bundle.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + extra + [out / "manifest.json"]:
            z.write(p, arcname=p.name)
    return zpath, ok, bad
