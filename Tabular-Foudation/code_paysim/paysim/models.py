"""Tạo mô hình, dự đoán xác suất lớp gian lận và chọn ngưỡng trên validation."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def build_model(model_id: str, seed: int, lightgbm_params: dict | None = None):
    if model_id == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(**C.LOGISTIC_PARAMS, random_state=seed))
    if model_id == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**C.CATBOOST_PARAMS, random_seed=seed, thread_count=C._n_jobs())
    if model_id == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**(lightgbm_params or C.lightgbm_params()), random_state=seed)
    if model_id == "TabICLv2":
        from tabicl import TabICLClassifier  # cần tải checkpoint từ Hugging Face ở lần chạy đầu
        return TabICLClassifier(**C.tabicl_params(), random_state=seed)
    raise ValueError(f"Mô hình không được hỗ trợ: {model_id}")


def predict_positive_probability(model, x_eval, batch_size: int = 2048) -> np.ndarray:
    """Xác suất lớp 1, dự đoán theo lô để TabICLv2 không tràn RAM."""
    positions = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positions) != 1:
        raise ValueError(f"Không tìm thấy duy nhất lớp dương 1 trong classes_={model.classes_}.")
    pos = int(positions[0])
    pieces = [np.asarray(model.predict_proba(x_eval[s:s + batch_size]))[:, pos]
              for s in range(0, len(x_eval), batch_size)]
    return np.concatenate(pieces)


def threshold_from_validation(y_true, probability):
    """Ngưỡng làm F1 lớn nhất trên đường precision-recall của validation (không dùng test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])
