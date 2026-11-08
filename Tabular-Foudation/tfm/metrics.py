"""Các chỉ số đánh giá cho phân loại nhị phân (xác suất lớp 1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, roc_auc_score)


def ece_score(y, p1, bins: int = 15) -> float:
    """Expected Calibration Error theo độ tin cậy của lớp được dự đoán."""
    y = np.asarray(y); p1 = np.asarray(p1, dtype=float)
    pred = (p1 >= 0.5).astype(int)
    conf = np.where(pred == 1, p1, 1 - p1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(y, p1) -> float:
    return float(np.mean((np.asarray(p1, dtype=float) - np.asarray(y)) ** 2))


def all_metrics(y, p1) -> dict:
    y = np.asarray(y); p1 = np.clip(np.asarray(p1, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p1 >= 0.5).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, p1)) if both else float("nan"),
        "avg_precision": float(average_precision_score(y, p1)) if both else float("nan"),
        "log_loss": float(log_loss(y, np.c_[1 - p1, p1], labels=[0, 1])),
        "ece": ece_score(y, p1),
        "brier": brier(y, p1),
    }


def fit_temperature(y, p1) -> float:
    """Temperature scaling: tìm T để log-loss trên tập validation nhỏ nhất."""
    from scipy.optimize import minimize_scalar
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-7, 1 - 1e-7)
    z = np.log(p1 / (1 - p1))
    y = np.asarray(y)

    def nll(t):
        q = 1 / (1 + np.exp(-z / t))
        return log_loss(y, np.clip(q, 1e-7, 1 - 1e-7), labels=[0, 1])
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(p1, t: float):
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-np.log(p1 / (1 - p1)) / t))
