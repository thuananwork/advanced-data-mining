"""Nhà máy mô hình. Mỗi mô hình có fit(X, y) và predict_proba(X)[:, 1]. Thiếu thư viện thì ném ModelUnavailable."""
from __future__ import annotations

import inspect
import os
import time

import numpy as np

TABICL_CKPT = "tabicl-classifier-v2-20260212.ckpt"
ALL_MODELS = ["TabICLv2", "TabPFNv2", "TabDPT", "XGBoost", "CatBoost", "LightGBM",
              "RandomForest", "LogisticRegression"]
FOUNDATION = {"TabICLv2", "TabPFNv2", "TabDPT"}


class ModelUnavailable(RuntimeError):
    pass


def threads() -> int:
    return min(4, os.cpu_count() or 2)


class _Median:
    """Điền thiếu bằng trung vị rồi chuyển cho mô hình trong (cho mô hình không xử lý NaN)."""
    def __init__(self, model, scale=False):
        self.model, self.scale = model, scale

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.imp = SimpleImputer(strategy="median").fit(X)
        Xi = self.imp.transform(X)
        self.sc = StandardScaler().fit(Xi) if self.scale else None
        self.model.fit(self.sc.transform(Xi) if self.sc else Xi, y)
        return self

    def predict_proba(self, X):
        Xi = self.imp.transform(X)
        return self.model.predict_proba(self.sc.transform(Xi) if self.sc else Xi)


class _TabDPT:
    def __init__(self, seed, n_ctx=1024, retrieval=True):
        self.seed, self.n_ctx, self.retrieval = seed, n_ctx, retrieval

    def fit(self, X, y):
        try:
            from tabdpt import TabDPTClassifier
        except ImportError as e:
            raise ModelUnavailable("chưa cài tabdpt (pip install 'tabdpt==1.1.*')") from e
        from sklearn.impute import SimpleImputer
        self.imp = SimpleImputer(strategy="median").fit(X)
        sig = inspect.signature(TabDPTClassifier).parameters
        kw = {k: v for k, v in {"device": "cpu", "compile": False, "use_flash_attention": False}.items() if k in sig}
        self.m = TabDPTClassifier(**kw)
        self.m.fit(self.imp.transform(X), y)
        self.n = len(y)
        return self

    def predict_proba(self, X):
        sig = inspect.signature(self.m.predict_proba).parameters
        opts = {"context_size": min(self.n, self.n_ctx), "n_ensembles": 1, "seed": self.seed,
                "use_retrieval": self.retrieval, "temperature": 1.0, "permute_classes": False}
        return self.m.predict_proba(self.imp.transform(X), **{k: v for k, v in opts.items() if k in sig})


def build(name: str, seed: int):
    t = threads()
    if name == "TabICLv2":
        try:
            import torch
            from tabicl import TabICLClassifier
        except ImportError as e:
            raise ModelUnavailable("chưa cài tabicl (pip install 'tabicl>=2.0')") from e
        torch.set_num_threads(t)
        return TabICLClassifier(device="cpu", n_estimators=1, batch_size=1, use_amp=False, use_fa3=False,
                                offload_mode="auto", checkpoint_version=TABICL_CKPT,
                                n_jobs=t, random_state=seed)
    if name == "TabPFNv2":
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as e:
            raise ModelUnavailable("chưa cài tabpfn (pip install tabpfn)") from e
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=1, random_state=seed,
            fit_mode="low_memory", show_progress_bar=False)
    if name == "TabDPT":
        return _TabDPT(seed)
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ModelUnavailable("chưa cài xgboost") from e
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
                             colsample_bytree=0.9, n_jobs=t, random_state=seed, eval_metric="logloss")
    if name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ModelUnavailable("chưa cài catboost") from e
        return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=seed,
                                  thread_count=t, verbose=False, allow_writing_files=False)
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ModelUnavailable("chưa cài lightgbm") from e
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                              subsample=0.9, colsample_bytree=0.9, n_jobs=t, random_state=seed, verbose=-1)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return _Median(RandomForestClassifier(n_estimators=300, n_jobs=t, random_state=seed))
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return _Median(LogisticRegression(max_iter=2000), scale=True)
    raise ValueError(f"Mô hình không hỗ trợ: {name}")


def fit_predict(name: str, seed: int, Xtr, ytr, *Xs):
    """Huấn luyện và trả (danh sách xác suất lớp 1 cho từng Xs, fit_sec, predict_sec)."""
    m = build(name, seed)
    t0 = time.perf_counter(); m.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    outs = []
    for X in Xs:
        p = np.asarray(m.predict_proba(X))
        classes = getattr(m, "classes_", None)
        col = 1
        if classes is not None and 1 in list(classes):
            col = list(classes).index(1)
        outs.append(p[:, col] if p.ndim == 2 else p)
    return outs, fit_s, time.perf_counter() - t0
