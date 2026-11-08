"""Chia theo thời gian, lấy mẫu đánh giá cố định và mẫu train phân tầng dùng chung cho mọi mô hình."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

from .config import EVAL_SAMPLE_SEED, TRAIN_FRACTION, VALIDATION_FRACTION


def hash_indices(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def sample_eval_indices(indices, max_rows, seed):
    """Lấy ngẫu nhiên (không phân tầng) tối đa max_rows dòng; max_rows=None thì lấy hết."""
    if max_rows is None or len(indices) <= max_rows:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_rows, replace=False))


def temporal_split(steps: np.ndarray, y: np.ndarray, max_eval_rows, dataset_sha256: str):
    """60% step đầu = train pool, 20% kế = validation, 20% cuối = test. Không xáo trộn thời gian."""
    unique_steps = np.sort(np.unique(steps))
    if len(unique_steps) < 5:
        raise ValueError("Cần tối thiểu 5 giá trị step để chia theo thời gian.")
    n_train = int(len(unique_steps) * TRAIN_FRACTION)
    n_val = int(len(unique_steps) * VALIDATION_FRACTION)
    n_train = min(max(n_train, 1), len(unique_steps) - 2)
    n_val = min(max(n_val, n_train + 1), len(unique_steps) - 1)
    train_end, val_end = float(unique_steps[n_train - 1]), float(unique_steps[n_val - 1])
    train_pool = np.flatnonzero(steps <= train_end)
    val_period = np.flatnonzero((steps > train_end) & (steps <= val_end))
    test_period = np.flatnonzero(steps > val_end)
    val_idx = sample_eval_indices(val_period, max_eval_rows, EVAL_SAMPLE_SEED)
    test_idx = sample_eval_indices(test_period, max_eval_rows, EVAL_SAMPLE_SEED + 1)
    if any(np.unique(y[idx]).size != 2 for idx in (train_pool, val_idx, test_idx)):
        raise ValueError("Train/validation/test cần có cả hai lớp.")
    config = {
        "dataset_sha256": dataset_sha256,
        "strategy": "chronological by unique step; first 60% train, next 20% validation, final 20% test",
        "train_step_end": train_end, "validation_step_end": val_end,
        "train_pool_rows": int(len(train_pool)), "validation_period_rows": int(len(val_period)),
        "test_period_rows": int(len(test_period)), "validation_eval_rows": int(len(val_idx)),
        "test_eval_rows": int(len(test_idx)), "validation_prevalence": float(y[val_idx].mean()),
        "test_prevalence": float(y[test_idx].mean()),
        "validation_indices_sha256": hash_indices(val_idx), "test_indices_sha256": hash_indices(test_idx),
        "max_eval_rows_per_split": max_eval_rows, "eval_sample_seed": EVAL_SAMPLE_SEED,
    }
    return train_pool, val_idx, test_idx, config


def build_train_subsets(train_pool, y, seeds, train_sizes):
    """Mẫu train phân tầng theo lớp lấy từ train pool; mỗi (seed, cỡ) có đúng cùng chỉ số cho mọi mô hình."""
    subsets = {}
    for seed in seeds:
        subsets[str(seed)] = {}
        for n_train in train_sizes:
            if n_train > len(train_pool):
                raise ValueError(f"Train size {n_train} lớn hơn train pool {len(train_pool)}.")
            if n_train < 2:
                raise ValueError("Mỗi train subset cần tối thiểu hai mẫu.")
            chosen, _ = train_test_split(train_pool, train_size=n_train, stratify=y[train_pool], random_state=seed)
            if np.unique(y[chosen]).size != 2:
                raise ValueError(f"Subset seed={seed}, n={n_train} không có đủ hai lớp.")
            subsets[str(seed)][str(n_train)] = np.sort(chosen).astype(int).tolist()
    return subsets


def check_no_leakage(subsets, val_idx, test_idx, train_pool, steps, config):
    """Kiểm tra tự động: không dòng nào của validation/test nằm trong train; train chỉ từ giai đoạn đầu."""
    val_set, test_set = set(map(int, val_idx)), set(map(int, test_idx))
    if val_set & test_set:
        raise AssertionError("Validation và test có dòng chung.")
    pool = set(map(int, train_pool))
    for seed, by_size in subsets.items():
        for size, idx in by_size.items():
            idx_set = set(idx)
            if not idx_set <= pool:
                raise AssertionError(f"Train subset seed={seed}, n={size} có dòng ngoài train pool.")
            if idx_set & (val_set | test_set):
                raise AssertionError(f"Rò rỉ: train seed={seed}, n={size} trùng validation/test.")
            if steps[list(idx)].max() > config["train_step_end"]:
                raise AssertionError("Train chứa step sau ranh giới train.")
    if steps[list(val_set)].min() <= config["train_step_end"] or steps[list(test_set)].min() <= config["validation_step_end"]:
        raise AssertionError("Ranh giới thời gian bị vi phạm.")
