"""Chia train/val/test theo seed, tách theo nhóm (dòng trùng nhau cùng một nhóm), phân tầng theo nhãn."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_split(y, groups, seed: int, test_frac=0.2, val_frac=0.2):
    n = len(y)
    idx = np.arange(n)
    k_test = max(2, round(1 / test_frac))
    tr_pool, te = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=seed).split(idx, y, groups))
    k_val = max(2, round(1 / val_frac))
    sub = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=seed).split(tr_pool, y[tr_pool], groups[tr_pool]))
    pool, val = tr_pool[sub[0]], tr_pool[sub[1]]
    assert not (set(groups[pool]) & set(groups[te])) and not (set(groups[val]) & set(groups[te]))
    assert not (set(groups[pool]) & set(groups[val]))
    return np.sort(pool), np.sort(val), np.sort(te)


def stratified_subset(pool, y, n: int, seed: int):
    """Lấy n mẫu phân tầng từ pool (nếu n >= len(pool) thì lấy hết)."""
    pool = np.asarray(pool)
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y[pool]):
        ids = pool[y[pool] == c]
        k = max(1, int(round(n * len(ids) / len(pool))))
        out.append(rng.choice(ids, size=min(k, len(ids)), replace=False))
    return np.sort(np.concatenate(out))
