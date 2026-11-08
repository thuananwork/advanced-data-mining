"""Hai kiểu làm bẩn tập train: nhiễu nhãn và mất cân bằng lớp. Test/validation luôn giữ sạch."""
from __future__ import annotations

import numpy as np


def flip_labels(y_train, rate: float, seed: int):
    """Đảo ngẫu nhiên một tỉ lệ nhãn (nhiễu đối xứng). Giữ lại ít nhất 1 mẫu mỗi lớp."""
    y = np.array(y_train, copy=True)
    if rate <= 0:
        return y
    rng = np.random.default_rng(seed)
    k = int(round(rate * len(y)))
    flip = rng.choice(len(y), size=k, replace=False)
    y2 = y.copy(); y2[flip] = 1 - y2[flip]
    return y2 if len(np.unique(y2)) == 2 else y


def imbalance_subset(pool, y, n: int, minority_frac: float, seed: int):
    """Lấy tối đa n mẫu từ pool sao cho lớp thiểu số chiếm minority_frac. Trả về (chỉ số, lớp thiểu số)."""
    pool = np.asarray(pool)
    rng = np.random.default_rng(seed)
    counts = {c: int((y[pool] == c).sum()) for c in (0, 1)}
    mino = min(counts, key=counts.get); majo = 1 - mino
    n_min = max(2, int(round(n * minority_frac)))
    n_maj = n - n_min
    n_min = min(n_min, counts[mino]); n_maj = min(n_maj, counts[majo])
    a = rng.choice(pool[y[pool] == mino], size=n_min, replace=False)
    b = rng.choice(pool[y[pool] == majo], size=n_maj, replace=False)
    return np.sort(np.concatenate([a, b])), mino
