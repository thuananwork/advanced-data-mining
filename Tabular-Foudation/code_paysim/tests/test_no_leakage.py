"""Test nhanh trên dữ liệu giả: đặc trưng lịch sử không nhìn vào tương lai/cùng step; chia tập không rò rỉ."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paysim.data import build_transfer_frame
from paysim.splits import build_train_subsets, check_no_leakage, temporal_split


def toy():
    rows = [  # step, type, amount, orig, dest, fraud
        (1, "PAYMENT", 10, "A", "M", 0), (1, "TRANSFER", 20, "A", "B", 0), (2, "TRANSFER", 30, "A", "B", 0),
        (2, "TRANSFER", 40, "A", "C", 0), (3, "TRANSFER", 50, "A", "B", 1),
    ]
    df = pd.DataFrame(rows, columns=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"])
    df.insert(0, "_source_row", range(len(df)))
    return df


def test_history_excludes_same_and_future_step():
    out = build_transfer_frame(toy())
    a = out.set_index("amount")
    assert a.loc[20, "sender_prior_attempt_count"] == 0      # step 1: chỉ có giao dịch cùng step
    assert a.loc[30, "sender_prior_attempt_count"] == 2      # step 2: thấy 2 giao dịch của step 1 (mọi loại)
    assert a.loc[30, "sender_prior_attempted_amount_sum"] == 30
    assert a.loc[50, "sender_prior_attempt_count"] == 4      # step 3 thấy step 1 và 2
    assert a.loc[50, "recipient_prior_attempt_count"] == 2   # B nhận 20 (step1) và 30 (step2)


def test_split_has_no_leakage():
    rng = np.random.default_rng(0)
    n = 4000
    steps = np.sort(rng.integers(1, 101, n))
    y = (rng.random(n) < 0.1).astype(np.int8)
    pool, v, t, cfg = temporal_split(steps, y, 500, "x")
    subs = build_train_subsets(pool, y, [1, 2], [200, 400])
    check_no_leakage(subs, v, t, pool, steps, cfg)
    assert steps[pool].max() < steps[v].min() and steps[v].max() < steps[t].min()


if __name__ == "__main__":
    test_history_excludes_same_and_future_step(); test_split_has_no_leakage(); print("OK")
