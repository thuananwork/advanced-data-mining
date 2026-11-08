"""Chạy lưới thí nghiệm. Mỗi lượt ghi ngay vào CSV nên có thể ngắt rồi chạy lại với --resume."""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from . import metrics as M
from .data import load_dataset
from .models import ModelUnavailable, fit_predict
from .paths import RESULTS
from .perturb import flip_labels, imbalance_subset
from .splits import make_split, stratified_subset

KEY = ["dataset", "scenario", "level", "model", "seed", "train_size"]


def scenarios(kind: str, noise, imbalance):
    if kind == "benchmark":
        return [("clean", 0.0)]
    out = [("clean", 0.0)]
    out += [("label_noise", float(r)) for r in noise if r > 0]
    out += [("imbalance", float(f)) for f in imbalance]
    return out


def _done(path):
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    d = d[d["status"].isin(["ok", "skipped"])]
    return {tuple(r) for r in d[KEY].itertuples(index=False, name=None)}


def run(kind, datasets, models, seeds, train_sizes, noise=(), imbalance=(), calibrate=True,
        resume=True, out_name=None):
    out_dir = RESULTS / (out_name or kind); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.csv"
    if not resume and path.exists():
        path.unlink()
    done = _done(path)
    n_total = 0
    for ds in datasets:
        X, y, groups, info = load_dataset(ds)
        print(f"[{ds}] n={info['n']} đặc trưng={info['n_features']} lớp dương={info['positive_rate']:.1%} "
              f"dòng trùng={info['n_duplicate_rows']} thiếu={info['n_missing']}")
        for seed in seeds:
            pool, val, test = make_split(y, groups, seed)
            for sc, level in scenarios(kind, noise, imbalance):
                for n in train_sizes:
                    n_eff = min(n, len(pool))
                    if sc == "imbalance":
                        tr, mino = imbalance_subset(pool, y, n_eff, level, seed)
                    else:
                        tr = stratified_subset(pool, y, n_eff, seed)
                    ytr = flip_labels(y[tr], level, seed * 1000 + 7) if sc == "label_noise" else y[tr]
                    for model in models:
                        key = (ds, sc, level, model, seed, n)
                        if key in done:
                            continue
                        row = dict(zip(KEY, key))
                        row.update(n_train_used=len(tr), train_pos_rate=float(ytr.mean()),
                                   n_val=len(val), n_test=len(test), status="failed", error="")
                        try:
                            (pv, pt), fs, ps = fit_predict(model, seed, X[tr], ytr, X[val], X[test])
                            row.update(M.all_metrics(y[test], pt))
                            if calibrate:
                                T = M.fit_temperature(y[val], pv)
                                m2 = M.all_metrics(y[test], M.apply_temperature(pt, T))
                                row.update(temperature=T, ece_ts=m2["ece"], brier_ts=m2["brier"],
                                           log_loss_ts=m2["log_loss"])
                            row.update(fit_seconds=fs, predict_seconds=ps, status="ok")
                        except ModelUnavailable as e:
                            row.update(status="skipped", error=str(e))
                        except Exception as e:  # noqa: BLE001
                            row.update(error=repr(e)); traceback.print_exc()
                        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
                        n_total += 1
                        print(f"  {ds} {sc}={level} n={n} seed={seed} {model}: {row['status']}"
                              + (f" acc={row.get('accuracy', float('nan')):.3f}" if row["status"] == "ok" else f" {row['error'][:80]}"))
    print(f"Xong {n_total} lượt mới → {path}")
    return path


# ---------------------------------------------------------------------------------------------
# Ensemble LightGBM + TabICLv2 (soft voting): p = (1 - a) * p_cây + a * p_nền_tảng
ENS_ALPHAS = [round(i * 0.05, 2) for i in range(21)]
ENS_VAL = "Ensemble_val"        # a chọn theo Average Precision trên validation SẠCH của từng lượt
ENS_HALF = "Ensemble_0.5"       # a = 0,5 cố định


def run_ensemble(datasets, seeds, train_sizes, noise=(), imbalance=(), foundation="TabICLv2", tree="LightGBM",
                 resume=True, out_name="ensemble"):
    """Với mỗi điều kiện: huấn luyện `tree` và `foundation` MỘT lần, rồi trộn xác suất.
    Ghi 4 dòng mỗi lượt: tree, foundation, Ensemble_val, Ensemble_0.5 (cùng định dạng runs.csv thường)."""
    from sklearn.metrics import average_precision_score
    out_dir = RESULTS / out_name; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.csv"
    if not resume and path.exists():
        path.unlink()
    done = _done(path)
    names = [tree, foundation, ENS_VAL, ENS_HALF]
    n_total = 0
    for ds in datasets:
        X, y, groups, info = load_dataset(ds)
        print(f"[{ds}] n={info['n']} lớp dương={info['positive_rate']:.1%}")
        for seed in seeds:
            pool, val, test = make_split(y, groups, seed)
            for sc, level in scenarios("robustness", noise, imbalance):
                for n in train_sizes:
                    n_eff = min(n, len(pool))
                    if sc == "imbalance":
                        tr, _ = imbalance_subset(pool, y, n_eff, level, seed)
                    else:
                        tr = stratified_subset(pool, y, n_eff, seed)
                    ytr = flip_labels(y[tr], level, seed * 1000 + 7) if sc == "label_noise" else y[tr]
                    if all((ds, sc, level, m, seed, n) in done for m in names):
                        continue
                    base = dict(dataset=ds, scenario=sc, level=level, seed=seed, train_size=n,
                                n_train_used=len(tr), train_pos_rate=float(ytr.mean()),
                                n_val=len(val), n_test=len(test), error="")
                    try:
                        (pv_t, pt_t), fs_t, ps_t = fit_predict(tree, seed, X[tr], ytr, X[val], X[test])
                        (pv_f, pt_f), fs_f, ps_f = fit_predict(foundation, seed, X[tr], ytr, X[val], X[test])
                    except ModelUnavailable as e:
                        print("  bỏ qua:", e); continue
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc(); print("  lỗi:", repr(e)); continue
                    # chọn a theo AP validation (hòa thì chọn a lớn hơn, nghiêng về mô hình nền tảng)
                    aps = [average_precision_score(y[val], (1 - a) * pv_t + a * pv_f) for a in ENS_ALPHAS]
                    best = max(range(len(ENS_ALPHAS)), key=lambda i: (round(aps[i], 12), ENS_ALPHAS[i]))
                    a_best = ENS_ALPHAS[best]
                    variants = {
                        tree: (pv_t, pt_t, fs_t, ps_t, float("nan")),
                        foundation: (pv_f, pt_f, fs_f, ps_f, float("nan")),
                        ENS_VAL: ((1 - a_best) * pv_t + a_best * pv_f, (1 - a_best) * pt_t + a_best * pt_f,
                                  fs_t + fs_f, ps_t + ps_f, a_best),
                        ENS_HALF: (0.5 * pv_t + 0.5 * pv_f, 0.5 * pt_t + 0.5 * pt_f, fs_t + fs_f, ps_t + ps_f, 0.5),
                    }
                    for m, (pv, pt, fs, ps, a) in variants.items():
                        if (ds, sc, level, m, seed, n) in done:
                            continue
                        row = dict(base, model=m, status="ok", alpha=a)
                        row.update(M.all_metrics(y[test], pt))
                        T = M.fit_temperature(y[val], pv)
                        m2 = M.all_metrics(y[test], M.apply_temperature(pt, T))
                        row.update(temperature=T, ece_ts=m2["ece"], brier_ts=m2["brier"], log_loss_ts=m2["log_loss"],
                                   fit_seconds=fs, predict_seconds=ps)
                        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
                        n_total += 1
                    print(f"  {ds} {sc}={level} n={n} seed={seed}: a={a_best} (xong {foundation}+{tree})")
    print(f"Xong {n_total} dòng mới → {path}")
    return path
