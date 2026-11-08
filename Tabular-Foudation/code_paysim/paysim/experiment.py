"""Tune LightGBM, chạy lưới (seed x cỡ train x mô hình) và trộn LightGBM + TabICLv2."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config as C
from .models import build_model, predict_positive_probability, threshold_from_validation
from .splits import hash_indices

METRICS = ["validation_average_precision", "test_average_precision", "test_roc_auc",
           "test_precision", "test_recall", "test_f1"]


def log(settings, *a):
    if settings.verbose:
        print(*a, flush=True)


def tune_lightgbm(X, y, subsets, val_idx, settings):
    """Chọn (num_leaves, min_child_samples) theo AP validation trung bình qua các seed."""
    from lightgbm import LGBMClassifier
    size = C.TUNE_TRAIN_SIZE if C.TUNE_TRAIN_SIZE in settings.train_sizes else max(settings.train_sizes)
    rows = []
    for cid, cand in enumerate(C.TUNING_CANDIDATES):
        for seed in settings.seeds:
            idx = np.asarray(subsets[str(seed)][str(size)], dtype=np.int64)
            model = LGBMClassifier(**{**C.lightgbm_params(), **cand, "random_state": seed})
            model.fit(X[idx], y[idx])
            ap = average_precision_score(y[val_idx], predict_positive_probability(model, X[val_idx]))
            rows.append({"candidate_id": cid, "seed": seed, "train_size": size, **cand,
                         "validation_average_precision": ap})
        log(settings, f"  tuning candidate {cid} {cand}")
    tuning = pd.DataFrame(rows)
    means = tuning.groupby("candidate_id")["validation_average_precision"].mean()
    best_id = int(means.idxmax())
    params = {**C.lightgbm_params(), **C.TUNING_CANDIDATES[best_id]}
    info = {"metric": "mean validation average precision across seeds", "train_size": size,
            "candidates": C.TUNING_CANDIDATES, "best_candidate_id": best_id,
            "best_candidate_mean_validation_ap": float(means.loc[best_id])}
    return tuning, params, info


def run_grid(X, y, subsets, val_idx, test_idx, lgbm_params, meta, settings, runs_path, preds_path):
    """Chạy toàn bộ lượt; ghi runs.csv sau mỗi lượt. Xác suất của LightGBM/TabICLv2 lưu để trộn."""
    y_val, y_test = y[val_idx], y[test_idx]
    done = pd.DataFrame()
    if settings.resume and runs_path.is_file():
        old = pd.read_csv(runs_path)
        done = old[old["status"].eq("ok")]
    else:
        runs_path.unlink(missing_ok=True)
        preds_path.unlink(missing_ok=True)
    rows = done.to_dict("records")
    finished = {(r["model"], int(r["seed"]), int(r["train_size"])) for r in rows}
    total, i, t0 = settings.expected_runs, 0, time.perf_counter()
    for seed in settings.seeds:
        for n in settings.train_sizes:
            idx = np.asarray(subsets[str(seed)][str(n)], dtype=np.int64)
            xt, yt = X[idx], y[idx]
            for mid in settings.models:
                i += 1
                if (mid, seed, n) in finished:
                    continue
                row = {"model": mid, "seed": seed, "train_size": n, "train_fraud": int(yt.sum()),
                       "train_prevalence": float(yt.mean()), "validation_size": len(val_idx),
                       "validation_fraud": int(y_val.sum()), "test_size": len(test_idx),
                       "test_fraud": int(y_test.sum()), "dataset_sha256": meta["dataset_sha256"],
                       "train_indices_sha256": hash_indices(idx),
                       "validation_indices_sha256": meta["validation_indices_sha256"],
                       "test_indices_sha256": meta["test_indices_sha256"], "status": "failed", "error": ""}
                t = time.perf_counter()
                log(settings, f"[{i}/{total}] {mid} seed={seed} train={n}")
                try:
                    model = build_model(mid, seed, lgbm_params)
                    s = time.perf_counter(); model.fit(xt, yt); row["fit_seconds"] = time.perf_counter() - s
                    s = time.perf_counter()
                    pv = predict_positive_probability(model, X[val_idx], settings.predict_batch_size)
                    row["validation_predict_seconds"] = time.perf_counter() - s
                    thr, f1v = threshold_from_validation(y_val, pv)
                    row.update(threshold_from_validation=thr, validation_f1_at_threshold=f1v,
                               validation_average_precision=average_precision_score(y_val, pv))
                    s = time.perf_counter()
                    pt = predict_positive_probability(model, X[test_idx], settings.predict_batch_size)
                    row["test_predict_seconds"] = time.perf_counter() - s
                    pred = (pt >= thr).astype(np.int8)
                    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
                    row.update(test_average_precision=average_precision_score(y_test, pt),
                               test_roc_auc=roc_auc_score(y_test, pt),
                               test_precision=precision_score(y_test, pred, zero_division=0),
                               test_recall=recall_score(y_test, pred, zero_division=0),
                               test_f1=f1_score(y_test, pred, zero_division=0),
                               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), status="ok")
                    if mid in C.ENSEMBLE_MODELS:
                        recs = []
                        for split, ix, lab, pr in [("validation", val_idx, y_val, pv), ("test", test_idx, y_test, pt)]:
                            recs.extend({"model": mid, "seed": seed, "train_size": n, "split": split,
                                         "row_index": int(a), "label": int(b), "probability": float(c)}
                                        for a, b, c in zip(ix, lab, pr))
                        pd.DataFrame(recs).to_csv(preds_path, mode="a", header=not preds_path.exists(), index=False)
                    log(settings, f"   AP={row['test_average_precision']:.4f} AUC={row['test_roc_auc']:.4f}")
                except Exception as exc:  # một mô hình lỗi không làm dừng cả lưới
                    row["error"] = repr(exc)[:2000]
                    log(settings, "   FAILED:", row["error"])
                row["run_seconds"] = time.perf_counter() - t
                rows.append(row)
                pd.DataFrame(rows).to_csv(runs_path, index=False)
    log(settings, f"Xong lưới trong {time.perf_counter() - t0:.1f}s")
    return pd.DataFrame(rows)


def summarize(runs, models):
    """Chỉ giữ nhóm (seed, cỡ train) có đủ mọi mô hình thành công và cùng chỉ số train/val/test."""
    ok = runs[runs["status"].eq("ok")]
    paired = ok.groupby(["seed", "train_size"]).filter(lambda g: set(g["model"]) == set(models))
    for _, g in paired.groupby(["seed", "train_size"]):
        for k in ["train_indices_sha256", "validation_indices_sha256", "test_indices_sha256"]:
            if g[k].nunique() != 1:
                raise ValueError(f"Các mô hình không dùng cùng {k}.")
    if paired.empty:
        return paired, pd.DataFrame()
    s = paired.groupby(["model", "train_size"])[METRICS].agg(["mean", "std", "count"])
    s.columns = ["_".join(c) for c in s.columns]
    return paired, s.reset_index()


def blend_ensemble(paired, preds_path):
    """alpha (trọng số TabICLv2) chọn trên validation AP; ngưỡng cũng chọn trên validation."""
    pred = pd.read_csv(preds_path)
    if pred.duplicated(["model", "seed", "train_size", "split", "row_index"]).any():
        raise ValueError("File dự đoán có hàng lặp.")
    pairs = [(int(s), int(n)) for s, n in paired.groupby(["seed", "train_size"]).groups]

    def aligned(seed, n, split):
        sel = pred[pred["seed"].eq(seed) & pred["train_size"].eq(n) & pred["split"].eq(split)]
        scores = sel.pivot(index="row_index", columns="model", values="probability").sort_index()
        labels = sel[["row_index", "label"]].drop_duplicates().set_index("row_index")["label"].sort_index()
        if set(scores.columns) != set(C.ENSEMBLE_MODELS) or not scores.index.equals(labels.index):
            raise ValueError(f"Thiếu hoặc lệch cặp mô hình: seed={seed}, n={n}, split={split}")
        return labels.to_numpy(np.int8), scores["LightGBM"].to_numpy(), scores["TabICLv2"].to_numpy()

    alphas = np.linspace(0.0, 1.0, 21)
    cache = {(s, n, sp): aligned(s, n, sp) for s, n in pairs for sp in ("validation", "test")}
    val_scores = []
    for a in alphas:
        val_scores.append(float(np.mean([
            average_precision_score(cache[(s, n, "validation")][0],
                                    (1 - a) * cache[(s, n, "validation")][1] + a * cache[(s, n, "validation")][2])
            for s, n in pairs])))
    best = float(alphas[int(np.argmax(val_scores))])
    rows = []
    for s, n in pairs:
        vl, vg, vt = cache[(s, n, "validation")]
        tl, tg, tt = cache[(s, n, "test")]
        vb, tb = (1 - best) * vg + best * vt, (1 - best) * tg + best * tt
        thr, f1v = threshold_from_validation(vl, vb)
        pr = (tb >= thr).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(tl, pr, labels=[0, 1]).ravel()
        rows.append({"model": C.ENSEMBLE_NAME, "seed": s, "train_size": n, "tabicl_weight": best,
                     "validation_average_precision": average_precision_score(vl, vb),
                     "validation_f1_at_threshold": f1v, "threshold_from_validation": thr,
                     "test_average_precision": average_precision_score(tl, tb),
                     "test_roc_auc": roc_auc_score(tl, tb),
                     "test_precision": precision_score(tl, pr, zero_division=0),
                     "test_recall": recall_score(tl, pr, zero_division=0),
                     "test_f1": f1_score(tl, pr, zero_division=0),
                     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    ens = pd.DataFrame(rows)
    es = ens.groupby("train_size")[METRICS].agg(["mean", "std", "count"])
    es.columns = ["_".join(c) for c in es.columns]
    return ens, es.reset_index(), best
