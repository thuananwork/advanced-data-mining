"""Tổng hợp kết quả: bảng mean±std, mức tụt Δ so với dữ liệu sạch, kiểm định cặp Wilcoxon, biểu đồ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import RESULTS

METRICS = ["accuracy", "macro_f1", "roc_auc", "avg_precision", "log_loss", "ece", "brier"]
HIGHER_BETTER = {"accuracy", "macro_f1", "roc_auc", "avg_precision"}


def summarize(kind: str):
    d = RESULTS / kind
    runs = pd.read_csv(d / "runs.csv")
    ok = runs[runs["status"] == "ok"].copy()
    if ok.empty:
        print("Chưa có lượt nào chạy thành công."); return
    g = ["dataset", "scenario", "level", "model", "train_size"]
    cols = [m for m in METRICS + ["ece_ts", "brier_ts"] if m in ok.columns]
    summ = ok.groupby(g)[cols].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index(); summ.to_csv(d / "summary.csv", index=False)

    if (ok["scenario"] != "clean").any():
        clean = ok[ok["scenario"] == "clean"].groupby(["dataset", "model", "train_size"])[METRICS].mean()
        rows = []
        for (ds, sc, lv, md, n), grp in ok[ok["scenario"] != "clean"].groupby(g):
            if (ds, md, n) not in clean.index:
                continue
            base = clean.loc[(ds, md, n)]
            for m in METRICS:
                # Δ theo TabFSBench: (chỉ số_i − chỉ số_0) / chỉ số_0 (âm = tệ đi với chỉ số "càng cao càng tốt")
                rows.append({"dataset": ds, "scenario": sc, "level": lv, "model": md, "train_size": n,
                             "metric": m, "clean": base[m], "perturbed": grp[m].mean(),
                             "delta": (grp[m].mean() - base[m]) / base[m] if base[m] else np.nan})
        dd = pd.DataFrame(rows); dd.to_csv(d / "delta_vs_clean.csv", index=False)

    ref = "TabICLv2"
    if ref in set(ok["model"]):
        rows = []
        keys = ["dataset", "scenario", "level", "train_size", "seed"]
        for metric in ["roc_auc", "log_loss", "ece"]:
            pv = ok.pivot_table(index=keys, columns="model", values=metric)
            for other in pv.columns:
                if other == ref:
                    continue
                pair = pv[[ref, other]].dropna()
                if len(pair) >= 6:
                    from scipy.stats import wilcoxon
                    try:
                        p = float(wilcoxon(pair[ref], pair[other]).pvalue)
                    except ValueError:
                        p = float("nan")
                    rows.append({"metric": metric, "vs": other, "pairs": len(pair),
                                 "mean_diff(ref-other)": float((pair[ref] - pair[other]).mean()), "wilcoxon_p": p})
        if rows:
            pd.DataFrame(rows).to_csv(d / "paired_tests.csv", index=False)
    _plots(d, ok)
    print("Đã ghi", d)


def _plots(d, ok):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for sc in ["label_noise", "imbalance"]:
        sub = ok[(ok["scenario"] == sc) | (ok["scenario"] == "clean")]
        if (ok["scenario"] == sc).sum() == 0:
            continue
        for ds, dsub in sub.groupby("dataset"):
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
            for ax, m in zip(axes, ["roc_auc", "log_loss", "ece"]):
                for model, g in dsub.groupby("model"):
                    cl = g[g["scenario"] == "clean"]; pe = g[g["scenario"] == sc]
                    xs, ys = [], []
                    if len(cl) and sc == "label_noise":
                        xs.append(0.0); ys.append(cl[m].mean())
                    for lv, gg in pe.groupby("level"):
                        xs.append(lv); ys.append(gg[m].mean())
                    ax.plot(xs, ys, marker="o", label=model)
                ax.set_title(m); ax.set_xlabel("tỉ lệ nhiễu nhãn" if sc == "label_noise" else "tỉ lệ lớp thiểu số trong train")
            axes[0].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(d / f"{ds}_{sc}.png", dpi=130); plt.close(fig)
