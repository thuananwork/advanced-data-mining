"""Chạy toàn bộ dự án bằng một lệnh (xem README.md).

  python run_all.py check                 kiểm tra môi trường, dữ liệu, thư viện
  python run_all.py benchmark             so sánh các mô hình trên Breast Cancer, Banknote, Wine, MAGIC, Pima
  python run_all.py robustness            Pima: nhiễu nhãn + mất cân bằng + hiệu chỉnh xác suất (điểm mới)
  python run_all.py ensemble              Pima: Ensemble LightGBM + TabICLv2 (nạp sạch, nhiễu nhãn, mất cân bằng)
  python run_all.py paysim                PaySim (gọi code_paysim/run_paysim.py)
  python run_all.py report                tổng hợp bảng, kiểm định, biểu đồ
  python run_all.py all                   tất cả theo thứ tự trên
Thêm --smoke để chạy thử nhanh, --resume để bỏ qua lượt đã xong.
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

from tfm import experiments, report
from tfm.data import SPECS, load_dataset
from tfm.models import ALL_MODELS
from tfm.paths import CODE_DIR, DATASETS

NOISE = [0.05, 0.1, 0.2, 0.3]
IMB = [0.3, 0.2, 0.1, 0.05, 0.02]


def cmd_check(_):
    print("Python", platform.python_version(), platform.platform())
    for mod, label in [("numpy", ""), ("pandas", ""), ("sklearn", ""), ("scipy", ""), ("matplotlib", ""),
                       ("lightgbm", ""), ("xgboost", ""), ("catboost", ""), ("torch", ""),
                       ("tabicl", "TabICLv2"), ("tabpfn", "TabPFNv2"), ("tabdpt", "TabDPT")]:
        print(f"  {mod:12s}", "OK" if importlib.util.find_spec(mod) else "THIẾU", label)
    for name in SPECS:
        try:
            _, _, _, info = load_dataset(name)
            print(f"  dữ liệu {name:14s} OK  n={info['n']} dòng trùng={info['n_duplicate_rows']}")
        except Exception as e:  # noqa: BLE001
            print(f"  dữ liệu {name:14s} LỖI {e}")
    print("  PaySim gốc:", "có" if (DATASETS / "paysim" / "PS_20174392719_1491204439457_log.csv").exists() or (DATASETS / "PS_20174392719_1491204439457_log.csv").exists() else "thiếu")


def _models(a):
    return a.models or ALL_MODELS


def cmd_benchmark(a):
    ds = a.datasets or ["breast_cancer", "banknote", "wine", "magic", "pima"]
    if a.smoke:
        return experiments.run("benchmark", ["breast_cancer"], _models(a), [42], [50], resume=a.resume, out_name="benchmark_smoke")
    experiments.run("benchmark", ds, _models(a), a.seeds or [42, 43, 44, 45, 46],
                    a.train_sizes or [50, 100, 200, 400], resume=a.resume)


def cmd_robustness(a):
    ds = a.datasets or ["pima"]
    if a.smoke:
        return experiments.run("robustness", ds[:1], _models(a), [42], [200], noise=[0.2], imbalance=[0.1],
                               resume=a.resume, out_name="robustness_smoke")
    experiments.run("robustness", ds, _models(a), a.seeds or [42, 43, 44, 45, 46],
                    a.train_sizes or [200], noise=NOISE, imbalance=IMB, resume=a.resume)


def cmd_ensemble(a):
    seeds = a.seeds or [42, 43, 44, 45, 46]
    if a.smoke:
        return experiments.run_ensemble(["pima"], [42], [100], noise=[0.2], imbalance=[0.1], resume=a.resume,
                                        foundation=a.foundation, tree=a.tree, out_name="ensemble_smoke")
    # phần sạch: 4 cỡ huấn luyện; phần nhiễu/mất cân bằng: n = 200 (giống nhóm robustness)
    experiments.run_ensemble(["pima"], seeds, a.train_sizes or [50, 100, 200, 400], resume=a.resume,
                             foundation=a.foundation, tree=a.tree)
    experiments.run_ensemble(["pima"], seeds, [200], noise=NOISE, imbalance=IMB, resume=True,
                             foundation=a.foundation, tree=a.tree)


def cmd_paysim(a):
    cmd = [sys.executable, "run_paysim.py"]
    if a.smoke: cmd += ["--smoke"]
    if a.resume: cmd += ["--resume"]
    if a.models and "TabICLv2" not in a.models: cmd += ["--skip-tabicl"]
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=CODE_DIR / "code_paysim", check=True)


def cmd_report(a):
    names = ["benchmark_smoke", "robustness_smoke", "ensemble_smoke"] if a.smoke else ["benchmark", "robustness", "ensemble"]
    from tfm.paths import RESULTS
    for k in names:
        if (RESULTS / k / "runs.csv").exists():
            report.summarize(k)


def cmd_all(a):
    for f in (cmd_check, cmd_benchmark, cmd_robustness, cmd_ensemble, cmd_paysim, cmd_report):
        f(a)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "benchmark", "robustness", "ensemble", "paysim", "report", "all"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS)
    p.add_argument("--datasets", nargs="+", choices=list(SPECS))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--train-sizes", type=int, nargs="+")
    p.add_argument("--foundation", default="TabICLv2", choices=ALL_MODELS)
    p.add_argument("--tree", default="LightGBM", choices=ALL_MODELS)
    a = p.parse_args()
    {"check": cmd_check, "benchmark": cmd_benchmark, "robustness": cmd_robustness, "ensemble": cmd_ensemble,
     "paysim": cmd_paysim, "report": cmd_report, "all": cmd_all}[a.command](a)


if __name__ == "__main__":
    main()
