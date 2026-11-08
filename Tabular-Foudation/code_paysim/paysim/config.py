"""Cấu hình thí nghiệm: hằng số, tham số mô hình và lớp Settings (có thể ghi đè từ dòng lệnh)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Thư mục code_paysim/ và thư mục gốc dự án (cha của code_paysim/).
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent

# Chỉ dùng 6 đặc trưng: step, amount và 4 đặc trưng lịch sử (số lần, tổng tiền của người gửi/người nhận
# trong các step TRƯỚC). Không dùng cột số dư (PaySim cảnh báo chúng gây rò rỉ nhãn) và không dùng isFlaggedFraud.
FEATURES = [
    "step",
    "amount",
    "sender_prior_attempt_count",
    "sender_prior_attempted_amount_sum",
    "recipient_prior_attempt_count",
    "recipient_prior_attempted_amount_sum",
]
HISTORY_FEATURES = FEATURES[2:]
REQUIRED_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
DATASET_FILENAMES = ["PS_20174392719_1491204439457_log.csv", "paysim.csv"]

MODEL_IDS = ["LogisticRegression", "CatBoost", "LightGBM", "TabICLv2"]
ENSEMBLE_MODELS = ("LightGBM", "TabICLv2")  # cặp mô hình được trộn (blend)
ENSEMBLE_NAME = "LightGBM+TabICLv2"

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_TRAIN_SIZES = [500, 1000, 2500, 5000]
DEFAULT_MAX_EVAL_ROWS = 20_000  # số dòng tối đa của mỗi tập validation/test (lấy mẫu cố định)
EVAL_SAMPLE_SEED = 4242  # seed lấy mẫu validation/test, giữ cố định giữa mọi mô hình và seed
TRAIN_FRACTION, VALIDATION_FRACTION = 0.60, 0.80  # ranh giới theo số step phân biệt
TUNE_TRAIN_SIZE = 5000  # cỡ train dùng để chọn tham số LightGBM
TUNING_CANDIDATES = [  # (num_leaves, min_child_samples)
    {"num_leaves": leaves, "min_child_samples": child}
    for leaves, child in [(7, 10), (7, 30), (15, 10), (15, 30), (31, 10), (31, 30)]
]
CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"

LOGISTIC_PARAMS = {"class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}
CATBOOST_PARAMS = {
    "iterations": 500, "depth": 6, "learning_rate": 0.05, "loss_function": "Logloss",
    "auto_class_weights": "Balanced", "verbose": False,
}


def _n_jobs() -> int:
    return min(4, os.cpu_count() or 2)


def lightgbm_params() -> dict:
    """Tham số LightGBM ban đầu; num_leaves và min_child_samples sẽ được thay bằng cấu hình tốt nhất khi tune."""
    return {
        "objective": "binary", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
        "min_child_samples": 20, "class_weight": "balanced", "reg_lambda": 1.0, "verbosity": -1,
        "n_jobs": _n_jobs(), "force_col_wise": True,
    }


def tabicl_params() -> dict:
    """TabICLv2 chạy CPU, một estimator (đúng như lần chạy trên Kaggle)."""
    return {
        "device": "cpu", "n_estimators": 1, "batch_size": 1, "use_amp": False, "use_fa3": False,
        "offload_mode": "auto", "checkpoint_version": CHECKPOINT, "n_jobs": _n_jobs(),
    }


@dataclass
class Settings:
    data_path: Path | None = None        # None: tự tìm (xem data.find_dataset)
    out_dir: Path = CODE_DIR / "results"  # nơi ghi kết quả; không đụng tới notebooks/paysim_fraud_report_bundle/
    seeds: list = field(default_factory=lambda: list(DEFAULT_SEEDS))
    train_sizes: list = field(default_factory=lambda: list(DEFAULT_TRAIN_SIZES))
    max_eval_rows: int | None = DEFAULT_MAX_EVAL_ROWS  # None: đánh giá toàn bộ giai đoạn
    models: list = field(default_factory=lambda: list(MODEL_IDS))
    predict_batch_size: int = 2048
    resume: bool = False                 # True: bỏ qua các lượt đã chạy xong (status=ok) trong runs.csv
    verbose: bool = True

    @property
    def expected_runs(self) -> int:
        return len(self.seeds) * len(self.train_sizes) * len(self.models)

    def smoke(self) -> "Settings":
        """Chạy thử nhanh: 1 seed, 2 cỡ train, 5.000 dòng đánh giá (ghi ra results_smoke để không đè kết quả 48 lượt)."""
        self.seeds, self.train_sizes, self.max_eval_rows = [42], [500, 1000], 5_000
        if self.out_dir == CODE_DIR / "results":
            self.out_dir = CODE_DIR / "results_smoke"
        return self
