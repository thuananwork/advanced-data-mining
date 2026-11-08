r"""Giao diện tương tác (Streamlit Demo) cho đồ án Tabular Foundation Models.

Tập trung vào 3 nội dung cốt lõi theo mạch trình bày 3 bước:
  - Bước 1: Lúc đầu TabICLv2 đơn lẻ ra số liệu như thế nào?
  - Bước 2: Khi đem so sánh trực tiếp TabICLv2 với LightGBM đơn lẻ thì ra sao?
  - Bước 3: Khi kết hợp cả hai lại (LightGBM + TabICLv2 Ensemble) thì đạt kết quả gì?

Chạy bằng lệnh:
    .\.venv\Scripts\streamlit.exe run demo_app.py
hoặc nhấp đúp vào file `run_demo.bat`.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import streamlit as st

from tfm.paths import CODE_DIR, RESULTS

st.set_page_config(
    page_title="Báo cáo Thực nghiệm: TabICLv2 vs LightGBM & Ensemble",
    page_icon="📊",
    layout="wide",
)

PAYSIM_RESULTS = CODE_DIR / "code_paysim" / "results"


@st.cache_data
def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    return None


def compute_ensemble_wilcoxon(df_ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["roc_auc", "macro_f1", "avg_precision", "log_loss", "ece", "brier"]:
        piv = df_ens.pivot(
            index=["scenario", "level", "train_size", "seed"],
            columns="model",
            values=metric,
        )
        pairs_to_check = [
            ("TabICLv2 (Bước 1)", "LightGBM (Bước 2)", "TabICLv2", "LightGBM"),
            ("Ensemble_val (Bước 3)", "TabICLv2 (Bước 1)", "Ensemble_val", "TabICLv2"),
            ("Ensemble_val (Bước 3)", "LightGBM (Bước 2)", "Ensemble_val", "LightGBM"),
            ("Ensemble_0.5 (Bước 3)", "TabICLv2 (Bước 1)", "Ensemble_0.5", "TabICLv2"),
            ("Ensemble_0.5 (Bước 3)", "LightGBM (Bước 2)", "Ensemble_0.5", "LightGBM"),
        ]
        for label_a, label_b, col_a, col_b in pairs_to_check:
            if col_a in piv.columns and col_b in piv.columns:
                diff = (piv[col_a] - piv[col_b]).dropna()
                nz = diff[diff != 0]
                p_val = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
                rows.append({
                    "Mô hình A": label_a,
                    "Mô hình B (Đối chiếu)": label_b,
                    "Chỉ số": metric,
                    "Số cặp điều kiện": len(diff),
                    "Chênh lệch TB (A - B)": round(float(diff.mean()), 4),
                    "Wilcoxon p-value": float(f"{p_val:.4g}"),
                    "Kết luận thống kê (p < 0.05)": "Có ý nghĩa" if p_val < 0.05 else "Tương đương (Không khác biệt)",
                })
    return pd.DataFrame(rows)


# ============================================================================
# HEADER & TỔNG QUAN
# ============================================================================
st.title("📊 Thực nghiệm Tabular Foundation Models: Từ TabICLv2 đến Kết hợp LightGBM + TabICLv2")
st.markdown(
    "**Học viên:** Nguyễn Thuận An · **Môn học:** Khám phá tri thức và Khai phá dữ liệu  \n"
    "**Mạch trình bày trọng tâm:** **(1) Kết quả `TabICLv2` ban đầu** $\\rightarrow$ "
    "**(2) Đối sánh trực tiếp `TabICLv2` với `LightGBM`** $\\rightarrow$ "
    "**(3) Kết hợp cả hai (`LightGBM + TabICLv2` Ensemble)** trên **Pima**, **PaySim** và **Benchmark 5 bộ dữ liệu**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Pima: LightGBM + TabICLv2", "260 / 260 dòng", "65 điều kiện × 4 cấu hình (AUC = 0.8352)")
c2.metric("2. PaySim: LightGBM + TabICLv2", "48 / 48 lượt", "Tăng Test AP ở cả 4 cỡ mẫu (lên 0.4903)")
c3.metric("3. Benchmark 5 Bộ dữ liệu", "800 / 800 lượt", "5 bộ dữ liệu × 8 mô hình × 5 seeds × 4 cỡ mẫu")
c4.metric("4. Độ bền Nhiễu & Imbalance", "400 / 400 lượt", "Nhiễu nhãn 5–30% & Imbalance 30%→2%")

st.divider()

tab_pima, tab_paysim, tab_bench, tab_cli = st.tabs([
    "🤝 1. LightGBM + TabICLv2 trên Pima (3 Bước: Đơn lẻ → So sánh → Kết hợp)",
    "💳 2. Bài toán PaySim (3 Bước: TabICLv2 → LightGBM → Kết hợp)",
    "📈 3. Benchmark 5 Bộ dữ liệu (800 lượt) & Độ bền (400 lượt)",
    "⚡ 4. Chạy Smoke Test CLI & Vị trí Mở Code",
])

# ============================================================================
# TAB 1: LIGHTGBM + TABICLV2 TRÊN PIMA (THEO 3 BƯỚC RÕ RÀNG)
# ============================================================================
with tab_pima:
    ens_runs = read_csv_safe(RESULTS / "ensemble" / "runs.csv")
    if ens_runs is None:
        st.error("Chưa tìm thấy file `results/ensemble/runs.csv`.")
    else:
        metrics_cols = ["accuracy", "macro_f1", "roc_auc", "avg_precision", "log_loss", "ece", "brier"]
        rename_metrics = {
            "accuracy": "Accuracy",
            "macro_f1": "Macro-F1",
            "roc_auc": "ROC-AUC (↑)",
            "avg_precision": "Avg Precision / PR-AUC (↑)",
            "log_loss": "Log-Loss (↓)",
            "ece": "ECE - Hiệu chỉnh (↓)",
            "brier": "Brier Score (↓)",
        }

        # --------------------------------------------------------------------
        # BƯỚC 1: LÚC ĐẦU TABICLV2 ĐƠN LẺ RA SỐ LIỆU NHƯ THẾ NÀO?
        # --------------------------------------------------------------------
        st.markdown("## 🟦 BƯỚC 1: Lúc đầu `TabICLv2` đơn lẻ ra số liệu như thế nào?")
        st.markdown(
            "Trước khi so sánh hay kết hợp, ta đánh giá mô hình nền tảng **`TabICLv2`** chạy độc lập (Zero-shot In-Context Learning trên CPU, không cần tinh chỉnh siêu tham số) "
            "trên bộ dữ liệu **Tiểu đường Pima** qua 3 kịch bản: **(a) Dữ liệu sạch theo cỡ mẫu**, **(b) Nhiễu nhãn 5% → 30%**, và **(c) Mất cân bằng lớp 30% → 2%**."
        )

        tabicl_only = ens_runs[ens_runs["model"] == "TabICLv2"]
        t_clean_200 = tabicl_only[
            (tabicl_only["scenario"] == "clean") & (tabicl_only["train_size"] == 200)
        ][metrics_cols].mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC (n=200 sạch)", f"{t_clean_200['roc_auc']:.4f}", "Xếp hạng phân lớp tốt")
        m2.metric("Macro-F1 (n=200 sạch)", f"{t_clean_200['macro_f1']:.4f}", "Cân bằng Precision/Recall")
        m3.metric("Average Precision (PR-AUC)", f"{t_clean_200['avg_precision']:.4f}", "Độ chính xác lớp dương")
        m4.metric("Log-Loss (↓ càng nhỏ càng tốt)", f"{t_clean_200['log_loss']:.4f}", "Xác suất đáng tin cậy")
        m5.metric("ECE - Sai số hiệu chỉnh (↓)", f"{t_clean_200['ece']:.4f}", "Không bị quá tự tin")

        s1_col1, s1_col2, s1_col3 = st.columns(3)
        with s1_col1:
            st.markdown("**1A. `TabICLv2` theo cỡ mẫu huấn luyện (Dữ liệu sạch)**")
            t_by_n = (
                tabicl_only[tabicl_only["scenario"] == "clean"]
                .groupby("train_size")[["roc_auc", "macro_f1", "avg_precision", "log_loss", "ece"]]
                .mean()
                .round(4)
                .reset_index()
            )
            st.dataframe(t_by_n, width="stretch", hide_index=True)

        with s1_col2:
            st.markdown("**1B. `TabICLv2` khi tăng Nhiễu nhãn (tại `n = 200`)**")
            t_by_noise = (
                tabicl_only[
                    (tabicl_only["train_size"] == 200)
                    & (tabicl_only["scenario"].isin(["clean", "label_noise"]))
                ]
                .groupby("level")[["roc_auc", "macro_f1", "avg_precision", "log_loss", "ece"]]
                .mean()
                .round(4)
                .reset_index()
                .rename(columns={"level": "Tỷ lệ nhiễu nhãn"})
            )
            st.dataframe(t_by_noise, width="stretch", hide_index=True)

        with s1_col3:
            st.markdown("**1C. `TabICLv2` khi Mất cân bằng lớp (tại `n = 200`)**")
            t_by_imb = (
                tabicl_only[
                    (tabicl_only["train_size"] == 200) & (tabicl_only["scenario"] == "imbalance")
                ]
                .groupby("level")[["roc_auc", "macro_f1", "avg_precision", "log_loss", "ece"]]
                .mean()
                .round(4)
                .reset_index()
                .sort_values("level", ascending=False)
                .rename(columns={"level": "Tỷ lệ lớp dương (Bệnh)"})
            )
            st.dataframe(t_by_imb, width="stretch", hide_index=True)

        st.divider()

        # --------------------------------------------------------------------
        # BƯỚC 2: SO SÁNH TRỰC TIẾP TABICLV2 VỚI LIGHTGBM ĐƠN LẺ
        # --------------------------------------------------------------------
        st.markdown("## 🟧 BƯỚC 2: Đem `TabICLv2` so sánh trực tiếp với `LightGBM` đơn lẻ")
        st.markdown(
            "Tiếp theo, ta đặt **`TabICLv2`** cạnh đại diện phổ biến nhất của nhóm cây quyết định tăng cường gradient là **`LightGBM`** trên cùng tập mẫu huấn luyện và kiểm thử."
        )

        # Bảng so sánh 1-đối-1 tại 3 điều kiện tiêu biểu
        comp_rows = []
        for sc_label, sc_name, lv_val in [
            ("1. Dữ liệu sạch (n=200, noise=0%)", "clean", 0.0),
            ("2. Nhiễu nhãn 20% (n=200, noise=20%)", "label_noise", 0.2),
            ("3. Nhiễu nhãn 30% (n=200, noise=30%)", "label_noise", 0.3),
            ("4. Mất cân bằng 5% (n=200, pos=5%)", "imbalance", 0.05),
            ("5. Mất cân bằng cực đoan 2% (n=200, pos=2%)", "imbalance", 0.02),
        ]:
            sub_pair = ens_runs[
                (ens_runs["train_size"] == 200)
                & (ens_runs["scenario"] == sc_name)
                & (ens_runs["level"] == lv_val)
            ]
            t_m = sub_pair[sub_pair["model"] == "TabICLv2"].mean(numeric_only=True)
            l_m = sub_pair[sub_pair["model"] == "LightGBM"].mean(numeric_only=True)
            comp_rows.append({
                "Kịch bản (tại n=200)": sc_label,
                "ROC-AUC (TabICLv2)": round(float(t_m["roc_auc"]), 4),
                "ROC-AUC (LightGBM)": round(float(l_m["roc_auc"]), 4),
                "Chênh lệch AUC (TabICLv2 - LGBM)": f"+{float(t_m['roc_auc'] - l_m['roc_auc']):.4f}",
                "AP (TabICLv2)": round(float(t_m["avg_precision"]), 4),
                "AP (LightGBM)": round(float(l_m["avg_precision"]), 4),
                "Log-Loss (TabICLv2 ↓)": round(float(t_m["log_loss"]), 4),
                "Log-Loss (LightGBM ↓)": round(float(l_m["log_loss"]), 4),
                "ECE (TabICLv2 ↓)": round(float(t_m["ece"]), 4),
                "ECE (LightGBM ↓)": round(float(l_m["ece"]), 4),
            })

        st.markdown("**Bảng đối chiếu trực diện 1-đối-1: `TabICLv2` vs `LightGBM` (Trung bình 5 seeds)**")
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        st.warning(
            "🔍 **Phát hiện quan trọng ở Bước 2 (Khi so sánh `TabICLv2` với `LightGBM`):**  \n"
            "- **Về khả năng phân lớp (ROC-AUC & AP):** `TabICLv2` vượt `LightGBM` ở **tất cả các kịch bản**. Trên toàn bộ 65 điều kiện ghép cặp, `TabICLv2` cao hơn trung bình **+0.0517 ROC-AUC** ($p = 1.31 \\times 10^{-11}$). Khi nhiễu nhãn lên 30%, `LightGBM` rơi xuống **0.6782** trong khi `TabICLv2` giữ vững **0.7496** (+0.0714).  \n"
            "- **Về độ tin cậy của xác suất (Log-Loss & ECE):** Trên dữ liệu nhỏ ($n=200$), các nhánh cây của `LightGBM` bị khớp quá mức và **quá tự tin (overconfident)**, làm `Log-Loss = 1.0182` và `ECE = 0.2197` (tệ hơn gấp đôi so với `Log-Loss = 0.4778` và `ECE = 0.0933` của `TabICLv2`)."
        )

        st.divider()

        # --------------------------------------------------------------------
        # BƯỚC 3: KẾT HỢP CẢ 2 CHÚNG NÓ LẠI (LIGHTGBM + TABICLV2 ENSEMBLE)
        # --------------------------------------------------------------------
        st.markdown("## 🟩 BƯỚC 3: Kết hợp cả hai lại (`LightGBM + TabICLv2` Ensemble)")
        st.markdown(
            """
            Sau khi biết điểm mạnh/yếu của từng mô hình, ta kết hợp xác suất dự đoán của cả hai theo công thức:
            $$p_{\\text{Ensemble}}(y=1 \\mid x) = (1 - \\alpha) \\cdot p_{\\text{LightGBM}}(y=1 \\mid x) + \\alpha \\cdot p_{\\text{TabICLv2}}(y=1 \\mid x)$$
            Ta khảo sát **2 cách kết hợp** để thấy rõ vai trò của trọng số $\\alpha$:
            1. **`Ensemble_0.5` ($\\alpha = 0.50$ cố định):** Trộn đều 50% `LightGBM` + 50% `TabICLv2`.
            2. **`Ensemble_val` ($\\alpha$ tối ưu tự động trên Validation sạch):** Quét $\\alpha \\in \\{0.00, 0.05, \\dots, 1.00\\}$ để tối đa hóa Average Precision trên tập Validation sạch (kết quả trung bình chọn **$\\alpha = 0.877 \\pm 0.239$** nghiêng về `TabICLv2`).
            """
        )

        order_models = ["TabICLv2", "LightGBM", "Ensemble_0.5", "Ensemble_val"]
        model_display_names = {
            "TabICLv2": "1. TabICLv2 (Đơn lẻ - Bước 1)",
            "LightGBM": "2. LightGBM (Đơn lẻ - Bước 2)",
            "Ensemble_0.5": "3A. Kết hợp 50/50 (Ensemble_0.5)",
            "Ensemble_val": "3B. Kết hợp Tối ưu Val (Ensemble_val)",
        }

        st.markdown("### 3.1. Bảng tổng hợp đầy đủ từ Đơn lẻ đến Kết hợp tại `n = 200` (5 seeds)")
        b3_c1, b3_c2, b3_c3 = st.columns(3)
        for col_ui, title, sc, lv in [
            (b3_c1, "🟢 (a) Dữ liệu Sạch (clean = 0%)", "clean", 0.0),
            (b3_c2, "🟠 (b) Nhiễu nhãn Nặng (label_noise = 30%)", "label_noise", 0.3),
            (b3_c3, "🔴 (c) Mất cân bằng Nặng (imbalance = 2%)", "imbalance", 0.02),
        ]:
            with col_ui:
                st.markdown(f"**{title}**")
                sub_df = ens_runs[
                    (ens_runs["train_size"] == 200)
                    & (ens_runs["scenario"] == sc)
                    & (ens_runs["level"] == lv)
                ]
                tbl = (
                    sub_df.groupby("model")[["roc_auc", "macro_f1", "avg_precision", "log_loss", "ece"]]
                    .mean()
                    .reindex(order_models)
                    .round(4)
                    .reset_index()
                )
                tbl["model"] = tbl["model"].map(model_display_names)
                st.dataframe(tbl, width="stretch", hide_index=True)

        st.markdown("### 3.2. Biểu đồ So sánh Trực quan: Đơn lẻ vs Kết hợp")
        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            sel_metric = st.selectbox(
                "Chọn chỉ số đánh giá trên biểu đồ:",
                ["roc_auc", "avg_precision", "macro_f1", "log_loss", "ece", "brier"],
                index=0,
            )
        with ctrl2:
            sel_scenario = st.selectbox(
                "Chọn kịch bản khảo sát:",
                [
                    "1. Theo cỡ mẫu huấn luyện n = 50, 100, 200, 400 (Dữ liệu sạch)",
                    "2. Theo tỷ lệ Nhiễu nhãn 0%, 5%, 10%, 20%, 30% (tại n = 200)",
                    "3. Theo tỷ lệ Lớp bệnh 30%, 20%, 10%, 5%, 2% (tại n = 200)",
                ],
                index=0,
            )

        fig_e, ax_e = plt.subplots(figsize=(9.5, 4.2))
        colors_map = {
            "TabICLv2": "#1f78b4",
            "LightGBM": "#d95f02",
            "Ensemble_0.5": "#7570b3",
            "Ensemble_val": "#1b9e77",
        }
        if sel_scenario.startswith("1."):
            df_plot = ens_runs[ens_runs["scenario"] == "clean"]
            x_col, x_label = "train_size", "Kích thước mẫu huấn luyện (n)"
        elif sel_scenario.startswith("2."):
            df_plot = ens_runs[
                (ens_runs["train_size"] == 200)
                & (ens_runs["scenario"].isin(["clean", "label_noise"]))
            ]
            x_col, x_label = "level", "Tỷ lệ nhiễu nhãn (Label Noise)"
        else:
            df_plot = ens_runs[
                (ens_runs["train_size"] == 200) & (ens_runs["scenario"] == "imbalance")
            ]
            x_col, x_label = "level", "Tỷ lệ lớp dương thiểu số (Class Imbalance)"

        for m_name in order_models:
            grp = (
                df_plot[df_plot["model"] == m_name]
                .groupby(x_col)[sel_metric]
                .mean()
                .reset_index()
                .sort_values(x_col)
            )
            lw = 2.8 if m_name in ("TabICLv2", "Ensemble_val") else 1.9
            ls = "-" if m_name != "Ensemble_0.5" else "--"
            ax_e.plot(
                grp[x_col],
                grp[sel_metric],
                marker="o",
                linewidth=lw,
                linestyle=ls,
                color=colors_map[m_name],
                label=model_display_names[m_name],
            )

        if sel_scenario.startswith("3."):
            ax_e.invert_xaxis()
        ax_e.set_xlabel(x_label)
        ax_e.set_ylabel(rename_metrics.get(sel_metric, sel_metric))
        ax_e.set_title(f"Từ Bước 1 (TabICLv2) → Bước 2 (LightGBM) → Bước 3 (Kết hợp) theo {sel_metric}")
        ax_e.grid(True, alpha=0.3)
        ax_e.legend()
        st.pyplot(fig_e)

        st.success(
            "✅ **Kết luận rút ra khi kết hợp `LightGBM + TabICLv2` trên Pima:**  \n"
            "1. **Trên dữ liệu sạch (`n = 200`):** Kết hợp cả hai (`Ensemble_val`) giúp nâng nhẹ mọi chỉ số phân lớp lên mức cao nhất: **ROC-AUC = 0.8352** (so với `0.8348` của TabICLv2 và `0.8167` của LightGBM), **Macro-F1 = 0.7288** (so với `0.7256` và `0.7023`), và **AP = 0.7201** (`Ensemble_0.5` đạt **AP = 0.7283**).  \n"
            "2. **Dưới mất cân bằng cực đoan (`2%` lớp bệnh):** `Ensemble_val` đạt **ROC-AUC = 0.7573** và **AP = 0.6186**, vượt qua cả `TabICLv2` đơn lẻ (`0.7564` / `0.6126`) lẫn `LightGBM` đơn lẻ (`0.6636` / `0.5621`).  \n"
            "3. **Dưới nhiễu nhãn nặng (`30%`):** Nếu trộn đều 50/50 (`Ensemble_0.5`), xác suất quá tự tin của `LightGBM` kéo ROC-AUC xuống `0.7176`. Ngược lại, cơ chế chọn trọng số trên Validation (`Ensemble_val`) tự động dồn trọng số về `TabICLv2` ($\\alpha \\approx 0.88$), nhờ đó bảo toàn hoàn toàn độ bền và đạt tương đương thống kê với `TabICLv2` trên toàn bộ 65 điều kiện ($p = 0.7088$)."
        )

        with st.expander("📋 Xem Bảng Kiểm định Cặp Wilcoxon trên toàn bộ 65 điều kiện (Pima Ensemble)"):
            st.dataframe(compute_ensemble_wilcoxon(ens_runs), width="stretch", hide_index=True)

# ============================================================================
# TAB 2: BÀI TOÁN PAYSIM (THEO 3 BƯỚC RÕ RÀNG: TABICLV2 -> LIGHTGBM -> KẾT HỢP)
# ============================================================================
with tab_paysim:
    ps_sum = read_csv_safe(PAYSIM_RESULTS / "summary.csv")
    ps_ens = read_csv_safe(PAYSIM_RESULTS / "ensemble_metrics_summary.csv")
    ps_ens_seeds = read_csv_safe(PAYSIM_RESULTS / "ensemble_summary.csv")

    if ps_sum is None or ps_ens is None:
        st.error("Chưa tìm thấy dữ liệu kết quả PaySim trong `code_paysim/results/`.")
    else:
        st.markdown(
            "**Bối cảnh thực nghiệm PaySim (Tích hợp từ Notebook 09):**  \n"
            "Khảo sát trên **532.909 giao dịch `TRANSFER`** của bộ dữ liệu tài chính di động PaySim (tỷ lệ gian lận thực tế chỉ **0,77%**, chia theo mốc thời gian `step` để chống rò rỉ tương lai). "
            "Thực nghiệm chạy đủ **48 lượt** (`4 mô hình × 3 seeds (42, 43, 44) × 4 cỡ mẫu (500, 1000, 2500, 5000)`)."
        )

        # --------------------------------------------------------------------
        # BƯỚC 1 TRÊN PAYSIM: TABICLV2 ĐƠN LẺ
        # --------------------------------------------------------------------
        st.markdown("## 🟦 BƯỚC 1: Lúc đầu `TabICLv2` đơn lẻ ra số liệu như thế nào trên PaySim?")
        tab_ps_df = (
            ps_sum[ps_sum["model"] == "TabICLv2"][
                [
                    "train_size",
                    "validation_average_precision_mean",
                    "test_average_precision_mean",
                    "test_average_precision_std",
                    "test_roc_auc_mean",
                    "test_precision_mean",
                    "test_recall_mean",
                    "test_f1_mean",
                ]
            ]
            .round(4)
            .rename(
                columns={
                    "train_size": "Cỡ mẫu Train (n)",
                    "validation_average_precision_mean": "Val AP (TB)",
                    "test_average_precision_mean": "Test AP / PR-AUC (TB)",
                    "test_average_precision_std": "Test AP (Std)",
                    "test_roc_auc_mean": "Test ROC-AUC (TB)",
                    "test_precision_mean": "Test Precision",
                    "test_recall_mean": "Test Recall",
                    "test_f1_mean": "Test F1",
                }
            )
        )
        st.dataframe(tab_ps_df, width="stretch", hide_index=True)
        st.info(
            "📌 **Nhận xét Bước 1 (TabICLv2 đơn lẻ trên PaySim):** Khi cỡ mẫu tăng từ `n = 500` lên `n = 5000`, "
            "`TabICLv2` cải thiện mạnh mẽ khả năng xếp hạng giao dịch gian lận: **Test Average Precision (PR-AUC)** tăng từ **0.3667** lên **0.4873**, "
            "và **Test ROC-AUC** tăng từ **0.9205** lên **0.9468**."
        )

        st.divider()

        # --------------------------------------------------------------------
        # BƯỚC 2 TRÊN PAYSIM: SO SÁNH TABICLV2 VỚI LIGHTGBM (VÀ CATBOOST, LR)
        # --------------------------------------------------------------------
        st.markdown("## 🟧 BƯỚC 2: So sánh `TabICLv2` với `LightGBM` đơn lẻ (cùng CatBoost & LogisticRegression)")
        lgb_ps = ps_sum[ps_sum["model"] == "LightGBM"].set_index("train_size")
        tab_ps = ps_sum[ps_sum["model"] == "TabICLv2"].set_index("train_size")
        cat_ps = ps_sum[ps_sum["model"] == "CatBoost"].set_index("train_size")
        lr_ps = ps_sum[ps_sum["model"] == "LogisticRegression"].set_index("train_size")

        step2_rows = []
        for n_sz in [500, 1000, 2500, 5000]:
            ap_t = float(tab_ps.loc[n_sz, "test_average_precision_mean"])
            ap_l = float(lgb_ps.loc[n_sz, "test_average_precision_mean"])
            auc_t = float(tab_ps.loc[n_sz, "test_roc_auc_mean"])
            auc_l = float(lgb_ps.loc[n_sz, "test_roc_auc_mean"])
            f1_t = float(tab_ps.loc[n_sz, "test_f1_mean"])
            f1_l = float(lgb_ps.loc[n_sz, "test_f1_mean"])
            step2_rows.append({
                "Cỡ mẫu (n)": n_sz,
                "Test AP (TabICLv2)": round(ap_t, 4),
                "Test AP (LightGBM)": round(ap_l, 4),
                "Chênh lệch AP (TabICLv2 - LGBM)": f"+{ap_t - ap_l:.4f}",
                "Test AUC (TabICLv2)": round(auc_t, 4),
                "Test AUC (LightGBM)": round(auc_l, 4),
                "Test F1 (TabICLv2)": round(f1_t, 4),
                "Test F1 (LightGBM)": round(f1_l, 4),
                "Test AP (CatBoost)": round(float(cat_ps.loc[n_sz, "test_average_precision_mean"]), 4),
                "Test AP (LogisticRegression)": round(float(lr_ps.loc[n_sz, "test_average_precision_mean"]), 4),
            })

        st.dataframe(pd.DataFrame(step2_rows), width="stretch", hide_index=True)
        st.warning(
            "🔍 **Phát hiện quan trọng ở Bước 2 trên PaySim (Tính chất bổ trợ giữa `TabICLv2` và `LightGBM`):**  \n"
            "- **`TabICLv2` mạnh hơn hẳn về xếp hạng xác suất toàn cục (Test AP / PR-AUC):** Ở cả 4 cỡ mẫu, `TabICLv2` đều có Test AP cao hơn `LightGBM` (tại $n=5000$, `TabICLv2` đạt **0.4873** so với **0.4388** của `LightGBM`, tức cao hơn **+0.0486**).  \n"
            "- **Nhưng `LightGBM` lại ổn định ở ngưỡng cắt cục bộ (Test F1):** Tại $n=2500$ và $n=5000$, `LightGBM` giữ được **Test F1 = 0.5409 – 0.5437** nhờ các quy tắc chia ngưỡng cứng trên đặc trưng số tiền và lịch sử tài khoản. Chính sự khác biệt có tính bù trừ này là lý do thôi thúc ta kết hợp cả hai ở Bước 3!"
        )

        st.divider()

        # --------------------------------------------------------------------
        # BƯỚC 3 TRÊN PAYSIM: KẾT HỢP CẢ HAI LẠI (LIGHTGBM + TABICLV2)
        # --------------------------------------------------------------------
        st.markdown("## 🟩 BƯỚC 3: Kết hợp cả hai lại (`LightGBM + TabICLv2` Ensemble với `α = 0.95`)")
        st.markdown(
            "Trên tập Validation của PaySim, thuật toán quét lưới trọng số tự động chọn **$\\alpha = 0.95$** "
            "(tức **95% xác suất của `TabICLv2` + 5% xác suất của `LightGBM`**). Kết quả đối sánh trước và sau khi kết hợp:"
        )

        ens_ps_idx = ps_ens.set_index("train_size")
        step3_rows = []
        for n_sz in [500, 1000, 2500, 5000]:
            ap_t = float(tab_ps.loc[n_sz, "test_average_precision_mean"])
            ap_l = float(lgb_ps.loc[n_sz, "test_average_precision_mean"])
            ap_e = float(ens_ps_idx.loc[n_sz, "test_average_precision_mean"])
            auc_t = float(tab_ps.loc[n_sz, "test_roc_auc_mean"])
            auc_l = float(lgb_ps.loc[n_sz, "test_roc_auc_mean"])
            auc_e = float(ens_ps_idx.loc[n_sz, "test_roc_auc_mean"])
            f1_t = float(tab_ps.loc[n_sz, "test_f1_mean"])
            f1_l = float(lgb_ps.loc[n_sz, "test_f1_mean"])
            f1_e = float(ens_ps_idx.loc[n_sz, "test_f1_mean"])
            step3_rows.append({
                "Cỡ mẫu (n)": n_sz,
                "1. AP TabICLv2 (Đơn lẻ)": round(ap_t, 4),
                "2. AP LightGBM (Đơn lẻ)": round(ap_l, 4),
                "3. AP Kết hợp (LGBM + TabICLv2)": round(ap_e, 4),
                "Tăng so với TabICLv2": f"+{ap_e - ap_t:.4f}",
                "Tăng so với LightGBM": f"+{ap_e - ap_l:.4f}",
                "ROC-AUC Kết hợp": round(auc_e, 4),
                "F1 TabICLv2": round(f1_t, 4),
                "F1 Kết hợp": round(f1_e, 4),
            })

        df_step3_ps = pd.DataFrame(step3_rows)
        st.dataframe(df_step3_ps, width="stretch", hide_index=True)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_ps1, ax_ps1 = plt.subplots(figsize=(6.5, 4.0))
            x = np.arange(len(df_step3_ps["Cỡ mẫu (n)"]))
            width = 0.26
            ax_ps1.bar(x - width, df_step3_ps["1. AP TabICLv2 (Đơn lẻ)"], width, label="1. TabICLv2 (Đơn lẻ)", color="#1f78b4")
            ax_ps1.bar(x, df_step3_ps["2. AP LightGBM (Đơn lẻ)"], width, label="2. LightGBM (Đơn lẻ)", color="#d95f02")
            ax_ps1.bar(x + width, df_step3_ps["3. AP Kết hợp (LGBM + TabICLv2)"], width, label="3. Kết hợp (0.05 LGBM + 0.95 TabICLv2)", color="#1b9e77")
            ax_ps1.set_xticks(x)
            ax_ps1.set_xticklabels([f"n = {n}" for n in df_step3_ps["Cỡ mẫu (n)"]])
            ax_ps1.set_ylabel("Test Average Precision (PR-AUC)")
            ax_ps1.set_title("So sánh Cột: Bước 1 vs Bước 2 vs Bước 3 trên PaySim")
            ax_ps1.set_ylim(0.30, 0.53)
            ax_ps1.grid(True, axis="y", alpha=0.3)
            ax_ps1.legend()
            st.pyplot(fig_ps1)

        with p_col2:
            fig_ps2, ax_ps2 = plt.subplots(figsize=(6.5, 4.0))
            n_vals = df_step3_ps["Cỡ mẫu (n)"]
            ax_ps2.plot(n_vals, df_step3_ps["1. AP TabICLv2 (Đơn lẻ)"], marker="o", linewidth=2.2, color="#1f78b4", label="1. TabICLv2 (Đơn lẻ)")
            ax_ps2.plot(n_vals, df_step3_ps["2. AP LightGBM (Đơn lẻ)"], marker="s", linewidth=2.0, linestyle="--", color="#d95f02", label="2. LightGBM (Đơn lẻ)")
            ax_ps2.plot(n_vals, df_step3_ps["3. AP Kết hợp (LGBM + TabICLv2)"], marker="^", linewidth=2.8, color="#1b9e77", label="3. Kết hợp (LightGBM + TabICLv2)")
            ax_ps2.set_xlabel("Kích thước tập huấn luyện (n)")
            ax_ps2.set_ylabel("Test Average Precision (PR-AUC)")
            ax_ps2.set_title("Đường cong Học: Mô hình Kết hợp dẫn đầu ở mọi cỡ mẫu")
            ax_ps2.grid(True, alpha=0.3)
            ax_ps2.legend()
            st.pyplot(fig_ps2)

        st.success(
            "🏆 **Kết luận Bước 3 trên PaySim:** Khi kết hợp **95% `TabICLv2` + 5% `LightGBM`**, mô hình lai (Ensemble) "
            "**vượt qua cả `TabICLv2` đơn lẻ lẫn `LightGBM` đơn lẻ ở tất cả 4 cỡ mẫu huấn luyện**:  \n"
            "- Tại $n=500$: AP tăng từ `0.3667` (TabICLv2) và `0.3607` (LightGBM) lên **`0.3708`**.  \n"
            "- Tại $n=1000$: AP tăng từ `0.4576` và `0.4111` lên **`0.4606`**.  \n"
            "- Tại $n=2500$: AP tăng từ `0.4483` và `0.4397` lên **`0.4582`** (đồng thời F1 tăng từ `0.3534` lên `0.3819`).  \n"
            "- Tại $n=5000$: AP đạt đỉnh **`0.4903`** (+0.0030 so với TabICLv2 và +0.0515 so với LightGBM), cao nhất trong toàn bộ 48 lượt thực nghiệm PaySim."
        )

        if ps_ens_seeds is not None:
            with st.expander("📋 Xem chi tiết 12 lượt kết hợp `LightGBM + TabICLv2` theo từng Seed (42, 43, 44)"):
                st.dataframe(ps_ens_seeds.round(4), width="stretch", hide_index=True)

# ============================================================================
# TAB 3: BENCHMARK 5 BỘ DỮ LIỆU (800 LƯỢT) & ĐỘ BỀN 8 MÔ HÌNH (400 LƯỢT)
# ============================================================================
with tab_bench:
    st.header("📈 Benchmark 8 Mô hình trên 5 Bộ dữ liệu (800/800 lượt)")
    bench_sum = read_csv_safe(RESULTS / "benchmark" / "summary.csv")
    bench_pairs = read_csv_safe(RESULTS / "benchmark" / "paired_tests.csv")

    if bench_sum is not None:
        st.markdown("### 1. Đối chiếu nhanh `TabICLv2` vs `LightGBM` trên cả 5 Bộ dữ liệu (tại cỡ mẫu `n = 200`)")
        b200 = bench_sum[bench_sum["train_size"] == 200]
        quick_rows = []
        for ds_name in ["pima", "breast_cancer", "banknote", "wine", "magic"]:
            t_row = b200[(b200["dataset"] == ds_name) & (b200["model"] == "TabICLv2")]
            l_row = b200[(b200["dataset"] == ds_name) & (b200["model"] == "LightGBM")]
            if not t_row.empty and not l_row.empty:
                auc_t = float(t_row["roc_auc_mean"].iloc[0])
                auc_l = float(l_row["roc_auc_mean"].iloc[0])
                f1_t = float(t_row["macro_f1_mean"].iloc[0])
                f1_l = float(l_row["macro_f1_mean"].iloc[0])
                ll_t = float(t_row["log_loss_mean"].iloc[0])
                ll_l = float(l_row["log_loss_mean"].iloc[0])
                quick_rows.append({
                    "Bộ dữ liệu": ds_name,
                    "ROC-AUC (TabICLv2)": round(auc_t, 4),
                    "ROC-AUC (LightGBM)": round(auc_l, 4),
                    "Chênh lệch AUC (TabICLv2 - LGBM)": f"{auc_t - auc_l:+.4f}",
                    "Macro-F1 (TabICLv2)": round(f1_t, 4),
                    "Macro-F1 (LightGBM)": round(f1_l, 4),
                    "Log-Loss (TabICLv2 ↓)": round(ll_t, 4),
                    "Log-Loss (LightGBM ↓)": round(ll_l, 4),
                })
        st.dataframe(pd.DataFrame(quick_rows), width="stretch", hide_index=True)

        st.markdown("### 2. Đường cong Học theo Cỡ mẫu (`n = 50, 100, 200, 400`) của toàn bộ 8 Mô hình")
        b_col1, b_col2 = st.columns(2)
        ds_choice = b_col1.selectbox(
            "Chọn bộ dữ liệu:",
            ["pima", "breast_cancer", "banknote", "wine", "magic"],
            index=0,
        )
        metric_choice = b_col2.selectbox(
            "Chọn chỉ số đánh giá:",
            ["roc_auc", "accuracy", "macro_f1", "avg_precision", "log_loss", "ece", "brier"],
            index=0,
        )

        sub = bench_sum[bench_sum["dataset"] == ds_choice].copy()
        mean_col = f"{metric_choice}_mean"

        fig_b, ax_b = plt.subplots(figsize=(9.5, 4.2))
        for m_name, grp in sub.groupby("model"):
            grp_s = grp.sort_values("train_size")
            lw = 2.8 if m_name in ("TabICLv2", "LightGBM") else (2.2 if m_name in ("TabPFNv2", "TabDPT") else 1.4)
            ls = "-" if m_name in ("TabICLv2", "TabPFNv2", "TabDPT") else "--"
            ax_b.plot(grp_s["train_size"], grp_s[mean_col], marker="o", linewidth=lw, linestyle=ls, label=m_name)
        ax_b.set_title(f"Đường cong học 8 mô hình trên bộ `{ds_choice}` ({metric_choice})")
        ax_b.set_xlabel("Kích thước tập huấn luyện (n)")
        ax_b.set_ylabel(metric_choice)
        ax_b.grid(True, alpha=0.3)
        ax_b.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig_b)

        st.markdown(f"### 3. Bảng chi tiết 8 mô hình trên bộ `{ds_choice}` (`results/benchmark/summary.csv`)")
        cols_show = ["model", "train_size", "accuracy_mean", "macro_f1_mean", "roc_auc_mean", "avg_precision_mean", "log_loss_mean", "ece_mean"]
        st.dataframe(
            sub[[c for c in cols_show if c in sub.columns]].sort_values(["train_size", "roc_auc_mean"], ascending=[True, False]).round(4),
            width="stretch",
            hide_index=True,
        )

    if bench_pairs is not None:
        st.markdown("### 4. Kiểm định cặp Wilcoxon trên toàn bộ 100 điều kiện Benchmark (`TabICLv2` vs 7 mô hình)")
        st.dataframe(bench_pairs, width="stretch", hide_index=True)

    st.divider()
    st.markdown("### 5. Độ bền Nhiễu nhãn (5–30%) & Mất cân bằng (30%→2%) trên 8 Mô hình (400 lượt)")
    img_col1, img_col2 = st.columns(2)
    noise_png = RESULTS / "robustness" / "pima_label_noise.png"
    imb_png = RESULTS / "robustness" / "pima_imbalance.png"
    if noise_png.is_file():
        img_col1.image(str(noise_png), caption="Độ bền khi tăng tỷ lệ nhiễu nhãn (5% -> 30%) trên Pima (n=200)")
    if imb_png.is_file():
        img_col2.image(str(imb_png), caption="Độ bền khi lớp thiểu số giảm từ 30% xuống 2% trên Pima (n=200)")

    rob_pairs = read_csv_safe(RESULTS / "robustness" / "paired_tests.csv")
    if rob_pairs is not None:
        with st.expander("📋 Xem Kiểm định cặp Wilcoxon dưới Nhiễu nhãn & Mất cân bằng (`results/robustness/paired_tests.csv`)"):
            st.dataframe(rob_pairs, width="stretch", hide_index=True)

# ============================================================================
# TAB 4: CHẠY SMOKE TEST CLI & SƠ ĐỒ MÃ NGUỒN ĐỂ THUYẾT TRÌNH
# ============================================================================
with tab_cli:
    st.subheader("⚡ Chạy Trực tiếp Chế độ `--smoke` & Bản đồ Mã nguồn (Dùng khi Bảo vệ)")
    st.markdown("Nhấn nút bên dưới để thực thi trực tiếp lệnh CLI từ giao diện và xem kết quả trả về:")

    b1, b2, b3, b4 = st.columns(4)
    cmd_to_run = None
    if b1.button("1️⃣ Chạy `run_all.py check` (~3s)"):
        cmd_to_run = [sys.executable, "run_all.py", "check"]
    if b2.button("2️⃣ Chạy `ensemble --smoke` (~15s)"):
        cmd_to_run = [sys.executable, "run_all.py", "ensemble", "--smoke"]
    if b3.button("3️⃣ Chạy `benchmark --smoke` (~40s)"):
        cmd_to_run = [sys.executable, "run_all.py", "benchmark", "--smoke"]
    if b4.button("4️⃣ Chạy `paysim --smoke --skip-tabicl` (~8s)"):
        cmd_to_run = [sys.executable, "code_paysim/run_paysim.py", "--smoke", "--skip-tabicl"]

    if cmd_to_run is not None:
        with st.spinner(f"Đang chạy: {' '.join(cmd_to_run)} ..."):
            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd_to_run,
                cwd=CODE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.perf_counter() - t0
        st.code(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), language="text")
        st.info(f"Hoàn tất trong {elapsed:.1f} giây (Exit code: {proc.returncode}).")

    st.markdown(
        """
        ---
        ### 🗺️ Chỉ dẫn Mở Code khi Thuyết trình trước Giảng viên
        1. **`Tabular-Foudation/run_all.py` (Dòng 48–98):** Điểm vào điều phối toàn bộ 5 lệnh (`check`, `benchmark`, `robustness`, `ensemble`, `paysim`, `report`) và cờ `--smoke` / `--resume`.
        2. **`Tabular-Foudation/tfm/experiments.py` (Dòng 89–155):** Hàm `run_ensemble()` thực hiện kết hợp xác suất `LightGBM + TabICLv2` theo cả 2 cơ chế: `Ensemble_0.5` (cố định $\\alpha=0.5$) và `Ensemble_val` (tối ưu $\\alpha$ trên Validation sạch).
        3. **`Tabular-Foudation/tfm/models.py` (Dòng 68–134):** Hàm `build()` và `fit_predict()` khởi tạo 3 mô hình nền tảng (`TabICLv2`, `TabPFNv2`, `TabDPT`) chạy trên CPU cùng 5 mô hình truyền thống.
        4. **`Tabular-Foudation/code_paysim/paysim/experiment.py` (Dòng 131–165):** Hàm `blend_ensemble()` kết hợp `LightGBM + TabICLv2` trên tập PaySim.
        """
    )
