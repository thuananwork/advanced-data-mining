# Mô Hình Nền Tảng Cho Dữ Liệu Bảng Trong Bối Cảnh Ít Mẫu (Tabular Foundation Models in Few-Shot Regimes)

Đồ án môn học **Khám phá tri thức và Khai thác dữ liệu (GCOS108)** – Chương trình Thạc sĩ Công nghệ Thông tin, Trường Đại học Công nghệ TP.HCM (HUTECH).

* **Học viên thực hiện:**
  * Phan Chí Thanh – MSSV: `2541863049`
  * Nguyễn Thuận An – MSSV: `2541863001`
* **Lớp:** `25SCT33`
* **Giảng viên hướng dẫn:** TS. Bùi Danh Hường

---

## 1. Giới thiệu tổng quan

Dự án nghiên cứu và thực nghiệm toàn diện trên CPU các **Mô hình nền tảng cho dữ liệu bảng (Tabular Foundation Models – TFMs)** dựa trên học trong ngữ cảnh (*In-Context Learning – ICL*) và mạng khớp dữ liệu tiên nghiệm (*Prior-Data Fitted Networks – PFNs*), trọng tâm là **TabICLv2**, đặt trong phép đối chiếu công bằng với:
* **Các mô hình nền tảng SOTA cùng họ:** `TabPFNv2`, `TabDPT`, `LoCalPFN`.
* **Các mô hình cây quyết định tăng cường gradient (GBDTs) và cổ điển:** `LightGBM` (đối chứng chính), `XGBoost`, `CatBoost`, `Random Forest`, `Logistic Regression`.
* **Mô hình kết hợp (Hybrid Probability Blending):** Trộn xác suất có trọng số giữa `LightGBM` và `TabICLv2` theo tập validation sạch:

$$p_{\text{blend}}(x) = (1 - \alpha)\,p_{\text{LightGBM}}(x) + \alpha\,p_{\text{TabICLv2}}(x), \quad \alpha \in [0, 1]$$

Dự án được tổ chức xuyên suốt theo **tiến trình 3 bước**:
1. **Bước 1 – Đánh giá `TabICLv2` đơn lẻ:** Khả năng học trong ngữ cảnh không cần cập nhật gradient khi ít mẫu ($n \in \{50, 100, 200, 400\}$ và mở rộng tới $n = 5.000$), chất lượng hiệu chuẩn xác suất (`Log-loss`, `ECE`, `Brier score`) và độ bền trước nhiễu nhãn / mất cân bằng lớp.
2. **Bước 2 – So sánh trực tiếp với `LightGBM` và 6 mô hình đối chứng:** Kiểm định cặp phi tham số Wilcoxon trên 100 điều kiện benchmark và 50 điều kiện nhiễu/lệch lớp.
3. **Bước 3 – Kết hợp `LightGBM + TabICLv2` (`Ensemble_val` vs. `Ensemble_0.5`):** Chứng minh việc chọn trọng số $\alpha^*$ tự động trên tập validation sạch giúp cải thiện đỉnh `Average Precision` trên bài toán phát hiện gian lận `PaySim` đồng thời tự động hạ trọng số của `LightGBM` khi nhãn bị nhiễu trên `Pima Diabetes`, giữ vững độ bền thống kê của `TabICLv2`.

---

## 2. Bộ dữ liệu thực nghiệm (6 bộ dữ liệu)

Toàn bộ dữ liệu được lưu trữ phân tách theo từng thư mục con trong `datasets/`:

| Bộ dữ liệu | Thư mục | Số mẫu ($N$) | Đặc trưng ($d$) | Tỷ lệ lớp dương | Giao thức chia tập & Thực nghiệm |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pima Indians Diabetes** | `datasets/pima/` | 768 | 8 | 34,9% | Chia nhóm phân tầng (`StratifiedGroupKFold`, 5 seed); xử lý `0 -> NaN` ở 5 cột lâm sàng; Benchmark + Độ bền nhiễu nhãn ($5\%\text{–}30\%$) + Lệch lớp ($30\%\text{–}2\%$) + Ensemble |
| **PaySim Mobile Fraud** | `datasets/paysim/` | 532.909 | 14 | 1,39% (6,88% ở test) | Chia theo mốc thời gian (`step <= 445` pool, `446–594` val, `> 594` test = 12.029 giao dịch); $n \in \{500, 1.000, 2.500, 5.000\} \times 3$ seed + Blend |
| **Breast Cancer Wisconsin** | `datasets/breast_cancer/` | 569 | 30 | 62,7% | Khám phá ban đầu ($n \in \{50, 100\}$, seed 42) & Benchmark 5 seed ($n \in \{50, 100, 200, 400\}$) |
| **Banknote Authentication** | `datasets/banknote/` | 1.372 | 4 | 44,5% | Kiểm toán rò rỉ bản sao đặc trưng (24 dòng trùng) & Benchmark chia theo nhóm (5 seed) |
| **White Wine Quality (Binary)** | `datasets/wine/` | 4.898 | 11 | 66,5% | Nhị phân hóa `quality >= 6`; loại bỏ giao nhóm (937 dòng trùng); Khám phá ($n \le 1.000$) & Benchmark 5 seed |
| **MAGIC Gamma Telescope** | `datasets/magic/` | 19.020 | 10 | 64,8% | Loại bỏ giao nhóm (115 dòng trùng); Khám phá ($n \le 5.000$) & Benchmark 5 seed ($n \le 400$) |

> **Lưu ý về file dữ liệu PaySim:** Do tệp gốc `PS_20174392719_1491204439457_log.csv` có dung lượng ~470 MB (vượt giới hạn 100 MB của GitHub), tệp này được đưa vào `.gitignore`. Để chạy lại từ dữ liệu thô PaySim, tải bộ dữ liệu từ [Kaggle PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) và đặt vào `datasets/paysim/PS_20174392719_1491204439457_log.csv`. Toàn bộ kết quả chạy hoàn chỉnh (48 lượt cơ sở + 12 lượt Blend + 18 lượt tuning) đã được lưu sẵn trong `Tabular-Foudation/code_paysim/results/`.

---

## 3. Tóm tắt kết quả thực nghiệm chính (1.536 lượt đánh giá)

### 3.1. Benchmark 5 bộ dữ liệu ít mẫu (800 lượt chạy – 5 bộ $\times$ 8 mô hình $\times$ 4 cỡ mẫu $\times$ 5 seed)
Kết quả trung bình `ROC-AUC` trên tập kiểm tra và kiểm định cặp Wilcoxon ($N = 100$ cặp điều kiện so với `TabICLv2`):

| Mô hình | $n = 50$ | $n = 100$ | $n = 200$ | $n = 400$ | Trung bình | $\Delta\text{ROC-AUC}$ (TabICLv2 $-$ đối thủ) | $\Delta\text{Log-loss}$ | $\Delta\text{ECE}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TabDPT** | **0,881** | **0,900** | **0,909** | **0,917** | **0,902** | $-0,005$ ($p = 0,018$) | $+0,008$ ($p = 0,095$) | $-0,000$ ($p = 0,433$) |
| **TabICLv2** | 0,871 | 0,894 | 0,907 | 0,916 | 0,897 | — | — | — |
| **TabPFNv2** | 0,876 | 0,895 | 0,904 | 0,914 | 0,897 | $-0,000$ ($p = 0,064$) | $+0,000$ ($p = 0,080$) | $+0,001$ ($p = 0,864$) |
| **Random Forest** | 0,860 | 0,883 | 0,895 | 0,906 | 0,886 | $+0,011$ ($p < 0,001$) | $-0,041$ ($p < 0,001$) | $-0,014$ ($p < 0,001$) |
| **CatBoost** | 0,853 | 0,877 | 0,893 | 0,904 | 0,882 | $+0,015$ ($p < 0,001$) | $-0,095$ ($p < 0,001$) | $-0,043$ ($p < 0,001$) |
| **Logistic Regression** | 0,861 | 0,874 | 0,883 | 0,891 | 0,877 | $+0,020$ ($p < 0,001$) | $-0,044$ ($p < 0,001$) | $-0,018$ ($p < 0,001$) |
| **XGBoost** | 0,848 | 0,873 | 0,885 | 0,899 | 0,876 | $+0,021$ ($p < 0,001$) | $-0,091$ ($p < 0,001$) | $-0,044$ ($p < 0,001$) |
| **LightGBM** | 0,844 | 0,869 | 0,882 | 0,897 | 0,873 | **$+0,024$ ($p < 0,001$)** | **$-0,505$ ($p < 0,001$)** | **$-0,094$ ($p < 0,001$)** |

### 3.2. Độ bền trước nhiễu nhãn & mất cân bằng lớp và Tổ hợp `LightGBM + TabICLv2` trên Pima Diabetes (400 lượt độ bền + 260 dòng Ensemble)
Tại $n = 200$ (trung bình 5 seed, tập kiểm tra 154 mẫu luôn sạch):

| Mô hình | Sạch ($n=200$) | Nhiễu 5% | Nhiễu 10% | Nhiễu 20% | Nhiễu 30% | Lệch lớp 30% | Lệch lớp 20% | Lệch lớp 10% | Lệch lớp 5% | Lệch lớp 2% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TabICLv2** | 0,835 | 0,825 | 0,835 | 0,784 | 0,750 | 0,831 | 0,827 | 0,822 | 0,809 | 0,756 |
| **LightGBM** | 0,800 | 0,779 | 0,750 | 0,705 | 0,598 | 0,816 | 0,799 | 0,776 | 0,718 | 0,752 |
| **Ensemble_0.5** ($\alpha = 0,5$) | 0,835 | 0,812 | 0,818 | 0,750 | 0,718 | 0,824 | 0,809 | 0,809 | 0,789 | **0,761** |
| **Ensemble_val** ($\alpha^*$ theo val) | **0,835** | **0,825** | **0,835** | **0,786** | **0,750** | **0,833** | **0,825** | **0,822** | **0,809** | 0,757 |
| *Trọng số $\alpha^*$ trung bình* | *0,89* | *0,96* | *0,99* | *0,97* | *1,00* | *0,90* | *0,75* | *0,96* | *0,89* | *0,68* |

* Trên 50 điều kiện nhiễu/lệch lớp, `TabICLv2` vượt `LightGBM` **+0,058 ROC-AUC** ($p < 0,001$), giảm **1,063 Log-loss** ($p < 0,001$) và giảm **0,102 ECE** ($p < 0,001$).
* Trên toàn bộ 65 điều kiện kết hợp (`results/ensemble/paired_tests.csv`), trộn ngây thơ `Ensemble_0.5` bị kéo tụt **$-0,0090$ ROC-AUC** ($p = 0,00059$) do xác suất quá tự tin của `LightGBM` dưới nhiễu nhãn; trong khi `Ensemble_val` tự động đẩy $\alpha^* \to 0,96\text{–}1,00$ khi có nhiễu, giữ độ bền tương đương thống kê với `TabICLv2` ($p = 0,7088$) và cải thiện đỉnh ở dữ liệu sạch (`ROC-AUC = 0,8352`, `Macro-F1 = 0,7288`) cũng như ở mức lệch lớp cực đoan 2% (`ROC-AUC = 0,7573`).

### 3.3. Phát hiện gian lận giao dịch PaySim (48 lượt cơ sở + 12 lượt Blend)
Trung bình $\pm$ độ lệch chuẩn `Average Precision` (PR-AUC, %) qua 3 seed trên cửa sổ kiểm tra tương lai (`step > 594`, 12.029 giao dịch):

| Cỡ tập huấn luyện ($n$) | Logistic Regression | CatBoost | LightGBM | TabICLv2 | **Blend (`LightGBM + TabICLv2`, $\alpha = 0,95$)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **500** | 33,27 ± 11,51 | 31,68 ± 4,33 | 36,07 ± 8,09 | 36,67 ± 8,48 | **37,08 ± 8,32** |
| **1.000** | 37,74 ± 2,73 | 41,47 ± 7,13 | 41,11 ± 4,46 | 45,76 ± 10,60 | **46,06 ± 10,61** |
| **2.500** | 39,02 ± 7,60 | 43,39 ± 7,21 | 43,97 ± 3,74 | 44,83 ± 7,75 | **45,82 ± 6,90** |
| **5.000** | 47,23 ± 4,85 | 46,37 ± 4,21 | 43,88 ± 0,58 | 48,73 ± 5,32 | **49,03 ± 5,58** |

---

## 4. Cấu trúc thư mục dự án

```text
advanced-data-mining/
├── datasets/                        # 6 bộ dữ liệu chia theo từng thư mục riêng
│   ├── pima/                        # pima_diabetes.csv (768 x 8)
│   ├── paysim/                      # PS_20174392719_1491204439457_log.csv (tải từ Kaggle)
│   ├── breast_cancer/               # breast_cancer_wisconsin.csv (569 x 30)
│   ├── banknote/                    # banknote_authentication.csv (1.372 x 4)
│   ├── wine/                        # wine_quality_white_binary.csv (4.898 x 11)
│   └── magic/                       # magic_gamma_telescope.csv (19.020 x 10)
├── results/                         # Toàn bộ kết quả thực nghiệm (CSV + biểu đồ)
│   ├── benchmark/                   # 800 lượt benchmark (runs.csv, summary.csv, paired_tests.csv)
│   ├── robustness/                  # 400 lượt độ bền Pima (runs.csv, summary.csv, paired_tests.csv)
│   ├── ensemble/                    # 260 dòng kết hợp LightGBM + TabICLv2 trên Pima
│   ├── figures/                     # Các biểu đồ tổng hợp tự động từ CSV
│   └── *.csv                        # Kết quả 28 lượt thăm dò ban đầu
├── splits/                          # Chỉ số chia tập cố định (JSON) của đợt thăm dò seed 42
├── notebooks/                       # 9 Jupyter Notebooks thực nghiệm & phân tích từng bước
├── docs/                            # Tài liệu quy trình và nhật ký tiến độ
└── Tabular-Foudation/               # Mã nguồn chính của hệ thống thực nghiệm & ứng dụng Demo
    ├── tfm/                         # Package lõi: nạp dữ liệu, chia nhóm, mô hình, độ đo, biểu đồ
    ├── code_paysim/                 # Pipeline thực nghiệm riêng cho bài toán gian lận PaySim
    │   ├── paysim/                  # Các module xử lý dữ liệu theo thời gian, mô hình & blend
    │   ├── results/                 # Kết quả 48 lượt chạy PaySim + 12 lượt Blend + biểu đồ
    │   └── run_paysim.py            # CLI chạy thực nghiệm PaySim
    ├── vendor/LoCalPFN/             # Mã nguồn nghiên cứu kiến trúc LoCalPFN
    ├── run_all.py                   # CLI chạy toàn bộ Benchmark, Độ bền, Ensemble và vẽ biểu đồ
    ├── demo_app.py                  # Ứng dụng Web Streamlit trình diễn trực quan 3 bước
    └── requirements.txt             # Danh sách thư viện Python
```

---

## 5. Hướng dẫn cài đặt và chạy thực nghiệm

### 5.1. Cài đặt môi trường ảo
Yêu cầu **Python 3.11 – 3.14** trên Windows, macOS hoặc Linux (chạy hoàn toàn trên CPU):

```bash
cd Tabular-Foudation
python -m venv .venv

# Kích hoạt môi trường ảo (Windows PowerShell):
.\.venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit
```

### 5.2. Khởi chạy ứng dụng Demo tương tác (Streamlit)
Ứng dụng web trình diễn trực quan câu chuyện thực nghiệm 3 bước (`TabICLv2` $\to$ `So sánh LightGBM` $\to$ `Kết hợp LightGBM + TabICLv2`) cùng tính năng suy luận trực tiếp (*Live Inference*) trên từng bệnh nhân Pima Diabetes:

```bash
streamlit run Tabular-Foudation/demo_app.py
```

### 5.3. Chạy lại hoặc mở rộng thực nghiệm từ dòng lệnh
Tất cả các kịch bản đều hỗ trợ cơ chế `--resume` (tự động bỏ qua các lượt đã có `status=ok` trong CSV):

```bash
# 1. Chạy thử nhanh (smoke test) để kiểm tra đường ống:
python Tabular-Foudation/run_all.py --smoke

# 2. Chạy toàn bộ Benchmark (800 lượt), Độ bền Pima (400 lượt), và xuất bảng/hình:
python Tabular-Foudation/run_all.py --mode all

# 3. Chạy riêng thực nghiệm kết hợp LightGBM + TabICLv2 trên Pima (260 dòng):
python Tabular-Foudation/run_all.py --mode ensemble

# 4. Chỉ tổng hợp lại các bảng thống kê Wilcoxon và vẽ lại biểu đồ từ CSV sẵn có:
python Tabular-Foudation/run_all.py --mode report

# 5. Chạy thực nghiệm phát hiện gian lận PaySim (smoke test hoặc đầy đủ):
python Tabular-Foudation/code_paysim/run_paysim.py --smoke
python Tabular-Foudation/code_paysim/run_paysim.py
```

---

## 6. Tài liệu tham khảo chính

1. S. Müller et al., *"Transformers can do Bayesian inference,"* ICLR, 2022.
2. N. Hollmann et al., *"TabPFN: A transformer that solves small tabular classification problems in a second,"* ICLR, 2023.
3. N. Hollmann et al., *"Accurate predictions on small data with a tabular foundation model,"* Nature, vol. 637, pp. 319–326, 2025.
4. J. Qu et al., *"TabICL: A tabular foundation model for in-context learning on large data,"* ICML, 2025.
5. J. Qu et al., *"TabICLv2: A better, faster, scalable, and open tabular foundation model,"* arXiv:2602.11139, 2026.
6. J. Ma et al., *"TabDPT: Scaling tabular foundation models on real data,"* NeurIPS, 2025.
7. V. Thomas et al., *"Retrieval & fine-tuning for in-context tabular models (LoCalPFN),"* NeurIPS, 2024.
8. G. Ke et al., *"LightGBM: A highly efficient gradient boosting decision tree,"* NeurIPS, 2017.
