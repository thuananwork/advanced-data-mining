# Tabular-Foudation: Code Thực nghiệm & Giao diện Báo cáo

Thư mục này chứa toàn bộ mã nguồn thực nghiệm và giao diện Web của đề tài. Dữ liệu, kết quả và chia tập nằm ở gốc dự án: `../datasets`, `../results`, `../splits`.

## 1. Khởi chạy Nhanh (Windows)

```powershell
cd D:\ThacSi\KhamPhaTriThucVaKhaiThacDuLieu\DoAnMonHoc\advanced-data-mining\Tabular-Foudation
.\run_demo.bat              # Mở giao diện Web trực quan (http://localhost:8501)
.\run.bat                   # Chạy toàn bộ pipeline thực nghiệm (có --resume)
.\run.bat check             # Kiểm tra môi trường, thư viện và 6 bộ dữ liệu
.\run.bat ensemble --smoke  # Chạy thử nhanh LightGBM + TabICLv2 (~15 giây)
```

## 2. Các Lệnh trong `run_all.py`

| Lệnh | Nội dung thực nghiệm | Thư mục kết quả |
| :--- | :--- | :--- |
| `check` | Kiểm tra thư viện và 6 bộ dữ liệu đầu vào | Màn hình console |
| `ensemble` | Kết hợp `LightGBM + TabICLv2` trên Pima (260 dòng: Sạch, Nhiễu nhãn, Mất cân bằng) | `../results/ensemble/` |
| `paysim` | Tích hợp Notebook 09: `LogisticRegression`, `CatBoost`, `LightGBM`, `TabICLv2` & Ensemble (48 lượt) | `code_paysim/results/` |
| `benchmark` | 8 mô hình × 5 bộ dữ liệu × 5 seeds × 4 cỡ mẫu (800 lượt) | `../results/benchmark/` |
| `robustness` | Độ bền 8 mô hình trên Pima dưới nhiễu nhãn 5–30% & mất cân bằng 30%→2% (400 lượt) | `../results/robustness/` |
| `report` | Xuất bảng `summary.csv`, `delta_vs_clean.csv`, `paired_tests.csv` và biểu đồ `.png` | `../results/*/` |

## 3. Cấu trúc Thư mục Code

- `demo_app.py`, `run_demo.bat`: Giao diện Web trình bày 3 bước (`TabICLv2` $\rightarrow$ So sánh `LightGBM` $\rightarrow$ Kết hợp `LightGBM + TabICLv2`).
- `run_all.py`, `run.bat`, `run_windows.ps1`: Điều phối chạy toàn bộ thực nghiệm.
- `tfm/`: Gói thực nghiệm chính (`data.py`, `splits.py`, `perturb.py`, `models.py`, `metrics.py`, `experiments.py`, `report.py`).
- `code_paysim/`: Gói thực nghiệm gian lận tài chính PaySim tách từ Notebook 09 (`run_paysim.py`, `paysim/`, `results/`, `tests/`).
- `tools/`, `vendor/LoCalPFN/`: Công cụ hỗ trợ cho Notebook 04.
