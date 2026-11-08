# Mô hình nền tảng cho dữ liệu bảng trong bối cảnh ít mẫu

Đồ án **Khám phá tri thức và Khai thác dữ liệu (GCOS108)** tại HUTECH. Dự án lấy **TabPFNv2** làm bài nền tảng, so sánh **TabICLv2**, **TabDPT** và các mô hình đối chứng trên CPU. Phần mở rộng kết hợp xác suất của **LightGBM** và **TabICLv2**, với trọng số chọn trên tập validation.

## Thực nghiệm

| Thực nghiệm | Kết quả chính |
| --- | --- |
| [Benchmark](results/benchmark/) | Trên Breast Cancer, Banknote, White Wine, MAGIC và Pima với 50–400 mẫu huấn luyện, ROC-AUC trung bình của TabDPT là 0,902, TabICLv2 và TabPFNv2 cùng đạt 0,897 sau khi làm tròn. |
| [Độ bền](results/robustness/) và [kết hợp trên Pima](results/ensemble/) | Với 30% nhiễu nhãn, phương án chọn trọng số trên validation chọn dùng hoàn toàn TabICLv2 ở cả 5 seed. |
| [Phát hiện gian lận trên PaySim](Tabular-Foudation/code_paysim/results/) | Ở 5.000 mẫu huấn luyện, tổ hợp đạt AP trung bình 49,03% so với 48,73% của TabICLv2 qua 3 seed. |

## Chạy thử

Dùng Python 3.11–3.13. Chạy từ gốc repo trong PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Tabular-Foudation/requirements.txt streamlit

$env:PYTHONUTF8 = "1"
python Tabular-Foudation/run_all.py check
python Tabular-Foudation/run_all.py benchmark --smoke --models LightGBM
python -m streamlit run Tabular-Foudation/demo_app.py
```

Trên Linux hoặc macOS, kích hoạt môi trường bằng `source .venv/bin/activate`. Lần đầu chạy mô hình nền tảng cần mạng để tải checkpoint.

Các lệnh `benchmark`, `robustness`, `ensemble`, `paysim`, `report` và `all` chạy qua [run_all.py](Tabular-Foudation/run_all.py). Để chạy lại PaySim, tải [PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) và đặt tệp CSV gốc tại `datasets/paysim/PS_20174392719_1491204439457_log.csv`.

Mã nguồn nằm trong [Tabular-Foudation](Tabular-Foudation/), các bước phân tích nằm trong [notebooks](notebooks/).

**Thực hiện** · Nguyễn Thuận An, Phan Chí Thanh<br>
**Hướng dẫn** · TS. Bùi Danh Hường
