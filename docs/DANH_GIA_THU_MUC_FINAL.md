# Đọc kỹ thư mục Final (bản làm bằng ChatGPT)

Ngày: 20/10/2026 · Thư mục: DoAnMonHoc\Final (report/, source-code/)

## 1. Thư mục này có gì

- **report/**: bài tổng quan Word `Survey_TFM_SmallData_2026.docx` (khoảng 640 dòng, 20 công trình + 6 nền tảng), bộ 20 slide, `Khao_sat_4_SOTA_TFM.docx` (khảo sát bốn phương pháp), `Kich_ban_thuyet_trinh_giua_ky.docx` (kịch bản nói và câu hỏi thầy có thể hỏi).
- **source-code/Tabular_4SOTA_CPU**: 5 notebook chạy TabICLv2, TabPFNv2, TabDPT, LoCalPFN và so sánh. Dữ liệu là Breast Cancer Wisconsin (569 dòng), kết quả nằm trong `results/*.csv`.
- **source-code/tabicl_official**: bản sao mã chính thức của TabICL (commit ngày 09/10/2026).
- **source-code/talent_benchmark**: một script chạy nhiều bộ dữ liệu TALENT cho kế hoạch cuối kỳ.
- **_to_delete/**: bản cũ của báo cáo và slide, cùng log; chưa động tới.

## 2. Phần đã kiểm tra và khớp

Mình đối chiếu trực tiếp với các PDF gốc trong thư mục 60_paper:

- Các số công bố đều đúng: TabPFNv2 0,939 so với 0,752 của CatBoost (Nature); TabICLv2 hạng 4,66 trên TALENT và 4,82 trên TabArena; TabDPT 0,884 so với 0,872 của TabPFNv2 (độ chính xác, OpenML-CC18); LoCalPFN IQM AUC 0,958 so với 0,917 (TabPFN) và 0,943 (XGBoost); TabICL 200 bộ TALENT, 171 bộ có tối đa 10 lớp, nhanh tới 10 lần.
- Các bảng kết quả trong báo cáo và slide khớp từng chữ số với các file CSV.
- Cách chia dữ liệu đúng như mô tả: 364/91/114 dòng, không giao nhau, tập 50 mẫu nằm trong tập 100 mẫu.
- Mình chạy lại ba mô hình truyền thống trên đúng cách chia đó và ra đúng số trong báo cáo: hồi quy logistic 111 và 110, rừng ngẫu nhiên 109 và 108, kNN 106 và 107 (trên 114 mẫu). Khoảng tin cậy Wilson 92,5%–99,1% và 90,1%–98,1% cũng đúng.
- Tên và chữ ký trong báo cáo khá thận trọng: nhiều chỗ đã tự nói "chỉ là thử nghiệm sơ bộ", "không xếp hạng chung được".

## 3. Chỗ có vấn đề

**a. Không khớp với 20 bài bạn đang đọc.** File `TOM_TAT_Y_CHINH_20_BAI.docx` của bạn dùng 20 bài đầu theo số thứ tự của bản 60 bài. Bài tổng quan của Final dùng 20 bài khác: 14 bài trùng, 6 bài của bạn không có (AMLB, PFNs4BO, ForecastPFN, TabPFN Unleashed, Do-PFN, APT) và có 6 bài thay vào (TabFlex, TransTab, XTab, CARTE, TabLLM, TabuLa-8B). Tất cả 6 bài thêm vẫn nằm trong 60 bài, nhưng thầy sẽ thấy hai danh sách không giống nhau.

**b. Danh mục tham khảo lọt luật nơi đăng của thầy.** Thầy loại bài chỉ có arXiv. Trong bài tổng quan:
- [9] LoCalPFN và [15] TabuLa-8B được ghi chỉ là arXiv, trong khi bản 60 bài ghi cả hai là NeurIPS 2024. Nên sửa thành tên hội nghị.
- [22] TabTransformer và [23] SAINT chỉ có arXiv và không nằm trong 60 bài. Nên bỏ hoặc thay bằng bài trong 60 bài (NODE, TabNet, "Revisiting Deep Learning Models" đều có trong danh sách).
- [21] còn dính tên file `01_01063-AAAI21.ArikS-ML.pdf` trong dòng trích dẫn.
- [26] Borisov ghi năm 2022, bản 60 bài ghi IEEE TNNLS 2024.
- Phụ lục B của chính bài tổng quan cũng nhắc là phải đối chiếu lại nơi đăng trước khi nộp.

**c. TabDPT chạy khác cấu hình bài gốc.** Code đặt `use_retrieval=False`, `n_ensembles=1` và `context_size` tối đa 256. Báo cáo ghi "chưa xác nhận đủ tùy chọn tìm mẫu", nhưng thực tế là đã tắt. Nên ghi thẳng như vậy để thầy không hiểu nhầm.

**d. Phần mở rộng cuối kỳ chưa được chạy.** Script TALENT chỉ mới chạy được phần mô hình truyền thống. README của nó nói rõ phần gọi TabICLv2, TabPFNv2, TabDPT chưa chạy thật. Báo cáo nói đã tải 300 bộ TALENT nhưng dữ liệu không nằm trong thư mục Final nên mình không kiểm được.

**e. Số CatBoost không tái lập được.** Báo cáo nêu CatBoost đúng 108 và 109 mẫu, nhưng không có script hay CSV nào cho số này (mình cũng không cài được CatBoost để chạy lại). Ba mô hình còn lại thì đã kiểm chứng được.

**f. Các lượt chạy là của máy khác.** Output trong notebook ghi đường dẫn `/Users/thanhpc02/...`, Python 3.14, CPU ARM (máy Mac). Các con số thời gian vì vậy không áp dụng cho máy của bạn (Windows, không có GPU NVIDIA, khoảng 12 GB RAM). Bạn nên tự chạy lại ít nhất một lần.

**g. Kịch bản nhắc tới file không có.** Kịch bản thuyết trình nhắc `TFM_SmallData_Survey_Slides_2026_v2.pptx`, trong khi thư mục chỉ có `TFM_SmallData_Survey_Slides_2026.pptx`.

**h. Bốn phương pháp khác kế hoạch của bạn.** Kế hoạch bạn lưu trước đó là TabICLv2 chạy thật, ba phương pháp còn lại (TabPFNv2, TabICL, TabDPT) chỉ trình bày theo quy trình và số công bố. Folder Final lại chọn TabPFNv2, TabDPT, LoCalPFN, và tự chạy thật cả ba mô hình đầu. TabICL (bài số 4) bị thay bằng LoCalPFN. Bạn cần quyết định theo kế hoạch nào.

**i. Dữ liệu TALENT.** Bài báo TALENT riêng đã bị loại khỏi 60 bài (51 trích dẫn, không phải bài về TabPFN). Muốn dùng bộ dữ liệu TALENT cho bài toán con thì nên ghi nguồn qua TabICL / TabICLv2 (đều nằm trong 60 bài), vì thầy yêu cầu bài toán con lấy từ 60 bài.

## 4. Liên hệ với đề xuất bài toán con

Hướng phát triển trong Final (nhãn nhiễu, mất cân bằng lớp, dữ liệu thiếu) trùng với hướng B và C trong `DE_XUAT_BAI_TOAN_CON.md`; hướng A (calibration) là phần bổ sung mà báo cáo Final cũng đã liệt kê ở vấn đề G5 nhưng chưa làm. Cả ba hướng đều có cơ sở trong 60 bài (47 nhãn nhiễu, 49 mất cân bằng, 52 và 53 calibration).

## 5. Đã sửa ngày 20/10/2026

Bản gốc được sao lưu trong `Final\_backup_truoc_khi_sua`. Đã sửa các mục b, c, d (phần câu chữ), e, f, g:
- Tài liệu tham khảo: [9] và [15] ghi NeurIPS 2024; [22] TabTransformer thay bằng NODE (ICLR 2020) và [23] SAINT thay bằng NPT (NeurIPS 2021), cả hai nằm trong 60 bài; bỏ tên file khỏi [21]; [26] sửa năm 2024. Slide 4 và kịch bản đổi SAINT thành NODE.
- TabDPT: ghi rõ chức năng tìm mẫu liên quan bị tắt (bài tổng quan, khảo sát, slide 16).
- CatBoost: thêm `source-code/Tabular_4SOTA_CPU/tools/run_baselines.py`, chạy lại ra `results/baselines.csv`, khớp 108 và 109 mẫu.
- Thêm ghi chú máy macOS ARM cho phần thời gian, bỏ câu "đã tải 300 bộ TALENT", sửa tên file slide trong kịch bản.
- Bài tổng quan đã sửa lưu thành `Survey_TFM_SmallData_2026_da_sua.docx` vì bản gốc đang mở trong Word.
Chưa sửa: mục a (20 bài khác danh sách của bạn) và mục h (kế hoạch bốn phương pháp), chờ hướng mới.
