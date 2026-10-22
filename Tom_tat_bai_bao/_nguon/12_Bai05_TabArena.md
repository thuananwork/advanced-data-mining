# Bài 5. TabArena: A Living Benchmark for Machine Learning on Tabular Data

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025, Datasets and Benchmarks Track; PDF là arXiv 2506.16791v4 (3/11/2025) |
| Tác giả | Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, Frank Hutter (AWS, Univ. Freiburg, Univ. Mannheim, INRIA Paris, ENS/PSL, Prior Labs, ELLIS Tübingen) |
| Mã nguồn / dữ liệu | tabarena.ai (bảng xếp hạng công khai, mã tái lập, kết quả tính sẵn kể cả dự đoán của mọi mô hình); dựng trên AutoGluon/TabRepo |
| Vị trí trong đề tài | **Benchmark chính của TabICLv2** (bài 7 báo cáo kết quả trên TabArena). Cũng là bộ dữ liệu và giao thức nên dùng cho thí nghiệm final-term so TFM với XGBoost/CatBoost. |

## 1. Bài này làm gì trong một đoạn

Các benchmark dữ liệu bảng hiện nay đều **tĩnh**: không được sửa khi phát hiện lỗi, không cập nhật phiên bản mô hình, không thêm mô hình mới, nên baseline "kẹt theo thời gian" và nhiều bộ dữ liệu đã lỗi thời hoặc rò rỉ. Bài đề xuất **TabArena**, benchmark "sống" đầu tiên cho dữ liệu bảng: xem benchmark như phần mềm có phiên bản, có đội bảo trì, cộng đồng góp thêm. Bản khởi tạo v0.1 gồm **51 bộ dữ liệu** chọn thủ công từ 1.053 bộ, **16 mô hình** (14 mô hình tiên tiến cộng 2 baseline, trong đó có 3 mô hình nền tảng bảng TabPFNv2, TabICL, TabDPT), tổng cộng huấn luyện ~**25 triệu** mô hình, tốn ~**15 năm** thời gian chạy (wall-clock, chưa song song). Kết luận chính: tinh chỉnh kèm ensembling các cấu hình siêu tham số mới cho thấy hiệu năng đỉnh của mô hình, học sâu đã bắt kịp GBDT khi có ngân sách và ensembling, và mô hình nền tảng vượt trội trên dữ liệu nhỏ.

## 2. Vấn đề và động cơ

- Benchmark dữ liệu bảng nhiều nhưng không thống nhất, và bị nghi ngờ về chọn lọc dữ liệu và quy trình đánh giá: dữ liệu lỗi thời, giấy phép có vấn đề, không phải bài toán bảng thật, rò rỉ dữ liệu, dùng dữ liệu không i.i.d. với chia ngẫu nhiên.
- Benchmark hiếm khi được bảo trì sau khi công bố; các nghiên cứu sau lặp lại sai sót và không so với SOTA thật sự.
- Hai bài kết quả cũ cho kết luận khác nhau (cây thắng hay NN thắng) vì khác thiết kế; TabArena muốn tổng hợp điểm tốt của từng benchmark trước.
- Phạm vi của v0.1 (Focus Statement): phân loại và hồi quy trên dữ liệu **i.i.d., cỡ nhỏ đến trung bình**. Tác giả **chủ ý không xét**: dữ liệu không i.i.d. (chuỗi thời gian, nhóm đối tượng, dịch phân phối), dữ liệu rất nhỏ (< 500 mẫu huấn luyện, few-shot), dữ liệu lớn (> 250.000 mẫu huấn luyện), và các bài toán khác như phân cụm, phát hiện bất thường, sống còn.

## 3. Thiết kế TabArena (mục 2)

### 3.1. Mô hình và tối ưu siêu tham số
Mô hình (Bảng 1):
- **Cây:** RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, EBM (Explainable Boosting Machine).
- **Mạng nơ-ron:** FastaiMLP, TorchMLP, RealMLP, TabM (bản mini), ModernNCA.
- **Mô hình nền tảng:** TabPFNv2, TabICL, TabDPT.
- **Baseline:** Linear/Logistic Regression, KNN.

Nguyên tắc: mỗi mô hình phải là baseline nổi tiếng hoặc được cài đặt **cùng với tác giả**; chỉ chạy mô hình **trong giới hạn của chính nó**. Giới hạn: TabPFNv2 ≤ 10.000 mẫu huấn luyện, ≤ 500 đặc trưng, ≤ 10 lớp; TabICL chỉ phân loại, ≤ 100.000 mẫu, ≤ 500 đặc trưng; TabDPT không có giới hạn kích thước vì dùng truy hồi ngữ cảnh (kích thước ngữ cảnh 1.024 như bài gốc; mặc định của mã là 128 nhưng nhóm thấy kém nên ghi đè).

Mỗi mô hình ba thành phần: (1) cài đặt trong framework **AutoGluon** (AbstractModel, giao diện scikit-learn: tiền xử lý, CV trong, tối ưu siêu tham số, huấn luyện song song theo fold, early stopping, unit test); (2) quy trình HPO được chọn lọc; (3) cải tiến kiểm định và ensembling.

- **Kiểm định chéo và ensemble sau (post-hoc):** mặc định **CV 8 fold** (phân tầng theo lớp), dùng cross-validation ensemble. Riêng mô hình nền tảng thì **huấn luyện lại (refit) trên train+validation** thay vì CV ensemble theo khuyến nghị của tác giả TabPFN và TabICL (vì chúng không dùng validation để early stopping nên hưởng lợi khi có thêm dữ liệu trong ngữ cảnh).
- **HPO:** 1 cấu hình mặc định + **200 cấu hình ngẫu nhiên cố định** cho mọi mô hình trừ TabICL và TabDPT (hai mô hình này chỉ chạy mặc định vì bài gốc không quy định HPO). Chọn cấu hình tốt nhất theo điểm CV trong 8 fold. Giới hạn **1 giờ** cho một cấu hình trên một phần huấn luyện (ít hơn 1% cấu hình chạm ngưỡng; EBM không bị dừng sớm ở ngưỡng này). Phần cứng khuyến nghị: 8 CPU, 32 GB RAM, 100 GB đĩa; 1 GPU 48 GB VRAM cho mô hình chạy GPU.
- **Ba chế độ báo cáo mỗi mô hình:** **Default** (siêu tham số mặc định), **Tuned** (cấu hình tốt nhất sau HPO), **Tuned + Ensembled** (ensemble sau của các cấu hình bằng greedy ensemble selection, 40 bước, Caruana 2004).

### 3.2. Chọn dữ liệu: 1.053 → 51 bộ
Lọc thủ công (chỉ khử trùng lặp và lọc kích thước là tự động được). Mười tiêu chí: duy nhất; **i.i.d.** (chia ngẫu nhiên hợp lý); không phải dữ liệu phi bảng như ảnh; dữ liệu thật (không sinh từ hàm tất định); công bố cho bài toán dự đoán thực tế; **500 ≤ số mẫu huấn luyện ≤ 250.000**; không có tiền xử lý gây rò rỉ không đảo ngược; giấy phép cho phép dùng khoa học; tải được tự động qua API công khai; không có vấn đề đạo đức rõ ràng.

Quan sát khi chọn dữ liệu (Phụ lục B.3):
- Quá trình tự động hóa không khả thi vì thiếu siêu dữ liệu: cần đọc bài gốc mới biết chia thế nào là phù hợp hoặc nhãn có tất định không.
- Rất nhiều bộ bị loại do giấy phép (chủ yếu Kaggle).
- Ít nhất 16 bộ là ảnh chữ số/chữ cái viết tay, đã lỗi thời nên loại.
- Rất nhiều bộ "tiny" (< 500 mẫu); trong 142 bộ tiny có biết ngày công bố, chỉ 4 bộ công bố sau 2010; nhiều bộ xuất phát từ tài liệu giáo trình.
- Nhiều bộ từ AutoML Challenges có tên bị che và trùng lặp; trong 254 bộ có phiên bản thay thế, đa số từ PMLB là dữ liệu nhân tạo (118 phương trình Feynman, 62 hàm Friedman).
- **Phiên bản không nhất quán:** cùng tên và nguồn nhưng khác tiền xử lý, đặc trưng, thậm chí khác nhãn giữa các benchmark, nên so sánh hiệu năng từng bộ giữa các benchmark thường không hợp lệ (do hệ thống phiên bản của OpenML gây hiểu lầm).
- **Chỉ 51 bộ thỏa i.i.d., 68 bộ thì không**, khẳng định Rubachev et al. (TabReD) rằng các benchmark trước đã dùng chia ngẫu nhiên sai chỗ.

Các bộ gồm phân loại nhị phân, đa lớp và hồi quy; nguồn UCI, Kaggle, OpenML, khác. Bảng B.2 liệt kê N, d, số lớp C, % hạng mục và cờ thuộc nhóm con TabPFNv2 hoặc TabICL.

### 3.3. Thiết kế đánh giá
- **Lặp lại:** bộ có < 2.500 mẫu dùng **CV ngoài 3 fold lặp 10 lần**; các bộ còn lại lặp 3 lần; phân loại phân tầng theo lớp.
- **Thước đo chính: Elo** (so sánh cặp, hiệu 400 điểm tương ứng xác suất thắng 10:1 = 91%); cố định 1000 Elo cho Random Forest mặc định; khoảng tin cậy 95% bằng bootstrap 200 vòng (cài đặt Bradley-Terry ổn định). Điểm cơ sở dùng cho Elo: **ROC AUC (nhị phân), log-loss (đa lớp), RMSE (hồi quy)**. Lý do chọn Elo: mỗi bộ dữ liệu đóng góp ngang nhau, không bị lệch về dữ liệu nhỏ hay ít nhiễu; điểm yếu: bỏ qua độ lớn chênh lệch.
- **Thước đo bổ sung:** điểm chuẩn hóa (thang tuyến tính, mô hình tốt nhất = 1, trung vị = 0), hạng trung bình, hạng điều hòa, số lần thắng, **improvability** (lỗi của mô hình lớn hơn mô hình tốt nhất bao nhiêu % trên mỗi bộ, rồi lấy trung bình), thời gian huấn luyện và dự đoán trên 1.000 mẫu.
- **Pipeline tham chiếu:** AutoGluon 1.3 (preset `best_quality`, 4 giờ) đại diện cho hiệu năng mà một người làm thực tế dễ đạt.
- **Lưu siêu dữ liệu:** thời gian, kết quả cho nhiều thước đo, siêu tham số, phần cứng, dự đoán trên validation và test của mọi mô hình và các mô hình theo fold, để mô phỏng ensemble mà không phải huấn luyện lại.
- **Phần cứng thực nghiệm:** CPU trên AWS M6i.2xlarge (8 nhân Xeon); GPU cho TabM, ModernNCA và các mô hình nền tảng trên NVIDIA L40S 48 GB; mỗi lượt chạy 32 GB RAM, 100 GB đĩa.
- **Bảo trì:** giao thức nộp mô hình, dữ liệu và kết quả lên bảng xếp hạng (Phụ lục E); TabArena-Lite (xem dưới).

## 4. Kết quả (mục 3 và phụ lục A)

### 4.1. Bảng xếp hạng chính (Hình 1; Bảng A.1 có điền khuyết)
Ghi chú: TabPFNv2 và TabICL chỉ chạy được trên một phần dữ liệu; trong Bảng A.1 phần thiếu được **điền bằng kết quả của Random Forest mặc định** (phạt mô hình không áp dụng được). Elo của một số dòng (khoảng tin cậy 95% khá rộng, ± 40–110 điểm):

| Mô hình | Elo | Ghi chú |
|---|---|---|
| RealMLP (Tuned+Ens.) | 1564 | cao nhất trong bảng, có khoảng tin cậy rộng |
| TabM (Tuned+Ens.) | 1541 | |
| LightGBM (Tuned+Ens.) | 1527 | |
| CatBoost (Tuned+Ens.) | 1482 | |
| CatBoost (Tuned) | 1469 | CatBoost đứng đầu khi chỉ tinh chỉnh |
| XGBoost (Tuned+Ens.) | 1440 | |
| CatBoost (Default) | 1427 | |
| ModernNCA (Tuned+Ens.) | 1425 | |
| TabPFNv2 (Tuned+Ens.) | 1405 | điền khuyết |
| TabICL (Default) | 1383 | điền khuyết |
| TabPFNv2 (Default) | 1309 | điền khuyết |
| TabDPT (Default) | 1290 | |
| RandomForest (Default) | 1000 | mốc cố định |
| KNN (Default) | 614 | thấp nhất |

(Trích theo Bảng A.1; có thể chênh với dòng trong Hình 1 chính vì Hình 1 không điền khuyết cho TabPFNv2 và TabICL. Cột "số lần thắng" chưa kiểm chứng được do bản trích cột bị lệch, mình không chép.)

Năm phát hiện chính của bài:
1. **Ensemble sau của các cấu hình mới tạo ra hiệu năng đỉnh.** Khi chỉ tinh chỉnh (Tuned) CatBoost hạng nhất, đúng như McElfresh (bài 33). Nhưng sau ensemble thì mạng nơ-ron (TabM, RealMLP) là mô hình đơn mạnh nhất trung bình. **Ba mô hình đầu (TabM, LightGBM, RealMLP) đều kém hạng tư (CatBoost) nếu không ensemble.** Vậy hiệu năng đỉnh bị đánh giá sai nếu không ensemble.
2. **Mô hình nền tảng dẫn đầu trên dữ liệu nhỏ (Hình 4):** trên 33 bộ tương thích TabPFNv2 (≤ 10K mẫu huấn luyện, ≤ 500 đặc trưng) và 36 bộ phân loại tương thích TabICL (≤ 100K, ≤ 500): **TabPFNv2 vượt các phương pháp khác với khoảng cách lớn**, và TabPFNv2 tinh chỉnh + ensemble còn vượt cả **AutoGluon**, xác nhận kết quả của bài Nature. TabICL tốt nhưng chỉ trên phân loại.
3. **Pareto hiệu quả (Hình 5):** EBM và CatBoost đã tinh chỉnh cho thời gian suy luận tốt nhất; điểm Pareto tiếp theo cải thiện mạnh nhưng chậm khoảng **15 lần (TabM-TE) và 100 lần (RealMLP-TE)** khi suy luận. GBDT hiệu quả khi xét chi phí huấn luyện; RealMLP chỉ vượt GBDT sau thời gian huấn luyện đáng kể và ensemble từ 25 cấu hình trở lên.
4. **Chi phí:** TabPFNv2 và TabICL gần như không cần huấn luyện (chi phí chủ yếu ở suy luận, Hình 6); thời gian suy luận của mô hình nền tảng lớn hơn nhiều so với GBDT.
5. **Ensemble qua nhiều mô hình tốt hơn mọi mô hình đơn (Hình 7):** ensemble mô phỏng gồm mọi mô hình TabArena (portfolio học theo Salinas và Erickson, 200 cấu hình, leave-one-dataset-out) vượt cả **AutoGluon 1.3 (4 giờ)**. Trọng số trung bình lớn nhất rơi vào ModernNCA và RealMLP, **không nhất thiết là mô hình hạng cao nhất**.

### 4.2. Phân tích phương pháp đánh giá
- **Ngộ nhận về holdout (Hình 6 phải):** dùng holdout thay vì CV để chọn mô hình thì **mọi mô hình bị đánh giá thấp** và lệch về phía mô hình đã dùng ensemble nội bộ. Do đó cần CV lồng (nested).
- **Quá khớp trên validation ("overtuning") (Phụ lục A.6):** hầu hết mô hình bão hòa ở ~201 cấu hình; **ModernNCA đạt đỉnh ở khoảng 25 cấu hình rồi giảm**. Trên validation nó đứng hạng 1, trên test chỉ hạng 6. **Mạng nơ-ron quá khớp validation nhiều hơn các mô hình khác**, kể cả khi có ít cấu hình; ModernNCA chỉ cần ensemble 10 cấu hình ngẫu nhiên đã quá khớp hơn mọi mô hình phi-học-sâu ensemble 201 cấu hình. **TabM là ngoại lệ**, bền với quá khớp nhờ ensemble nội bộ, early stopping chung cho cả ensemble và không gian tìm kiếm vừa phải. Hệ quả: ensemble liên mô hình ưu ái ModernNCA, RealMLP (vì quá khớp validation), và ít chọn TabM dù test tốt. Tác giả kêu gọi nhà phát triển mô hình xử lý vấn đề này.
- **Kiểm định thống kê (Phụ lục A.5):** biểu đồ độ khác biệt tới hạn (Friedman + hậu kiểm Nemenyi, α = 0,05, AutoRank): **luôn có một nhóm mô hình đứng đầu không khác biệt có ý nghĩa** gồm ít nhất một mô hình học sâu và một GBDT (và TabPFNv2/TabICL khi áp dụng được). Tỉ lệ thắng theo cặp (Hình A.5) giữa các mô hình mạnh khá vừa phải: **không có lời giải một-cỡ-vừa-mọi-bài**.
- **GPU và CPU (Phụ lục A.3):** TabM và ModernNCA huấn luyện nhanh hơn rõ trên GPU; trên CPU khoảng 16% lượt chạy bị dừng ở giới hạn 1 giờ, nên chạy CPU sẽ làm hại hiệu năng của chúng; RealMLP chênh ít.
- **"GBDT vs học sâu là đối lập giả":** cả hai họ đều đóng góp vào ensemble vượt xa từng họ; tác giả cho rằng quan trọng hơn là tìm mô hình tốt trong ensemble thay vì thắng GBDT. Họ cũng đoán thực hành vẫn ưu tiên GBDT vì chất lượng mã và bảo trì của học sâu kém hơn, thời gian huấn luyện (nhất là CPU), thiếu chức năng, thói quen, tài liệu, và "tuyên bố quá mức" trong bài học thuật.
- **TabArena-Lite (Phụ lục A.4):** tập con gồm mọi bộ dữ liệu nhưng chỉ **1 fold ngoài**; kết quả tương tự TabArena đầy đủ. Khuyến nghị cho nghiên cứu học thuật: chạy mô hình mới trên TabArena-Lite với 1 cấu hình mặc định (và tùy chọn khoảng 25 cấu hình ngẫu nhiên); mô hình mới mà vượt các mô hình khác ở ít nhất một bộ dữ liệu, dù không SOTA trung bình, vẫn đáng công bố.

## 5. Hạn chế (tác giả nêu và suy ra)
1. **200 cấu hình ngẫu nhiên cố định** cho mọi mô hình (để nghiên cứu ensemble), nên chưa phân tích được phương sai của HPO ngẫu nhiên hay thuật toán HPO nâng cao; ngoài ra hạt giống cố định (phiên bản sau sẽ bỏ).
2. **Giới hạn thời gian cho mỗi cấu hình** làm kết quả phụ thuộc phần cứng trong các trường hợp chạm ngưỡng; người dùng khác nhau dùng phần cứng khác nhau thì kém so sánh được.
3. **Chỉ 51 bộ** do tiêu chuẩn chọn lọc chặt, nên khả năng thống kê hạn chế, đa dạng chưa đủ; tác giả kêu gọi cộng đồng bổ sung.
4. **Không có kỹ thuật đặc trưng (feature engineering)** nên xếp hạng có thể đổi.
5. **Chưa phủ** dữ liệu rất nhỏ (< 500), lớn (> 250.000), không i.i.d.; mô hình nền tảng hiện chỉ áp dụng được cho một phần (TabPFNv2 ≤ 10K, TabICL chỉ phân loại), nên phải đánh giá riêng và điền khuyết có thể thiên vị.
6. **TabICL, TabDPT chỉ chạy mặc định**, không được tinh chỉnh, nên so sánh với các mô hình đã tinh chỉnh 200 cấu hình chưa hoàn toàn công bằng (nhưng bài nói rõ).
7. **Rủi ro làm bẩn bảng xếp hạng:** nhà cung cấp mô hình có thể tối ưu hóa quá mức trên dữ liệu TabArena hoặc dùng làm dữ liệu tiền huấn luyện; tác giả đề xuất giữ benchmark "sống" như biện pháp (Phụ lục E.6).
8. **Xung đột lợi ích** được tác giả khai báo: Holzmüller là tác giả RealMLP và TabICL; Purucker và Hutter là một phần tác giả TabPFNv2; Hutter thuộc Prior Labs; Erickson, Purucker, Desai phát triển AutoGluon.
9. **Không đánh giá hiệu chuẩn, nhãn nhiễu hay mất cân bằng lớp** như biến điều khiển (chỉ ROC AUC, log-loss, RMSE là thước đo xếp hạng; dữ liệu không cố ý chèn nhiễu).

## 6. Liên hệ với bài Nature và đề tài
- Bài Nature (bài 1) khẳng định TabPFNv2 vượt tập hợp cây/AutoML trên dữ liệu nhỏ; **TabArena là bên thứ ba độc lập kiểm chứng** điều đó (TabPFNv2 tinh chỉnh + ensemble vượt AutoGluon), đồng thời cho thấy: ưu thế chỉ nằm trong giới hạn kích thước, và các mô hình học sâu ensemble (TabM, RealMLP) có thể mạnh hơn trên bộ lớn.
- TabICLv2 (bài 7) báo cáo kết quả trên TabArena và so với AutoGluon 1.5, nên đọc bài này để hiểu Elo, improvability và các dòng "Default/Tuned/Tuned+Ensembled" mà bài 7 dùng.
- Đối với thí nghiệm final-term (TFM vs XGBoost/CatBoost): nên bắt chước **giao thức** của TabArena: CV 8 fold, ba chế độ Default/Tuned/Tuned+Ens., đo ROC AUC / log-loss, kiểm định Friedman, và cảnh giác overtuning. Cấu hình **CatBoost (Default) Elo 1427 so với TabICL (Default) 1383 và TabPFNv2 (Default) 1309 (có điền khuyết)** cho thấy với mặc định, các mô hình nền tảng chưa chắc thắng CatBoost trên toàn bộ benchmark.
- Khoảng trống: giữa 51 bộ không có kịch bản điều khiển nhãn nhiễu/mất cân bằng, nên có thể dùng TabArena làm "nền sạch" để chèn nhiễu (như phụ lục L.6 của TabICLv2 đã làm cho 6 bộ nhị phân).

## 7. Từ khóa cần nhớ
- **51 bộ từ 1.053 bộ; 16 mô hình; ~25 triệu mô hình; ~15 năm wall-clock.**
- **Default / Tuned / Tuned + Ensembled**; **CV 8 fold**; **200 cấu hình ngẫu nhiên**; **greedy ensemble selection 40 bước**.
- **Elo** (1000 = RandomForest mặc định; ROC AUC / log-loss / RMSE); **Improvability**; **TabArena-Lite** (1 fold ngoài).
- **Holdout fallacy**, **overtuning** (ModernNCA, RealMLP); **TabM bền nhất**.
- **TabPFNv2 ≤ 10K mẫu, 500 đặc trưng, 10 lớp; TabICL ≤ 100K, chỉ phân loại.**
