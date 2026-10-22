# Bài 33. When Do Neural Nets Outperform Boosted Trees on Tabular Data? (TabZilla)

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2023, Datasets and Benchmarks Track; PDF là arXiv 2305.02997v4 (15/7/2024) |
| Tác giả | Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Benjamin Feuer, Chinmay Hegde, Ganesh Ramakrishnan, Micah Goldblum, Colin White |
| Mã nguồn / dữ liệu | github.com/naszilla/tabzilla (bộ benchmark, mã, toàn bộ kết quả thô trên 500 nghìn mô hình); TabZilla được công bố như một collection trên OpenML |
| Vị trí trong đề tài | Nguồn của bộ benchmark **TabZilla**, cũng là bộ dữ liệu LoCalPFN (bài 10) dùng. Là bằng chứng nền: TabPFN v1 đã đứng đầu trung bình dù chỉ thấy 3.000 mẫu. |

## 1. Bài này làm gì trong một đoạn

Cuộc tranh luận "mạng nơ-ron (NN) hay cây tăng cường gradient (GBDT) tốt hơn trên dữ liệu bảng" thường dựa trên ít hơn 50 bộ dữ liệu hoặc baseline chưa được tinh chỉnh kỹ. Bài này chạy **19 thuật toán trên 176 bộ dữ liệu phân loại** (mỗi thuật toán tối đa 30 bộ siêu tham số, 10 fold, tổng **538.650 mô hình**), rồi hỏi hai câu: (1) việc chọn họ thuật toán có thật sự quan trọng không, (2) tính chất nào của bộ dữ liệu khiến NN hoặc GBDT thắng. Kết luận chính: cuộc tranh luận "NN vs GBDT" bị **thổi phồng**; ngoại lệ đáng chú ý là **TabPFN**, tốt nhất trung bình dù chỉ dùng 3.000 mẫu. Từ phân tích, nhóm phát hành **TabZilla Benchmark Suite**: 36 bộ dữ liệu "khó nhất".

## 2. Vấn đề và động cơ

- Dữ liệu bảng là loại dữ liệu lâu đời và phổ biến nhất (y tế, tài chính, quảng cáo...), nhưng khác với thị giác và ngôn ngữ, NN chưa vượt hẳn các phương pháp khác. Nhiều công trình lập luận cho NN (TabNet, NODE, SAINT, FT-Transformer...), nhiều công trình khác lập luận cho GBDT (Grinsztajn, Shwartz-Ziv...).
- Hạn chế của các nghiên cứu trước: ít bộ dữ liệu (thường dưới 50), baseline không được tinh chỉnh đầy đủ, và chỉ hỏi "ai có hạng trung bình tốt hơn" mà không tìm hiểu nguyên nhân.
- Công trình gần nhất là Grinsztajn et al. (bài 29): 7 thuật toán, 45 bộ, kích thước 3.000–10.000 (có 7 bộ đúng 50.000), kiểm soát chặt (giới hạn tỉ lệ kích thước/đặc trưng, bỏ đặc trưng hạng mục cardinality cao, bỏ đặc trưng số cardinality thấp). Bài này **không kiểm soát** như vậy: kích thước từ **32 đến 1.025.009**, nên thấy được các hiện tượng mà bài kia bỏ sót (ví dụ GBDT tốt hơn trên bộ "bất thường"), và xét **hàng trăm siêu đặc trưng (metafeature)** thay vì vài cái.
- Bài duy nhất trước đó dùng nhiều hơn 50 bộ là TabPFN (179 bộ nhưng đều ≤ 2.000 mẫu).

## 3. Thiết lập thí nghiệm (mục 2)

- **19 thuật toán:**
  - 3 GBDT: CatBoost, LightGBM, XGBoost.
  - 11 mạng nơ-ron: DANet, FT-Transformer, hai MLP (rtdl và MLP thường), NODE, ResNet, SAINT, STG, TabNet, **TabPFN**, VIME.
  - 5 baseline: Decision Tree, KNN, Logistic Regression (LinearModel), Random Forest, SVM.
  - Thêm một số thuật toán chỉ chạy được một phần (NAM, DeepFM, TabTransformer) nên chỉ so sánh sơ bộ trong phụ lục.
- **TabPFN\*** (ký hiệu TabPFN-3k trong phụ lục): do bộ nhớ và thời gian của TabPFN tăng bậc hai theo số mẫu huấn luyện, trên bộ lớn hơn 3.000 mẫu chỉ **lấy ngẫu nhiên 3.000 mẫu** từ tập huấn luyện.
- **176 bộ phân loại từ OpenML**, gồm phần lớn bộ trong OpenML-CC18 (bài 36), OpenML Benchmarking Suite và một số bộ bổ sung. Khoảng một nửa là phân loại nhị phân (tối đa 100 lớp); khoảng một nửa số tập huấn luyện nhỏ hơn 2.300 mẫu. Do giới hạn tính toán không dùng bộ lớn hơn 1,1 triệu mẫu.
- **Siêu đặc trưng:** 965 đặc trưng từ thư viện PyMFE, thuộc năm nhóm: general (số mẫu, số lớp...), statistical (độ lệch, độ nhọn...), information-theoretic (entropy của nhãn...), landmarking (hiệu năng của baseline như 1-NN trên mẫu con), model-based (thống kê của mô hình cây). Thêm log của mọi đặc trưng dương hẳn.
- **Giao thức:** dùng 10 fold train/test có sẵn của OpenML; chia tiếp mỗi fold huấn luyện thành train/validation. Mỗi thuật toán và mỗi lần chia chạy tối đa 10 giờ, tối đa **30 bộ siêu tham số** (1 mặc định + 29 ngẫu nhiên, dùng Optuna), mỗi bộ siêu tham số tối đa 2 giờ trên V100 32 GiB. Báo cáo hiệu năng test của cấu hình tốt nhất trên validation.
- **Thước đo chính:** accuracy; ngoài ra log loss (ít hòa hơn nên dùng cho kiểm định thống kê), F1, ROC-AUC (phụ lục). Khi lấy trung bình giữa các bộ dữ liệu dùng **ADTM** (average distance to the minimum, thang 0–1 sau khi chọn siêu tham số tốt nhất). Xếp hạng: nếu hòa thì cùng nhận hạng tốt nhất.
- **Tiền xử lý tối thiểu:** chỉ điền giá trị thiếu (NaN), để giảm yếu tố gây nhiễu; phụ lục thử thêm quantile scaling.

## 4. Kết quả (mục 2.1 và phụ lục D)

### 4.1. Không thuật toán nào thống trị (Bảng 1, 98 bộ chạy trọn vẹn)

Bảng 1 xếp hạng theo accuracy (hạng trung bình; Mean Acc là accuracy chuẩn hóa 0–1; thời gian huấn luyện trung bình trên 1.000 mẫu):

| Thuật toán | Họ | Hạng TB | Mean Acc | Thời gian (s/1000 mẫu) |
|---|---|---|---|---|
| CatBoost | GBDT | 5,50 | 0,87 | 21,70 |
| TabPFN\* | PFN | 5,89 | 0,83 | 0,25 |
| XGBoost | GBDT | 6,87 | 0,81 | 0,81 |
| ResNet | NN | 7,65 | 0,75 | 16,01 |
| SAINT | NN | 7,90 | 0,73 | 169,54 |
| NODE | NN | 7,90 | 0,74 | 138,36 |
| FT-Transformer | NN | 8,14 | 0,76 | 27,67 |
| Random Forest | baseline | 8,26 | 0,76 | 0,35 |
| LightGBM | GBDT | 8,46 | 0,76 | 0,87 |
| SVM | baseline | 9,09 | 0,69 | 30,40 |
| ... | | | | |
| TabNet | NN | 12,69 | 0,54 | 34,95 |
| KNN | baseline | 13,69 | 0,45 | 0,01 |
| VIME | NN | 14,98 | 0,37 | 16,81 |

(Trích 13 trong 19 dòng; mỗi thuật toán đều từng đứng hạng 1 trên ít nhất một bộ, trừ VIME. Trong văn bản chính có câu "CatBoost chỉ đạt hạng trung bình 5,06"; con số trong bảng của bản PDF này là 5,50, tức bản bài viết chưa đồng nhất, nên mình ghi theo bảng.)

- **Gần như thuật toán nào cũng từng đứng nhất trên ít nhất một bộ và đứng bét trên ít nhất một bộ khác.** CatBoost, thuật toán tốt nhất, chỉ đạt hạng trung bình ~5,5, nên không cách tiếp cận nào thống trị.
- **Thuật toán nhanh nhất không phải baseline đơn giản mà là TabPFN:** thời gian huấn luyện 0,25 s/1.000 mẫu, nhanh hơn CatBoost hai bậc độ lớn, mà accuracy chuẩn hóa chỉ kém một chút (0,83 so với 0,87), dù chỉ thấy một phần dữ liệu ở nhiều bộ. Thời gian suy luận của TabPFN (2,36 s/1.000 mẫu) cao hơn các thuật toán khác (nguồn: văn bản chính).
- **Trên 57 bộ nhỏ (kích thước ≤ 1.250, Bảng 2):** **TabPFN đứng đầu về hạng trung bình (4,88)**, trước CatBoost (5,37), ResNet (6,75), Random Forest (7,65); mean acc chuẩn hóa 0,84 so với 0,85 của CatBoost.
- **Kiểm định thống kê (Hình 3, 98 bộ):** Friedman bác bỏ giả thuyết đồng nhất với p < 10^-20; sau đó Wilcoxon signed-rank với hiệu chỉnh Holm-Bonferroni (p < 0,05). Theo log loss, **TabPFN thắng mọi thuật toán khác trung bình và khác biệt có ý nghĩa thống kê**; kế đó là CatBoost, XGBoost, SAINT.
- **Thời gian vs hiệu năng (Hình 2):** NN cần thời gian dài nhất; GBDT vừa nhanh vừa mạnh; TabPFN đạt hiệu năng hàng đầu với thời gian huấn luyện ít hơn mọi GBDT lẫn NN.
- **Trên toàn bộ 176 bộ (Bảng 10, accuracy, một số thuật toán không chạy xong mọi bộ):** CatBoost hạng 5,21 (mean acc 0,88), XGBoost 5,60 (0,88), ResNet 6,80 (0,79), LightGBM 6,98 (0,83), SAINT 7,55, NODE 7,57; thứ tự hầu như giống Bảng 1 (không có dòng TabPFN riêng trong Bảng 10).

### 4.2. Họ thuật toán và vai trò của tinh chỉnh (Hình 4)

- **Venn "high-performing"** (một thuật toán đạt accuracy chuẩn hóa ≥ 0,99): chia ba họ GBDT / NN / baseline. Phân bố trên 176 bộ **khá cân bằng giữa ba họ**, GBDT hơi nhỉnh. Khi siết ngưỡng lên 0,9999 (Hình 12), GBDT là họ duy nhất đạt ngưỡng ở **42%** bộ, NN ở **30%**; nhưng chênh lệch dưới 0,1% nên ít ý nghĩa thực tế.
- **Chọn thuật toán vs tinh chỉnh:** với khoảng **một phần ba số bộ dữ liệu**, tinh chỉnh nhẹ (30 vòng random search) trên CatBoost (hoặc ResNet) cải thiện log loss nhiều hơn việc chọn giữa "NN tốt nhất mặc định" và "GBDT tốt nhất mặc định". Tức nhiều khi không cần phân biệt GBDT hay NN, chỉ cần một thuật toán mạnh được tinh chỉnh.
- **Nhiều thuật toán khá tốt ngay cả khi chưa tinh chỉnh** (Bảng 9): CatBoost (mặc định) hạng 10,12; CatBoost tinh chỉnh 8,05; XGBoost mặc định 12,02; ResNet mặc định 13,94.

### 4.3. Tính chất bộ dữ liệu liên quan tới thắng thua (mục 2.2)

Cách làm: tính tương quan Pearson của các siêu đặc trưng với (a) chênh log loss chuẩn hóa giữa NN tốt nhất và GBDT tốt nhất (ký hiệu Δℓℓ), (b) hiệu năng từng thuật toán so với tất cả, (c) chênh lệch giữa các cặp thuật toán hàng đầu (CatBoost, XGBoost, SAINT, ResNet). Kiểm tra giá trị dự đoán bằng meta-model huấn luyện theo kiểu **leave-one-dataset-out** (giữ một bộ làm test, 175 bộ còn lại huấn luyện). TabPFN bị loại khỏi phân tích "GBDT vs NN" vì cơ chế quá khác biệt.

- **NN kém hơn GBDT trên bộ lớn.** XGBoost đứng đầu trên cả 7 bộ lớn nhất. Kích thước (log số mẫu) là siêu đặc trưng tương quan âm mạnh nhất với hiệu năng tương đối của LightGBM (khoảng tin cậy 95% của tương quan [−0,36; −0,28]) và XGBoost ([−0,27; −0,18]). GBDT cũng hưởng lợi khi **tỉ lệ số mẫu/số đặc trưng cao**, vì mỗi lần tách nút được tính trên nhiều điểm hơn. Tuy vậy "NN" không đồng nhất: TabPFN rất tốt trên bộ nhỏ, còn TabNet tốt hơn ở bộ lớn; Hình 13 cho thấy CatBoost thắng TabNet cho đến cỡ ~1.500 mẫu rồi hai bên tương đương.
- **GBDT thích bộ dữ liệu "bất thường" (irregular).** CatBoost và XGBoost thắng ResNet và SAINT khi phân phối đặc trưng nặng đuôi, lệch, hoặc phương sai rất khác nhau giữa các đặc trưng (cao về độ lệch và độ nhọn). "Độ bất thường" trong Hình 5 là tổ hợp tuyến tính của năm siêu đặc trưng đã chuẩn hóa: trị riêng nhỏ nhất của ma trận hiệp phương sai đặc trưng (hệ số −0,33), độ lệch của độ lệch chuẩn các đặc trưng (0,23), độ lệch của khoảng giá trị (0,22), IQR của trung bình điều hòa (0,21), độ lệch chuẩn của độ nhọn (0,21). GBDT cũng tốt hơn khi lớp **mất cân bằng nhiều** hơn (rõ nhất với SAINT).
- **Siêu đặc trưng tương quan nhất với từng thuật toán (Bảng 3):** CatBoost: "độ nhiễu của đặc trưng" ((Σ entropy đặc trưng − Σ thông tin tương hỗ với nhãn)/Σ thông tin tương hỗ) [0,25; 0,34]; ResNet: tương quan chính tắc trung bình giữa đặc trưng số và nhãn [−0,28; −0,19]; FT-Transformer: số lớp [0,23; 0,32]; SAINT: độ nhiễu của đặc trưng [0,19; 0,29].
- **Meta-model và cây quyết định chọn thuật toán (Hình 15):** cây độ sâu 3 chọn giữa ResNet, SAINT, TabPFN, CatBoost, XGBoost. Các nhánh chính: số mẫu ≤ 4.211 (nhánh nhỏ có TabPFN, chia tiếp theo ngưỡng 1.132 mẫu), độ thưa của đặc trưng, tần suất lớp lớn nhất ≤ 0,792, số đặc trưng số không NaN ≤ 65. Chọn đặc trưng tiến tham lam (forward selection) với meta-model XGBoost, dự đoán chênh log loss CatBoost − ResNet: năm đặc trưng đầu là số đặc trưng phân phối chuẩn (kiểm định Shapiro-Wilk), trung vị của giá trị nhỏ nhất của các đặc trưng, trung vị độ thưa, IQR của trung bình các đặc trưng, trung bình của trung bình điều hòa. Độ chính xác của các cây (Bảng 17) có trong phụ lục; mình không chép con số vì bản trích văn bản của bảng này bị lệch cột.
- **Khuyến nghị thực hành của tác giả:** thử baseline đơn giản trước, rồi tinh chỉnh nhẹ CatBoost; sau đó thử NN và GBDT khác dựa trên siêu đặc trưng của bộ dữ liệu. Nên xem phân tích **theo từng thuật toán** thay vì xu hướng chung "GBDT vs NN".

### 4.4. Thí nghiệm phụ (phụ lục D)

- **Ablation TabPFN và CatBoost (Bảng 13, 98 bộ, accuracy):** CatBoost (tuần tự toàn tập huấn luyện) hạng 6,12; **TabPFN-3k hạng 6,43**; CatBoost-1k (chỉ 1.000 mẫu) 7,02; XGBoost 7,85; **TabPFN-1k 7,94**. Nghĩa là chỉ 1.000 mẫu, TabPFN vẫn ngang XGBoost; còn CatBoost chỉ giảm nhẹ khi bị cắt còn 1.000 mẫu.
- **Hồi quy (Bảng 18):** 12 thuật toán trên 17 bộ hồi quy (tối ưu R²). Kết luận tương tự phân loại: GBDT rất mạnh, đặc biệt CatBoost, mọi thuật toán đều từng tốt và từng tệ.
- **Quantile scaling (Bảng 19, 36 bộ TabZilla):** cải thiện các thuật toán đơn giản (Decision Tree, MLP, Random Forest, SVM) nhưng ít tác động lên thuật toán mạnh.
- **Tinh chỉnh bổ sung (Bảng 20):** 100 vòng Optuna (TPE) cho XGBoost, CatBoost, LightGBM, Random Forest trên 36 bộ TabZilla; như dự kiến hiệu năng cải thiện so với 30 vòng ngẫu nhiên và so với mặc định. Không chạy thêm cho NN vì tốn tài nguyên.

## 5. TabZilla Benchmark Suite (mục 3)

36 bộ "khó nhất" trong 176 bộ, chọn theo ba tiêu chí:
1. **Khó với baseline:** loại bộ nào mà một baseline đạt log loss chuẩn hóa trong vòng 20% so với thuật toán tốt nhất.
2. **Khó với đa số thuật toán:** thuật toán tốt thứ tư trong 19 thuật toán kém thuật toán tốt nhất ít nhất 7%, tức chỉ một, hai hoặc ba thuật toán nổi bật (tiêu chí này gần như bao hàm tiêu chí 1).
3. **Khó với GBDT:** thêm mọi bộ mà GBDT kém thuật toán tốt nhất ≥ 10%, để tránh điểm mù ở những nơi GBDT yếu.

Kích thước trong bộ từ **148 đến hơn 1 triệu mẫu** (ví dụ credit-g 1.000, MiniBooNE 130.064, poker-hand 1.025.009, albert 425.240, airlines 539.382), độ nhọn đặc trưng rất đa dạng. Bảng 4 liệt kê ba thuật toán tốt nhất ở từng bộ: phần lớn là GBDT (CatBoost, XGBoost, LightGBM), nhưng còn thấy ResNet, SAINT, TabNet, STG, SVM, TabPFN, DANet.

## 6. Hạn chế (tác giả nêu và suy ra)
1. Chủ yếu **phân loại** (hồi quy chỉ có 17 bộ trong phụ lục); chưa có chuỗi thời gian, chưa nghiên cứu **định lượng bất định**, chưa xét tỉ lệ đặc trưng hạng mục.
2. Chỉ **30 vòng tìm kiếm ngẫu nhiên** cho mọi thuật toán (100 vòng TPE chỉ cho cây và chỉ trên TabZilla); NN tốn kém nên không tinh chỉnh sâu.
3. Một số thuật toán (NAM, DeepFM, TabTransformer) **không chạy xong** trên mọi bộ vì bộ nhớ và thời gian nên chỉ so sánh sơ bộ. Có 98 bộ chạy trọn vẹn cho Bảng 1, còn lại 78 bộ gặp vấn đề bộ nhớ/thời gian với một số thuật toán.
4. TabPFN bị cắt còn **3.000 mẫu**, và bản TabPFN dùng là v1 (tối đa 100 đặc trưng, 10 lớp); kết quả TabPFN cho bộ lớn không phải của mô hình đầy đủ.
5. Tiền xử lý rất tối thiểu (chỉ điền NaN), có thể thiệt cho NN; phụ lục chỉ thử quantile scaling.
6. **Không đánh giá hiệu chuẩn, nhãn nhiễu hay mất cân bằng lớp** như một biến điều khiển; mất cân bằng chỉ xuất hiện như một siêu đặc trưng quan sát.
7. Phân tích tương quan siêu đặc trưng là **quan hệ thống kê**, không phải nhân quả.

## 7. Liên hệ với bài Nature và đề tài
- Đây là **bằng chứng thực nghiệm đầu tiên ở quy mô lớn** cho thấy một mô hình tiền huấn luyện trên dữ liệu tổng hợp (TabPFN v1) vượt cả CatBoost trên bộ nhỏ, và là cơ sở cho mô tả "TabPFN nổi bật ở dữ liệu nhỏ" trong bài Nature (bài 1) và các bài sau.
- Kết luận "tinh chỉnh nhẹ một GBDT mạnh thường đủ" chính là lý do bài Nature, TabArena (bài 5) và đề tài final-term phải so TFM với **XGBoost/CatBoost đã tinh chỉnh**, không chỉ với mặc định.
- TabZilla là bộ dữ liệu của LoCalPFN (bài 10), và là nguồn tính chất "dữ liệu bất thường, mất cân bằng lớp thì GBDT tốt hơn" mà đề tài về nhãn nhiễu/mất cân bằng cần kiểm tra với TFM.
- Khoảng trống: chưa so TFM hiện đại (TabPFNv2, TabICLv2) theo từng siêu đặc trưng (độ bất thường, mất cân bằng, nhiễu nhãn) trên cùng bộ dữ liệu.

## 8. Từ khóa cần nhớ
- **176 bộ, 19 thuật toán, 538.650 mô hình; 10 fold; 30 vòng tinh chỉnh.**
- **TabPFN\*** = TabPFN với mẫu con 3.000; **ADTM** = trung bình khoảng cách tới giá trị nhỏ nhất (thang 0–1).
- **Irregularity:** GBDT thắng NN trên bộ lớn, bất thường, nhiều mẫu so với đặc trưng; chọn thuật toán nhiều khi kém quan trọng hơn tinh chỉnh.
- **TabZilla Suite:** 36 bộ khó nhất (148 đến hơn 1 triệu mẫu).
