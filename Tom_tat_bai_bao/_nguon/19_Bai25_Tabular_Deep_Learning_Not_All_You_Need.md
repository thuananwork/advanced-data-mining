# Bài 25. Tabular Data: Deep Learning is Not All You Need

| Mục | Thông tin |
|---|---|
| Nơi đăng | arXiv 2106.03253v2 (23/11/2021); bản được trích dẫn là Information Fusion 81 (2022) |
| Tác giả | Ravid Shwartz-Ziv, Amitai Armon (IT AI Group, Intel) |
| Mã nguồn / dữ liệu | 11 bộ dữ liệu công khai (OpenML, Kaggle, UCI, MSLR, PASCAL); PDF không nêu kho mã riêng |
| Vị trí trong đề tài | Bài mở đầu cho cuộc tranh luận "deep learning có thắng cây trên dữ liệu bảng không?". Cùng nhóm với bài 29 (Grinsztajn) và bài 33 (McElfresh). Là lý do TFM cần được so với XGBoost đã tinh chỉnh công bằng. |

## 1. Bài này làm gì trong một đoạn

Nhiều mô hình học sâu cho dữ liệu bảng (TabNet, NODE, DNF-Net) tự công bố là vượt GBDT, nhưng mỗi bài dùng một bộ dữ liệu riêng và tinh chỉnh không đồng đều. Tác giả huấn luyện bốn mô hình sâu (TabNet, NODE, DNF-Net, 1D-CNN) và XGBoost với **cùng một quy trình tinh chỉnh** trên 11 bộ dữ liệu, trong đó 9 bộ lấy từ chính các bài gốc (mỗi bài 3 bộ) và 2 bộ Kaggle mới. Kết luận: mỗi mô hình sâu chỉ tốt nhất trên các bộ **trong bài của nó**, và kém rõ rệt trên bộ chưa từng thấy; **XGBoost nhìn chung thắng** và cần ít tinh chỉnh hơn nhiều; nhưng **ensemble giữa mô hình sâu và XGBoost** còn tốt hơn XGBoost đơn.

## 2. Vấn đề và động cơ

- Dữ liệu bảng là kiểu dữ liệu phổ biến nhất trong ứng dụng thực (y tế, tài chính, sản xuất, khí hậu, cơ sở dữ liệu quan hệ). GBDT (XGBoost, LightGBM, CatBoost) được xem là lựa chọn mặc định.
- Mạng nơ-ron gặp khó với dữ liệu bảng: thiếu tính cục bộ, dữ liệu thưa/thiếu giá trị, đặc trưng hỗn hợp (số, thứ tự, hạng mục), không có tri thức tiền nghiệm về cấu trúc, và thiếu khả năng diễn giải.
- Vấn đề phương pháp luận: không có benchmark chuẩn kiểu ImageNet/GLUE; một số mô hình không có mã mở; các bài so sánh thường **không tối ưu đều** cho mọi mô hình. Do đó không rõ mô hình sâu có thực sự thắng GBDT hay không.
- Hai câu hỏi của bài: (1) mô hình có chính xác hơn không, nhất là trên bộ **không xuất hiện trong bài gốc**? (2) thời gian huấn luyện và tìm siêu tham số so với mô hình khác như thế nào?

## 3. Các mô hình được so sánh (mục 2)

- **XGBoost:** GBDT; mô hình mới được tạo từ phần dư của mô hình trước, hạ gradient để giảm mất mát; một trong những cài đặt GBDT phổ biến nhất.
- **Hai nhóm mô hình sâu:**
  - *Cây khả vi:* làm trơn hàm quyết định ở nút trong để định tuyến khả vi (Hazimeh 2020; Kontschieder 2015; Popov 2020).
  - *Dựa trên attention:* attention giữa đặc trưng của một mẫu hoặc giữa các mẫu (Arik & Pfister 2021; Huang 2020; Somepalli 2021). Còn có điều chuẩn hóa theo trọng số (Kadra 2021), mô hình tích đặc trưng (Beutel 2018), 1D-CNN.
- **Bốn mô hình được chọn** (đã được tuyên bố vượt cây và được ngành chú ý):
  - **TabNet:** encoder gồm các bước quyết định tuần tự, mỗi bước dùng mặt nạ thưa học được (sparsemax) để chọn đặc trưng cho từng dòng; chọn đặc trưng mềm thay vì ngưỡng cứng.
  - **NODE:** tập hợp các cây quyết định "oblivious" (độ sâu bằng nhau, mỗi tầng một đặc trưng) khả vi, lan truyền ngược được.
  - **DNF-Net:** mô phỏng công thức chuẩn tắc tuyển (DNF) bằng mạng, thay luận lý cứng bằng bản mềm khả vi; khối DNNF gồm lớp kết nối đầy đủ và lớp hội mềm trên các literal; mô hình là ensemble của các DNNF.
  - **1D-CNN:** mô hình đơn tốt nhất trong một cuộc thi Kaggle dữ liệu bảng (Baosenguo 2021); lớp kết nối đầy đủ tạo thêm đặc trưng có "tính cục bộ", sau đó các lớp 1D-Conv có kết nối tắt. Vì được đề xuất cho một cuộc thi cụ thể và bài không chạy trên dữ liệu đó, **cả 11 bộ đều "mới" với 1D-CNN**.
- **Ensemble (mục 2.2):** năm bộ phân loại gồm TabNet, NODE, DNF-Net, 1D-CNN và XGBoost; hai cách kết hợp: (1) hỗn hợp **đều** p(y|x) = Σ_k p_θk(y|x, θ_k); (2) **trung bình có trọng số** theo validation loss chuẩn hóa, p(y|x) = Σ_k l_k^val p_θk(y|x, θ_k); trọng số đều là trường hợp riêng. Mỗi mô hình huấn luyện trên **toàn bộ** tập huấn luyện. (Công thức chỉ để minh họa; ý chính là trung bình dự đoán của các mô hình khác loại.)

## 4. Thiết lập thực nghiệm (mục 3.1, Phụ lục A–B)

- **11 bộ dữ liệu (Bảng 1):**

| Bộ | Đặc trưng | Số lớp | Mẫu | Nguồn | Bài gốc |
|---|---|---|---|---|---|
| Gesture Phase | 32 | 5 | 9,8k | OpenML | DNF-Net |
| Gas Concentrations | 129 | 6 | 13,9k | OpenML | DNF-Net |
| Eye Movements | 26 | 3 | 10,9k | OpenML | DNF-Net |
| Epsilon | 2000 | 2 | 500k | PASCAL 2008 | NODE |
| YearPrediction | 90 | 1 (hồi quy) | 515k | Million Song | NODE |
| Microsoft (MSLR) | 136 | 5 | 964k | MSLR-WEB10K | NODE |
| Rossmann Store Sales | 10 | 1 (hồi quy) | 1018k | Kaggle | TabNet |
| Forest Cover Type | 54 | 7 | 580k | Kaggle | TabNet |
| Higgs Boson | 30 | 2 | 800k | Kaggle | TabNet |
| Shrutime | 11 | 2 | 10k | Kaggle | Mới |
| Blastchar | 20 | 2 | 7k | Kaggle | Mới |

  Từ 10 đến 2000 đặc trưng, 1–7 lớp, 7k–1M mẫu; có cả đặc trưng "không đồng nhất" (đơn vị đo khác nhau) lẫn "đồng nhất" (pixel, từ). Dữ liệu chuẩn hóa về trung bình 0, phương sai 1 theo thống kê tập huấn luyện; tiền xử lý và huấn luyện theo từng bài gốc.
- **Tối ưu siêu tham số:** HyperOpt (tối ưu Bayes), **1000 bước** mỗi bộ, tối ưu trên tập validation; điểm khởi đầu lấy từ bài gốc; mỗi mô hình có **6–9** siêu tham số chính (với mô hình sâu gồm learning rate, số lớp, số nút). Không gian tìm kiếm ở Phụ lục B, ví dụ XGBoost: số cây [100, 4000], eta log-đều [e⁻⁷, 1], max_depth [1, 10], subsample/colsample [0,2, 1], min_child_weight log-đều [e⁻¹⁶, e⁵], alpha/lambda/gamma ∈ {0} ∪ log-đều [e⁻¹⁶, e²]; NODE: số lớp [1, 10], tổng số cây {256..2048}, độ sâu cây [4, 9], batch size {512..8192}; TabNet: feature dim và output dim [20, 60], số bước [1, 8], relaxation factor [0,3, 2]; DNF-Net: số công thức [256, 2048]; 1D-CNN: kích thước lớp ẩn [100, 4000], số lớp [1, 6]. CatBoost cũng được tìm tương tự (learning rate, random strength, bagging temperature, l2, leaf estimation iterations).
- **Chia dữ liệu theo bài gốc:** Epsilon, YearPrediction, MSLR, Shrutime, Blastchar: chia phân tầng ngẫu nhiên train 80% / validation 20% (theo Phụ lục B); Eye, Gesture, Gas: validation 10%, test 20%, train 70%; Forest Cover: dùng chia sẵn của tác giả bộ dữ liệu; Rossmann: dữ liệu 2014 để huấn luyện và validation (100k mẫu làm validation), 2015 để kiểm tra; Higgs: 500k mẫu validation, sau tối ưu thì huấn luyện lại trên toàn bộ tập huấn luyện. Early stopping bằng validation cho mọi mô hình. Nếu chia ngẫu nhiên: 3 lần chia, báo trung bình và sai số chuẩn của trung bình (SEM); nếu không: 4 hạt giống khác nhau trên cùng phép chia.
- **Chỉ số:** cross-entropy (nhân 100) cho phân loại; MSE cho hồi quy (YearPrediction, Rossmann). Huấn luyện bằng Adam, không giảm learning rate; dừng sau 100 epoch liên tiếp không cải thiện validation; batch size cũng được tối ưu.
- **Kiểm định:** Friedman (phi tham số, mức ý nghĩa 95%), sau đó so sánh từng cặp.
- **Định nghĩa "unseen":** bộ dữ liệu của một mô hình mà bài gốc của nó không dùng (không phải bộ chưa huấn luyện).

## 5. Kết quả

### 5.1. Mô hình sâu có tổng quát hóa sang bộ khác không? (Bảng 2)

Sai số kiểm tra (thấp hơn là tốt hơn), trung bình 4 lần chạy:

| Mô hình | Rossmann (TabNet) | CoverType (TabNet) | Higgs (TabNet) | Gas (DNF-Net) | Eye (DNF-Net) | Gesture (DNF-Net) |
|---|---|---|---|---|---|---|
| XGBoost | 490,18 ± 1,19 | 3,13 ± 0,09 | 21,62 ± 0,33 | 2,18 ± 0,20 | **56,07 ± 0,65** | 80,64 ± 0,80 |
| NODE | 488,59 ± 1,24 | 4,15 ± 0,13 | 21,19 ± 0,69 | 2,17 ± 0,18 | 68,35 ± 0,66 | 92,12 ± 0,82 |
| DNF-Net | 503,83 ± 1,41 | 3,96 ± 0,11 | 23,68 ± 0,83 | **1,44 ± 0,09** | 68,38 ± 0,65 | 86,98 ± 0,74 |
| TabNet | **485,12 ± 1,93** | 3,01 ± 0,08 | **21,14 ± 0,20** | 1,92 ± 0,14 | 67,13 ± 0,69 | 96,42 ± 0,87 |
| 1D-CNN | 493,81 ± 2,23 | 3,51 ± 0,13 | 22,33 ± 0,73 | 1,79 ± 0,19 | 67,9 ± 0,64 | 97,89 ± 0,82 |
| Simple Ensemble | 488,57 ± 2,14 | 3,19 ± 0,18 | 22,46 ± 0,38 | 2,36 ± 0,13 | 58,72 ± 0,67 | 89,45 ± 0,89 |
| Deep Ensemble w/o XGBoost | 489,94 ± 2,09 | 3,52 ± 0,10 | 22,41 ± 0,54 | 1,98 ± 0,13 | 69,28 ± 0,62 | 93,50 ± 0,75 |
| Deep Ensemble w XGBoost | 485,33 ± 1,29 | **2,99 ± 0,08** | 22,34 ± 0,81 | 1,69 ± 0,10 | 59,43 ± 0,60 | **78,93 ± 0,73** |

| Mô hình | YearPrediction (NODE) | MSLR (NODE) | Epsilon (NODE) | Shrutime (mới) | Blastchar (mới) |
|---|---|---|---|---|---|
| XGBoost | 77,98 ± 0,11 | 55,43 ± 2e-2 | 11,12 ± 3e-2 | 13,82 ± 0,19 | 20,39 ± 0,21 |
| NODE | 76,39 ± 0,13 | 55,72 ± 3e-2 | **10,39 ± 1e-2** | 14,61 ± 0,10 | 21,40 ± 0,25 |
| DNF-Net | 81,21 ± 0,18 | 56,83 ± 3e-2 | 12,23 ± 4e-2 | 16,8 ± 0,09 | 27,91 ± 0,17 |
| TabNet | 83,19 ± 0,19 | 56,04 ± 1e-2 | 11,92 ± 3e-2 | 14,94 ± 0,13 | 23,72 ± 0,19 |
| 1D-CNN | 78,94 ± 0,14 | 55,97 ± 4e-2 | 11,08 ± 6e-2 | 15,31 ± 0,16 | 24,68 ± 0,22 |
| Simple Ensemble | 78,01 ± 0,17 | 55,46 ± 4e-2 | 11,07 ± 4e-2 | 13,61 ± 0,14 | 21,18 ± 0,17 |
| Deep Ensemble w/o XGBoost | 78,99 ± 0,11 | 55,59 ± 3e-2 | 10,95 ± 1e-2 | 14,69 ± 0,11 | 24,25 ± 0,22 |
| Deep Ensemble w XGBoost | **76,19 ± 0,21** | **55,38 ± 1e-2** | 11,18 ± 1e-2 | **13,10 ± 0,15** | **20,18 ± 0,16** |

(In đậm là giá trị tốt nhất trong cột theo bảng của bài. Việc trích bảng từ PDF có thể lệch cột; mình đã đối chiếu từng cột với phần văn bản. YearPrediction và Rossmann là MSE; các bộ còn lại là cross-entropy ×100.)

Quan sát của tác giả:
- Hầu hết mô hình **kém hơn trên bộ chưa thấy** so với trên bộ trong bài gốc. Mỗi mô hình sâu chỉ tốt nhất trên bộ của chính nó (DNF-Net tốt nhất trên Gas; TabNet trên Rossmann và Higgs; NODE trên Epsilon), không có mô hình sâu nào thắng nhất quán. (Riêng 1D-CNN trông tốt hơn vì mọi bộ đều mới với nó.)
- **XGBoost thắng các mô hình sâu trên 8 trong 11 bộ ở phần các bộ chưa thấy**, có ý nghĩa (p < 0,005); điển hình là Eye Movements (56,07 so với 67–68) và Gesture (80,64 so với 87–98).
- Ensemble "mô hình sâu + XGBoost" tốt nhất ở phần lớn bộ; trên 7/11 bộ, ensemble hoặc XGBoost tốt hơn có ý nghĩa mô hình sâu đơn (p < 0,005).
- Kết quả của từng mô hình trên bộ gốc **khớp với số công bố trong bài gốc**, nên khác biệt không do cài đặt sai.

### 5.2. Hiệu năng tương đối (Bảng 3)
Trung bình hình học của mức tăng so với mô hình tốt nhất, trên các bộ chưa thấy của mỗi mô hình (thấp hơn tốt hơn):

| Mô hình | Suy giảm tương đối trung bình (%) |
|---|---|
| Deep Ensemble w XGBoost | 2,32 |
| Simple Ensemble (không sâu) | 3,15 |
| XGBoost | 3,34 |
| 1D-CNN | 7,56 |
| Deep Ensemble w/o XGBoost | 6,91 |
| TabNet | 10,51 |
| DNF-Net | 11,96 |
| NODE | 14,21 |

(Phần văn bản làm tròn hơi khác: XGBoost 3,4, 1D-CNN 7,5, DNF-Net 11,8, NODE 14,2; mình ghi theo bảng.) Ensemble đầy đủ tốt nhất với 2,32% (mọi p-value, trừ một, dưới 0,005).

**Giải thích khả dĩ của tác giả** cho việc mô hình sâu kém ở bộ chưa thấy: (1) thiên lệch chọn lọc — mỗi bài chọn bộ mà mô hình hoạt động tốt; (2) tối ưu siêu tham số kỹ hơn trên các bộ của bài gốc. Cài đặt sai bị loại trừ vì tái lập được kết quả gốc.

### 5.3. Có cần cả XGBoost lẫn mạng sâu không?
Ensemble "cổ điển" (XGBoost + SVM + CatBoost) kém hơn nhiều ensemble "mô hình sâu + XGBoost"; ensemble mô hình sâu **không có** XGBoost cũng không tốt (mọi p-value < 0,005). Kết luận: cần cả hai loại để có lợi thế.

### 5.4. Chọn tập con mô hình trong ensemble (Hình 1, Shrutime)
Ba cách chọn: theo validation loss, theo độ bất định từng mẫu, hoặc ngẫu nhiên. **Trung bình dự đoán theo validation loss là tốt nhất; chỉ cần 3 mô hình đã gần tối ưu**; chọn ngẫu nhiên tệ nhất; khác biệt ở ba mô hình đầu có ý nghĩa (p < 0,005).

### 5.5. Tối ưu khó đến đâu? (Hình 2, Shrutime)
- XGBoost nhanh hơn mô hình sâu **hơn một bậc độ lớn** về thời gian, nhưng tác giả không dùng thời gian vì phụ thuộc cài đặt (so một thư viện trưởng thành với mã nguồn bài báo là không công bằng). Thay vào đó, đo **số vòng tối ưu siêu tham số đến khi đạt bình nguyên** như thước đo độ bền, không phụ thuộc phần mềm.
- XGBoost hội tụ về hiệu năng tốt trong **ít vòng hơn** (và mỗi vòng cũng ngắn hơn). Ba nguyên nhân khả dĩ: (1) dùng tối ưu Bayes, quy trình khác có thể cho kết quả khác; (2) siêu tham số khởi đầu của XGBoost vốn bền hơn vì có thể được đề xuất dựa trên nhiều bộ hơn (có thể tìm được khởi đầu tốt hơn cho mô hình sâu); (3) XGBoost có đặc tính nội tại dễ tối ưu hơn.

## 6. Kết luận của tác giả
Mô hình sâu yếu hơn XGBoost trên các bộ không thuộc bài gốc; khó tối ưu hơn nhiều. Khi bị giới hạn thời gian, XGBoost cho kết quả tốt nhất và dễ tối ưu nhất; để tối đa hóa hiệu năng thì cần thêm mô hình sâu vào ensemble. Nghiên cứu sau phải kiểm tra **hệ thống trên nhiều bộ đa dạng** và báo cáo **độ dễ tìm siêu tham số**; một hướng khác là mô hình sâu dễ tối ưu hơn.

## 7. Hạn chế (tác giả nêu và suy ra)
1. **Chỉ 11 bộ dữ liệu**, trong đó hầu hết rất lớn (hàng trăm nghìn mẫu); chỉ 2 bộ nhỏ (Shrutime 10k, Blastchar 7k). Tác giả thừa nhận bất kỳ tập bộ dữ liệu nào cũng không đại diện đủ sự đa dạng và "không có bữa trưa miễn phí".
2. **Ensemble không công bằng về chi phí:** năm mô hình được huấn luyện và tinh chỉnh đầy đủ; tăng chi phí tính toán (tác giả có xét chọn tập con).
3. Tối ưu siêu tham số bằng **một thuật toán duy nhất** (Bayes/HyperOpt); khác biệt về độ bền có thể phụ thuộc vào điều này và vào điểm khởi đầu.
4. Chỉ dùng **XGBoost** làm đại diện cây (CatBoost chỉ trong ensemble cổ điển); không có Random Forest hay LightGBM làm đối thủ chính.
5. Chỉ ba mô hình sâu + 1D-CNN; **không có mô hình tiền huấn luyện/foundation** (bài đi trước TabPFN).
6. Đo độ khó tối ưu chỉ minh họa trên **một bộ (Shrutime)**; số liệu thời gian không nêu cụ thể.
7. Không đánh giá nhãn nhiễu, mất cân bằng lớp, hiệu chuẩn hay dữ liệu hạng mục cardinality cao; chỉ cross-entropy và MSE.
8. Dùng sai số chuẩn trên 3–4 lần chạy; số lần chạy nhỏ.

## 8. Liên hệ với bài Nature và đề tài
- Là một trong hai bài (cùng bài 29) cho thấy **quy trình tinh chỉnh công bằng** quyết định kết luận. Bài Nature và TabArena (bài 5) thừa hưởng bài học này: so tinh chỉnh đều cho mọi mô hình, báo cáo cả cấu hình mặc định.
- Cảnh báo "kết quả tốt trên bộ của chính bài" tương tự rủi ro cho TFM: kiểm tra TabICLv2 trên bộ **ngoài** bộ dữ liệu bài gốc là cần thiết.
- Ensemble sâu + cây cho thấy hướng bổ sung (ví dụ ensemble TFM + CatBoost trong thí nghiệm final-term) và AutoGluon đã làm điều này trong AMLB.
- Mức "độ khó tối ưu" là điểm lợi của TFM: không cần tìm siêu tham số.

## 9. Từ khóa cần nhớ
- 11 bộ (9 từ bài gốc + 2 Kaggle mới), HyperOpt **1000 bước**, 6–9 siêu tham số; TabNet, NODE, DNF-Net, 1D-CNN vs XGBoost.
- XGBoost thắng 8/11 bộ chưa thấy (p < 0,005); suy giảm tương đối: **ensemble sâu+XGB 2,32% < XGBoost 3,34% < NODE 14,21%**.
- Ensemble cổ điển (XGBoost+SVM+CatBoost) kém; chọn 3 mô hình theo validation loss gần tối ưu; XGBoost cần ít vòng tối ưu hơn.
