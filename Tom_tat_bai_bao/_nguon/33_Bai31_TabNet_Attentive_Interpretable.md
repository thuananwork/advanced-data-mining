# Bài 31. TabNet: Attentive Interpretable Tabular Learning

| Mục | Thông tin |
|---|---|
| Nơi đăng | AAAI 2021; arXiv:1908.07442v5, 9/12/2020 |
| Tác giả | Sercan Ö. Arık, Tomas Pfister (Google Cloud AI) |
| Mã nguồn | Bài ghi sẽ có mã nguồn mở (bản triển khai TensorFlow/PyTorch sau này) |
| Lưu ý về PDF | Bảng 1–8 và 10 đọc rõ. Hình 4 bị lẫn nhãn. Bảng 9 (xếp hạng độ quan trọng của Adult) và Bảng 11 (ablation) bị lệch cột nên chỉ dùng các nhận xét bằng lời và những chênh lệch đọc được chắc chắn. |
| Vị trí trong đề tài | Một trong những **mạng sâu "kinh điển" đầu tiên cho bảng** (attention theo cột tuần tự, giải thích được, tiền huấn luyện tự giám sát). Thường xuất hiện làm baseline yếu trong các benchmark (bài 33 McElfresh, TabArena, TabFlex). Đọc để hiểu **vì sao hướng "kiến trúc đặc biệt cho bảng" chưa thắng GBDT** trước khi TFM xuất hiện. |

## 1. Bài này làm gì trong một đoạn

Mạng nơ-ron sâu thành công với ảnh, văn bản, âm thanh nhờ có kiến trúc chuẩn, nhưng bảng chưa có, và họ cây quyết định vẫn thống trị. TabNet đề xuất một kiến trúc chuẩn cho bảng, nhận **dữ liệu thô không tiền xử lý**, huấn luyện bằng gradient descent, và dùng **attention tuần tự (sequential attention)** để mỗi **bước quyết định** (decision step) chọn một tập con đặc trưng làm việc. Việc chọn đặc trưng **theo từng mẫu (instance-wise)**, thưa, học được, vừa dùng năng lực mô hình cho đặc trưng quan trọng, vừa cho **khả năng giải thích** cục bộ (mặt nạ từng bước) lẫn toàn cục (độ quan trọng tổng hợp). Bài còn lần đầu cho thấy **tiền huấn luyện tự giám sát bằng dự đoán đặc trưng bị che** cải thiện đáng kể kết quả khi dữ liệu không nhãn nhiều. Trên các tập từ 10 nghìn đến 10,5 triệu mẫu, TabNet vượt hoặc ngang GBDT và các mạng khác, mặc dù trên các tập "đã bão hòa" thì chỉ ngang hoặc kém nhẹ.

## 2. Bối cảnh

- Cây quyết định hấp dẫn vì (i) hiệu quả biểu diễn cho ranh giới quyết định xấp xỉ siêu phẳng (phổ biến trong bảng), (ii) dễ giải thích (theo nút quyết định, hoặc SHAP cho ensemble), (iii) huấn luyện nhanh. MLP và CNN chồng lên nhau bị quá tham số hóa và thiếu thiên kiến quy nạp phù hợp.
- Vì sao vẫn đáng thử học sâu cho bảng: kỳ vọng hiệu năng cao với dữ liệu rất lớn; huấn luyện đầu cuối cho phép kết hợp bảng với ảnh/âm thanh, giảm kỹ thuật đặc trưng, học từ dòng dữ liệu, và học biểu diễn (thích nghi miền, mô hình sinh, bán giám sát).
- **Công trình liên quan:** chọn đặc trưng toàn cục (forward selection, Lasso) so với **theo mẫu** (L2X của Chen 2018 tối đa hóa thông tin tương hỗ; INVASE của Yoon 2019 dùng actor-critic); cây quyết định và ensemble (Random Forest, XGBoost, LightGBM); tích hợp DNN vào cây (cây mềm, Neural Decision Forest, soft binning của Yang 2018, Tanno 2018 adaptive neural tree); học tự giám sát với dự đoán đầu vào bị che (BERT cho văn bản). TabNet khác ở chỗ: **chọn đặc trưng mềm với độ thưa điều khiển được, trong một kiến trúc duy nhất làm cả chọn đặc trưng lẫn ánh xạ đầu ra**.

## 3. Phương pháp

### 3.1. Ý tưởng "cây bằng khối DNN"
Hình 3 minh họa dùng các khối DNN quen thuộc để tạo ranh giới giống cây: mặt nạ nhân thưa chọn đặc trưng, biến đổi tuyến tính + bias biểu diễn ngưỡng, ReLU chọn vùng, cộng gộp các vùng; hằng số lớn làm ranh giới sắc nét. TabNet dựa trên chức năng này nhưng (i) học chọn đặc trưng thưa theo mẫu, (ii) nhiều bước tuần tự, mỗi bước góp một phần vào quyết định, (iii) xử lý phi tuyến đặc trưng đã chọn để tăng năng lực, (iv) mô phỏng ensemble bằng chiều cao hơn và nhiều bước hơn.

### 3.2. Bộ mã hóa (encoder)
Đầu vào: đặc trưng số thô; đặc trưng hạng mục qua embedding học được (một chiều); chỉ dùng batch normalization, không chuẩn hóa toàn cục. Cùng D đặc trưng f ∈ ℝ^{B×D} (B: kích thước batch) được đưa vào **N_steps bước quyết định**; bước i nhận thông tin đã xử lý của bước i−1.
- **Chọn đặc trưng:** mặt nạ học được M[i] ∈ ℝ^{B×D}; M[i] = sparsemax(P[i−1] · h_i(a[i−1])), với h_i là lớp FC + BN, a[i−1] là phần thông tin chuyển từ bước trước. **Sparsemax** (Martins & Astudillo 2016) chiếu Euclid lên simplex xác suất, tạo ra mặt nạ **thưa thật sự** (nhiều phần tử đúng bằng 0), tốt hơn softmax cho mục đích giải thích. Σⱼ M[i]_{b,j} = 1.
- **Prior scale:** P[i] = ∏_{j≤i} (γ − M[j]) cho biết một đặc trưng đã được dùng bao nhiêu ở các bước trước. γ = 1 buộc mỗi đặc trưng chỉ dùng ở đúng một bước; γ lớn cho phép dùng lại nhiều bước. P[0] khởi tạo toàn 1 (không có prior); nếu có đặc trưng không dùng (tự giám sát) thì phần tương ứng đặt 0.
- **Điều chuẩn độ thưa (entropy):** L_sparse = Σ_i Σ_b Σ_j −M_{b,j}[i] log(M_{b,j}[i] + ε) / (N_steps · B), cộng vào loss với hệ số λ_sparse; có lợi cho dữ liệu có nhiều đặc trưng dư thừa.
- **Xử lý đặc trưng (feature transformer):** [d[i], a[i]] = f_i(M[i] · f), d[i] ∈ ℝ^{B×N_d} cho quyết định, a[i] ∈ ℝ^{B×N_a} cho bước kế. Khối gồm 4 lớp (2 lớp **chung** cho mọi bước, 2 lớp **riêng từng bước**), mỗi lớp = FC + ghost BN + GLU, nối phần dư có hệ số √0,5 để ổn định phương sai. **Ghost BN** (Hoffer 2017) cho phép batch lớn với batch ảo nhỏ (B_V, động lượng m_B); riêng lớp đầu vào không dùng ghost BN.
- **Đầu ra:** d_out = Σ_i ReLU(d[i]), rồi ánh xạ tuyến tính W_final · d_out (phân loại thêm softmax lúc huấn luyện, argmax lúc suy luận). Phép cộng ReLU các bước mô phỏng cách cây tổng hợp các vùng.

### 3.3. Giải thích được
- **Cục bộ:** mặt nạ M_{b,j}[i] = 0 nghĩa là đặc trưng j của mẫu b không đóng góp ở bước i.
- **Toàn cục:** độ quan trọng tổng hợp M_agg-b,j = Σ_i η_b[i] M_{b,j}[i] / chuẩn hóa, với η_b[i] = Σ_c ReLU(d_{b,c}[i]) là trọng số tầm quan trọng của bước i (nếu d < 0 thì bước đó không đóng góp).

### 3.4. Tiền huấn luyện tự giám sát
Thêm **bộ giải mã (decoder)** gồm các feature transformer và FC mỗi bước, cộng các đầu ra để tái tạo đặc trưng. Nhiệm vụ: **dự đoán các cột bị che** từ các cột còn lại. Mặt nạ nhị phân S ~ Bernoulli(p_s) mỗi vòng lặp; encoder nhận (1 − S)·f̂, decoder xuất S·f̂, P[0] = 1 − S để encoder chỉ chú ý đặc trưng đã biết. Loss tái tạo chuẩn hóa theo độ lệch chuẩn tổng thể của đặc trưng. Mục tiêu trực giác: các cột bảng phụ thuộc nhau (học vấn đoán được từ nghề nghiệp, giới tính đoán được từ quan hệ gia đình).

## 4. Thiết lập thí nghiệm

- **Tập dữ liệu:** 6 bộ tổng hợp (Syn1–Syn6, của Chen 2018, 10k mẫu); Forest Cover Type; Poker Hand; Sarcos (hồi quy động lực học cánh tay robot); Higgs Boson (10,5M mẫu); Rossmann Store Sales (Kaggle); KDD (Appetency, Churn, Upselling, Census Income); Mushroom; Adult Census Income. Các số của đối thủ **trích từ bài gốc** (cùng chia train/val/test), chỉ TabNet được tinh chỉnh trên tập kiểm chứng.
- **Huấn luyện:** Adam, khởi tạo Glorot uniform, loss softmax cross-entropy (phân loại) / MSE (hồi quy), huấn luyện đến hội tụ; giảm learning rate theo hàm mũ.
- **Không gian siêu tham số:** N_d, N_a ∈ {8, 16, 24, 32, 64, 128}; N_steps ∈ [3, 10]; γ ∈ {1,0; 1,2; 1,5; 2,0}; λ_sparse ∈ {0 đến 0,1}; B ∈ {256…32768}; B_V ∈ {256…4096}; learning rate ∈ {0,005…0,025}; mọi mô hình so sánh cũng được tinh chỉnh với cùng số bước tìm kiếm.

## 5. Kết quả

### 5.1. Chọn đặc trưng theo mẫu (Bảng 1, AUC trên Syn1–Syn6)

| Mô hình | Syn1 | Syn2 | Syn3 | Syn4 | Syn5 | Syn6 |
|---|---|---|---|---|---|---|
| Không chọn | .578 | .789 | .854 | .558 | .662 | .692 |
| Cây | .574 | .872 | .899 | .684 | .741 | .771 |
| Lasso | .498 | .555 | .886 | .512 | .691 | .727 |
| L2X | .498 | .823 | .862 | .678 | .709 | .827 |
| INVASE | .690 | .877 | .902 | .787 | .784 | .877 |
| Chọn toàn cục (biết trước đặc trưng) | .686 | .873 | .900 | .774 | .784 | .858 |
| **TabNet** | .682 | **.892** | .897 | .776 | **.789** | **.878** |

- Với Syn1–Syn3 (đặc trưng quan trọng giống nhau mọi mẫu), TabNet gần bằng chọn toàn cục; với Syn4–Syn6 (đặc trưng phụ thuộc mẫu), TabNet cải thiện so với chọn toàn cục. TabNet vượt cây, Lasso, L2X và ngang INVASE.
- **Kích thước:** các đối thủ dùng mô hình dự đoán 43k tham số (INVASE tổng 101k vì có thêm hai mô hình actor-critic), TabNet chỉ 26k (Syn1–Syn3) và 31k (Syn4–Syn6).

### 5.2. Dữ liệu thực
- **Forest Cover Type (Bảng 2, accuracy):** XGBoost 89,34%; LightGBM 89,28%; CatBoost 85,14%; AutoML Tables 94,95%; **TabNet 96,99%**. Một TabNet không tìm siêu tham số kỹ vẫn vượt AutoML Tables (ensemble DNN, GBDT, AdaNet với tìm kiếm rất kỹ).
- **Poker Hand (Bảng 3):** DT 50,0%; MLP 50,0%; Deep neural DT 65,1%; XGBoost 71,1%; LightGBM 70,0%; CatBoost 66,6%; **TabNet 99,2%**; luật viết tay 100%. Mối quan hệ vào–ra là tất định nhưng dữ liệu mất cân bằng và đòi hỏi các phép sắp xếp/xếp hạng; TabNet thắng nhờ xử lý phi tuyến sâu mà không quá khớp (nhờ chọn đặc trưng theo mẫu).
- **Sarcos (Bảng 4, MSE kiểm tra / kích thước):** Random forest 2,39 (16,7K); Stochastic DT 2,11 (28K); MLP 2,13 (0,14M); Adaptive neural tree 1,23 (0,60M); Gradient boosted tree 1,44 (0,99M); **TabNet-S 1,25 (6,3K)**; TabNet-M 0,28 (0,59M); **TabNet-L 0,14 (1,75M)**. Ở vùng mô hình rất nhỏ TabNet-S ngang mô hình tốt nhất trước đó với số tham số ít hơn khoảng 100 lần; không giới hạn kích thước thì MSE thấp hơn gần một bậc.
- **Higgs Boson (Bảng 5, 10,5M mẫu, accuracy / kích thước):** Sparse evolutionary MLP 78,47% (81K); GBT-S 74,22% (0,12M); GBT-M 75,97% (0,69M); MLP 78,44% (2,04M); GBT-L 76,98% (6,96M); TabNet-S 78,25% (81K); **TabNet-M 78,84% (0,66M)**. Với dữ liệu rất lớn, DNN vượt các biến thể cây ngay cả khi ensemble cực lớn; TabNet thắng MLP với biểu diễn gọn hơn, và ngang sparse evolutionary training cùng số tham số nhưng độ thưa **có cấu trúc** nên tận dụng tốt phần cứng đa nhân.
- **Rossmann Store Sales (Bảng 6, MSE):** MLP 512,62; XGBoost 490,83; LightGBM 504,76; CatBoost 489,75; **TabNet 485,12**. Đặc trưng thời gian (ngày) có độ quan trọng cao; lợi ích của chọn đặc trưng theo mẫu thấy ở các dịp lễ.
- **KDD (Bảng 8, accuracy):**

| Mô hình | Appetency | Churn | Upselling | Census |
|---|---|---|---|---|
| XGBoost | 98,2 | 92,7 | 95,1 | 95,8 |
| CatBoost | 98,2 | 92,8 | 95,1 | 95,7 |
| TabNet | 98,2 | 92,7 | 95,0 | 95,5 |

  Các bộ này đã bão hòa; TabNet ngang hoặc kém nhẹ hơn XGBoost/CatBoost.

### 5.3. Giải thích (Hình 5, Hình 6, Bảng 9)
- **Tổng hợp:** trên Syn2 (đầu ra chỉ phụ thuộc X3–X6), mặt nạ tổng hợp gần như bằng 0 với đặc trưng vô ích. Trên Syn4 (đầu ra phụ thuộc X1–X2 hoặc X3–X6 tùy giá trị chỉ báo X11), TabNet chọn đặc trưng theo mẫu chính xác: dành một mặt nạ cho chỉ báo X11. (Các hình minh họa huấn luyện với 10M mẫu để mặt nạ sắc nét hơn.)
- **Mushroom:** TabNet đạt 100% accuracy; đặc trưng "Odor" (riêng nó đủ cho hơn 98,5% accuracy) được TabNet gán **43%** độ quan trọng, trong khi LIME, Integrated Gradients, DeepLift gán dưới 30%.
- **Adult Census Income:** xếp hạng độ quan trọng của TabNet nhất quán với SHAP, Skater, XGBoost: nhóm quan trọng nhất (Age, Capital gain/loss, Education, Relationship) và ít quan trọng nhất (Native country, Race, Gender, Work class). TabNet đạt 85,7% accuracy.

### 5.4. Tự giám sát
- **Higgs (Bảng 7, TabNet-M, trung bình 15 lần):**

| Số mẫu có nhãn | Chỉ giám sát | Có tiền huấn luyện |
|---|---|---|
| 1k | 57,47 ± 1,78 | 61,37 ± 0,88 |
| 10k | 66,66 ± 0,88 | 68,06 ± 0,39 |
| 100k | 72,92 ± 0,21 | 73,19 ± 0,15 |

- **Forest Cover Type (Bảng 10):** 1k: 65,91 ± 1,02 → 67,86 ± 0,63; 10k: 78,85 ± 1,24 → 79,22 ± 0,78.
- Lợi ích lớn nhất khi dữ liệu không nhãn nhiều hơn rất nhiều dữ liệu có nhãn; mô hình hội tụ nhanh hơn rõ (hữu ích cho học liên tục và thích nghi miền).

### 5.5. Ablation (Phụ lục, Bảng 11, trên Forest Cover Type; cấu hình gốc N_d = N_a = 64, γ = 1,5, N_steps = 5, λ = 1e−4, B = 16384, 470k tham số)
- Giảm năng lực (N_d, N_a = 32; hoặc N_steps = 3) giảm hiệu năng dần dần (khoảng 0,8–2 điểm phần trăm); tăng lên quá mức (N_steps = 9) không có lợi và có thể gây vấn đề tối ưu.
- Thay feature transformer bằng **một lớp chung** vẫn cho kết quả mạnh với mô hình rất gọn (khoảng 35K tham số), cho thấy tầm quan trọng của thiên kiến quy nạp từ chọn đặc trưng và attention tuần tự; tăng độ sâu là cách hiệu quả để đẩy hiệu năng; **chia sẻ một phần** tham số giữa các bước giảm kích thước mà không giảm hiệu năng, tốt hơn cả "toàn bộ riêng" lẫn "toàn bộ chung"; **GLU tốt hơn ReLU** (khoảng 1,4 điểm trên cấu hình một lớp).
- γ gần 1 hoặc λ_sparse quá cao có thể ràng buộc quá chặt và hại hiệu năng; nhưng γ đủ thấp và λ đủ cao tạo thiên kiến quy nạp có lợi. **Batch lớn** (nhờ ghost BN) giúp rõ: batch nhỏ (4096) làm giảm khoảng 3 điểm so với 16384 (số trong bảng, thứ tự cột có thể lệch).
- **Hướng dẫn siêu tham số:** N_steps ∈ [3, 10] (nhiều đặc trưng mang thông tin thì cần nhiều bước hơn, nhưng quá lớn thì quá khớp); N_d = N_a là lựa chọn hợp lý; γ lớn đi cùng N_steps lớn; batch lớn (1–10% kích thước tập huấn luyện); learning rate ban đầu lớn rồi giảm dần.

## 6. Hạn chế

Tác giả không có mục hạn chế riêng; nhận xét đã viết: trên bộ dữ liệu bão hòa (KDD) TabNet chỉ ngang hoặc kém nhẹ GBDT; hiệu năng phụ thuộc tinh chỉnh năng lực mô hình; N_steps và N_d, N_a quá lớn gây quá khớp.

Nhận xét thêm của mình:
1. Các số đối thủ **trích lại từ bài khác**, chứ không chạy lại cùng điều kiện; GBDT ở một số bảng (ví dụ Cover Type 89%) rất thấp so với các benchmark sau này (cùng bộ này, ensemble hiện đại cao hơn nhiều), nên lợi thế của TabNet trong bài có thể do baseline chưa được tinh chỉnh.
2. Các đánh giá độc lập sau đó (bài 29 Grinsztajn, bài 33 McElfresh, Gorishniy 2021) thường thấy **TabNet kém MLP/ResNet tinh chỉnh và GBDT**, trái với kết quả trong bài này. Đây là ví dụ rõ về chênh lệch giữa kết quả trong bài gốc và benchmark công bằng.
3. Mỗi thí nghiệm có một chia dữ liệu và ít lặp lại, không có kiểm định thống kê giữa các mô hình (chỉ Bảng 7, 10 có độ lệch chuẩn).
4. Khả năng giải thích dựa trên mặt nạ attention, nhưng bài chỉ chứng minh trên dữ liệu tổng hợp có đáp án biết trước và vài ví dụ thực, không có đánh giá định lượng về độ trung thực của giải thích.
5. Không xét nhiễu nhãn, mất cân bằng (ngoài Poker Hand), hay hiệu chỉnh xác suất.

## 7. Liên hệ với bài Nature và đề tài

- **Bối cảnh lịch sử:** TabNet đại diện cho giai đoạn 2019–2021 "tìm kiến trúc đặc biệt cho bảng"; sau đó FT-Transformer (bài 30) đưa ra baseline đơn giản hơn và mạnh hơn, và TabPFN/TabPFNv2 (bài 3, 1) chuyển sang hướng hoàn toàn khác (tiền huấn luyện trên dữ liệu tổng hợp, ICL).
- **Tự giám sát:** ý tưởng tiền huấn luyện bằng dự đoán cột bị che tái xuất hiện trong các TFM huấn luyện trên dữ liệu thật (TabSTAR, ConTextTab, TabDPT dùng huấn luyện kiểu dự đoán cột).
- **Giải thích:** nếu báo cáo bàn về khả năng giải thích của TFM, TabNet là điểm so sánh cho giải thích tích hợp trong kiến trúc (khác SHAP hậu nghiệm cho TabPFN).

## 8. Từ khóa cần nhớ

- **Sequential attention + sparsemax:** mỗi bước quyết định chọn đặc trưng thưa theo mẫu; mặt nạ vừa để tính vừa để giải thích.
- **Prior scale (γ), λ_sparse:** điều khiển mức tái sử dụng đặc trưng và độ thưa; **ghost BN** cho batch lớn.
- **Feature transformer:** 2 lớp chung + 2 lớp riêng, FC + BN + GLU, phần dư √0,5.
- **Tự giám sát:** dự đoán cột bị che; lợi ích lớn khi ít nhãn (Higgs 1k: 57,47 → 61,37).
- **Số liệu:** Cover Type 96,99% (so với XGBoost 89,34%); Poker Hand 99,2%; Sarcos MSE 0,14; Higgs 78,84%; Rossmann 485,12; "Odor" 43%.
- **Điểm yếu thực tế:** bảo hòa trên KDD, và đánh giá độc lập sau này thường xếp TabNet thấp.
