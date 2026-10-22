# Bài 16. ConTextTab: A Semantics-Aware Tabular In-Context Learner

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025; arXiv:2506.10707v4 [cs.LG], 3/11/2025 |
| Tác giả | Marco Spinaci (SAP France), Marek Polewczyk, Maximilian Schambach, Sam Thelin (SAP SE); ba người đầu đồng tác giả chính |
| Mã nguồn | https://github.com/SAP-samples/sap-rpt-1-oss (có cả checkpoint) |
| Lưu ý về PDF | Bảng 1, 2 và 3 bị lệch cột khi trích chữ. Mình ghép lại Bảng 1 theo thứ tự hàng và đối chiếu với văn bản; Bảng 2 (ablation) hầu như không đọc được theo từng ô nên chỉ dùng phần văn bản. Các hình (CD diagram, win ratio, runtime) chỉ lấy ý từ chú thích và văn bản. |
| Vị trí trong đề tài | Mô hình TFM **huấn luyện trên dữ liệu thật** có hiểu ngữ nghĩa (tên cột, giá trị hạng mục, ngày tháng). Đối chứng cho nhóm TFM tổng hợp (TabPFNv2, TabICL) và gần TabDPT về tinh thần dùng dữ liệu thật. |

## 1. Bài này làm gì trong một đoạn

TabPFNv2 và TabICL đều tiền huấn luyện hoàn toàn trên dữ liệu **tổng hợp** thuần số, nên chúng không có khái niệm về ngữ nghĩa: tên cột không được dùng, giá trị hạng mục chỉ được mã hóa one-hot/ordinal, ngày tháng không được hiểu. Ở phía ngược lại, các mô hình ICL dựa trên LLM (điển hình TabuLa-8B) hiểu ngữ nghĩa nhưng token hóa văn bản rất tốn chỗ (tối đa chỉ 32 dòng ngữ cảnh), mất cấu trúc 2D và không bất biến với hoán vị hàng/cột. ConTextTab nối hai thế giới: giữ kiến trúc **table-native** của TabPFNv2 (attention xen kẽ theo cột và theo hàng) nhưng thay lớp mã hóa bằng các embedding riêng cho từng kiểu dữ liệu (văn bản bằng một mô hình nhúng câu nhỏ, ngày bằng ba embedding ngày/tháng/năm, số bằng phép chiếu một chiều) và **tiền huấn luyện trên dữ liệu thật** (bộ T4, 2,18 triệu bảng). Trên 203 tập (91 hồi quy, 112 phân loại) ở 5 benchmark, mô hình **đạt hạng 1 trên benchmark CARTE giàu ngữ nghĩa** (có ý nghĩa thống kê so với mọi mô hình ngoại trừ CatBoost và AutoGluon đã tinh chỉnh/ensemble), và **ngang bằng các phương pháp hiện có trên các benchmark không ngữ nghĩa**, nhưng thua cây tăng cường và RealMLP được tinh chỉnh ensemble khi dữ liệu lớn.

## 2. Bài toán và động cơ

- Dữ liệu bảng ngoài đời thật chứa nghĩa: tên cột ("Giá", "Mô tả"), giá trị hạng mục, văn bản tự do, ngày tháng, ngày lễ. Mô hình tiền huấn luyện trên dữ liệu tổng hợp bỏ phí phần này.
- Tác giả lập luận nghĩa chỉ học được từ **nhiều bảng thật**; vì thế họ dùng T4 (khoảng 3 triệu bảng tách từ kho TabLib, do TabuLa-8B tạo ra).
- Hạn chế của hướng LLM (TabuLa-8B): tối đa 32 dòng ngữ cảnh, token hóa không đồng đều (mỗi ô thành số token khác nhau nên mất cấu trúc 2D), không bất biến với hoán vị hàng/cột.
- Hạn chế của TFM tổng hợp: dữ liệu không có giá trị có nghĩa; tên cột không dùng; hạng mục mã hóa bỏ nghĩa; không có kiểu ngày/giờ.

## 3. Phương pháp

### 3.1. Mã hóa theo kiểu dữ liệu
- **Văn bản** (cả cột hạng mục lẫn văn bản tự do): mỗi ô qua một mô hình nhúng câu có sẵn; chọn mô hình nhỏ và nhanh **all-MiniLM-L6-v2** vì mỗi bảng phải nhúng rất nhiều giá trị, rồi một lớp tuyến tính học được đưa về chiều d.
- **Ngày:** nhúng riêng ngày, tháng, năm rồi **cộng ba vector**; giúp so sánh hai ngày và biết ngày đặc biệt (ngày lễ lặp lại), tiết kiệm token hơn cách tách thành nhiều đặc trưng kiểu AutoGluon.
- **Số:** mã hóa một chiều: cắt theo phân vị 2%–98%, chuẩn hóa trung bình 0 và phương sai 1 (theo bất đẳng thức Chebyshev giá trị bị chặn trong (−7,1; 7,1) nên tránh nổ gradient), rồi nhân với một vector học được và cộng bias. NaN thay bằng 0, nên bias đóng vai trò cờ "is-NaN".
- **Tên cột:** nhúng bằng cùng mô hình văn bản, qua một lớp tuyến tính riêng rồi **cộng vào embedding ô**; thay vai trò mã hóa vị trí của TabPFN/TabICL.
- Sau khi cộng, qua layer norm. Mọi embedding trên đều **tương đẳng với hoán vị hàng và cột**, giúp dự đoán tái lập hơn và giảm nhu cầu bagging.

### 3.2. Backbone
Giữ gần như nguyên kiến trúc TabPFN: các lớp self-attention xen kẽ "ngang" (giữa các cột, không mặt nạ) và "dọc" (giữa các hàng, có mặt nạ để mỗi hàng chỉ nhìn ngữ cảnh). Cỡ mô hình gọi theo BERT: "base" = 12 lớp, chiều ẩn 768. **Chia sẻ trọng số** giữa các khối (như RNN trải theo độ sâu): không ảnh hưởng hiệu năng nên là mặc định. Cấu hình mặc định (n = 12, d = 768, d_ff = 3072) có **172 triệu tham số, giảm còn 16 triệu tham số huấn luyện được** khi chia sẻ trọng số.

### 3.3. Giải mã
- **Phân loại:** MLP rồi cross-entropy. Hạn chế: không vượt được số lớp đã thấy khi tiền huấn luyện nếu không dùng cách phân cấp như TabPFN/TabDPT/TabICL, và không dùng được nghĩa nhãn lớp. Có thêm mã hóa riêng cho cột đích, phá tính bất biến theo hoán vị nhãn nhưng hiệu quả với bài toán ít lớp.
- **Hồi quy:** dự đoán trực tiếp giá trị số đã cắt và chuẩn hóa, huấn luyện bằng L2.

### 3.4. Các kiến trúc thay thế đã thử
- **Soft binning (mã hóa số) + bin averaging (giải mã):** chia số theo phân vị, nội suy giữa hai vector bin gần nhất; hồi quy chuyển thành phân loại bin, dự đoán là kỳ vọng Σ p_i q_i.
- **Đầu phân cụm có giám sát:** chiếu đầu ra cột đích bằng MLP hai lớp ra vector x, tính cosine similarity giữa dòng truy vấn và dòng ngữ cảnh, so với ma trận kề (1 nếu cùng lớp) bằng binary cross-entropy. Giữ nghĩa nhãn, giữ bất biến, **không giới hạn số lớp**; hữu ích khi nhiều lớp mà mỗi lớp ít mẫu.
- **ISAB (Induced Set Attention Block):** thay một số khối attention theo hàng bằng khối dùng "inducing points" để giảm độ phức tạp bậc hai; chỉ áp dụng cho attention theo hàng, dùng cho m khối đầu (ví dụ 3), rồi một khối MAB, sau đó attention chuẩn.

## 4. Thiết lập thí nghiệm

### 4.1. Huấn luyện và suy luận
- **Dữ liệu:** T4, bỏ bảng dưới 150 dòng còn **2,18 triệu bảng** (trung vị 750 dòng, 9 cột). Mỗi lần: 1000 dòng ngẫu nhiên, 50–900 dòng làm truy vấn, phần còn lại làm ngữ cảnh; chọn một cột đích, loại cột ngày, cột số có hơn 50% NaN và cột có hơn 20% giá trị duy nhất; lấy mẫu tăng cột không phải số để tỉ lệ hồi quy:phân loại gần cân bằng.
- **Huấn luyện:** 4–10 triệu bước (2–5 epoch); batch hiệu dụng 256 (128 cho cỡ "mini") bằng tích lũy gradient; AdamW, learning rate tối đa 1e-4, khởi động 1000 bước; cắt gradient. Mô hình "base" trên **một GPU H100**, khoảng 10 bảng/giây, mất **4–12 ngày**.
- **Curriculum:** thử thêm bước hai với dữ liệu của TabDPT (123 bảng, trung vị 11k dòng, 34 cột), tăng lên 4000 dòng; không cải thiện đáng kể.
- **Suy luận:** bagging 8 phần: mỗi phần lấy c dòng ngữ cảnh có hoàn lại từ tập train, dự đoán riêng rồi trung bình; mặc định **c = 8192** (lúc huấn luyện c không quá 950), nên tối đa 8c điểm ngữ cảnh; nếu hơn 500 cột thì mỗi bag lấy tối đa 500 cột.

### 4.2. Đánh giá
- **Benchmark:** OpenML-CC18 (phân loại), OpenML-CTR23 (hồi quy), **TALENT-Tiny** (45 bộ chọn từ hơn 300 bộ của TALENT), **TabReD** (8 bộ lớn, thực tế), **CARTE** (nhiều đặc trưng ngữ nghĩa, ít đặc trưng số). Tổng **91 hồi quy + 112 phân loại = 203 tập**, 5 đến 3000 cột, 400 đến khoảng 400k dòng huấn luyện. **Không dùng cross-validation**, chỉ một split kiểm tra cố định mỗi tác vụ.
- **Đối chứng:** TabPFN v2.1.0, TabICL v0.1.1 (chỉ phân loại), TabDPT v1.1 (k=2048), CARTE, RealMLP, TabM, XGBoost, LightGBM, CatBoost (mặc định D, meta-tuned TD, HPO với Parzen-tree, và HPO + ensemble CV 5-fold), cùng baseline scikit-learn, và **AutoGluon v1.2** làm chuẩn vàng (không tính vào xếp hạng tổng vì là stack nhiều mô hình).
- **Thước đo:** accuracy (phân loại), R² (hồi quy) với **soft-clipping** (R² âm ánh xạ qua tanh về [−1, 0)); hạng trung bình trên toàn bộ tập, **hòa nếu cách nhau dưới 0,005**; phụ lục có CD diagram, win ratio và p-value Wilcoxon hai phía.

## 5. Kết quả (Bảng 1, ghép theo thứ tự hàng; acc và R² theo %)

| Mô hình | Hạng chung | CARTE: hạng | CARTE: Acc | CARTE: R² | CC18 Acc | CTR23 R² | TALENT-Tiny Acc | TALENT-Tiny R² |
|---|---|---|---|---|---|---|---|---|
| AutoGluon v1.2 (tham chiếu) | n/a | n/a | 78,7 | 73,8 | 88,5 | 68,6 | 87,9 | 73,7 |
| **ConTextTab** | 2,96 | **1,55** | **76,9** | **72,4** | 86,6 | 73,3 | 87,6 | 76,1 |
| LightGBM (HPO, CV) | 2,85 | 3,00 | 73,4 | 67,5 | 87,5 | 68,8 | 86,9 | 74,1 |
| CatBoost (HPO, CV) | 2,71 | 3,06 | 75,9 | 68,2 | 87,7 | 68,8 | 87,2 | 76,4 |
| RealMLP (HPO, CV) | **2,50** | 3,20 | 73,4 | 67,5 | **88,3** | 75,0 | 88,3 | 76,4 |
| TabM (HPO, CV) | 2,98 | 3,71 | 73,2 | 66,9 | 87,9 | 74,3 | 87,5 | 75,7 |
| XGBoost (HPO, CV) | 3,11 | 3,78 | 73,1 | 66,7 | 87,5 | 70,9 | 86,8 | 73,6 |
| TabPFN v2.1.0 | 3,63 | 5,76 | 72,3 | 65,0 | 87,4 | 75,1 | 87,8 | 74,4 |
| TabDPT v1.1 | 4,18 | 5,98 | 72,7 | 65,1 | 87,8 | 73,2 | 86,7 | 74,8 |
| HistGradBoost | 5,27 | 6,27 | 72,5 | 64,8 | 86,1 | 65,5 | 86,3 | 67,6 |
| Random forest | 6,22 | 8,00 | 71,5 | 63,3 | 85,7 | 67,9 | 85,8 | 70,6 |
| Naive | 10,51 | 11,0 | 53,0 | −1,8 | 47,0 | −8,4 | 53,4 | −19,2 |

(TabICL và mô hình CARTE chỉ có một phần cột nên lược bớt. Cột TabReD: hạng tốt nhất thuộc LightGBM và XGBoost cùng 1,25; ConTextTab 4,12; TabPFN 4,25; TabDPT 6,88; cây tinh chỉnh đạt khoảng 86,0 acc và 64,9 R².)

Nhận xét:
1. **CARTE:** ConTextTab hạng 1,55, hơn xa các TFM khác (TabPFN 5,76). Phụ lục A.2: thắng TabPFN ở **96% tập** của CARTE với p rất nhỏ; thắng AutoGluon ở 65% tập nhưng p = 0,03, nên gain không lớn như số trung bình gợi ý. Trên benchmark không ngữ nghĩa, TabPFN thắng ConTextTab ở 49%–66% tập với p = 0,09–0,64 (không khác biệt thống kê).
2. **Benchmark không ngữ nghĩa:** ConTextTab "ngang bằng" nhưng hạng sau cây HPO+CV và RealMLP; các chênh lệch phần lớn không có ý nghĩa thống kê, trừ OpenML-CTR23 nơi nó kém mô hình tốt nhất có ý nghĩa (nhưng không khác cây ensemble đã tinh chỉnh).
3. **Theo kích thước dữ liệu:** trên CARTE với dữ liệu con từ 128 dòng đến đủ, ConTextTab dẫn đầu mọi cỡ và **vượt AutoGluon đến 2048 mẫu huấn luyện**. Trên cả 203 tập: dưới 1000 dòng TabPFN tốt nhất; trên 1000 dòng ConTextTab vượt TabPFN; cạnh tranh với AutoGluon đến khoảng 10.000 dòng; sau đó **cây tăng cường và RealMLP vượt cả ConTextTab và TabPFN**. Bộ đánh giá bị chi phối bởi các tập dưới 10k dòng và dưới 100 cột.
4. **TabReD:** cây tinh chỉnh thậm chí vượt AutoGluon ở vài chỉ số.
5. **Thời gian chạy:** bảng giả 100 cột, 1000–10.000 dòng ngữ cảnh: 0,1 s đến 9 s mỗi lần dự đoán một dòng; chậm hơn TabPFN/TabICL do nhúng văn bản, khoảng **gấp đôi ở 10.000 dòng**; tăng bậc hai theo số dòng. XGBoost và RealMLP bản HPO+CV có thời gian fit+predict **dài hơn hai bậc độ lớn**.

### 5.1. Ablation
- **Ngữ nghĩa giá trị ô** (chỉ CARTE): thay bằng ordinal, MinHashEncoder, GapEncoder (skrub) hay bộ mã hóa AutoGluon thì hiệu năng giảm rõ.
- **Ngữ nghĩa tên cột:** thay bằng col1…colN mất khoảng **1% accuracy và 2% R²**, win rate của mô hình gốc 92%. Làm giàu tên cột bằng mô tả do LLM (gemma3-12b) chỉ tăng nhẹ, win rate 57%, p = 0,4, **không có ý nghĩa thống kê**.
- **Cỡ mô hình:** nhỏ hơn "base" thì kém; "large" không cải thiện đáng kể.
- **Lượng dữ liệu tiền huấn luyện:** cần ít nhất khoảng **100.000 bảng** để đạt chất lượng SOTA rồi tăng thêm ít lợi, nên bước curriculum với 123 bảng của TabDPT không giúp. Tác giả cho rằng huấn luyện bị giới hạn bởi lượng hoặc độ đa dạng của T4 (thiếu bảng dài và rộng).
- **Ngữ cảnh khi suy luận:** tăng đơn điệu (thử 1024 đến 20000), mạnh hơn ở tập lớn, vượt cả ngữ cảnh lớn nhất lúc huấn luyện (950) nhưng bão hòa với dữ liệu rất lớn.
- **Đầu phân loại:** cross-entropy mặc định nhỉnh hơn đầu phân cụm vì hầu hết bộ đánh giá có nhãn không mô tả và ít lớp.
- **ISAB:** phân loại tương đương, hồi quy giảm vừa phải, nhưng bỏ được độ phức tạp bậc hai.
- **Bagging:** có lợi nhất quán nhưng biên nhỏ nhờ kiến trúc đã tương đẳng; ngược lại ensemble CV của baseline tăng rất mạnh, nên cách tận dụng ensemble cho ICL còn là câu hỏi mở.
- **Không chia sẻ trọng số:** hơi kém hơn.
- **Khác:** mô hình nhúng câu khác (multilingual-e5-small, gte-multilingual-base) không khác đáng kể; cắt đuôi 2% và binning làm giảm nhẹ chất lượng; **fine-tune** đồng thời trên 203 tập với cột đích cố định gây quá khớp, tránh được nếu chọn cột đích ngẫu nhiên.

## 6. Hạn chế

Tác giả nêu:
1. **Rò rỉ dữ liệu:** huấn luyện trên dữ liệu thật có thể chứa tập đánh giá. Với CARTE họ so sánh embedding tên cột và giá trị, **không phát hiện** rò rỉ; mỗi bảng chỉ thấy vài lần nên nhớ vẹt khó xảy ra.
2. **Mọi TFM table-native** (ConTextTab, TabPFN, TabDPT, TabICL) **không mở rộng tốt lên dữ liệu rất lớn**; tăng ngữ cảnh và bagging không giải quyết hết. Với dữ liệu lớn, phương pháp cổ điển (đặc biệt AutoGluon) vẫn tốt nhất.
3. Thêm dữ liệu tiền huấn luyện cho lợi ích thấp; cần dữ liệu thật đa dạng hơn và benchmark giàu ngữ nghĩa hơn (CARTE chỉ có nhãn nhị phân, chủ yếu bảng nhỏ).
4. Đầu phân cụm không cho lợi ích vì thiếu benchmark nhãn có nghĩa, nhiều lớp.

Nhận xét thêm của mình:
- Mỗi tập chỉ **một split cố định**, không CV hay lặp seed, nên chênh lệch nhỏ khó kết luận.
- Không kiểm tra **nhãn nhiễu, mất cân bằng, hiệu chuẩn xác suất**; thước đo chỉ accuracy và R² (không AUC, log-loss).
- Lợi thế chủ yếu đến từ **ngữ nghĩa văn bản**; với dữ liệu thuần số nó chỉ ngang TabPFN.

## 7. Liên hệ với bài Nature và đề tài

- **So với TabPFNv2 (bài 1):** cùng khung attention xen kẽ hàng/cột và ICL; khác ở dữ liệu tiền huấn luyện (thật so với tổng hợp) và mã hóa ngữ nghĩa. Dữ liệu tổng hợp đủ cho bảng thuần số, nhưng không đủ cho dữ liệu có nghĩa.
- **So với TabDPT (bài 8):** cả hai dùng dữ liệu thật, TabDPT dùng retrieval kiểu kNN còn ConTextTab dùng bagging và ngữ cảnh lớn; trên CARTE ConTextTab hạng 1,55 so với 5,98.
- **Đề tài:** nếu bộ dữ liệu có cột hạng mục/văn bản/ngày, ConTextTab đáng đưa vào so sánh; nếu thuần số, TabPFNv2/TabICLv2 đủ. Với khoảng trống nhiễu/mất cân bằng: bài không đụng tới.

## 8. Từ khóa cần nhớ

- **Table-native + semantic:** attention ngang/dọc, embedding theo kiểu dữ liệu (văn bản MiniLM, ngày = ngày + tháng + năm, số 1 chiều, tên cột cộng vào ô).
- **T4:** 2,18 triệu bảng thật; 172M tham số (16M học được khi chia sẻ trọng số); 4–12 ngày trên 1 H100; suy luận 8 bag × 8192 dòng.
- **Kết quả:** hạng 1 trên CARTE (1,55), ngang trên benchmark không ngữ nghĩa, vượt AutoGluon đến 2048 mẫu; cây/RealMLP thắng trên 10k dòng; bỏ tên cột mất ~1% acc / 2% R².
- **Hạn chế:** không mở rộng tốt lên dữ liệu lớn; thiếu benchmark ngữ nghĩa; không có thí nghiệm nhiễu/mất cân bằng.
