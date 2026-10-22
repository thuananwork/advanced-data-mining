# Bài 15. TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems

| Mục | Thông tin |
|---|---|
| Nơi đăng | arXiv:2502.02527v1 [cs.LG], 4/2/2025 (bản tiền công bố) |
| Tác giả | Si-Yang Liu, Han-Jia Ye (Nanjing University, State Key Laboratory for Novel Software Technology; liên hệ: yehj@lamda.nju.edu.cn) |
| Mã nguồn | Không thấy nêu trong bản PDF |
| Lưu ý về PDF | Bảng kết quả chi tiết (Bảng 4 trên 186 tập, Bảng 5 trên 20 tập chiều cao) chỉ có mã tập và con số bị lệch cột; mình chỉ dùng các số trong phần văn bản và Bảng 3. Các hình (CD diagram, bias–variance) chỉ lấy ý từ chú thích và văn bản. Phần tóm tắt (Abstract) bị đưa lên cùng trang đầu. |
| Vị trí trong đề tài | Một **biến thể cải tiến TabPFN (v1)** dựa trên phân tích **bias–variance**; liên quan trực tiếp LoCalPFN (bài 10), TabDPT (bài 8), TuneTables, MixturePFN. Hữu ích để hiểu vì sao các biến thể TabPFN hay tăng chất lượng và vì sao chúng đánh đổi. |

## 1. Bài này làm gì trong một đoạn

TabPFN dự đoán nhãn bằng học trong ngữ cảnh (ICL): đưa cả tập huấn luyện làm "prompt" rồi một lượt chạy xuôi cho ra nhãn của mọi mẫu kiểm tra. Bản v1 chỉ xử lý được tối đa 100 đặc trưng, 10 lớp và vài nghìn mẫu. Nhiều biến thể ra đời nhưng cải thiện rời rạc và thường tăng chi phí suy luận. Bài này dùng khung **phân rã bias–variance** (Nagler, 2023) để phân loại các biến thể: mỗi cách thường chỉ giảm **bias hoặc variance**, hiếm khi cả hai. Từ đó đề xuất **BETA** (Bagging and Encoder-based Fine-tuning for TabPFN Adaptation): (1) gắn một **bộ mã hóa đặc trưng nhẹ** trước TabPFN đóng băng để giảm bias; (2) dùng **nhiều bộ mã hóa** bằng **Batch Ensemble** để giảm variance mà chi phí thấp; (3) lúc suy luận dùng **bootstrapped sampling** (bagging) tạo nhiều tập hỗ trợ trong cùng một lượt chạy; (4) xử lý hơn 10 lớp bằng **ECOC**. Trên khoảng 200 tập phân loại của TALENT và 20 tập chiều cao, BETA hơn hoặc ngang các phương pháp SOTA ở thời điểm đó (xét độ chính xác), trong khi suy luận gần bằng TabPFN gốc.

## 2. Bài toán và kiến thức nền

### 2.1. TabPFN và hạn chế
- TabPFN hai giai đoạn: tiền huấn luyện trên tập tổng hợp, rồi suy luận ICL không cập nhật tham số. Dữ liệu có d < d_max được đệm không (zero-padding) lên d_max (ví dụ 100); nếu d > d_max thì kiến trúc không xử lý được.
- Xác suất dự đoán là softmax của logit q_θ(X̃_q, D̃_train) trên nhãn; thường chạy nhiều lần xáo trộn đặc trưng rồi lấy trung bình (ensemble).
- **PFN-style batching:** gộp toàn bộ mẫu kiểm tra vào một prompt, mặt nạ attention cho tập hỗ trợ chỉ nhìn tập hỗ trợ, tập truy vấn nhìn cả hỗ trợ và chính nó, không nhìn truy vấn khác (tránh rò rỉ giữa các mẫu kiểm tra).
- Tác giả ghi chú rằng TabPFN-v2 (Nature) là công trình **đồng thời**: nó giảm một phần các hạn chế về kích thước và số đặc trưng nhưng chưa giải quyết hết; các cải tiến của bài mang tính tổng quát, bổ trợ được cho v2. (Bản triển khai của BETA dựa trên checkpoint TabPFN v1.)

### 2.2. Các biến thể TabPFN trước đó (hai hướng)
Dự đoán p_θ(y_q | X_q, D_train) phụ thuộc cả **tập ngữ cảnh D_train** và **tham số θ**, nên có hai hướng:
1. **Chọn ngữ cảnh (context selection)** để mở rộng quy mô, vì Transformer có bộ nhớ tăng bậc hai theo độ dài ngữ cảnh: lấy mẫu ngẫu nhiên (hại với tập lớn); sketching (CoreSet, K-Means, Data Distillation); TuneTables (mã hóa tập dữ liệu thành biểu diễn học được, nén); chọn ngữ cảnh riêng từng mẫu: MixturePFN (chia tập huấn luyện thành nhiều tập con, gán tập phù hợp nhất), LocalPFN và TabDPT (láng giềng gần nhất, kNN). **Nhược điểm:** làm hỏng PFN-style batching, giảm mạnh hiệu quả suy luận.
2. **Fine-tune** mô hình để khớp tập hạ nguồn: fine-tune toàn bộ (TabForestPFN, LocalPFN) tốn tài nguyên; TuneTables (toàn bộ hoặc prompt-tuning); MixturePFN dùng adapter.

### 2.3. Phân tích tổng quát hóa bằng bias–variance
Tác giả dùng khung của Nagler (2023) để phân rã sai số tổng quát hóa của TabPFN (và biến thể) thành bình phương bias và variance của q_θ so với p₀(y|x) thật, thí nghiệm trên hai tập thật **Adult** và **Bank** (Hình 2, với kích thước tập huấn luyện 200 đến 12.800). Các biến thể so sánh: TabPFN gốc (không giới hạn độ dài ngữ cảnh), TabPFN-1000 (lấy mẫu con 1000), TabPFN-en-16 (một mẫu, xáo đặc trưng 16 lần), TabPFN-Bagging (bootstrap), KNN-PFN, TabPFN-finetune (fine-tune toàn bộ), và BETA. Các phát hiện:
- Tăng độ dài ngữ cảnh **giảm variance**, bias gần như bão hòa (phù hợp Nagler).
- **kNN chọn ngữ cảnh giảm bias nhưng tăng variance** (vì dùng tập con cục bộ).
- **Fine-tune giảm bias** (khớp prior với tập dữ liệu) nhưng **tăng variance**, nhất là khi tập huấn luyện nhỏ (quá khớp), nên cần điều chuẩn cẩn thận.
- **Ensemble xáo đặc trưng (en-16) giảm variance nhưng có thể tăng bias** (vì biến đổi đặc trưng không giám sát).
- **Bagging giảm mạnh variance, giữ bias ổn định**, với chi phí thấp hơn ensemble.
- Kết luận: các biến thể ảnh hưởng một trong hai phía; cần phương pháp xử lý **đồng thời cả hai**. Bảng 1 so sánh các phương pháp (TabPFN, TuneTables, TabForestPFN, LocalPFN, MixturePFN, BETA) theo tác động bias/variance, dữ liệu lớn, đặc trưng chiều cao, đa lớp, hiệu quả suy luận.

## 3. Phương pháp BETA

BETA có giai đoạn fine-tune và giai đoạn suy luận (Hình 3), giữ **TabPFN đóng băng**.

1. **Bộ mã hóa đặc trưng nhẹ (giảm bias).** Biến dữ liệu có chiều bất kỳ thành biểu diễn có chiều cố định (d_max) phù hợp với TabPFN: cấu trúc chính là MLP hai lớp (chiều ẩn và đầu ra 100, ReLU) cộng **kích hoạt tuần hoàn (periodic activation)** (Gorishniy 2022) để tăng khả năng biểu diễn; có thể xếp chồng nhiều khối. Tự nhiên giải quyết cả vấn đề đặc trưng chiều cao (giảm chiều). Mã hóa được học bằng cách tối thiểu hóa loss dự đoán của TabPFN đóng băng trên tập hỗ trợ/truy vấn đã mã hóa, nên chỉ một lượng tham số nhỏ được tinh chỉnh ("fine-tune một phần").
2. **Nhiều bộ mã hóa + Batch Ensemble (giảm variance).** K bộ mã hóa cùng được huấn luyện chung, mỗi bộ học một phép biến đổi khác nhau của dữ liệu; để rẻ, các lớp tuyến tính trong bộ mã hóa được thay bằng phiên bản **Batch Ensemble**: ma trận trọng số W **dùng chung** và các hệ số theo thành viên (r_k, s_k, b_k) nên số tham số tăng rất ít nhưng giữ được tính đa dạng giữa các thành viên.
3. **Bootstrapped sampling lúc suy luận (giảm variance).** Với mỗi bộ mã hóa k, tạo tập hỗ trợ bootstrap D_bootstrap bằng cách lấy mẫu **có hoàn lại** N_sub mẫu từ tập huấn luyện; dự đoán cuối là trung bình (đồng đều, hoặc có trọng số theo độ tin cậy) các xác suất qua các tập bootstrap/bộ mã hóa. Tất cả nằm trong **một** lượt chạy xuôi của TabPFN nên không tốn thêm chi phí đáng kể so với ensemble xáo đặc trưng, và **giữ nguyên PFN-style batching**. Tác giả cho rằng kỹ thuật bootstrap hầu như bị bỏ qua trong các biến thể TabPFN trước.
4. **ECOC cho hơn 10 lớp.** Dùng Error-Correcting Output Codes (Dietterich & Bakiri, 1995): các bộ mã hóa khác nhau trong cùng một mô hình xử lý các bài toán nhị phân con, không cần huấn luyện nhiều bộ phân loại riêng.
- **Cấu hình mặc định:** kích thước ngữ cảnh 1000, 16 lần lấy mẫu bootstrap; không tìm kiếm siêu tham số cho BETA và các biến thể TabPFN (dùng mặc định) để so sánh công bằng.

## 4. Thiết lập thí nghiệm

- **Dữ liệu:** benchmark **TALENT** (Ye et al., 2024), gồm 120 tập phân loại nhị phân và 80 tập đa lớp từ nhiều lĩnh vực (y tế, sinh học, tài chính, giáo dục, vật lý). Loại những tập trùng với tập validation của TabPFN. Thêm **20 tập chiều rất cao** từ kho scikit-feature (đã loại tập quá 10 lớp), và **12 tập có hơn 10 lớp** từ TALENT cho thí nghiệm ECOC. (Bài nói "hơn 200 tập"; Bảng 4 phụ lục ghi trung bình trên **186 tập**; Hình 5 dùng **173 tập**; Hình 6 nhắc **17** tập chiều cao dù nơi khác nói 20: các con số này không thống nhất hoàn toàn trong bản PDF.)
- **Chia dữ liệu:** TALENT: train/val/test = 64%/16%/20%; trên mỗi tập chạy nhiều seed, trung bình 15 lần chạy ngẫu nhiên (với tập chiều cao: 5 lần chia × 3 seed = 15 lần). Chỉ số: **accuracy**. Kiểm định Wilcoxon–Holm mức 0,05 cho CD diagram.
- **Đối chứng:** phương pháp cổ điển (KNN, SVM, XGBoost, CatBoost, Random Forest, LightGBM), mạng sâu cho bảng (MLP, ResNet, NODE, TabNet, DANets, FT-Transformer, AutoInt, DCNv2, SNN, TabTransformer, GrowNet, TabR, ModernNCA, MLP-PLR, RealMLP, ExcelFormer, TabM...), và các biến thể TabPFN (kNN-PFN, LocalPFN, MixturePFN, TuneTables). Các mô hình tuning bằng **Optuna hơn 100 vòng**, cố định siêu tham số tốt nhất rồi chạy 15 seed. Tiền xử lý theo Gorishniy et al. (2021): chuẩn hóa số, one-hot cho hạng mục. Phần cứng: 4 GPU RTX 4090 và 4 GPU A6000.

## 5. Kết quả

1. **Dưới 10 lớp (CD diagram, Hình 4):** BETA **tốt hơn** các phương pháp khác, kể cả ensemble như TabM, cây truyền thống và các biến thể TabPFN. Tác giả nhấn mạnh tiềm năng của mô hình tiền huấn luyện cho bảng.
2. **Theo kích thước dữ liệu (Hình 5):** trên 173 tập sắp theo số dòng, mức cải thiện tương đối của BETA so với TabPFN **tăng theo kích thước dữ liệu**; riêng ở các tập lớn nhất, **LocalPFN nhỉnh hơn BETA một chút**, còn ở các tập khác BETA tốt hơn.
3. **Hơn 10 lớp (Hình 7, 12 tập):** TabPFN gốc không xử lý được; BETA (ECOC) đạt **độ chính xác trung bình cao nhất** so với KNN, XGBoost, Random Forest, MLP, ModernNCA, TabM; XGBoost và Random Forest **suy giảm** khi số lớp tăng.
4. **Chiều cao (Hình 6, 20 tập):** BETA có **hạng trung bình thấp nhất**; hai phương pháp khác xếp thứ hai và ba; TabPFN gốc và các biến thể khác xếp cuối. FT-T và TabR không chạy được do giới hạn bộ nhớ.
5. **Bias–variance thực nghiệm (Hình 2):** BETA có **sai số tổng quát hóa thấp nhất ở mọi cỡ dữ liệu**; bias giảm nhờ bộ mã hóa fine-tune, variance giảm nhờ bagging và nhiều bộ mã hóa. Ở tập rất nhỏ (vài trăm mẫu), fine-tune bộ mã hóa có thể tăng nhẹ bias vì quá khớp; ngay cả khi mỗi mẫu bootstrap phủ toàn bộ tập, nhiều bộ mã hóa vẫn tạo đa dạng và giảm variance.
6. **Ablation (Hình 8):** tăng số đường mã hóa K → cải thiện tương đối so với TabPFN rõ hơn. Bỏ **periodic activation**, bỏ **Batch Ensemble**, hoặc fine-tune toàn bộ tham số (không "partial parameter tuning") đều làm kết quả kém đi; bản fine-tune toàn bộ có cận trên cải thiện lớn nhất nhưng **chi phí tính toán cao hơn nhiều** và cải thiện trung vị thấp hơn BETA.
7. **Hiệu quả (Bảng 3, trung bình qua các tập):**

| Chỉ số | BETA | TabPFN | LocalPFN | MixturePFN | FT-T | TabR |
|---|---|---|---|---|---|---|
| Thời gian suy luận (giây) | 0,91 | 0,78 | 12,51 | 3,24 | 0,36 | 0,38 |
| Hạng trung bình | **6,76** | 15,63 | 12,15 | 12,86 | 13,27 | 9,26 |
| Dung lượng checkpoint (MB, ước tính số tham số học được) | 0,60 | 0 | 98,65 | 21,42 | 7,15 | 14,07 |

   BETA suy luận gần bằng TabPFN (0,91 so với 0,78 giây) và nhanh hơn LocalPFN khoảng **14 lần**, mà hạng tốt hơn rõ rệt. Số tham số học thêm chỉ khoảng 0,6 MB.
8. **Trực quan hóa t-SNE (Bank, KDD, Spambase):** MLP tạo cụm gọn theo lớp; TabR và ModernNCA tạo nhiều cụm cho một lớp; BETA **không ép** mỗi lớp thành một cụm gọn, phù hợp với cách TabPFN được tiền huấn luyện (không cần biểu diễn tách hoàn toàn); các đường mã hóa khác nhau cho các không gian biểu diễn đa dạng.

## 6. Hạn chế

Tác giả nêu:
- Chỉ **phân loại**; chưa mở rộng sang hồi quy (do thiết kế TabPFN v1).
- Giả định **i.i.d.**: huấn luyện và kiểm tra cùng phân phối; chưa xử lý covariate shift hay concept drift (bài chủ ý không giải quyết, vì là bài toán riêng).
- Phụ thuộc prior của TabPFN và độ nhạy với đặc điểm tập dữ liệu; cần thận trọng khi triển khai ở ứng dụng rủi ro cao.

Nhận xét thêm của mình:
1. Dựa trên **TabPFN v1** (checkpoint v1) nên kết quả không trực tiếp chuyển sang TabPFNv2 (bài Nature) hay TabICL; tác giả chỉ nói các cải tiến "có thể bổ trợ".
2. Đánh giá chủ yếu bằng **accuracy** (không AUC, log-loss, hiệu chuẩn), và dữ liệu TALENT; số tập, số seed ghi không nhất quán trong bản PDF (186 / 173 / 17 / 20).
3. Mỗi biến thể TabPFN dùng **siêu tham số mặc định**, còn các mô hình khác tinh chỉnh Optuna; BETA tuy vậy vẫn có bước huấn luyện bộ mã hóa riêng cho mỗi tập, nên "không cập nhật tham số" của ICL thuần không còn đúng (chi phí huấn luyện không được báo cáo ở thân bài).
4. Không có thí nghiệm về nhãn nhiễu hay mất cân bằng.

## 7. Liên hệ với bài Nature và đề tài

- **Cung cấp lăng kính bias–variance** cho mọi cải tiến TabPFN: tăng ngữ cảnh/bagging → giảm variance; chọn ngữ cảnh cục bộ (LocalPFN, TabDPT) và fine-tune → giảm bias nhưng có thể tăng variance. Hữu ích để giải thích kết quả của LoCalPFN (bài 10) và TabDPT (bài 8) trong ma trận so sánh.
- **So với TabPFNv2 (bài 1):** v2 đã có xáo trộn đặc trưng ensemble, nhưng BETA cho thấy bagging bootstrap và bộ mã hóa nhẹ là hướng bổ sung, không cần tiền huấn luyện lại.
- **Với đề tài nhiễu nhãn:** bagging giảm variance có thể giúp bền với nhiễu (như Random Forest, bài 23 đã thử nhiễu nhãn 5%), nhưng bài này không kiểm tra; là câu hỏi mở có thể làm trong final-term (bagging bootstrap có giúp TabICLv2/TabPFNv2 chống nhiễu nhãn?).

## 8. Từ khóa cần nhớ

- **BETA** = Bagging + Encoder-based fine-tuning cho TabPFN; mã hóa MLP 2 lớp (100 chiều) + periodic activation + Batch Ensemble; bootstrap 16 lần, ngữ cảnh 1000; ECOC cho >10 lớp.
- **Chẩn đoán bias–variance:** ngữ cảnh dài → giảm variance; kNN-context & fine-tune → giảm bias, tăng variance; en-16 → giảm variance, có thể tăng bias; bagging → giảm variance giữ bias.
- **Kết quả:** hạng trung bình 6,76 so với TabPFN 15,63 và LocalPFN 12,15; suy luận 0,91 s so với 12,51 s của LocalPFN; checkpoint 0,60 MB.
- **Hạn chế:** chỉ phân loại, dựa trên TabPFN v1, giả định i.i.d.
