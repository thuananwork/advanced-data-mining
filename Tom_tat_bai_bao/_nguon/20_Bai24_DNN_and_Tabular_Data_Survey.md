# Bài 24. Deep Neural Networks and Tabular Data: A Survey

| Mục | Thông tin |
|---|---|
| Nơi đăng | IEEE Transactions on Neural Networks and Learning Systems (bản gửi IEEE, 6/2022); arXiv 2110.01889v3 (29/6/2022) |
| Tác giả | Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, Gjergji Kasneci (Data Science and Analytics Research group, Đại học Tübingen) |
| Mã nguồn | github.com/kathrinse/TabSurvey (benchmark mở, chạy trong Docker) |
| Lưu ý về PDF | Bản PDF text mất trang đầu của phần tiêu đề; phần dưới đây bám theo đúng nội dung đọc được (tóm tắt, mục I–IX). Phần tài liệu tham khảo được bỏ qua. |
| Vị trí trong đề tài | Bài khảo sát tổng quan về học sâu cho dữ liệu bảng **trước thời TFM**; dùng làm bản đồ phân loại phương pháp và làm nền cho luận điểm "GBDT vẫn là SOTA cho dữ liệu nhỏ và vừa". |

## 1. Bài này làm gì trong một đoạn

Đây là khảo sát có hệ thống đầu tiên về mạng nơ-ron sâu cho **dữ liệu bảng không đồng nhất** (heterogeneous), gồm ba mảng: **suy luận** (phân loại/hồi quy), **sinh dữ liệu bảng** và **giải thích mô hình**. Tác giả đề xuất một **phân loại thống nhất ba nhánh**: (1) biến đổi dữ liệu, (2) kiến trúc chuyên biệt (hybrid và transformer), (3) mô hình điều chuẩn. Ngoài khảo sát, bài đóng góp một **benchmark mở** so 11 mô hình sâu với các mô hình học máy cổ điển trên 5 bộ dữ liệu thực. Kết luận: **các mô hình cây tăng cường gradient (XGBoost, LightGBM, CatBoost) vẫn thắng hầu hết**, trừ bộ rất lớn HIGGS, nơi SAINT dẫn đầu; tác giả nói tiến bộ của học sâu trên dữ liệu bảng đang "đình trệ".

## 2. Vấn đề và động cơ (mục I)

- Dữ liệu bảng là dạng dữ liệu phổ biến nhất, trong chẩn đoán y tế, phân tích tài chính (rủi ro tín dụng), dự đoán tỉ lệ nhấp (CTR), hệ gợi ý, dự đoán rời bỏ khách hàng, an ninh mạng, phát hiện gian lận, v.v. Khác với ảnh/âm thanh, dữ liệu bảng **không đồng nhất** (đặc trưng số dày và hạng mục thưa), tương quan giữa đặc trưng yếu hơn tương quan không gian/ngữ nghĩa trong ảnh hay văn bản. Kadra et al. gọi dữ liệu bảng là "lâu đài chưa chinh phục được" của mạng nơ-ron.
- Lý do vẫn muốn dùng mạng sâu: linh hoạt, huấn luyện lặp hiệu quả, hữu ích cho AutoML, **sinh dữ liệu bảng** (giảm mất cân bằng lớp), học đa phương thức, chưng cất, học liên kết.
- Ba nhiệm vụ cốt lõi: (1) suy luận, (2) sinh dữ liệu (điền giá trị thiếu, cân bằng lớp, quyền riêng tư), (3) giải thích (quy định như CCPA, GDPR đòi quyền được giải thích; cần cho kiểm toán và gỡ lỗi).
- Bảy mục tiêu của khảo sát: tổng hợp tài liệu; phân loại; trình bày hiện trạng sinh dữ liệu; tổng quan giải thích; **so sánh thực nghiệm rộng**; thảo luận lý do học sâu kém trên dữ liệu bảng; liệt kê vấn đề mở.
- Công trình liên quan (mục II): Hancock & Khoshgoftaar (mã hóa hạng mục); Sahakyan et al. (giải thích cho dữ liệu bảng); Gorishniy et al. (đánh giá nhiều mô hình sâu; ResNet được tinh chỉnh ngang một số mô hình sâu SOTA); **Shwartz-Ziv & Armon** (bài 25: nhiều mô hình sâu bị XGBoost thách thức, không mô hình sâu nào thắng tất cả). Bài này tích hợp thêm sinh dữ liệu và giải thích.

## 3. Định nghĩa, lịch sử, thách thức, phân loại (mục III)

- **Định nghĩa:** mạng nơ-ron sâu = mạng truyền thẳng y = f(x) ≈ f̂(x; W). Dữ liệu bảng = tập có số đặc trưng cố định, mỗi đặc trưng liên tục hoặc hạng mục (nhị phân, thứ tự, hạng mục cardinality cao). Ví dụ Bảng I lấy từ Adult (tuổi, học vấn, nghề, giới tính, thu nhập).
- **Lịch sử ngắn:** dữ liệu bảng là dạng dữ liệu cổ nhất được phân tích thống kê; học sâu phát triển theo hướng dữ liệu đồng nhất. Thương mại điện tử/quảng cáo thúc đẩy CTR với nhiều biến hạng mục thưa; sau đó là làn sóng điều chuẩn (Shavitt & Segal) và transformer/tự giám sát.
- **Bốn nguyên nhân khiến học sâu kém trên dữ liệu bảng:**
  1. **Dữ liệu chất lượng thấp:** thiếu giá trị, ngoại lai, lỗi/không nhất quán, cỡ mẫu nhỏ so với số chiều, **mất cân bằng lớp** (do thu thập đắt đỏ). Cây hiện đại xử lý thiếu và thang đo lạ nội tại.
  2. **Thiếu hoặc phức tạp phụ thuộc không gian:** phải học cấu trúc từ đầu; thiên lệch quy nạp của CNN không phù hợp.
  3. **Phụ thuộc tiền xử lý:** hiệu năng tùy chiến lược; hạng mục dẫn đến ma trận thưa (one-hot) hoặc thứ tự giả (ordinal); có mất thông tin.
  4. **Tầm quan trọng của từng đặc trưng:** đổi nhỏ của một đặc trưng hạng mục/nhị phân có thể lật hẳn dự đoán; cây chọn một đặc trưng và ngưỡng và "bỏ qua phần còn lại".
- **Phân loại thống nhất (Hình 1):** Data Transformation (Single-Dimensional Encoding, Multi-Dimensional Encoding); Specialized Architectures (Hybrid: fully differentiable / partly differentiable; Transformer-based); Regularization Models.

## 4. Phương pháp học sâu cho dữ liệu bảng (mục IV, Bảng II)

### 4.1. Biến đổi dữ liệu
- **Mã hóa một chiều (từng đặc trưng):** ordinal/label (thứ tự giả); one-hot (không thứ tự nhưng thưa, tăng chiều); binary encoding (log c cột); **leave-one-out / target encoding** (thay hạng mục bằng trung bình nhãn, loại mẫu hiện tại; dùng trong CatBoost); hash-based encoding (kích thước cố định, chọn tay).
- **Mã hóa đa chiều:**
  - **VIME** (tự giám sát và bán giám sát): hai nhiệm vụ tiền huấn luyện: ước lượng vector mặt nạ (giá trị nào bị nhiễu) và khôi phục vector đặc trưng; bộ mã hóa MLP; bán giám sát dùng mất mát nhất quán L = L_s + β·L_u trên nhiều bản nhiễu của một mẫu.
  - **SuperTML** (chuyển bảng thành ảnh đen trắng để dùng CNN) và **IGTD** (tối ưu để khoảng cách pixel phản ánh khoảng cách đặc trưng); IGTD chỉ thử trên biểu hiện gen và mô tả phân tử thuốc, nên tính tổng quát chưa rõ; có thể thất bại nếu đặc trưng độc lập.

### 4.2. Kiến trúc chuyên biệt
**Hybrid, khả vi hoàn toàn** (huấn luyện đầu-cuối bằng gradient):
- **NODE:** ensemble cây quyết định oblivious khả vi (entmax, tách mềm), lấy cảm hứng từ CatBoost; trong cài đặt chính thức, chuỗi được chuyển sang số bằng leave-one-out.
- **SDT** (Frosst & Hinton): chưng cất mạng sâu vào cây mềm, dễ diễn giải nhưng còn khoảng cách hiệu năng; **SDTR** mở rộng cho bảng không đồng nhất và hồi quy, hứa hẹn trong bộ nhớ thấp.
- **Net-DNF:** cây quyết định là dạng chuẩn tắc tuyển (DNF); kiến trúc mô phỏng GBDT; chỉ thử trên dữ liệu không thiếu, chủ yếu đặc trưng số; không nêu cách xử lý hạng mục cardinality cao.
- **DNN2LR:** mạng sâu kết hợp đặc trưng rồi đưa vào mô hình tuyến tính (giữ diễn giải).
- **Wide&Deep** (tuyến tính trên đặc trưng thủ công + mạng sâu, embedding cho hạng mục), **DeepFM** (thay đặc trưng thủ công bằng Factorization Machine học được), **NON** (network-on-network: mạng theo cột, mạng liên cột, mạng hợp nhất; không so với cây; tốn thời gian).

**Hybrid, khả vi một phần** (kết hợp cây không khả vi):
- **DeepGBM:** CatNN (hạng mục thưa) + GBDT2NN (chưng cất tri thức từ chỉ số lá của GBDT); dự đoán trực tuyến nhanh hơn GBDT; nhược điểm: chỉ số lá là đặc trưng hạng mục "meta"; không rõ ảnh hưởng của thiếu giá trị, thang đo, nhiễu.
- **TabNN:** chưng cất nhóm đặc trưng từ GBDT, gom cụm, xây mạng; không có cài đặt công khai, dùng thực tế hạn chế.
- **BGNN:** dùng cấu trúc cây từ GBDT làm đồ thị cho GNN; vượt đối thủ về độ chính xác và thời gian huấn luyện.

**Transformer:**
- **TabNet:** các bước quyết định tuần tự, mặt nạ thưa theo mẫu (sparsemax), chọn đặc trưng mềm; có diễn giải cục bộ và toàn cục; có bộ giải mã cho học tự giám sát hai giai đoạn.
- **TabTransformer:** self-attention tạo embedding ngữ cảnh cho đặc trưng hạng mục, rồi MLP; bền hơn với thiếu/nhiễu; cho tiền huấn luyện không giám sát (masked language modelling hoặc replace token detection).
- **ARM-Net:** mô hình quan hệ thích ứng; tương tác đặc trưng bậc tùy ý với attention thưa trong không gian mũ.
- **SAINT:** self-attention + **attention giữa các mẫu (inter-sample)**, giống láng giềng gần nhất, tăng độ bền với thiếu/nhiễu; tiền huấn luyện tương phản; dùng CutMix và mixup.
- **Non-Parametric Transformer (NPT):** dùng cả tập dữ liệu một lúc, attention giữa các điểm dữ liệu (lưu ý: **đề tài đã loại NPT khỏi danh sách đọc** ở các bước trước; ở đây chỉ nêu vì bài khảo sát nhắc tên).

### 4.3. Mô hình điều chuẩn
- **RLN** (Shavitt & Segal): hệ số điều chuẩn có thể học cho từng trọng số, bằng "Counterfactual Loss"; mạng rất thưa, cho tầm quan trọng đặc trưng; ngang GBDT nhưng đánh giá trên dữ liệu chủ yếu số, không bàn hạng mục.
- **Regularization cocktail** (Kadra et al.): kết hợp 13 kỹ thuật điều chuẩn; MLP được điều chuẩn tốt có thể vượt kiến trúc cây; nhược điểm: tối ưu siêu tham số và điều chuẩn theo từng bộ **tốn thời gian hơn GBDT nhiều**.

## 5. Sinh dữ liệu bảng (mục V)

- **Mục đích:** tăng dữ liệu, điền giá trị thiếu, **cân bằng lớp**, bảo vệ quyền riêng tư.
- Khó vì hỗn hợp liên tục/rời rạc và phân phối khác nhau. Cổ điển: Copulas, Bayesian networks (Chow-Liu). Học sâu: **GAN** (vanilla, WGAN, WGAN-GP, Cramér GAN, Boundary-seeking GAN, VeeGAN) và **VAE**.
- Một số mô hình (Bảng III): **medGAN** (hồ sơ y tế, toàn bộ đặc trưng rời rạc), **TableGAN** (DCGAN, chuyển bản ghi thành ma trận; không rõ thiên lệch ảnh có phù hợp), **CTGAN** và **TVAE** (Xu et al.): chuẩn hóa theo mode (mixture component) cho phân phối đa đỉnh/lệch, conditional GAN, Gumbel-softmax cho hạng mục; TVAE vượt GAN của họ; **PATE-GAN** (riêng tư vi phân), actGAN, VAEM, OVAE, ITS-GAN, TAEI (AE+SMOTE), Causal-TGAN, Copula-Flow. Fan et al.: khuyên dùng mạng đầy đủ kết nối với mất mát GAN thường kèm điều chỉnh chống mode-collapse và chuẩn hóa của Xu et al.
- **Đánh giá chất lượng:** (i) "machine learning efficacy" (huấn luyện trên dữ liệu thật vs sinh rồi kiểm tra trên dữ liệu thật); (ii) quan sát phân phối từng đặc trưng (CDF, tán xạ); (iii) kiểm định Kolmogorov-Smirnov; (iv) log-likelihood trên tập kiểm tra (chống quá khớp); (v) **Distance to Closest Record (DCR)** để phát hiện ghi nhớ mẫu. Không đủ chỉ một thước đo: mô hình ghi nhớ sẽ đạt efficacy cao nhưng thất bại DCR.

## 6. Giải thích mô hình (mục VI)

- **Feature highlighting:** gán điểm quan trọng đặc trưng cho từng dự đoán: mô hình tuyến tính cục bộ theo thiết kế; truyền lan độ liên quan và gradient (khi biết tham số); mô hình thay thế agnostic (LIME, **KernelSHAP**).
- **Counterfactual:** thay đổi nhỏ nhất của vector đặc trưng để đảo kết quả (recourse). Ba hướng: giả định đặc trưng độc lập (bộ giải tổ hợp, tối ưu theo gradient với ràng buộc khả thi và đa dạng), mô hình sinh (VAE) để giữ phụ thuộc đặc trưng, và recourse bền với dịch chuyển mô hình/nhiễu. Xử lý biến hạng mục cardinality cao là vấn đề mở.

## 7. Thực nghiệm (mục VII)

### 7.1. Dữ liệu (Bảng IV)
Chưa có benchmark chuẩn kiểu MNIST/ImageNet cho dữ liệu bảng; khảo sát 18 bài trong Bảng II dùng **hơn 100 bộ dữ liệu khác nhau**, ít trùng nhau. Bài chọn 5 bộ phủ nhiều đặc tính:

| Bộ | Mẫu | Đặc trưng số | Đặc trưng hạng mục | Nhiệm vụ | Số lớp |
|---|---|---|---|---|---|
| HELOC (FICO; dự đoán trả nợ hạn mức vay thế chấp trong 2 năm) | 9.871 | 21 | 2 | Nhị phân | 2 |
| Adult Income | 32.561 | 6 | 8 | Nhị phân | 2 |
| HIGGS (vật lý hạt, mô phỏng Monte Carlo; chỉ số liên tục) | 11 M | 27 | 1 | Nhị phân | 2 |
| Covertype (loại rừng) | 581.012 | 52 | 2 | Đa lớp | 7 |
| California Housing | 20.640 | 8 | 0 | Hồi quy | – |

Với HIGGS, biến thứ 21 được nhị phân hóa thành hạng mục 3 nhóm vì DeepFM, DeepGBM, TabTransformer cần ít nhất một đặc trưng hạng mục.

### 7.2. Thiết lập
- Tối ưu siêu tham số bằng **Optuna, 100 vòng cho mỗi mô hình**, mỗi cấu hình kiểm định chéo **5 fold phân tầng**, cùng cách chia fold cho mọi bộ. Không gian tìm kiếm dựa vào bài gốc và khuyến nghị tác giả khung phần mềm (công bố online).
- Tiền xử lý thống nhất: chuẩn hóa trung bình 0, phương sai 1 cho đặc trưng số; **mã hóa ordinal cho hạng mục** (theo bài [54], chiến lược này ngang các mã hóa nâng cao); giá trị thiếu thay bằng 0 cho mô hình tuyến tính và mạng thuần. Với LightGBM, DeepFM, DeepGBM, TabNet, TabTransformer, SAINT, khai báo rõ đặc trưng hạng mục để dùng chức năng riêng (embedding...).
- Chạy trong Docker, mã công khai.

### 7.3. Kết quả (Bảng V; stratified 5-fold CV; trung bình ± độ lệch chuẩn)

| Mô hình | HELOC Acc | HELOC AUC | Adult Acc | Adult AUC | HIGGS Acc | HIGGS AUC | Covertype Acc | Covertype AUC | Cal. Housing MSE |
|---|---|---|---|---|---|---|---|---|---|
| Linear Model | 73,0 | 80,1 | 82,5 | 85,4 | 64,1 | 68,4 | 72,4 | 92,8 | 0,528 |
| KNN | 72,2 | 79,0 | 83,2 | 87,5 | 62,3 | 67,1 | 70,2 | 90,1 | 0,421 |
| Decision Tree | 80,3 | 89,3 | 85,3 | 89,8 | 71,3 | 78,7 | 79,1 | 95,0 | 0,404 |
| Random Forest | 82,1 | 90,0 | 86,1 | 91,7 | 71,9 | 79,7 | 78,1 | 96,1 | 0,272 |
| XGBoost | 83,5 | 92,2 | 87,3 | 92,8 | 77,6 | 85,9 | 97,3 | 99,9 | 0,206 |
| LightGBM | 83,5 | 92,3 | 87,4 | 92,9 | 77,1 | 85,5 | 93,5 | 99,7 | 0,195 |
| CatBoost | 83,6 | 92,4 | 87,2 | 92,8 | 77,5 | 85,8 | 96,4 | 99,8 | 0,196 |
| Model Trees | 82,6 | 91,5 | 85,0 | 90,4 | 69,8 | 76,7 | – | – | 0,385 |
| MLP | 73,2 | 80,3 | 84,8 | 90,3 | 77,1 | 85,6 | 91,0 | 76,1 | 0,263 |
| DeepFM | 73,6 | 80,4 | 86,1 | 91,7 | 76,9 | 83,4 | – | – | 0,260 |
| DeepGBM | 78,0 | 84,1 | 84,6 | 90,8 | 74,5 | 83,0 | – | – | 0,856 |
| RLN | 73,2 | 80,1 | 81,0 | 75,9 | 71,8 | 79,4 | 77,2 | 92,0 | 0,348 |
| TabNet | 81,0 | 90,0 | 85,4 | 91,1 | 76,5 | 84,9 | 93,1 | 99,4 | 0,346 |
| VIME | 72,7 | 79,2 | 84,8 | 90,5 | 76,9 | 85,5 | 90,9 | 82,9 | 0,275 |
| TabTransformer | 73,3 | 80,1 | 85,2 | 90,6 | 73,8 | 81,9 | 76,5 | 72,9 | 0,451 |
| NODE | 79,8 | 87,5 | 85,6 | 91,1 | 76,9 | 85,4 | 89,9 | 98,7 | 0,276 |
| Net-DNF | 82,6 | 91,5 | 85,7 | 91,3 | 76,6 | 85,1 | 94,2 | 99,1 | – |
| STG | 73,1 | 80,0 | 85,4 | 90,9 | 73,9 | 81,9 | 81,8 | 96,2 | 0,285 |
| NAM | 73,3 | 80,7 | 83,4 | 86,6 | 53,9 | 55,0 | – | – | 0,725 |
| SAINT | 82,1 | 90,7 | 86,1 | 91,6 | **79,8** | **88,3** | 96,3 | 99,8 | 0,226 |

(Độ lệch chuẩn nhỏ, thường ≤ 0,3 điểm; hai chỗ đáng chú ý: RLN trên Adult AUC 75,9 ± 8,2 và MLP trên Covertype AUC 76,1 ± 3,0. Dấu "–": mô hình không áp dụng được cho loại tác vụ đó. Số ghi theo PDF; bảng gốc có thể bị lệch cột khi trích, mình đã kiểm theo hàng.)

Phát hiện chính:
- Trừ HIGGS rất lớn, **điểm tốt nhất vẫn thuộc cây tăng cường**; XGBoost và CatBoost thắng mọi mô hình sâu trên bộ nhỏ và vừa, bộ hồi quy và bộ đa lớp (Covertype: XGBoost 97,3% acc; mô hình sâu tốt nhất là Net-DNF 94,2%, SAINT 96,3%).
- Trên **HIGGS (11 triệu mẫu, chủ yếu đặc trưng liên tục)**, SAINT (79,8% acc, 88,3 AUC) vượt các mô hình cổ điển: gợi ý rằng với bộ rất lớn, kiến trúc nơ-ron hiện đại có thể có lợi thế.
- Trong nhóm mô hình sâu, **SAINT** nhất quán cạnh tranh; các mô hình khác phụ thuộc bộ dữ liệu (DeepFM tốt nhất nhóm sâu trên Adult và nhì trên California Housing nhưng yếu trên HELOC).
- Phù hợp quan sát từ các cuộc thi Kaggle.

### 7.4. Thời gian chạy (Hình 3–4)
Huấn luyện của mô hình dựa trên boosting nhanh hơn đa số mô hình sâu, nhưng **suy luận trên HIGGS (11 M mẫu)** thì XGBoost chậm hơn nhiều: **5995 giây** cho XGBoost so với **10,18 giây** cho MLP và **282 giây** cho SAINT (tất cả chạy trên cùng GPU). Hình 5 so sánh kích thước mô hình (số tham số) trên Adult.

### 7.5. Đánh giá diễn giải (Hình 6, Bảng VI)
- Đánh giá tính trung thực của quy gán đặc trưng (fidelity) bằng loại dần đặc trưng quan trọng nhất (MoRF) hoặc ít quan trọng nhất (LeRF), huấn luyện lại sau mỗi lần; 15 lần chạy trên Adult. Với MoRF, TabNet và TabTransformer (lấy đường chéo của attention) cho kết quả tốt nhất; với LeRF chỉ TabNet tốt hơn đáng kể.
- So với **KernelSHAP** (không có chân lý nhưng được dùng rộng) bằng tương quan hạng Spearman trên 750 mẫu Adult: rất thấp, có khi âm:

| Mô hình, loại attention | Spearman với KernelSHAP |
|---|---|
| TabTransformer, theo cột | −0,01 ± 0,008 |
| TabTransformer, đường chéo | 0,00 ± 0,010 |
| TabNet | 0,07 ± 0,009 |
| SAINT, theo cột | −0,04 ± 0,007 |
| SAINT, đường chéo | 0,01 ± 0,007 |

  Kết luận: các mô hình transformer **không cho quy gán đặc trưng đáng tin ngay từ đầu**; cần benchmark diễn giải kỹ hơn.

## 8. Thảo luận và vấn đề mở (mục VIII–IX)

- **Cây ensemble vẫn là SOTA:** sau hơn 6 năm từ XGBoost và hơn 20 năm từ bài gradient boosting gốc, "hiện trạng tốt nhất cho dữ liệu bảng gần như không đổi". Với bộ rất lớn, học sâu có thể cạnh tranh. Tác giả cho rằng có thể cần **"tái định hướng cơ bản"** của lĩnh vực; thay vì thêm mô hình phức tạp, cần hiểu **lý do** của khoảng cách hiệu năng; kết luận "không" cho câu hỏi dùng học sâu hiện tại có lợi cho dữ liệu bảng, nhất là **dữ liệu nhỏ không đồng nhất phổ biến trong ứng dụng**.
- **Benchmark thống nhất:** không có đồng thuận; lựa chọn bộ dữ liệu, siêu tham số và mã hóa (ví dụ one-hot) có thể đổi kết luận (Shwartz-Ziv & Armon).
- **Xu hướng:** tiền xử lý nhẹ có thể tăng hiệu năng rõ rệt; kiến trúc hướng tới transformer và tiền huấn luyện tự giám sát; điều chuẩn là hướng quan trọng; sinh dữ liệu cần thiết cho dữ liệu nhỏ và riêng tư (và khó ngang suy luận); dữ liệu luồng (concept drift) vẫn ưu tiên cây tăng dần.
- **Vấn đề mở (mục VIII-B):** phân tích lý thuyết thông tin của mã hóa (hầu hết lossy); hiệu quả tính toán của mô hình hybrid; điều chuẩn chuyên biệt (kể cả bất công giữa các lớp do weight decay và tăng cường dữ liệu); quy trình sinh mới (diffusion chưa được áp dụng cho bảng); diễn giải/phản thực tế với dữ liệu hỗn hợp; chuyển giao học sâu sang luồng dữ liệu; **transfer learning cho dữ liệu bảng ("chưa có cách chung hiệu quả")**; tăng cường dữ liệu cho bảng (SMOTE-NC kém, cải thiện trong không gian tiềm ẩn chỉ nhỏ); học tự giám sát chuyên cho bảng.
- **Kết luận chính:** mô hình sâu vẫn kém ensemble cây trên dữ liệu **nhỏ và vừa (dưới khoảng 1 triệu mẫu)**; chỉ SAINT thắng trên HIGGS.

## 9. Hạn chế (tác giả nêu và suy ra)
1. **Chỉ 5 bộ dữ liệu** (3 nhị phân, 1 đa lớp, 1 hồi quy), 3 bộ có thông tin hạng mục rất ít (HIGGS 1, Cal. Housing 0); chỉ 1 bộ rất lớn nên kết luận "học sâu thắng khi dữ liệu lớn" dựa trên một bộ duy nhất. Tác giả thừa nhận lựa chọn khác (bộ dữ liệu, siêu tham số, mã hóa) có thể cho kết quả khác.
2. **Chỉ 100 vòng Optuna** (số vòng ít với mô hình sâu có nhiều siêu tham số; bài 25 dùng 1000 vòng); tiền xử lý thống nhất (ordinal) có thể bất lợi cho một số mô hình sâu.
3. Hồi quy chỉ một bộ; thiếu mô hình trên đa lớp cho vài mô hình (DeepFM, DeepGBM, NAM, Model Trees không áp dụng).
4. Đánh giá diễn giải chỉ trên Adult và chỉ một số mô hình transformer; **KernelSHAP không phải chân lý**.
5. Không đánh giá nhãn nhiễu, mất cân bằng lớp, hiệu chuẩn hay dữ liệu cực nhỏ (< 1k mẫu); dù bài nêu "mất cân bằng lớp" là thách thức, **không có thí nghiệm kiểm tra**.
6. Khảo sát đóng ở 2022, **trước TabPFN** (v1 ra cuối 2022) và trước TFM; không có mô hình tiền huấn luyện cho dữ liệu bảng.
7. Thời gian suy luận XGBoost trên HIGGS (5995 giây) có thể do cách cài đặt/GPU của họ; không nêu chi tiết cấu hình.

## 10. Liên hệ với bài Nature và đề tài
- Cung cấp **bản đồ phân loại** để định vị TabPFN/TabICL: chúng không thuộc ba nhánh (chúng là mô hình tiền huấn luyện trên dữ liệu tổng hợp, dùng học trong ngữ cảnh), cho thấy TFM là hướng mới đáp lại mục "transfer learning cho dữ liệu bảng" mà khảo sát liệt kê là vấn đề mở.
- Bằng chứng nền: **dưới ~1M mẫu cây vẫn thắng**; đúng vùng mà TFM nhắm tới (bài Nature: ≤ 10k mẫu, ≤ 500 đặc trưng). Cần chứng minh TFM vượt cây trên chính bộ này.
- Bài nêu **mất cân bằng lớp và dữ liệu bẩn** là nguyên nhân học sâu kém nhưng không đo: củng cố khoảng trống nghiên cứu (đánh giá TFM với nhãn nhiễu/mất cân bằng) của đề tài.
- Mã TabSurvey là nguồn tham khảo để cài đặt benchmark XGBoost/CatBoost cho thí nghiệm final-term.

## 11. Từ khóa cần nhớ
- 3 nhánh: **data transformation / specialized architectures (hybrid, transformer) / regularization**; nhiệm vụ: suy luận, sinh, giải thích.
- 4 nguyên nhân học sâu kém: dữ liệu chất lượng thấp, thiếu phụ thuộc không gian, phụ thuộc tiền xử lý, tầm quan trọng của từng đặc trưng.
- Benchmark: 5 bộ (HELOC, Adult, HIGGS, Covertype, California Housing), Optuna 100 vòng, CV 5 fold; **SAINT thắng HIGGS (88,3 AUC)**, còn lại cây thắng; XGBoost suy luận HIGGS **5995 s** vs MLP 10,18 s, SAINT 282 s.
- Spearman giữa attention và KernelSHAP ≈ 0 (−0,04 đến 0,07).
