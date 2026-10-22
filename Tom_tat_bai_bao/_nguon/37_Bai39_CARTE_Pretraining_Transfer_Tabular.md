# Bài 39. CARTE: Pretraining and Transfer for Tabular Learning

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2024 (PMLR 235; arXiv:2402.16785v2) |
| Tác giả | Myung Jun Kim, Léo Grinsztajn, Gaël Varoquaux (SODA, Inria Saclay; Probabl.ai) |
| Mã nguồn | github.com/soda-inria/carte |
| Lưu ý về PDF | Bảng 1 (tỉ lệ giảm hiệu năng khi thiếu giá trị) và Bảng 2 (thời gian) bị trích lệch cột, nên mình chỉ ghi kết luận định tính và bậc độ lớn. Hình 4–8 chỉ là đồ thị; số liệu cụ thể lấy từ phần chữ và phụ lục (thứ hạng trung bình của Hình 9). |
| Vị trí trong đề tài | Hướng **tiền huấn luyện từ cơ sở tri thức (YAGO) trên đồ thị**, mạnh ở bảng có **chuỗi ký tự/hạng mục nhiều giá trị**. Cùng nhóm tác giả với Grinsztajn (bài 29); là đối chứng cho TransTab (bài 34), ConTextTab (bài 24), TabSTAR (bài 25). |

## 1. Bài này làm gì trong một đoạn

Học chuyển giao trên bảng gặp trở ngại tích hợp dữ liệu: các bảng khác nhau có cột khác tên, khác thứ tự, và cùng một thực thể có thể viết khác nhau (khớp lược đồ, khớp thực thể). CARTE (Context-Aware Representation of Table Entries) tránh việc này bằng cách **biểu diễn mỗi dòng bảng như một đồ thị hình sao nhỏ**: nút lá là giá trị ô, cạnh là tên cột, và mọi chuỗi (giá trị phân loại, tên cột) được biến thành embedding bởi mô hình ngôn ngữ (FastText) nên có từ vựng mở. Một mạng chú ý đồ thị (graph attention) được **tiền huấn luyện tự giám sát bằng học tương phản trên đồ thị con của YAGO3** (18,1 triệu bộ ba, 6,3 triệu thực thể), rồi tinh chỉnh cho bảng đích. Trên 51 bộ dữ liệu có nhiều chuỗi ký tự (40 hồi quy, 11 phân loại) với số mẫu huấn luyện nhỏ (32–2048), CARTE vượt 42 baseline, kể cả CatBoost và TabPFN, và còn học chung được nhiều bảng chưa khớp cột.

## 2. Bối cảnh

- Học sâu thường thua mô hình dựa trên cây trên bảng (Grinsztajn 2022, Shwartz-Ziv 2021, Gardner 2022); McElfresh 2023 cho rằng NN tốt ở vài loại bảng. Tác giả lập luận học sâu cần **kiến thức nền** để có lợi thế.
- **Chuyển giao trên bảng** trước đây chủ yếu giữa các bảng cùng cột (Somepalli 2021 SAINT, Levin 2023). XTab (Zhu 2023) cho cột khác nhau nhưng chưa vượt CatBoost; TransTab (bài 34) học chéo bảng nhưng trong một chủ đề hẹp (thử nghiệm lâm sàng).
- **TabPFN** (Hollmann 2023) pretrain trên dữ liệu tổng hợp, mạnh ở dữ liệu nhỏ, nhưng không xử lý riêng cột phân loại (nơi cây vốn mạnh). **TabLLM** biểu diễn dòng thành token cho LLM, nhưng LLM kém với số.
- **Mục rời rạc:** cơ sở dữ liệu quan tâm nhiều tới chuỗi ký tự; Cerda & Varoquaux 2022 (mã hóa chuỗi), TableVectorizer của skrub, KEN (embedding thực thể trên đồ thị tri thức, nhưng buộc phải khớp từng ô với thực thể Wikipedia).
- Bài toán tác giả đặt ra: **không cần khớp tường minh** (không khớp lược đồ, không khớp thực thể), chỉ học cấu trúc ngầm để cải thiện tác vụ phía sau.

## 3. Phương pháp

### 3.1. Biểu diễn đồ thị của một dòng (graphlet)
Dòng i của bảng k cột → đồ thị hình sao Gᵢ(X, E) có k − pᵢ nút lá (pᵢ = số cột thiếu giá trị ở dòng đó; **ô thiếu đơn giản bị bỏ**, không cần điền khuyết).
- Nút lá = giá trị ô; cạnh = tên cột. Cả hai được khởi tạo bằng embedding FastText (d = 300) cho chuỗi/hạng mục và tên cột.
- **Nút số:** khởi tạo bằng tích của giá trị số với embedding tên cột, ví dụ X(239) = 239 × E(H index).
- **Nút tâm:** trung bình các nút lá; sau này đóng vai trò nút đọc ra (readout) cho biểu diễn cả dòng.
Lợi ích: (1) giữ ngữ cảnh (ô được hiểu theo tên cột); (2) không cần xử lý chuỗi (khử trùng lặp, mã hóa hạng mục), từ vựng mở nên lỗi chính tả hay biến thể ("North America"/"Northern America") tự được gom gần; (3) các bảng khác lược đồ nằm chung một miền đồ thị nên không cần khớp cột/thực thể.

### 3.2. Tiền huấn luyện từ YAGO3
- **Graphlet:** với mỗi thực thể, lấy đồ thị con 2-hop (tối đa 100 quan hệ 1-hop và 10 quan hệ 2-hop). Nút tâm là một token, kèm một nút "has name" chứa tên thực thể, để giống cấu trúc đồ thị từ bảng.
- **Lấy mẫu batch:** 90% thực thể có ≥ 6 quan hệ 1-hop (trung vị khoảng 15, gần với bảng nhiều cột), 10% còn lại từ phần còn lại. Mẫu dương = **bản cắt** của graphlet gốc (xóa ngẫu nhiên 30–70% cạnh).
- **Kiến trúc:** Transformer encoder biến thành mạng chú ý đồ thị. Query chỉ dùng nút: Q = X·W_Q. Key và Value dùng tích phần tử (nút ⊙ cạnh): K = (X ⊙ E)·W_K, V = (X ⊙ E)·W_V. Ý tưởng tích phần tử cho quan hệ (tên cột) lấy từ embedding đồ thị tri thức (Balazevic 2019; KEN). Điểm chú ý A_ij = softmax_j(Q_i·K_ij/√d) chỉ trên các nút kề (mặt nạ theo cấu trúc đồ thị). Cập nhật nút: X' = σ(ΣⱼAᵢⱼVᵢⱼ); cạnh cũng được cập nhật qua E' = σ(E·W_E). Tầng cuối là chú ý không cập nhật cạnh, rồi lấy nút tâm (Aggregation & Readout) và một lớp tuyến tính.
- **Mất mát:** tương phản theo Chen 2020 (SimCLR): graphlet gốc và bản cắt là cặp dương, các graphlet khác trong batch là âm; cosine + InfoNCE.
- **Thông số:** 12 lớp attention × 12 head, ẩn 300, ~9,3 triệu tham số; batch 256 (128 thực thể + 128 mẫu dương); 1.000.000 bước (~40 epoch trên YAGO); AdamW, lịch cosine lr từ 5×10⁻⁶ đến 1×10⁻⁴, warmup 10.000 bước, dropout 0,1, GELU; huấn luyện trên GPU V100 32GB.

### 3.3. Tinh chỉnh cho tác vụ phía sau
- Chỉ **tái sử dụng một phần**: các lớp khởi tạo nút/cạnh và lớp Aggregation & Readout; dùng **1 lớp attention** (bằng số hop tối đa k = 1) vì bảng đích đơn giản hơn YAGO (hình sao, ít biến thiên cấu trúc), mạng sâu bị **over-smoothing** (Phụ lục C.4, Hình 11: biểu diễn lấy từ các lớp 2, 4, 8, 12 càng sâu càng kém hữu ích). Thêm các lớp tuyến tính làm bộ phân loại/hồi quy.
- **Bảng đơn:** biến đổi lũy thừa Yeo–Johnson cho biến số; **bagging** (Breiman 1996) nhiều mô hình với các lần chia train/validation khác nhau cho early stopping, trung bình đầu ra.
- **Chuyển giao một bảng nguồn → bảng đích:** tinh chỉnh chung trên cả hai bảng, không cần khớp cột nhưng cần nhãn tương tự; nhãn nguồn được biến đổi (Yeo–Johnson ngược) để khớp moment bậc một của nhãn đích; khác loại tác vụ thì nhị phân hóa nhãn hồi quy hoặc chuẩn hóa nhãn nhị phân. Batch 64, trong đó 8 mẫu từ bảng đích; early stopping trên validation của đích (thường dừng trước khi đi hết dữ liệu nguồn nên ít quá khớp nguồn). Hyperparameter lấy từ cấu hình bảng đơn.
- **Học chung nhiều bảng:** huấn luyện từng cặp (nguồn_i, đích) và một mô hình chỉ bảng đích, rồi **ensemble bằng softmax** trên điểm validation chia cho độ lệch chuẩn (nhiệt độ); nếu mọi nguồn đều tốt như nhau thì trọng số đều, nếu một nguồn trội thì dự đoán neo vào nguồn đó.

## 4. Thiết lập thí nghiệm

- **Dữ liệu:** 51 bộ (40 hồi quy, 11 phân loại) lấy từ các nghiên cứu ML trước và Kaggle, chủ đề đời sống (tai nạn, bầu cử, lương, thực phẩm, nhà hàng, giá xe, rượu vang, phim…). Chọn bảng có **cột có nghĩa và nhiều giá trị chuỗi rời rạc**, khác nhiều bộ UCI. Tiền xử lý tối thiểu: bỏ cột chỉ có một giá trị hoặc thiếu hơn nửa số dòng.
- **So với bộ của TabLLM (Bảng 3):** bộ của bài có phần cột số 0,194 (TabLLM: 0,613); phần cột có |C| > 10 là 0,625 (TabLLM: 0,043); tỉ số cardinality/kích thước dữ liệu 0,263 (TabLLM: 0,001), tức dữ liệu của bài có hạng mục nhiều giá trị hơn hẳn.
- **Kích thước tập huấn luyện:** 32, 64, 128, 256, 512, 1024, 2048 (phần còn lại là test); 10 lần chia train/test; R² cho hồi quy, AUROC cho phân loại. Chuẩn hóa điểm: kiểu Grinsztajn (cận dưới ρ = 0 hồi quy, 0,5 phân loại), và biểu đồ khác biệt tới hạn (Friedman + hậu kiểm Conover).
- **Baseline (42 tổ hợp):** CatBoost (text là categorical), TabVec (TableVectorizer của skrub: one-hot cho |C| thấp, Gamma-Poisson cho |C| cao) + XGB/HGB/RF, MLP, ResNet, Ridge/Logistic, S-LLM (mỗi dòng thành câu, mã hóa bằng e5-small-v2 rồi XGBoost; hai cách xử lý số: CN nối ngoài LLM, EN nhúng như chuỗi), TabPFN (text mã hóa target encoding), kèm các biến thể bagging, target encoding, FastText/LLM.
- **Tinh chỉnh siêu tham số:** baseline dùng tìm ngẫu nhiên 100 lần với CV 5-fold (không gian lấy từ Grinsztajn 2022 và bài CatBoost); **CARTE chỉ tìm lưới trên learning rate** ([2,5; 5; 7,5] × 10^−k); TabPFN dùng mặc định.
- **Nhiều bảng:** nhóm các bảng cùng chủ đề (giá rượu, điểm rượu, bia, xe cũ, phim, drama, anime, đồ em bé, xe máy, lương nhân viên, điểm nhà hàng, điểm tạp chí…), mỗi nhóm 2–5 bảng; số mẫu đích 32–256; tổng 245–275 trường hợp.

## 5. Kết quả

### 5.1. Học trên một bảng (Hình 4, 9)
- CARTE **vượt đều các baseline ở mọi kích thước tập huấn luyện**, cả hồi quy lẫn phân loại, theo điểm chuẩn hóa và theo biểu đồ khác biệt tới hạn.
- Thứ hạng trung bình trong bảng 42 baseline (hồi quy): **CARTE 4,33**; CatBoost 14,68; S-LLM-CN-XGB 15,17; TabVec-LLM-XGB 15,45; TabVec-FT-XGB 15,57; … cách biệt lớn.
- **Bagging** (chia train/validation khác nhau) cũng có lợi cho các mạng nơ-ron khác (MLP/ResNet), ít tác dụng với mô hình tuyến tính, cây hay TabPFN; gợi ý đây là mẹo chung cho học sâu khi dữ liệu nhỏ.
- **Bền với giá trị thiếu (Bảng 1):** bỏ ngẫu nhiên 10% và 30% đặc trưng (cả train và test) ở 64 và 512 mẫu: CARTE giảm hiệu năng ít hơn CatBoost, TabVec-XGB, TabVec-RF (ý chính của tác giả; số liệu cụ thể của bảng bị lệch cột trong PDF).
- **Chi phí tính toán (Bảng 2):** CARTE chậm hơn nhiều, khoảng hàng trăm giây ở mức tập huấn luyện 512 trong khi CatBoost cỡ vài giây; khoảng cách tăng theo kích thước dữ liệu. Tác giả thừa nhận cần tối ưu thêm.

### 5.2. Không cần khớp thực thể (Hình 5, Bảng 6)
Giả thuyết giải thích hiệu năng là CARTE đã học thông tin về nhiều thực thể từ YAGO. Kiểm tra trên 4 bộ (company employees, movies, US accidents, US elections) với tỉ lệ thực thể khớp YAGO lần lượt 32%, 84%, 67%, 98%, thử khớp thủ công và dùng tên gốc: **KEN cần khớp thực thể mới có lợi, còn CARTE bền với biến thể tên** (khớp thủ công hay không đều tương tự, kể cả khi thực thể không có trong YAGO); ablation (Hình 10) cho thấy biểu diễn mức chuỗi có ngữ nghĩa là then chốt.

### 5.3. So với TabLLM (Hình 6, Bảng 5)
Trên 9 bộ của TabLLM (chủ yếu cột số hoặc hạng mục ít giá trị): **TabPFN tốt nhất**, sau đó CARTE, XGBoost, TabLLM. CARTE vẫn cạnh tranh, riêng bộ "bank" (có số lẫn hạng mục nhiều giá trị) thì CARTE thắng. Tác giả mô tả CARTE nằm giữa TabPFN (số) và TabLLM (token chuỗi).

### 5.4. Học qua nhiều bảng
- **Một bảng nguồn (Hình 7):** mọi phương pháp đều hưởng lợi từ chuyển giao so với chỉ bảng đích, nhưng **chỉ CARTE cải thiện đều đặn mà không cần khớp cột thủ công**; CatBoost cần khớp cột (đường chấm) mới có lợi; S-LLM lợi ích giảm nhanh khi tăng số mẫu.
- **Nhiều bảng nguồn (Hình 8):** thêm tối đa 3 nguồn (1 đích + 3 nguồn), trung vị cải thiện và cả cận dưới biến thiên đều tốt lên (nhiều nguồn thì dễ có nguồn hữu ích hơn, nên tình huống xấu nhất cũng tốt hơn).
- **Phụ lục C.6:** CARTE có và không khớp lược đồ cho kết quả tương đương trên 275 trường hợp (kiểm định t hai phía p = 0,728): không cần khớp lược đồ.

### 5.5. Ablation (Phụ lục C.3–C.4)
Bỏ hoặc thay từng thành phần (khởi tạo bằng MinHash của skrub thay cho FastText, bỏ attention…) đều làm giảm hiệu năng, nên mỗi thành phần đều quan trọng. Mạng sâu hơn bị over-smoothing nên dùng 1 lớp khi tinh chỉnh.

## 6. Hạn chế

Tác giả nêu:
- Chi phí tính toán lớn hơn nhiều baseline, cần nghiên cứu/tối ưu thêm; kiến trúc cần cải thiện cho tập huấn luyện lớn.
- Chưa tinh chỉnh số với kỹ thuật embedding số (Gorishniy 2022), chưa tận dụng đủ thông tin huấn luyện hay attention giàu hơn như LLM; chưa kết hợp với ý tưởng meta-learning của TabPFN (mạnh với bảng thuần số).
- Về tác động: cảnh báo chi phí năng lượng của cuộc đua mô hình lớn.

Nhận xét thêm của mình:
1. **Kết luận phụ thuộc vào cách chọn benchmark:** 51 bộ được chọn vì có nhiều chuỗi ký tự; với bảng thuần số hoặc hạng mục ít (kiểu TabLLM) TabPFN vẫn tốt hơn. Nghĩa là CARTE bổ sung chứ không thay TabPFN.
2. **Lợi thế một phần đến từ kiến thức nền của YAGO/FastText** (tên người, địa danh, sản phẩm), chứ không từ khả năng "học từ bảng" thuần túy; cần bảng có thực thể thật trong đời sống.
3. Tập huấn luyện nhỏ (≤ 2048) là điểm mạnh nhưng cũng là giới hạn của kết luận; chưa biết CARTE ra sao khi dữ liệu lớn.
4. So sánh không hoàn toàn đồng đều: baseline tìm siêu tham số 100 lần, CARTE chỉ tìm lr (có lợi cho CARTE về công sức tuning nhưng pretrain đã tốn lớn); bagging cho CARTE nhưng các biến thể bagging của baseline cũng được thử, nên kết quả khá đáng tin.
5. Không đánh giá nhiễu nhãn hay mất cân bằng; chỉ có thí nghiệm giá trị thiếu.
6. Về sau Grinsztajn và nhóm đã nghiên cứu hướng này thành TabLib/"skrub", và ConTextTab/TabSTAR dùng LLM embedding thay cho FastText+YAGO (bài 24, 25).

## 7. Liên hệ với bài Nature và đề tài

- **Bổ sung cho TabPFN v2:** bài Nature cho thấy TabPFN v2 mạnh ở dữ liệu số hoặc dữ liệu hạng mục ít; CARTE chỉ ra điểm yếu còn lại là **cột chuỗi/hạng mục nhiều giá trị**, mà mô hình ngôn ngữ giúp được. Đây là một lý do cho các TFM mới có thêm embedding ngữ nghĩa (ConTextTab, TabSTAR, bài 11 mục D?).
- **Bagging** như một cách giảm phương sai ở dữ liệu nhỏ: liên hệ với TabM (bài 32) và BETA (bài 15); gợi ý cho câu hỏi cuối kỳ "ensemble có bền với nhiễu nhãn không".
- Bằng chứng thêm rằng "mô hình pretrain + ít dữ liệu" thắng cây ở dữ liệu nhỏ, giống kết luận bài Nature.

## 8. Từ khóa cần nhớ

- **Graphlet:** dòng bảng thành đồ thị sao; nút = giá trị ô, cạnh = tên cột; embedding FastText.
- **Pretrain:** YAGO3 (18,1 triệu bộ ba, 6,3 triệu thực thể), tương phản với bản cắt; 12 lớp, 9,3 triệu tham số, 1 triệu bước.
- **Fine-tune:** 1 lớp attention (tránh over-smoothing), Yeo–Johnson, bagging, chỉ tìm learning rate.
- **Kết quả:** vượt 42 baseline trên 51 bộ (thứ hạng hồi quy 4,33 so với 14,68 của CatBoost); không cần khớp thực thể hay lược đồ; TabPFN vẫn tốt hơn ở bảng thuần số.
- **Hạn chế:** chậm; lợi thế gắn với dữ liệu có chuỗi ký tự.
