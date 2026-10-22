# Bài 40. Making Pre-trained Language Models Great on Tabular Prediction (TP-BERTa)

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICLR 2024 (arXiv:2403.01841v2) |
| Tác giả | Jiahuan Yan, Bo Zheng, Hongxia Xu, Yiheng Zhu, Danny Z. Chen, Jimeng Sun, Jian Wu, Jintai Chen (Zhejiang Univ., Notre Dame, UIUC) |
| Mã nguồn | github.com/jyansir/tp-berta |
| Lưu ý về PDF | Bảng 1 (thứ hạng) bị lệch nhãn dòng khi trích chữ; mình giải mã lại theo thứ tự cột và đối chiếu với phần chữ ("Ours_s tốt hơn Ours_j", "TP-BERTa kém khi α = 0"). Bảng 14 (mất cân bằng) bị rời cột, nên mình ghi kết luận định tính cùng vài số đọc chắc. Bảng 6–10 (danh sách dữ liệu chi tiết) không chép lại. |
| Vị trí trong đề tài | Ý tưởng **"mô hình ngôn ngữ làm nền tảng cho bảng"**: dùng RoBERTa hiểu tên cột và giá trị, tiền huấn luyện trên 202 bảng thật. Cùng nhóm hướng ngữ nghĩa với TransTab (bài 34), CARTE (bài 39), ConTextTab (bài 24), TabSTAR (bài 25). Có một thí nghiệm phụ về **mất cân bằng nhãn** (Phụ lục E), liên quan trực tiếp tới khoảng trống nghiên cứu. |

## 1. Bài này làm gì trong một đoạn

Mô hình ngôn ngữ (LM) hiểu tên cột và giá trị phân loại rất tốt nhờ tiền huấn luyện trên kho văn bản khổng lồ, nhưng gặp hai rào cản với bảng: (1) không nhạy với **giá trị số liên tục** (token hóa chuỗi chữ số làm mất ý nghĩa độ lớn), và (2) biến bảng thành một câu dài khiến LM phải tự ghép đúng cặp "tên-giá trị" và phải chịu hoán vị cột. TP-BERTa (Tabular Prediction adapted BERT) dựa trên RoBERTa giải hai vấn đề: **Relative Magnitude Tokenization (RMT)** rời rạc hóa số thành "token độ lớn" học được bằng cây C4.5 theo nhãn, và **Intra-Feature Attention (IFA)** hợp nhất từng cặp (tên, giá trị) thành một vector trước khi vào LM. Tiền huấn luyện giám sát trên 202 bảng (101 phân loại nhị phân và 101 hồi quy, ~10 triệu mẫu), kiểm tra trên 145 bảng nhỏ (80 nhị phân, 65 hồi quy): **vượt mọi mạng sâu cho bảng và cạnh tranh với GBDT**, dù chỉ dùng siêu tham số mặc định, nhưng **yếu ở bảng toàn số**.

## 2. Bối cảnh

- GBDT vẫn thường thắng DNN trên bảng cỡ vừa và ít chiều (Grinsztajn 2022). Trong thị giác và ngôn ngữ, tính chuyển giao của DNN giúp giảm nhu cầu dữ liệu, nhưng trên bảng thì chưa khai thác được do **tính không đồng nhất giữa các bảng** (cột khác nhau).
- **Công trình liên quan:** TransTab (bài 34) chỉ hợp cho bảng lâm sàng có tên cột chung; XTab (Zhu 2023) dùng bộ mã hóa riêng từng bộ dữ liệu, nên không chuyển giao hết kiến thức. Với LM: GReaT (Borisov 2023) và TapTap (Zhang 2023) tinh chỉnh GPT-2 trên câu mẫu, coi số như chuỗi nên kém nhạy với số (Qian 2023). CT-BERT (Ye 2024, cùng thời) dùng kỹ thuật như TransTab (nhân giá trị với embedding tên cột).
- **Ba vấn đề của LM trên bảng, IFA giải quyết:** (1) mục tiêu không phụ thuộc hoán vị cột, nhưng LM có mã hóa vị trí; (2) khi nhồi toàn bộ "[Gender] is female, [Blood Pressure] is 123.8" vào LM, attention đầy đủ dễ tạo tín hiệu ghép sai tên-giá trị; (3) chuỗi dài làm tốn tính toán khi số cột lớn. Đã có đề xuất cho (1) (tăng cường hoán vị hoặc bỏ mã hóa vị trí) nhưng chưa ai xử lý (2), (3).

## 3. Phương pháp

### 3.1. Relative Magnitude Tokenization (RMT)
- **Rời rạc hóa theo nhãn:** với mỗi cột số, áp thuật toán rời rạc hóa C4.5 (Kohavi & Sahami 1996) lên tập huấn luyện, nghĩa là dựng một cây quyết định theo nhãn trên cột đó; biên các lá chia miền giá trị thành các bin (tương tự binning có mục tiêu của Gorishniy 2022 – bài 35). Mỗi giá trị được gán chỉ số bin k (e_k ≤ x < e_{k+1}). Tối đa **n_bin = 256** bin.
- **Token độ lớn:** thêm 256 từ mới vào từ vựng RoBERTa (embedding khởi tạo ngẫu nhiên, **dùng chung cho mọi cột và mọi bộ dữ liệu**). Embedding cuối của một giá trị = embedding token của bin × chính giá trị: RMT(x) = E_k^extra × x (vẫn giữ giá trị chính xác trong bin).
- **Loss triplet nhận biết độ lớn (Eq. 3–5):** muốn các embedding bin gần nhau khi chỉ số gần nhau. Với ba bin k₁, k₂, k₃ thỏa |k₁−k₂| < |k₁−k₃|, loss = max(d(f(k₁), f(k₂)) − d(f(k₁), f(k₃)) + m, 0), f(k) = LayerNorm(Linear(E_k)), khoảng cách L2, biên m = (|k₁−k₃| − |k₁−k₂|)/n_bin.
- **Biến phân loại/nhị phân/chuỗi:** được chuyển thành văn bản có nghĩa (giá trị "0" của "gender" → "male") rồi xử lý như tên cột. Mỗi đặc trưng i có embedding là phép nối [E_name ⊗ E_value] (độ dài l₁ + l₂; l₂ = 1 với số).

### 3.2. Intra-Feature Attention (IFA)
Một **khối multi-head self-attention duy nhất, dùng chung cho mọi đặc trưng và mọi bộ dữ liệu**, nhận embedding của một cặp (tên, giá trị) cộng token [CLS] và nén thành **một vector** ĥ(i) = đầu ra tại vị trí [CLS]. Chi tiết:
- Mã hóa vị trí: [CLS] và token độ lớn nhận vị trí 0; các từ của tên cột nhận vị trí 1…l₁, giữ thứ tự chữ trong tên cột.
- **Bỏ mã hóa vị trí ở vector value** (Eq. 7, V không cộng P): để tránh tín hiệu hằng số làm méo embedding độ lớn khởi tạo ngẫu nhiên và được điều chuẩn cẩn thận.
- Hệ quả: thông tin từ cặp (tên, giá trị) của đặc trưng khác **không thể làm nhiễu** biểu diễn của đặc trưng i; chuỗi vào LM có độ dài bằng số đặc trưng (không phải tổng số từ), nên nhanh hơn.

### 3.3. Mô hình tổng thể
Chuỗi vào RoBERTa: X = [e_CLS, ĥ₁, …, ĥₙ] ∈ ℝ^{(1+n)×d}, **không mã hóa vị trí** giữa các đặc trưng → dự đoán không phụ thuộc thứ tự cột. Dự đoán từ [CLS] của RoBERTa-Encoder qua đầu dự đoán riêng cho từng bộ dữ liệu: Dropout(Linear₁(Tanh(Linear₂(x)))). Loss tiền huấn luyện L = L_sup + λ·L_reg với λ = 0,1 (BCE cho nhị phân, MSE cho hồi quy; L_reg là triplet ở trên). Khi tinh chỉnh chỉ dùng L_sup (loss triplet đã hội tụ khi tiền huấn luyện). **Bỏ lớp đa lớp** khỏi nghiên cứu chính (quy về nhiều bài toán nhị phân, theo Grinsztajn).

## 4. Thiết lập thí nghiệm

- **Dữ liệu:** lấy từ cơ sở dữ liệu ngữ nghĩa lớn **TabPertNet** (Ye 2024). Bảng có ≥ 10.000 mẫu và ≤ 32 đặc trưng dùng để **tiền huấn luyện**; bảng < 10.000 mẫu dùng làm **tác vụ phía sau** (chế độ "bảng điển hình": cỡ vừa, ít chiều, như Grinsztajn). Loại bỏ trùng lặp, và loại thủ công bảng có tên cột vô nghĩa ("v1, x1") hay hạng mục không ánh xạ được.
  - Tiền huấn luyện: 101 nhị phân + 101 hồi quy, ~10 triệu mẫu.
  - Phía sau: 80 nhị phân + 65 hồi quy = 145 bảng.
- **Tiền huấn luyện:** khởi tạo từ RoBERTa-base; 30 epoch, khởi động tuyến tính 6% bước rồi giảm tuyến tính về 0, batch tổng 512 trên 4 GPU A100 40GB; mỗi bước lấy một batch từ một bộ dữ liệu ngẫu nhiên; 5% dữ liệu huấn luyện làm validation, chọn checkpoint theo loss validation trung bình trên mọi bộ. Ba phiên bản: chỉ nhị phân (~72 giờ), chỉ hồi quy (~98 giờ), cả hai (~143 giờ).
- **Tinh chỉnh:** chia (64%, 16%, 20%) train/val/test, giữ phân phối nhãn khi nhị phân; **siêu tham số cố định** (lr 1e-5, không weight decay, batch 64) cho mọi bộ dữ liệu, không tuning.
- **Baseline:** XGBoost, CatBoost (mặc định (d) và tinh chỉnh (t)); MLP, AutoInt, DCNv2, TabNet, SAINT, FT-Transformer (cả (d) và (t)); mô hình chéo bảng TransTab (mặc định, không tiền huấn luyện chéo bảng vì cần bảng chồng cột) và XTab (checkpoint tốt nhất trên validation). Tuning bằng Optuna theo Gorishniy 2021.
- **Chỉ số:** AUC (nhị phân), RMSE (hồi quy), báo cáo bằng **thứ hạng trung bình** trên các bộ dữ liệu.
- **Đặc trưng nhóm bảng:** α = (số cột phân loại)/(số cột số); β = (tổng Shapley cột phân loại)/(tổng Shapley cột số) (Shapley từ XGBoost mặc định), để chia bảng theo kiểu đặc trưng chủ đạo.

## 5. Kết quả

### 5.1. So sánh tổng thể (Bảng 1, thứ hạng trung bình; thấp hơn là tốt hơn)

| Mô hình | Nhị phân (80 bộ) | Hồi quy (65 bộ) |
|---|---|---|
| **TP-BERTa (d), tiền huấn luyện đơn tác vụ (Ours_s)** | **5,8** | **4,3** |
| TP-BERTa (d), tiền huấn luyện chung (Ours_j) | 8,4 | 6,9 |
| CatBoost (t) | 5,9 | 5,5 |
| XGBoost (t) | 6,2 | 4,5 |
| CatBoost (d) | 6,7 | 5,5 |
| FT-Transformer (d) | 7,1 | 7,8 |
| XGBoost (d) | 7,7 | 7,7 |
| SAINT (t) | 8,2 | 7,6 |
| AutoInt (t) | 8,0 | 8,3 |
| DCNv2 (t) | 7,9 | 8,4 |
| MLP (t) | 8,6 | 8,5 |
| XTab (t) | 9,8 | 12,4 |
| TransTab (d) | 11,0 | 12,1 |
| TabNet (t) | 12,1 | 12,6 |

(Một số hàng khác như FT-Transformer (t) nằm giữa; các giá trị trên đọc lại từ bảng 1 của bài và đối chiếu với phần chữ.)

- Xu hướng giống Grinsztajn: GBDT vẫn vượt DNN cổ điển và nâng cao trong chế độ này; nhưng **TP-BERTa (tiền huấn luyện đơn tác vụ, không tuning) đứng đầu hoặc ngang đầu**, vượt XTab và TransTab khá xa.
- Giải thích của tác giả: LM tiền huấn luyện khai thác ngữ nghĩa tên cột và giá trị, chuyển được mẫu "male & female" sang "boy & girl" mà không cần cột chồng nhau hay bộ mã hóa riêng.
- **Tiền huấn luyện chung (Ours_j) luôn kém** các bản đơn tác vụ: có thể do bản chất phân loại và hồi quy không tương thích, chiến lược đầu dự đoán riêng không phù hợp, hoặc cần LM nền mạnh hơn.

### 5.2. Theo kiểu đặc trưng (Bảng 1, Hình 2)
- **Khi cột phân loại chiếm ưu thế (α hoặc β lớn), TP-BERTa càng tốt:** ví dụ nhị phân α ≥ 1: 4,4 so với CatBoost (d) 7,4, CatBoost (t) 7,1, XGBoost (t) 6,5; hồi quy α ≥ 1: 3,9 so với XGBoost (t) 3,3 (XGBoost (t) tốt hơn ở nhóm này), CatBoost (d) 5,5.
- **Khi toàn bộ đặc trưng là số (α = 0): TP-BERTa kém** (nhị phân 7,5; so với CatBoost (t) 4,9, CatBoost (d) 6,0): mất độ chính xác khi biểu diễn số bằng LM; GBDT hoặc FT-Transformer là lựa chọn tốt hơn, nhất là phân loại.
- Quan sát thêm: XGBoost mặc định phụ thuộc mạnh vào tuning nên không ổn định; CatBoost dùng mặc định vẫn thường là lựa chọn tốt, nhất là khi cột phân loại chiếm đa số.
- **Theo kích thước dữ liệu (Hình 5):** không thấy xu hướng đặc biệt; lựa chọn chủ yếu do phân phối kiểu đặc trưng quyết định.

### 5.3. Vì sao LM bị bỏ qua: so sánh mã hóa số và ablation IFA (Bảng 2, 80 bộ nhị phân, dùng TP-BERTa không tiền huấn luyện, khởi tạo từ RoBERTa)
Chỉ tính các bộ dữ liệu có AUC thay đổi > 0,5% (loại bớt dao động theo seed).

| Thay thế | |Δ| ≤ 0,5% | Δ < −0,5% | Δ > 0,5% | Sai khác trung bình |
|---|---|---|---|---|
| Value2Str (GReaT: số như chuỗi) | 16 | 54 | 10 | −12,45% |
| VMFE (nhân giá trị với embedding tên cột, như CT-BERT/FT-T/TransTab) | 34 | 36 | 10 | −3,44% |
| Bỏ IFA (đưa cả tên và giá trị vào LM như trước) | 14 | 52 | 14 | −4,17% |

- RMT tốt hơn cả hai cách mã hóa số trước đó; còn 10 bộ lại tốt hơn khi thay RMT (có lẽ do embedding độ lớn khởi tạo ngẫu nhiên, không đủ dữ liệu để học chính xác khi tinh chỉnh trực tiếp).
- Bỏ IFA làm giảm AUC ở 52 bộ và tăng thời gian huấn luyện **1,32 lần**: LM bối rối khi xử lý một đống cặp tên-giá trị không ghép nối.

### 5.4. Hiệu quả của tiền huấn luyện (Bảng 3)
So TP-BERTa tiền huấn luyện đơn tác vụ với (a) khởi tạo ngẫu nhiên và (b) khởi tạo từ RoBERTa không tiền huấn luyện trên bảng:

| Đối chứng | |Δ| ≤ 0,5% | Δ < −0,5% | Δ > 0,5% | Δ < −3% | Δ > 3% | Sai khác TB |
|---|---|---|---|---|---|---|
| Ngẫu nhiên | 29 | 41 | 10 | 26 | 5 | −3,16% |
| RoBERTa | 26 | 35 | 19 | 21 | 6 | −2,79% |

Tiền huấn luyện trên bảng cho thêm khoảng 2,8–3,2% AUC trung bình trên các bộ thay đổi đáng kể; trọng số RoBERTa tốt hơn ngẫu nhiên (đã mang tri thức ngữ nghĩa).

### 5.5. Thiết kế khác (Bảng 5, Hình 3–4)
- **Số token độ lớn:** n_bin = 128 làm AUC giảm trung bình 2,06%, n_bin = 32 giảm 3,59%; một số bảng lại thích n_bin nhỏ (có thể do biểu diễn quá mức). Gợi ý: tiền huấn luyện với n_bin lớn và tìm n_bin riêng cho từng bộ phía sau.
- **Loss triplet:** giúp hội tụ nhanh và giảm quá khớp (Hình 3); chỉ dùng khi tiền huấn luyện.
- **Thêm mã hóa vị trí vào value vector:** giảm trung bình 2,35%, nên bỏ như thiết kế.
- **Trực quan hóa (Hình 4, t-SNE):** 256 token độ lớn ban đầu rải ngẫu nhiên, sau tiền huấn luyện nằm trên một đa tạp đều đặn, token gần chỉ số thì gần nhau, tức mô hình thực sự học được thứ tự độ lớn.

### 5.6. Các tình huống khác (Phụ lục E)
- **Nhãn mất cân bằng (Bảng 14, dùng bản tiền huấn luyện nhị phân):** lọc các bộ có tỉ lệ lớp thiểu số p < 1/3 (32 bộ), < 1/5 (18 bộ), < 1/8 (12 bộ), < 1/20 (4 bộ). TP-BERTa tốt hơn baseline khi mất cân bằng vừa phải (thứ hạng 6,2 ở p < 1/3 và 5,2 ở p < 1/5, so với 6,4–6,6 của CatBoost/XGBoost tinh chỉnh và FT-T tinh chỉnh). **Ở mức cực đoan (p < 1/20, chỉ 4 bộ), GBDT thắng áp đảo** (XGBoost tinh chỉnh hạng 3,1), TP-BERTa (6,3) vẫn vượt phần lớn DNN. Đây là thử nghiệm duy nhất trong các bài đã đọc khảo sát mất cân bằng trên mô hình nền tảng/pretrain, nhưng **chỉ dựa trên thứ hạng và lọc tập dữ liệu có sẵn**, không chủ động tạo mất cân bằng hay đo hiệu chỉnh.
- **Đa lớp:** 32 bộ đa lớp bổ sung (bản tiền huấn luyện nhị phân); TP-BERTa cạnh tranh, thứ hạng tốt (khoảng 5,1 và 3,5 theo hai thước đo trong bảng).
- **Y tế:** 25 bộ trong 145 bộ chính; TP-BERTa (không tuning) vượt GBDT đã tuning và các DNN, gợi ý dùng dữ liệu rẻ hơn từ miền khác để giảm phụ thuộc vào dữ liệu y tế đắt.

## 6. Hạn chế

Tác giả nêu (Phụ lục A):
- Cần **tên cột và giá trị có ngữ nghĩa rõ**; bảng có tên cột khó hiểu (vật lý lượng tử, mã hóa che đậy cho quyền riêng tư y tế, federated learning) không tận dụng được lợi thế của LM; với miền hiếm có thể dùng LM tiền huấn luyện trên kho miền đó.
- LM có nhiều tham số/siêu tham số nên tuning tốn thời gian; vì vậy dùng siêu tham số mặc định (cho kết quả đủ tốt, ít tốn thời gian hơn các mô hình tuning).
- Tốt hơn ở bảng nhiều cột phân loại, kém ở bảng toàn số; một số bảng thích ít token độ lớn hơn.

Nhận xét thêm của mình:
1. **Rất ít bằng chứng so với TabPFN hay mô hình ICL:** không đưa TabPFN vào baseline (cùng thời với TabPFN v1), nên chưa biết vị trí so với các TFM chủ chốt của đề tài.
2. Điều kiện "bảng điển hình" (< 10.000 mẫu, ≤ 32 đặc trưng) chính là miền mà TabPFN mạnh; do đó TP-BERTa so với TabPFN v2/TabICL v2 là phép so sánh tự nhiên còn thiếu.
3. **Rò rỉ ngữ nghĩa khả dĩ:** tiền huấn luyện trên 101 bộ lấy từ cùng kho (TabPertNet) với các bộ phía sau; tác giả đã loại trùng, nhưng cùng một miền dữ liệu và chủ đề (Kaggle/OpenML) vẫn có thể cho lợi thế so với baseline không biết gì về miền.
4. Bản chung (Ours_j) kém hơn bản đơn tác vụ và tiền huấn luyện tốn 72–143 giờ trên 4 A100: chi phí khá lớn so với TabPFN (tiền huấn luyện một lần trên dữ liệu tổng hợp).
5. Chia dữ liệu một lần (64/16/20) và xếp hạng trung bình, độ lệch chuẩn thứ hạng lớn (±3–4), nên chênh 0,1–0,5 hạng không có ý nghĩa thống kê.
6. **Mất cân bằng chỉ là phân tích hậu nghiệm**; không có thí nghiệm nhiễu nhãn.

## 7. Liên hệ với bài Nature và đề tài

- **Hai trường phái "tiền huấn luyện":** TabPFN v2 dùng **dữ liệu tổng hợp** (SCM + cây) và ICL, không dùng tên cột; TP-BERTa/CARTE/TransTab dùng **bảng thật + ngữ nghĩa cột**. Bài này là ví dụ cho lập luận ở bài 11 (position paper) và dẫn tới ConTextTab/TabSTAR, những mô hình lai hai trường phái.
- **Điểm yếu chung ở dữ liệu toàn số:** đúng chỗ TabPFN v2 mạnh, nên hai hướng bổ sung nhau.
- **Mất cân bằng:** kết quả "GBDT thắng ở p < 1/20" là gợi ý rằng khi mất cân bằng cực đoan thì mô hình nền tảng chưa chắc có lợi; đây là lý do để kiểm tra riêng TabPFN/TabICL ở bài toán cuối kỳ.
- Cập nhật nhận định khoảng trống: **chưa có benchmark khảo sát hệ thống nhiễu nhãn và hiệu chỉnh cho TFM**; mất cân bằng mới chỉ được khảo sát gián tiếp (bài này, trên mô hình LM).

## 8. Từ khóa cần nhớ

- **TP-BERTa:** RoBERTa + RMT + IFA, tiền huấn luyện giám sát trên 202 bảng (~10 triệu mẫu).
- **RMT:** bin C4.5 theo nhãn (≤ 256), token độ lớn dùng chung, nhân với giá trị, loss triplet theo khoảng cách chỉ số.
- **IFA:** một MHSA dùng chung, gộp (tên, giá trị) thành một vector, bỏ vị trí ở value → bất biến thứ tự cột, chuỗi ngắn.
- **Kết quả:** hạng 5,8 (nhị phân) và 4,3 (hồi quy) với siêu tham số mặc định; kém ở bảng toàn số; RMT hơn Value2Str 12,45% và VMFE 3,44%; bỏ IFA −4,17%; tiền huấn luyện +2,8–3,2%.
- **Mất cân bằng:** tốt ở mức vừa phải, GBDT thắng ở p < 1/20.
