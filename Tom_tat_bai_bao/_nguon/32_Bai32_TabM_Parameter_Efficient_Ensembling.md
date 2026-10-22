# Bài 32. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICLR 2025; arXiv:2410.24210v3, 18/2/2025 |
| Tác giả | Yury Gorishniy, Akim Kotelnikov, Artem Babenko (Yandex, HSE) |
| Mã nguồn | github.com/yandex-research/tabm |
| Lưu ý về PDF | Hình 3 (hạng và điểm tương đối) bị lẫn nhãn khi trích chữ: mình chỉ lấy con số hạng tốt nhất (TabM†mini 1,7 ± 1,2) và điểm trung bình ở Hình 2 (đọc rõ). Bảng 18–20 (kết quả từng bộ dữ liệu) hầu như là số thô, không tóm tắt chi tiết. |
| Vị trí trong đề tài | **Baseline học sâu mạnh nhất cho bảng ở thời điểm hiện tại** (không phải TFM). Xuất hiện trong TabArena và là đối thủ MLP-based mà TFM cần so sánh. Ý tưởng "ensemble tham số hiệu quả" có liên hệ với BETA (bài 15), vốn dùng bagging/ensemble để giảm phương sai của TabPFN. |

## 1. Bài này làm gì trong một đoạn

Bài chỉ ra cơ hội bị bỏ qua để làm MLP cho bảng tốt hơn nhiều: **ensemble hiệu quả về tham số**. TabM là **một mô hình duy nhất bắt chước một ensemble của k = 32 MLP**, cho ra nhiều dự đoán trên mỗi mẫu; các MLP ẩn được huấn luyện **đồng thời** và **dùng chung hầu hết trọng số** (kiểu BatchEnsemble của Wen 2020, với hai cải tiến nhỏ về tham số hóa và khởi tạo). Trên 46 bộ dữ liệu công khai (gồm 9 bộ chia theo miền thực tế), TabM là mô hình học sâu tốt nhất xét hạng, cạnh tranh với GBDT, nhanh hơn rõ rệt các mô hình dựa trên attention và truy xuất, và có thể chạy trên bộ dữ liệu 6,5–13 triệu mẫu. Phân tích cho thấy các dự đoán thành phần của TabM **yếu và quá khớp khi xét riêng, nhưng trung bình lại mạnh**.

## 2. Bối cảnh

- GBDT vẫn là baseline cổ điển mạnh; học sâu cho bảng (attention, truy xuất, MLP) được cho là ngang hoặc vượt GBDT trên benchmark học thuật. Nhưng từ góc nhìn thực dụng, có ba vấn đề: (1) mức độ và tính nhất quán của cải thiện so với MLP đơn giản ít được phân tích rõ; (2) hiệu quả (thời gian huấn luyện, thông lượng suy luận) ít được chú ý; (3) tiến bộ trên benchmark học thuật chưa chắc chuyển sang bài toán thực (Rubachev 2024, TabReD).
- **Deep ensemble** (nhiều mô hình cùng kiến trúc huấn luyện độc lập với seed khác nhau) thường mạnh hơn mô hình đơn và hữu ích cho ước lượng bất định/OOD (Lakshminarayanan 2017), nhưng đắt. **Ensemble tham số hiệu quả** (BatchEnsemble, MIMO, FiLM-Ensemble, Packed-Ensemble...) cố đạt hiệu quả tương tự với chi phí thấp. Các nỗ lực ensemble trong DL bảng trước (GrowNet, NODE) không hứa hẹn.
- Cải thiện MLP cho bảng nhờ thay đổi kiến trúc, regularization, kỹ thuật huấn luyện (Kadra 2021, Gorishniy 2022, Holzmüller 2024) đều trực giao với hướng của TabM.

## 3. Phương pháp

### 3.1. BatchEnsemble (nền tảng)
Với một lớp tuyến tính l(x) = Wx + b, deep ensemble truyền thống có (Wᵢ, bᵢ) riêng cho từng thành viên. BatchEnsemble dùng **W chung**, còn mỗi thành viên i chỉ có ba vector "adapter" rᵢ, sᵢ, bᵢ ∈ ℝᵈ: lᵢ(xᵢ) = sᵢ ⊙ (W(rᵢ ⊙ xᵢ)) + bᵢ, tương đương Wᵢ = W ⊙ (sᵢrᵢᵀ). Có thể gói k thành viên vào một mô hình xử lý đồng thời k đối tượng: l_BE(X) = ((X ⊙ R)W) ⊙ S + B, X ∈ ℝ^{k×d}. Thêm một thành viên chỉ thêm 3d tham số mỗi lớp (không đáng kể so với d² + d); chi phí thời gian thấp hơn nhiều so với ×k nhờ chạy song song.

### 3.2. Chuỗi biến thể dẫn đến TabM (k = 32 cố định)

| Biến thể | Mô tả | Điểm tương đối so với MLP (trung bình ± độ lệch chuẩn trên 46 bộ) |
|---|---|---|
| MLP | Baseline: N khối Dropout(ReLU(Linear)) và đầu tuyến tính | 0,00 ± 0,0% |
| FT-Transformer | Baseline attention | 0,39 ± 1,6% |
| **MLP×32** | Deep ensemble 32 MLP huấn luyện độc lập | 0,96 ± 1,6% |
| TabM_packed | 32 MLP độc lập gói trong một mô hình (Packed-Ensemble); huấn luyện đồng thời, dừng sớm theo hiệu năng cả ensemble | 1,42 ± 2,3% |
| TabM_naive | MLP + BatchEnsemble (3N adapter: R, S, B ở mọi lớp) | 1,86 ± 2,4% |
| TabM_bad | TabM_naive bỏ adapter R đầu tiên | 1,00 ± 2,3% |
| **TabM_mini** | Chỉ giữ adapter R đầu tiên, bỏ 3N − 1 adapter còn lại (MiniEnsemble) | 1,96 ± 2,8% |
| **TabM** | Như TabM_naive nhưng khởi tạo mọi adapter nhân R, S (trừ R đầu) bằng 1; ban đầu hành xử như TabM_mini, nhưng được tự do tăng khả năng biểu diễn | **2,15 ± 2,8%** |
| TabM♠ (dùng chung batch huấn luyện) | Cài đặt đơn giản hơn | 2,07 ± 3,6% |
| TabM†_mini (embedding đặc trưng phi tuyến piecewise-linear) | | 2,92 ± 4,1% |
| TabM†×5_mini (ensemble 5 TabM†_mini) | | 3,20 ± 4,4% |

Bài học rút ra:
1. **Huấn luyện đồng thời các thành viên** cho phép **dừng sớm theo hiệu năng của cả ensemble** và tinh chỉnh siêu tham số như một mô hình (MLP×32 dừng từng MLP theo điểm riêng, không tối ưu cho ensemble).
2. **Chia sẻ trọng số** hoạt động như một **dạng regularization** hiệu quả trên bảng: TabM_naive tốt hơn TabM_packed dù ít tham số hơn.
3. **Adapter R đầu tiên** là then chốt: nó ánh xạ k đầu vào đến k không gian biểu diễn khác nhau trước khi các đặc trưng được trộn lần đầu; bỏ nó (TabM_bad) làm hiệu năng giảm.
4. Siêu tham số mới duy nhất so với MLP là k (đặt cố định 32, không tinh chỉnh). Tốc độ học tối ưu của TabM thường cao hơn của MLP (giả thuyết: batch hiệu dụng lớn hơn).

### 3.3. Hai điều chỉnh thực dụng
- **♠ dùng chung batch huấn luyện:** các submodel dùng cùng batch; mất ít hiệu năng nhưng đơn giản và hiệu quả hơn.
- **† embedding đặc trưng phi tuyến:** dùng bản sửa của embedding piecewise-linear (Gorishniy 2022, biến thể "Q-L"/"T-L" với phân khoảng theo quantile); kết hợp tốt với TabM (adapter R đầu có kích thước k × m·d_e).
- Cũng có thể deep ensemble thêm trên TabM (TabM†×5_mini).

### 3.4. Vì sao MLP và BatchEnsemble (Phụ lục A.1)
BatchEnsemble đơn giản, linh hoạt, dễ cụ thể hóa, phân tích và cắt tỉa từng submodel; MIMO ghép đầu vào (k = 32 và 1000 đặc trưng embedding 32 chiều cho ~1 triệu chiều đầu vào) và FiLM-Ensemble cần lớp chuẩn hóa. MLP được chọn vì **hiệu quả**: lợi thế "chi phí thấp hơn ×k" chỉ có khi mô hình gốc chưa dùng hết phần cứng, điều không đúng với attention (nhiều đặc trưng) hoặc truy xuất (nhiều mẫu). Các backbone phức tạp hơn không cho lợi ích rõ trong thử nghiệm sơ bộ.

## 4. Thiết lập thí nghiệm

- **Benchmark 46 bộ dữ liệu:** 38 bộ từ Gorishniy 2024 (28 từ Grinsztajn 2022 và 10 bộ khác), 8 bộ TabReD (Rubachev 2024) và Microsoft. Kích thước tập huấn luyện: nhỏ nhất 1,8K, trung vị 12K, trung bình 76K, lớn nhất 723K; số đặc trưng: 3, 20 (trung vị), 108 (trung bình), 986; **28 hồi quy và 18 phân loại**; **37 bộ chia ngẫu nhiên** và **9 bộ chia "domain-aware"** (chia thực tế theo thời gian, có dịch chuyển phân phối tự nhiên). Loại các bộ có chia sai hoặc rò rỉ nhãn theo Rubachev (Bike Sharing Demand, compass, electricity, SGEMM, sulfur, visualizing soil, weather cũ) và một số bộ bất thường (rl, yprop_4_1).
- **Quy trình (theo Gorishniy 2024):** tinh chỉnh trên validation (Optuna TPE, thường 50–100 vòng, 25 vòng cho một số bộ TabReD); huấn luyện lại từ đầu với nhiều seed (đa số 15); điểm cuối là trung bình test qua các seed. Mạng: cross-entropy / MSE, AdamW, không lịch learning rate, không tăng cường dữ liệu, cắt gradient toàn cục 1,0, batch size theo bộ dữ liệu, patience = 16. Tiền xử lý: biến đổi quantile (có sửa nhẹ), one-hot cho hạng mục.
- **Thước đo:** RMSE (hồi quy), ROC-AUC (phân loại bộ TabReD) hoặc accuracy (còn lại); xét cả **điểm tương đối so với MLP**: (score/baseline − 1) × 100%, với hồi quy đổi RMSE sang R² để cân thang đo.
- **Cách tính hạng:** mô hình A tốt hơn B nếu A_mean − A_std > B_mean; các mô hình "không tệ hơn" mô hình tốt nhất cùng hạng 1, mô hình đầu tiên tệ hơn nhận hạng 2 và trở thành mốc mới; vì vậy hạng giống như "tầng".
- **Baseline:** MLP, FT-Transformer, SAINT, T2G-Former, ExcelFormer, TabR, ModernNCA, XGBoost, LightGBM, CatBoost, và (phụ lục) MLP-PLR, ResNet, SNN, DCNv2, AutoInt, MLP-Mixer, Trompt, TabPFN. Dấu † và ‡ là mô hình dùng embedding phi tuyến (piecewise-linear hoặc tuần hoàn).

## 5. Kết quả

### 5.1. Hiệu năng (Hình 3)
- **TabM_mini†** có hạng trung bình tốt nhất **1,7 ± 1,2** (hạng 1 là hạng tốt nhất), cao hơn rõ MLP (5,5 ± 3,2 là mốc tệ nhất trong sơ đồ). TabM là mô hình sâu hàng đầu theo hạng.
- Nhiều mô hình sâu (attention, truy xuất) **không hơn, thậm chí tệ hơn MLP** trên một số bộ dữ liệu đáng kể, nên kém tin cậy; thứ hạng thay đổi nhất ở các bộ chia theo miền thực tế. Một đặc trưng quan trọng là phần yếu nhất của hồ sơ hiệu năng (phân vị 10 hoặc 25): MLP† là lựa chọn thực dụng tốt giữa MLP thường và TabM, đơn giản hơn TabR và ModernNCA.
- Kết luận: TabM xác nhận là baseline học sâu tin cậy; **MLP, kể cả TabM, tạo thành tuyến mô hình mạnh và thực dụng hơn attention hay truy xuất**.

### 5.2. Hiệu quả (Hình 4, Hình 10, Bảng 2, Bảng 3)
- **Thời gian huấn luyện (bộ > 100K mẫu) và thông lượng suy luận:** MLP đơn nhanh nhất, TabM đứng thứ hai; attention và truy xuất chậm hơn nhiều (thang thời gian từ khoảng 2 phút đến hơn 3 giờ).
- **Bảng 2, hai bộ rất lớn (bản đầy đủ của TabReD Weather 13M mẫu, 103 đặc trưng và Maps Routing 6,5M mẫu, 986 đặc trưng; dùng lại siêu tham số của bản nhỏ):** RMSE (thời gian huấn luyện), theo thứ tự cột XGBoost, MLP, TabM†♠∗mini, TabM†mini, FT-T:
  - Maps Routing: 0,1601 (28 phút), 0,1592 (15 phút), **0,1583 (2 giờ)**, 0,1582 (13,5 giờ), 0,1594 (45,5 giờ).
  - Weather: 1,4234 (10 phút), 1,4842 (15 phút), **1,4090 (1,3 giờ)**, 1,4112 (3,3 giờ), 1,4409 (13,5 giờ).
  - TabR: hết bộ nhớ (OOM) ở cả hai. Bản TabM♠∗ tối ưu (AMP + torch.compile) nhanh hơn rõ bản thường.
- **Số tham số (Bảng 3, trung bình trên các siêu tham số tinh chỉnh):** TabM 1,4M ± 1,3M, MLP 1,0M, FT-T 1,2M, T2G 2,1M, TabR 858K, ModernNCA 1,0M, SAINT 175,4M ± 565,4M.

## 6. Phân tích

### 6.1. Các submodel riêng lẻ (Hình 5)
Thiết lập đơn giản hóa (độ sâu 3, độ rộng 512, không dropout/weight decay, **không dừng sớm**): so TabM_mini với k = 32 và k = 1 (tức một MLP thường) trên bốn bộ dữ liệu (Churn, House, Microsoft, Otto). Kết quả: dự đoán tập thể (trung bình) tốt hơn dự đoán riêng cả ở loss huấn luyện lẫn kiểm tra. Ở tọa độ train–test, các submodel **quá khớp khi đứng riêng** so với MLP nền nhưng dự đoán tập thể khái quát hóa tốt hơn hẳn, bằng chứng của **sự đa dạng đáng kể** giữa các submodel. **Submodel tốt nhất của TabM (TabM[B]) không tốt hơn MLP đơn giản** (−0,06 ± 1,8%). Tóm tắt: sức mạnh của TabM đến từ **dự đoán tập thể của các submodel yếu nhưng đa dạng**.

### 6.2. Chọn submodel sau huấn luyện (Hình 6)
Có thể cắt các đầu dự đoán và hàng adapter tương ứng. Chọn tham lam tập con tốt nhất trên validation (TabM[G]) cho điểm 2,02 ± 2,6% so với 2,15 ± 2,8% của TabM đầy đủ, chỉ giữ trung bình **8,8 ± 6,6 submodel trên 32**, hữu ích cho suy luận nhanh.

### 6.3. Phụ thuộc vào k (Hình 7)
Trên 17 bộ dữ liệu (trừ Covertype và các bộ TabReD): mạng càng "lớn" (n lớp, độ rộng d) càng chứa được nhiều submodel hiệu quả (các đường theo d tách ra từ k = 2, 4); **k quá lớn có hại** (chia sẻ trọng số giới hạn số submodel cùng tồn tại); cấu hình quá hẹp (d = 64) hoặc quá nông (n = 1) kém, ít nhất ở dữ liệu cỡ vừa và lớn.

### 6.4. Giảm nơ-ron chết (5.4)
Mỗi nơ-ron chung của TabM_mini được dùng k lần mỗi lượt truyền xuôi, nên có nhiều cơ hội kích hoạt hơn. Tỷ lệ nơ-ron chết (không bao giờ kích hoạt trên 2048 mẫu huấn luyện) là **0,29 ± 0,17 với k = 1** và **0,14 ± 0,09 với k = 32**, trung bình trên 46 bộ.

## 7. Phụ lục: chi tiết đáng nhớ

- **TabPFN (bản v1) trên benchmark này:** chỉ chạy được trên bộ có ≤ 100 đặc trưng, không áp dụng cho hồi quy (nên bỏ qua một phần lớn bộ dữ liệu), dùng mẫu con 10K cho mỗi seed. Kết quả **không cạnh tranh**, vì TabPFN chuyên cho dữ liệu nhỏ.
- **Hạn chế thực dụng (A.4):** backbone MLP đơn giản nhất, chưa thử backbone có lớp chuẩn hóa (cần thêm adapter nhân và cộng sau chuẩn hóa như FiLM-Ensemble); embedding đối tượng của TabM là **một tập k vector**, chưa rõ cách gộp thành một vector khi cần; nguyên tắc: k nhánh dự đoán **không được tương tác** với nhau (không attention, pooling) và phải huấn luyện riêng.
- **Siêu tham số tinh chỉnh TabM (Bảng 6):** k = 32; số lớp UniformInt[1, 5]; độ rộng [64, 1024]; dropout {0, U[0, 0,5]}; lr log-uniform [1e−4, 5e−3]; weight decay {0, log-uniform [1e−4, 1e−1]}.
- **Phần cứng:** chủ yếu một GPU A100; một số thử nghiệm trên 2080Ti.
- **Hạn chế benchmark hiệu quả:** tinh chỉnh không nhận biết ngân sách tính toán, có thể chọn cấu hình nặng cho lợi ích rất nhỏ.

## 8. Hạn chế

Tác giả nêu: chỉ thử backbone MLP đơn giản; chưa thử các lớp chuẩn hóa; chưa đánh giá cho ước lượng bất định và phát hiện OOD (hướng tương lai, lấy cảm hứng từ Lakshminarayanan 2017); và hướng mở rộng ensemble hiệu quả sang miền khác ngoài bảng.

Nhận xét thêm của mình:
1. Tất cả vẫn là mô hình **huấn luyện riêng cho từng bộ dữ liệu** kèm tinh chỉnh siêu tham số, nên không so được trực tiếp về chi phí với TFM dùng ICL (không huấn luyện) như TabPFNv2.
2. Chênh lệch giữa các biến thể (ví dụ 1,86% so với 1,96% so với 2,15%) nhỏ so với độ lệch chuẩn qua các bộ dữ liệu (2,3–2,8%), nên thứ tự các biến thể phụ không chắc chắn; kết luận mạnh là ensemble hiệu quả hơn MLP đơn.
3. Cách tính hạng "tầng" rộng tay với mô hình có độ lệch chuẩn lớn, nên hạng thấp (ít tầng) dễ bị nén.
4. Thông điệp "bias–variance": ensemble giảm phương sai, trùng với phân tích của BETA (bài 15) cho TabPFN; đây là điểm nối tự nhiên giữa DL bảng cổ điển và TFM.
5. Không thí nghiệm nhiễu nhãn hay mất cân bằng; tuy nhiên ensemble thường làm mô hình bền hơn với nhiễu.

## 9. Liên hệ với bài Nature và đề tài

- **Baseline học sâu cho thí nghiệm cuối kỳ:** TabM (hoặc MLP†) là đại diện "DL bảng tốt nhất" bên cạnh XGBoost/CatBoost và TabPFN; dùng được thư viện chính thức.
- **Cầu nối khái niệm:** TabM chứng minh rằng *ensemble thành phần yếu nhưng đa dạng* quan trọng; BETA làm tương tự cho TFM (bagging/ensemble giảm phương sai), nên một câu hỏi nghiên cứu: *ensemble/bagging có làm TFM bền hơn với nhiễu nhãn không?*
- **Đánh giá hiệu quả:** thang thời gian huấn luyện và thông lượng suy luận của bài là mẫu tốt để báo cáo chi phí cho cả TFM.

## 10. Từ khóa cần nhớ

- **TabM:** MLP ensemble tham số hiệu quả, k = 32, adapter R, S, B kiểu BatchEnsemble; khởi tạo adapter bằng 1 trừ R đầu.
- **Hai lý do hiệu quả:** huấn luyện đồng thời (dừng sớm theo ensemble) + chia sẻ trọng số (regularization).
- **Phát hiện:** submodel yếu và quá khớp riêng lẻ, mạnh tập thể; nơ-ron chết giảm từ 0,29 xuống 0,14; submodel tốt nhất không hơn MLP.
- **Số liệu:** 46 bộ dữ liệu (28 hồi quy, 18 phân loại; 9 bộ domain-aware); điểm tương đối +2,15% (TabM), +2,92% (TabM†mini), +3,20% (ensemble 5 TabM†mini); TabM†mini hạng 1,7; xử lý bộ 6,5M–13M mẫu; cắt còn khoảng 8,8 trên 32 submodel vẫn giữ 2,02%.
