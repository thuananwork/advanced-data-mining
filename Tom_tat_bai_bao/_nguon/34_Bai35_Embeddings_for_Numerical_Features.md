# Bài 35. On Embeddings for Numerical Features in Tabular Deep Learning

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2022; arXiv:2203.05556v4, 26/10/2023 |
| Tác giả | Yury Gorishniy, Ivan Rubachev, Artem Babenko (Yandex, HSE) |
| Mã nguồn | github.com/yandex-research/tabular-dl-num-embeddings |
| Lưu ý về PDF | Bảng 3–6, 8–10, 12 đọc rõ (có bảng bị tách dòng nhưng thứ tự cột rõ). Bảng 7 (số tham số) đọc được dạng hệ số nhân. Bảng 18–19 (kết quả có độ lệch chuẩn) là phụ lục dài, không tóm tắt từng ô. |
| Vị trí trong đề tài | Bài **kỹ thuật tiền xử lý/biểu diễn số**: embedding cho đặc trưng số (PLE, Periodic) là thành phần chuẩn của nhiều mô hình mới (TabM†, TabR, ModernNCA, TabPFNv2 ở mức token ô, TabICL). Giải thích vì sao MLP có thể đạt ngang attention. |

## 1. Bài này làm gì trong một đoạn

Các mô hình dạng Transformer cho bảng (FT-Transformer, AutoInt) biến mỗi **giá trị số vô hướng** thành vector nhiều chiều trước khi trộn qua backbone, nhưng văn liệu chỉ tập trung cải tiến backbone và để phần embedding rất đơn giản (thường là một lớp tuyến tính). Bài chứng minh rằng **embedding cho đặc trưng số** là một bậc tự do bị bỏ quên và có tác động lớn. Hai họ embedding được đề xuất: (1) **mã hóa tuyến tính từng đoạn (PLE, piecewise linear encoding)** dựa trên chia khoảng giá trị (binning), và (2) **hàm kích hoạt tuần hoàn (periodic)** dựa trên sin/cos với hệ số học được. Cả hai cải thiện đáng kể nhiều backbone (MLP, ResNet, Transformer), giúp **MLP đơn giản đạt ngang các mô hình attention**, thu hẹp khoảng cách với GBDT trên các bộ dữ liệu "thân thiện với GBDT", và lần đầu cho mạng sâu ngang GBDT trên California Housing và Adult.

## 2. Bối cảnh

- Mạng sâu cho bảng vẫn thường thua ensemble cây (GBDT), và các kiến trúc phức tạp không hơn MLP/ResNet được tinh chỉnh kỹ (Gorishniy 2021, Kadra 2021).
- Transformer cho bảng buộc phải ánh xạ giá trị vô hướng sang vector; FT-Transformer chỉ dùng **một lớp tuyến tính**. Trong dự đoán tỷ lệ nhấp (CTR), có nơi đã xử lý đặc trưng số phi tầm thường (AutoDis của Guo 2021) nhưng vẫn dựa trên lớp tuyến tính và kích hoạt thông thường.
- **Binning** là kỹ thuật tiền xử lý cổ điển (rời rạc hóa; Dougherty 1995). **Kích hoạt tuần hoàn** đến từ biểu diễn nơ-ron ẩn (NeRF, SIREN), Fourier features (Tancik 2020) và NLP/CV. Khác các công trình đó, bài này phải **embedding riêng từng đặc trưng trước khi trộn** (không trộn đặc trưng bằng lớp tuyến tính trước khi qua hàm tuần hoàn như Tancik).

## 3. Phương pháp

### 3.1. Khung chung
Với đặc trưng số thứ i, embedding là z_i = f_i(x_i^{(num)}) ∈ ℝ^{d_i}; các f_i tính **độc lập** nhau, cùng dạng hàm nhưng **tham số không chia sẻ** giữa các đặc trưng. Dùng cho MLP-like thì nối (concatenate) mọi z_i thành vector phẳng; Transformer thì đưa trực tiếp các embedding như token.

### 3.2. Piecewise linear encoding (PLE)
Chia miền giá trị của đặc trưng thành T khoảng B_t = [b_{t−1}, b_t). Mã hóa PLE(x) = [e_1, …, e_T] với
- e_t = 0 nếu x < b_{t−1} và t > 1;
- e_t = 1 nếu x ≥ b_t và t < T;
- e_t = (x − b_{t−1}) / (b_t − b_{t−1}) trong các trường hợp còn lại.

Tính chất: là **tiền xử lý** tính một lần; T = 1 tương đương biểu diễn vô hướng; vì giá trị số có thứ tự nên các thành phần ứng với khoảng nằm hoàn toàn bên trái x được đặt bằng 1 (giống mã hóa nhãn trong hồi quy thứ tự); các giá trị ngoài [b₀, b_T] vẫn được bao phủ; mọi thành phần nằm trong [0, 1] và bất biến với dịch/co giãn. Với mô hình attention vốn bất biến thứ tự đầu vào, cần thêm **một lớp tuyến tính sau PLE** (không chia sẻ giữa đặc trưng); lớp này tương đương gán mỗi khoảng một embedding v_t và lấy tổng có trọng số e_t cộng bias v₀: f_i(x) = v₀ + Σ e_t·v_t = Linear(PLE(x)).

**Hai cách tạo khoảng:**
- **Dựa trên phân vị (quantile, "PLE_q" / ký hiệu Q):** b_t = phân vị thứ t/T của đặc trưng trên tập huấn luyện; bỏ khoảng có kích thước 0. Không giám sát.
- **Nhận biết nhãn (target-aware, "PLE_t" / ký hiệu T):** chia đệ quy theo kiểu tham lam bằng nhãn, tương đương xây một cây quyết định chỉ trên đặc trưng đó và nhãn rồi lấy các vùng lá làm khoảng (tinh thần thuật toán "C4.5 Discretization" của Kohavi & Sahami); b₀ = min, b_T = max trên tập huấn luyện. Siêu tham số: số lá tối đa (2–256), số mẫu tối thiểu mỗi lá (1–128), lợi ích thông tin tối thiểu (log-uniform 1e−9 đến 0,01).

### 3.3. Embedding tuần hoàn (Periodic)
f_i(x) = Periodic(x) = concat[sin(v), cos(v)], v = [2πc₁x, …, 2πc_k x], với c_j là tham số học được khởi tạo từ N(0, σ). Khác Tancik 2020: không trộn đặc trưng khi embedding, và **huấn luyện** hệ số tiền kích hoạt thay vì cố định (gần với Li 2021 "learnable Fourier features"). σ là siêu tham số quan trọng cần tinh chỉnh (cùng k, mọi đặc trưng chung).

### 3.4. Lớp khả vi đơn giản và các kết hợp
Có thể đặt thêm các lớp thông thường ở trên PLE hoặc Periodic. Quy ước tên "Backbone-Embedding": L = Linear; LR = ReLU∘Linear; LRLR = ReLU∘Linear∘ReLU∘Linear; Q / T = PLE_q / PLE_t; Q-L, Q-LR, Q-LRLR, T-L, T-LR, T-LRLR; P = Periodic; PL, PLR, PLRLR = Linear/ReLU xếp trên Periodic; AutoDis (Guo 2021). "Transformer-L" chính là FT-Transformer.

## 4. Thiết lập thí nghiệm

- **11 bộ dữ liệu công khai** (thiên về bài toán thân thiện với GBDT, quy mô vừa và lớn):

| Viết tắt | Tên | #Train | #Đặc trưng số | Hạng mục | Loại / thước đo |
|---|---|---|---|---|---|
| GE | Gesture Phase | 6.318 | 32 | 0 | Đa lớp (5), Acc. |
| CH | Churn Modelling | 6.400 | 10 | 1 | Nhị phân, Acc. |
| CA | California Housing | 13.209 | 8 | 0 | Hồi quy, RMSE |
| HO | House 16H | 14.581 | 16 | 0 | Hồi quy, RMSE |
| AD | Adult | 26.048 | 6 | 8 | Nhị phân, Acc. |
| OT | Otto Group | 39.601 | 93 | 0 | Đa lớp (9), Acc. |
| HI | Higgs Small (98K) | 62.751 | 28 | 0 | Nhị phân, Acc. |
| FB | Facebook Comments | 157.638 | 50 | 1 | Hồi quy, RMSE |
| SA | Santander Customer Transaction | 128.000 | 200 | 0 | Nhị phân, Acc. |
| CO | Covertype | 371.847 | 54 | 0 | Đa lớp (7), Acc. |
| MI | MSLR-WEB10K (fold 1) | 723.412 | 136 | 0 | Hồi quy (pointwise), RMSE |

- **Quy trình (theo Gorishniy 2021):** biến đổi quantile (scikit-learn) cho mọi bộ trừ Otto (không tiền xử lý); chuẩn hóa nhãn hồi quy; tinh chỉnh Optuna TPE trên validation (100 vòng; Transformer 50 vòng với SA, CO, MI); 15 seed, ensemble ba nhóm năm mô hình; AdamW, patience = 16, không lịch learning rate; one-hot cho hạng mục (CatBoost dùng hỗ trợ nội tại). Một kết quả được xem là tốt hơn nếu trung bình tốt hơn và độ lệch chuẩn nhỏ hơn chênh lệch.
- **Backbone:** MLP, ResNet, Transformer (từ Gorishniy 2021). Không đưa vào các mô hình có attention giữa các đối tượng (SAINT, NPT, Hopfield) vì thành phần phi tham số này trực giao với chủ đề.

## 5. Kết quả

### 5.1. Embedding đơn giản (Bảng 3, MLP)

| | GE↑ | CH↑ | CA↓ | HO↓ | AD↑ | OT↑ | HI↑ | FB↓ | SA↑ | CO↑ | MI↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 0,632 | 0,856 | 0,495 | 3,204 | 0,854 | 0,818 | 0,720 | 5,686 | 0,912 | 0,964 | 0,747 |
| MLP-L | 0,639 | 0,861 | 0,475 | 3,123 | 0,856 | 0,820 | 0,723 | 5,684 | 0,916 | 0,963 | 0,748 |
| MLP-LR | 0,642 | 0,860 | 0,471 | 3,084 | 0,857 | 0,819 | 0,726 | 5,625 | 0,923 | 0,963 | 0,746 |

MLP hưởng lợi từ embedding; LR cải thiện nhỏ nhưng đều; MLP-L "dư thừa" vẫn tốt hơn MLP và có thể **gộp (fuse) vào lớp đầu của MLP sau huấn luyện** nên không tốn chi phí suy luận. LRLR và AutoDis quá đắt so với lợi ích.

### 5.2. PLE (Bảng 4)
- MLP-Q: GE 0,653, CA 0,464, AD 0,859, SA 0,922, CO 0,968; MLP-T: CA 0,447, HO 3,149, AD 0,864, FB 5,577, SA 0,923, CO 0,967; MLP-T-LR: CA **0,439**, AD **0,868**, HI 0,724; MLP-Q-LR: FB **5,394**.
- Transformer: Transformer-L (FT-T) GE 0,632, CA 0,465, AD 0,858, CO 0,971; Transformer-T-L: GE 0,663, AD 0,871, CO 0,974; Transformer-T-LR: GE 0,665, CA 0,442, AD 0,870.
- Kết luận: PLE thường có lợi cho cả MLP lẫn Transformer, đặc biệt ở **CA và AD**; thêm lớp khả vi lên trên PLE giúp thêm, nhưng các biến thể đắt nhất (Q-LRLR, T-LRLR) không đáng. Ưu thế của bin theo cây (T) so với phân vị (Q) có thể do benchmark thiên về GBDT, nên bài không khẳng định chung.

### 5.3. Embedding tuần hoàn (Bảng 5)
- MLP-P: AD 0,869, SA 0,923, HO 3,129; nhưng kém MLP ở OT (0,807 < 0,818) và FB (5,845 > 5,686). MLP-PL và MLP-PLR sửa các chỗ đó: **MLP-PLR** GE 0,674, HO 3,050, AD 0,870, SA 0,924, CO 0,970; Transformer-PLR: GE 0,646, AD 0,870, HI 0,730.
- Quy tắc mặc định: **luôn đặt thành phần khả vi (Linear, ReLU) lên trên Periodic**; PLR thường tốt hơn PL nhưng PL có lớp tuyến tính cuối có thể gộp sau huấn luyện; PLRLR không khác PLR đáng kể nên không đáng chi phí.

### 5.4. So sánh toàn diện với GBDT (Bảng 6, ensemble, hạng trung bình trên 11 bộ)

| Mô hình | Hạng TB | | Mô hình | Hạng TB |
|---|---|---|---|---|
| CatBoost | 3,6 ± 2,9 | | ResNet | 6,7 ± 3,3 |
| XGBoost | 4,6 ± 2,7 | | ResNet-LR | 5,6 ± 2,7 |
| MLP | 8,5 ± 2,6 | | ResNet-Q-LR | 4,7 ± 2,0 |
| MLP-LR | 5,5 ± 2,7 | | ResNet-T-LR | 4,1 ± 1,9 |
| MLP-Q-LR | 5,1 ± 1,9 | | ResNet-PLR | 3,2 ± 1,3 |
| MLP-T-LR | 5,1 ± 1,7 | | Transformer-L | 5,9 ± 2,2 |
| **MLP-PLR** | **3,0 ± 2,4** | | Transformer-LR | 5,2 ± 2,2 |
| | | | Transformer-Q-LR | 4,4 ± 2,2 |
| | | | Transformer-T-LR | 3,7 ± 2,2 |
| | | | Transformer-PLR | 3,9 ± 2,5 |

- Embedding cho cải thiện rõ rệt ở ba backbone: hạng MLP **8,5 → 3,0** (MLP-PLR). Embedding LR đơn giản là baseline nhất quán; **PLR cho hiệu năng trung bình tốt nhất** (σ cần tinh chỉnh); PLE (T-LR, Q-LR) cho embedding tốt, lại **đơn giản, giải thích được, rẻ** (không có hàm tuần hoàn).
- **Sau khi có embedding, MLP-like đạt ngang mô hình dựa trên Transformer.**
- **So với GBDT:** benchmark vốn thiên về GBDT (xem khoảng cách CatBoost/XGBoost với MLP, ResNet, Transformer-L gốc). Với hầu hết cặp backbone–bộ dữ liệu, embedding đúng là đủ để thu hẹp khoảng cách; ngoại lệ gồm MI và các cặp ResNet & GE, Transformer & FB, Transformer & GE, Transformer & OT. Lần đầu mạng sâu **ngang GBDT trên California Housing và Adult** (CatBoost CA 0,430 so với MLP-T-LR 0,435, ResNet-T-LR 0,425, Transformer-T-LR 0,423; AD: CatBoost 0,873 so với MLP-PLR 0,874). Hiệu quả vẫn là vấn đề so với GBDT.

## 6. Phân tích

- **Kích thước mô hình (Bảng 7):** embedding làm tăng số tham số khá lớn. Ví dụ MLP-LR trên CH tăng khoảng ×1931 (từ 1,5K tham số), trên HI ×68, nhưng thời gian huấn luyện chỉ tăng khoảng 1,5 lần ở CH. Mô hình MLP/ResNet có embedding vẫn **nhanh hơn Transformer**.
- **Ablation PLE (Bảng 8):** so PLE với (a) mã hóa nhị phân kiểu nhiệt kế ("thermometer") và (b) one-blob (rộng bản T−γ, tinh chỉnh γ). PLE tuyến tính từng đoạn tốt nhất hoặc ngang: ví dụ MLP-Q: GE 0,653 so với 0,652 (binary) và 0,613 (one-blob); CH 0,854 so với 0,815 (binary); MLP-T trên FB 5,577 so với 5,572 (binary) và 5,475 (one-blob). Vậy "piecewise linear" là mặc định tốt.
- **PLE như tiền xử lý (Bảng 9):** vì thành phần nằm trong [0, 1] và bất biến với dịch/co giãn, PLE ít phụ thuộc vào tiền xử lý. MLP thường mà không tiền xử lý hầu như **vô dụng** (CA 1,118; FB 13,125; MI 0,844), còn với chuẩn hóa thông thường hay quantile thì tốt hơn nhiều (CA 0,509 hoặc 0,495); MLP-Q (none) và MLP-Q (quantile) rất gần nhau (CA 0,463 và 0,464), MLP-T cũng vậy (0,447 cả hai).
- **Góc nhìn "feature engineering" (Bảng 10):** thử đưa Periodic (hệ số ngẫu nhiên cố định, giữ đặc trưng gốc) vào XGBoost: CA 0,436 → 0,441, HO 3,160 → 3,184, HI 0,724 → 0,724, tức **không giúp GBDT**, trong khi MLP-PL tốt hơn MLP (CA 0,495 → 0,467, HO 3,204 → 3,113, HI 0,720 → 0,727). Kết luận: lợi ích gắn với đặc điểm tối ưu của DL chứ không phải kỹ thuật đặc trưng chung.
- **Fourier features gốc (Phụ lục D.2, Bảng 12):** MLP với Fourier features không chia riêng từng đặc trưng kém hơn cả MLP thường (GE 0,612 so với 0,632, CH 0,845 so với 0,856, FB 5,767 so với 5,686), nên **embedding từng đặc trưng riêng là quan trọng**.
- **Thí nghiệm tổng hợp thân thiện GBDT (Phụ lục D.1, Hình 5):** 10.000 mẫu, 8 đặc trưng, nhãn sinh bởi 16 cây độ sâu 6; PLE (Q) cải thiện cả MLP và Transformer; tăng số khoảng cải thiện thêm trong thiết lập này, nhưng thực tế quá nhiều khoảng gây quá khớp nên cần tinh chỉnh T trên validation. MLP tinh chỉnh đạt 0,2229 ± 0,0055 (vẫn kém CatBoost).

## 7. Hạn chế

Tác giả nêu:
- Chưa giải thích được ở mức cơ bản vì sao các embedding giúp tối ưu hóa.
- Mọi đặc trưng dùng cùng một dạng hàm embedding; cách dùng hàm khác nhau cho từng đặc trưng có thể tốt hơn.
- Embedding làm tăng số tham số (Bảng 7); hiệu quả tính toán so với GBDT vẫn là vấn đề.

Nhận xét thêm của mình:
1. Benchmark có chủ đích **thiên về GBDT** và chỉ gồm 11 bộ cỡ vừa đến lớn; không có bộ dữ liệu nhỏ, nơi TFM mạnh nhất.
2. Hạng trung bình ± độ lệch chuẩn qua 11 bộ là thước đo thô (chính tác giả nói "không phải thước đo tốt cho kết luận tinh").
3. Kết luận so với GBDT là về **ensemble** DL so với ensemble GBDT, bỏ qua chi phí; không nói về mô hình đơn.
4. Chỉ xét đặc trưng **số**; hạng mục dùng one-hot, tức là phần "embedding" cho hạng mục không được nghiên cứu.
5. Không đánh giá nhiễu nhãn, mất cân bằng hay hiệu chỉnh.

## 8. Liên hệ với bài Nature và đề tài

- **Trong TFM:** TabPFNv2 và TabICL cũng phải xử lý giá trị số: mã hóa mỗi ô thành token qua tiền xử lý riêng từng cột (chuẩn hóa/biến đổi mũ) rồi tuyến tính; bài này cho thấy các thiết kế thay thế (PLE, Periodic) cải thiện đáng kể mô hình học riêng từng bộ dữ liệu, gợi ý hướng cải tiến encoder số cho TFM.
- **Trong TabM (bài 32):** TabM†_mini sử dụng embedding piecewise-linear (biến thể "Q-L"/"T-L") và đạt kết quả tốt nhất; đọc bài 35 để hiểu nền tảng.
- **Cho thí nghiệm cuối kỳ:** khi dùng MLP làm baseline, nên dùng MLP-PLR (hoặc MLP-T-LR) thay vì MLP trơn để baseline công bằng.

## 9. Từ khóa cần nhớ

- **PLE:** mã hóa từng đoạn, e_t ∈ [0, 1]; bin theo phân vị (Q) hoặc theo cây có nhãn (T); sau đó Linear (L) hoặc Linear + ReLU (LR).
- **Periodic:** sin/cos của 2π·c·x, c học được khởi tạo N(0, σ); PLR = Periodic + Linear + ReLU; σ quan trọng.
- **Embedding từng đặc trưng độc lập, không chia sẻ tham số.**
- **Số liệu:** 11 bộ dữ liệu; hạng MLP 8,5 → MLP-PLR 3,0 (CatBoost 3,6); MLP-L có thể gộp vào lớp đầu; Periodic không giúp XGBoost; MLP-Q/T ít nhạy với tiền xử lý.
- **Thông điệp:** embedding đặc trưng số là yếu tố thiết kế quan trọng; MLP có embedding ≈ Transformer.
