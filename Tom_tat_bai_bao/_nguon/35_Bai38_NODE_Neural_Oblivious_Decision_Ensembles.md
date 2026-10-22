# Bài 38. Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (NODE)

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICLR 2020 (bản arXiv:1909.06312v2, 19/9/2019) |
| Tác giả | Sergei Popov, Stanislav Morozov, Artem Babenko (Yandex) |
| Mã nguồn | github.com/Qwicen/node (PyTorch) |
| Lưu ý về PDF | Bảng 1–2 bị tách dòng khi trích chữ; mình ghép theo thứ tự cột của tiêu đề (Epsilon, YearPrediction, Higgs, Microsoft, Yahoo, Click) và kiểm lại bằng phần nhận xét của bài. Bảng 3, 4 đọc rõ. |
| Vị trí trong đề tài | Bài **"cây khả vi"** kinh điển: đưa cây quyết định không-nhớ (oblivious) của CatBoost vào mạng sâu huấn luyện đầu cuối. Được dùng làm baseline trong bài 30 (FT-Transformer), bài 33 (McElfresh) và TabFlex. Liên quan đến lập luận "vì sao cây vẫn mạnh" và tới prior dựa trên cây của TabPFNv2/Mitra. |

## 1. Bài này làm gì trong một đoạn

Trên dữ liệu bảng không đồng nhất, các mô hình nông như GBDT (XGBoost, LightGBM, CatBoost) vẫn thường tốt nhất, và chưa có kiến trúc sâu nào vượt chúng một cách hệ thống. NODE là một kiến trúc DNN **khái quát hóa CatBoost**: biến các **cây quyết định không-nhớ (oblivious decision tree, ODT)** thành khả vi bằng phép biến đổi **entmax** (cho phép chọn đặc trưng chia thưa và định tuyến mềm), rồi xếp nhiều lớp cây nối chặt kiểu **DenseNet** và huấn luyện đầu cuối bằng lan truyền ngược, tạo thành "GBDT sâu huấn luyện end-to-end". Trên 6 bộ dữ liệu lớn (Epsilon, YearPrediction, Higgs, Microsoft, Yahoo, Click), NODE thắng CatBoost và XGBoost ở cấu hình mặc định trên cả 6 bộ, và với cấu hình tinh chỉnh thắng trên hầu hết (trừ Yahoo và Microsoft nơi XGBoost tinh chỉnh tốt nhất).

## 2. Bối cảnh

- **Hiện trạng:** ensemble cây (GBDT, rừng ngẫu nhiên) là lựa chọn hàng đầu cho bảng; nhiều cuộc thi Kaggle vẫn do GBDT thắng. Các gói GBDT khác nhau không chênh nhiều.
- **Cây không-nhớ (ODT):** cây độ sâu d dùng **cùng một đặc trưng và cùng ngưỡng cho mọi nút ở cùng độ sâu**, nên cây tương đương một bảng 2^d ô tra cứu. ODT yếu hơn cây thường, nhưng trong ensemble ít quá khớp, kết hợp tốt với boosting (CatBoost), và suy luận rất hiệu quả (d phép chia nhị phân song song, rồi tra bảng).
- **Cây khả vi:** làm "mềm" hàm chia của nút nội để định tuyến khả vi (Kontschieder 2015, Yang 2018 deep neural decision trees, Lay 2018). Nhược điểm các cách làm trước: REINFORCE, Gumbel-softmax cần thời gian huấn luyện dài; soft binning của Yang liệt kê mọi quyết định có thể.
- **Entmax (Peters 2019):** biến đổi chiếu điểm số thành phân phối rời rạc, tổng quát hóa softmax (α = 1) và sparsemax (α = 2); α-entmax tạo phân phối **thưa** (nhiều xác suất đúng bằng 0) nhưng mượt hơn sparsemax, thích hợp tối ưu hóa gradient. Ý nghĩa với cây: học chia theo một tập đặc trưng nhỏ, có thể chỉ một đặc trưng như cây cổ điển.
- **Kiến trúc nhiều lớp không khả vi:** Deep Forest (Zhou & Feng 2017), mGBDT (Feng 2018): xếp chồng rừng/GBDT mà không huấn luyện đầu cuối; kết quả cải thiện chỉ biên.
- **DNN riêng cho bảng trước đó** (TabNN của Ke 2018, regularization learning networks) hầu như không so sánh với GBDT được tinh chỉnh kỹ.

## 3. Phương pháp

### 3.1. ODT khả vi và lớp NODE
Một lớp NODE gồm **m cây ODT khả vi cùng độ sâu d**, nhận chung vector x ∈ ℝⁿ.
- ODT cổ điển: h(x) = R[1(f₁(x) − b₁), …, 1(f_d(x) − b_d)], với R ∈ ℝ^{2×…×2} (d chiều) là tensor phản hồi, f_i là đặc trưng chia, b_i là ngưỡng, 1(·) là hàm Heaviside.
- **Chọn đặc trưng mềm:** f̂_i(x) = Σⱼ xⱼ · entmax_α(F_ij), với ma trận chọn đặc trưng F ∈ ℝ^{d×n} học được; α = 1,5.
- **So sánh mềm:** thay Heaviside bằng entmax hai lớp σ_α(x) = entmax_α([x, 0]) và chia thang: c_i(x) = σ_α((f̂_i(x) − b_i)/τ_i), với b_i (ngưỡng) và τ_i (thang) học được vì các đặc trưng có thang khác nhau.
- **Tensor lựa chọn:** C(x) = [c₁; 1−c₁] ⊗ [c₂; 1−c₂] ⊗ … ⊗ [c_d; 1−c_d] (tích ngoài), cùng kích thước R.
- **Đầu ra cây:** ĥ(x) = Σ_{i₁…i_d ∈ {0,1}^d} R_{i₁…i_d} · C_{i₁…i_d}(x). Bằng đúng cây cổ điển khi chọn đặc trưng và ngưỡng đều đạt trạng thái one-hot.
- Đầu ra lớp: nối đầu ra của m cây. Với phân loại, đầu ra nhiều chiều ĥ(x) ∈ ℝ^{|C|}.

### 3.2. Kiến trúc nhiều lớp kiểu DenseNet
k lớp NODE xếp liên tiếp, **mỗi lớp nhận đầu vào là phép nối (concatenation) của đặc trưng gốc và đầu ra của mọi lớp trước**. Một cây ở lớp i có thể dựa vào chuỗi đến i−1 lớp đầu ra làm đặc trưng, nên học được cả luật nông lẫn sâu. Dự đoán cuối = **trung bình mọi cây của mọi lớp**. Đầu ra mỗi cây có chiều l tùy ý (siêu tham số, thường 1–3); chỉ |C| tọa độ đầu cho phân loại, hoặc tọa độ đầu cho hồi quy, được dùng khi lấy trung bình.

### 3.3. Huấn luyện và suy luận
- **Tiền xử lý:** biến đổi quantile (QuantileTransformer) đưa mỗi đặc trưng về phân phối chuẩn, quan trọng cho huấn luyện ổn định và hội tụ nhanh.
- **Khởi tạo data-aware (Mishkin & Matas):** F_ij ~ U(0,1); ngưỡng b khởi tạo bằng giá trị đặc trưng quan sát trong batch đầu; τ_i sao cho mọi mẫu của batch đầu nằm trong vùng tuyến tính của σ_α (để có gradient khác 0); R ~ N(0,1).
- **Tối ưu:** SGD mini-batch đầu cuối trên mọi tham số (F, b, τ, R); loss cross-entropy / MSE (bất kỳ loss khả vi nào dùng được); bộ tối ưu Quasi-Hyperbolic Adam (Ma & Yarats 2018) với tham số khuyến nghị; trung bình tham số qua c = 5 checkpoint liên tiếp (Izmailov 2018); dừng sớm theo tập kiểm chứng; learning rate 10⁻³.
- **Suy luận:** sau huấn luyện, có thể tính trước bộ chọn entmax và lưu dạng vector thưa (định dạng COO) để suy luận nhanh.

## 4. Thiết lập thí nghiệm

- **Sáu bộ dữ liệu:**

| Bộ | Train | Test | Tác vụ | Thước đo | Ghi chú |
|---|---|---|---|---|---|
| Epsilon | 400K | 100K | Nhị phân | Lỗi phân loại | PASCAL Challenge 2008, nhiều chiều |
| YearPrediction | 463K | 51,6K | Hồi quy | MSE | Million Song Dataset, năm phát hành 1922–2011 |
| Higgs | 10,5M | 500K | Nhị phân | Lỗi | UCI ML Higgs |
| Microsoft | 723K | 241K | Xếp hạng (pointwise → hồi quy) | MSE | MSLR-WEB10K, 136 đặc trưng |
| Yahoo | 544K | 165K | Xếp hạng (pointwise) | MSE | Yahoo LETOR |
| Click | 800K | 200K | Nhị phân | Lỗi | KDD Cup 2012, lấy mẫu 500.000 dương + 500.000 âm; hạng mục dùng Leave-One-Out |

  Mỗi bộ cố định chia train/val/test (lấy 20% train làm validation; Yahoo dùng chia gốc).
- **Đối thủ:** CatBoost, XGBoost, FCNN (mạng nơ-ron kết nối đầy đủ), và (một phần) mGBDT và Deep Forest.
- **Hai chế độ:** (1) **siêu tham số mặc định:** CatBoost/XGBoost chỉ chọn số cây (đến 2048) theo validation; NODE mặc định một lớp, 2048 cây độ sâu 6 (kế thừa từ CatBoost); không so FCNN vì cần tinh chỉnh nhiều; (2) **siêu tham số tinh chỉnh:** TPE (Hyperopt) 50 bước cho CatBoost, XGBoost, FCNN; NODE tìm lưới trên số lớp {2, 4, 8}, tổng số cây {1024, 2048}, độ sâu {6, 8}, chiều đầu ra cây {2, 3}, batch tối đa vừa bộ nhớ GPU. FCNN: 2–7 lớp, 128–1024 đơn vị, dropout [0, 0,5], Adam, batch 1024.
- **Báo cáo:** trung bình ± độ lệch chuẩn trên 10 seed.

## 5. Kết quả

### 5.1. Siêu tham số mặc định (Bảng 1)

| Mô hình | Epsilon↓ | YearPred↓ | Higgs↓ | Microsoft↓ | Yahoo↓ | Click↓ |
|---|---|---|---|---|---|---|
| CatBoost | 0,1119 ± 2e−4 | 80,68 ± 0,04 | 0,2434 ± 2e−4 | 0,5587 ± 2e−4 | 0,5781 ± 3e−4 | 0,3438 ± 1e−3 |
| XGBoost | 0,1144 | 81,11 | 0,2600 | 0,5637 | 0,5756 | 0,3461 |
| **NODE** | **0,1043 ± 4e−4** | **77,43 ± 0,09** | **0,2412 ± 5e−4** | **0,5584 ± 3e−4** | **0,5666 ± 5e−4** | **0,3309 ± 3e−4** |

NODE (một lớp, 2048 cây, độ sâu 6, vẫn huấn luyện đầu cuối) **thắng cả CatBoost và XGBoost trên cả sáu bộ**. (Lỗi/MSE: thấp hơn là tốt hơn.)

### 5.2. Siêu tham số tinh chỉnh (Bảng 2)

| Mô hình | Epsilon↓ | YearPred↓ | Higgs↓ | Microsoft↓ | Yahoo↓ | Click↓ |
|---|---|---|---|---|---|---|
| CatBoost | 0,1113 ± 4e−4 | 79,67 ± 0,12 | 0,2378 ± 1e−4 | 0,5565 ± 2e−4 | 0,5632 ± 3e−4 | 0,3401 ± 2e−3 |
| XGBoost | 0,1112 ± 6e−4 | 78,53 ± 0,09 | 0,2328 ± 3e−4 | **0,5544 ± 1e−4** | **0,5420 ± 4e−4** | 0,3334 ± 2e−3 |
| FCNN | 0,1041 ± 2e−4 | 79,99 ± 0,47 | 0,2140 ± 2e−4 | 0,5608 ± 4e−4 | 0,5773 ± 1e−3 | 0,3325 ± 2e−3 |
| **NODE** | **0,1034 ± 3e−4** | **76,21 ± 0,12** | **0,2101 ± 5e−4** | 0,5570 ± 2e−4 | 0,5692 ± 2e−4 | **0,3312 ± 2e−3** |
| mGBDT | OOM | 80,67 | OOM | OOM | OOM | OOM |
| Deep Forest | 0,1179 | – | 0,2391 | – | – | 0,3333 |

- NODE thắng ở **4/6** bộ (Epsilon, YearPrediction, Higgs, Click); **XGBoost tinh chỉnh thắng Yahoo và Microsoft**. Giải thích của tác giả: ở Yahoo XGBoost vượt CatBoost rất xa nên cây không-nhớ có thể là thiên kiến quy nạp không phù hợp; hướng tương lai là mở rộng NODE sang cây không "oblivious".
- FCNN thắng GBDT ở một số bộ (Epsilon, Higgs), thua ở bộ khác (YearPrediction, Yahoo, Microsoft); còn NODE là "công cụ phổ quát" cho hiệu năng cao nhất ở đa số.
- mGBDT và Deep Forest chỉ so sánh được một phần (không công bố mã, không scale: nhiều lỗi hết bộ nhớ, Deep Forest chỉ có cho phân loại); các GBDT được tinh chỉnh đúng cách vượt cả hai.

## 6. Phân tích

### 6.1. Hàm chọn (Bảng 3)
Thử bốn hàm cho cả chọn đặc trưng chia và định tuyến cây, với 1, 2, 4, 8 lớp:

| Bộ | Hàm | 1 lớp | 2 lớp | 4 lớp | 8 lớp |
|---|---|---|---|---|---|
| YearPrediction (MSE) | softmax | 78,41 | 77,61 | 77,58 | 77,47 |
| | Gumbel-softmax | 79,39 | 79,31 | 79,69 | 80,49 |
| | sparsemax | 78,13 | 76,81 | 76,60 | 76,31 |
| | **entmax** | 77,43 | 77,05 | 76,21 | **76,17** |
| Epsilon (lỗi) | softmax | 0,1045 | 0,1041 | 0,1034 | 0,1036 |
| | Gumbel-softmax | 0,1979 | 0,2884 | 0,2908 | 0,3081 |
| | sparsemax | 0,1083 | 0,1052 | 0,1058 | 0,1058 |
| | **entmax** | 0,1043 | **0,1031** | 0,1033 | 0,1036 |

- Entmax (α = 1,5) tốt nhất hoặc ngang tốt nhất trên cả hai bộ; softmax và sparsemax không phổ quát (sparsemax tốt hơn softmax trên YearPrediction nhưng kém trên Epsilon).
- **Gumbel-softmax không học được kiến trúc sâu** (lỗi tăng khi thêm lớp), có thể vì tính ngẫu nhiên khiến đầu ra lớp trước quá nhiễu làm đặc trưng cho lớp sau.
- Độ sâu giúp ở YearPrediction (77,43 → 76,17 với entmax) nhưng ít ở Epsilon.

### 6.2. Độ quan trọng đặc trưng (Hình 3, Higgs)
Dùng độ quan trọng hoán vị (Breiman 2001) trên 10.000 mẫu Higgs cho đặc trưng gốc và đặc trưng học ở từng lớp: đặc trưng ở lớp đầu được dùng nhiều nhất và quan trọng giảm theo độ sâu; nhưng **cây ở lớp sâu đóng góp nhiều hơn vào đầu ra cuối**. Có tương quan nghịch giữa độ quan trọng đặc trưng và đóng góp vào phản hồi: **lớp đầu tạo đặc trưng giàu thông tin, lớp sau dùng chúng để dự đoán chính xác**.

### 6.3. Thời gian chạy (Bảng 4, YearPrediction, 1024 cây độ sâu 6, trung bình 5 lần)

| Cấu hình | Huấn luyện | Suy luận |
|---|---|---|
| NODE 8 lớp, GPU 1080Ti | 7 phút 42 giây | 8,56 giây |
| XGBoost, CPU Xeon 28 nhân | 5 phút 39 giây | 5,94 giây |
| XGBoost, GPU 1080Ti | 1 phút 13 giây | 4,45 giây |
| CatBoost, CPU Xeon | 41 giây | 4,62 giây |

NODE chậm hơn nhưng cùng bậc với GBDT tối ưu cao, dù chỉ viết bằng PyTorch thuần (không có kernel tùy biến). (GPU rẻ hơn khoảng một nửa CPU Xeon dùng so sánh.)

## 7. Hạn chế

Tác giả nêu: NODE nên được mở rộng sang cây không "oblivious" (Yahoo, Microsoft); hướng phát triển là đưa lớp NODE vào pipeline đa phương thức huấn luyện bằng lan truyền ngược (thay CNN cho ảnh, RNN cho chuỗi).

Nhận xét thêm của mình:
1. Chỉ **6 bộ dữ liệu lớn** (từ 463K mẫu đến 10,5M); tất cả là phân loại nhị phân hoặc hồi quy, không có dữ liệu nhỏ; kết luận "thắng GBDT" chưa chắc đúng ở dữ liệu nhỏ hoặc có hạng mục nhiều (các benchmark độc lập sau này như bài 30, 33, 29 cho kết quả lẫn lộn).
2. Chênh lệch nhiều chỗ nhỏ so với độ lệch chuẩn (ví dụ Microsoft 0,5584 so với 0,5587); và tinh chỉnh GBDT chỉ 50 bước TPE, GBDT tối đa 2048 cây.
3. Bài đánh giá trên lỗi/MSE, không xét AUC hay hiệu chỉnh; đã nhìn từng mô hình chạy một bộ dữ liệu và seed.
4. NODE là mô hình huấn luyện riêng cho từng bộ dữ liệu (dù nhanh), không có khả năng ICL/chuyển giao.
5. Đề tài nhiễu/mất cân bằng: không đề cập; CatBoost/ODT từ lâu được cho là bền với quá khớp nhưng không có thí nghiệm nhiễu nhãn.

## 8. Liên hệ với bài Nature và đề tài

- **Prior dựa trên cây:** bài Nature cho thấy prior của TabPFNv2 gồm cả cây (cùng SCM); Mitra (bài 9) kết hợp prior cây trực tiếp (DSRF). NODE là minh chứng sớm rằng *thiên kiến quy nạp kiểu cây* (chia theo ngưỡng, chọn đặc trưng thưa) cực kỳ hữu ích cho bảng, và có thể tích hợp vào mạng sâu.
- **Giải thích vì sao GBDT mạnh:** kết hợp với Grinsztajn (bài 29): dữ liệu bảng thích hợp hàm không trơn, chọn đặc trưng thưa; entmax/ODT là một cách tái tạo điều đó trong mạng nơ-ron.
- **Thí nghiệm cuối kỳ:** NODE ít cần thiết hơn so với TabM/FT-T/XGBoost, nhưng nếu muốn một "mạng cây" thì đây là đại diện.

## 9. Từ khóa cần nhớ

- **ODT (oblivious decision tree):** cùng đặc trưng, cùng ngưỡng ở mỗi tầng; bảng 2^d ô.
- **Entmax (α = 1,5):** chọn đặc trưng chia thưa và so sánh mềm; tốt hơn softmax, sparsemax, Gumbel-softmax.
- **NODE:** m cây ODT khả vi mỗi lớp; DenseNet-nối chặt; trung bình mọi cây mọi lớp; 2–8 lớp, ≤ 2048 cây.
- **Số liệu:** 6 bộ dữ liệu (400K đến 10,5M mẫu); mặc định thắng 6/6, tinh chỉnh thắng 4/6 (XGBoost thắng Yahoo, Microsoft); huấn luyện 7 phút 42 giây so với 41 giây (CatBoost).
- **Phát hiện:** lớp đầu tạo đặc trưng, lớp sau dự đoán; Gumbel-softmax không học được mạng sâu.
