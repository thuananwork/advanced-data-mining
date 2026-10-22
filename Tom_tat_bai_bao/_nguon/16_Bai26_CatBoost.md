# Bài 26. CatBoost: unbiased boosting with categorical features

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2018 (theo danh mục đề tài; bài này được các tài liệu khác trích dẫn là NeurIPS'18, trang 6639–6649); PDF là arXiv 1706.09516v5 (20/1/2019), đầu trang ghi "Preprint. Work in progress" |
| Tác giả | Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (Yandex; MIPT) |
| Mã nguồn / dữ liệu | github.com/catboost/catboost; mã thí nghiệm: github.com/catboost/benchmarks/tree/master/quality_benchmarks; có cài đặt GPU hiệu quả |
| Vị trí trong đề tài | Baseline cây tăng cường được dùng nhiều nhất trong các bài TFM (đứng đầu TabArena khi chỉ tinh chỉnh, bài 5; **top 2 của TabZilla**, bài 33). Đối thủ chính của TFM trong thí nghiệm final-term. |

## 1. Bài này làm gì trong một đoạn

Tác giả chỉ ra rằng **mọi cài đặt tăng cường gradient hiện có** gặp một vấn đề thống kê chưa được nhận ra: **dịch chuyển dự đoán (prediction shift)**, một dạng rò rỉ nhãn (target leakage). Mô hình sau vài bước tăng cường đã dùng nhãn của mọi mẫu huấn luyện để tính gradient, nên phân phối của F(x_k) | x_k đối với mẫu huấn luyện lệch so với F(x) | x đối với mẫu kiểm tra. Một vấn đề tương tự xảy ra với **thống kê mục tiêu (target statistics, TS)** dùng để mã hóa đặc trưng hạng mục. Giải pháp là một **nguyên lý sắp thứ tự (ordering principle)**: (1) **ordered boosting**, thay thế thuật toán tăng cường cổ điển, và (2) thuật toán xử lý đặc trưng hạng mục bằng **ordered TS**. Cả hai được cài đặt trong thư viện mã nguồn mở **CatBoost** ("Categorical Boosting"), vượt XGBoost và LightGBM trên 9 bộ dữ liệu phổ biến.

## 2. Vấn đề và động cơ

- Tăng cường gradient đã nhiều năm là phương pháp chính cho các bài toán có đặc trưng không đồng nhất, dữ liệu nhiễu và phụ thuộc phức tạp (tìm kiếm web, hệ gợi ý, dự báo thời tiết). Về bản chất là xây ensemble bằng hạ gradient trong không gian hàm: F^t = F^{t−1} + α h^t, với h^t ≈ −g^t (xấp xỉ bình phương tối thiểu gradient âm của mất mát), h là cây quyết định nhị phân.
- **Đặc trưng hạng mục** (tập giá trị rời rạc không so sánh được): one-hot mã hóa gây bùng nổ số đặc trưng khi cardinality cao (ví dụ "user ID"). Cách thông dụng thay thế: gom nhóm theo TS (kỳ vọng nhãn theo hạng mục), hoặc dùng TS làm đặc trưng số (Micci-Barreca). LightGBM chuyển hạng mục thành thống kê gradient ở mỗi bước (tốn thời gian và bộ nhớ, và gom các hạng mục đuôi vào một cụm nên mất thông tin; chính tác giả LightGBM cũng cho rằng với cardinality cao nên chuyển sang số). TS chỉ cần lưu một số mỗi hạng mục, nên là cách hiệu quả nhất với ít mất thông tin nhất; bài tập trung vào cách tính TS.

## 3. Đặc trưng hạng mục và rò rỉ nhãn (mục 3)

TS ước lượng x̂ᵢ_k ≈ E(y | xⁱ = xⁱ_k). Các phiên bản:
- **Greedy TS:** trung bình nhãn của các mẫu cùng hạng mục, làm trơn bằng prior p với tham số a: x̂ = (Σ_j 1{xⁱ_j = xⁱ_k}·y_j + a·p) / (Σ_j 1{xⁱ_j = xⁱ_k} + a). Vấn đề: x̂ⁱ_k được tính bằng chính nhãn y_k của mẫu k, nên gây **dịch chuyển có điều kiện** (phân phối x̂|y khác nhau giữa huấn luyện và kiểm tra). Ví dụ cực đoan: đặc trưng hạng mục mà mọi giá trị đều duy nhất và P(y = 1 | xⁱ = A) = 0,5; khi huấn luyện, x̂ = (y_k + a p)/(1 + a), một lần tách ở ngưỡng t = (0,5 + a p)/(1 + a) phân loại hoàn hảo tập huấn luyện, nhưng mọi mẫu kiểm tra có x̂ = p, nên mô hình dự đoán toàn 0 hoặc toàn 1, độ chính xác 0,5.
- Hai tính chất mong muốn: **P1** E(x̂ⁱ | y = v) = E(x̂ⁱ_k | y_k = v); **P2** dùng hiệu quả toàn bộ dữ liệu huấn luyện cho cả tính TS lẫn học mô hình.
- Cách tránh dịch chuyển: tính TS của x_k trên tập D_k ⊂ D \ {x_k}.
  - **Holdout TS:** chia D thành hai phần, một phần để tính TS, phần còn lại để huấn luyện. Thỏa P1 nhưng vi phạm P2 (giảm dữ liệu).
  - **Leave-one-out TS:** D_k = D \ x_k cho mẫu huấn luyện, D_k = D cho mẫu kiểm tra. Nghe có vẻ tốt nhưng **vẫn rò rỉ**: với đặc trưng hằng (mọi mẫu cùng giá trị A), x̂ = (n⁺ − y_k + a p)/(n − 1 + a) tiết lộ y_k và cho phép phân loại hoàn hảo bằng một lần tách.
- **Ordered TS (CatBoost):** lấy cảm hứng từ học trực tuyến (mẫu đến tuần tự theo thời gian). Tạo "thời gian giả" bằng **một hoán vị ngẫu nhiên σ** của các mẫu huấn luyện; với mỗi mẫu, chỉ dùng "lịch sử" đứng trước nó: D_k = {x_j : σ(j) < σ(k)}; mẫu kiểm tra dùng D_k = D. Thỏa P1 và P2. Nhược điểm: các mẫu đầu hoán vị có TS phương sai cao, nên CatBoost dùng **các hoán vị khác nhau ở các bước tăng cường khác nhau**.

## 4. Dịch chuyển dự đoán và ordered boosting (mục 4)

### 4.1. Prediction shift
Trong thực tế, kỳ vọng ở phương trình (2) không biết, nên xấp xỉ bằng cùng tập D: h^t = argmin (1/n) Σ_k (−g^t(x_k, y_k) − h(x_k))². Chuỗi dịch chuyển: (1) phân phối có điều kiện của gradient g^t(x_k, y_k) | x_k ở mẫu huấn luyện lệch so với mẫu kiểm tra; (2) h^t lệch so với nghiệm lý tưởng; (3) ảnh hưởng khả năng tổng quát hóa của F^t. Nguyên nhân: gradient mỗi bước được ước lượng bằng nhãn của chính các điểm mà F^{t−1} đã được xây trên đó. Các cách làm trước đó như iterated bagging (Breiman) và lấy mẫu con mỗi vòng (Friedman) chỉ giảm nhẹ, không loại bỏ vấn đề (Phụ lục E chứng minh residual out-of-bag vẫn bị dịch chuyển).

**Định lý 1** (trường hợp hồi quy bình phương, m = 2 đặc trưng Bernoulli(1/2) độc lập, y = c₁x₁ + c₂x₂, N = 2 bước với gốc cây độ sâu 1, α = 1):
1. Nếu dùng hai mẫu độc lập D₁, D₂ cỡ n cho h¹ và h²: E F²(x) = f\*(x) + O(1/2ⁿ) (ước lượng không chệch).
2. Nếu dùng cùng một tập D cho cả hai bước: **E F²(x) = f\*(x) − (1/(n − 1))·c₂(x₂ − ½) + O(1/2ⁿ)**, tức **chệch tỉ lệ nghịch với n** và phụ thuộc vào hệ số hàm đích (ở đây tỉ lệ với c₂); chệch nghiêm trọng hơn với bộ nhỏ.

### 4.2. Ordered boosting
Với dữ liệu vô hạn, chỉ cần lấy mẫu độc lập D^t mỗi bước. Với dữ liệu hữu hạn, để residual r^{I−1}(x_k, y_k) không dịch chuyển thì F^{I−1} phải được huấn luyện **không dùng x_k**. Ý tưởng: duy trì n mô hình hỗ trợ M₁..M_n, M_i chỉ học từ i mẫu đầu của hoán vị σ; để tính residual của mẫu thứ j dùng M_{j−1} (Thuật toán 1). Vì cần n mô hình nên không khả thi trực tiếp; CatBoost cài đặt một biến thể thực tế dựa trên cây quyết định. **Kết hợp hạng mục + ordered boosting:** phải dùng **cùng hoán vị σ_cat = σ_boost** để nhãn y_i không được dùng cho M_i (cả TS lẫn gradient); đảm bảo lý thuyết ở Phụ lục F, bằng chứng thực nghiệm ở Phụ lục G.

## 5. Cài đặt thực tế (mục 5)

- **Hai chế độ:** *Plain* (GBDT chuẩn kèm ordered TS tích hợp) và *Ordered* (biến thể hiệu quả của Thuật toán 1).
- **Nhiều hoán vị:** sinh s + 1 hoán vị độc lập; σ₁..σ_s dùng để đánh giá tách (cấu trúc cây), σ₀ để chọn giá trị lá. Dùng một hoán vị thì phương sai dự đoán cao với các mẫu đầu, nên dùng nhiều.
- **Cây oblivious (cây bảng quyết định):** cùng một tiêu chí tách cho cả một tầng; cây cân bằng, ít quá khớp, và **suy luận nhanh hơn nhiều**. Thuật toán 2 xây cây: ở mỗi bước chọn tách tốt nhất theo **độ tương tự cos** giữa vector gradient xấp xỉ (dùng gradient chỉ dựa trên các mẫu đứng trước trong hoán vị) và giá trị lá; duy trì các mô hình hỗ trợ M_{r,j}; cùng một cấu trúc cây được thêm vào mọi M nhưng với giá trị lá khác nhau.
- **Giảm độ phức tạp:** thay vì O(s·n²) giá trị M_{r,j}(i), chỉ giữ M′_{r,j}(i) := M_{r,2^j}(i) cho j = 1..⌈log₂ n⌉, tức O(s·n); độ phức tạp tiệm cận bằng GBDT chuẩn với ordered TS (Bảng 1: CalcGradient O(s·n), Build T O(|C|·n), tính giá trị lá O(n), cập nhật M O(s·n), tính ordered TS O(N_TS,t·n)); so với các loại TS khác, ordered TS làm chậm s lần ba thủ tục (CalcGradient, cập nhật M, tính TS).
- **Tổ hợp đặc trưng hạng mục:** dùng tổ hợp (nối) các đặc trưng hạng mục làm đặc trưng hạng mục mới để bắt phụ thuộc bậc cao (ví dụ user ID + chủ đề quảng cáo). Số tổ hợp tăng theo hàm mũ nên xây theo cách tham lam: với mỗi lần tách, kết hợp các đặc trưng hạng mục (và tổ hợp) đã dùng trong cây hiện tại với mọi đặc trưng hạng mục còn lại; đổi sang TS ngay.
- **Chi tiết khác:** lấy mẫu con theo Bayesian bootstrap (trọng số w_i = a_i^t, nhân với gradient); loại bỏ phần đầu hoán vị khi tính cosine để giảm phương sai.
- **Giá trị lá cuối** tính bằng quy trình tăng cường chuẩn, dùng σ₀. Khi suy luận trên mẫu mới, TS tính từ toàn bộ tập huấn luyện.

## 6. Thực nghiệm (mục 6, Phụ lục C, D, G)

### 6.1. Thiết lập
- **9 bộ dữ liệu (Bảng 7):** Adult (dự đoán thu nhập > 50K, điều tra dân số Mỹ 1994), Amazon (Kaggle Amazon Employee), Click (KDD Cup 2012 track 2, lấy 1% mẫu và hạ mẫu lớp đa số về tỉ lệ khoảng 5:1), Epsilon (nhị phân; phiên bản nhị phân hóa), Appetency / Churn / Upselling (KDD Cup 2009), Internet (KDD internet usage, đa lớp được nhị phân hóa bằng cách gán lớp đa số là dương, lớp còn lại âm), Kick (Kaggle "Don't Get Kicked!").
- **Giao thức:** chia ngẫu nhiên 80/20 train/test; CV 5 fold phân tầng theo lớp trên tập train để tinh chỉnh; thiếu giá trị: hạng mục coi như hạng mục đặc biệt, đặc trưng số thay bằng 0 kèm đặc trưng giả nhị phân. XGBoost, LightGBM và cấu hình "raw" của CatBoost dùng cùng bước tiền xử lý ordered TS (trên một hoán vị ngẫu nhiên độc lập; giá trị TS được coi là đặc trưng số). **Tinh chỉnh 50 bước TPE (Hyperopt) tối thiểu hóa logloss**, ví dụ eta log-đều [e⁻⁷, 1], max_depth [2, 10], colsample [0,5, 1], min_child_weight, alpha/lambda/gamma (hỗn hợp 0,5 tại 0 + 0,5 log-đều) cho XGBoost; learning_rate, num_leaves, feature_fraction, bagging_fraction, min_sum_hessian, min_data_in_leaf, lambda_l1/l2 cho LightGBM; learning_rate, random_strength, one_hot_max_size, l2_leaf_reg, bagging_temperature, gradient_iterations cho CatBoost. Riêng **Epsilon** dùng tham số mặc định (thời gian chạy quá lớn) và chỉ tinh chỉnh số cây.
- **Thước đo:** logloss và zero-one loss.

### 6.2. Kết quả chính (Bảng 2: CatBoost tuyệt đối; baseline hiển thị tăng tương đối)

| Bộ | CatBoost logloss / 0-1 | LightGBM (tăng tương đối) | XGBoost (tăng tương đối) |
|---|---|---|---|
| Adult | 0,270 / 0,127 | +2,4% / +1,9% | +2,2% / +1,0% |
| Amazon | 0,139 / 0,044 | +17% / +21% | +17% / +21% |
| Click | 0,392 / 0,156 | +1,2% / +1,2% | +1,2% / +1,2% |
| Epsilon | 0,265 / 0,109 | +1,5% / +4,1% | +11% / +12% |
| Appetency | 0,072 / 0,018 | +0,4% / +0,2% | +0,4% / +0,7% |
| Churn | 0,232 / 0,072 | +0,1% / +0,6% | +0,5% / +1,6% |
| Internet | 0,209 / 0,094 | +6,8% / +8,6% | +7,9% / +8,0% |
| Upselling | 0,166 / 0,049 | +0,3% / +0,1% | +0,04% / +0,3% |
| Kick | 0,286 / 0,095 | +3,5% / +4,4% | +3,2% / +4,1% |

- CatBoost (chế độ Ordered) **vượt cả hai trên cả 9 bộ**; cải thiện có ý nghĩa thống kê (kiểm định t một phía ghép cặp, p ≪ 0,01) trừ ba bộ Appetency, Churn, Upselling.
- **Cấu hình "raw" gần GBDT cổ điển** (GPU, Plain, border_count 255, gradient_iterations 1, random_strength 0, depth 6, chỉ tinh chỉnh sample_rate, learning_rate, l2_leaf_reg; hạng mục đã mã hóa ordered TS cho mọi thuật toán, khác duy nhất là dùng cây oblivious) chênh nhẹ so với baseline: tốt hơn 0,2% về zero-one loss, kém 0,2% về logloss. Chứng tỏ cài đặt GBDT của CatBoost là nền tảng hợp lý (cây oblivious còn tăng tốc suy luận).
- **Thời gian (Epsilon, 16 luồng; Bảng 6):** CatBoost Plain và LightGBM nhanh nhất; **Ordered chậm hơn khoảng 1,7 lần**.

### 6.3. Phân tích
- **Ordered vs Plain (Bảng 3, Hình 2):** Ordered đặc biệt có lợi trên **bộ nhỏ** (Adult, Internet < 40.000 mẫu huấn luyện; ví dụ Plain kém Ordered +1,1% logloss ở Adult và +3,9% ở Internet): phù hợp với Định lý 1 (chệch lớn hơn khi n nhỏ). Thí nghiệm lọc ngẫu nhiên cho thấy khi bộ nhỏ lại, Plain càng kém so với Ordered. Ở cấu hình raw (Phụ lục G), lợi thế của Ordered vẫn còn (0,5% logloss khi σ_boost độc lập với σ_cat; 0,6% logloss và 0,9% zero-one khi σ_boost = σ_cat), chứng thực cả tính đúng đắn lý thuyết của việc dùng chung hoán vị.
- **So sánh TS (Bảng 4, thay đổi tương đối so với ordered TS):** Ordered TS vượt rõ rệt các cách khác. Holdout là tốt nhất trong số baseline ở đa số bộ (không dịch chuyển có điều kiện) nhưng kém vì dùng dữ liệu không hiệu quả. Leave-one-out thường tốt hơn Greedy nhưng có thể rất tệ (ví dụ trên Adult: +5,5% / +3,7%, hay Internet +3,9%); Greedy có thể rất tệ: Amazon +40% / +32%, Internet +57% / +50%, Click +13%/+6,7%. Lý do: Greedy chịu ảnh hưởng hạng mục tần suất thấp, LOO chịu cả hạng mục tần suất cao (như mọi đặc trưng của Adult).
- **Tổ hợp đặc trưng:** tăng c_max (số đặc trưng được kết hợp) từ 1 lên 2 giảm logloss trung bình **1,86%** (tối đa 11,3%); lên 3 giảm 2,04%; tăng thêm hầu như không đổi.
- **Số hoán vị s:** tăng s giảm nhẹ logloss (3 hoán vị: −0,19%; 9 hoán vị: −0,38% so với s = 1).
- GPU: CatBoost có cài đặt GPU rất hiệu quả (so sánh thời gian nằm ngoài phạm vi, xem GitHub).

## 7. Hạn chế (tác giả nêu và suy ra)
1. Phân tích lý thuyết (Định lý 1) chỉ cho trường hợp **hồi quy bình phương, hai đặc trưng Bernoulli, gốc cây độ sâu 1**; phần còn lại dựa vào thực nghiệm. Tác giả giới hạn mục 4 ở trường hợp này (các phương pháp ở mục 4.2 nói áp dụng được cho các tác vụ khác).
2. **Chậm hơn 1,7 lần** ở chế độ Ordered; tốn thêm bộ nhớ cho các mô hình hỗ trợ.
3. Chỉ **9 bộ dữ liệu**, toàn bộ **phân loại nhị phân** (đa lớp nhị phân hóa ở Internet), không có hồi quy, không có bộ dữ liệu nhỏ chuẩn kiểu OpenML; một số bộ (Click, Epsilon) được mẫu con hoặc dùng tham số mặc định; lợi thế trên ba bộ (Appetency, Churn, Upselling) không có ý nghĩa thống kê.
4. Tinh chỉnh 50 bước TPE cho mọi thuật toán; các thuật toán có số siêu tham số khác nhau nên mức tinh chỉnh không hoàn toàn tương đương.
5. Phụ thuộc vào ngẫu nhiên của hoán vị (nhiều hoán vị chỉ giảm nhẹ phương sai).
6. Không đánh giá nhãn nhiễu, mất cân bằng lớp hay hiệu chuẩn như yếu tố điều khiển (mất cân bằng chỉ xuất hiện ngầm trong bộ Click và Kick).

## 8. Liên hệ với bài Nature và đề tài
- CatBoost là baseline cây mạnh nhất trong hầu hết benchmark tổng hợp (hạng nhất ở bài 33 và bài 5 khi chỉ tinh chỉnh), nên mọi TFM phải so với nó; ordered boosting giải thích vì sao nó **tốt trên dữ liệu nhỏ**, chính là vùng mà TFM cạnh tranh.
- Mã hóa hạng mục bằng ordered TS cho thấy cách **tránh rò rỉ nhãn** — liên quan đến thí nghiệm nhãn nhiễu: nếu nhãn sai, TS (và gradient) bị nhiễm nhãn sai; câu hỏi mở là CatBoost có bền hơn XGBoost/LightGBM khi nhãn nhiễu, điều bài không đo.
- Trong thí nghiệm final-term, cần dùng CatBoost với **một cấu hình mặc định và một cấu hình tinh chỉnh** (đúng quy trình TabArena), lưu ý hạng mục nên để CatBoost xử lý nguyên bản.

## 9. Từ khóa cần nhớ
- **Prediction shift** = rò rỉ nhãn trong gradient; **conditional shift** của TS; **Định lý 1:** chệch −c₂(x₂ − ½)/(n − 1).
- **Ordered boosting** (n mô hình hỗ trợ M_i theo hoán vị σ), **ordered TS**, **σ_cat = σ_boost**, **s + 1 hoán vị**.
- **Cây oblivious**, **tổ hợp đặc trưng hạng mục**, **Bayesian bootstrap**; chế độ **Plain/Ordered** (chậm 1,7×).
- 9 bộ dữ liệu; CatBoost thắng cả 9 bộ; tinh chỉnh **50 bước TPE** tối thiểu hóa logloss.
