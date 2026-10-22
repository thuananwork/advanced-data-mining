# Bài 28. XGBoost: A Scalable Tree Boosting System

| Mục | Thông tin |
|---|---|
| Nơi đăng | KDD '16 (ACM SIGKDD), San Francisco, 13–17/8/2016, trang 785–794; PDF là arXiv 1603.02754v3 (10/6/2016) |
| Tác giả | Tianqi Chen, Carlos Guestrin (University of Washington) |
| Mã nguồn / dữ liệu | github.com/dmlc/xgboost (Python, R, Julia, tích hợp scikit-learn; bản phân tán chạy trên Hadoop, MPI, Sun Grid Engine, Flink, Spark; thư viện rabit cho allreduce) |
| Vị trí trong đề tài | **Baseline cây tăng cường gradient** kinh điển mà mọi TFM (TabPFN, TabICL, TabDPT, LoCalPFN) phải so sánh; đối thủ trực tiếp của thí nghiệm final-term. |

## 1. Bài này làm gì trong một đoạn

Tăng cường cây (gradient tree boosting) là kỹ thuật hiệu quả hàng đầu cho dữ liệu bảng. Bài mô tả **XGBoost**, một **hệ thống** tăng cường cây có khả năng mở rộng từ đầu đến cuối: một thuật toán mới cho dữ liệu **thưa** (sparsity-aware), một **weighted quantile sketch** có chứng minh lý thuyết để chia ngưỡng xấp xỉ khi mỗi mẫu có trọng số, và các tối ưu hệ thống (cấu trúc khối cột theo từng đặc trưng, truy cập nhận biết cache, tính toán ngoài bộ nhớ có nén và chia mảnh khối). Hệ thống chạy nhanh hơn các giải pháp phổ biến hơn **10 lần** trên một máy và mở rộng tới **hàng tỉ mẫu** trong môi trường phân tán hoặc hạn chế bộ nhớ.

## 2. Vấn đề và động cơ

- Hai nhân tố tạo nên ứng dụng học máy thành công: mô hình thống kê hiệu quả và **hệ thống học có khả năng mở rộng**.
- Tăng cường cây là kỹ thuật nổi bật trong nhiều ứng dụng: kết quả SOTA trên nhiều benchmark phân loại; biến thể LambdaMART cho bài toán xếp hạng; được đưa vào pipeline thực tế cho dự đoán tỉ lệ nhấp quảng cáo; là lựa chọn mặc định cho ensemble ở các cuộc thi như Netflix Prize.
- Bằng chứng về tác động: trong **29 giải pháp thắng cuộc** được Kaggle đăng blog năm 2015, **17** dùng XGBoost (8 chỉ dùng XGBoost, còn lại phối hợp với mạng nơ-ron); phương pháp phổ biến thứ hai là mạng nơ-ron sâu, 11 giải pháp. Ở KDDCup 2015, mọi đội trong top 10 đều dùng XGBoost, và các đội thắng báo cáo rằng ensemble chỉ hơn một XGBoost cấu hình tốt rất ít. Các bài toán đa dạng: dự báo doanh số cửa hàng, phân loại sự kiện vật lý năng lượng cao, phân loại văn bản web, hành vi khách hàng, phát hiện chuyển động, CTR, phân loại mã độc, phân loại sản phẩm, rủi ro bảo hiểm, tỉ lệ bỏ học khóa học trực tuyến.
- Các công trình tăng cường cây song song trước đó chưa khai thác **tính toán ngoài bộ nhớ (out-of-core)**, **truy cập nhận biết cache** và **nhận biết thưa**; một hệ thống đầu cuối kết hợp mọi yếu tố là điểm mới.

Đóng góp (mục 1): (1) hệ thống tăng cường cây đầu cuối có khả năng mở rộng cao; (2) weighted quantile sketch có cơ sở lý thuyết; (3) thuật toán nhận biết thưa cho học cây song song; (4) cấu trúc khối nhận biết cache cho học cây ngoài bộ nhớ. Ngoài ra còn cải tiến hàm mục tiêu có điều chuẩn.

## 3. Mô hình tăng cường cây (mục 2)

### 3.1. Hàm mục tiêu có điều chuẩn
- Tập dữ liệu n mẫu, m đặc trưng. Mô hình ensemble gồm **K hàm cộng** (cây hồi quy CART): ŷ_i = Σ_k f_k(x_i), f ∈ F = { f(x) = w_{q(x)} } với q: R^m → T ánh xạ mẫu tới chỉ số lá, T là số lá và w là điểm số liên tục của lá. Khác cây quyết định thông thường, mỗi lá chứa **điểm số liên tục**.
- Mục tiêu: L(φ) = Σ_i l(ŷ_i, y_i) + Σ_k Ω(f_k), với **Ω(f) = γT + ½λ‖w‖²**. Nghĩa là phạt số lá (γ) và độ lớn trọng số lá (λ, điều chuẩn L2). l là hàm mất mát lồi khả vi. Khi điều chuẩn bằng 0, mục tiêu quay về tăng cường cây truyền thống. Điều chuẩn làm trơn trọng số lá để tránh quá khớp, ưu tiên mô hình dùng các hàm đơn giản và có tính dự đoán. Tương tự Regularized Greedy Forest (RGF) nhưng mục tiêu và thuật toán đơn giản hơn, dễ song song hơn.

### 3.2. Huấn luyện theo kiểu cộng và xấp xỉ bậc hai
Vì mô hình có hàm làm tham số nên không tối ưu trong không gian Euclid; huấn luyện theo lượt cộng: ở bước t thêm f_t để tối thiểu hóa L^(t) = Σ_i l(y_i, ŷ_i^(t−1) + f_t(x_i)) + Ω(f_t). Dùng **xấp xỉ Taylor bậc hai**: L^(t) ≈ Σ_i [ l(y_i, ŷ^(t−1)) + g_i f_t(x_i) + ½ h_i f_t²(x_i) ] + Ω(f_t), trong đó g_i, h_i là đạo hàm bậc nhất và bậc hai của mất mát tại dự đoán hiện tại (phương pháp bậc hai gốc từ Friedman). Bỏ hằng số thu được L̃^(t) = Σ_i [ g_i f_t(x_i) + ½ h_i f_t²(x_i) ] + Ω(f_t).

Với cấu trúc cây q cố định, trọng số tối ưu của lá j là **w_j\* = − Σ_{i∈I_j} g_i / (Σ_{i∈I_j} h_i + λ)**, và giá trị mục tiêu tối ưu L̃^(t)(q) = −½ Σ_j (Σ_{i∈I_j} g_i)² / (Σ_{i∈I_j} h_i + λ) + γT (điểm chất lượng cấu trúc cây, giống độ tạp nhưng suy ra cho nhiều hàm mất mát). Không liệt kê được mọi cấu trúc, nên dùng thuật toán tham lam: bắt đầu từ một lá, lần lượt thêm nhánh. **Giảm mất mát khi tách** (mọi tách được đánh giá bằng công thức này):

L_split = ½ [ G_L²/(H_L + λ) + G_R²/(H_R + λ) − (G_L + G_R)²/(H_L + H_R + λ) ] − γ,

với G, H là tổng g_i, h_i trong nút trái/phải. Số hạng −γ khiến nhánh chỉ được giữ nếu cải thiện đủ lớn (tương đương cắt tỉa).

### 3.3. Shrinkage và lấy mẫu cột
Hai kỹ thuật chống quá khớp thêm: **shrinkage** (nhân trọng số cây mới thêm với hệ số η sau mỗi bước, giống tốc độ học, chừa chỗ cho cây sau cải thiện) và **lấy mẫu con theo cột** (đặc trưng), mượn từ Random Forest. Theo phản hồi người dùng, lấy mẫu cột chống quá khớp hiệu quả hơn lấy mẫu hàng truyền thống và còn tăng tốc thuật toán song song.

## 4. Thuật toán tìm điểm tách (mục 3)

- **Thuật toán tham lam chính xác (exact greedy, Alg. 1):** liệt kê mọi điểm tách trên mọi đặc trưng; cần **sắp xếp dữ liệu theo giá trị đặc trưng** rồi duyệt tích lũy g, h. Mạnh nhưng tốn kém, và không khả thi khi dữ liệu không vừa bộ nhớ hoặc ở môi trường phân tán. Các hệ thống đơn máy khác (scikit-learn, gbm của R) hỗ trợ nó; XGBoost đơn máy cũng hỗ trợ.
- **Thuật toán xấp xỉ (Alg. 2):** đề xuất các điểm ứng viên theo phân vị của phân phối đặc trưng, ánh xạ giá trị liên tục vào các thùng, gộp thống kê trong thùng và tìm tách tốt nhất trong các điểm đề xuất. Hai biến thể: **global** (đề xuất một lần ở đầu xây cây, cần nhiều ứng viên hơn vì không được tinh chỉnh) và **local** (đề xuất lại sau mỗi lần tách, hợp với cây sâu và cần ít ứng viên hơn). Thực nghiệm trên Higgs 10M (Hình 3, AUC test theo số vòng): local với eps = 0,3 tiến gần exact greedy; global cần eps nhỏ hơn (0,05) để đạt cùng độ chính xác; phương pháp phân vị đạt độ chính xác của exact greedy ở mức xấp xỉ hợp lý.
- **Weighted quantile sketch (mục 3.3, Phụ lục A):** muốn chọn các điểm ứng viên sao cho |r_k(s_{k,j}) − r_k(s_{k,j+1})| < ε (khoảng 1/ε ứng viên), với hàm hạng r_k(z) = Σ_{x<z} h / Σ h, tức mỗi mẫu có trọng số **h_i** (đạo hàm bậc hai). Lý do dùng h_i làm trọng số: có thể viết lại mục tiêu thành ½ Σ_i h_i (f_t(x_i) − g_i/h_i)² + Ω(f_t) + hằng số, tức một **bình phương sai số có trọng số** với nhãn g_i/h_i và trọng số h_i. Các thuật toán quantile sketch cũ (GK và mở rộng) chỉ cho dữ liệu trọng số bằng nhau; trước đây phải sắp xếp trên mẫu con ngẫu nhiên (có thể thất bại) hoặc dùng heuristic không có bảo đảm. Bài đưa ra cấu trúc dữ liệu hỗ trợ hai thao tác **merge** và **prune** với bảo đảm sai số tương đương GK (merge hai tóm tắt có sai số ε1, ε2 thì ra ε = max(ε1, ε2); prune về b + 1 phần tử làm sai số tăng thêm 1/b), nên có thể cắm vào mọi khung dùng GK. Phụ lục A chứng minh cấu trúc này (định nghĩa tóm tắt phân vị có trọng số, tóm tắt ε-xấp xỉ, hàm truy vấn, phép gộp và cắt tỉa).
- **Tách nhận biết thưa (mục 3.4, Alg. 3):** đầu vào thưa do (1) giá trị thiếu, (2) nhiều giá trị 0, (3) hệ quả của kỹ thuật đặc trưng như one-hot. Cách làm: thêm một **hướng mặc định** cho mỗi nút cây (Hình 4); khi giá trị bị thiếu, mẫu đi theo hướng mặc định. Có hai lựa chọn hướng ở mỗi nhánh và **hướng mặc định tối ưu được học từ dữ liệu**: thuật toán chỉ duyệt các mục không thiếu I_k (liệt kê hai lượt, một lượt gán giá trị thiếu sang phải, một lượt sang trái, chọn điểm tách và hướng có gain lớn nhất). Độ phức tạp tuyến tính theo số mục không thiếu. Cũng áp dụng được khi "không có" nghĩa là giá trị do người dùng chỉ định. Là cách xử lý thống nhất mọi dạng thưa (các thuật toán trước chỉ tối ưu cho dữ liệu dày hoặc cần thủ tục riêng như mã hóa hạng mục). Thực nghiệm trên Allstate-10K (Hình 5): nhanh hơn bản không nhận biết thưa **hơn 50 lần**.

## 5. Thiết kế hệ thống (mục 4)

- **Khối cột cho học song song (4.1):** phần tốn thời gian nhất là sắp xếp dữ liệu. Lưu dữ liệu trong các khối bộ nhớ ở định dạng **cột nén (CSC)**, mỗi cột đã sắp theo giá trị đặc trưng; tính một lần trước huấn luyện và tái dùng. Exact greedy dùng một khối duy nhất và một lần quét tuyến tính thu thống kê cho mọi lá; thuật toán xấp xỉ dùng nhiều khối (mỗi khối một tập hàng), có thể phân tán trên nhiều máy hoặc lưu đĩa. Thu thống kê theo cột song song được; lấy mẫu cột cũng dễ vì chọn tập con cột trong khối. **Độ phức tạp thời gian:** exact greedy với thuật toán nhận biết thưa gốc là O(K d ‖x‖₀ log n) (K số cây, d độ sâu tối đa, ‖x‖₀ số mục không thiếu); với cấu trúc khối chỉ còn **O(K d ‖x‖₀ + ‖x‖₀ log n)** (tiền xử lý được khấu hao), tiết kiệm một hệ số log n. Với thuật toán xấp xỉ: O(K d ‖x‖₀ log q) giảm còn O(K d ‖x‖₀ + ‖x‖₀ log B), B là số hàng tối đa của khối, q là số ứng viên đề xuất (thường 32–100).
- **Truy cập nhận biết cache (4.2):** cấu trúc khối khiến truy cập thống kê gradient theo chỉ số hàng không liên tục; khi thống kê không vừa cache CPU sẽ thiếu cache (cache miss). Giải pháp cho exact greedy: **prefetching nhận biết cache** (mỗi luồng một bộ đệm, nạp thống kê gradient vào đệm rồi tích lũy theo lô nhỏ); **nhanh gấp hai** trên tập lớn (10M mẫu; Hình 7). Cho thuật toán xấp xỉ: chọn **kích thước khối** đúng (khối nhỏ quá thì song song kém, lớn quá thì thiếu cache); thực nghiệm chọn **2^16 mẫu mỗi khối** (Hình 9).
- **Khối cho tính toán ngoài bộ nhớ (4.3):** chia dữ liệu thành nhiều khối lưu trên đĩa, dùng luồng riêng nạp trước khối vào bộ đệm để tính toán song song với đọc đĩa. Hai kỹ thuật tăng thông lượng đĩa: **nén khối** (nén theo cột, giải nén trên luồng riêng; chỉ số hàng lưu dưới dạng offset 16-bit trừ chỉ số bắt đầu của khối, nên 2^16 mẫu mỗi khối; tỉ lệ nén khoảng **26–29%**) và **chia mảnh khối (sharding)** lên nhiều đĩa, mỗi đĩa một luồng nạp trước, luồng huấn luyện đọc xen kẽ.

Bảng 1 so sánh các hệ thống tăng cường cây: XGBoost là hệ duy nhất hỗ trợ đủ exact greedy, xấp xỉ global/local, nhận biết thưa, ngoài bộ nhớ và song song; pGBRT không có exact greedy, thưa, ngoài bộ nhớ; Spark MLlib, H2O hỗ trợ một phần; scikit-learn và R gbm không song song/không thưa.

## 6. Đánh giá đầu cuối (mục 6)

### 6.1. Triển khai và dữ liệu
XGBoost mã nguồn mở, hỗ trợ phân loại có trọng số, xếp hạng, hàm mục tiêu do người dùng định nghĩa; cài đặt phân tán trên rabit; tích hợp Hadoop, MPI, Sun Grid Engine, Flink, Spark và nền tảng Alibaba Tianchi. Bốn bộ dữ liệu (Bảng 2):

| Bộ | Số mẫu | Số đặc trưng | Nhiệm vụ |
|---|---|---|---|
| Allstate | 10 triệu | 4.227 | Phân loại yêu cầu bồi thường bảo hiểm (đơn giản hóa chỉ dự đoán khả năng có yêu cầu) |
| Higgs Boson | 10 triệu | 28 | Phân loại sự kiện vật lý năng lượng cao (mô phỏng Monte Carlo; 21 tính chất động học + 7 đại lượng suy ra) |
| Yahoo LTRC | 473 nghìn | 700 | Xếp hạng (20 nghìn truy vấn, ~22 tài liệu mỗi truy vấn; chia train/test chính thức) |
| Criteo | 1,7 tỉ | 67 | Dự đoán tỉ lệ nhấp quảng cáo (hơn 1 TB ở LibSVM; xử lý ID thành thống kê CTR trung bình và số đếm của 10 ngày đầu) |

Đơn máy: Dell PowerEdge R420, hai Xeon E5-2470 8 nhân (2,3 GHz), 64 GB RAM. Thiết lập chung: độ sâu tối đa 8, shrinkage 0,1, không lấy mẫu cột (trừ khi nêu).

### 6.2. Kết quả
- **Phân loại (Higgs-1M, 500 cây, exact greedy; Bảng 3):** XGBoost 0,6841 s/cây, AUC test **0,8304**; XGBoost (colsample = 0,5) 0,6401 s, AUC 0,8245; scikit-learn 28,51 s, AUC 0,8302; R gbm 1,032 s, AUC 0,6224. Tức XGBoost **nhanh hơn scikit-learn hơn 10 lần** (thực tế ~40 lần) với AUC tương đương; R gbm nhanh hơn scikit-learn nhưng kém hẳn vì chỉ mở rộng một nhánh cây. Lấy mẫu cột cho kết quả hơi kém ở bộ này, có thể do ít đặc trưng quan trọng và hưởng lợi khi chọn tham lam từ tất cả.
- **Xếp hạng (Yahoo LTRC, 500 cây; Bảng 4):** XGBoost 0,826 s/cây, NDCG@10 **0,7892**; XGBoost (colsample = 0,5) 0,506 s, **0,7913**; pGBRT (hệ tốt nhất đã công bố trước đó, chỉ hỗ trợ thuật toán xấp xỉ) 2,576 s, 0,7915. XGBoost chạy nhanh hơn; lấy mẫu cột vừa giảm thời gian vừa nhỉnh hiệu năng (giúp chống quá khớp).
- **Ngoài bộ nhớ (Criteo, một máy AWS c3.8xlarge: 32 vcore, hai SSD 320 GB, 60 GB RAM; Hình 11):** nén khối tăng tốc 3 lần; chia mảnh sang hai đĩa tăng thêm 2 lần. Thuật toán cơ bản chỉ xử lý được 200 triệu mẫu; hệ thống hết bộ nhớ đệm tệp từ 400 triệu mẫu nhưng phương pháp nén + chia mảnh chỉ chậm dần tuyến tính. Xử lý được **1,7 tỉ mẫu trên một máy**.
- **Phân tán (cụm YARN trên EC2, 32 máy m3.2xlarge: 8 nhân ảo, 30 GB RAM, hai SSD 80 GB; dữ liệu trên S3; Hình 12–13):** so với Spark MLlib và H2O (hai hệ trong bộ nhớ): XGBoost nhanh hơn Spark MLlib hơn 10 lần mỗi vòng lặp và 2,2 lần so với bản H2O đã tối ưu (nhưng H2O tải dữ liệu chậm nên thời gian đầu cuối của nó kém hơn); Spark chậm đột ngột khi hết bộ nhớ; XGBoost mở rộng mượt tới toàn bộ 1,7 tỉ mẫu nhờ tính toán ngoài bộ nhớ. Mở rộng theo số máy **gần tuyến tính** (hơi siêu tuyến tính vì thêm máy thì thêm bộ nhớ đệm tệp); chỉ cần **bốn máy** để xử lý toàn bộ 1,7 tỉ mẫu.

## 7. Hạn chế (tác giả nêu và suy ra)
1. Bài là bài **hệ thống**: các đóng góp là hiệu quả và khả năng mở rộng, không phải một thuật toán học mới về mặt biểu đạt; thực nghiệm chủ yếu về tốc độ, chỉ báo cáo độ chính xác trên hai bộ (Higgs-1M, Yahoo LTRC) và không so sánh với mạng nơ-ron hay dữ liệu bảng nhỏ chuẩn.
2. Đánh giá chỉ trên **4 bộ dữ liệu quy mô lớn** (Allstate, Higgs, Yahoo LTRC, Criteo), không có dữ liệu nhỏ, nhãn nhiễu, mất cân bằng lớp hay hiệu chuẩn.
3. Hiệu năng phụ thuộc siêu tham số (độ sâu, shrinkage, điều chuẩn λ, γ, tỉ lệ lấy mẫu cột); bài dùng một cấu hình cố định (độ sâu 8, η = 0,1) cho các so sánh và không đề cập tinh chỉnh.
4. Tối ưu cache và khối phụ thuộc phần cứng (kích thước khối 2^16 chọn thực nghiệm).
5. Thuật toán xấp xỉ có sai số phụ thuộc ε; local và global khác nhau về số ứng viên.

## 8. Liên hệ với bài Nature và đề tài
- Mọi bài TFM đều lấy XGBoost (cùng CatBoost, LightGBM) làm đối chứng; bài Nature báo cáo TabPFNv2 vượt XGBoost/CatBoost/RF **đã tinh chỉnh** trên dữ liệu ≤ 10.000 mẫu; LoCalPFN (bài 10) vượt XGBoost trên 95 bộ TabZilla; TabArena (bài 5) xếp CatBoost đầu khi chỉ tinh chỉnh.
- Hai thuộc tính của XGBoost hữu ích cho thí nghiệm final-term: (i) nhận biết giá trị thiếu qua hướng mặc định (TabICLv2 điền khuyết bằng trung bình, bài 7); (ii) điều chuẩn L2 + γ và shrinkage là các siêu tham số cần tinh chỉnh khi so công bằng.
- Phù hợp với phát hiện của Grinsztajn (bài 29): cây học hàm hằng theo từng đoạn nên không bị thiên kiến "hàm quá trơn" của mạng nơ-ron.
- Khoảng trống: bài gốc không xét nhãn nhiễu hay mất cân bằng lớp; hàm mất mát log-loss với tăng cường bậc hai nhạy với nhãn sai (nhãn sai làm g_i, h_i lệch). Đây là điểm cần đo trực tiếp với XGBoost trong thí nghiệm final-term.

## 9. Từ khóa cần nhớ
- **Mục tiêu có điều chuẩn:** Ω(f) = γT + ½λ‖w‖²; **xấp xỉ bậc hai** với g_i, h_i; **w\* = −G/(H + λ)**; **giảm mất mát khi tách** L_split.
- **Shrinkage η** và **lấy mẫu cột**.
- **Exact greedy / approximate (global, local)**; **weighted quantile sketch** (trọng số = h_i); **hướng mặc định cho giá trị thiếu**.
- **Khối cột CSC đã sắp, prefetch nhận biết cache, khối 2^16, nén 26–29%, sharding**; **1,7 tỉ mẫu trên một máy**; **17/29** giải Kaggle 2015.
