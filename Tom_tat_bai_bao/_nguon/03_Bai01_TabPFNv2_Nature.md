# Bài 1. Accurate predictions on small data with a tabular foundation model (TabPFN v2, Nature 2025)

| Mục | Thông tin |
|---|---|
| Nơi đăng | Nature, tập 637, trang 319–326, 9/1/2025 (Open Access) |
| Tác giả | Noah Hollmann, Samuel Müller (đồng tác giả chính), Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter (Freiburg, Charité, Prior Labs, ELLIS Tübingen) |
| Mã nguồn | priorlabs.ai/tabpfn-nature (có Zenodo). Mã sinh dữ liệu prior **không** được công bố. |
| Vị trí trong đề tài | Đây là "bài survey" gốc của đề tài; mọi so sánh trong báo cáo giữa kỳ xoay quanh bài này. |

## 1. Bài này làm gì trong một đoạn

TabPFN v2 là một **mô hình nền tảng cho dữ liệu bảng** (tabular foundation model). Nó là một Transformer được huấn luyện **một lần** trên khoảng **130 triệu bảng dữ liệu tổng hợp** sinh từ các mô hình nhân quả cấu trúc (SCM). Khi gặp bảng thật, ta đưa cả tập huấn luyện lẫn các dòng cần dự đoán vào mạng; mạng dự đoán ngay **trong một lần chạy xuôi** (học trong ngữ cảnh, ICL), không cập nhật trọng số. Trên bảng tới **10.000 mẫu, 500 đặc trưng**, trong **2,8 giây** nó vượt một ensemble các baseline mạnh nhất đã được tinh chỉnh **4 giờ** (nhanh hơn 5.140 lần với phân loại, 3.000 lần với hồi quy). Ngoài dự đoán, cùng mô hình còn **sinh dữ liệu, ước lượng mật độ, trích embedding, tinh chỉnh (fine-tune) và giải thích bằng SHAP**.

## 2. Vấn đề và động cơ

- Dữ liệu bảng có mặt khắp nơi (y sinh, vật lý hạt, kinh tế, khí hậu), và 76% bộ dữ liệu trên OpenML có dưới 10.000 dòng. Nhiệm vụ cốt lõi: điền cột nhãn dựa vào các cột còn lại.
- Học sâu làm nên cách mạng ở ảnh và văn bản, nhưng với bảng thì **GBDT (cây tăng cường) thống trị suốt 20 năm**. Lý do: bảng không đồng nhất. Cột có nhiều thang đo và kiểu (Boolean, phân loại, thứ tự, nguyên, thực), có thiếu dữ liệu, đặc trưng vô ích, ngoại lệ. Cùng một giá trị có thể mang nghĩa khác nhau ở hai bảng (khác với văn bản, nơi nghĩa của một từ khá ổn định).
- Cây quyết định có nhược điểm: dự đoán ngoài phân phối kém, khó chuyển tri thức giữa các bộ dữ liệu, và không truyền gradient nên khó ghép với mạng nơ-ron.
- Ý tưởng bài đưa ra: tiếp tục xu hướng "thành phần thiết kế tay được thay bằng thành phần học end-to-end" (SIFT/HOG → tích chập, văn phạm → Transformer) sang dữ liệu bảng.

## 3. Ý tưởng cốt lõi: học trong ngữ cảnh có nền tảng lý thuyết

TabPFN kế thừa PFN (bài 6) và TabPFN v1 (bài 3). Quy trình ba bước:
1. **Sinh dữ liệu:** định nghĩa một quá trình sinh (gọi là *prior*) tạo ra các bảng đa dạng với nhiều quan hệ đặc trưng–nhãn. Sinh hàng triệu bảng; với mỗi bảng, che nhãn của một phần mẫu để mô phỏng bài toán học có giám sát.
2. **Tiền huấn luyện (một lần):** huấn luyện Transformer dự đoán nhãn bị che, với các mẫu còn lại làm ngữ cảnh.
3. **Dự đoán trên dữ liệu thật:** đưa mẫu huấn luyện thật làm ngữ cảnh, mô hình dự đoán nhãn mẫu kiểm tra bằng ICL.

Về lý thuyết (Müller 2022): mạng xấp xỉ **phân phối dự đoán hậu nghiệm** p(y_test | X_test, X_train, y_train) cho prior là các bảng tổng hợp, tức nó trả về một dự đoán kiểu Bayes cho phân phối bảng nhân tạo đó. Bài nhấn mạnh đây là "lập trình thuật toán theo ví dụ vào-ra": thay vì viết chỉ dẫn, ta mô tả hành vi mong muốn bằng cách sinh dữ liệu thể hiện hành vi đó rồi huấn luyện mô hình mã hóa thuật toán ấy. Dùng dữ liệu tổng hợp còn tránh được các vấn đề của mô hình nền tảng: bản quyền, quyền riêng tư, lẫn dữ liệu kiểm tra vào dữ liệu huấn luyện, và thiếu dữ liệu.

## 4. So với TabPFN v1

Phiên bản mới **lớn hơn 50 lần** về cỡ dữ liệu, hỗ trợ **hồi quy, biến phân loại, giá trị thiếu**, và bền với **đặc trưng vô ích, ngoại lệ** (những điểm yếu của v1: tối đa 1.000 mẫu, 100 đặc trưng, phân loại số thuần túy).

## 5. Kiến trúc dành riêng cho bảng

- **Mỗi ô (cell) của bảng có một biểu diễn riêng** (trong Hình 1b: mỗi nút là một ô). Mỗi lớp thực hiện **chú ý hai chiều**: (a) mỗi ô chú ý tới các đặc trưng khác trong cùng dòng (chú ý đặc trưng); rồi (b) chú ý tới cùng đặc trưng ở các mẫu khác trong cùng cột (chú ý mẫu); sau đó là MLP. Mỗi nhánh con có residual và layer norm nửa độ chính xác. Có **12 lớp** chú ý hai chiều.
- Kiến trúc **bất biến với thứ tự mẫu và thứ tự đặc trưng** và cho phép **ngoại suy tới bảng lớn hơn khi huấn luyện** cả về số mẫu lẫn số đặc trưng.
- Chú ý mẫu: **mẫu kiểm tra chỉ chú ý tới mẫu huấn luyện**, không chú ý lẫn nhau, nên các dự đoán không ảnh hưởng nhau và không làm đổi biểu diễn của tập huấn luyện.
- **Nhúng đặc trưng ngẫu nhiên**: mỗi đặc trưng nhận một vector ngẫu nhiên (qua một lớp tuyến tính học được) cộng vào mọi embedding của đặc trưng đó, để phân biệt các đặc trưng có cùng thống kê (ví dụ hai cột chứa cùng giá trị khác thứ tự).
- **Mã hóa nhóm đặc trưng**: trong 6 kiến trúc phân loại thử, 3 kiến trúc (và 4 trong 5 kiến trúc hồi quy, kể cả mặc định) dùng một vị trí để mã hóa **hai đặc trưng** của một mẫu.
- **Tiền xử lý đầu vào**: z-normalize theo đặc trưng trên tập huấn luyện, thêm một đầu vào cho mỗi ô cho biết ô đó bị thiếu hay không (giá trị thiếu đặt bằng 0), mã hóa tuyến tính vào chiều embedding.
- **Tách huấn luyện và suy luận**: vì biểu diễn mẫu huấn luyện không phụ thuộc mẫu kiểm tra, có thể **đệm (cache) key/value** của tập huấn luyện và dùng lại cho nhiều lần suy luận. Dùng biến thể multi-query attention (key/value đầu tiên lặp lại cho mẫu kiểm tra) để chỉ lưu một cặp key–value cho mỗi ô. Với 10.000 mẫu, 10 đặc trưng: suy luận nhanh hơn khoảng **300 lần trên CPU (32 s xuống 0,1 s) và 6 lần trên GPU**; với 100 đặc trưng thì 800 lần CPU và 30 lần GPU.
- **Độ phức tạp**: thời gian bậc hai theo số mẫu và số đặc trưng, O(n² + m²); bộ nhớ tuyến tính O(n·m). Tối ưu (layer norm half-precision, flash attention, activation checkpointing, tính trạng thái tuần tự) giảm bộ nhớ 4 lần, còn dưới 1.000 byte mỗi ô, nên dự đoán được bảng tới **50 triệu ô** (ví dụ 5 triệu dòng × 10 đặc trưng) trên một GPU H100.
- **Đầu ra hồi quy**: phân phối hằng số từng khúc (**Riemann / bucketized**) thay cho một giá trị đơn, nên dự đoán được phân phối đa đỉnh. Biên các khúc lấy từ các phân vị 1/5.000 của nhãn lấy mẫu từ prior; hai đầu dùng nửa chuẩn để có support đầy đủ. Mọi biến đổi nhãn (z-score, power transform) được đảo ngược lên biên các khúc khi dự đoán.

## 6. Prior: sinh dữ liệu bằng mô hình nhân quả (phần then chốt)

Đây là nơi "thuật toán" thực sự được quyết định. Quy trình (Hình 2):
1. **Lấy siêu tham số mức cao** cho từng bảng: số mẫu, số đặc trưng, số nút, độ phức tạp đồ thị, độ khó.
2. **Dựng đồ thị có hướng không chu trình** bằng phương pháp "growing network with redirection" (gắn kết ưu tiên, mạng phi tỷ lệ). Hai siêu tham số: số nút N (log-uniform) và xác suất chuyển hướng P (gamma); P nhỏ thì đồ thị dày hơn. Đồ thị có thể là một thành phần liên thông hoặc gộp nhiều đồ thị rời; đồ thị rời cho đặc trưng **độc lập với nhãn** (mô phỏng đặc trưng vô ích).
3. **Truyền dữ liệu khởi tạo** (nhiễu chuẩn, đều hoặc hỗn hợp, có mức phụ thuộc giữa các mẫu qua nguyên mẫu trộn tuyến tính) qua đồ thị. Mỗi cạnh dùng một trong bốn loại ánh xạ: (i) mạng nơ-ron nhỏ với kích hoạt ngẫu nhiên (identity, log, sigmoid, trị tuyệt đối, sin, tanh, hạng, bình phương, lũy thừa, smooth ReLU, bậc thang, modulo); (ii) **rời rạc hóa phân loại** (gán vào điểm gần nhất trong K điểm ngẫu nhiên, K ≥ 2); (iii) **cây quyết định** để mã hóa quy tắc cục bộ; (iv) cộng nhiễu Gauss. Lưu biểu diễn trung gian ở mỗi nút.
4. **Chọn ngẫu nhiên vị trí nút đặc trưng và nút nhãn** để đọc ra một dòng dữ liệu.
5. **Hậu xử lý**: bóp méo Kumaraswamy (phi tuyến), lượng tử hóa đặc trưng liên tục thành K xô, và **đánh dấu thiếu giá trị theo cơ chế MCAR** (thiếu hoàn toàn ngẫu nhiên) với tỷ lệ ρ_miss.
6. **Nhãn**: hồi quy lấy một đặc trưng liên tục không hậu xử lý; phân loại lấy một đặc trưng phân loại tối đa **10 lớp** (nên mô hình chỉ gốc hỗ trợ 10 lớp; có thể mở rộng bằng OVO, OVR hay ECOC).

Nhờ vậy mô hình tự học cách xử lý thiếu giá trị, đặc trưng vô ích, ngoại lệ... mà không cần luật viết tay. Vì mọi siêu tham số của prior là phân phối, PPD ngầm biểu diễn một **ensemble Bayes tích phân qua không gian siêu tham số**. Prior được chọn bằng tìm kiếm ngẫu nhiên trên tập phát triển (development datasets) không trùng tập kiểm tra.

## 7. Huấn luyện

- Hàm mất mát PFN: cross-entropy giữa nhãn bị che của bảng tổng hợp và dự đoán của mô hình, L = E[−log qθ(y_test | X_test, D_train)]; cực tiểu hóa nó tương đương xấp xỉ hậu nghiệm Bayes cho prior đã chọn.
- **~2.000.000 bước, batch 64 bảng ⇒ ~130 triệu bảng tổng hợp** cho mỗi mô hình (phần mở đầu viết "hơn 100 triệu"). Một lần huấn luyện mất khoảng **2 tuần** trên một node **8 GPU RTX 2080 Ti**.
- Số mẫu huấn luyện mỗi bảng lấy đều tới **2.048**, tập validation cố định **128**; số đặc trưng lấy từ Beta (k = 0,95, b = 8,0) co giãn về 1–160; mỗi bảng giới hạn dưới 75.000 ô.
- Adam, warm-up tuyến tính + cosine annealing, thử tốc độ học trong [0,0001; 0,0005] chọn theo mất mát cuối thấp nhất.
- Mô hình huấn luyện **một lần** và dùng cho **mọi bảng mới** (như GPT).

## 8. Suy luận (inference)

- Luôn dùng một ensemble nhỏ: mỗi thành viên **xáo thứ tự đặc trưng** (xấp xỉ bất biến hoán vị), với phân loại còn **xáo nhãn**, áp **nhiệt độ softmax** (mặc định T = 0,9) để hiệu chuẩn.
- Mỗi thành viên dùng một tập con trong các tiền xử lý: **Quantile + giữ bản gốc** (nhân đôi số đặc trưng), **xáo nhãn phân loại ít giá trị**, **SVD** (nối vào đặc trưng), **loại ngoại lệ > 12 độ lệch chuẩn**, **power transform Yeo–Johnson**, **one-hot**. Với PHE/HPO thêm log, exp, biến đổi KDI.
- Tùy chọn (tắt mặc định): đặc trưng đa thức bậc 2 (tối đa 50 cặp), lấy mẫu con trong mỗi thành viên; **thêm đặc trưng định danh mẫu ngẫu nhiên** để xử lý mẫu trùng lặp do kiến trúc bất biến hoán vị.
- Cấu hình **mặc định = ensemble 4 thành viên (phân loại) hoặc 8 thành viên (hồi quy)**.
- **Nhóm dữ liệu bằng random forest** (chỉ dùng khi HPO/PHE): tách tập huấn luyện thành các nhóm đồng nhất hơn qua cây (lá tối thiểu 500–2.000 mẫu) rồi huấn luyện một TabPFN cho mỗi lá, hợp với bảng rất không đồng nhất.
- **TabPFN (PHE)**: ensemble hậu kỳ (post hoc ensembling) chỉ gồm các cấu hình TabPFN; dùng holdout lặp (không phải CV, vì ICL ít quá khớp) và **greedy ensemble selection 25 vòng**, bỏ thành viên trọng số 0 rồi huấn luyện lại bằng toàn bộ dữ liệu.

## 9. Thiết lập thí nghiệm

- **Bộ kiểm tra chính:** 29 bộ phân loại từ **AutoML Benchmark** và 28 bộ hồi quy từ AutoML Benchmark + **OpenML-CTR23**, đều có ≤ 10.000 mẫu, ≤ 500 đặc trưng, ≤ 10 lớp (Extended Data Table 3–4). Còn đánh giá thêm: benchmark của **Grinsztajn** (22 bộ: 15 số + 7 có phân loại, bài 29), **TabZilla** (102 bộ ít hơn 10.000 dòng, bài 28), và **5 cuộc thi Kaggle Tabular Playground Series Season 3**.
- **Baseline:** random forest, XGBoost, CatBoost, LightGBM, hồi quy logistic/ridge, SVM, MLP, (KNN chỉ ở hồi quy, rất kém), và **AutoGluon 1.0** (ensemble chồng có tinh chỉnh).
- **Giao thức:** mỗi bộ × mỗi phương pháp **10 lần lặp** (khác hạt giống và khác phép chia **90/10**, chia theo OpenML); điểm được **chuẩn hóa theo từng bộ dữ liệu** (1,0 = tốt nhất, 0 = tệ nhất trong số mọi baseline); khoảng tin cậy 95%; kiểm định Wilcoxon signed-rank hai phía. Các baseline được tinh chỉnh bằng **tìm kiếm ngẫu nhiên với CV 5 lần**, ngân sách từ 30 giây tới 4 giờ, ứng viên đầu tiên là cấu hình mặc định. Mọi phương pháp dùng 8 nhân CPU; TabPFN dùng thêm một GPU RTX 2080 Ti (baseline không được lợi từ GPU: Extended Data Fig. 2d).
- **Thước đo:** ROC AUC (One-vs-Rest) và độ chính xác cho phân loại; R² và −RMSE cho hồi quy. Lưu ý các baseline được tinh chỉnh theo ROC AUC / RMSE nên các thước đo phụ kém đại diện hơn.

## 10. Kết quả

### 10.1. Phân tích định tính (Hình 3)
- Trên các hàm đồ chơi một đặc trưng (hàm bậc thang, |x|, sin(x)+x, x², có nhiễu đồng/dị phương sai): **hồi quy tuyến tính** thất bại thảm hại ở nhiều hàm; **MLP** kém ở hàm không trơn (rõ nhất ở bậc thang); **CatBoost** chỉ khớp hằng số từng khúc nên sai số xấp xỉ nhưng không thất bại thảm hại; **TabPFN** khớp tốt cả hàm trơn lẫn không trơn, kể cả bậc thang, **và trả về cả phân phối** (độ bất định) miễn phí.
- **Thí nghiệm khe đôi** (double-slit): dự đoán phân phối cường độ ánh sáng đa đỉnh sau khi quan sát vị trí 1.000 photon. TabPFN làm trong **1,2 giây** một lần chạy xuôi; CatBoost phải huấn luyện nhiều mô hình phân vị rồi dựng lại phân phối, kết quả kém hơn dù đã tinh chỉnh, và mất **169,3 giây** ở cấu hình mặc định. TabPFN dự đoán mật độ rất thấp chính xác hơn và ít nhiễu hơn.

### 10.2. Kết quả định lượng chính (Hình 4)
Chuẩn hóa theo bộ dữ liệu, trung bình:
- **Phân loại, ROC AUC chuẩn hóa:** TabPFN mặc định **0,939** so với CatBoost mặc định **0,752** (hơn 0,187); khi tinh chỉnh: TabPFN **0,952** so với CatBoost **0,822** (hơn 0,13).
- **Hồi quy, −RMSE chuẩn hóa:** TabPFN mặc định **0,923** so với CatBoost **0,872** (hơn 0,051); khi tinh chỉnh: **0,968** so với **0,875** (hơn 0,093).
- **Tốc độ:** TabPFN mặc định, trung bình **2,8 giây (phân loại) và 4,8 giây (hồi quy)**, vượt mọi baseline kể cả khi chúng được tinh chỉnh 4 giờ: tăng tốc **5.140 lần** và **3.000 lần**. Với từng bộ dữ liệu, có bộ CatBoost thắng, nhưng TabPFN thắng ở đa số (Hình 4b).
- **Trên benchmark của Grinsztajn** (nơi trước đây cây thắng) và TabZilla, TabPFN cũng vượt hẳn mọi baseline (Extended Data Fig. 2).
- **Kaggle (Extended Data Table 6):** TabPFN mặc định thắng CatBoost mặc định trên cả **5 cuộc thi** TPS S3 có dưới 10.000 mẫu (điểm private trung bình 5 hạt giống, chưa dùng thủ thuật Kaggle như kỹ nghệ đặc trưng).

### 10.3. Độ bền với đặc tính dữ liệu (Hình 5a, 5b)
- **Đặc trưng vô ích** (thêm đặc trưng xáo) và **ngoại lệ** (mỗi ô nhân với số ngẫu nhiên trong [0, hệ số ngoại lệ] với xác suất 2%): TabPFN **rất bền**, khác MLP.
- **Bỏ mẫu / bỏ đặc trưng** làm giảm hiệu năng mọi phương pháp, nhưng **TabPFN chỉ với một nửa số mẫu vẫn bằng baseline tốt nhì dùng toàn bộ mẫu**.
- Chia tập kiểm tra theo: có/không biến phân loại, có/không giá trị thiếu, số mẫu (1–1.999, 2.000–3.999, 4.000–10.000), số đặc trưng (1–19, 20–39, 40–500): **không đặc điểm nào làm hiệu năng tương đối của TabPFN giảm rõ rệt**. Tác giả cảnh báo **không nên suy ra rằng nó mở rộng tốt ngoài 10.000 mẫu / 500 đặc trưng**. Extended Data Fig. 1: khác biệt chủ yếu tinh tế; biến động lớn nhất ở bộ có ngoại lệ trong nhãn (khoảng tin cậy còn chồng nhau).

### 10.4. So với ensemble đã tinh chỉnh (Hình 5c, 5d)
- Phân loại: sau chỉ **2,8 giây**, TabPFN mặc định đã vượt AutoGluon dù AutoGluon được tới 4 giờ. **TabPFN (PHE)** đạt ROC AUC chuẩn hóa trung bình **0,971**, so với **0,939** của TabPFN mặc định và **0,914** của AutoGluon (win rate TabPFN-PHE vs AutoGluon, Wilcoxon P = 0,0024).
- Hồi quy: tinh chỉnh quan trọng hơn. TabPFN (PHE) vượt AutoGluon 4 giờ sau ngân sách tối thiểu 300 giây, tức nhanh hơn **48 lần** (P = 0,0101). Ngân sách tối thiểu là 300 giây vì AutoGluon không trả kết quả đáng tin với ngân sách nhỏ hơn.

### 10.5. Khả năng của mô hình nền tảng (Hình 6, proof-of-concept)
Thực hiện trên German Credit và mfeat-factors:
- **Ước lượng mật độ** (hàm mật độ cho đặc trưng số, hàm khối xác suất cho phân loại) dùng phân rã p(x, y | D) = Π_j p(x_j | x_<j, D) · p(y | x, D), dùng hồi quy/phân loại TabPFN cho từng thành phần; lấy trung bình trên N_j = 24 hoán vị thứ tự đặc trưng (Janossy pooling) để giảm phụ thuộc thứ tự. Dùng được để **phát hiện bất thường** (gian lận, lỗi thiết bị, dữ liệu chất lượng thấp).
- **Sinh dữ liệu tổng hợp** bằng cách lấy mẫu lần lượt từng đặc trưng theo phân rã trên, dùng cho tăng cường dữ liệu, chia sẻ dữ liệu bảo vệ riêng tư, mô phỏng kịch bản.
- **Embedding**: lấy biểu diễn cột nhãn ở lớp cuối; trên mfeat-factors cho tách lớp rõ hơn dữ liệu gốc trên hai thành phần PCA đầu, dùng được cho imputation và phân cụm.
- **Fine-tuning**: thử trên bộ đường sin với độ lệch pha khác nhau giữa dữ liệu tinh chỉnh và kiểm tra, **50 lần chạy** (Extended Data Fig. 4); có **chuyển giao tích cực** ngay cả khi nhãn khác hẳn (độ lệch π thì phải dự đoán nhãn đảo ngược), hiệu năng tốt hơn khi phân phối giống hơn. Ví dụ ứng dụng: tinh chỉnh trên nhiều nghiên cứu y khoa để có mô hình chẩn đoán chung.
- **Giải thích bằng SHAP** (Extended Data Fig. 3, bộ "Default of Credit Card Clients"): hồi quy logistic dễ giải thích nhưng kém chính xác và hàm học được vô lý ở rìa; CatBoost chính xác nhưng khó giải thích (mẫu không rõ, SHAP phân tán); **TabPFN vừa chính xác vừa học hàm trơn, dễ hiểu**.

## 11. Hướng dẫn sử dụng và hạn chế (User guide, tác giả nêu)

- Dùng tốt nhất với ≤ 10.000 mẫu, 500 đặc trưng. Với **bảng lớn hơn** và **hồi quy rất không trơn**, CatBoost / XGBoost / AutoGluon có thể thắng.
- **Hạn chế:**
  1. Tốc độ suy luận có thể **chậm hơn** các phương pháp tối ưu như CatBoost: với 10.000 dòng × 10 cột, TabPFN mất **0,2 giây (GPU) hoặc 0,6 giây (CPU)** để dự đoán một mẫu, trong khi CatBoost mặc định mất **0,0002 giây**. Ngoài ra TabPFN không tối ưu cho suy luận thời gian thực (hướng MotherNet làm nhanh hơn 4 lần so với XGBoost nhưng giảm chất lượng).
  2. **Bộ nhớ tăng tuyến tính** theo cỡ dữ liệu, có thể cấm đoán với bảng rất lớn.
  3. Đánh giá chỉ tới 10.000 mẫu / 500 đặc trưng, **khả năng mở rộng cần nghiên cứu thêm**.
  4. Về thiết kế: tối đa **10 lớp** gốc; mã sinh prior **không công bố** (khó tái lập phần prior).
- Khuyến nghị thực hành: TabPFN tự xử lý giá trị thiếu, mã hóa biến phân loại và chuẩn hóa; nó kỳ vọng đặc trưng và nhãn **xấp xỉ phân phối chuẩn** sau tiền xử lý (nếu biết đặc trưng theo phân phối log thì nên lấy mũ); nên tiếp tục dùng kỹ nghệ đặc trưng và hiểu biết miền; chạy trên GPU nhanh hơn 1–3 bậc.
- **Hướng tương lai:** mở rộng cho bảng lớn (TuneTables), dịch chuyển phân phối theo thời gian (Drift-Resilient TabPFN), tinh chỉnh trên các tác vụ bảng liên quan (retrieval & fine-tuning), nền tảng lý thuyết (Nagler 2023, bài 12), prior chuyên biệt cho chuỗi thời gian, đa phương thức, ECG, ảnh thần kinh, dữ liệu di truyền.

## 12. Liên hệ và ý nghĩa cho đề tài

- Bài này là nền tảng của mọi so sánh tiếp theo: TabICL (và TabICLv2), TabDPT, LoCalPFN, Mitra... đều lấy TabPFN v2 làm mốc.
- Điểm yếu tạo ra khoảng trống nghiên cứu: giới hạn **10.000 mẫu / 500 đặc trưng / 10 lớp**, **độ phức tạp bậc hai**, **chỉ dựa vào prior MCAR** cho giá trị thiếu, **chưa đánh giá với nhãn nhiễu hay mất cân bằng lớp** (bài không có thí nghiệm nào về hai chủ đề này), và việc chỉ có benchmark trên bảng nhỏ.
- Bảng kiểm tra để đối chiếu khi đọc bài: Extended Data Table 1 và 2 (kết quả tổng hợp, mọi phương pháp), Table 5 (không gian siêu tham số và cấu hình mặc định), Table 6 (Kaggle).

## 13. Từ khóa cần nhớ
- **ICL (học trong ngữ cảnh):** dự đoán không cập nhật trọng số, đưa cả tập huấn luyện vào đầu vào.
- **SCM (mô hình nhân quả cấu trúc):** nguồn sinh dữ liệu giả để huấn luyện.
- **Chú ý hai chiều:** đặc trưng rồi mẫu, trên từng ô.
- **PHE:** post hoc ensembling các cấu hình TabPFN.
- **Riemann distribution:** đầu ra hồi quy theo phân phối hằng số từng khúc.
- **Chuẩn hóa theo bộ dữ liệu (0 = tệ nhất, 1 = tốt nhất):** thang điểm dùng trong mọi hình.
