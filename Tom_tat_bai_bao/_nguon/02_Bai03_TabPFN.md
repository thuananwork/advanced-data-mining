# Bài 3. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICLR 2023 (CORE A*) |
| Tác giả | Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter (Freiburg, Charité Berlin, Bosch) |
| Trích dẫn (Google Scholar, 07/10/2026) | 1453 |
| Mã nguồn | github.com/automl/TabPFN |
| Vị trí trong đề tài | Phiên bản đầu tiên của TabPFN. Bài Nature (TabPFNv2) là bản mở rộng của bài này. |

## 1. Bài này làm gì trong một đoạn

Bài giới thiệu **TabPFN**: một Transformer đã được huấn luyện sẵn (một lần, offline) để làm phân loại trên bảng dữ liệu nhỏ mà **không cần huấn luyện lại và không cần chỉnh siêu tham số**. Đưa tập huấn luyện và các mẫu cần dự đoán vào cùng lúc, mạng trả về dự đoán cho toàn bộ tập kiểm tra trong **một lần chạy xuôi**, dưới một giây. Nó được huấn luyện theo khung PFN (bài 6) trên hàng triệu bảng dữ liệu giả sinh từ một prior mới, dựa trên mô hình nhân quả cấu trúc (SCM) và mạng nơ-ron Bayes (BNN), có ưu tiên cho cấu trúc đơn giản. Phạm vi của bản này: tối đa 1.000 mẫu huấn luyện, 100 đặc trưng số, không thiếu giá trị, tối đa 10 lớp.

## 2. Vấn đề và động cơ

- Dữ liệu bảng là dạng dữ liệu phổ biến nhất trong ứng dụng thực tế nhưng học sâu ít được chú ý. Phân loại bảng vẫn do **cây tăng cường (GBDT)** thống trị, chủ yếu vì huấn luyện nhanh và bền.
- Mỗi lần gặp bảng mới phải khớp mô hình lại từ đầu, tìm siêu tham số, thường mất hàng giờ.
- Bài đề xuất thay đổi hẳn cách làm: **không khớp mô hình mới cho từng bảng**, mà thay bước đó bằng một lần chạy xuôi qua một Transformer đã học cách giải các bài toán phân loại bảng từ trước.
- Điểm khác biệt mang tính ý niệm so với các mô hình thông thường: với mạng nơ-ron và GBDT, "điều ta tin trước" (inductive bias) bị giới hạn bởi cái gì dễ cài đặt (điều chuẩn L2, dropout, độ sâu cây). Với PFN, có thể **viết một chương trình sinh dữ liệu** thể hiện đúng điều mình tin, rồi để mạng học xấp xỉ suy luận cho prior đó.

## 3. Nền tảng: PFN (nhắc lại ngắn)

(Chi tiết xem bài 6.) Với prior p(D), PFN học qθ(y | xtest, Dtrain) bằng cách tối thiểu hóa cross-entropy trên các điểm bị che trong những bảng giả. Khi dùng thật, đưa ⟨Dtrain, xtest⟩ vào mạng thu được phân phối dự đoán hậu nghiệm (PPD) trong một lần chạy xuôi, không cập nhật tham số nào, đó là **học trong ngữ cảnh (ICL)**. Quá trình huấn luyện PFN thực hiện **một lần cho một prior**, như một phần của phát triển thuật toán.

Công thức tích phân của PPD mà mạng xấp xỉ: p(y | x, D) tỉ lệ với tích phân của p(y | x, φ) p(D | φ) p(φ) theo mọi giả thuyết φ. Ở đây "giả thuyết" φ là một cơ chế sinh dữ liệu cụ thể, ví dụ một mô hình nhân quả cấu trúc cụ thể. PPD vì thế tương đương với **một ensemble vô hạn các cơ chế sinh dữ liệu**, mỗi cơ chế được cân theo xác suất tiên nghiệm và độ khớp với dữ liệu.

Điểm mới so với PFN gốc cho dữ liệu bảng (Müller 2022, bài 6) chỉ xử lý 30 mẫu huấn luyện, phân loại nhị phân cân bằng, 60 đặc trưng:
1. Mở rộng prior để xử lý **phân loại đa lớp và mất cân bằng lớp**.
2. Nghiên cứu **tiền xử lý** (loại ngoại lệ, biến đổi lũy thừa khi suy luận), xoay chỉ số cột và nhãn khi huấn luyện.
3. **Ensemble** trên nhiều cách tiền xử lý khác nhau.
4. Sửa kiến trúc để nhanh hơn (ma trận chú ý từ (n+m)² còn n² + n·m).
5. **Prior SCM mới**, cộng prior BNN cũ.

## 4. Kiến trúc

- Transformer **12 lớp**, kích thước embedding 512, lớp feed-forward ẩn 1024, **4 đầu chú ý**, tổng cộng **25,82 triệu tham số**. Tối ưu Adam với warm-up tuyến tính và cosine annealing; thử 3 tốc độ học {0,001; 0,0003; 0,0001}, chọn tốc độ cho mất mát huấn luyện cuối thấp nhất.
- Mỗi mẫu huấn luyện (vector đặc trưng + nhãn) thành **một token**; các token huấn luyện chú ý lẫn nhau; các mẫu kiểm tra **chỉ chú ý tới các mẫu huấn luyện** (Hình 1b), không nhìn nhau và không nhìn chính mình. Đầu vào là tập (không có thứ tự), nên bất biến hoán vị mẫu huấn luyện.
- Thay một mô-đun self-attention của PFN gốc bằng **hai mô-đun dùng chung trọng số**: một self-attention giữa các mẫu huấn luyện, một cross-attention từ mẫu kiểm tra tới mẫu huấn luyện. Tương đương về mặt khái niệm, chỉ khác mặt nạ (mẫu kiểm tra không còn nhìn chính nó, nhưng thông tin vị trí hiện tại vẫn đi qua nhánh residual).
- **Số đặc trưng thay đổi**: huấn luyện với số chiều rút ngẫu nhiên đều tới 100; khi ít hơn 100, **đệm số 0** và co giãn các đặc trưng theo hệ số K/k để giữ cùng độ lớn.
- Huấn luyện: **18.000 bước × 512 bảng giả = 9.216.000 bảng**, mỗi bảng cố định kích thước 1.024 mẫu, chia ngẫu nhiên huấn luyện/kiểm tra. Mất **20 giờ** trên máy có **8 GPU Nvidia RTX 2080 Ti**. Đường cong học phẳng dần sau khoảng 10 triệu bảng và khá nhiễu (do prior sinh ra nhiều loại bảng khác nhau).
- Một mô hình duy nhất được dùng cho **mọi thí nghiệm** trong bài.

## 5. Prior cho dữ liệu bảng (phần quan trọng nhất của bài)

**Nguyên tắc 1: hoàn toàn xác suất.** Hầu như mọi siêu tham số của prior cũng là phân phối (không phải một giá trị), nên PPD tích phân luôn trên không gian siêu tham số, ví dụ kiến trúc BNN. Mô hình trộn hai prior: BNN và SCM, mỗi cái lại là hỗn hợp nhiều kiến trúc và siêu tham số. Khi huấn luyện, mỗi bảng được lấy từ BNN hoặc SCM **với xác suất bằng nhau**.

**Nguyên tắc 2: đơn giản (Occam's razor).** Đồ thị ít nút và ít tham số có xác suất cao hơn. Thể hiện qua phân phối log-chuẩn (rời rạc hóa) cho số lớp và số nút, dropout theo phân phối Beta.

**Prior SCM (mô hình nhân quả cấu trúc).**
- Lý do chọn nhân quả: dữ liệu bảng thường có quan hệ nhân quả giữa các cột; cơ chế nhân quả là prior mạnh trong suy luận của người. Bài nói công việc của mình ở "bậc 1,5 của thang nhân quả của Pearl": không làm suy luận nhân quả, nhưng giả định dữ liệu do SCM sinh ra và chỉ dự đoán dựa trên liên kết quan sát.
- Mỗi SCM gồm các phép gán zi = fi(cha của i, εi) trên một đồ thị có hướng không chu trình (DAG), với nhiễu εi.
- Thuật toán lấy mẫu (Phụ lục C.1): (1) lấy số lớp l và số nút h, dựng đồ thị có cấu trúc như một MLP l lớp ẩn h; (2) lấy trọng số cho mỗi cạnh; (3) **bỏ ngẫu nhiên một số cạnh** để thu được DAG; (4) chọn k nút làm đặc trưng và 1 nút làm nhãn; (5) lấy phân phối nhiễu từ một siêu phân phối. Mỗi hàm fi là ánh xạ affine ngẫu nhiên theo sau một hàm kích hoạt (một hàm cho mỗi bảng, lấy từ Tanh, LeakyReLU, ELU, Identity).
- Với mỗi mẫu: lấy nhiễu, lan truyền qua đồ thị, đọc giá trị tại các nút đặc trưng và nút nhãn. Vì vậy nhãn có thể là nguyên nhân hoặc hệ quả của đặc trưng.

**Prior BNN.** Lấy kiến trúc và trọng số ngẫu nhiên, đầu vào x ngẫu nhiên, lan truyền qua mạng có nhiễu và lấy đầu ra làm nhãn.

**Chuyển nhãn liên tục thành nhãn phân loại (mục 4.5).** SCM và BNN chỉ cho nhãn số thực. Cách làm: lấy số lớp Nc từ một phân phối trên số nguyên; lấy Nc − 1 ngưỡng ngẫu nhiên trong tập các giá trị nhãn; ánh xạ mỗi giá trị vào khoảng chứa nó; cuối cùng **xáo ngẫu nhiên thứ tự các lớp** để bỏ quan hệ thứ tự. Cách này sinh được cả bài toán đa lớp lẫn mất cân bằng.

**Tinh chỉnh cho đặc thù dữ liệu bảng (Phụ lục C.2).**
- Tiền xử lý: chuẩn hóa trung bình 0, độ lệch chuẩn 1; khi suy luận dùng thêm **biến đổi lũy thừa (Yeo-Johnson)** vì dữ liệu thật hay có thang mũ. Mọi thống kê chỉ tính từ mẫu huấn luyện; thời gian tiền xử lý được tính vào thời gian suy luận.
- Đặc trưng tương quan: dùng "lấy mẫu đặc trưng theo khối" để các đặc trưng gần nhau tương quan mạnh hơn, giống dữ liệu thật (Hình 20 so sánh ma trận tương quan của bộ PC4 thật với bộ giả).
- Hàm không đều: khuếch đại sự khác biệt tầm quan trọng của các đặc trưng bằng một trọng số nhân cho mọi cạnh đi ra của từng đặc trưng đầu vào; làm thưa theo khối; nhiễu có trung bình và độ lệch chuẩn riêng cho mỗi nút; đầu vào x lấy từ hỗn hợp Gauss, Zipf và đa biến.
- **Giá trị thiếu (NaN)**: không có xử lý đặc biệt, **thay bằng 0** khi suy luận.
- **Đặc trưng phân loại**: chọn ngẫu nhiên một tỷ lệ pcat đặc trưng rồi rời rạc hóa như cách làm với nhãn; có thêm xác suất xáo thứ tự các loại. Trong huấn luyện dùng xác suất 20% có đặc trưng phân loại.
- Một số siêu tham số của prior (Bảng 5): lớp MLP từ 1 tới 6 (trung bình tối đa 6); số nút ẩn trung bình tối đa 130; độ lệch chuẩn nhiễu Gauss; số nút ở lớp 1 của SCM; dropout trọng số 0,9·Beta(a, b), a, b ~ U(0,1; 5).

**Ablation prior (Bảng 4):** chỉ BNN cho mất mát CE trung bình 0,811 và ROC AUC 0,865; chỉ SCM cho 0,771 và 0,881; SCM + BNN cho 0,776 và 0,883. Kết luận: prior SCM mạnh hơn BNN rõ rệt; trộn thêm BNN không tạo khác biệt lớn so với SCM thuần. Cải thiện của prior mới so với prior cũ khoảng 2% trong thiết lập nhỏ, lớn hơn chênh lệch giữa TabPFN cuối cùng và mọi baseline trừ KNN và SAINT.

## 6. Dự đoán và ensemble khi suy luận

- Một lượt chạy xuôi cho toàn bộ tập kiểm tra.
- Phiên bản ensemble: **32 lượt chạy xuôi** trên các phiên bản dữ liệu khác nhau, mỗi phiên bản khác nhau ở: xoay chỉ số cột đặc trưng, xoay chỉ số nhãn lớp, và có/không áp biến đổi lũy thừa (xác suất 0,5). Mỗi tổ hợp chỉ dùng một lần (nên số thành viên không vượt quá 2·k·j với k đặc trưng và j lớp).
- Bản "TabPFN n.e." (no ensembling) nhanh hơn, không có ensemble.

## 7. Thực nghiệm

### 7.1. Bài toán đồ chơi (Hình 4)
So với KNN, hồi quy logistic, MLP đơn giản, Gaussian Process, cây quyết định, CatBoost, Auto-sklearn 2.0 (không chỉnh siêu tham số) trên moons, circles, và hai cặp lớp/đặc trưng của iris và wine. TabPFN mô hình hóa đúng biên quyết định; giống GP, độ bất định lớn ở nơi xa các mẫu đã thấy; vẽ được hình tròn với độ tin cậy cao ở ngoài vùng trộn lẫn; dự đoán "trơn, trực quan, hiệu chuẩn tốt".

### 7.2. Bảng dữ liệu thật (OpenML-CC18)
**Chọn dữ liệu.** Lấy mọi bộ trong OpenML-CC18 có tối đa 2.000 mẫu (1.000 cho huấn luyện), 100 đặc trưng, 10 lớp: **30 bộ**. Chia thành **18 bộ số thuần túy không thiếu giá trị** (phân tích chính trong bài) và 12 bộ còn lại có đặc trưng phân loại hoặc thiếu giá trị (kết quả ở phụ lục). Lý do tập trung vào bảng nhỏ: phổ biến trong thực tế; học sâu yếu nhất ở đây; và TabPFN sẽ tốn nhiều chi phí hơn với bảng lớn.

**Baseline.** 5 mô hình ML chuẩn: KNN, hồi quy logistic, XGBoost, LightGBM, CatBoost (mỗi mô hình dùng kiểm chứng chéo 5 lần trên các cấu hình siêu tham số rút ngẫu nhiên tới khi hết ngân sách hoặc đủ 10.000 cấu hình, chọn cấu hình ROC AUC OVO tốt nhất rồi huấn luyện lại trên toàn bộ tập huấn luyện); 2 hệ AutoML: **AutoGluon** (ensemble chồng) và **Auto-sklearn 2.0** (tối ưu Bayes và ensemble có trọng số); thêm hai mô hình học sâu: Regularization Cocktails (Kadra) và SAINT. Không cố chạy TabNet, NPT vì chúng được đánh giá trên bộ lớn hơn nhiều và dùng tinh chỉnh riêng.

**Giao thức.** 5 lần lặp, mỗi lần một hạt giống khác và một lần chia **50% huấn luyện – 50% kiểm tra**, mọi phương pháp dùng chung phép chia. Báo ROC AUC (one-vs-one cho đa lớp), hạng, số lần thắng, khoảng tin cậy 95%; ngân sách thời gian của baseline {30, 60, 300, 900, 3600} giây.

**Kết quả chính (18 bộ số thuần túy, ngân sách 60 phút cho baseline; Bảng 1).** Giá trị đọc theo thứ tự cột: LightGBM, CatBoost, XGBoost, Auto-sklearn 2.0, AutoGluon, TabPFN không ensemble, TabPFN, TabPFN + AutoGluon.
- ROC AUC OVO trung bình: 0,920; 0,924; 0,924; 0,929; 0,930; 0,932; **0,934**; 0,934.
- Độ chính xác trung bình: 0,862; 0,864; 0,866; 0,870; 0,881; 0,873; 0,879; **0,886**.
- Cross-entropy trung bình (thấp hơn là tốt hơn): 0,750; 0,747; 0,759; 0,813; 0,714; 0,727; 0,716; 0,711.
- (Lưu ý: thứ tự cột trong bảng trích từ PDF có thể lệch; khi trích số của một phương pháp cụ thể hãy đối chiếu Bảng 1 trong PDF. Kết luận chắc chắn: TabPFN ≥ AutoGluon/Auto-sklearn về ROC AUC và vượt GBDT chỉnh siêu tham số.)

**Về tốc độ và đánh đổi (Hình 5 và văn bản).**
- TabPFN dự đoán trong chưa đầy một giây trên một GPU mà **ngang với kết quả AutoML sau một giờ** và **vượt GBDT đã chỉnh**.
- TabPFN không ensemble mất trung bình **1,30 giây trên CPU và 0,05 giây trên GPU** cho một bộ dữ liệu, ngang với baseline mạnh nhất sau 5 phút; tức nhanh hơn **230 lần trên CPU và 5.700 lần trên GPU**. Bản ensemble 32 hoán vị mất 0,62 giây trên GPU.
- Hồi quy logistic và KNN cho kết quả nhanh với ngân sách nhỏ nhưng tệ nhất; GBDT khá hơn nhưng vẫn thua TabPFN và hai hệ AutoML.
- Không phương pháp nào, kể cả TabPFN, tốt nhất trên mọi bộ; có bộ mà ngay cả cấu hình mặc định của baseline thắng TabPFN (ví dụ pm10 số thuần túy nhưng TabPFN kém; monks-problem2 có biến phân loại nhưng TabPFN rất mạnh).
- Chi phí phát triển của từng phương pháp (huấn luyện trước của TabPFN, meta-learning của Auto-sklearn khoảng 3.360 giờ CPU, thiết kế không gian siêu tham số của GBDT) **không tính vào thời gian** vì không nằm ở phía người dùng.

**Mở rộng sang cả 30 bộ (Phụ lục B.1, Bảng 2).** Khi gồm cả bộ có đặc trưng phân loại và giá trị thiếu, TabPFN vẫn mạnh nhưng không bằng trường hợp số thuần túy; Bảng 2 cho ROC AUC OVO trung bình của 8 phương pháp từ 0,884 (LightGBM) tới 0,898 (TabPFN + AutoGluon); TabPFN 0,894. Hình 6 cho thấy TabPFN vượt trội hơn khi **không có đặc trưng phân loại và không có giá trị thiếu**; hiệu năng đa lớp ngang nhị phân.

**OpenML-AutoML Benchmark (Phụ lục B.2, Bảng 3).** Trên 5 bộ nhỏ (vehicle, eucalyptus, blood-transfusion-service-center, Australian, credit-g), dùng đúng script, chia dữ liệu (10-fold, tức 90/10) và kết quả baseline có sẵn: TabPFN mất trung bình **4,4 giây** trên một CPU so với 60 phút của baseline và **tốt hơn mọi baseline** về cross-entropy trung bình, độ chính xác và OpenML metric (độ chính xác trung bình 0,794 và CE 0,449 so với AutoGluon 0,793 và 0,454). Ba trong năm bộ này cũng nằm trong tập kiểm thử chính, một bộ (Australian) nằm trong tập meta-validation; thí nghiệm này xác nhận baseline được chỉnh tốt, thời gian tái lập được, và TabPFN không bị khớp quá với phép chia hay thước đo.

**Tổng quát hóa theo cỡ mẫu (Hình 10).** Khi huấn luyện prior, kích thước bảng giả tối đa 1.024. Thử 18 bộ của OpenML-AutoML Benchmark, lấy 10.000 mẫu mỗi bộ, huấn luyện tới 5.000 mẫu, 5.000 mẫu kiểm tra: **TabPFN tổng quát hóa vượt cỡ mẫu đã thấy**.

**Ensemble với AutoGluon.** Lỗi của TabPFN ít tương quan với lỗi của các baseline (tương quan điểm chuẩn hóa theo bộ dữ liệu thấp hơn giữa các baseline mạnh; Hình 11). Do đó trung bình dự đoán của TabPFN và AutoGluon ("TabPFN + AutoGluon") **vượt mọi phương pháp khác**. Và vì TabPFN dự đoán gần như "miễn phí", ghép nó vào hệ AutoML không tốn thêm.

### 7.3. Phân tích thiên kiến quy nạp (Phụ lục B.3)
Thiết kế theo ba thách thức của Grinsztajn (bài 29): mô hình bảng cần (1) khớp được hàm không đều; (2) bền với đặc trưng vô ích; (3) bảo toàn hướng của dữ liệu.
- **Hàm không đều**: GBDT học mẫu không trơn, MLP học hàm trơn tần số thấp, TabPFN học hàm khá trơn do prior ưa SCM đơn giản; nhưng khi có nhiều mẫu hơn, TabPFN cũng khớp hàm phức tạp hơn (Hình 8: đánh đổi giữa độ phức tạp và số mẫu).
- **Đặc trưng vô ích** (thêm bản sao xáo giá trị): TabPFN và MLP **kém bền** hơn LightGBM khi thêm đặc trưng vô ích; LightGBM gần như không đổi. Gỡ dần đặc trưng theo độ quan trọng (do Random Forest xếp): độ chính xác của TabPFN và GBDT không giảm nhiều khi bỏ tới 30% đặc trưng, MLP còn tốt hơn khi bỏ 20% đặc trưng kém nhất. Đề xuất: bổ sung đặc trưng vô ích vào prior.
- **Xoay đặc trưng**: LightGBM mất nhiều độ chính xác nhất, MLP không đổi, TabPFN chỉ hơi kém; tức TabPFN **gần bất biến theo phép xoay** nhưng vẫn tốt hơn khi không xoay. Theo Ng (2004) và Grinsztajn, thuật toán bất biến xoay có độ phức tạp mẫu tăng ít nhất tuyến tính theo số đặc trưng vô ích; điều này giải thích sự suy giảm của TabPFN khi thêm đặc trưng vô ích.

## 8. Dữ liệu dùng trong bài
- **Meta-test**: 30 bộ OpenML-CC18 (tối đa 2.000 mẫu, 100 đặc trưng, 10 lớp; Bảng 7, ví dụ balance-scale, mfeat-*, breast-w, cmc, credit-g, diabetes, tic-tac-toe, vehicle, eucalyptus, pc1/pc3/pc4, kc2, banknote, blood-transfusion, ilpd, qsar-biodeg, wdbc, cylinder-bands, dresses-sales, MiceProtein, car, steel-plates-fault, climate-model-simulation-crashes).
- **Meta-validation**: **150 bộ** OpenML (một bộ, "flags", bị bỏ vì không đủ lần chia, còn 149), dùng để định hướng chọn siêu tham số của prior; đã loại bộ trùng với meta-test, bộ thời gian (FOREX), bộ nhân tạo (Univ, Friedman).
- Phần cứng baseline: CPU Intel Xeon Gold 6242 2,80 GHz, 1 CPU, tối đa 6 GB RAM; TabPFN thêm một RTX 2080 Ti.
- Không gian siêu tham số của baseline: lấy theo Shwartz-Ziv và Armon (bài 25) cho CatBoost và XGBoost; thử cả không gian của một Kaggle grandmaster nhưng nó cho kết quả tệ hơn ở mọi ngân sách; LightGBM, KNN, GP, hồi quy logistic tự thiết kế.

## 9. Hạn chế (tác giả nêu)
1. **Chi phí bậc hai**: thời gian và bộ nhớ của Transformer tăng bậc hai theo số mẫu; suy luận với trên 100.000 mẫu khó trên GPU phổ thông. Có các phương pháp chú ý tuyến tính có thể gắn vào.
2. Giới hạn cứng của mô hình đã huấn luyện: **tối đa 100 đặc trưng và 10 lớp**.
3. Tập trung vào **số thuần túy, không thiếu giá trị**; với đặc trưng phân loại hoặc giá trị thiếu hiệu năng thường kém hơn.
4. Prior chưa tính đến **đặc trưng vô ích**, nên hiệu năng giảm khi thêm chúng.
5. Chỉ phân loại, chưa hồi quy.
6. Baseline thường nhanh hơn về thời gian dự đoán thuần túy vì baseline tách bước khớp và bước dự đoán, còn TabPFN thực hiện cùng lúc (Bảng 2).
7. Hướng tương lai tác giả nêu: mở rộng cho bảng lớn hơn, xử lý tốt hơn biến phân loại và giá trị thiếu, thêm đặc trưng vô ích vào prior, hồi quy, tích hợp vào AutoML và dùng cho các tác vụ khác như giải thích, kỹ nghệ đặc trưng (đối chiếu mục Discussion trong PDF để lấy danh sách đầy đủ).

## 10. Liên hệ với bài Nature (bài 1)
Bài Nature giải quyết gần như toàn bộ danh sách hạn chế ở trên: hỗ trợ **hồi quy**, **biến phân loại**, **giá trị thiếu**, **đặc trưng vô ích**, mở rộng lên **10.000 mẫu, 500 đặc trưng**, thêm các khả năng như sinh dữ liệu, ước lượng mật độ, trích embedding, tinh chỉnh; thay prior sinh dữ liệu bằng phiên bản giàu hơn. Khung PFN, ý tưởng SCM và ICL giữ nguyên.

## 11. Từ khóa cần nhớ
- **SCM**: mô hình nhân quả cấu trúc, đồ thị có hướng không chu trình với mỗi nút là một hàm của cha và nhiễu.
- **Prior-fitting**: huấn luyện PFN trên bảng giả sinh từ prior.
- **Ensemble xoay**: lấy trung bình dự đoán qua các hoán vị cột/nhãn và biến đổi lũy thừa.
- **OVO ROC AUC**: ROC AUC một-đối-một, trung bình cho đa lớp.
