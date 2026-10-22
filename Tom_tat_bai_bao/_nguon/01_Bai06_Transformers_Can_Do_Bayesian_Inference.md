# Bài 6. Transformers Can Do Bayesian Inference (PFN)

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICLR 2022 (CORE A*) |
| Tác giả | Samuel Müller, Noah Hollmann, Sebastian Pineda, Josif Grabocka, Frank Hutter (Freiburg, Charité Berlin, Bosch) |
| Trích dẫn (Google Scholar, 07/10/2026) | 616 |
| Mã nguồn | github.com/automl/TransformersCanDoBayesianInference |
| Vị trí trong đề tài | Nền tảng lý thuyết của TabPFN và bài Nature. Đọc đầu tiên. |

## 1. Bài này làm gì trong một đoạn

Bài đề xuất **Prior-Data Fitted Networks (PFN)**: một mạng Transformer được huấn luyện để dự đoán nhãn của một điểm dữ liệu khi biết cả một tập các điểm dữ liệu khác làm "ngữ cảnh". Điều đặc biệt là mạng không học từ dữ liệu thật, mà học từ rất nhiều tập dữ liệu **được sinh ra từ một phân phối tiên nghiệm (prior)** do người dùng chọn. Sau khi học xong, đưa một tập dữ liệu thật vào, mạng trả lời ngay trong một lần chạy xuôi, và câu trả lời xấp xỉ kết quả của suy luận Bayes ứng với prior đó. Nói cách khác, mạng đã học cách "làm suy luận Bayes" thay vì phải chạy thuật toán suy luận chậm (MCMC, suy luận biến phân) cho từng tập dữ liệu.

## 2. Bối cảnh và vấn đề

- Học sâu thành công khi có nhiều dữ liệu, nhưng nhiều bài toán thực tế chỉ có ít dữ liệu. Với ít dữ liệu, cần đưa kiến thức có sẵn vào mô hình (qua kiến trúc, điều chuẩn, hay augmentation). Nếu không, định lý "no free lunch" nói rằng không có phương pháp nào tốt cho mọi bài toán.
- Cách có nguyên tắc nhất để đưa kiến thức có sẵn vào là **suy luận Bayes**. Nó có bốn ưu điểm mà bài nêu: có nền tảng lý thuyết khi prior phù hợp; tính đúng xác suất các sự kiện khác nhau; có độ hiệu chuẩn tốt (calibrated); dễ diễn giải vì prior mô tả điều ta kỳ vọng.
- Khó khăn: **phân phối dự đoán hậu nghiệm (PPD)** thường không tính được. Các cách xấp xỉ truyền thống là MCMC (như NUTS, chính xác nhưng chậm) và suy luận biến phân (VI, nhanh hơn nhưng cần biết mật độ). Cả hai đều đòi hỏi những điều kiện khó đáp ứng với prior phức tạp.
- Điều kiện duy nhất của PFN thấp hơn nhiều: **chỉ cần lấy mẫu được từ prior**. Định nghĩa một prior chỉ là viết một chương trình sinh ra các bộ dữ liệu có giám sát.

## 3. Ý tưởng cốt lõi

Cho một prior p(D) trên các tập dữ liệu. Quy trình huấn luyện (Thuật toán 1 trong bài):

1. Lấy một tập dữ liệu D ∪ {(x, y)} từ prior (thực chất là chọn một "nhiệm vụ" rồi sinh các điểm của nó).
2. Che nhãn của một số điểm (các điểm "truy vấn").
3. Cho mạng qθ nhận phần còn lại D và điểm truy vấn x, dự đoán phân phối của y.
4. Tối thiểu hóa −log qθ(y | x, D) bằng gradient, lặp rất nhiều lần.

Khi dùng thật: đưa tập dữ liệu huấn luyện thật Dtrain và điểm kiểm tra xtest vào mạng, nhận về qθ(ytest | xtest, Dtrain). Bài gọi bước này là *suy luận (Bayesian inference)*, khác với việc *huấn luyện bản thân PFN*. Mạng thực hiện **học trong ngữ cảnh (in-context learning)**: không cập nhật tham số nào khi gặp tập dữ liệu mới.

Bài phân biệt PFN với hai họ gần nó:
- **Meta-learning / Neural Processes**: cũng học cách học trong một lần chạy xuôi, nhưng đó là kiến trúc kiểu quá trình ngẫu nhiên; PFN khác ở chỗ dùng dữ liệu sinh tức thời từ prior để chứng minh rằng mạng học được *suy luận Bayes*.
- **Suy luận mô phỏng khấu hao (amortized simulation-based inference)**: cũng chỉ dùng mẫu từ prior, nhưng thường cần biết mô hình cụ thể sinh ra dữ liệu và xấp xỉ posterior của tham số; PFN mô hình hóa trực tiếp PPD mà không bao giờ dựng posterior.

## 4. Cơ sở toán học (giữ ở mức cần để hiểu)

- Hàm mất mát gọi là **Prior-Data Negative Log-Likelihood**: ℓθ = E[−log qθ(y | x, D)], trung bình trên các tập dữ liệu và điểm lấy từ prior.
- **Insight 1**: ℓθ bằng kỳ vọng của cross-entropy giữa PPD thật p(·|x, D) và xấp xỉ qθ. Hệ quả 1.1: ℓθ bằng kỳ vọng của KL-divergence giữa PPD thật và qθ, cộng một hằng số. Nghĩa là tối thiểu hóa mất mát chính là kéo qθ về sát PPD thật.
- Hệ quả 1.2: nếu họ qθ đủ lớn để chứa đúng PPD thì giá trị tối ưu của θ cho đúng PPD thật. (Ghi chú 2024 trong bài: kết quả này được Foong và cộng sự công bố trước năm 2020.)
- Điều này cũng cho phép dùng Prior-Data NLL để **so sánh các phương pháp xấp xỉ** với nhau: giá trị càng thấp thì càng gần PPD thật.

## 5. Kiến trúc và phân phối đầu ra

**Kiến trúc**
- Dựa trên bộ mã hóa Transformer, **không dùng mã hóa vị trí**, nên mạng không phụ thuộc thứ tự các điểm trong tập dữ liệu (bất biến hoán vị). Mỗi cặp (x, y) được mã hóa bằng các lớp tuyến tính rồi cộng lại thành một token.
- Điểm truy vấn được đưa cùng lượt cho hiệu quả. Dùng mặt nạ chú ý: mọi điểm huấn luyện nhìn được mọi điểm huấn luyện khác; mỗi điểm truy vấn nhìn được mọi điểm huấn luyện nhưng **không nhìn các điểm truy vấn khác**, nên các truy vấn không ảnh hưởng nhau.
- Khi huấn luyện cố định tổng số điểm N = n + m (n điểm có nhãn, m điểm truy vấn) và lấy mẫu n ngẫu nhiên để mạng quen với các cỡ dữ liệu khác nhau. Vì n nhỏ thì có nhiều truy vấn hơn, họ lấy mẫu mỗi n với trọng số 1/(N − n) để mỗi cỡ dữ liệu được nhìn số lần tương đương.
- Đầu ra: softmax cho phân loại nhiều lớp, sigmoid cho phân loại nhị phân, phân phối Riemann cho hồi quy.

**Phân phối Riemann (cho hồi quy)**
- Mạng thần kinh làm tốt phân loại hơn nhiều so với dự đoán trực tiếp một phân phối liên tục. Nên bài rời rạc hóa trục y thành các khoảng (bucket) và coi bài toán hồi quy như phân loại vào các bucket. Mật độ thu được là một biểu đồ cột.
- Cách chọn biên các bucket: sao cho mỗi bucket có **xác suất bằng nhau theo dữ liệu từ prior** (p(y ∈ b) = 1/|B|). Xác suất này ước lượng trên một mẫu lớn từ prior.
- Với prior cần miền giá trị không giới hạn, thay bucket ngoài cùng ở mỗi phía bằng một phân phối nửa chuẩn (half-normal) đã co giãn để một nửa khối lượng xác suất nằm trong biên của bucket đó.
- **Định lý 1**: phân phối Riemann với miền hữu hạn có thể mô hình hóa mọi phân phối có mật độ hữu hạn, liên tục hầu khắp nơi và phủ toàn miền, với sai số KL nhỏ tùy ý.
- Ablation (Hình 9): dùng phân phối Riemann thay phân phối chuẩn cho hiệu quả rõ rệt. Việc bỏ mã hóa vị trí chỉ cải thiện chậm khi chuỗi dài hơn.

## 6. Thí nghiệm

### 6.1. Xấp xỉ Gaussian Process cố định (mục 5.1)
- Prior: GP trung bình 0, nhân RBF, length scale 0.6, nhiễu 1e-4 trên đường chéo, output scale 1; điểm đầu vào lấy đều trong hình lập phương đơn vị. Thí nghiệm có nhiều điểm (Hình 4a) dùng 5 chiều đầu vào.
- Với GP cố định, PPD tính được chính xác, nên có thể kiểm tra PFN lệch bao nhiêu.
- Kết quả: trung bình và khoảng tin cậy 95% của PFN "gần như không phân biệt được" với GP thật (chỉ vài chênh lệch nhỏ được đánh dấu mũi tên đỏ). PFN còn tốt hơn rõ rệt Attentive Neural Processes (Hình 8). Với bộ dữ liệu tới 2000 điểm, càng huấn luyện trên nhiều bộ dữ liệu giả (500K, 1M, 2M, 4M) thì Prior-Data NLL càng sát mức tối ưu của GP.

### 6.2. GP với hyper-prior (mục 5.2)
- Thực tế người ta đặt phân phối cho siêu tham số của GP (hyper-prior), nên PPD không còn tính được đúng. Hai cách xấp xỉ phổ biến để so sánh: **MLE-II** (tìm siêu tham số bằng MAP) và **MCMC bằng NUTS**.
- Thiết lập (theo BoTorch SingleTaskGP, chỉ đổi phân phối nhiễu): nhân Matérn ν = 2.5; nhiễu ~ Gamma(0.0001, 1.0); output scale ~ Gamma(2.0, 0.15); length scale ~ Gamma(3.0, 6.0). NUTS dùng bản cài của Pyro, mỗi bộ siêu tham số lấy mẫu là một GP và trung bình các đầu ra.
- Kết quả (Hình 4b): PFN xấp xỉ PPD trên prior **sát hơn cả MLE-II và NUTS**; suy luận nhanh hơn MLE-II hơn 200 lần và nhanh hơn NUTS từ 1.000 đến 8.000 lần.

### 6.3. Mạng nơ-ron Bayes (BNN) (mục 5.3)
- Prior dữ liệu: lấy trọng số ngẫu nhiên cho một BNN Φ, đặc trưng Xi i.i.d. chuẩn, yi = Φ(Xi). Đánh giá trên 100 bộ dữ liệu với m = 200 điểm kiểm tra và n = 100 điểm đã biết.
- Hai kiến trúc BNN trong Hình 5: (a) 3 đặc trưng, 2 lớp, ẩn 5; (b) 8 đặc trưng, 2 lớp, ẩn 64.
- Kết quả: để đạt cùng chất lượng, PFN nhanh hơn Bayes-by-Backprop (SVI) khoảng **1.000 lần** và nhanh hơn NUTS khoảng **10.000 lần**.

### 6.4. Phân loại dữ liệu bảng nhỏ (mục 6) – phần liên quan nhất tới TabPFN
**Hai prior phân loại**
- *Prior GP*: dùng prior GP có hyper-prior ở trên, rồi chuyển thành phân loại bằng cách đặt lớp 1 nếu giá trị lớn hơn trung vị của y, ngược lại lớp 0.
- *Prior BNN theo kiến trúc*: lấy mẫu **kiến trúc** của BNN từ một phân phối (số lớp, số đơn vị ẩn, độ thưa của lớp, độ lệch chuẩn của nhiễu Gauss trên mỗi đơn vị, hàm kích hoạt), rồi lấy trọng số, sinh đặc trưng chuẩn, truyền xuôi và nhị phân hóa đầu ra. Nhờ đó PFN "Bayes" luôn cả về kiến trúc: không cần tìm kiếm kiến trúc cho từng bộ dữ liệu.

**Thiết lập dữ liệu**
- Lấy các bộ dữ liệu từ OpenML AutoML Benchmark; bỏ bộ nào có trên 100 đặc trưng hoặc có giá trị thiếu; còn 20 bộ (bảng chi tiết trong phụ lục liệt kê 21 dòng, gồm kr-vs-kp, credit-g, vehicle, wine, kc1, airlines, bank-marketing, blood-transfusion, phoneme, covertype, numerai28.6, connect-4, car, Australian, segment, jungle chess, sylvine, MiniBooNE, dionis, jannis, helena).
- Mỗi bộ được đơn giản hóa thành **phân loại nhị phân cân bằng**. Với mỗi bộ lấy 20 tập con, mỗi tập 100 mẫu; **chỉ cho nhãn 30 mẫu đầu**, đánh giá trên 70 mẫu còn lại. Mọi mô hình dùng chung các tập con.
- Sáu bộ khác để tinh chỉnh siêu tham số của prior và dải siêu tham số kiểm chứng chéo của các baseline: haberman, ionosphere, sa-heart, cleve, four, german.
- Số đặc trưng thay đổi, nên cố định tối đa 60 đặc trưng, thêm số 0 cho đủ, co giãn đặc trưng theo số đặc trưng đang dùng. Mỗi tập con được chuẩn hóa chỉ từ chính các mẫu mà mô hình dùng, không dùng thông tin của tập kiểm tra.

**Baseline**: XGBoost, CatBoost (có biết cột nào là phân loại), hồi quy logistic, KNN, GP (nhân RBF), BNN (Bayes-by-Backprop bằng Pyro). Mọi baseline dùng grid search với kiểm chứng chéo 5 lần.

**Kết quả (Bảng 1 trong bài; ROC AUC, 30 mẫu huấn luyện)**
- PFN-BNN mạnh nhất tổng thể, hạng trung bình **2,786**; PFN-GP ngang các baseline tốt nhất.
- Kiểm định Wilcoxon signed-rank (trên từng 20 tập con của 21 bộ dữ liệu) cho thấy PFN-BNN tốt hơn có ý nghĩa so với các baseline (p-value rất nhỏ, như 3,6e-13).
- **Tốc độ**: PFN chạy trên GPU V100 cho cả 20 bộ chỉ mất khoảng **13 giây** (4 phút 23 giây trên CPU); XGBoost với grid search cần khoảng **20 giờ**, tức PFN nhanh hơn hơn 5.000 lần, và còn nhanh hơn cả KNN.
- **Độ hiệu chuẩn**: PFN-BNN có Expected Calibration Error (100 bin) khoảng **0,025**, thấp hơn BNN thường và XGBoost, hai baseline giữ được bất định tốt nhất. Hình 10 cho thấy đường hiệu chuẩn của PFN sát đường chéo lý tưởng.
- (Lưu ý của mình: đây là các số trong văn bản và bảng tóm tắt của bài. Bảng 1 và Bảng 7 có thứ tự cột bị xáo khi trích văn bản nên mình không gán số cho từng baseline; muốn số của từng phương pháp thì mở Bảng 1 và Bảng 7 trong PDF.)

### 6.5. Học vài mẫu trên Omniglot (mục 7)
- Prior "nét bút": mỗi lớp là một tổ hợp ngẫu nhiên các đoạn thẳng, mỗi mẫu của lớp là phiên bản của tổ hợp đó với nhiễu vị trí. Chỉ cần 55 dòng mã (Hình 11).
- Bài toán: 5-shot 5-way (5 lớp, mỗi lớp 5 mẫu), theo thiết lập của Rothfuss và cộng sự: một nhiệm vụ cho 5 lớp đầu của mỗi bảng chữ cái, 30 bảng huấn luyện và 20 bảng đánh giá.
- Huấn luyện PFN trên prior nét bút rồi **tinh chỉnh** trên các nhiệm vụ huấn luyện Omniglot (tốc độ học 1e-5, 500.000 mẫu).
- Kết quả độ chính xác kiểm tra: PFN **0,865 ± 0,019**; PACOH-NN 0,885 ± 0,090; Vanilla BNN 0,795 ± 0,006; BMAML 0,764 ± 0,025; MLAP 0,700 ± 0,0135; MAML 0,693 ± 0,013. Tức PFN ngang mức tốt nhất.

## 7. Chi tiết huấn luyện đáng nhớ
- Huấn luyện PFN với prior GP: 100 epoch, 2 giờ 44 phút; với prior BNN: 3 giờ 14 phút. Một mô hình duy nhất được dùng lại cho mọi bộ dữ liệu.
- Siêu tham số của prior (hyper-prior) được chọn bằng tìm kiếm trên một nhóm dữ liệu kiểm chứng riêng, tương tự chọn dải tìm kiếm của grid search, rồi dùng chung cho mọi bài toán cùng loại (ví dụ mọi bài toán phân loại bảng nhỏ).

## 8. Kết luận và hướng tiếp theo của tác giả
Tác giả nêu bốn hướng: (i) tìm các prior mới mà nay đã khả thi nhờ PFN; (ii) kiến trúc phù hợp hơn Transformer thường; (iii) mở rộng PFN lên bài toán thực tế lớn hơn; (iv) dùng PFN cho suy luận mô phỏng khấu hao.

## 9. Hạn chế (nhìn từ bài và nhận xét của mình)
- Chất lượng chỉ tốt khi dữ liệu thật phù hợp với prior đã chọn. PFN chỉ xấp xỉ posterior của đúng prior đó, nên nếu prior sai thì kết quả cũng sai theo.
- Chi phí huấn luyện trước đáng kể (nhiều giờ GPU), chỉ được bù khi dùng lại mô hình nhiều lần.
- Thí nghiệm bảng chỉ ở quy mô nhỏ (100 mẫu mỗi tập con, tối đa 60 đặc trưng, nhị phân cân bằng). Bài chưa chứng minh được cho bảng lớn, nhiều lớp, có giá trị thiếu hay cột phân loại. Đây chính là những chỗ TabPFN rồi bài Nature về sau mở rộng.

## 10. Liên hệ với TabPFN và bài Nature
- TabPFN (bài 3) chính là PFN áp dụng cho phân loại bảng nhỏ: bài này đặt khung, TabPFN chọn prior (nhân quả có ưu tiên cấu trúc đơn giản) và quy mô cụ thể.
- Bài Nature (bài 1) giữ nguyên ý tưởng: sinh dữ liệu giả, huấn luyện Transformer đoán nhãn bị che, rồi suy luận trong một lần chạy xuôi. Phân phối Riemann (bucket có xác suất bằng nhau theo prior) chính là cơ chế hồi quy mà Nature dùng.
- Mặt nạ chú ý "điểm truy vấn chỉ nhìn điểm huấn luyện" ở đây cũng là cơ chế của TabPFN.

## 11. Từ khóa cần nhớ
- **Prior**: phân phối trên các nhiệm vụ hoặc hàm, mô tả điều ta kỳ vọng trước khi thấy dữ liệu.
- **PPD (posterior predictive distribution)**: phân phối dự đoán của y tại x sau khi đã thấy dữ liệu D.
- **PFN**: mạng được huấn luyện để xấp xỉ PPD cho một prior bằng cách học từ dữ liệu sinh ra từ prior đó.
- **In-context learning**: mạng dùng tập dữ liệu như đầu vào để dự đoán, không cập nhật tham số.
- **Prior-Data NLL**: mất mát huấn luyện và cũng là thước đo độ sát với PPD thật.
