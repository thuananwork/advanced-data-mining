# Bài 13. PFNs4BO: In-Context Learning for Bayesian Optimization

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2023 (PMLR 202; arXiv:2305.17535v5) |
| Tác giả | Samuel Müller, Matthias Feurer, Noah Hollmann, Frank Hutter (Univ. Freiburg, Prior Labs, Charité) |
| Mã nguồn | github.com/automl/PFNs4BO (kèm mô hình đã huấn luyện) |
| Lưu ý về PDF | Bố cục hai cột nên các bảng/hình bị lẫn chữ; số liệu mình chỉ lấy ở Bảng 1 (đọc rõ) và phần chữ. Kết quả HPO được trình bày chủ yếu bằng đường cong (Hình 4, 5), nên mình chỉ tóm tắt xu hướng, không chép số. |
| Vị trí trong đề tài | Bài **ngoài phạm vi dữ liệu bảng**: dùng PFN làm mô hình thay thế (surrogate) cho tối ưu Bayes. Giúp hiểu **bản chất PFN** (xấp xỉ phân phối dự đoán hậu nghiệm bằng ICL trên một prior tùy ý) và vì sao prior quyết định mọi thứ, nền tảng của TabPFN. Là bài cùng nhóm tác giả (Müller, Hollmann, Hutter). |

## 1. Bài này làm gì trong một đoạn

Quá trình Gauss (GP) là mô hình thay thế phổ biến nhất trong tối ưu Bayes (BO) nhưng có nhiều hạn chế: chi phí lập phương theo số quan sát, giả định phân phối chuẩn đồng thời, prior phải biểu diễn được bằng một hàm nhân (kernel) hợp lệ, và siêu tham số kernel thường chỉ được ước lượng theo kiểu Empirical Bayes (cực đại hóa likelihood) thay vì Bayes đầy đủ. Bài này dùng **Prior-data Fitted Networks (PFN)** làm mô hình thay thế cho BO: một Transformer được huấn luyện một lần, ngoại tuyến, trên các tập dữ liệu lấy mẫu từ một prior tùy ý, rồi cho ra phân phối dự đoán hậu nghiệm trong một lần chạy xuôi nhờ học trong ngữ cảnh. Tác giả cho thấy PFN có thể (i) xấp xỉ sát một GP đơn giản, (ii) mô phỏng prior của HEBO (GP nâng cao) và một mạng nơ-ron Bayes (BNN), và (iii) dễ dàng mở rộng prior: gợi ý của người dùng về vị trí cực trị, bỏ qua chiều thừa, và học hàm thu thập không cận thị. Đánh giá trên 3 bộ HPO (HPO-B, Bayesmark, PD1) với 105 tác vụ, PFN với prior HEBO+ **đạt hoặc vượt các baseline BO tiên tiến**.

## 2. Bối cảnh

### 2.1. Tối ưu Bayes (BO)
BO tối ưu hàm hộp đen có nhiễu với ít lần đánh giá nhất: lặp lại giữa (a) mô hình hóa kết quả bằng mô hình xác suất f̂(y|x, D_k) từ mọi quan sát, và (b) dùng **hàm thu thập** α (EI, PI, UCB…) để chọn điểm truy vấn tiếp theo, cân bằng khai thác và khám phá. Thuật toán 1: lặp (khớp mô hình → chọn x tối đa hóa α → đánh giá f(x) → cập nhật D). Ứng dụng chính ở đây: **tối ưu siêu tham số (HPO)**. HEBO (Cowen-Rivers 2022, thắng cuộc thi black-box của NeurIPS) là GP tiên tiến nhất, dùng biến đổi phi tuyến đầu vào/đầu ra.

### 2.2. Hạn chế của GP
Độ phức tạp lập phương; giả định đồng thời chuẩn (không khớp với dữ liệu đuôi dài); kernel dừng (stationary) khó xử lý hàm không dừng hay phương sai thay đổi; prior phải mã hóa thành kernel nên khó biểu diễn khái niệm hạng mục hay phân cấp; và tuning siêu tham số kernel kiểu Empirical Bayes bị "lời nguyền số chiều".

### 2.3. PFN (Müller 2022)
- **Prior-fitting:** giả sử có thể lấy mẫu tập dữ liệu D ~ p(D) từ prior. Huấn luyện mạng q_θ để dự đoán (x_test, y_test) ∈ D khi điều kiện hóa trên phần còn lại D_train = D \ {(x_test, y_test)}, tối thiểu hóa cross-entropy trên mẫu giữ lại: E[−log q_θ(y_test | x_test, D_train)]. Việc này xấp xỉ **phân phối dự đoán hậu nghiệm (PPD)** Bayes. Chỉ làm một lần cho mỗi prior.
- **Suy luận:** bộ dữ liệu thật D và điểm kiểm thử x đưa vào làm đầu vào; một lần chạy xuôi cho q_θ(y|x, D) qua học trong ngữ cảnh. Không cần huấn luyện lại hay khớp siêu tham số trên dữ liệu thật.
- **Kiến trúc:** Transformer; các vị trí huấn luyện (x_i, y_i) chỉ chú ý lẫn nhau, vị trí kiểm thử chỉ chú ý các vị trí huấn luyện; đầu vào là tập (bất biến hoán vị); độ dài truy vấn thay đổi được.
- **Đầu hồi quy là phân phối Riemann:** rời rạc hóa trục y thành các bucket có xác suất bằng nhau theo prior (mỗi bucket 1/|B|), biến hồi quy thành phân loại; hai đầu mút thay bằng nửa chuẩn để hỗ trợ không giới hạn. Trước đó đã chứng minh tốt hơn đầu hồi quy thường.
- **Công việc liên quan dùng Transformer làm surrogate:** Maraval 2022 (PFN trên hàm đồ chơi, một mạng cho mỗi số chiều, huấn luyện rất ít epoch), Nguyen & Grover 2022 (Transformer neural process, đầu ra phân phối chuẩn), và **Optformer** (Chen 2022, mô hình hóa cả quỹ đạo BO từ các lần tối ưu trước, không phải phương pháp Bayes, dùng cả tên chiều tối ưu).

## 3. Phương pháp: BO với PFN

- **Thay vào Thuật toán 1:** mô hình GP (khớp Empirical Bayes mỗi vòng) được thay bằng PFN q_θ(· | D) đã huấn luyện. Chi phí huấn luyện trả một lần cho mỗi prior, thay vì trả ở mỗi vòng BO. **Một PFN dùng chung cho mọi không gian tìm kiếm** với số chiều khác nhau (huấn luyện trên số chiều lấy ngẫu nhiên 1–18; đệm không cho thiếu chiều và nhân hệ số K/k để giữ độ lớn đầu vào, như TabPFN).
- **Hàm thu thập:** vì đầu ra là phân phối Riemann (hằng từng đoạn), EI, PI, UCB đều tính **chính xác** bằng giải tích; thử cả ba và thấy **EI đơn giản hoạt động ổn định** (so sánh ở Bảng 4 phụ lục).
- **Tối ưu bằng gradient lúc suy luận:** PFN là Transformer nên khả vi, gradient truyền từ đầu ra về đầu vào. Dùng để (i) tối ưu hàm thu thập: lấy ngẫu nhiên 100.000 ứng viên cộng các quan sát, chọn 100 ứng viên có α cao nhất, rồi tối ưu bằng L-BFGS-B, chọn ứng viên có EI lớn nhất sau tối ưu; biến nguyên/luận lý được làm tròn xác suất, bỏ ứng viên đã thử. (ii) **Input warping:** biến đổi đặc trưng đầu vào bằng CDF Kumaraswamy w(x; a, b) = 1 − (1 − xᵃ)ᵇ, với a, b tuned theo từng đặc trưng để tối đa hóa xác suất quan sát lại cùng y (thay cho likelihood dữ liệu, vì cái sau đắt: cần phân tích thành chuỗi xác suất có điều kiện).
- **Mẹo BO (phụ lục D):** biến đổi lũy thừa cho đầu ra (như HEBO) giúp xử lý phương sai thay đổi và làm phẳng các ngoại lai ở phía xấu; đưa đầu vào về siêu lập phương [0,1]ᵈ và chuẩn hóa giá trị quan sát theo min/max cho nhất quán.

## 4. Các prior

1. **GP đơn giản:** kernel RBF, trung bình 0, siêu tham số cố định; lấy x ~ đều trên siêu lập phương, y ~ N(0, K). Dùng để so với hậu nghiệm đúng (ground truth tính được).
2. **HEBO-inspired (HEBO và HEBO+):** kernel Matérn 3/2 với thang độ dài theo chiều, thang đầu ra và nhiễu là các biến ngẫu nhiên độc lập có phân phối p(φ; ψ). HEBO+ là biến thể có chỉnh sửa (không dùng kernel tuyến tính; siêu tham số của prior tìm bằng BO trên 7 không gian tìm kiếm validation của HPO-B: outputscale ~ Gamma(0,8452; 0,3993), lengthscale ~ Gamma(1,2107; 1,5212), log ε ~ N(−4,63; 0,5)).
3. **BNN:** MLP (tanh) có số lớp lấy đều 8–15, số nút ẩn 36–150, trọng số chuẩn với độ lệch chuẩn lấy đều [0,089; 0,193], 14,5% trọng số bị đặt về 0 (rồi nhân lại hệ số), thêm nhiễu Gaussian trước kích hoạt (độ lệch 0,0003–0,0014) và nhiễu đầu ra, input warping Bayes. Mỗi batch lấy mẫu kiến trúc 16 lần cho batch 128 mẫu (hiệu quả hơn nhưng làm các gradient trong batch tương quan). Cùng họ với prior BNN của TabPFN.

## 5. Mở rộng prior (điểm mạnh riêng của PFN)

- **Input warping trong prior:** lấy mẫu siêu tham số warp rồi biến đổi đầu vào (hữu ích nhưng không đủ để bỏ hẳn bước warp sau prior-fitting).
- **Chiều thừa (spurious dimensions):** thêm ngẫu nhiên đặc trưng không liên quan khi huấn luyện (chúng không đi vào GP/NN sinh đầu ra); bản HEBO+ cuối dùng **30% chiều vô ích**. Với GP + Empirical Bayes rất khó mã hóa xác suất một đặc trưng là vô ích. Cải thiện rõ ở ba không gian lớn nhất của HPO-B (Bảng 3 phụ lục).
- **Prior của người dùng:** người dùng chỉ định khoảng I chứa cực trị và độ tự tin ρ ∈ [0,1]; prior p(D|ρ, I) = ρ·p(D | m ∈ I) + (1−ρ)·p(D), với m là vị trí cực đại. Huấn luyện **một PFN** nhận I (15 khoảng trên mỗi chiều) và ρ làm đầu vào bổ sung qua một vị trí thêm với bộ mã hóa tuyến tính riêng (giống style embedding ở mô hình ngôn ngữ). Tích hợp trực tiếp, đúng nguyên tắc Bayes, thay vì warp không gian, tái trọng số hậu nghiệm hay hàm thu thập như các phương pháp trước.
- **Không cận thị (non-myopic):** hàm thu thập Knowledge Gradient α_KG(x; D) = E_{p(y|x,D)}[τ(D ∪ {(x, y)}) − τ(D)], τ(D) = max_x E[y|x,D]. Ước lượng Monte Carlo tốn M·N; còn PFN có thể **học xấp xỉ trực tiếp** bằng hai bước: học phân phối của giá trị trung bình rồi huấn luyện PFN thứ hai cho mục tiêu y' = icdf(q(μ), 0,999) (dùng cùng một PFN với style embedding chỉ chế độ). Kết quả: hỗn hợp KG và EI (tung đồng xu mỗi bước) tốt hơn EI trên bốn không gian tìm kiếm ≤ 3 chiều của HPO-B; chưa mở rộng tới không gian lớn hay nhìn xa nhiều bước.

## 6. Thí nghiệm

### 6.1. Tính đúng của xấp xỉ (trên mẫu prior)
- PFN trùng khớp hậu nghiệm GP thật chỉ khi đủ mẫu prior-fitting: **≥ 20 triệu mẫu** (còn ≤ 100K mẫu thì xấp xỉ kém) (Hình 11–12).
- Với prior HEBO, PFN và GP + Empirical Bayes (cài đặt lại trong GPyTorch) xấp xỉ tốt như nhau về likelihood trên mẫu prior (Bảng 8).
- BO trên mẫu từ prior HEBO (1000 tập dữ liệu rời rạc, mỗi tập 1000 đánh giá, mỗi số chiều): kết quả **hai bên ngang nhau, phần lớn các lần chạy tìm cùng cực đại**:

| Số đặc trưng | PFN thắng | Empirical Bayes thắng | Hòa |
|---|---|---|---|
| 1 | 239 | 206 | 555 |
| 2 | 154 | 144 | 702 |
| 10 | 171 | 169 | 660 |

### 6.2. HPO (mỗi bộ chạy 50 đánh giá; tổng 105 tác vụ trên 19 không gian tìm kiếm)
- **HPO-B** (51 tác vụ, 9 không gian kiểm thử, rời rạc; cây quyết định, SVM, mô hình tuyến tính, rừng ngẫu nhiên, XGBoost). 5 lần lặp. Baseline: Random Search, GP, DNGO, Deep-kernel GP (DGP), HEBO. Kết quả: prior BNN xấp xỉ baseline tốt nhất, **prior HEBO+ vượt các baseline** (xếp hạng và regret trung bình, Hình 4 trên). Phát hiện: khoảng 10% siêu tham số trong HPO-B thiếu thang log; input warping lúc suy luận khắc phục được điều này (chỉ dùng cho HPO-B).
- **Bayesmark** (liên tục, 9 phương pháp scikit-learn, huấn luyện trên 4 bộ dữ liệu, 36 tác vụ, 10 lần lặp; tối ưu độ chính xác). Baseline: HEBO (bản thi đấu), HyperOpt, PySOT, Random. Với thiết kế khởi tạo Sobol thông thường: PFN-BNN bị HEBO vượt nhưng vượt các baseline khác; HEBO+ ngang hoặc kém HEBO một chút. Vì PFN là Bayes đầy đủ nên không cần thiết kế khởi tạo lớn: dùng **một điểm khởi tạo duy nhất** (ở giữa không gian hơi tốt hơn Sobol; ở **góc dưới của không gian rõ ràng vượt mọi cách khác**), dù lúc đầu có vẻ kém.
- **PD1** (tuning mạng sâu lớn: ResNet, Transformer ngôn ngữ; 5 chiều chung; 18 tác vụ; 5 seed, một điểm khởi tạo chung). PFN-BNN bị HEBO vượt rõ, nhưng **PFN-HEBO+ tốt hơn HEBO**. **Prior người dùng** (một prior chung dựa trên vị trí cực trị của 18 tác vụ, không quá mạnh: tối đa đặt nửa trọng số vào một phần tư không gian) **cải thiện mạnh** so với HEBO+; so với quasi-random lấy mẫu từ prior người dùng thì tốt hơn rõ, nên prior không làm bài toán BO trở nên tầm thường.

## 7. Hạn chế (theo tác giả)

1. PFN kém hơn khi dữ liệu có **likelihood rất thấp theo prior** (ngoài phân phối prior); GP ít bị vấn đề này.
2. Không tìm được thiết lập vừa khôi phục được chiều bị gán nhãn log sai vừa giữ hiệu năng mạnh cho không gian đã đúng, nên chỉ dùng input warping cho HPO-B.
3. Không lấy được mẫu đồng thời (joint samples) nhiều điểm như GP, nên khó áp dụng hàm thu thập như noisy EI.
4. Tập trung vào tác vụ đo độ chính xác; **hoạt động kém hơn với tác vụ phi-độ-chính-xác** (động lực khác nhau, kể cả sau biến đổi lũy thừa; chỉ PD1 có loại này).
5. PFN mới được chứng minh tốt đến khoảng **1000 điểm dữ liệu** (Hollmann 2023).
Hướng tương lai: prior cho hàm không dừng, nhiễu phương sai thay đổi, biến hạng mục, hàm gián đoạn, hàng trăm/nghìn chiều vô ích; các quá trình ngẫu nhiên khác (Student-t) hay đầu ra không chuẩn (log-normal); học prior người dùng qua chuyển giao.

Nhận xét thêm của mình:
1. Bài này **không dùng dữ liệu bảng** cho bài toán phân loại/hồi quy thông thường; nó là bằng chứng cho khung PFN, không phải cho TabPFN. Hiệu năng đo bằng regret/thứ hạng BO, khác hẳn AUC/accuracy.
2. Kết quả "ngang GP" trên mẫu từ prior là điều **kỳ vọng** (PFN học chính prior đó); giá trị thực nằm ở các prior GP không viết được bằng kernel (BNN, chiều vô ích) và ở thử nghiệm HPO thật, nơi chênh lệch với HEBO nhỏ và phụ thuộc vào benchmark (PFN-BNN thua HEBO ở PD1).
3. Điều bài đúc kết có ích cho đề tài: **chất lượng PFN bị giới hạn bởi mức độ khớp của prior với dữ liệu thực** và **cần rất nhiều mẫu prior-fitting** (≥ 20 triệu). Đây là cơ sở để hiểu vì sao TabPFN v2 đầu tư vào prior (bài Nature) và vì sao dữ liệu "ngoài prior" (nhiễu nhãn, mất cân bằng nặng) có thể khiến PFN suy giảm.
4. Hạn chế 1 (dữ liệu ngoài prior) chính là cơ chế nghi ngờ cho câu hỏi cuối kỳ về nhiễu nhãn: nếu prior của TabPFN không chứa nhiễu nhãn kiểu thực tế thì mô hình có thể kém bền.

## 8. Liên hệ với bài Nature và đề tài

- **Cùng một khung:** TabPFN = PFN với prior là hỗn hợp SCM + cây trên dữ liệu bảng; PFNs4BO = PFN với prior GP/BNN cho BO. Hiểu bài này giúp thấy rõ mọi thành phần của TabPFN (Riemann head, đệm số chiều, hoán vị bất biến, prior-fitting một lần) đã xuất hiện ở đây.
- **Điểm chung quan trọng với đề tài:** "mô hình Bayes xấp xỉ" nên có hành vi hợp lý về độ bất định (PPD), đáng kiểm tra khi có nhiễu hay mất cân bằng (hiệu chỉnh xác suất).
- Prior người dùng gợi ý một hướng tương lai cho TFM: truyền tri thức chuyên gia vào prior, nhưng chưa có trong TabPFN v2.

## 9. Từ khóa cần nhớ

- **PFN:** Transformer xấp xỉ PPD qua ICL, huấn luyện một lần trên mẫu từ prior (cross-entropy trên điểm giữ lại).
- **Riemann distribution:** rời rạc hóa y thành bucket đều xác suất → EI/PI/UCB tính chính xác.
- **Prior:** GP, HEBO(+), BNN; mở rộng: warping, 30% chiều vô ích, prior người dùng (ρ, I), KG học bằng PFN.
- **Số liệu:** cần ≥ 20 triệu mẫu prior; Bảng 1 gần như hòa với GP; 105 tác vụ, 19 không gian, 50 đánh giá; huấn luyện < 24 giờ trên 8 GPU RTX 2080 Ti; embedding 512, 6 lớp.
- **Hạn chế:** ngoài phân phối prior, tác vụ phi-độ-chính-xác, ≤ ~1000 điểm.
