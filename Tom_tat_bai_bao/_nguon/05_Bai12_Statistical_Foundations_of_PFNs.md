# Bài 12. Statistical Foundations of Prior-Data Fitted Networks

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2023 (PMLR 202); arXiv 2305.11097 |
| Tác giả | Thomas Nagler (Khoa Thống kê, LMU Munich; Munich Center for Machine Learning) |
| Mã mô phỏng | Script R công khai (gist của tác giả) để tái lập Hình 1 |
| Vị trí trong đề tài | Bài lý thuyết duy nhất trong nhóm đọc đầu: giải thích về mặt thống kê vì sao một mạng PFN đã "đóng băng" vẫn học thêm được khi đưa vào tập dữ liệu lớn hơn lúc huấn luyện. Bài Nature (bài 1) trích dẫn đây là nền tảng lý thuyết. |

## 1. Bài này làm gì trong một đoạn

Các PFN (TabPFN, bài 3 và 1) được huấn luyện trên bảng nhỏ (tới khoảng 1.000 mẫu) mà khi suy luận vẫn **cải thiện thêm khi nhận bảng lớn hơn**, điều không hiển nhiên vì tham số đã cố định. Bài cung cấp **khung lý thuyết**: (i) định nghĩa chính xác phân phối dự đoán hậu nghiệm (PPD) và PFN; (ii) chỉ ra khi nào PPD học được; (iii) giải thích bằng góc nhìn **tần suất (frequentist)**: PFN là một "bộ dự đoán chưa huấn luyện nhưng đã được tinh chỉnh trước", nên hành vi của nó phụ thuộc vào **phương sai** và **độ chệch (bias)** của bộ dự đoán đó; (iv) kết luận kiến trúc Transformer hiện tại đảm bảo **phương sai tiến về 0** nhưng **không đảm bảo bias tiến về 0** vì thiếu tính **địa phương hóa (localization)**; (v) đề xuất một cách địa phương hóa đơn giản sau huấn luyện (kNN rồi chạy PFN) và kiểm chứng bằng mô phỏng.

## 2. Bối cảnh và vấn đề

- PFN (Müller 2022, bài 6): huấn luyện Transformer xấp xỉ PPD trên các tập dữ liệu mô phỏng nhỏ từ nhiều tác vụ; sau đó **đóng băng**, khi suy luận chỉ cần đưa tập huấn luyện mới và các đặc trưng kiểm tra để có dự đoán trong một lần chạy xuôi, không cần huấn luyện hay chỉnh tham số.
- Thực nghiệm (bài 6, bài 3): PFN đạt hiệu năng hàng đầu cho cỡ dữ liệu gần cỡ huấn luyện; điều lạ là **độ chính xác còn tăng khi đưa vào tập lớn hơn** cỡ đã thấy (TabPFN huấn luyện với n tối đa khoảng 1.000). Đây là một ví dụ của **học trong ngữ cảnh (ICL)** vốn đã được quan sát ở LLM.
- Câu hỏi của bài: **vì sao** mạng cố định vẫn học được khi suy luận? Bài tách rõ **prior** (phân phối tác vụ) khỏi **kiến trúc mạng**.
- Các nghiên cứu liên quan được nhắc: ICL ở LLM (Brown, Wei, Dong survey); Garg 2022 (Transformer học hàm tuyến tính, mạng hai lớp, cây quyết định trong ngữ cảnh); giải thích ICL qua gradient descent ngầm (Dai, von Oswald, Akyürek); Olsson 2022 về cụm các đầu chú ý phối hợp; Transformer Neural Processes (Nguyen & Grover 2022) cùng ý tưởng.

## 3. Khung lý thuyết (mục 2)

**Mô hình thống kê.** Bài toán phân loại với nhãn Y và đặc trưng X ∈ R^d; dữ liệu D_n = (Y_i, X_i)_{i=1..n} i.i.d. từ p₀; mục tiêu là xác suất lớp có điều kiện p₀(y | x). Theo thống kê Bayes phi tham số, coi p₀ là một thực hiện của một tham số ngẫu nhiên vô hạn chiều p có phân phối Π (**prior**). Quá trình sinh dữ liệu: (1) rút p ~ Π; (2) rút i.i.d. D_n và (Y, X) từ p.

**PPD.** π(y | x, D_n) = P(Y = y | X = x, D_n). Nếu prior phân rã độc lập thành phần p(y | x) và p(x) thì
π(y | x, D_n) = ∫ p(y | x) dΠ(p | D_n),
tức PPD là **kỳ vọng hậu nghiệm** của p(y | x). Lưu ý (Remark 2.1): các tác giả TabPFN dùng prior phân rã như vậy nhưng không nói rõ; prior không phân rã sẽ cho công thức khác, trong đó việc quan sát x lại mang thông tin về p(y | x), điều phản trực giác.

**PFN.** Là xấp xỉ số học họ PPD. PPD tối đa hóa log-likelihood có điều kiện kỳ vọng E_Π[log q(Y | X, D_n)] trên mọi hàm q hợp lệ (Định lý 2.2), tương đương cực tiểu hóa **KL kỳ vọng** giữa q và π (Remark 2.3). Huấn luyện θ* = argmax E_{Π_N} E_Π [log q_θ(Y | X, D_N)], với Π_N là phân phối trên cỡ mẫu (nên q_θ* bắt chước cả họ PPD, không chỉ phần tử thứ n). Thực tế dùng Monte-Carlo: θ̂ = argmax Σ_j log q_θ(Y_j | X_j, D^(j)) trên m tập dữ liệu mô phỏng có cỡ N_j ~ Π_N. Mô hình q_θ thường **sai đặc tả**, nên (3) cho xấp xỉ KL-tối ưu trong lớp {q_θ}. Với kiến trúc Transformer, một mạng cố định nhận số vector đặc trưng kiểm tra và độ dài tập dữ liệu tùy ý.

## 4. Khi nào PPD học được (mục 3)

PPD hoàn toàn do prior Π quyết định. Mong muốn là hậu nghiệm Π(p | x, D_n) tập trung quanh p₀ khi n tăng. Với dữ liệu phi tham số, chọn prior rất khó: support phải đủ rộng, và prior không được dồn khối lượng vào vùng sai. **Định lý 3.1**: dưới điều kiện (A1), (A2), tồn tại p* ∈ P (P = support của Π) sao cho π(y | x, D_n) → p*(y | x) gần như chắc chắn với P₀-hầu hết (y, x); và p* là **xấp xỉ KL-tối ưu của p₀ trong P**, kể cả khi p₀ nằm ngoài support. Hệ quả: nếu P đủ lớn, p* gần p₀; điều này giải thích vì sao PPD học được, và vì sao việc xấp xỉ PPD bằng PFN có ý nghĩa; prior càng tốt thì PPD càng đáng làm mục tiêu.

## 5. PFN xấp xỉ PPD như thế nào (mục 4)

Bốn yếu tố ảnh hưởng: prior dữ liệu Π, prior cỡ mẫu Π_N, mô hình q_θ, và cỡ Monte-Carlo m. Với lớp mô hình cố định, chất lượng của θ̂ theo lý thuyết cực tiểu hóa rủi ro kinh nghiệm: θ̂ = θ* + O_p(m^(-1/2)) (Phụ lục B). Hai điểm đáng chú ý:
- **Độ phức tạp tương hỗ giữa prior và mô hình**: nếu Π chỉ có mô hình đơn giản thì PFN tối ưu cũng chỉ sinh hàm đơn giản; ngược lại mô hình đơn giản không hưởng lợi từ prior phức tạp. Cần đủ dung lượng ở cả hai.
- Định nghĩa θ_n* = argmax E_Π[log q_θ(Y | X, D_n)] (tham số KL-tối ưu cho cỡ n). PPD thay đổi theo n, nên θ_n* cũng đổi; kỳ vọng độ phức tạp của π(y | x, D_n) tăng theo n, còn ở n = 1 thì PPD thực chất là mô hình trung bình của prior và gần hằng số. Vì vậy **cỡ tập huấn luyện N_j hoạt động như bộ điều chuẩn độ phức tạp** của mạng.
- TabPFN dùng Π_N đều trên {1, …, 1023}; giới hạn nhỏ vì chi phí Transformer bậc hai theo N_j.
- Nghịch lý: TabPFN chưa từng thấy n lớn hơn khoảng 1.000 mà vẫn cải thiện; PFN không đảm bảo hành xử như họ PPD cho prior ngầm nào ở cỡ mẫu chưa thấy, nên Định lý 3.1 không áp dụng cho q_θ̂. Phần tiếp theo trả lời câu hỏi này.

## 6. Vì sao PFN học được lúc suy luận (mục 5)

**Góc nhìn tần suất.** Với mỗi cỡ n gặp lúc suy luận, coi q_θ̂(y | x, ·) là một **bộ dự đoán chưa huấn luyện** ánh xạ tập dữ liệu D_n tới một phân phối có điều kiện; θ là tập "tham số điều chỉnh" được chọn qua meta-learning; Π và Π_N chỉ là phân phối trên các tác vụ ta muốn bộ dự đoán làm tốt. Phân rã sai số:
q_θ(y | x, D_n) − p₀(y | x) = [q_θ − E q_θ] (**phương sai**) + [E q_θ − p₀] (**bias**).
Thực nghiệm cho thấy sai số giảm khi n tăng; câu hỏi là đặc tính cấu trúc nào của PFN gây ra điều đó.

1. **Đối xứng (5.2).** Transformer chuẩn là hàm đối xứng của các mẫu trong D_n (hợp lý với dữ liệu i.i.d.). **Bổ đề 5.1**: mọi bộ dự đoán f đều có phiên bản đối xứng hóa f̃ cùng kỳ vọng, phương sai không lớn hơn; nên đối xứng là tối ưu theo MSE. Nhưng đối xứng **không dẫn tới học được gì**: q = 1/|Y| cũng đối xứng và không học được.
2. **Phương sai và độ nhạy giảm dần (5.3).** Giả sử khi hai tập D_n, D_n′ khác nhau đúng một mẫu thì |q_θ(·|x, D_n) − q_θ(·|x, D_n′)| ≤ L·n^(-α) (điều kiện 5). **Định lý 5.2**: khi đó độ lệch q_θ − E q_θ ≲ n^(1/2−α) với xác suất cao; nếu **α > 1/2** thì phương sai do D_n tiến về 0 (gần như chắc chắn, Bổ đề 5.3). Tức nếu ảnh hưởng của từng mẫu đủ nhỏ, phương sai biến mất.
3. **Bias và nhu cầu địa phương hóa (5.4).** Bias xác định bởi dãy E[q_θ(y|x,D_n)]. Không biết mô hình cụ thể thì khó nói nhiều (có ví dụ bias hằng, giảm, tăng theo n). Điều kiện **cần** để bias tiến về 0 trên lớp P đủ giàu: bộ dự đoán phải **địa phương** (**Định lý 5.4**): bất kể nhãn của các mẫu xa x bị lật hầu như tùy ý (thay bằng nhãn ngẫu nhiên từ một p̃ khác), q_θ không đổi khi n → ∞; tức chỉ các mẫu có X_i gần x mới đóng góp. Kết quả chỉ có nghĩa khi P đủ giàu; chỉ là điều kiện cần (bộ dự đoán hằng q = 1 địa phương nhưng bias không đổi). Nếu bias biến mất với mọi p ∈ P thì q_θ thực chất chỉ dùng n_n = o(n) mẫu hiệu dụng, nên (5) khó giữ với α = 1, phù hợp với cận dưới đánh đổi bias–phương sai của Derumigny & Schmidt-Hieber (2020).

## 7. Các PFN cụ thể (mục 6)

**Bổ đề 6.1**: bộ dự đoán dạng trung bình có chuẩn hóa Σ g(y,x,Y_i,X_i) / Σ 1{X_i ∈ A_n(x)} với g bị chặn bởi K, nếu n^η·P{X_i ∈ A_n(x)} → c (η ∈ (0, 1/2), c > 0) thì thỏa Định lý 5.2 với α = 1 − η, L = 4K/c.

1. **Window smoother (cửa sổ trơn).** q_θ(y|x,D_n) = Σ 1(Y_i = y)1(‖X_i − x‖ < θ) / Σ 1(‖X_i − x‖ < θ); θ* là băng thông KL-tối ưu cho dữ liệu từ prior. PFN thu được chỉ là bộ trơn cửa sổ với siêu tham số đã tinh chỉnh. Có α = 1 nên phương sai tiến về 0; nhưng **bias không đổi** (P₀(Y = y | ‖X − x‖ < θ) − p₀(y|x)), được tối ưu cho cỡ dữ liệu từ Π × Π_N. Nên PFN vẫn "học" khi có thêm dữ liệu, nhưng chỉ nhờ giảm phương sai. Nếu thay băng thông bằng a_n·θ với a_n → 0 (tối ưu MSE là a_n = n^(-1/(4+d)), tốc độ bình phương bias và phương sai n^(-4/(4+d))) thì bias biến mất, nhưng lúc suy luận mô hình cố định không thể đổi số tham số hay thang này theo n.
2. **Cây phân loại (classification trees) và Bayesian model averaging.** Một cây có S vị trí chia là tham số; bias không phụ thuộc n, và muốn giảm bias cần tăng S theo n, điều mô hình cố định không làm được. Giải pháp: **tập hợp K cây với trọng số theo BIC** w(D_n; θ_k) ∝ exp{−BIC(D_n; θ_k)}. Khi n tăng, mô hình trôi về trung bình có trọng số của các thành viên tốt nhất, nên **bias giảm theo n**; θ* khi đó sinh ra các thành viên đa dạng và phức tạp hơn.
3. **Mạng Transformer một lớp (6.3).** Định nghĩa chú ý H đầu: a_j^(h) = SoftMax(vᵀ W_q^(h) V_1, …)_j; u′ = Σ_h Σ_j a_j^(h) W_v^(h) V_j; u = LayerNorm(v + u′); z′ = W_{r,2} ReLU(W_{r,1} u); z = LayerNorm(u + z′); q = SoftMax(W_o z). **Định lý 6.2**: với chuẩn các ma trận bị chặn, **phương sai ≲ n^(-1/2) với xác suất cao**, *bất kể θ* (vì chú ý softmax tự nhiên cho trọng số giảm dần với từng mẫu). **Định lý 6.3**: E[q_θ(·|x,D_n)] → q̄_θ(·|x), trong đó q̄ giống q_θ nhưng u′ được thay bằng Σ_h W_v^(h) E_{V~g_h}[V] với g_h(s) ∝ exp(vᵀ W_q^(h) s) p₀(s) là **phân phối p₀ nghiêng theo hàm mũ** (exponentially tilted). Mỗi đầu chú ý nhìn một "khía cạnh" của p₀. Điều này **địa phương hóa một phần** (tăng ảnh hưởng của mẫu giống v) nhưng **không theo nghĩa của Định lý 5.4**: mẫu xa v vẫn có ảnh hưởng, nên **không thể kỳ vọng bias tiến về 0**. Dù vậy bias giới hạn vẫn có thể nhỏ nếu phần mạng phía sau xử lý tổng các tóm tắt khía cạnh thành xấp xỉ tốt của p₀. Với n nhỏ (n = 1, mọi trọng số chú ý bằng 1) các khía cạnh đóng góp ngang nhau, nên bias có thể giảm **nếu có nhiều đầu chú ý (H > 1)** và phần sau dùng chúng có ý nghĩa, giống hiệu ứng "trung bình mô hình" ở cây; nhưng chỉ đúng với các cỡ n mà θ đã được tinh chỉnh. Với cỡ lớn hơn thì **không có lý do kỳ vọng bias của mạng đã tinh chỉnh còn giảm**.
4. **Địa phương hóa sau huấn luyện (6.4, "Localized PFN").** Áp dụng cho mọi mạng đã huấn luyện: để dự đoán tại x, (1) dựng tập huấn luyện thu gọn D_n(x) chỉ giữ **k_n láng giềng gần nhất** của x; (2) dự đoán nhãn tối đa hóa q_θ(·|x, D_n(x)). Trực giác: giới hạn ở lân cận làm hàm mục tiêu p(y|·) "phẳng hơn" (dễ xấp xỉ) với cái giá giảm cỡ mẫu; cơ chế tương tự window smoother.

## 8. Kiểm chứng số (mục 6.5, Hình 1)

- **Thiết lập:** mô phỏng 500 tập D_n từ p₀(1 | X) = 1/2 + sin(1ᵀX)/2, Y ∈ {0,1}, X ~ N(0, I₅). Chạy TabPFN đã huấn luyện của Hollmann et al. (pip phiên bản 0.1.8). Tính bình phương bias trung bình và phương sai trung bình trên 100 điểm X_test ~ N(0, I₅), với n tới khoảng 4.000.
- **Kết quả:** phương sai giảm **đúng theo tốc độ 1/n**; bias **giảm tới n ≈ 1.000 rồi không tiếp tục giảm** (không tiến về 0), khớp với dự đoán lý thuyết: (i) phương sai biến mất theo 1/n; (ii) bias không biến mất, chỉ giảm tới cỡ n mà mạng đã được tinh chỉnh.
- **TabPFN địa phương hóa** với k_n = min{500, ⌈n^(4/(d+4))⌉}: bias **tiếp tục giảm sau n = 1.000**, đổi lại phương sai tăng nhẹ.
- Ý nghĩa: với n lớn lúc suy luận, TabPFN học chủ yếu qua **giảm phương sai**, hệ quả của kiến trúc Transformer, bất kể tham số θ̂.

## 9. Thảo luận và hướng mở (mục 7)

- Prior của TabPFN dựa trên SCM (Pearl) có ý nghĩa riêng: cặp (Y, X) được sinh bởi một cơ chế nhân quả có nhiễu (không nhất thiết hướng X → Y); vì cơ chế có thể phức tạp tùy ý nên prior về bản chất là **phi tham số**. Tốc độ co hẹp hậu nghiệm tương ứng là câu hỏi mở.
- Ba yếu tố quyết định khả năng học của PFN: **độ nhạy với từng mẫu**, **khả năng chọn mô hình con** (nhiều đầu chú ý / trung bình mô hình), và **địa phương hóa**. Có thể làm bias biến mất bằng cách điều chỉnh tham số theo n nhưng làm tăng độ nhạy; cách này tùy kiến trúc. Địa phương hóa sau huấn luyện là cách đơn giản; thiết kế nghiêm túc nên đưa địa phương hóa vào cả huấn luyện trước và tính đến hiệu quả tính toán; có thể bổ sung cơ chế trung bình Bayes. Hướng khác: kiến trúc Transformer chú ý thưa/tuyến tính (Longformer, BigBird, Reformer) và **bagging/boosting** các PFN để vượt giới hạn kích thước (TabPFN nhanh nên dùng làm thành viên ensemble).
- Nhận xét của tác giả: "tiềm năng đầy đủ của PFN còn chưa được khai thác".

## 10. Hạn chế (của bài)
- Lý thuyết chỉ cho **bài toán phân loại với dữ liệu i.i.d.**; kiến trúc phân tích là **Transformer một lớp** (tác giả lập luận kết quả chuyển sang mạng nhiều lớp vì cơ chế chính trên D_n không đổi, nhưng không chứng minh).
- Một số kết quả chỉ là **điều kiện cần** (Định lý 5.4) hoặc phụ thuộc giả thiết trên prior và support (Định lý 3.1).
- Kiểm chứng số chỉ trên một mô hình mô phỏng đơn giản (d = 5) với **TabPFN v1**; không thử trên dữ liệu thật hay TabPFN v2.
- Không có thí nghiệm về nhãn nhiễu, mất cân bằng hay hiệu chuẩn.

## 11. Liên hệ với đề tài
- Giải thích vì sao TabPFN v1/v2 hoạt động khi n > 1.000 và **vì sao nó có thể bão hòa**: bias không giảm sau cỡ huấn luyện. Điều này tạo động cơ cho hướng "cải thiện bằng địa phương hóa", cụ thể là **LoCalPFN** (một trong 4 SOTA của đề tài) và retrieval/fine-tuning trong bài Nature (tham chiếu [50]).
- Gợi ý khoảng trống: lý thuyết này **giả sử nhãn sạch**; chưa có phân tích tương ứng khi nhãn bị nhiễu (một ứng viên cho đề tài cuối kỳ).

## 12. Từ khóa cần nhớ
- **PPD:** phân phối dự đoán hậu nghiệm, kỳ vọng hậu nghiệm của p(y|x).
- **Untrained predictor:** góc nhìn tần suất xem mạng PFN đã đóng băng là bộ dự đoán có siêu tham số được tinh chỉnh trước.
- **Phương sai ~ 1/n, bias không đổi:** hành vi của Transformer; **địa phương hóa** cần để bias tiến về 0.
- **Exponential tilting:** phân phối p₀ nhân hàm mũ của độ tương tự, xuất hiện ở giới hạn chú ý.
- **Localized TabPFN:** chỉ giữ k láng giềng gần nhất của điểm kiểm tra làm ngữ cảnh.
