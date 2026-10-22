# Bài 18. Do-PFN: In-Context Learning for Causal Effect Estimation

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025 (arXiv:2506.06039v3) |
| Tác giả | Jake Robertson, Arik Reuter, Siyuan Guo, Noah Hollmann, Frank Hutter, Bernhard Schölkopf (Prior Labs, ELLIS Tübingen, Freiburg, MPI-IS, Cambridge) |
| Mã nguồn | github.com/jr2021/Do-PFN (mô hình, mã sinh dữ liệu tiền huấn luyện, dữ liệu case study) |
| Lưu ý về PDF | Hầu hết kết quả là biểu đồ hộp, sơ đồ khác biệt tới hạn (CD) và đường cong calibration, không có bảng số; mình chỉ tóm tắt kết luận của tác giả. Một số công thức trong chú thích hình (cấu trúc phương trình của case study) bị vỡ nên chỉ mô tả bằng lời. |
| Vị trí trong đề tài | PFN cho **suy luận nhân quả** (dự đoán hiệu ứng can thiệp từ dữ liệu quan sát), xây trên **kiến trúc TabPFN v2** và **prior SCM** giống bài Nature. Cho thấy khung TabPFN mở rộng được sang bài toán khó hơn dự đoán. Có phân tích **hiệu chỉnh độ bất định (PICP)**, **độ bền với nhiễu** và **ngoài phân phối prior**, những chủ đề gần với đề tài. |

## 1. Bài này làm gì trong một đoạn

Ước lượng hiệu ứng nhân quả (ví dụ thuốc có giảm nguy cơ ung thư không) là trung tâm của y học, kinh tế, khoa học xã hội, nhưng các phương pháp hiện có hoặc cần dữ liệu can thiệp (RCT), hoặc cần biết đồ thị nhân quả thật, hoặc dựa vào giả định "không nhiễu ẩn" (unconfoundedness) vốn khó kiểm chứng. Bài này hỏi: nếu TabPFN đã thành công ở dự đoán nhờ tiền huấn luyện trên dữ liệu SCM tổng hợp, liệu cách làm đó có chuyển sang bài toán nhân quả không? Do-PFN được tiền huấn luyện trên dữ liệu lấy từ **rất nhiều cấu trúc nhân quả (SCM) tổng hợp, có cả can thiệp**, để từ dữ liệu quan sát dự đoán **phân phối kết quả có điều kiện sau can thiệp** (CID) p(y | do(t), x) mà không cần biết đồ thị nhân quả. Trên các nghiên cứu tình huống tổng hợp, bán tổng hợp (RealCause) và hai bộ dữ liệu thật có đồ thị được thừa nhận, Do-PFN ước lượng CID, CATE và ATE chính xác, bền hơn nhiều baseline khi giả định "không nhiễu ẩn" bị vi phạm, và có độ bất định được hiệu chỉnh khá tốt.

## 2. Bối cảnh

- **SCM (mô hình nhân quả cấu trúc):** một DAG G_ψ gồm K nút (biến z_k), mỗi nút sinh từ cha qua z_k = f_k(z_PA(k), ε_k) với nhiễu ε_k.
- **Can thiệp do(t)** với biến điều trị nhị phân T ∈ {0,1}: xóa mọi cạnh vào nút T và gán T = t. Hiệu ứng lên y là p(y | do(t), ψ). **CID** p(y | do(t), x) thêm điều kiện lên vector đặc trưng x; từ đó **CATE** τ(x) = E[y|do(1), x] − E[y|do(0), x].
- **Nguồn dữ liệu:** RCT (tiêu chuẩn vàng nhưng đắt/phi đạo đức/bất khả thi) so với dữ liệu quan sát D^ob = {(y_j, t_j, x_j)} (dễ có hơn, nhưng cần giả định mạnh).
- **Hai khung cổ điển:** do-calculus của Pearl (cần SCM/đồ thị giả định, rồi dùng điều chỉnh back-door, front-door); Neyman–Rubin (kết quả tiềm năng, cần ignorability/unconfoundedness): causal trees/forests, T-, S-, X-learner, doubly robust/DML.
- **PFN và suy luận Bayes được khấu hao (amortized):** học ánh xạ D ↦ p(y | x, D) bằng cách mô phỏng nhiều bộ (D_i, y_i, x_i) (neural process, simulation-based inference); PFN nhấn mạnh tiền huấn luyện quy mô lớn và bộ mô phỏng thực tế (prior).
- **Nhân quả được khấu hao trước đó:** khám phá nhân quả (Ke 2022, Lorch 2022, Dhir 2025); Sauter 2025 học dịch chuyển phân phối khi can thiệp nhưng không vượt baseline điều kiện hóa ngay cả với hai biến; Nilforoshan 2023 zero-shot CATE cho điều trị mới; Bynum 2025 (đồng thời) chỉ SCM ≤ 3 nút và ước lượng điểm, bỏ qua bất định; CausalPFN (Balazadeh 2025, đồng thời) cùng hướng.

## 3. Phương pháp

### 3.1. Giả định và mục tiêu
Đặt một **prior p(ψ) trên SCM** (do đó trên cả đồ thị). Mỗi SCM ψ ~ p(ψ) mô phỏng được dữ liệu quan sát p(y^ob, t^ob, x^ob | ψ) bằng cách lan truyền nhiễu ε ~ p(ε), và dữ liệu can thiệp p(y^in, x^in | ψ, do(t^in)) bằng cách lan truyền qua SCM đã can thiệp; có thêm prior p(t^in) cho giá trị can thiệp (chỉ cần đủ hỗ trợ). CID mong muốn:
p(y^in | do(t^in), x^in) = ∫ p(y^in | do(t^in), x^in, ψ) p(ψ | x^in) dψ.
Điều này mở rộng cách tiếp cận do-calculus (vốn dùng một đồ thị cố định) thành tích phân trên nhiều đồ thị, và **bao gồm cả các kịch bản không có unconfoundedness**. Lưu ý p(ψ|x^in) ≠ p(ψ) vì biết x^in đã cho thông tin về SCM.

### 3.2. Thuật toán 1: prior-fitting bằng SGD
Lặp qua N SCM: (1) lấy ψ_i ~ p(ψ); (2) lấy số điểm quan sát M^ob ~ Uniform({M_min..M_max}) và sinh D^ob_i = {(y, t, x)}; (3) lấy M^in = M_max − M^ob điểm can thiệp: mỗi điểm lấy lại nhiễu ε_k, lấy x^in_k ~ p(x | ψ_i, ε_k) (giá trị trước điều trị), lấy t^in_k ~ p(t^in), rồi **can thiệp do(t^in_k) và sinh y^in_k với cùng nhiễu ε_k**; (4) tính L_i(θ) = Σ_k −log q_θ(y^in_k | do(t^in_k), x^in_k, D^ob_i) và một bước giảm gradient. Lặp qua hàng triệu SCM để học "meta" cách suy luận nhân quả trong ngữ cảnh. Suy luận: đưa (t^in, x^in) cùng toàn bộ D^ob vào, Transformer trả q_θ(y | do(t), x, D^ob) bằng một lượt chạy xuôi.

### 3.3. Chi tiết prior (phụ lục C)
- **DAG:** lấy bằng sắp xếp tô-pô các đỉnh.
- **Cơ chế:** mô hình nhiễu cộng (ANM) f_k(z_PA, ε_k) = γ(Σ_l w_l z_l) + ε_k, γ chọn ngẫu nhiên trong {x², ReLU, tanh}; trọng số w_l ~ U(−1/√|PA|, 1/√|PA|) (khởi tạo Kaiming).
- **Kích thước:** M^ob lấy đều 10–2200 (M_min = 10, M_max = 2200). Phiên bản v1 huấn luyện trên đồ thị tới 10 nút (48 giờ); **v1.1 tới 60 nút (96 giờ)**; đều trên một GPU RTX 2080.
- Nhiễu mặc định Gaussian (các biến thể Do-PFN-Mixed dùng thêm Laplace, Student-t, Gumbel).

### 3.4. Kiến trúc
Transformer giống **TabPFN v2** (Hollmann 2025), thêm một chỉ báo nội bộ cho biết cột đầu là biến điều trị và phần còn lại là biến phụ. **7,3 triệu tham số**. Đầu ra là phân phối thanh (bar distribution) như TabPFN.

### 3.5. Lý thuyết
- **Mệnh đề 1:** SGD theo Thuật toán 1 tối thiểu hóa kỳ vọng KL thuận giữa CID thật p(y^in | x^in, do(t^in), ψ) và q_θ(y^in | do(t^in), x^in, D^ob), kỳ vọng lấy theo quá trình sinh dữ liệu của prior. Nghĩa là xấp xỉ tối ưu **theo prior đã chọn**; không khẳng định ước lượng được mọi hiệu ứng nhân quả theo nghĩa cổ điển (và vẫn đúng ngay cả khi nhiễu không độc lập, tức trường hợp không Markov).
- **Ba nguồn bất định (phân rã Eq. 4):** (i) bất định ngẫu nhiên (aleatoric) từ nhiễu của SCM; (ii) bất định do **không định danh được** (nhiều SCM tương đương quan sát, p(ψ|r)); (iii) bất định nhận thức (epistemic) về lớp tương đương Markov của SCM thật do dữ liệu hữu hạn, p(r | D^ob).
- **Tính nhất quán (Định lý, phụ lục B):** khi |D^ob| → ∞, phần (iii) biến mất, còn (i) và (ii) giữ lại; hậu nghiệm hội tụ tới p(y^in | x^in, do(t^in), [ψ_0]), tức tập trung vào lớp tương đương quan sát của SCM thật. Nếu SCM định danh được thì hội tụ về CID dưới đúng ψ_0.

## 4. Thiết lập thí nghiệm

- **Sáu nghiên cứu tình huống tổng hợp** (mỗi cái 100 bộ dữ liệu độc lập, thay đổi tham số SCM, số mẫu, độ lệch chuẩn nhiễu, trọng số cạnh và phi tuyến): Observed Confounder, Observed Mediator, Confounder + Mediator, Unobserved Confounder, Back-Door Criterion, Front-Door Criterion; cùng ba nghiên cứu ablation (bộ nhỏ M_max ~ U[5,100]; đồ thị phức tạp K ~ U[4,10] nút; và "Common Effect"). Độ lệch chuẩn nhiễu ngoại sinh σ_exo ~ U[1,3], nhiễu cộng σ_ε = 0,3·β, β ~ Beta(1,5).
- **Bán tổng hợp:** RealCause (Neal 2020; 4 nhóm bộ dữ liệu, mô hình sinh khớp dữ liệu thật nhưng bảo đảm unconfoundedness), cùng hai bộ dữ liệu thật có đồ thị được thừa nhận: **Amazon Sales** (ảnh hưởng của sự kiện mua sắm lên lợi nhuận điện thoại, cấu trúc trung gian phức tạp) và **Law School Admissions** (ảnh hưởng của "Race" lên điểm năm nhất, qua UGPA và LSAT; tác giả ghi rõ không bàn về công bằng thuật toán và không đồng ý với việc coi chủng tộc là biến nhị phân).
- **Baseline dự đoán CID:** Random Forest, TabPFN v2, **Dont-PFN** (cùng kiến trúc, tiền huấn luyện trên chính prior nhưng dự đoán kết quả quan sát — đối chứng loại bỏ yếu tố "prior nhân quả" và "mục tiêu can thiệp"), Do-PFN-Short (20 giờ, đồ thị ≤ 5 nút), Do-PFN-Graph (5 giờ, chỉ trên đồ thị cố định của case study), Do-PFN-Mixed (24 giờ, nhiễu hỗn hợp), DoWhy (Int./Cntf.) là baseline "tiêu chuẩn vàng" biết đồ thị thật và dùng TabPFN xấp xỉ phương trình cấu trúc.
- **Baseline CATE/ATE:** S-, T-, X-Learner (TabPFN và HPO bằng FLAML 60 giây), DML (TabPFN), Causal Forest (DML), TARNet, DragonNet, DoWhy-CATE.
- **Chỉ số:** MSE chuẩn hóa; kiểm định Friedman + hậu kiểm Nemenyi (Autorank), biểu đồ CD; độ phủ khoảng dự đoán (PICP).

## 5. Kết quả

- **Dự đoán CID (Hình 3, hàng đầu):** Do-PFN **tốt hơn có ý nghĩa** Random Forest, TabPFN v2 và Dont-PFN. Khoảng cách Do-PFN – Dont-PFN cho thấy tiền huấn luyện theo Thuật toán 1 xấp xỉ một thứ khác hẳn PPD quan sát thông thường. Điểm then chốt: mẫu ngữ cảnh D^ob và mẫu truy vấn D^in đến từ **hai quá trình sinh khác nhau** (SCM gốc và SCM đã can thiệp), gây lệch giữa p(t^ob, y^ob) và p(t^in, y^in) đúng khi có cạnh có hướng từ điều trị sang biến phụ. Khi t và x độc lập ("Common Effect" – tương đương RCT), hiện tượng này biến mất và các mô hình hồi quy bảng thông thường ngang Do-PFN. Với hiệu ứng nền nhỏ, TabPFN v2 và Do-PFN tương tự nhau.
- **Điều chỉnh không cần biết đồ thị:** Do-PFN cạnh tranh với DoWhy (Int.) trong dự đoán CID; ngay cả Do-PFN-Short cũng cạnh tranh với Do-PFN-Graph (huấn luyện chỉ trên đúng đồ thị đã biết), cho thấy mô hình thực hiện điều chỉnh như áp dụng tiêu chí front-door/back-door mà không có đồ thị. Do-PFN vượt có ý nghĩa Do-PFN-Graph và DoWhy (Int.) (Hình 12). Mô hình chính thắng Do-PFN-Graph và Short, gợi ý **số lượng bộ dữ liệu tiền huấn luyện là yếu tố then chốt**; Do-PFN-Mixed chỉ huấn luyện nửa thời gian mà ngang mô hình chính, gợi ý nhiễu đa dạng giúp hội tụ.
- **CATE/ATE (Hình 3 hàng 2–3):** Do-PFN **vượt có ý nghĩa** meta-learner (S/T/X), DML/causal forest, DragonNet, TARNet, kể cả trên các case study khá đơn giản. Hiệu năng baseline phụ thuộc vào việc unconfoundedness có đúng hay không (Hình 11): "Observed Confounder" và "Back-Door" thỏa, còn "Observed Mediator", "Confounder + Mediator", "Unobserved Confounder", "Front-Door" vi phạm; hầu hết phương pháp xấu đi khi vi phạm, còn **Do-PFN ổn định nhất ở cả hai trường hợp**. Ở CATE, Do-PFN gần "tiêu chuẩn vàng" DoWhy (Cntf.) hơn so với CID: phân rã độ chệch (Hình 14) cho thấy Do-PFN dự đoán CID hơi cao nhưng độ chệch gần như bằng nhau cho do(0) và do(1), nên **triệt tiêu khi lấy hiệu CATE**.
- **Ablation trong phân phối:** (i) kích thước dữ liệu 5–2000: MSE giảm và ổn định dần khi tăng mẫu, cạnh tranh với DoWhy (Cntf.); (ii) đồ thị 4–10 nút, 2–43 cạnh có hướng: cạnh tranh DoWhy (Cntf.) (số DAG khác nhau của 10 nút là 4,17×10¹⁸, nên việc nhận diện đồ thị là khó về tổ hợp); (iii) **nhiễu cộng lớn làm hiệu năng giảm** (sai số không thể giảm) nhưng thêm dữ liệu có thể bù một phần, hiệu ứng mạnh hơn DoWhy (Cntf.); (iv) hiệu ứng nền (base-rate ATE): TabPFN v2 phân kỳ khi ATE lớn, Do-PFN ổn định, tuy nhiên cũng kém đi ở nhóm ATE lớn nhất, có thể do prior cho nhiều SCM ATE nhỏ (nhiễu cộng và phi tuyến "ép" hiệu ứng) — cần cách sinh dữ liệu có hiệu ứng mạnh giữ được qua chuỗi nhân quả dài.
- **Hiệu chỉnh độ bất định (PICP, phụ lục E.5):** nhìn chung **calibration tốt**. Ở case study "Unobserved Confounder" (không định danh), entropy đầu ra tăng đúng như lý thuyết, và PICP cho thấy mô hình hiệu chỉnh đúng mức tăng bất định đó. Ở các case study có thể định danh, mô hình **hơi thiếu tự tin (under-confident)**: nguyên nhân gồm sai số rời rạc hóa của phân phối thanh và loss chưa tối thiểu hoàn toàn; vì tối thiểu KL thuận nên dẫn tới underdispersion theo Murphy 2023, McNamara 2024. "Observed Mediator" cũng bất định cao vì nguồn ngoại sinh duy nhất là biến nhị phân nên hiệu ứng liên tục chỉ đến từ nhiễu cộng.
- **Ngoài phân phối prior (Hình 4):** bền với nhiễu **Laplace, Student-t, Gumbel** và hỗn hợp (dù prior chỉ dùng Gaussian), và với các hàm kích hoạt/phi tuyến sau khác; nhưng **nhân đôi phương sai nhiễu Gaussian làm hiệu năng giảm mạnh**, và thử trên trung bình điểm của các hàm dùng khi huấn luyện làm hiệu năng giảm rõ. Với đồ thị lớn hơn (tới 50 nút) vượt phạm vi, **v1 giảm đáng kể**; v1.1 (tiền huấn luyện đồ thị tới 60 nút) cải thiện rõ ở đồ thị 21–50 nút.
- **RealCause (Hình 5):** bản v1.1 mạnh trong ba trong bốn bộ; các bộ này được mô phỏng để unconfoundedness đúng tuyệt đối, vốn **có lợi cho baseline**, nhưng Do-PFN vẫn cạnh tranh. Trên đường Pareto tốc độ – độ chính xác, Do-PFN luôn nằm trên mặt Pareto ở cả bốn nhóm (Hình 8): chi phí tiền huấn luyện trả một lần, suy luận một lượt chạy xuôi.
- **Đồ thị đã biết (Hình 6):** Do-PFN v1 mạnh nhất trên **Law School**, không kém có ý nghĩa so với tốt nhất trên **Amazon Sales** (cấu trúc trung gian phức tạp), "vượt các phương pháp khác trong kịch bản trung gian".

## 6. Hạn chế

Tác giả nêu:
1. Khả năng khái quát hóa **phụ thuộc hoàn toàn vào prior SCM** có phản ánh đủ độ phức tạp nhân quả thực tế hay không; hiện chỉ kiểm chứng trên dữ liệu tổng hợp và bán tổng hợp (cần khảo sát hệ thống về độ bền khi prior không khớp thực tế và về phương pháp thiết kế/kiểm định prior). Tác giả lạc quan nhờ các bằng chứng khác rằng prior tổng hợp cho hiệu năng thật tốt (TabPFN v2, Hoo 2024, Reuter 2025) và có một số kết quả lý thuyết (Nagler 2023).
2. Lý thuyết bảo đảm cho suy luận khấu hao còn đang phát triển, khác với ước lượng không khấu hao khi biết cấu trúc nhân quả.
3. Phạm vi mới ở bước đầu: chỉ điều trị nhị phân, chưa có loại can thiệp rộng hơn, phản thực tế (counterfactual), dữ liệu không i.i.d. hay các đặc điểm/mô thức khác.

Nhận xét thêm của mình:
1. **Chưa có bằng chứng trên dữ liệu thật có ground truth can thiệp:** hai bộ "đồ thị đã biết" thực chất vẫn dựa trên mô phỏng bằng DoWhy từ đồ thị được thừa nhận, nên kết quả vẫn là "bán tổng hợp"; hợp lý với lập luận chính của bài rằng chưa có dữ liệu thật đủ tốt.
2. Nhiều so sánh mạnh (Hình 3) nằm trong chính phân phối của prior (SCM ANM với ReLU/tanh/bình phương, nhiễu Gaussian), nên lợi thế của Do-PFN có phần "sân nhà". Phần OOD giúp giảm lo ngại nhưng cho thấy vẫn nhạy với phương sai nhiễu và kích thước đồ thị.
3. Chỉ hiển thị biểu đồ hộp/CD, thiếu bảng số nên khó đối chiếu độ lớn thật của lợi thế.
4. **Điểm có ích cho đề tài:** bài chứng minh rằng độ bất định của PFN **đo được và có thể hiệu chỉnh tốt trong phân phối**, nhưng có thiên kiến thiếu tự tin do KL thuận và rời rạc hóa; và **kém đi rõ khi nhiễu lớn hơn mức prior nhìn thấy** (nhân đôi phương sai). Đây là bằng chứng gián tiếp cho giả thuyết rằng TabPFN có thể nhạy với nhiễu vượt phạm vi prior, đáng kiểm tra với nhiễu nhãn.

## 7. Liên hệ với bài Nature và đề tài

- Dùng đúng **kiến trúc TabPFN v2 và prior SCM** của bài Nature, thay mục tiêu từ "dự đoán y quan sát" sang "dự đoán y sau can thiệp". Dont-PFN so với Do-PFN tách bạch đóng góp của mục tiêu huấn luyện.
- Cho thấy **tính linh hoạt của prior**: chỉ cần thay bộ mô phỏng (thêm can thiệp, tăng kích thước đồ thị) là có khả năng mới; ngược lại, mọi giới hạn nằm ở prior. Phù hợp luận điểm bài 11 (position paper) về TFM.
- Về robustness (đề tài cuối kỳ): phân tích **ngoài phân phối** và **PICP** của bài là khuôn mẫu phương pháp tốt để làm theo khi kiểm tra TabPFN/TabICL dưới nhiễu nhãn hoặc mất cân bằng: (1) so sánh trong và ngoài phân phối prior, (2) kiểm tra calibration bằng độ phủ khoảng dự đoán, (3) phân biệt giữa thiếu dữ liệu và nhiễu không thể giảm.

## 8. Từ khóa cần nhớ

- **Do-PFN:** PFN dự đoán CID p(y | do(t), x) từ dữ liệu quan sát; không cần đồ thị, không cần unconfoundedness.
- **Prior:** SCM ngẫu nhiên (DAG, ANM, ReLU/tanh/x²), M^ob ∈ [10, 2200] điểm quan sát + điểm can thiệp cùng nhiễu; đồ thị ≤ 10 nút (v1, 48 giờ) hoặc ≤ 60 nút (v1.1, 96 giờ); 7,3 triệu tham số, 1 GPU RTX 2080.
- **Lý thuyết:** tối thiểu KL thuận theo prior; ba nguồn bất định (ngẫu nhiên, không định danh, nhận thức); nhất quán đến lớp tương đương Markov.
- **Kết quả:** vượt TabPFN v2, RF, Dont-PFN ở CID; vượt meta-learner/DML/DragonNet/TARNet ở CATE khi unconfoundedness vi phạm; calibration tốt nhưng hơi under-confident; bền với nhiễu không Gaussian nhưng nhạy phương sai nhiễu gấp đôi; Pareto-tối ưu tốc độ.
- **Hạn chế:** phụ thuộc prior; mới chỉ điều trị nhị phân; thiếu dữ liệu thật có ground truth.
