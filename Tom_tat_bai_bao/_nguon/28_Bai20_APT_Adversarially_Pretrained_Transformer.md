# Bài 20. Zero-shot Meta-learning for Tabular Prediction Tasks with Adversarially Pre-trained Transformer (APT)

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2025 (42nd International Conference on Machine Learning, Vancouver, PMLR 267); arXiv:2502.04573v2 [cs.LG], 10/6/2025 |
| Tác giả | Yulun Wu (UC Berkeley), Doron L. Bergman (Capital One) |
| Mã nguồn | Có trong phụ lục (mã và checkpoint cuối cùng, không chọn checkpoint theo kết quả đánh giá) |
| Lưu ý về PDF | Bảng 1 và Bảng 5 bị lệch cột khi trích chữ; mình ghép hạng, ROC-AUC theo thứ tự hàng (thứ tự thuật toán cố định theo hạng trung bình) và đối chiếu với văn bản. Cột thời gian chạy không đọc được rõ nên chỉ dùng ghi chú trong tóm tắt (thời gian trung bình dưới 1 giây). |
| Vị trí trong đề tài | Biến thể **cải tiến việc tạo dữ liệu tiền huấn luyện** của TabPFN (v1) bằng **tác tử dữ liệu đối kháng**, cộng một khối đầu ra không giới hạn số lớp. Cùng mạch với Mitra (bài 9) về "thiết kế prior", EquiTabPFN (bài 17) về số lớp. |

## 1. Bài này làm gì trong một đoạn

TabPFN được tiền huấn luyện trên dữ liệu sinh từ **MLP ngẫu nhiên thưa**: mỗi bộ dữ liệu là một MLP khởi tạo ngẫu nhiên, nên tập phân phối dữ liệu tổng hợp tuy rộng vẫn hạn chế (hai bộ sinh ngẫu nhiên độc lập cho phân phối giống nhau gần như hoàn toàn). Ngoài ra TabPFN không xử lý được số lớp vượt mức khi tiền huấn luyện (bị cố định bởi lớp dense cuối). Bài đề xuất **APT (Adversarially Pre-trained Transformer)**: (1) **tác tử dữ liệu đối kháng (adversarial data agents)**: một phần các bộ sinh dữ liệu là MLP được **cập nhật gradient để làm tăng sai số dự đoán của mô hình**, liên tục dịch chuyển phân phối dữ liệu sinh ra và thử thách mô hình; (2) **khối trộn (mixture block)**: đầu ra cho phân loại được thay bằng cơ chế "query-key" giữa mẫu kiểm tra và các mẫu huấn luyện, có cổng thưa, rồi **tổng cộng xác suất theo nhãn** (scatter-sum), nên **số lớp bất kỳ**. Trên 35 tập nhỏ của OpenML-CC18 (không lọc theo số lớp, đặc trưng hạng mục hay thiếu giá trị), APT đứng **hạng 1** với ROC-AUC trung bình 0,921 và thời gian chạy gần TabPFN (dưới 1 giây trung bình); và trên OpenML-CTR23 hồi quy, tiền huấn luyện đối kháng giúp TabPFN tốt hơn.

## 2. Bối cảnh

- Học máy meta (meta-learning) nhằm "học cách học" để thích nghi nhanh với tập mới. Tác giả định nghĩa **zero-shot meta-learning (ZSML)** theo nghĩa "không có bước tối ưu hóa nào trên tập hạ nguồn, không cần tập hạ nguồn giống tập tiền huấn luyện", khác ý nghĩa "zero-shot" trong mô hình ngôn ngữ/thị giác lớn (không thấy ví dụ nào). Công trình trước của Verma et al. (2020) dùng GAN có điều kiện thuộc tính lớp nhưng vẫn dựa vào phân phối giống tập huấn luyện.
- PFN (Müller 2021): transformer tiền huấn luyện trên dữ liệu tổng hợp lấy từ prior để suy luận Bayes xấp xỉ qua ICL; TabPFN (Hollmann 2022) là PFN cho bảng. Với dữ liệu bảng, GBDT (CatBoost, XGBoost, LightGBM) vẫn khó bị học sâu đánh bại trên bài toán một-mô-hình-một-tập.
- Hai điểm yếu của TabPFN theo bài: (1) phân phối dữ liệu tổng hợp hạn chế; (2) hạn chế số lớp là điểm yếu quan trọng của một ZSML vì lúc suy luận có thể gặp số lớp chưa từng thấy. Hạn chế số đặc trưng và kích thước dữ liệu có lời giải đồng thời (TabPFNv2, TabICL) và không mâu thuẫn với khối trộn (chỉ sửa lớp cuối).

## 3. Phương pháp

### 3.1. Tiền huấn luyện với bộ sinh dữ liệu thường và tác tử đối kháng
- Mỗi lần lặp, ghép lô gồm m bộ dữ liệu từ m bộ sinh {g₁,...,g_m}; mỗi bộ sinh tạo n mẫu. Theo TabPFN, đặc trưng và biến đáp là giá trị của các nơ-ron chọn ngẫu nhiên trong MLP thưa có nhiễu, cộng tiền xử lý (chuẩn hóa theo mẫu trong từng bộ dữ liệu, cắt ở 4 độ lệch chuẩn).
- **Bộ sinh thường** (ordinary data generator): MLP khởi tạo ngẫu nhiên (siêu tham số như số lớp, kích thước ẩn, hàm kích hoạt, dropout dùng làm thưa hóa, thang nhiễu lấy ngẫu nhiên).
- **Tác tử đối kháng** (adversarial data agent): MLP có tham số η được **cập nhật bằng gradient ascent** trên sai số dự đoán của mô hình q_θ: hàm mục tiêu L(g_η) = E_{X,y∼g_η} log q_θ(y_{l+1:n} | X_{l+1:n}, {X_{1:l}, y_{1:l}}); nghĩa là tác tử muốn sinh dữ liệu làm mô hình **sai nhiều hơn**, còn meta-learner muốn sai ít hơn. Hai mục tiêu đối nghịch trực tiếp nên gradient cho meta-learner chỉ là gradient của tác tử đổi dấu: **cả hai cập nhật được trong một lượt lan truyền ngược**, hiệu quả hơn huấn luyện đối kháng thông thường.
- **So với GAN:** tác tử là bộ sinh, meta-learner là bộ phân biệt, nhưng không có mẫu thật/giả; điểm khác then chốt là **sản phẩm cuối là bộ phân biệt (meta-learner)**, không phải bộ sinh.
- **Rời rạc hóa khả vi (soft discretization):** để lan truyền gradient qua đặc trưng hạng mục, tác giả mở rộng "ranking discretization" của TabPFN thành xấp xỉ liên tục theo ý Gumbel-Softmax/Concrete với siêu tham số nhiệt độ τ; giá trị liên tục được ánh xạ vào N_j hạng mục dựa trên N_j − 1 phân vị Gaussian lấy ngẫu nhiên lúc khởi tạo. Thử chọn nhiều nơ-ron làm logit lớp cho kết quả kém hơn nhiều nên giữ cách dùng một nơ-ron.
- **Đặt lại (reset) tác tử:** cứ mỗi N_e bước (2000 bước) tác tử bị khởi tạo lại. Mục đích: cân bằng giữa việc khám phá miền mới của khởi tạo ngẫu nhiên và việc khai thác của đối kháng; tránh hội tụ về cân bằng Nash hoặc bế tắc với meta-learner (như tập dữ liệu không còn tín hiệu); tác giả mô tả như để bộ phân biệt "định kỳ thắng nhẹ" bộ sinh.
- **Cấu hình:** 12,5% bộ sinh trong mỗi lô là tác tử đối kháng; learning rate của tác tử 10⁻¹, weight decay 10⁻⁵, nhiệt độ rời rạc hóa 10⁻², 2000 bước gradient giữa hai lần reset. Siêu tham số còn lại **thừa kế từ TabPFN, không tinh chỉnh**: learning rate 10⁻⁴, 12 khối, chiều ẩn 512, chiều feed-forward 1024, 4 đầu attention, batch hiệu dụng 64, tổng 6.400.000 bộ dữ liệu huấn luyện.

### 3.2. Khối trộn (mixture block) cho phân loại số lớp tùy ý
- Với mỗi điểm kiểm tra, embedding sau các khối transformer (q) truy vấn embedding của các điểm huấn luyện (K): hai MLP cho hai nhóm logit: (1) nhóm cho **trọng số softmax có tỉ lệ** (mức độ giống lớp của từng điểm huấn luyện); (2) nhóm lấy mẫu **cổng nhị phân mềm qua phân phối Concrete** để làm thưa các trọng số này. Xác suất lớp là **tổng cộng có cổng** (scatter-sum) các trọng số, chỉ mục theo nhãn gốc.
- **Diễn giải theo cây:** cổng quyết định tập con dữ liệu huấn luyện cùng "lá" với điểm kiểm tra (chia tách theo từng điểm, không chung), trọng số quyết định tầm quan trọng tương đối của từng nhãn trong lá; khác cây ở chỗ bỏ phiếu có trọng số, chia tách theo điểm. Không đổi độ phức tạp tính toán theo dữ liệu hay số chiều; chỉ bỏ lớp dense cuối và thêm hai khối multi-head attention + feed-forward. Thí nghiệm sơ bộ cho thấy **cổng thưa là then chốt** (không cổng, khối trộn kém).
- Hồi quy vẫn dùng lớp dense cuối thông thường (khối trộn chỉ dùng cho phân loại).

### 3.3. Thích nghi cho dữ liệu lớn (phụ lục B, đơn giản, không phải trọng tâm)
- **Nhiều mẫu:** ước lượng phân phối dự đoán sau bằng các lô con (tối đa 3000, theo TabZilla): p(y|X_test, D_train) ≈ q_θ(y|X_test, b)·p(b|D_train); phân loại trộn dự đoán theo tỉ lệ kích thước lô; hồi quy dùng ước lượng trung bình có trọng số nghịch phương sai.
- **Nhiều đặc trưng:** **patch embedding** cho số đặc trưng bất kỳ (chia đặc trưng thành các patch kích thước d*, đệm không patch cuối, qua dense, attention tùy chọn có mã hóa vị trí tương đối, rồi average pooling), nằm giữa dense toàn phần và token hóa từng đặc trưng.

## 4. Thiết lập thí nghiệm

- **Phân loại:** OpenML-CC18 (68 tập tabular sau khi loại 4 tập ảnh: mnist_784, CIFAR-10, Devnagari-Script, Fashion-MNIST). Kết quả chính trên **35 tập nhỏ (≤2000 mẫu)**; khác TabPFN gốc: **không lọc** theo số lớp (≤10), số đặc trưng (<100), đặc trưng hạng mục, hoặc thiếu giá trị (nên số tập tăng từ 18 lên 35); chia train/test 80/20 (thay vì 50/50). Với tập hơn 100 đặc trưng, lấy mẫu con 100 đặc trưng như TabZilla.
- **Hồi quy:** OpenML-CTR23 (35 tập; 12 tập nhỏ ≤2000 mẫu).
- **Đối chứng:** top GBDT của TabZilla (CatBoost, XGBoost, LightGBM), ResNet, SAINT, và 5 mô hình học máy chuẩn (SVM, LASSO-Logistic, Ridge-Logistic, Random Forest, KNN), cùng TabPFN. Ngân sách tính toán **1 giờ mỗi tập mỗi lần chia** (so với 10 giờ trong TabZilla). Kết quả của các baseline thừa hưởng từ TabPFN (2022) hoặc TabZilla.
- **Thước đo:** ROC-AUC một-chống-một (OVO) cho đa lớp; hạng và số lần thắng theo ROC-AUC trung bình trên các lần chia; hai kiểu độ lệch chuẩn (std của trung bình theo kiểu TabPFN; trung bình của std theo kiểu TabZilla).

## 5. Kết quả

### 5.1. Phân loại (Bảng 1, 35 tập nhỏ, 1 giờ)

| Thuật toán | Hạng trung bình | ROC-AUC trung bình |
|---|---|---|
| **APT** | **3,86** | **0,921** |
| CatBoost | 4,03 | 0,918 |
| TabPFN | 4,57 | 0,913 |
| SVM | 4,89 | 0,904 |
| XGBoost | 5,37 | 0,914 |
| LightGBM | 5,60 | 0,917 |
| LASSO-Logistic | 6,69 | 0,908 |
| Ridge-Logistic | 6,91 | 0,907 |
| Random Forest | 7,17 | 0,908 |
| ResNet | 7,69 | 0,825 |
| KNN | 9,57 | 0,884 |
| SAINT | 9,97 | 0,759 |

(Ghép theo thứ tự hàng của bảng gốc; bảng cũng có hạng trung vị, số lần thắng và thời gian.)

- APT đạt **SOTA với thời gian chạy như TabPFN** (trung bình dưới 1 giây theo tóm tắt). Lợi thế so với CatBoost nhỏ (AUC 0,921 so với 0,918; hạng 3,86 so với 4,03).
- ResNet và SAINT (học sâu theo quy trình học có giám sát chuẩn) **kém**, một phần do ngân sách 1 giờ thay vì 10 giờ.
- **Phân tích theo đặc điểm tập dữ liệu (Hình 4):** TabPFN sụt giảm đáng kể khi có **đặc trưng hạng mục hoặc giá trị thiếu**; APT **xử lý tốt hơn** hai trường hợp này (hạng cải thiện rõ ở các tập có hạng mục và có thiếu giá trị).

### 5.2. Phân tích định tính tác tử đối kháng
- Dữ liệu 2D sinh từ hai bộ sinh thường độc lập (2000 tập, 100.000 điểm) cho phân phối gần như trùng nhau; còn dữ liệu từ tác tử đối kháng **đa dạng hơn** và làm tản các đỉnh mật độ tập trung (Hình 3).
- **KL-divergence:** giữa hai bộ sinh thường: 0,134 ± 0,141; giữa bộ sinh thường và tác tử đối kháng: **0,813 ± 0,072** (tức tác tử sinh phân phối khác hẳn).
- Lo ngại tác tử sinh dữ liệu **không còn tín hiệu** không xảy ra: tương quan Pearson giữa đặc trưng và đáp trên dữ liệu 2D là **0,311 ± 0,026** (đối kháng) so với 0,268 ± 0,013 (thường), có thể một phần do tần suất reset cao.

### 5.3. Khối trộn: khái quát hóa số lớp và tốc độ tiền huấn luyện
- **Số lớp chưa thấy (Bảng 2):** APT tiền huấn luyện với tối đa 10 lớp, so với APT không có khối trộn nhưng tiền huấn luyện tới tối đa 26 lớp (mở rộng cho tập letter, isolet, vowel, texture trong OpenML-CC18 có hơn 10 lớp). APT khớp mô hình kia ở **3/4 tập** (ví dụ ROC-AUC của letter 0,975, texture 0,970 so với 0,939 ở một cấu hình so sánh). Nghĩa là không cần chọn số lớp tối đa theo tập đánh giá trước khi tiền huấn luyện, điều không hợp với tinh thần zero-shot.
- **Tốc độ học (Hình 5):** khối trộn giúp mô hình **học nhanh hơn rõ rệt**: ROC-AUC đạt 0,70 chỉ sau khoảng 40 bước gradient (theo văn bản), so với mô hình không có khối trộn chậm hơn nhiều, trong 30.000 bước đầu.
- **Chi phí:** khối trộn không làm mô hình nặng hơn đáng kể (nêu rõ thời gian huấn luyện trên H100 trong bài).

### 5.4. Ablation (Bảng 3 phân loại, Bảng 4 hồi quy)
- **Phân loại, tập nhỏ (ROC-AUC trung bình / hạng):** APT 0,921 / 2,11; **APT không khối trộn** 0,917 / 2,09; **TabPFN có khối trộn** 0,914 / 2,55; **TabPFN** 0,913 / 2,49. Trên toàn bộ tập: 0,918 / 0,917 / 0,914 / 0,914 (hạng 2,1; 2,1; 2,6; 2,4). Mô hình có và không có khối trộn **không áp đảo nhau** về AUC và hạng, đúng như dự kiến (khối trộn không nhằm tăng hiệu năng ở số lớp đã thấy); phần cải thiện chủ yếu đến từ **tiền huấn luyện đối kháng**.
- **Hồi quy (APT = TabPFN + tiền huấn luyện đối kháng, không khối trộn):** MSE trung bình trên 12 tập nhỏ: TabPFN 0,412 ± 0,077, số thắng 3,8; APT 0,344 ± 0,068, 8,2 thắng. Trên toàn bộ: TabPFN 0,340 ± 0,025, 6,4 thắng; APT **0,306 ± 0,023, 28,6 thắng**. Dùng đúng phân phối lấy mẫu dữ liệu và siêu tham số của TabPFN để tách riêng đóng góp của đối kháng; tác giả lưu ý các siêu tham số này có thể đã tối ưu quá cho phân loại.

## 6. Hạn chế

Tác giả nêu:
- **Không vượt GBDT trên tập dữ liệu lớn**; còn **độ phức tạp bậc hai** như TabPFN.
- Hướng mở rộng: dùng các cải tiến transformer/state-space cho ngữ cảnh dài; mở rộng khối trộn sang học có giám sát thường; khám phá thay lớp dense cuối trong hồi quy (có thể ảnh hưởng đến thiên kiến quy nạp của DNN so với GBDT).
- Kết quả chủ yếu trên **bảng nhỏ**, và việc chọn bộ siêu tham số thừa kế từ TabPFN chưa tối ưu cho hồi quy.
- Tính hữu ích của ngữ nghĩa văn bản ở LLM chỉ "có ý nghĩa nhưng không bảo đảm cho dữ liệu bảng rộng" (nhận xét trong phần liên quan).

Nhận xét thêm của mình:
1. Dựa trên **TabPFN v1**, không so sánh với TabPFN-v2, TabICL hay Mitra (xuất bản gần như đồng thời); khó biết đối kháng còn lợi thế khi prior đã giàu hơn (SCM, cây). Mitra (bài 9) đạt ý tương tự "đa dạng prior" bằng cách trộn cây, không đối kháng.
2. Khác biệt AUC giữa APT và CatBoost chỉ **0,003**; với độ lệch chuẩn khoảng 0,003 của trung bình, ý nghĩa thống kê của vị trí số một không mạnh, dù bài so sánh nhiều kiểu std.
3. Dữ liệu đánh giá hạn chế ở **35 tập nhỏ** và 12 tập hồi quy nhỏ; bản gốc dùng ngân sách 1 giờ thay vì 10 giờ nên kết quả baseline sâu bị bất lợi.
4. Cơ chế đối kháng "làm khó mô hình" gợi nhớ **nhiễu**, nhưng bài không kiểm tra độ bền với nhiễu nhãn/mất cân bằng thật; KL-divergence và tương quan là số liệu về dữ liệu sinh, không phải độ bền.

## 7. Liên hệ với bài Nature và đề tài

- **Thiết kế prior** là chủ đề thống nhất: bài Nature dùng SCM + cây; Mitra (bài 9) thêm tiêu chí đa dạng/khác biệt; APT thêm tác tử đối kháng; EquiTabPFN (bài 17) sửa đầu ra số lớp. Có thể đặt ba bài vào cùng một mục "cải tiến prior/kiến trúc cho TFM" trong báo cáo.
- **Hạn chế số lớp** mà APT giải bằng khối trộn cũng được EquiTabPFN giải bằng kiến trúc tương đẳng; TabPFNv2/TabICL giải bằng ECOC/phân cấp.
- **Với đề tài nhiễu/mất cân bằng:** tác tử đối kháng tạo ra dữ liệu "khó"; một câu hỏi tự nhiên là liệu prior đối kháng có làm mô hình bền hơn với nhiễu thực sự không. Chưa có bằng chứng.

## 8. Từ khóa cần nhớ

- **Zero-shot meta-learning (ZSML):** không tối ưu hóa nào trên tập hạ nguồn.
- **Adversarial data agents:** MLP sinh dữ liệu cập nhật gradient ascent trên sai số của mô hình; 12,5% bộ sinh; reset mỗi 2000 bước; một lượt lan truyền ngược cập nhật cả hai phía.
- **Mixture block:** truy vấn–khóa + cổng Concrete thưa + scatter-sum theo nhãn → số lớp bất kỳ, học nhanh hơn.
- **Số liệu:** 35 tập CC18 nhỏ; AUC 0,921, hạng 3,86; KL 0,813 so với 0,134; tương quan 0,311 so với 0,268; hồi quy MSE 0,306 so với 0,340, 28,6 thắng so với 6,4.
- **Hạn chế:** không vượt GBDT trên dữ liệu lớn; bậc hai; dựa trên TabPFN v1.
