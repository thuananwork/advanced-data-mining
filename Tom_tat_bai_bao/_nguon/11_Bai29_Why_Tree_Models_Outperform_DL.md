# Bài 29. Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2022 Datasets and Benchmarks Track (theo thông tin công bố của bài); PDF là arXiv 2207.08815v1 (18/7/2022), đề đầu trang ghi "Preprint. Under review" |
| Tác giả | Léo Grinsztajn (Soda, Inria Saclay), Edouard Oyallon (ISIR, CNRS, Sorbonne), Gaël Varoquaux (Soda, Inria Saclay) |
| Mã nguồn / dữ liệu | github.com/LeoGrin/tabular-benchmark: mã, bộ 45 dữ liệu trên OpenML và bảng kết quả thô của toàn bộ ~20.000 giờ tính toán tìm siêu tham số |
| Vị trí trong đề tài | Bài kinh điển giải thích **vì sao cây thắng mạng sâu** trên dữ liệu bảng; nền của câu hỏi "vì sao TFM có thể thắng" và của kiểu benchmark có tính đến chi phí tinh chỉnh. |

## 1. Bài này làm gì trong một đoạn

Bài làm hai việc. (1) Tạo **một benchmark chuẩn cho dữ liệu bảng**: 45 bộ dữ liệu chọn lọc theo tiêu chí rõ ràng, quy trình tinh chỉnh siêu tham số có tính biến thiên (random search ~400 vòng, mô phỏng nhiều ngân sách), đánh giá cả MLP, ResNet, FT-Transformer, SAINT và Random Forest, GradientBoosting, XGBoost. Kết quả: **mô hình cây vẫn là SOTA trên dữ liệu cỡ trung bình (~10.000 mẫu)**, kể cả khi không tính đến việc chúng huấn luyện nhanh hơn. (2) Điều tra bằng thực nghiệm **vì sao**: biến đổi dữ liệu để thu hẹp hoặc nới rộng khoảng cách giữa NN và cây, từ đó chỉ ra ba thiên kiến quy nạp (inductive bias) khác nhau, tạo thành ba thách thức cho NN chuyên cho bảng.

## 2. Vấn đề và động cơ

- Cây tập hợp (XGBoost, Random Forest) vẫn là công cụ ưa thích của người làm thực tế và các cuộc thi. NN bảng là hướng nghiên cứu sôi động, một phần vì cây **không khả vi** nên khó ghép và huấn luyện chung với các khối học sâu.
- Hầu hết bài báo NN bảng tuyên bố bằng hoặc hơn cây, nhưng độ tin cậy bị nghi ngờ: một ResNet đơn giản đã cạnh tranh được với nhiều mô hình mới (Gorishniy 2021), nhiều mô hình thất bại trên dữ liệu mới (Shwartz-Ziv và Armon 2021). Nguyên nhân: không có benchmark chuẩn cho dữ liệu bảng như ImageNet hay SuperGLUE, mỗi bài chọn dữ liệu và quy trình khác nhau; bộ dữ liệu bảng công khai thường nhỏ nên đánh giá nhiễu; công sức tinh chỉnh không đều (Lipton và Steinhardt 2019); không tính đến độ bất định thống kê (Bouthillier 2021).
- OpenML-CC18/CC100 và AutoML Benchmark có dữ liệu bảng nhưng lẫn ảnh và dữ liệu nhân tạo, nên ít được dùng trong nghiên cứu NN bảng.
- Các giải thích trước đó cho lợi thế của cây chỉ là suy đoán (Klambauer 2017, Borisov 2021); Kadra 2021 cho rằng MLP đủ biểu đạt nhưng thiếu điều chuẩn phù hợp. Bài này tuyên bố là **nghiên cứu thực nghiệm đầu tiên** về nguyên nhân.

## 3. Benchmark (mục 3)

### 3.1. Chọn 45 bộ dữ liệu
Tiêu chí:
- **Cột không đồng nhất** (loại ảnh, tín hiệu nơi mọi cột cùng bản chất).
- **Không quá nhiều chiều:** tỉ lệ d/n < 1/10.
- **Có tài liệu** (bỏ bộ ít thông tin, giữ bộ ẩn tên cột nếu rõ là đặc trưng không đồng nhất).
- **i.i.d.:** bỏ chuỗi thời gian, dữ liệu dạng luồng.
- **Dữ liệu thật** (bỏ dữ liệu nhân tạo, giữ vài bộ mô phỏng có ý nghĩa thực tế như Higgs).
- **Không quá nhỏ:** ≥ 4 đặc trưng và ≥ 3.000 mẫu.
- **Không quá dễ:** loại bộ nào mà Logistic/Linear Regression mặc định đạt điểm chênh lệch tương đối dưới 5% so với cả ResNet mặc định và HistGradientBoosting mặc định (dùng chênh lệch tương đối thay vì "cây đơn đạt hoàn hảo" vì Bayes rate mỗi bộ khác nhau).
- **Không tất định:** bỏ bộ mà nhãn là hàm tất định của dữ liệu (poker, cờ vua), vì rất khác dữ liệu thực.

### 3.2. Loại bỏ các vấn đề phụ (để tập trung vào thách thức riêng của dữ liệu bảng)
- **Cỡ trung bình:** cắt tập huấn luyện còn **10.000 mẫu** nếu lớn hơn; chế độ lớn (50.000 mẫu) xét trong phụ lục.
- **Không có giá trị thiếu:** bỏ cột thiếu nhiều, sau đó bỏ mọi hàng còn thiếu.
- **Lớp cân bằng:** phân loại nhị phân hóa bằng hai lớp đông nhất và lấy một nửa mẫu mỗi lớp.
- **Đặc trưng hạng mục có ≤ 20 giá trị;** đặc trưng số phải có ≥ 10 giá trị phân biệt (đặc trưng số có 2 giá trị đổi thành hạng mục).

### 3.3. Quy trình tinh chỉnh có tính biến thiên
- Random search ~**400 vòng** mỗi mô hình trên mỗi bộ dữ liệu (cây trên CPU, NN trên GPU), luôn bắt đầu từ siêu tham số mặc định. Không gian tìm kiếm lấy từ Hyperopt-Sklearn, bài gốc của mỗi mô hình, và Gorishniy (MLP, ResNet, XGBoost).
- Để nghiên cứu hiệu năng theo ngân sách tìm kiếm: với n vòng đầu, chọn cấu hình tốt nhất trên validation rồi đánh giá trên test; **lặp 15 lần với thứ tự random search bị xáo** để ước lượng kiểu bootstrap kỳ vọng điểm test.
- NN chạy 300 epoch với early stopping và checkpointing (kiên nhẫn 40 cho MLP, ResNet, FT-Transformer; 10 cho SAINT).
- **Chia dữ liệu:** 70% train (hoặc đến mức tối đa kích thước cho phép), phần còn lại 30% làm validation và 70% làm test, validation/test cắt tối đa 50.000 mẫu. Số fold phụ thuộc kích thước test (1 fold nếu hơn 6.000 mẫu; 2 fold nếu 3.000–6.000; 3 fold nếu 1.000–3.000; 5 fold nếu dưới 1.000).

### 3.4. Tổng hợp qua các bộ dữ liệu
Dùng accuracy (phân loại) và R² (hồi quy), chuẩn hóa theo kiểu **ADTM** (distance to the minimum) nhưng điểm thấp nhất lấy là **phân vị 10% (phân loại) hoặc 50% (hồi quy)** của lỗi test, vì điểm tệ nhất thường do mô hình ngoại lai; điểm R² âm được cắt về 0.

### 3.5. Chuẩn bị dữ liệu tối thiểu
Với NN: **Gaussian hóa đặc trưng** bằng QuantileTransformer; hồi quy: log-biến đổi mục tiêu khi đuôi nặng (và thêm siêu tham số Gaussian hóa mục tiêu); mô hình không xử lý hạng mục tự nhiên thì dùng OneHotEncoder.

## 4. Mô hình được so sánh (mục 4.1)
- **Cây:** Random Forest, GradientBoostingTrees (HistGradientBoosting khi có đặc trưng hạng mục) của scikit-learn, XGBoost.
- **Học sâu:** MLP (Gorishniy, thêm ReduceOnPlateau), ResNet (MLP + dropout, chuẩn hóa, skip connection), **FT-Transformer** (trường hợp "tốt nhất" cho NN theo Gorishniy), **SAINT** (Transformer có attention giữa các mẫu, mô hình sâu tốt nhất trong tổng quan Borisov).

## 5. Kết quả benchmark (mục 4.2)

Hình 1 (số liệu) và Hình 2 (cả số và hạng mục) chia ra: phân loại 15 bộ và hồi quy 19 bộ (chỉ đặc trưng số), phân loại 7 bộ và hồi quy 14 bộ (có đặc trưng hạng mục), cỡ trung bình.

- **Tinh chỉnh siêu tham số không biến NN thành SOTA:** mô hình cây tốt hơn ở **mọi ngân sách random search**, khoảng cách vẫn lớn ngay cả sau nhiều vòng. Điều này chưa tính rằng mỗi vòng của NN chậm hơn; khi vẽ theo **thời gian** (Hình 7, 8 ở phụ lục) ưu thế của cây còn rõ hơn: sau một thời gian tìm kiếm như nhau, điểm của cây luôn cao hơn xa NN, dù NN và cây gần nhau ở vài bộ sau ít vòng.
- **Đặc trưng hạng mục không phải điểm yếu chính của NN:** chỉ dùng đặc trưng số thì khoảng cách có hẹp hơn một chút so với khi có hạng mục, nhưng **phần lớn khoảng cách vẫn còn**.
- **Chế độ lớn (50.000 mẫu, phụ lục A.2, Hình 9–12):** chỉ giữ được ít bộ (phân loại 4 bộ + 2 bộ, hồi quy 3 bộ + 5 bộ), nên so sánh chưa rõ ràng; nhìn chung **tăng kích thước tập huấn luyện làm thu hẹp khoảng cách** giữa NN và cây. Tác giả để dành nghiên cứu chặt cho tương lai.

## 6. Vì sao cây thắng? Ba phát hiện (mục 5)

Phương pháp: áp dụng các biến đổi dữ liệu làm thu hẹp hoặc nới rộng khoảng cách hiệu năng, giới hạn ở phân loại, đặc trưng số, cỡ trung bình để đơn giản. Hai thuộc tính chung của phương pháp tốt trên dữ liệu bảng là: là **ensemble** (bagging hoặc boosting) và học yếu là **cây quyết định**; điểm mấu chốt dường như là yếu tố "cây".

**Phát hiện 1. NN thiên về nghiệm quá trơn (Hình 3).** Làm trơn hàm mục tiêu trên tập huấn luyện bằng bộ làm trơn nhân Gaussian (ma trận hiệp phương sai = hiệp phương sai dữ liệu × lengthscale bình phương; chỉ giữ 5 đặc trưng quan trọng nhất theo Random Forest và ước lượng hiệp phương sai bằng MinCovDet bền vững với ngoại lai). Khi lengthscale nhỏ, **làm trơn hạ thấp rõ rệt độ chính xác của mô hình cây nhưng hầu như không ảnh hưởng NN**. Kết luận: hàm mục tiêu trong các bộ này **không trơn**, NN khó khớp các hàm bất thường, phù hợp với *spectral bias* (NN thiên về hàm tần số thấp, Rahaman 2019); mô hình cây học hàm hằng theo từng đoạn nên không có thiên kiến này. Ví dụ (Hình 20): trên bộ electricity với 2 đặc trưng quan trọng nhất, Random Forest đạt độ chính xác huấn luyện hoàn hảo và test **85%**, còn MLP mặc định **80%**; Random Forest học được các mẫu bất thường theo trục ngày mà MLP bỏ qua. Phát hiện này gợi ý giải thích cho lợi ích của kích hoạt ExU (Neural-GAM) và của embedding tuần hoàn / binning theo mục tiêu (Gorishniy 2022).

**Phát hiện 2. Đặc trưng vô ích làm hại các NN kiểu MLP (Hình 4–5).**
- Bộ bảng chứa nhiều đặc trưng vô ích: loại tới 50% đặc trưng ít quan trọng nhất (theo Random Forest) thì độ chính xác của GBT thay đổi không nhiều, và GBT huấn luyện trên chính các đặc trưng bị loại có điểm rất thấp (đến 20% loại, và vẫn thấp đến 50%), tức đó là đặc trưng vô ích chứ không chỉ dư thừa.
- **Loại** đặc trưng vô ích thu hẹp khoảng cách giữa MLP (ResNet) và các mô hình khác (FT-Transformer, cây); **thêm** đặc trưng vô ích (mẫu từ Gauss chuẩn, không tương quan với nhãn và đặc trưng khác) làm khoảng cách nới rộng. MLP kém bền với đặc trưng vô ích hơn.

**Phát hiện 3. Dữ liệu không bất biến theo phép quay, nên quy trình học cũng không nên (Hình 6).**
- MLP là bất biến quay theo nghĩa của Ng (2004), và Ng đã chứng minh mọi thủ tục học bất biến quay có độ phức tạp mẫu **tệ nhất tăng ít nhất tuyến tính theo số đặc trưng vô ích**: muốn bỏ đặc trưng vô ích, thuật toán trước hết phải tìm lại hướng gốc của các đặc trưng, nên thông tin ở hướng trục bị mất.
- Thí nghiệm **quay ngẫu nhiên** dữ liệu: chỉ ResNet là bất biến quay (như dự đoán); quay làm **đảo ngược thứ hạng**: NN vượt cây và ResNet vượt FT-Transformer. Nếu bỏ nửa đặc trưng kém quan trọng trước khi quay, hiệu năng của mọi mô hình trừ ResNet giảm, nhưng ít hơn so với khi quay toàn bộ.
- Diễn giải: giống thị giác, có một **cơ sở tự nhiên (hệ trục gốc)** mã hóa thiên kiến dữ liệu; đặc trưng bảng thường có nghĩa riêng (tuổi, cân nặng), nên trộn tuyến tính chúng làm hỏng thông tin. Điều này giải thích vì sao thêm lớp **embedding cho đặc trưng số** (Somepalli 2021, Gorishniy 2022) cải thiện NN: lớp đó phá tính bất biến quay, và nhiều loại embedding khác nhau đều có ích cho thấy bản thân việc phá bất biến là phần quan trọng. Hướng tương lai: tìm cách phá bất biến quay rẻ hơn embedding.

**Ba thách thức cho NN bảng (kết luận):** (1) bền với đặc trưng vô ích, (2) giữ hướng của dữ liệu, (3) học dễ dàng các hàm bất thường.

## 7. Hạn chế (tác giả nêu và suy ra)
1. Phạm vi thu hẹp có chủ ý: **cỡ trung bình (≤ 10.000 mẫu huấn luyện)**, bỏ dữ liệu thiếu, ép **lớp cân bằng** (nhị phân hóa), bỏ đặc trưng hạng mục cardinality cao; vì vậy kết luận **không áp dụng thẳng cho mất cân bằng lớp, giá trị thiếu, nhãn nhiễu**. Tác giả tự nêu các câu hỏi mở: bộ rất nhỏ, rất lớn, dữ liệu thiếu, đặc trưng hạng mục cardinality cao.
2. Chỉ so MLP, ResNet, FT-Transformer, SAINT và ba mô hình cây; **không có mô hình nền tảng bảng**, mô hình ICL (TabPFN ra đời gần cùng thời, chưa được xét), hay các NN đặc thù khác (TabNet, NODE...).
3. Chỉ có **một mô hình nhiều bộ** theo thiết kế, nên với chế độ lớn chỉ có rất ít bộ (4–8 bộ mỗi thiết lập), khó kết luận.
4. Các biến đổi (làm trơn, quay) là thí nghiệm kiểm soát, không chứng minh nhân quả hoàn toàn; việc làm trơn chỉ dùng 5 đặc trưng quan trọng nhất để khả thi.
5. So sánh thời gian chạy không chặt chẽ vì dùng nhiều loại GPU/CPU khác nhau (phụ lục A.3).

## 8. Liên hệ với bài Nature và đề tài
- Ba phát hiện này là **lý do** các mô hình nền tảng như TabPFN/TabICL có thể khác: chúng mã hóa thiên kiến "tập trung vào đặc trưng riêng lẻ" qua attention theo cột/hàng, và được tiền huấn luyện trên prior chứa nhiều đặc trưng vô ích và hàm bất thường (bài 3, 6).
- Benchmark của bài (45 bộ, ~10.000 mẫu) là kiểu bộ dữ liệu mà TFM hợp nhất; cả bài 33 (TabZilla) lẫn TabArena (bài 5) mở rộng cách làm này. Chú ý bài này **cân bằng lớp và bỏ dữ liệu thiếu**, trong khi đề tài final-term quan tâm tới nhãn nhiễu/mất cân bằng: đó là đúng những yếu tố mà benchmark này đã loại ra, nên là một khoảng trống để kiểm tra với TFM.

## 9. Từ khóa cần nhớ
- **45 bộ dữ liệu, ~10.000 mẫu huấn luyện, random search ~400 vòng, 15 lần xáo thứ tự;** ~20.000 giờ tính toán.
- **Ba phát hiện:** NN thiên về hàm trơn (spectral bias); MLP không bền với đặc trưng vô ích; dữ liệu không bất biến quay mà MLP thì có (Ng 2004).
- **Embedding số phá bất biến quay** là lý do chúng giúp NN bảng.
