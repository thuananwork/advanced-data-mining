# Bài 36. OpenML Benchmarking Suites (OpenML-CC18)

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2021, Datasets and Benchmarks Track; PDF là arXiv 1708.03731v3 (22/11/2021) |
| Tác giả | Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael G. Mantovani, Jan N. van Rijn, Joaquin Vanschoren (xếp theo thứ tự chữ cái) |
| Mã nguồn / dữ liệu | Mã BSD-3 tại github.com/openml; gói Python `openml` (PyPI), R `OpenML` (CRAN), Java `org.openml.openmlweka` (Maven); trang suite: openml.org/s/99 (CC18), openml.org/s/{218, 269, 271} (các suite của AutoML Benchmark) |
| Vị trí trong đề tài | Nguồn của **OpenML-CC18**, một trong các benchmark mà bài Nature (bài 1) và TabPFN v1 (bài 3) dùng; cách tạo suite là nền của TabZilla (bài 33) và TabArena (bài 5). |

## 1. Bài này làm gì trong một đoạn

Nghiên cứu học máy cần benchmark thuật toán **diễn giải được, so sánh được và tái lập được**, nhưng thực tế mỗi bài chọn dữ liệu khác nhau, thí nghiệm quy mô nhỏ, khó tái lập, đôi khi mâu thuẫn. Bài đề xuất **OpenML benchmarking suite**: một tập các **tác vụ OpenML** (không chỉ tập dữ liệu) được chọn lọc kỹ để đánh giá thuật toán trong điều kiện xác định, cùng bộ công cụ để tạo, truy xuất, chạy và chia sẻ kết quả (Python, Java, R). Bằng bộ công cụ đó, nhóm tạo suite phân loại đầu tiên, **OpenML-CC18** (72 tác vụ phân loại), rồi điểm lại các ứng dụng của nó và của suite AutoML Benchmark.

## 2. Vấn đề và động cơ

- Benchmark là "ngọn hải đăng" cho nghiên cứu: giúp cộng đồng theo dõi tiến bộ và xác định vấn đề khó. Muốn vậy cần tập tác vụ **được thiết kế tốt và đủ khó**, dễ truy cập, đánh giá theo cách chuẩn hóa và kết quả phải **dễ tìm, dễ diễn giải, tái lập được, so sánh trực tiếp**.
- Kho dữ liệu đã có từ lâu (UCI, LIBSVM, UCR cho chuỗi thời gian, Mulan cho đa nhãn, PMLB, Kaggle, KEEL cho dữ liệu mất cân bằng và dữ liệu thiếu), nhưng chưa đem lại benchmark chuẩn hóa so sánh được giữa các nghiên cứu. Hậu quả: thí nghiệm nhỏ cần thận trọng khi diễn giải, khó tái lập, và thậm chí mâu thuẫn nhau. Một điểm hay bị chỉ trích: tư duy cạnh tranh, dồn sức "thống trị SOTA trên vài bộ dữ liệu" thay vì phân tích có hệ thống, kể cả kết quả âm khi thuật toán phổ biến thất bại.
- OpenML đã có dữ liệu và tác vụ, nhưng chưa có cách dễ dàng tạo và chia sẻ **bộ benchmark được thiết kế tốt** cùng kết quả chạy trên đó.

## 3. OpenML và khái niệm suite (mục 3–4)

OpenML (Vanschoren 2013) tổ chức mọi thứ thành bốn khối máy đọc được:
1. **Dữ liệu (data).**
2. **Tác vụ (task):** tham chiếu dữ liệu, kiểu tác vụ (phân loại/hồi quy), thuộc tính đích, **quy trình đánh giá** (k-fold CV, hold-out), **các phép chia cụ thể** và thước đo.
3. **Flow:** mô tả pipeline học máy giải tác vụ.
4. **Run:** kết quả (dự đoán, đánh giá) khi chạy một flow trên một tác vụ.

OpenML đã có truy cập lập trình tới mọi thứ, ghi nhật ký thí nghiệm đầy đủ và tự động chia sẻ; hàng triệu thí nghiệm công khai gắn với đúng dữ liệu, pipeline, siêu tham số.

**Định nghĩa:** *OpenML benchmarking suite* là tập các tác vụ OpenML được chọn cẩn thận để đánh giá thuật toán trong điều kiện chính xác. Dùng **tác vụ chứ không phải tập dữ liệu** làm đơn vị để thí nghiệm có thể so sánh và tái lập (vì quy trình đánh giá và phép chia đã cố định).

Ưu điểm so với tập dữ liệu tĩnh:
- Tạo benchmark dễ (lọc từ hàng nghìn bộ).
- Mỗi suite có **ID duy nhất**, trang riêng; **quyền sở hữu và tính bất biến** (chỉ người tạo mới sửa/xóa; khắc phục cơ chế gắn thẻ cộng đồng cũ cho phép ai cũng thêm tác vụ).
- Cộng đồng thực hành; mở rộng/lọc con suite có sẵn; tái lập dựa trên tác vụ máy đọc được; làm nghiên cứu benchmark bằng cách gắn run vào suite (tái dùng dữ liệu run); cộng tác báo lỗi dữ liệu.
- **Benchmark động:** vì dùng lâu sẽ bị quá khớp, suite có thể sửa và mở rộng (ví dụ hằng năm) để theo dõi tiến bộ theo thời gian.

## 4. Cách dùng (mục 5, Hình 2)

Các mở rộng nền tảng: khái niệm *benchmark suite* (tập tác vụ) và *benchmark study* (tập kết quả run); bộ lọc dữ liệu trong API và web để chỉ định ràng buộc; script và notebook hỗ trợ tạo và đánh giá chất lượng suite (ví dụ lọc bộ quá dễ); chú thích (tag) cộng tác và tự động cho các loại dữ liệu cần xử lý riêng (đa nhãn, chuỗi thời gian, nhân tạo).

Ba trường hợp dùng:
1. **Tạo suite mới:** xác định ràng buộc (kích thước tối thiểu, mức mất cân bằng lớp, không phải chuỗi thời gian...), quá trình lặp (xem phân phối tác vụ đang chọn và kết quả hiện có), mô tả mục tiêu và cân nhắc đạo đức trong trang suite.
2. **Truy xuất suite:** tải bằng ID hoặc bí danh qua thư viện khách; dữ liệu có siêu dữ liệu đầy đủ (ví dụ có giá trị thiếu thì đánh dấu để mã người dùng tự điều chỉnh).
3. **Chạy benchmark:** tích hợp sẵn với scikit-learn, mlr/mlr3, Weka (tích hợp học sâu đang phát triển); mã tùy chỉnh bọc qua giao diện scikit-learn. Kết quả có thể gói thành *study* và công bố lên OpenML (cần API key). Hình 2 là mã đầy đủ ba ngôn ngữ chạy cây quyết định độ sâu 1 trên mọi tác vụ CC18 và đăng kết quả.

## 5. OpenML-CC18 (mục 6, Bảng 1)

**72 tác vụ phân loại** từ các bộ dữ liệu được tuyển chọn trong hàng nghìn bộ của OpenML, thay thế trực tiếp cho nhiều bộ benchmark thông thường; chủ ý **cỡ trung bình** để thực tế. Là kế thừa của OpenML100 (100 bộ phân loại, ràng buộc lỏng hơn, có một số lỗi).

### 5.1. Tiêu chí thiết kế
Tất cả bộ dữ liệu đã kiểm chứng và có giấy phép công khai tới giữa năm 2018 thỏa mãn:
- (a) **500 ≤ số quan sát ≤ 100.000.**
- (b) **< 5.000 đặc trưng** sau one-hot (tránh sự cố bộ nhớ).
- (c) Nhãn có ≥ 2 lớp và **không lớp nào có < 20 quan sát** (đảm bảo đủ mẫu mỗi lớp trong CV 10 fold).
- (d) **Tỉ lệ lớp thiểu số / đa số > 0,05**, loại các bộ **mất cân bằng nặng** vì cần xử lý riêng cả thuật toán lẫn thước đo.
- (e) Không thưa (sparse).
- (f) Không phụ thuộc thời gian giữa các mẫu (chuỗi thời gian, luồng dữ liệu).
- (g) Không cần lấy mẫu theo nhóm (nhiều điểm cho một đối tượng phải vào cùng một phần chia).

Tiêu chí "có quan điểm" để tránh bộ có vấn đề:
- Cố gắng loại **dữ liệu nhân tạo** (khó đánh giá độ khó); loại cả bộ nghi ngờ là mô phỏng hay nhân tạo.
- Loại bộ là **tập con** của bộ lớn hơn (lựa chọn kích thước/biến/lớp con là chủ quan).
- Loại tác vụ có **nhãn bị biến đổi** (gộp lớp, rời rạc hóa biến liên tục).
- Loại bộ **không có nguồn hay tham chiếu** (tránh hộp đen).

Và để đảm bảo **đủ thách thức:** loại bộ có thể phân loại hoàn hảo bằng **một thuộc tính hay decision stump**, và bộ mà cây quyết định đạt 100% accuracy trong CV 10 fold (dấu hiệu bộ quá dễ).

### 5.2. Sai sót tự nhận (hậu kiểm)
- `sick` là phiên bản mới của `hypothyroid` với vài lớp bị gộp; `electricity` có đặc trưng liên quan thời gian; `balance_scale` là bộ nhân tạo; `mnist_784` cần gom mẫu theo người viết. Nhóm hứa sửa trong phiên bản sau và sàng lọc hơn 900 bộ mới đưa lên OpenML từ khi tạo CC18, đồng thời lên kế hoạch benchmark động cập nhật định kỳ.
- Một số bộ có thể có vấn đề đạo đức nhưng không tác động nếu dùng đúng mục đích (benchmark thuật toán, không xây mô hình triển khai thật).

### 5.3. Đặc điểm kết quả (Hình 3)
Phân phối điểm (AUC trung bình theo lớp, có trọng số) của **3,8 triệu thí nghiệm** với hàng nghìn pipeline chia sẻ trên các tác vụ CC18: có tác vụ khó hơn tác vụ khác, nhiều tác vụ có dải điểm rộng, và với tất cả tác vụ đều tồn tại mô hình kém (AUC 0,5).

### 5.4. Ứng dụng của CC18 (mục 6.2)
Nghiên cứu điền khuyết lặp (Van Wolputte và Blockeel), định lượng bất định của bộ phân loại (König et al.), mạng sâu học siêu đặc trưng (De Bie et al.), xem tác vụ đa lớp như bài toán bandit ngữ cảnh (Bibaut et al.), thử thuật toán phác thảo phân vị (Mitchell et al.). **Cardoso et al. (2021)** dùng Item Response Theory phân tích 60 bộ (trừ bộ lớn nhất): CC18 gồm cả bộ dễ và khó; **nhìn chung suite không quá thách thức** nhưng có nhiều bộ phù hợp để phân biệt phân loại tốt và kém, và đề xuất hai tập con (một khó, một mô phỏng suite đầy đủ). OpenML100 (tiền thân) được dùng để lập phân loại học thuật toán (Fabra-Boluda), đo tầm quan trọng siêu tham số (van Rijn và Hutter; Probst et al.) và chiến lược tinh chỉnh random forest.

## 6. Các suite OpenML khác (mục 7)

### 6.1. AutoML Benchmark Suite
Dùng chính công nghệ suite; một phần lấy từ CC18, phần khác từ cuộc thi AutoML (Guyon 2019) và các so sánh AutoML trước. So với CC18, **nới lỏng** vài ràng buộc vì giả định AutoML phải xử lý được sự phức tạp:
- Không giới hạn 100.000 mẫu hay 5.000 đặc trưng.
- **Không giới hạn mất cân bằng lớp** (framework tự dùng kỹ thuật như SMOTE).
- Gồm dữ liệu thưa (chuyển sang dày cho công cụ không hỗ trợ), và (bản mới) có bộ hồi quy.

Và **siết** vài ràng buộc: bộ lọc "quá dễ" xét thêm kết quả OpenML của nhiều learner; số bài toán ảnh bị giới hạn. Trước đó chuẩn gần nhất là 21 bài của Auto-WEKA (một phần ba là ảnh), nhưng không thành chuẩn: các bài cùng hội thảo dùng bộ dữ liệu khác nhau. AMLB (bài 2) dựa trên suite này (cho 71 phân loại + 33 hồi quy).

### 6.2. Khác
OpenML100-friendly (con của OpenML100 không thiếu và chỉ có đặc trưng số), Foreign Exchange. **Nhóm mời cộng đồng tạo suite cho các tác vụ khác, bộ lớn hơn hoặc nhiều chiều hơn, bộ mất cân bằng hay cực kỳ nhiễu, văn bản, chuỗi thời gian...**

## 7. Hạn chế và hướng tương lai (mục 8)
1. **Quá khớp benchmark:** chưa được chứng minh nhưng sẽ ngày càng có khả năng; giải pháp là cập nhật định kỳ và mời cộng đồng thêm tác vụ khó (như thị giác và NLP).
2. **Ghi công:** curation tốn công (nhóm kiểm tra và sửa tay CC18 nhiều tháng), cần cách để suite có thể được trích dẫn.
3. **Chưa có cách tự động hóa** tạo suite chất lượng cao; hậu kiểm như Cardoso et al. là cần thiết.
4. **Chi phí tính toán:** dù chọn cỡ trung bình, một số bộ vẫn quá nặng nên nhiều nhóm chỉ dùng tập con của CC18.
5. **Chưa đủ rộng:** chưa phục vụ mô hình lớn/dữ liệu lớn; chưa có ràng buộc tài nguyên (bộ nhớ, thời gian) trong đặc tả tác vụ và suite; chưa tích hợp datasheet cho đạo đức.
6. Bản thân CC18 cố ý **loại dữ liệu mất cân bằng nặng** (tỉ lệ thiểu số/đa số ≤ 0,05), **loại dữ liệu thưa, chuỗi thời gian, dữ liệu nhóm**; và không có thiết kế điều khiển nhãn nhiễu.

## 8. Liên hệ với bài Nature và đề tài
- CC18 là **nền chung** của hầu hết kết quả TFM đời đầu (TabPFN v1, bài Nature): vì lọc theo ≤ 100.000 mẫu, không mất cân bằng nặng, không thiếu theo yêu cầu, nó thuận lợi cho TFM.
- **Đây là bằng chứng gián tiếp cho khoảng trống nhãn nhiễu/mất cân bằng:** benchmark chủ lực của giai đoạn này **cố ý loại** bộ mất cân bằng nặng (> 1:20) và không điều khiển nhãn nhiễu, nên việc đánh giá TFM trên các điều kiện đó phải tự tạo (như phụ lục L.6 của TabICLv2 đã làm). Cần đối chiếu khi lập ma trận 4 SOTA.
- AMLB (bài 2), TabZilla (bài 33, 176 bộ trong đó có CC18) và TabArena (bài 5, tuyển chọn từ 1.053 bộ) đều thừa hưởng ý tưởng suite/tác vụ và bài học "benchmark phải sống, cần sửa lỗi".

## 9. Từ khóa cần nhớ
- **Benchmarking suite = tập tác vụ OpenML** (dữ liệu + quy trình đánh giá + phép chia + thước đo); **study = tập run**.
- **OpenML-CC18:** 72 tác vụ phân loại; 500–100.000 mẫu; < 5.000 đặc trưng; mỗi lớp ≥ 20 mẫu; **tỉ lệ thiểu số/đa số > 0,05**; không thưa, không chuỗi thời gian, không nhóm; loại dữ liệu quá dễ và nhân tạo.
- Tiền thân **OpenML100**; hậu duệ **AutoML Benchmark suite** (lỏng hơn về kích thước và mất cân bằng, có hồi quy).
