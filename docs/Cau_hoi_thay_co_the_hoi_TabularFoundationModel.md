# Bộ câu hỏi thầy có thể hỏi về Tabular Foundation Model (TFM)

Dành cho: Nguyễn Thuận An · Môn Khám phá tri thức và Khai thác dữ liệu · Tháng 10/2026

**Cách dùng.** Thầy sẽ hỏi kỹ phần **khái niệm chung về TFM** trước (Phần A đến J). Câu hỏi riêng về **TabICLv2** chỉ đến sau khi đã trình bày xong TFM (Phần K). Mỗi câu có một **câu trả lời ngắn** (đủ để nói trong 20–30 giây) và, nếu cần, ý mở rộng. Chỗ nào chưa chắc chắn hoặc là số của tác giả bài báo thì có ghi rõ. Cuối tệp có danh sách **việc cần tự kiểm tra lại** trước khi bảo vệ.

Quy ước nguồn: [1] PFN (Müller et al., ICLR 2022) · [2] TabPFN (Hollmann et al., 2023) · [3] TabPFNv2 (Nature, 2025) · [4] TabICL (ICML 2025) · [5] TabICLv2 (ICML 2026) · [8] TabDPT (NeurIPS 2025) · [9] LoCalPFN (NeurIPS 2024) · [18] bài quan điểm về TFM · [19] cơ sở thống kê của PFN (Nagler, ICML 2023) · [20] TabArena.

---

## A. Khái niệm nền tảng

### A1. Dữ liệu dạng bảng là gì, vì sao nó "khó" hơn ảnh hay văn bản?
Dữ liệu bảng gồm các dòng (mẫu) và cột (thuộc tính) với **kiểu dữ liệu hỗn hợp** (số, phân loại, thiếu giá trị). Ảnh và văn bản có cấu trúc cục bộ ổn định (điểm ảnh gần nhau, từ gần nhau) nên mạng sâu khai thác tốt. Bảng thì **thứ tự cột không có ý nghĩa**, mỗi bảng có số cột và ý nghĩa cột khác nhau, kích thước thường nhỏ, nên rất khó dùng chung một mô hình cho nhiều bảng.

### A2. Mô hình nền tảng (foundation model) là gì? Tabular foundation model là gì?
Là mô hình **huấn luyện trước một lần trên lượng lớn dữ liệu đa dạng**, sau đó dùng lại cho nhiều bài toán mà không phải huấn luyện từ đầu. Với dữ liệu bảng, TFM được huấn luyện trước trên rất nhiều bảng khác nhau (chủ yếu là bảng giả lập), khi gặp bảng mới chỉ cần đưa các mẫu có nhãn vào là **dự đoán ngay** (xem A4 và D).

### A3. "Base" của TFM là gì? Nó xuất phát từ ý tưởng nào?
Nền của TFM là **Prior-Data Fitted Network (PFN)** của Müller và cộng sự [1]: một **Transformer** được huấn luyện để xấp xỉ **suy luận Bayes** trên rất nhiều bài toán giả lập. TabPFN [2] là ứng dụng đầu tiên của PFN cho phân loại bảng nhỏ. Mọi biến thể sau (TabPFNv2, TabICL, TabICLv2, TabDPT, LoCalPFN) đều xoay quanh khung "Transformer + học trong ngữ cảnh + huấn luyện trước". Về lý thuyết, bài [19] của Nagler giải thích vì sao cách làm này có cơ sở thống kê.

### A4. Nói đơn giản TFM hoạt động thế nào?
Ba bước: (1) **huấn luyện trước** trên rất nhiều bộ dữ liệu giả lập để học "cách học"; (2) khi có bảng mới, **đưa các mẫu đã có nhãn cùng mẫu cần dự đoán** vào mô hình; (3) mô hình **dự đoán trong một lượt chạy xuôi**, không cập nhật trọng số. Điều nó học không phải là một bộ dữ liệu cụ thể mà là **kỹ năng suy ra quy luật từ vài ví dụ**.

### A5. Vì sao lĩnh vực này nóng lên trong 3 năm gần đây?
Dữ liệu bảng rất phổ biến (y tế, tài chính) nhưng học sâu lâu nay thường không vượt được cây quyết định tăng cường. Từ 2022–2023, PFN và TabPFN cho thấy học trong ngữ cảnh khả thi; TabPFNv2 lên Nature 2025 thu hút cả học thuật và công nghiệp. Thêm vào đó, ý tưởng "huấn luyện trước một lần, dùng nhiều lần" từ mô hình ngôn ngữ, nhiều bài toán thật chỉ có ít nhãn, các bộ đánh giá (TALENT, TabArena) và mã nguồn mở đã chín muồi.

### A6. Vì sao có nhiều biến thể trong thời gian ngắn?
Mỗi biến thể nhắm vào **một điểm yếu của TabPFN gốc**: quy mô và tốc độ (TabICL, TabICLv2, TabFlex), dữ liệu giả lập chưa đủ thật (TabDPT, Mitra), mẫu ngữ cảnh chưa chọn lọc (LoCalPFN), chưa dùng tên cột và văn bản (CARTE, TabLLM, TabSTAR), thiếu lý thuyết (EquiTabPFN, Nagler).

### A7. Vì sao gọi là "foundation" khi nó chỉ dự đoán trên bảng?
Câu hỏi phản biện hợp lý. Trả lời an toàn: gọi là "nền tảng" theo nghĩa **huấn luyện trước một lần và tái sử dụng cho nhiều bảng**, giống tinh thần của mô hình ngôn ngữ. Tuy vậy nó **không "tổng quát" như mô hình ngôn ngữ lớn**: nó chuyên cho dự đoán trên bảng, và còn hạn chế về quy mô, văn bản, giải thích. Bài quan điểm [18] lập luận rằng đây là hướng nên ưu tiên nghiên cứu, nhưng đó là một **quan điểm**, chưa phải kết luận đã được chứng minh.

---

## B. Dữ liệu giả lập: "tự sinh ra hàng triệu bảng giả" là thế nào?

### B1. Dữ liệu có sẵn mà sao lại tự sinh ra hàng triệu bảng giả?
Vì **dữ liệu bảng thật rất ít và rất khác nhau**, lại bị ràng buộc bản quyền và riêng tư, nên không đủ để huấn luyện một mô hình lớn. Bảng giả thì **sinh được vô hạn, đa dạng và kiểm soát được**. Điều quan trọng là mô hình không cần nhớ một bộ dữ liệu nào, nó cần học **kỹ năng suy ra quy luật từ vài mẫu**, và kỹ năng đó có thể học từ các quy luật sinh ngẫu nhiên.

### B2. Bảng giả được sinh ra như thế nào?
Theo mã nguồn của TabICLv2 (thư mục `src/tabicl/prior/`):
1. **Tạo quy luật ẩn ngẫu nhiên**, gọi là Structural Causal Model (SCM, mô hình nhân quả cấu trúc). Chương trình lấy vài biến gốc ngẫu nhiên rồi cho đi qua một **mạng nơ-ron nhỏ** (trọng số, hàm kích hoạt, nhiễu đều ngẫu nhiên) hoặc qua **các lớp cây** (cây quyết định, rừng ngẫu nhiên, XGBoost). Cấu hình mặc định cho thấy tỉ lệ khoảng 70% nhánh mạng nơ-ron và 30% nhánh cây.
2. **Lấy cột và nhãn** từ các giá trị trung gian và cuối của quy luật đó, nên giữa các cột có quan hệ nhân quả như dữ liệu thực.
3. **Biến thành bài toán phân loại**: nhãn liên tục được cắt theo ngưỡng hoặc thứ hạng thành các lớp; một phần cột (mặc định khoảng 20%) được chuyển thành biến phân loại.
4. **Xáo trộn ngẫu nhiên** thứ tự cột và nhãn để mô hình không học thuộc vị trí.

Lưu ý: đây là cách của **TabICLv2**; bộ sinh của TabPFNv2 có cùng tinh thần nhưng chi tiết khác, mình chưa đối chiếu từng chi tiết.

### B3. "Prior" (tiên nghiệm) ở đây nghĩa là gì?
Là **phân phối của tất cả các quy luật sinh dữ liệu mà ta cho là có thể xảy ra**. Việc chọn prior chính là chọn "thế giới giả định" mà mô hình sẽ học trong đó. Prior càng giống dữ liệu thật thì mô hình càng dự đoán tốt trên dữ liệu thật. Đó là lý do bài Mitra [6] nhấn mạnh vai trò của quy luật dùng để sinh dữ liệu.

### B4. Dữ liệu giả có thể khác dữ liệu thật, vậy rủi ro là gì?
Nếu bảng thật khác xa các quy luật đã sinh thì chất lượng giảm. Hai cách xử lý hiện nay: (a) làm prior **đa dạng hơn** (TabPFNv2, TabICLv2); (b) **huấn luyện trước trên bảng thực** (TabDPT [8]). Đổi lại, dùng bảng thực làm dữ liệu huấn luyện trước làm tăng nguy cơ **trùng dữ liệu với tập kiểm tra** (rò rỉ), cần kiểm tra kỹ.

### B5. Dữ liệu giả có giúp tránh rò rỉ không?
Giúp **giảm nguy cơ** vì mô hình chưa từng thấy bảng đánh giá trong lúc huấn luyện trước, nhưng không loại trừ hoàn toàn nếu prior vô tình giống một bảng cụ thể. Câu này là nhận định của mình, các bài báo không khẳng định tuyệt đối.

---

## C. PFN và suy luận Bayes

### C1. PFN là gì?
**Prior-Data Fitted Network**: mạng được huấn luyện trên dữ liệu rút từ một prior, sao cho khi xem một tập mẫu có nhãn, nó xuất ra **xác suất dự đoán** cho mẫu mới, xấp xỉ phân phối dự đoán hậu nghiệm Bayes.

### C2. Công thức Bayes ở slide 5 nghĩa là gì?
Nói bằng lời: xác suất nhãn của mẫu mới **bằng trung bình, theo mọi "quy luật" h có thể có, của xác suất nhãn dưới quy luật h, có trọng số là mức độ quy luật đó phù hợp với các mẫu đã thấy**.
`p(y* | x*, D) = ∫ p(y* | x*, h) · p(h | D) dh`
(D: tập mẫu có nhãn; x*: mẫu cần dự đoán; h: quy luật sinh dữ liệu). **Công thức chỉ để minh họa, không cần giải thích sâu.** PFN không tính tích phân này trực tiếp mà học xấp xỉ nó bằng mạng.

### C3. Hàm mất mát khi huấn luyện trước là gì?
Là **log-likelihood âm** (cross-entropy) của nhãn đúng trên rất nhiều bảng giả: mỗi lần bốc một bảng từ prior, tách ra phần ngữ cảnh và phần cần dự đoán, tối ưu để mô hình dự đoán đúng phần sau. Khi dùng (suy luận) thì **không cập nhật trọng số**.

### C4. Vì sao dự đoán "một lượt" mà vẫn giống học?
Vì quá trình "học" đã được **nén vào việc huấn luyện trước**: mô hình học một thuật toán học tổng quát. Lúc dùng, các mẫu có nhãn trong ngữ cảnh đóng vai trò dữ liệu huấn luyện, và lượt chạy xuôi của Transformer thực hiện thuật toán đó. Đây là điều gọi là **học trong ngữ cảnh**.

---

## D. Học trong ngữ cảnh (In-Context Learning, ICL)

### D1. ICL khác huấn luyện/tinh chỉnh thông thường chỗ nào?
Huấn luyện thông thường **cập nhật trọng số** bằng gradient trên bảng đích. ICL **giữ nguyên trọng số**, chỉ đưa mẫu có nhãn vào làm ngữ cảnh. Hệ quả: nhanh, không cần chỉnh tham số, nhưng chi phí tăng theo số mẫu ngữ cảnh. (LoCalPFN là ngoại lệ: có thêm bước tinh chỉnh.)

### D2. Hàm `fit()` trong thư viện làm gì nếu không huấn luyện?
Chủ yếu **lưu và tiền xử lý dữ liệu huấn luyện** (và tải checkpoint nếu cần); phần tính toán chính nằm ở `predict`, khi mô hình đọc ngữ cảnh cùng mẫu cần dự đoán. Vì vậy thời gian "fit" rất ngắn, ngoại trừ lần đầu tải trọng số (lý do con số 25,87 giây của TabICLv2 ở lượt chạy đầu trong thử nghiệm của bạn).

### D3. Số mẫu trong ngữ cảnh có giới hạn không? Vì sao?
Có, vì cơ chế chú ý có chi phí tăng theo bình phương số dòng. Bài TabPFN gốc và TabPFNv2 đánh giá tới khoảng 10.000 mẫu và 500 đặc trưng [3]; TabICL và TabICLv2 được thiết kế để mở rộng hơn (TabICLv2 theo tác giả xử lý cỡ triệu mẫu trong 50 GB bộ nhớ GPU).

### D4. Thành phần mẫu trong ngữ cảnh có quan trọng không?
Có thể rất quan trọng. LoCalPFN [9] dựa trên nhận xét rằng **không phải mẫu nào cũng hữu ích**, nên chọn các mẫu gần mẫu cần dự đoán. Trong thử nghiệm của bạn, độ chính xác TabPFNv2 giảm khi tăng từ 50 lên 100 mẫu (97,37% xuống 95,61%) là một gợi ý, nhưng chỉ trên một seed nên chưa đủ kết luận.

---

## E. Kiến trúc và chi phí

### E1. Cơ chế chú ý (attention) dùng để làm gì ở đây?
Để mô hình **so sánh mẫu cần dự đoán với các mẫu trong ngữ cảnh** (chú ý giữa các dòng) và, ở TabPFNv2, còn xét quan hệ **giữa các cột** (chú ý hai chiều). Có thể hình dung attention như một cách học độ tương đồng.

### E2. Vì sao chi phí tăng nhanh theo số dòng và số cột?
Chú ý giữa các dòng cần so sánh từng cặp dòng nên chi phí tăng theo **bình phương số dòng**; chú ý hai chiều thêm chi phí theo bình phương số cột. Các ký hiệu bậc phức tạp trên slide 6 chỉ là **ước lượng lý thuyết**, không phải thời gian đo thực.

### E3. TabPFN, TabPFNv2 và TabICL khác nhau thế nào về kiến trúc?
- **TabPFN 2023:** mỗi dòng là một đầu vào, Transformer chú ý giữa các dòng, hợp bảng nhỏ.
- **TabPFNv2:** chú ý hai chiều (dòng và cột), xử lý biến phân loại, giá trị thiếu, cả hồi quy.
- **TabICL:** tách ba bước: TFcol (biểu diễn từng cột) → TFrow (gộp thành vector dòng) → TFicl (học trong ngữ cảnh). Nhờ vậy không phải tính lại quan hệ giữa các cột ở bước dự đoán.

### E4. Kiến trúc phức tạp hơn có chắc chính xác hơn không?
Không. Mỗi kiến trúc giải quyết một vấn đề khác nhau (bảng nhỏ, bảng đa dạng, bảng nhiều dòng). Phải so sánh **trong cùng điều kiện** mới kết luận được.

### E5. Thứ tự cột và thứ tự dòng có ảnh hưởng không?
Với bảng, thứ tự không mang ý nghĩa, nên mô hình cần **bất biến theo hoán vị** hoặc xấp xỉ điều đó (thường qua xáo trộn ngẫu nhiên khi huấn luyện và khi suy luận). Bài EquiTabPFN [7] nghiên cứu cụ thể tính bất biến theo hoán vị nhãn. Đây là chi tiết kỹ thuật, nếu bị hỏi sâu có thể nói mình chưa đi sâu vào phần này.

---

## F. So sánh với phương pháp khác

### F1. TFM có thật sự tốt hơn cây quyết định (XGBoost, CatBoost) không?
Theo các bài báo, TabPFNv2 và TabICLv2 đạt **bằng hoặc tốt hơn** cây đã chỉnh tham số trên nhiều bộ, đặc biệt khi ít mẫu và không cần chỉnh gì. Tuy nhiên đây là số của tác giả, và trên bảng rất lớn cây vẫn là đối thủ mạnh và rẻ hơn. Trong thử nghiệm của nhóm trên bộ Wisconsin, chênh lệch rất nhỏ nên chưa kết luận được.

### F2. Cây quyết định liên quan gì đến TabPFN?
Không có cây bên trong TabPFN: nó là Transformer. Cây xuất hiện ở hai chỗ: (1) là **đối thủ và mốc so sánh**; (2) là **một nguồn sinh dữ liệu giả lập** (mã sinh dữ liệu của TabICLv2 có tệp `_tree_scm.py`, bên cạnh `_mlp_scm.py` và `_graph_scm.py`).

### F3. TFM khác kNN hay mạng nơ-ron huấn luyện riêng ở điểm nào?
kNN dùng độ đo khoảng cách cố định, TFM **học** cách so sánh mẫu từ rất nhiều bài toán. Mạng nơ-ron huấn luyện riêng phải học lại từ đầu trên từng bảng (cần nhiều mẫu và chỉnh tham số), còn TFM dùng kiến thức đã học trước. LoCalPFN [9] là ví dụ kết hợp: tìm mẫu gần (giống kNN) **cộng** tinh chỉnh.

### F4. Khi nào nên dùng TFM, khi nào nên dùng cây?
Nên cân nhắc TFM khi **ít nhãn, muốn kết quả nhanh mà ít chỉnh tham số**. Cây vẫn hợp khi bảng rất lớn, cần độ trễ thấp, cần giải thích, hoặc tài nguyên hạn chế. Cần đo thêm độ ổn định, chất lượng xác suất, tài nguyên và bảo mật dữ liệu trước khi triển khai.

### F5. Vì sao các mô hình đơn giản như hồi quy logistic cũng ngang trong thử nghiệm của em?
Vì bộ Breast Cancer Wisconsin **khá dễ** (đặc trưng số, hai lớp, tách khá rõ). Trên bộ dễ, mô hình đơn giản cũng gần đạt trần, nên không phân biệt được mô hình nền tảng với mô hình thường. Đây là lý do cần thêm nhiều bộ dữ liệu ở cuối kỳ.

---

## G. Đánh giá và so sánh công bằng

### G1. Vì sao không so sánh thẳng các số công bố (0,939; 4,66; 0,884; 0,958)?
Chúng đo trên **bộ dữ liệu khác nhau** (AMLB, TALENT, OpenML-CC18, TabZilla) và bằng **thước đo khác nhau** (ROC-AUC chuẩn hóa, hạng trung bình, độ chính xác, IQM AUC). Muốn so công bằng phải cùng dữ liệu, cùng cách chia, cùng chỉ số và điều kiện tính toán tương đương.

### G2. ROC-AUC, Macro-F1, Log-Loss đo điều gì?
- **Độ chính xác:** tỷ lệ dự đoán đúng.
- **Macro-F1:** F1 từng lớp rồi lấy trung bình đều, nhạy với lớp thiểu số.
- **ROC-AUC:** khả năng phân biệt các lớp ở nhiều ngưỡng.
- **Log-Loss:** chất lượng xác suất dự đoán (thấp hơn là tốt hơn); phạt nặng khi tự tin mà sai.

### G3. ROC-AUC chuẩn hóa là gì? Hạng trung bình là gì?
ROC-AUC chuẩn hóa (như 0,939 của TabPFNv2) là điểm đã được **đưa về cùng thang giữa các bộ dữ liệu**, không phải ROC-AUC thông thường. Hạng trung bình là thứ hạng của mô hình trên từng bộ dữ liệu rồi lấy trung bình, **càng thấp càng tốt** (như 4,66 của TabICLv2 trên TALENT), và không cho biết hơn nhau bao nhiêu điểm.

### G4. Các bộ đánh giá TALENT, TabArena, AMLB, OpenML-CC18, TabZilla là gì?
Đều là **tập hợp nhiều bộ dữ liệu bảng chuẩn** để so sánh phương pháp. TALENT có 300 bộ (120 nhị phân, 80 nhiều lớp, 100 hồi quy), TabArena [20] có 51 bảng và là bảng xếp hạng cập nhật liên tục, AMLB là bộ đánh giá AutoML, OpenML-CC18 là 72 bộ phân loại tuyển chọn, TabZilla có 95 bộ lấy từ OpenML trong bài LoCalPFN.

### G5. Rò rỉ dữ liệu (data leakage) là gì và có xảy ra ở đây không?
Là khi dữ liệu kiểm tra (hoặc rất giống) đã xuất hiện trong dữ liệu huấn luyện, làm kết quả cao giả tạo. Nguy cơ này **lớn nhất với mô hình huấn luyện trước trên bảng thực** (TabDPT). Với mô hình huấn luyện trên dữ liệu giả lập thì thấp hơn. Cần kiểm tra trùng lặp giữa huấn luyện và kiểm tra, đây là khoảng trống G3.

### G6. Một bộ dữ liệu, một seed thì kết luận được gì?
Chỉ là **thử nghiệm sơ bộ**. Với 114 mẫu kiểm tra, 97,37% (111 đúng) và 95,61% (109 đúng) chỉ khác nhau hai dự đoán; khoảng tin cậy Wilson 95% lần lượt là 92,5%–99,1% và 90,1%–98,1%, **chồng lên nhau gần hết**. Vì vậy không nói được mô hình nào hơn.

### G7. Vì sao phải lặp nhiều seed và nhiều bộ dữ liệu?
Với ít mẫu, kết quả thay đổi theo cách chọn mẫu huấn luyện và cách chia tập. Nhiều seed cho **trung bình và độ lệch chuẩn**, nhiều bộ dữ liệu cho biết kết quả có **tổng quát** hay không.

---

## H. Hạn chế và vấn đề còn mở

### H1. Hạn chế lớn nhất của TFM hiện nay là gì?
(1) **Chi phí khi bảng lớn** (chú ý bậc hai); (2) **dữ liệu giả lập có thể khác dữ liệu thật**; (3) **độ tin cậy của xác suất** chưa được kiểm tra kỹ khi ít mẫu; (4) **khả năng giải thích** hạn chế; (5) chưa tận dụng tốt **tên cột và văn bản** (trừ nhóm CARTE, TabSTAR); (6) cần cấu hình phần cứng (tải checkpoint, đôi khi GPU).

### H2. Xác suất dự đoán của TFM có đáng tin không?
Chưa có đủ bằng chứng. Nhóm mới đo Log-Loss, **chưa đo sai số hiệu chỉnh kỳ vọng (ECE) và điểm Brier**, nên chưa dám nói xác suất đã đáng tin. Đây là hướng cần kiểm tra thêm.

### H3. TFM xử lý dữ liệu thiếu, biến phân loại, nhãn nhiễu, lớp mất cân bằng thế nào?
TabPFNv2 hỗ trợ giá trị thiếu và biến phân loại trong thiết kế, nhưng **độ bền của TFM trước nhãn nhiễu, mất cân bằng lớp và dữ liệu thiếu chưa được các bài báo đánh giá kỹ**. Đây chính là ngách mà nhóm dự định kiểm tra ở cuối kỳ.

### H4. Có rủi ro đạo đức hay riêng tư không?
Khi triển khai thực tế cần xem xét **bảo mật dữ liệu** (dữ liệu có đưa lên dịch vụ ngoài hay chạy cục bộ), **độ ổn định** khi dữ liệu thay đổi, và **khả năng giải thích** trong các lĩnh vực nhạy cảm như y tế. Các thực nghiệm của nhóm chạy cục bộ trên dữ liệu công khai.

### H5. Những khoảng trống nghiên cứu nào bạn thấy?
Sáu hướng nêu trong báo cáo: (G1) ít nhãn, nhiều lần chia và khoảng tin cậy; (G2) khác biệt dữ liệu, thử lĩnh vực mới; (G3) rò rỉ dữ liệu; (G4) hiệu quả: độ chính xác cùng thời gian và bộ nhớ; (G5) độ tin cậy của xác suất và dữ liệu biến động; (G6) ảnh hưởng của tên cột. Ngách dự kiến cuối kỳ: **độ bền trước nhãn nhiễu, mất cân bằng lớp và dữ liệu thiếu**.

---

## I. Bốn phương pháp tiên tiến (tóm tắt để trả lời nhanh)

| | Năm / nơi đăng | Vấn đề | Ý tưởng | Số công bố |
|---|---|---|---|---|
| **LoCalPFN** [9] | NeurIPS 2024 | TabPFN tốn bộ nhớ, học thiếu khi dữ liệu phức tạp | Tìm mẫu gần nhất làm ngữ cảnh và tinh chỉnh; chỉ phân loại | IQM AUC 0,958 (95 bộ TabZilla), so với 0,917 TabPFN và 0,943 XGBoost |
| **TabPFNv2** [3] | Nature 2025 | TabPFN gốc chỉ hợp bảng nhỏ, số | Prior đa dạng hơn, chú ý hai chiều; phân loại và hồi quy | ROC-AUC chuẩn hóa 0,939 (29 bộ AMLB) |
| **TabDPT** [8] | NeurIPS 2025 (arXiv 10/2024) | Dữ liệu giả lập chưa chắc đủ | Học tự giám sát trên bảng thực kết hợp truy xuất mẫu; quy luật mở rộng | Độ chính xác 0,884 so với 0,872 của TabPFNv2 trên OpenML-CC18 |
| **TabICLv2** [5] | ICML 2026 | Cần tốt hơn, nhanh hơn, mở rộng được | Khung TFcol/TFrow/TFicl; prior mới, scalable softmax, Muon | Hạng 4,66 (TALENT), 4,82 (TabArena) |

### I1. Vì sao chọn bốn phương pháp này?
Mỗi phương pháp đại diện **một hướng cải tiến**: kiến trúc và mở rộng (TabPFNv2, TabICLv2), dữ liệu thực (TabDPT), chọn mẫu và tinh chỉnh (LoCalPFN). Lưu ý: danh hiệu "tiên tiến nhất" luôn gắn với một bộ đánh giá cụ thể.

### I2. Phương pháp nào chạy được, phương pháp nào không?
TabICLv2, TabPFNv2 và TabDPT **chạy được trên CPU** và có điểm số. LoCalPFN **không có số tự chạy**: dòng lệnh gốc chạy được (`num_steps=3`) nhưng không lưu điểm số hoặc xác suất, nên báo cáo dùng số công bố của tác giả và ghi rõ.

### I3. LoCalPFN và TabDPT có liên quan nhau không?
Có: hai bài **cùng nhóm tác giả (Layer 6)**. Họ thấy điểm yếu của TabPFN rồi thử hai hướng khác nhau: chọn mẫu thông minh hơn (LoCalPFN) và học trước từ dữ liệu thực (TabDPT).

### I4. Dữ liệu huấn luyện trước của từng phương pháp?
TabPFNv2 và TabICLv2: **dữ liệu giả lập**. TabDPT: **nhiều bảng thực** (danh sách cụ thể cần xem trong bài gốc). LoCalPFN: xuất phát từ TabPFN đã huấn luyện, rồi **tinh chỉnh trên từng bảng đích**.

---

## J. Thực nghiệm giữa kỳ của nhóm

### J1. Em đã chạy gì?
Ba mô hình (TabICLv2, TabPFNv2, TabDPT) trên CPU, bộ **Breast Cancer Wisconsin** (569 mẫu, 30 đặc trưng số, hai lớp), seed 42, chia 364/91/114 (nguồn/xác thực/kiểm tra), lấy 50 hoặc 100 mẫu huấn luyện, `n_estimators = 1`, không huấn luyện lại. Bốn chỉ số: độ chính xác, Macro-F1, ROC-AUC, Log-Loss.

### J2. Kết quả chính là gì?
Ở 100 mẫu: TabICLv2 97,37% (ROC-AUC 0,9977; Log-Loss 0,0650), TabDPT 96,49%, TabPFNv2 95,61%. Ở 50 mẫu: TabICLv2 và TabPFNv2 cùng 97,37%, TabDPT 96,49%. **Chênh lệch chỉ là hai dự đoán trên 114 mẫu, chưa đủ kết luận.**

### J3. Thời gian chạy cho thấy gì?
Dự đoán ở 100 mẫu: 0,067 giây (TabICLv2), 0,167 (TabPFNv2), 0,199 (TabDPT). Con số **25,87 giây của TabICLv2 ở lượt đầu không phải tốc độ suy luận**: lượt đó gồm cả tải trọng số (nhật ký có 4 lần thử lại do ngắt kết nối); lượt sau chỉ 0,07 giây. Cần đo riêng thời gian tải, chuẩn bị và dự đoán.

### J4. Em có đối chiếu với mô hình đơn giản không?
Có, chạy riêng ngoài các notebook, chỉ để đối chiếu, trên cùng cách chia: hồi quy logistic đúng 111/114 (50 mẫu) và 110/114 (100 mẫu); rừng ngẫu nhiên 109 và 108; CatBoost 108 và 109; kNN 106 và 107. Hồi quy logistic gần ngang các mô hình nền tảng, vì bộ dữ liệu dễ.

### J5. Hạn chế thử nghiệm của em?
Một bộ dữ liệu, một seed, hai mức số mẫu, cấu hình CPU rút gọn (không phải cấu hình đầy đủ của bài báo), LoCalPFN chưa chạy, chưa có khoảng tin cậy qua nhiều seed, chưa đo ECE và Brier. Đã tự nhận trong báo cáo.

### J6. Cuối kỳ em làm gì?
Chạy khoảng 8 bộ phân loại của TALENT (credit-g, kr-vs-kp, adult, eucalyptus, vehicle, car-evaluation, segment, shuttle), 5 seed, kèm các mô hình truyền thống; thêm **nhãn nhiễu, mất cân bằng lớp và dữ liệu thiếu** có kiểm soát; đo mức giảm hiệu năng, cộng ECE hoặc Brier. Script đã chuẩn bị trong thư mục `source-code/talent_benchmark`.

### J7. Các bộ dữ liệu đó thuộc lĩnh vực gì?
credit-g: tín dụng; adult: thu nhập từ điều tra dân số; kr-vs-kp: cờ vua; eucalyptus: lâm nghiệp; vehicle: nhận dạng hình ảnh; car-evaluation: đánh giá xe; segment: phân đoạn ảnh; shuttle: dữ liệu NASA. Trong TALENT tên cột đã bị bỏ, mô hình chỉ thấy các cột số; bốn phương pháp đều không dùng tên cột nên không ảnh hưởng, nhưng nhóm CARTE hay TabSTAR (dùng tên cột) thì không thử được trên bộ này. (Lĩnh vực của từng bộ do mình nhớ theo mô tả quen thuộc của UCI/OpenML, nên cần đối chiếu lại.)

---

## K. Về TabICLv2 (chỉ hỏi sau khi đã trình bày kỹ TFM)

### K1. TabICLv2 là gì và quan hệ với TabICL, TabPFN thế nào?
TabICLv2 (Qu, Holzmüller, Varoquaux, Le Morvan, ICML 2026) là **phiên bản thứ hai của TabICL**, thuộc cùng họ TFM với TabPFN: Transformer huấn luyện trước trên dữ liệu giả lập và dùng học trong ngữ cảnh. TabICL gốc đã tách xử lý thành ba bước TFcol, TFrow, TFicl để nhanh và mở rộng hơn TabPFN.

### K2. TabICLv2 cải tiến gì so với TabICL?
Không chỉ một chức năng mà **một nhóm cải tiến**: (1) **bộ sinh dữ liệu giả lập mới**, đa dạng hơn; (2) thay đổi kiến trúc, nổi bật là **scalable softmax** giúp tổng quát hóa sang bảng lớn; (3) huấn luyện trước bằng bộ tối ưu **Muon** thay AdamW; (4) hỗ trợ **cả phân loại lẫn hồi quy** (theo tài liệu của kho mã, bản v1.1 chỉ phân loại). Theo kho mã: độ chính xác tăng rõ so với v1 với thời gian chạy tương đương.
*Cách nói gọn nếu thầy hỏi:* "TabICLv2 kế thừa khung TabICL, cải tiến chủ yếu ở dữ liệu giả lập, cơ chế chú ý và cách huấn luyện, và hỗ trợ cả hồi quy, để tốt hơn, nhanh hơn và mở rộng hơn."

### K3. Số công bố của TabICLv2?
Hạng trung bình **4,66 trên TALENT** và **4,82 trên TabArena** (hạng, càng thấp càng tốt, đo trên hai bộ khác nhau nên không cộng gộp). Tác giả nói vượt RealTabPFN-2.5 mà không cần tinh chỉnh, và xử lý cỡ triệu mẫu trong 50 GB bộ nhớ GPU. Đây là **số của tác giả**.

### K4. Vì sao chọn TabICLv2 để thực nghiệm?
Là bài **mới nhất** trong bốn bài, **mã nguồn mở đủ** (suy luận, huấn luyện trước, sinh dữ liệu giả lập), cài và chạy bằng vài dòng lệnh, có checkpoint công khai và được đánh giá trên TALENT nên thuận lợi để tái lập và mở rộng.

### K5. TabICLv2 hơn ba mô hình còn lại ở điểm nào?
Theo bài báo: **quy mô và tốc độ** (cỡ triệu mẫu, nhanh hơn RealTabPFN-2.5) cùng hạng tốt trên TALENT và TabArena. So với TabDPT: học từ dữ liệu giả lập nên ít nguy cơ trùng với tập kiểm tra hơn (nhận định của mình). So với LoCalPFN: dùng thẳng, không tinh chỉnh từng bài toán, làm được hồi quy. **Không nói TabICLv2 "mở hơn" TabDPT**, vì TabDPT cũng công bố cả mã huấn luyện. Trong thử nghiệm của nhóm, TabICLv2 chỉ **nhỉnh hơn**, chưa vượt trội.

### K6. Thử nghiệm của em có kiểm chứng được điểm mạnh chính của TabICLv2 không?
**Chưa.** Điểm mạnh chính là quy mô lớn, còn Wisconsin chỉ có 569 mẫu. Thử nghiệm giữa kỳ cũng chạy CPU với một thành phần dự đoán, chưa thử hồi quy. Vì vậy cuối kỳ mở rộng sang nhiều bộ TALENT (một trong hai bộ đánh giá của chính bài báo).

### K7. Mã nguồn TabICLv2 nằm ở đâu?
Kho [soda-inria/tabicl](https://github.com/soda-inria/tabicl), đã tải về `source-code/tabicl_official`. Suy luận: `src/tabicl/_sklearn/`; kiến trúc: `src/tabicl/_model/` (có `ssmax.py` là scalable softmax); sinh dữ liệu giả lập: `src/tabicl/prior/`; huấn luyện trước: `src/tabicl/train/` (có `_muon.py`) cùng `scripts/train_v2_*`. Tệp trọng số (checkpoint) nằm trên Hugging Face, gói tự tải ở lần chạy đầu.

### K8. Em có tự huấn luyện lại TabICLv2 không?
Không. Huấn luyện trước cần nhiều tài nguyên GPU; nhóm chỉ dùng checkpoint có sẵn và đọc mã để hiểu cách sinh dữ liệu và cách huấn luyện.

---

## L. Câu hỏi "bẫy" và cách trả lời an toàn

- **"Vậy TFM tốt nhất rồi à?"** Không. Theo bài báo thì ngang hoặc hơn cây trên nhiều bộ, nhưng điều kiện đánh giá khác nhau và thử nghiệm của em chỉ là sơ bộ.
- **"Em tự chạy cả bốn mô hình à?"** Ba mô hình có số tự chạy; LoCalPFN dùng số công bố của tác giả, đã ghi rõ trong báo cáo.
- **"Số này của em hay của tác giả?"** Phân biệt rõ: các số ở phần bài báo là của tác giả; bảng 50 và 100 mẫu là em tự chạy.
- **"Sao không kết luận TabICLv2 tốt nhất?"** Vì chênh lệch chỉ hai dự đoán trên 114 mẫu, khoảng tin cậy chồng nhau, và mô hình đơn giản cũng ngang.
- **"Em đóng góp gì mới?"** Giữa kỳ: khảo sát, thực nghiệm đối chiếu có kiểm tra độc lập và chỉ ra các cạm bẫy khi đọc số liệu. Cuối kỳ: kiểm tra độ bền trước nhãn nhiễu, mất cân bằng và dữ liệu thiếu, một vấn đề các bài báo chưa đánh giá kỹ.
- **"Em có hiểu công thức Bayes không?"** Nói bằng lời (C2); công thức chỉ để minh họa, mô hình học xấp xỉ chứ không tính tích phân.

---

## M. Việc cần tự kiểm tra lại trước khi bảo vệ

1. Danh sách các bảng thực mà TabDPT dùng để huấn luyện trước (mình chưa tra).
2. Chi tiết prior của TabPFNv2 (mình chỉ đọc mã của TabICLv2) và các giới hạn số mẫu, số cột, số lớp của TabPFN gốc và TabPFNv2 (kể cả giới hạn khoảng 10 lớp; ảnh hưởng đến bộ `letter` 26 lớp).
3. Lĩnh vực của từng bộ trong TALENT (đối chiếu trên OpenML).
4. Số công bố 4,66 và 4,82 của TabICLv2 và các số của ba bài còn lại: đối chiếu lại với bảng trong bài gốc.
5. Chi tiết "TabICL v1 chỉ phân loại" (mình dựa vào tài liệu của kho mã, bản v1.1).
6. Việc TabICLv2 đã được kiểm tra trên các tập hồi quy trong thử nghiệm: **chưa**, chỉ mới thử phân loại.
