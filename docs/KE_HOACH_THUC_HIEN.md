# KẾ HOẠCH THỰC HIỆN: TabPFN trên dữ liệu nhỏ (An và Claude cùng theo)

Cập nhật lần đầu: 20/10/2026. Cách dùng: mỗi bước có "Xong khi" để biết đã đủ chưa. Làm xong đánh dấu [x], ghi ngày vào cột nhật ký ở cuối file. Mọi thay đổi hướng đi phải sửa file này trước.

Đề tài: Khảo sát các mô hình nền tảng cho dữ liệu bảng (Tabular Foundation Models) và đánh giá thực nghiệm mô hình TabPFN trên dữ liệu nhỏ.
Bài gốc (để đọc sâu và tái hiện): Nature 01/2025, "Accurate predictions on small data with a tabular foundation model".

## Hướng của thầy (bổ sung 21/10/2026)

Bài khảo sát phải đi theo mạch sau:
1. Chỉ ra 3 đến 4 phương pháp tiên tiến (SOTA).
2. Mỗi phương pháp chạy trên bộ dữ liệu nào.
3. Thực nghiệm như thế nào, kết quả gì, điểm yếu gì.
4. Từ đó nhìn ra khoảng trống nghiên cứu, tức một "ngách", để đề xuất cái của mình hoặc cải tiến hơn người ta.

Cách chia công việc:
- **Bốn phương pháp SOTA** (theo `Survey_TFM_SmallData_2026.docx`): TabPFNv2, TabICLv2, TabDPT, LoCalPFN.
- **Ba phương pháp** TabPFNv2, TabDPT, LoCalPFN: chỉ cần trình bày quá trình họ làm và kết quả họ công bố.
- **Cái của mình là TabICLv2**: tải code về, chạy trực tiếp, lấy kết quả đó so với ba phương pháp trên. Xong phần này là xong giữa kỳ.
- **Nếu kết quả của TabICLv2 chưa tốt**: nói rõ em sẽ phát triển thêm một ngách ở cuối kỳ để kết quả khả quan hơn. Ngách đã chọn: độ bền với nhãn nhiễu và mất cân bằng lớp, đo kèm độ tin cậy xác suất (quyết định D2).

Lưu ý khi so sánh: số của ba phương pháp kia là số công bố, đo trên bộ dữ liệu và chỉ số khác nhau, nên khi so với kết quả TabICLv2 tự chạy phải nói rõ điều đó. Muốn so công bằng hơn thì chạy lại cả ba (hoặc ít nhất TabPFNv2) trên cùng bộ dữ liệu và cùng cách chia, nếu phần cứng cho phép.

---

## 0. Những gì đã có (đến 20/10/2026)

- [x] Danh sách 60 bài đã kiểm chứng (50 A*, 1 A, 9 Q1), PDF trong `Bai_bao_Survey\60_paper`.
- [x] `HIEU_SAU_BAI_NATURE_TABPFN.docx`: giải thích bài Nature (11 trang, có bộ câu hỏi thầy có thể hỏi).
- [x] `BAO_CAO_KHAO_SAT_GIUA_KY.docx`: bản nháp báo cáo khảo sát 60 bài (22 trang).
- [x] `TOM_TAT_BAI_TONG_QUAN_FINAL_da_sua.docx`: tóm tắt bài của anh bạn.
- [x] Code của anh bạn trong `Final\Final\source-code\Tabular_4SOTA_CPU` (chạy TabICLv2, TabPFNv2, TabDPT trên Breast Cancer) kèm `tools/run_baselines.py` mình thêm. Tạm để đó, dùng làm mẫu cấu trúc chứ chưa dùng làm kết quả chính.

Việc tồn đọng cần xử lý (xem Giai đoạn 1): file PDF số 48 sai; hai file số 46 trùng; chưa chốt 20 bài đọc kỹ.

---

## 1. Lộ trình tổng thể

| Giai đoạn | Nội dung | Sản phẩm | Ai làm chính |
|---|---|---|---|
| 1 | Hiểu bài Nature, tóm tắt 40 bài, chọn 20 bài, lập bảng SOTA và khoảng trống | 40 file tóm tắt, bảng chọn 20 bài, bảng SOTA, báo cáo khảo sát bản chốt | An đọc và chọn, Claude viết và chỉnh |
| 2 | Slide giữa kỳ (phần khảo sát) | File slide + kịch bản nói | Claude làm, An duyệt |
| 3 | TabICLv2: tải code, chạy trực tiếp, so với 3 SOTA, hoàn tất giữa kỳ | Kết quả TabICLv2, slide so sánh, đoạn "ngách sẽ phát triển ở cuối kỳ" | Cùng làm |
| 4 | Cuối kỳ: thực nghiệm theo 10 bước quy trình, gồm phần mở rộng (ngách) | Code, dữ liệu, kết quả, báo cáo, slide | Cùng làm (chia ở mục 4) |

Nguyên tắc: không sang giai đoạn sau khi chưa đạt "Xong khi" của giai đoạn trước. Thứ tự đã chốt ngày 20/10: hiểu bài Nature, viết báo cáo, làm slide, rồi mới tới TabICLv2. Thầy yêu cầu thêm (21/10): phần chạy TabICLv2 thuộc giữa kỳ, nên nằm ở Giai đoạn 3, trước cuối kỳ.

---

## 2. Giai đoạn 1: hiểu bài Nature và chốt tài liệu giữa kỳ

### 2.1 Việc cần làm

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 1.1 | Đọc `HIEU_SAU_BAI_NATURE_TABPFN.docx` hết một lượt, đánh dấu chỗ chưa hiểu | An | Có danh sách câu hỏi của An |
| 1.2 | Giải đáp các chỗ chưa hiểu, bổ sung vào tài liệu | Claude | An nói "đã hiểu" từng mục |
| 1.3 | Tự trả lời không nhìn tài liệu 10 câu ở mục 10 của tài liệu Nature (ý tưởng, kiến trúc, dữ liệu giả, kết quả, hạn chế) | An | Trả lời đúng ít nhất 8/10; Claude chấm và hỏi thêm |
| 1.4 | Đọc `BAO_CAO_KHAO_SAT_GIUA_KY.docx`, ghi chỗ sai, thiếu, khó hiểu | An | Có danh sách sửa |
| 1.5 | Sửa báo cáo theo phản hồi, thêm tên giảng viên, lớp, mã học viên ở bìa nếu cần | Claude | Bản chốt |
| 1.6 | Tải lại PDF số 48 (GPT-3, khoảng 6,5 MB), xóa file 46 thừa (bản Self-Attention Between Datapoints, vì bài này là số 37) | An | Thư mục 60_paper đúng 60 file |
| 1.7 | Dò lại hạng CORE và Scimago của 11 bài mới thêm trên portal.core.edu.au và scimagojr.com | An (Claude hỗ trợ tra) | Bảng hạng đã xác nhận |
| 1.8 | Viết **mỗi bài một file tóm tắt cho 40 bài đầu tiên** của danh sách 60 (không cần ngắn, miễn giữ đủ ý chính): vấn đề, ý tưởng, phương pháp, dữ liệu, thực nghiệm và kết quả, điểm yếu, liên hệ với bài Nature. Đặt trong thư mục `Tom_tat_bai_bao`, kèm file mục lục `00_THU_TU_DOC.md` | Claude | Đủ 40 file + mục lục, An mở đọc được |
| 1.9 | An đọc 40 file, đánh dấu bài thấy quan trọng. Cùng chọn ra **20 bài tốt nhất** theo tiêu chí ở mục 2.3 | An + Claude | Danh sách 20 bài được cả hai đồng ý, ghi vào file này |
| 1.10 | Lập **bảng SOTA** (mục 2.2): mỗi phương pháp một dòng gồm làm gì, chạy trên dữ liệu nào, thực nghiệm ra sao, kết quả, điểm yếu | Claude | Bảng đủ 4 dòng, mọi số có nguồn |
| 1.11 | Từ bảng SOTA rút ra **khoảng trống và ngách** để đề xuất (mục 2.2) | Cùng | 2 đến 3 khoảng trống có bằng chứng, chọn 1 ngách |

### 2.2 Bảng SOTA và khoảng trống (theo ý thầy)

Mẫu bảng cần điền (4 dòng: TabPFNv2, TabICLv2, TabDPT, LoCalPFN):

| Phương pháp | Làm gì, quá trình | Chạy trên dữ liệu nào | Thực nghiệm thế nào | Kết quả công bố | Điểm yếu |
|---|---|---|---|---|---|

Từ cột "Điểm yếu" và các chỗ các phương pháp đều chưa đo, rút ra khoảng trống. Ứng viên đã thấy trong khảo sát: độ tin cậy của xác suất, độ bền với nhãn nhiễu và mất cân bằng lớp, dữ liệu rất ít mẫu (dưới vài trăm), chi phí dự đoán từng mẫu, dữ liệu đổi theo thời gian. Ngách đã chọn ở D2: độ bền với nhãn nhiễu và mất cân bằng lớp, đo kèm độ tin cậy xác suất. Khi viết phải chỉ ra được: ít nhất một bài trong 4 SOTA chưa đo điều này, và có số hoặc câu trích dẫn chứng minh.

### 2.3 Tiêu chí chọn 20 bài từ 40 bài

1. Liên quan trực tiếp tới bài Nature (TabPFN, PFN, học trong ngữ cảnh, dữ liệu giả).
2. Có mặt trong 4 SOTA hoặc là nền của chúng.
3. Phủ đủ các nhóm: nền PFN, cải tiến TabPFN, ngôn ngữ và chuyển giao, đối thủ (cây, học sâu), benchmark.
4. Có đủ ba thứ tải được: bài báo, bộ dữ liệu, code (nếu bài đó có thể là bài toán con).
5. Nơi đăng đạt quy tắc của thầy (Scimago Q1 hoặc CORE A*/A/B).
6. Không có hai bài làm cùng một việc nếu một bài đã đủ.

Cách chọn: An đánh dấu sau khi đọc 40 file; Claude đề xuất bộ 20 dựa trên tiêu chí; hai bên đối chiếu rồi chốt.

### 2.4 Xong khi
Có đủ: 40 file tóm tắt, danh sách 20 bài đã chốt, bảng SOTA 4 dòng, ngách đã có bằng chứng. An giải thích được bằng lời của mình, không nhìn tài liệu: (a) vì sao TabPFN không cần huấn luyện lại; (b) dữ liệu giả sinh ra thế nào; (c) chú ý hai chiều để làm gì; (d) kết quả chính và những chỗ không được suy diễn quá; (e) bốn hạn chế.

---

## 3. Giai đoạn 2: slide giữa kỳ (phần khảo sát)

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 2.1 | Chốt cấu trúc slide (khoảng 15 đến 18 slide cho 15 đến 20 phút): vấn đề, câu hỏi nghiên cứu, cách chọn 60 bài và 20 bài, nền tảng, dòng thời gian, bài Nature chi tiết (3 đến 4 slide), bốn SOTA (mỗi cái một slide: quá trình, dữ liệu, kết quả, điểm yếu), benchmark, lưu ý không so sánh được, khoảng trống và ngách, kế hoạch | Cùng | An duyệt dàn ý |
| 2.2 | Làm slide dạng bảng và hình đơn giản, ít chữ | Claude | File slide dựng được, kiểm tra từng trang |
| 2.3 | Viết kịch bản nói cho từng slide và 15 câu hỏi dự kiến kèm trả lời | Claude | An đọc thử một lượt |
| 2.4 | An tập nói, Claude đóng vai thầy hỏi | Cùng | An nói trọn bài mà không nhìn kịch bản |

Chú ý: slide ở giai đoạn này chưa có kết quả TabICLv2 của An; phần đó thêm ở Giai đoạn 3 và gắn vào bộ slide giữa kỳ cuối cùng.

---

## 3b. Giai đoạn 3: TabICLv2 (hoàn tất giữa kỳ)

Làm sau khi đã hiểu bài Nature và xong slide khảo sát (đúng thứ tự An đã chốt). Dùng bản rút gọn của quy trình 10 bước trên **một bộ dữ liệu**.

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 3.1 | Tải code TabICLv2 từ kho chính thức (`soda-inria/tabicl`), ghi phiên bản và mã commit. Kiểm tra giấy phép và cách tải trọng số | Claude, An xác nhận | Chạy được ví dụ mẫu của tác giả |
| 3.2 | Chọn 1 bộ dữ liệu (D1) và chia train/validation/test (Bước 02 đến 04) | Cùng | Sơ đồ chia, kiểm thử không rò rỉ |
| 3.3 | Mô hình nền (Bước 05) trên bộ đó | Claude viết, An đọc từng dòng | Bảng nền |
| 3.4 | Chạy TabICLv2 trực tiếp bằng code gốc (Bước 06) và đo kết quả | Cùng | Bảng kết quả TabICLv2 |
| 3.5 | So với kết quả công bố của TabPFNv2, TabDPT, LoCalPFN, nêu rõ điều kiện khác nhau; nếu máy cho phép, chạy lại TabPFNv2 (và TabDPT) trên cùng bộ và cùng cách chia | Cùng | Bảng so sánh có chú thích "số công bố" hay "tự chạy" |
| 3.6 | Nhận định: nếu TabICLv2 chưa tốt thì viết rõ "sẽ phát triển ngách ... ở cuối kỳ"; nếu tốt thì nêu vì sao và còn điểm yếu nào | Cùng | Đoạn nhận định có số liệu |
| 3.7 | Thêm vào slide giữa kỳ, tập nói lại | Claude làm, An tập | Bộ slide giữa kỳ hoàn chỉnh |

Xong khi: có kết quả TabICLv2 chạy từ code gốc trên 1 bộ dữ liệu và bản so sánh trung thực với 3 SOTA.

---

## 4. Giai đoạn 4: cuối kỳ, đi theo 10 bước của quy trình thầy đưa

Tên đề tài cuối kỳ (đề xuất): "Tái hiện và đánh giá TabPFN so với XGBoost và CatBoost trên dữ liệu bảng nhỏ".
Câu hỏi chính: Khi dữ liệu có rất ít mẫu, TabPFN có thật sự tốt hơn XGBoost và CatBoost như bài Nature nói không, tốt hơn bao nhiêu, và ở điều kiện nào thì không?

### Bước 01. Xác định bài toán (Define Problem)

- Bài toán: phân loại trên bảng nhỏ (đúng với hướng An đã chọn).
- Giả thuyết cần kiểm chứng (từ bài Nature): H1 TabPFN mặc định đạt kết quả tốt hơn XGBoost và CatBoost mặc định; H2 TabPFN mặc định vẫn ngang hoặc hơn khi hai đối thủ được chỉnh tham số; H3 ưu thế lớn nhất ở bảng rất nhỏ và giảm khi số mẫu tăng.
- Phần mở rộng (điều chưa ai làm trong bài, thầy yêu cầu): xem Bước 07. Cần An chốt (mục 6, quyết định D2).
- Xong khi: viết được phần "Bài toán, giả thuyết, phạm vi" một trang, thống nhất với An.

### Bước 02. Xác định biến mục tiêu và thời điểm dự đoán

- Mục tiêu: nhãn của từng bộ dữ liệu (nhị phân hoặc tối đa 10 lớp, đúng giới hạn của TabPFN).
- Thời điểm dự đoán: dữ liệu bảng không theo thời gian nên "thời điểm dự đoán" ở đây là lúc mô hình đã thấy tập huấn luyện và chưa thấy nhãn tập kiểm tra. Ghi rõ để không có thông tin từ tập kiểm tra lọt vào bước tiền xử lý (rò rỉ).
- Chọn bộ dữ liệu theo quy tắc D1: **làm một bộ thôi.** Chạy xong và ổn toàn bộ (mọi mô hình chính và SOTA) trên bộ 1 rồi mới sang bộ 2, rồi bộ 3. Mỗi bộ theo tiêu chí: lấy từ OpenML (tải được); thuộc danh sách phân loại của Nature (AutoML Benchmark, dưới 10.000 mẫu, dưới 500 cột, tối đa 10 lớp); khác nhau về đặc điểm: một bộ chỉ có cột số, một bộ có cột phân loại, một bộ có giá trị thiếu hoặc mất cân bằng lớp; có thể thêm German Credit vì bài Nature dùng minh họa.
- Việc cần làm: lấy danh sách ID OpenML trong code công bố của Nature (file `datasets/benchmark_dids.py`) hoặc Extended Data Table 3, rồi chọn.
- Xong khi: bảng mô tả bộ dữ liệu (nguồn, ID, số mẫu, số cột, loại cột, số lớp, tỉ lệ lớp, có thiếu không) và An đồng ý.

### Bước 03. Chuẩn bị đặc trưng (Prepare Features)

- Nguyên tắc: tiền xử lý tối thiểu và giống nhau cho mọi mô hình, để so sánh công bằng. TabPFN tự xử lý thiếu, mã hóa phân loại và chuẩn hóa; XGBoost cần mã hóa cột phân loại; CatBoost nhận cột phân loại trực tiếp.
- Mọi bước học từ dữ liệu (chuẩn hóa, mã hóa) chỉ khớp trên tập huấn luyện rồi áp lên tập khác.
- Xong khi: một hàm tiền xử lý duy nhất, có ghi chú vì sao mỗi mô hình xử lý khác nhau, và chạy được trên cả 2 đến 3 bộ.

### Bước 04. Chia huấn luyện, xác thực và kiểm tra (Train / Validation / Test)

- Tập kiểm tra cố định, tách rời hoàn toàn: dùng chia chuẩn của OpenML hoặc chia phân tầng một lần với seed cố định, tập kiểm tra không dùng để huấn luyện hay chọn tham số dù chỉ một lần.
- Từ phần còn lại lấy mẫu con huấn luyện với các cỡ n khác nhau để mô phỏng "dữ liệu nhỏ": ví dụ n = 20, 50, 100, 200, 500, 1.000 (cỡ cuối chỉ khi bộ có đủ mẫu).
- Tập xác thực: dùng để chọn tham số của XGBoost và CatBoost (kiểm định chéo 5 lần ngay trong tập huấn luyện, như Nature) và để chọn cấu hình TabPFN nếu có.
- Lặp lại 10 lần với các seed và các lần chia khác nhau (như Nature), báo cáo trung bình và khoảng tin cậy 95%.
- Xong khi: sơ đồ chia dữ liệu một trang, và kiểm thử tự động xác nhận không có dòng nào của tập kiểm tra nằm trong tập huấn luyện.

### Bước 05. Xây mô hình nền (Build Baseline)

Mục đích theo thầy: biết cái mình làm có tạo ra điều gì thật sự không, bằng cách so với những mô hình đã có để có "kết quả nền".

| Mô hình nền | Vì sao |
|---|---|
| Đoán lớp đa số (dummy) | Mức thấp nhất, biết bộ dữ liệu có dễ không |
| Hồi quy logistic | Đơn giản, thường mạnh trên bảng nhỏ |
| Rừng ngẫu nhiên | Nền cây quen thuộc |
| XGBoost và CatBoost mặc định | Hai đối thủ chính theo đề cương |

- Kết quả nền lưu thành file CSV riêng, không sửa sau khi đã so sánh.
- Xong khi: bảng kết quả nền trên mọi bộ và mọi cỡ n, kèm khoảng tin cậy.

### Bước 06. Huấn luyện mô hình (Train Models), áp dụng mô hình của mình

- Dùng thư viện TabPFN (bản cho bài Nature) ở cấu hình mặc định. Ghi rõ phiên bản thư viện, phiên bản trọng số, số estimator, thiết bị chạy.
- Lưu ý phần cứng: máy An không có GPU NVIDIA, RAM khoảng 11,9 GB nên chạy CPU, giữ n tối đa vài nghìn. Nếu cần chạy nhanh hơn, dùng GPU miễn phí trên Google Colab hoặc Kaggle (cần quyết định D4).
- Lưu ý rủi ro: trọng số TabPFN phải tải về lần đầu và có thể cần chấp nhận điều khoản trên nơi phát hành; kiểm tra sớm trước khi lên kế hoạch tiếp. Cần mạng cho lần tải đầu.
- Đo riêng thời gian: tải mô hình, chuẩn bị (fit), dự đoán, không gộp.
- Xong khi: một script chạy từ đầu tới cuối trên 1 bộ dữ liệu, cho ra bảng kết quả TabPFN cùng định dạng với bảng nền.

### Bước 07. Điều chỉnh (Tune)

- XGBoost và CatBoost: tìm kiếm ngẫu nhiên với kiểm định chéo 5 lần, ngân sách thời gian cố định (ví dụ 30 giây, 5 phút, 1 giờ) theo kiểu Nature; báo cáo cả bản mặc định và bản đã chỉnh.
- TabPFN: bản mặc định, và bản chỉnh nhẹ (số estimator, nhiệt độ softmax) trong cùng ngân sách để công bằng.
- Phần mở rộng (đề xuất, chờ An chốt D2): kiểm tra độ bền và độ tin cậy xác suất của TabPFN khi dữ liệu rất ít: thêm nhãn nhiễu (đổi nhãn ngẫu nhiên 5%, 10%, 20%), mất cân bằng lớp (giảm tỉ lệ lớp thiểu số), đo Log-Loss, sai số hiệu chỉnh (ECE) và biểu đồ độ tin cậy, so với XGBoost và CatBoost. Lý do chọn: là khoảng trống G5 trong báo cáo khảo sát; có bài nền trong danh sách 60 [47] nhãn nhiễu, [49] mất cân bằng, [52] calibration, [53] ensembles; không cần code lớn; có số liệu dễ so.
- Ứng viên thay thế (nếu D2 chọn khác): chia dữ liệu theo thời gian (drift); hoặc kết hợp TabPFN với tập hợp (PHE) trong cùng ngân sách.
- Xong khi: bảng "mặc định so với đã chỉnh" cho 3 mô hình, ghi rõ ngân sách; và kết quả phần mở rộng.

### Bước 08. Đánh giá cuối (Final Evaluation)

- Khóa cấu hình trước khi nhìn tập kiểm tra. Chỉ chạy trên tập kiểm tra một lần cho cấu hình đã chốt.
- Chỉ số: độ chính xác, ROC-AUC (một-với-tất-cả cho nhiều lớp, như Nature), Log-Loss, thời gian chuẩn bị và dự đoán. Có thêm điểm chuẩn hóa theo từng bộ để so với kiểu số của Nature, nhưng nói rõ không so trực tiếp với 0,939 của bài.
- Kiểm định: Wilcoxon theo cặp giữa TabPFN và từng đối thủ như Nature, kèm khoảng tin cậy 95%.
- So kết quả của mình với bài Nature và nêu rõ điều kiện khác nhau (số bộ, phần cứng, cỡ mẫu).
- Xong khi: bảng kết quả cuối có trung bình, khoảng tin cậy, p-value; kết luận được từng giả thuyết H1, H2, H3: ủng hộ, không ủng hộ, hoặc chưa đủ bằng chứng.

### Bước 09. Phân tích lỗi (Error Analysis)

- Ma trận nhầm lẫn từng mô hình; lớp nào sai nhiều, mẫu nào sai nhưng đối thủ đúng và ngược lại.
- Đường cong theo cỡ mẫu (n so với kết quả): TabPFN ưu thế ở n nào, đối thủ bắt kịp ở n nào.
- Độ tin cậy xác suất: biểu đồ độ tin cậy, ECE.
- Theo đặc điểm bảng: có cột phân loại, có thiếu, mất cân bằng.
- Xong khi: ít nhất 4 hình, 4 nhận xét có số, và danh sách trường hợp TabPFN thua kèm giải thích có cơ sở.

### Bước 10. Diễn giải (Interpretation)

- Trả lời câu hỏi chính bằng ngôn ngữ dễ hiểu: khi nào dùng TabPFN, khi nào không.
- Đối chiếu với giới hạn tác giả nêu: dự đoán từng mẫu chậm, bảng lớn, chỉ kiểm chứng tới 10.000 mẫu.
- Có thể dùng SHAP (đã có trong bài Nature) cho một ví dụ minh họa nếu còn thời gian.
- Hướng phát triển: các khoảng trống trong báo cáo khảo sát; nối sang TabICLv2.
- Xong khi: phần "Thảo luận và kết luận" không có câu nào vượt quá bằng chứng đã đo.

### Sản phẩm cuối kỳ (theo yêu cầu thầy)

- [ ] Code demo: sạch, có README chạy lại được, ghi seed và phiên bản thư viện.
- [ ] Dataset: ghi nguồn và ID, kèm script tải (không bắt buộc đóng gói dữ liệu).
- [ ] Bài báo gốc: PDF Nature và các bài được dùng.
- [ ] Báo cáo cuối kỳ (Word).
- [ ] Slide cuối kỳ và kịch bản nói.

---

## 5. TabICLv2 trong cuối kỳ

- TabICLv2 đã chạy ở Giai đoạn 3. Ở cuối kỳ, đưa nó vào cùng quy trình 10 bước, cùng dữ liệu và cùng cách chia như TabPFN, XGBoost, CatBoost.
- Ba phương pháp tiên tiến còn lại (TabPFNv2, TabDPT, LoCalPFN) chỉ trình bày quá trình và số công bố, trừ khi có lý do chạy lại.
- Phần mở rộng (ngách D2) áp dụng cho các mô hình chính để so với nhau.
- Câu hỏi cần xác nhận với thầy (D7): cuối kỳ lấy TabPFN (theo đề tài) làm mô hình tái hiện chính và TabICLv2 như mô hình bổ sung, hay ngược lại.

---

## 6. Các quyết định (đã chốt 21/10/2026)

| Mã | Quyết định | Nội dung đã chốt |
|---|---|---|
| D1 | Số bộ dữ liệu | **Một bộ thôi.** Chạy xong và ổn toàn bộ mô hình (chính và SOTA) trên bộ 1 rồi mới chạy bộ 2, rồi bộ 3 |
| D2 | Phần mở rộng (ngách) | Độ bền với nhãn nhiễu và mất cân bằng lớp, đo kèm độ tin cậy xác suất |
| D3 | Các mức cỡ mẫu n | 20, 50, 100, 200, 500, 1.000 |
| D4 | Chạy ở đâu | CPU máy An cho thử nghiệm nhỏ, Google Colab cho phần nặng |
| D5 | Dùng lại code anh bạn | Chỉ lấy cấu trúc thư mục và cách ghi kết quả; tự viết lại script chính để hiểu rõ từng dòng |
| D6 | Hai danh sách 20 bài khác nhau | Không chọn một trong hai. Thay bằng cách ở Giai đoạn 1: viết tóm tắt 40 bài, rồi An và Claude cùng chọn 20 bài tốt nhất theo tiêu chí mục 2.3 (giải thích bên dưới) |
| D7 | Hỏi giảng viên | Hỏi ngay xem thầy chấp nhận phần mở rộng ở D2 không, chạy 1 bộ dữ liệu hay 2 đến 3 bộ, và cuối kỳ lấy TabPFN hay TabICLv2 làm mô hình chính |

### Giải thích rõ D6

Hiện có hai danh sách 20 bài, và chúng khác nhau:
- **Danh sách của An:** 20 bài đầu trong danh sách 60 bài của thầy, được xếp theo mức liên quan với bài Nature. Danh sách này gồm các bài như AMLB, PFNs4BO, ForecastPFN, TabPFN Unleashed, Do-PFN, APT.
- **Danh sách của anh bạn:** 20 bài do ChatGPT chọn cho bài tổng quan trong `Survey_TFM_SmallData_2026.docx`. Hai danh sách trùng 14 bài. Danh sách của anh bạn có 6 bài mà danh sách của An không có (TabFlex, TransTab, XTab, CARTE, TabLLM, TabuLa-8B), và bỏ 6 bài của An.

Vì sao điều này quan trọng: 20 bài là phần nội dung chính của báo cáo và slide giữa kỳ, nên chọn khác bài thì nội dung khác. Cách cũ (chọn nguyên một danh sách) đều có điểm yếu: danh sách của An chưa được đọc kỹ, danh sách của anh bạn do ChatGPT chọn và thiếu vài bài rất gần với bài Nature.

Cách mới: Claude viết tóm tắt cho 40 bài đầu (mỗi bài một file). An đọc, đánh dấu. Rồi hai bên chọn 20 bài theo tiêu chí ở mục 2.3. Như vậy mỗi bài trong 20 bài đều có lý do được chọn, và An hiểu chúng. Không còn phải chọn giữa hai danh sách.

---|---|---|---|
| D1 | Chọn 2 hay 3 bộ dữ liệu | 3 bộ: một chỉ cột số, một có cột phân loại, một có thiếu hoặc mất cân bằng | Trước Bước 02 |
| D2 | Phần mở rộng | Độ bền với nhãn nhiễu và mất cân bằng lớp, đo kèm độ tin cậy xác suất | Trước Bước 07, tốt nhất sớm hơn |
| D3 | Các mức cỡ mẫu n | 20, 50, 100, 200, 500, 1.000 | Trước Bước 04 |
| D4 | Chạy ở đâu | CPU máy An cho thử nghiệm nhỏ; Colab hoặc Kaggle GPU nếu cần chạy đủ 10 lần lặp | Trước Bước 06 |
| D5 | Có dùng lại code anh bạn không | Chỉ lấy cấu trúc thư mục và cách ghi kết quả, tự viết lại script chính để hiểu rõ từng dòng | Trước Bước 06 |
| D6 | Cách xử lý 20 bài khác nhau giữa danh sách An và anh bạn | Theo danh sách An | Giai đoạn 1 (việc 1.8) |
| D7 | Khi nào hỏi giảng viên | Hỏi ngay nếu thầy chấp nhận phần mở rộng ở D2 và việc chạy trên 2 hay 3 bộ | Giữa kỳ |

---

## 7. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Không tải được trọng số TabPFN hoặc cần chấp nhận điều khoản | Lỗi tải ở lần chạy thử | Thử sớm ở đầu Giai đoạn 3; phương án dự phòng: Colab/Kaggle hoặc API chính thức |
| CPU chậm với nhiều lần lặp | Một lượt chạy quá lâu | Giảm số lần lặp xuống 5, giảm n lớn nhất, hoặc dùng GPU miễn phí |
| Kết quả không giống bài Nature | TabPFN không thắng rõ | Không che giấu; phân tích nguyên nhân (cỡ mẫu, bộ dữ liệu, cấu hình), vì vẫn là kết quả hợp lệ |
| Rò rỉ dữ liệu | Kết quả đẹp bất thường | Kiểm thử tự động ở Bước 04, khớp tiền xử lý chỉ trên tập huấn luyện |
| Chọn tham số trên tập kiểm tra | Kết quả cao nhờ may | Khóa cấu hình trước Bước 08, chỉ chạy tập kiểm tra một lần |
| So sánh không công bằng | Đối thủ chỉnh ít hơn | Cùng ngân sách thời gian và cùng cách chia |
| Hiểu sai bài Nature khi thầy hỏi | Trả lời mơ hồ ở câu hỏi về dữ liệu giả hoặc con số | Làm Giai đoạn 1 kỹ, luyện hỏi đáp |

---

## 8. Cách làm việc giữa An và Claude

- Claude không làm bước tiếp theo khi chưa có "Xong khi" của bước trước, trừ khi An cho phép.
- Mỗi lần xong một việc, Claude ghi vào nhật ký và nói rõ còn việc gì; An cũng cập nhật khi tự làm.
- Con số nào đưa vào báo cáo đều phải truy ra được từ file kết quả hoặc từ bài gốc.
- Điều không chắc thì ghi "chưa kiểm chứng", không đoán.
- File sinh ra đặt trong `advanced-data-mining` (docs, code, results), không để lẫn với thư mục của anh bạn.

---

## 9. Nhật ký

| Ngày | Việc đã làm | Việc kế tiếp |
|---|---|---|
| 20/10/2026 | Có tài liệu Nature, báo cáo khảo sát bản nháp, kế hoạch này | An đọc tài liệu Nature, trả lời D1 đến D7 |
| 21/10/2026 | An chốt D1 đến D7; thêm hướng của thầy (3–4 SOTA, TabICLv2 chạy trực tiếp ở giữa kỳ, ngách ở cuối kỳ); đổi cách chọn 20 bài thành 40 file tóm tắt; thêm thư mục `.claude` chứa quy tắc | Viết 40 file tóm tắt, rồi bảng chọn 20 bài |
| 21/10/2026 | Viết xong 40 file tóm tắt (bài 1–40) trong `Tom_tat_bai_bao/` kèm `00_THU_TU_DOC.md` (có đề xuất sơ bộ 20 bài) | An đọc, cùng chọn 20 bài, rồi làm ma trận 4 SOTA và khoảng trống |

| 21/10/2026 | Thêm bài 47, 49; bỏ giới hạn 20 bài, sắp 42 bài thành 8 nhóm đọc trong `00_THU_TU_DOC.md`; sửa nhận định sai về NPT | An đọc theo nhóm, cùng chọn bài quan trọng nhất cho TabICLv2 |
| 22/10/2026 | An yêu cầu loại bài 37 (NPT): đã xóa 2 file PDF NPT (bản 37 và bản số 46 thừa) và file tóm tắt Word; còn 41 file tóm tắt | An đọc theo `00_THU_TU_DOC.md` |
| 22/10/2026 | Gỡ bài 37 (NPT) khỏi `BAO_CAO_KHAO_SAT_GIUA_KY.docx` (còn 59 bài, giữ nguyên số thứ tự gốc); bản gốc nằm ở `Final/Final/_backup_truoc_khi_go_NPT/` | An đọc 41 file tóm tắt |
| 22/10/2026 | Dọn `Bai_bao_Survey`: xóa 3 thư mục nhóm rỗng/trùng, `DANH_SACH_60_BAI_TAI.md` (đã lỗi thời) và PDF Nature trùng ở ngoài cùng | — |

| 22/10/2026 | Viết lại 41 tóm tắt: **xong 41/41** (mỗi file 5–10 trang, đã đọc cả phụ lục; `00_THU_TU_DOC.md` đã cập nhật tên file). Phát hiện: bài 40 (TP-BERTa) có phân tích mất cân bằng nhãn hậu nghiệm, nên khoảng trống được diễn đạt lại là "chưa có khảo sát hệ thống". Chi tiết: Tom_tat_bai_bao/_nguon/TIEN_DO.md | An đọc 41 file, rồi cùng chọn bài quan trọng cho TabICLv2 |
