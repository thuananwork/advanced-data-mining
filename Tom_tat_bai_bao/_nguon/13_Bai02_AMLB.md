# Bài 2. AMLB: an AutoML Benchmark

| Mục | Thông tin |
|---|---|
| Nơi đăng | Journal of Machine Learning Research (JMLR); PDF là arXiv 2207.12560v2 (16/11/2023), vẫn mang khung bản thảo "Submitted 4/00; Published 10/00" của mẫu JMLR nên năm trên đầu trang không đáng tin |
| Tác giả | Pieter Gijsbers, Marcos L. P. Bueno, Stefan Coors, Erin LeDell, Sébastien Poirier, Janek Thomas, Bernd Bischl, Joaquin Vanschoren (TU Eindhoven, LMU Munich, H2O.ai, Radboud) |
| Mã nguồn / dữ liệu | openml.github.io/automlbenchmark (công cụ, kết quả, trực quan hóa tương tác); hai bộ OpenML: phân loại openml.org/s/271 (71 tác vụ), hồi quy openml.org/s/269 (33 tác vụ) |
| Vị trí trong đề tài | Một trong các benchmark mà bài Nature (bài 1) dùng. Nguồn của phép so sánh "TabPFN vs AutoGluon" và của quy ước điểm số: AUC (nhị phân), log-loss (đa lớp), RMSE (hồi quy). |

## 1. Bài này làm gì trong một đoạn

So sánh các framework AutoML thường bị làm sai: chọn dữ liệu thiên vị, cấu hình đối thủ sai, ngân sách không tương đương, bỏ qua lỗi. Bài đề xuất **AMLB** (AutoML Benchmark): một công cụ mã nguồn mở, tự động hóa trọn vòng đánh giá (cài framework, cấp tài nguyên, chạy, ghi kết quả, phân tích) và hai bộ benchmark chọn lọc trên OpenML (**71 tác vụ phân loại, 33 tác vụ hồi quy**). Trên đó, nhóm đánh giá **9 framework AutoML** (một số với nhiều cấu hình) với ngân sách **1 giờ và 4 giờ**, phân tích độ chính xác, đánh đổi độ chính xác và thời gian suy luận, lỗi của framework, và dùng cây Bradley-Terry để tìm tập tác vụ có thứ hạng khác nhau. Kết quả: **AutoGluon (preset best quality) có hạng trung bình cao nhất** nhưng suy luận rất chậm; các framework còn lại khá sát nhau.

## 2. Vấn đề và động cơ

- AutoML tự động hóa việc chọn thuật toán, tiền xử lý và siêu tham số. Theo khảo sát được trích, **69% trong 307 người làm nghề** dùng ít nhất một phần chọn mô hình hoặc cấu hình siêu tham số tự động.
- Sai sót thường gặp khi so sánh AutoML (mục 1.1, 2.1):
  - **Thiên lệch chọn dữ liệu:** quá ít dữ liệu, dữ liệu đã quá dễ, mỗi bài chọn một bộ khác nhau nên không so được và dễ chọn kết quả đẹp.
  - **Cấu hình sai đối thủ:** Balaji và Allen (2018) cấu hình H2O AutoML tối ưu log loss trong khi các đối thủ tối ưu F1, và vô hiệu HPO của một framework; Ferreira et al. (2021) dùng ngân sách 6 giờ, 1 giờ và "1000 thế hệ" cho các framework khác nhau mà không giải thích.
  - **Lỗi bị bỏ qua:** Truong et al. (2019) giới hạn 15 phút nên nhiều framework không trả kết quả; mà chính các bộ khó mới hay thất bại, nên framework thất bại ở bộ khó có thể trông tốt hơn. Zöller và Huber (2021) ghi nhận lỗi bộ nhớ, segmentation fault, Java crash.
  - So sánh với Kaggle khó diễn giải; benchmark đa nhãn của Wever et al. buộc phải cài đặt lại các framework.
- Cuộc thi AutoML (Guyon et al. 2019) cũng thấy nhiều phương pháp thất bại trên một số bộ do hết bộ nhớ.
- Mục tiêu: một benchmark chuẩn, tái lập được (theo định nghĩa ACM), dễ mở rộng, tích hợp **cùng tác giả của từng framework**.

## 3. Các framework được đánh giá (mục 3, Bảng 1)

Chỉ chọn mã nguồn mở, chạy không cần GPU (nên loại Auto-Keras, AutoPyTorch), tác giả chủ động hợp tác tích hợp. Chín framework:
- **AutoGluon-Tabular:** không tìm pipeline hay tinh chỉnh siêu tham số, mà dùng **tập mô hình định sẵn** được **xếp chồng (stacking) ba tầng**: tầng 1 các họ mô hình huấn luyện trực tiếp; tầng 2 các mô hình cùng loại nhưng nhận thêm dự đoán tầng 1; tầng 3 ensemble (Caruana 2004). Ba preset: best quality (B), high quality (HQ), high quality có giới hạn thời gian suy luận (HQIL).
- **auto-sklearn** (tối ưu Bayes pipeline scikit-learn, khởi động bằng meta-learning từ 208 bộ OpenML, tạo ensemble) và **auto-sklearn 2** (rút gọn không gian tìm kiếm, successive halving, portfolio pipeline không phụ thuộc dữ liệu; không hỗ trợ hồi quy).
- **FLAML** (tối ưu hóa chi phí tiết kiệm (CFO) cho XGBoost, CatBoost, LightGBM và vài mô hình scikit-learn; nhanh với ngân sách nhỏ).
- **GAMA** (lập trình di truyền, đa mục tiêu NSGA-II, bất đồng bộ).
- **H2O AutoML** (danh mục cấu hình dựng sẵn + tìm ngẫu nhiên, hai loại stacked ensemble; thiết kế cho suy luận nhanh).
- **LightAutoML** (thiết kế cho tài chính; chỉ mô hình tuyến tính và CatBoost, LightGBM; luật chuyên gia + TPE).
- **MLJAR** (cặp preset compete và perform; thêm bước tạo và chọn đặc trưng, hill climbing, stacking).
- **TPOT** (lập trình di truyền trên cây pipeline, giảm độ phức tạp pipeline).
- **NaiveAutoML** chạy nhưng bị loại khỏi phân tích chính vì không ổn định (Phụ lục E).

Loại khỏi bài: Auto-WEKA (kết quả kém năm 2019, không cập nhật), AutoXGBoost (ngừng), Ml-Plan, mlr3automl, ML.NET (lỗi kỹ thuật).

**Baseline:** constant predictor (tiên nghiệm lớp/trung bình); **Random Forest** (thêm 10 cây mỗi lần đến khi sắp vượt 90% bộ nhớ hoặc thời gian, hoặc đủ 2.000 cây); **Tuned Random Forest** (tối ưu max_features, thử tới 11 giá trị bằng CV 5 fold; baseline mạnh nhất, mô phỏng cách làm đầu tiên của người làm nghề).

## 4. Công cụ và thiết kế benchmark (mục 4–5)

### 4.1. Công cụ
- Tích hợp mỗi framework chỉ cần **< 250 dòng mã trong ≤ 4 tệp**, trao đổi dữ liệu qua đối tượng Python hoặc định dạng tệp phổ biến, nên hỗ trợ framework viết bằng C#, Java, Python, R. Cũng chạy được cục bộ để benchmark riêng tư.
- Benchmark định nghĩa bằng **OpenML tasks/suites** (mỗi tác vụ gồm dữ liệu, quy trình đánh giá, các phép chia, thuộc tính đích). Ràng buộc tài nguyên (thời gian, số nhân CPU, bộ nhớ, đĩa) khai báo riêng trong `constraints.yaml`. Ví dụ lệnh: `runbenchmark.py tunedrandomforest openml/s/271 1h8c`; chế độ `aws` chạy song song trên nhiều EC2.

### 4.2. Chọn dữ liệu (71 phân loại + 33 hồi quy)
Lấy từ các bài AutoML trước (Thornton 2013), cuộc thi (Guyon 2019) và benchmark (Bischl 2021), theo tiêu chí xác định trước:
- **Đủ khó:** nếu cây quyết định/RF/logistic đã đạt lỗi 0 hoặc mọi thuật toán cho kết quả như nhau thì không phân biệt được framework.
- **Đại diện bài toán thực tế:** hạn chế dữ liệu nhân tạo (giữ vài bộ như kr-vs-kp) và thị giác máy tính trên pixel thô.
- **Không có văn bản tự do** (nhiều framework chưa hỗ trợ nên xử lý như hạng mục).
- **Đa dạng lĩnh vực** (tránh thiên lệch như nhóm bộ chất lượng phần mềm jm1, kc1, kc2, pc1, pc3, pc4 trong CC18).
- **i.i.d.** (loại dữ liệu theo thời gian hoặc đo lặp lại).
- **Công khai trên OpenML**, không rò nhãn, không trùng gần (chỉ khác mã hóa hạng mục, imputation hoặc nhị phân hóa).
(Bộ airlines vi phạm các tiêu chí nhưng được giữ vì lý do lịch sử.) Số mẫu và số đặc trưng biến thiên nhiều bậc độ lớn; các bộ có giá trị thiếu và đặc trưng hạng mục được giữ có chủ ý, vì AutoML phải xử lý dữ liệu "bẩn". Hạn chế tự nhận: loại dữ liệu có văn bản làm giảm lựa chọn, đặc biệt bộ lớn.

### 4.3. Thước đo
**AUC (nhị phân), log loss (đa lớp), RMSE (hồi quy)** (cài đặt scikit-learn 1.2.2): hợp lý, phổ biến và phần lớn framework hỗ trợ tối ưu đúng thước đo đó, điều bắt buộc để so công bằng. Mô hình **không được hiệu chuẩn** trừ khi framework tự làm; AUC không nhạy với hiệu chuẩn nên cần lưu ý với kết quả nhị phân.

### 4.4. Xử lý kết quả thiếu
Thất bại **không ngẫu nhiên**: tương quan với kích thước và mất cân bằng lớp. Bỏ qua hay điền bằng điểm của fold khác sẽ thiên vị. Giải pháp: **điền bằng constant predictor (tiên nghiệm)**, một cách phạt nặng nhưng diễn giải được và công bằng cho framework chạy ổn; không dùng RF làm dự phòng vì RF cũng rất tốn thời gian ở bộ lớn và nhiều lỗi (segfault) không cứu được.

### 4.5. Thiết lập thực nghiệm
- **Phần cứng:** AWS m5.2xlarge (8 vCPU Xeon Platinum 8000, 32 GB RAM, đĩa gp3 100 GB), không GPU. Các framework dùng cấu hình mặc định, chỉ điều khiển: *mode* (do tác giả mỗi framework chọn), thời gian tìm kiếm (**1 giờ và 4 giờ**, kèm 1 giờ nới lỏng cho nạp dữ liệu, dự đoán, dọn dẹp, không báo cho framework), tài nguyên, thước đo mục tiêu (trùng thước đo đánh giá). Mỗi tác vụ chạy **10 fold**.
- **Phiên bản:** chạy tháng 6/2023; một số kết quả của tháng 9/2021 (đánh dấu *) vì phiên bản không khác đáng kể.

### 4.6. Hạn chế thiết kế (mục 5.3–5.4)
- **Không quy kết được nguyên nhân** hiệu năng cho một thành phần thiết kế (thư viện nền, không gian tìm kiếm, tiền xử lý, thuật toán tìm kiếm, đa tiến trình) vì framework khác nhau ở nhiều trục.
- **Chỉ ghi kết quả mô hình cuối**, không có hiệu năng theo thời gian (anytime); xấp xỉ bằng hai ngân sách 1 giờ và 4 giờ.
- **Không đánh giá định tính** (khả năng diễn giải, hỗ trợ, trải nghiệm).
- **Chỉ vài preset**, không tinh chỉnh "siêu-siêu tham số" của AutoML.
- **Rò rỉ meta-learning:** dữ liệu công khai nên auto-sklearn (208 bộ OpenML; loại 39 bộ của Gijsbers 2019 theo tên) và đặc biệt auto-sklearn 2 (portfolio và meta-model huấn luyện từ hàng trăm bộ, không loại được từng bộ) có thể đã thấy bộ kiểm tra; kết quả auto-sklearn 2 **lạc quan** và bị loại khỏi so sánh hạng.
- **Quá khớp benchmark:** bộ dữ liệu công khai nên không thể coi là chưa thấy; giữ bí mật thì mất tái lập.

## 5. Kết quả (mục 6)

### 5.1. Độ chính xác (Hình 2, 3, 4)
- Phân tích bằng **biểu đồ độ khác biệt tới hạn** (Friedman, p ≈ 0, hậu kiểm Nemenyi) sau khi điền thiếu bằng constant predictor, và boxplot với điểm **chuẩn hóa: Random Forest = 0, tốt nhất quan sát = 1** (âm nghĩa là kém RF).
- **AutoGluon(B) có hạng tốt nhất và TPOT có hạng tệ nhất** trong hầu hết các tình huống. AutoGluon(B) thường **không khác biệt có ý nghĩa** so với framework hạng hai, nhưng khác biệt có ý nghĩa so với các framework yếu nhất. Mọi baseline tệ hơn mọi framework AutoML trừ TPOT, dù **Tuned Random Forest là baseline mạnh**. Chỉ AutoGluon(B) và LightAutoML cho hạng tốt hơn một cách có ý nghĩa ở mọi thiết lập. Các framework còn lại nằm sát nhau.
- Hạng giống nhau vẫn có phân phối khác nhau: ví dụ AutoGluon(HQIL) và MLJAR(P) có hạng trung bình tương đương ở hồi quy 1 giờ, nhưng AutoGluon(HQIL) vượt RF nhất quán hơn dù trung vị thấp hơn; TPOT có hạng gần Tuned RF nhưng phương sai lớn.
- **Ngân sách thời gian:** tăng từ 1 lên 4 giờ chỉ cải thiện nhẹ; tác giả đoán các framework khác bị giới hạn bởi không gian tìm kiếm (ví dụ cần stacking mới đạt tốt nhất).

### 5.2. Cây Bradley-Terry (mục 6.2, Hình 5)
Cây phân hoạch đệ quy trên các so sánh cặp, tách theo tính chất bộ dữ liệu khi tham số mô hình BT không ổn định. Giá trị ở lá = xác suất một framework tốt nhất. Cây độ sâu 3 cho phân loại 1 giờ (chỉ dùng số mẫu và số đặc trưng):
- Bộ **nhỏ, ít chiều** (Node 3, 32 tác vụ): khác biệt giữa các framework hàng đầu không rõ; AutoGluon(B) vẫn được ưa thích, kèm AutoGluon(HQ) và auto-sklearn 2.
- Bộ **nhỏ nhưng rất rộng** (> 1.301 đặc trưng, Node 4, 5 tác vụ): khoảng cách tăng, AutoGluon(B) dẫn đầu, kế đó AutoGluon(HQ) và LightAutoML.
- Bộ **lớn** (> 8.237 mẫu): AutoGluon(B) vượt xa AutoGluon(HQ); FLAML thứ ba khi ít đặc trưng (≤ 43), AutoGluon(HQIL) khi nhiều đặc trưng hơn.
- Khoảng cách của AutoGluon(B) **tăng theo độ phức tạp** của bộ dữ liệu. Ngoại lệ: **bộ nhị phân nhỏ và cân bằng** (Hình 10), nơi nhiều framework được ưa thích hơn AutoGluon.

### 5.3. Độ chính xác và thời gian suy luận (mục 6.3)
Thời gian suy luận đo trên 1 hàng trong bộ nhớ hoặc 10.000 hàng từ đĩa, lấy trung vị 10 lần (loại TPOT do mã hóa dữ liệu; H2O không báo cáo trong bộ nhớ vì tuần tự hóa qua đĩa).
- **Độ chính xác cao đi kèm suy luận cực chậm:** AutoGluon(B) tạo ensemble nhiều tầng (stacking + ensembling) nên rất chậm. Thời gian suy luận giữa các framework **chênh nhau nhiều bậc độ lớn**.
- Ensemble nói chung chậm hơn (auto-sklearn, GAMA, MLJAR(B)); H2O AutoML và AutoGluon(HQIL) giữ suy luận nhanh nhờ mô hình tối ưu hoặc chọn ensemble có chủ ý; **FLAML rất nhanh** (CFO ngầm tính cả thời gian suy luận).
- Trên mặt Pareto (Hình 7) các framework chính xác nhất phải đánh đổi tốc độ suy luận; không phải framework nào cũng nằm trên biên Pareto. Nghĩa là báo cáo độ chính xác riêng lẻ cho bức tranh thiếu sót.

### 5.4. Lỗi của framework (mục 6.4, Hình 8–9, Phụ lục D)
Bốn nhóm lỗi: bộ nhớ, thời gian (vượt ngân sách quá 1 giờ nới lỏng), dữ liệu, cài đặt.
- **Nguyên nhân chính: bộ nhớ và thời gian**, nghĩa là kích thước bộ dữ liệu là nguyên nhân thất bại chủ yếu (không phải mọi phương pháp mở rộng tốt). Phân loại lỗi nhiều hơn hồi quy vì bộ lớn nhất phần lớn là phân loại và nhóm này đông hơn (71 so với 33).
- **Mất cân bằng lớp:** nhiều framework không tạo được mô hình hoặc dự đoán trên bộ rất mất cân bằng. Hai bộ nhỏ `yeast` và `wine-quality-white` có lớp thiểu số chỉ **5 mẫu**: khi một mẫu rơi vào tập test, chia validation nội bộ (kiểu CV 5 fold) cho ra phần chia không còn đủ lớp và đánh giá pipeline lỗi; chỉ thấy ở GAMA, LightAutoML, TPOT.
- MLJAR(B) có 284 lỗi cài đặt, 256 do ba lỗi riêng biệt, chỉ ở chế độ compete (nghi từ bước ensembling). AutoGluon(HQIL) 0.8.0 có lỗi hiệu chuẩn mô hình đã được sửa; các giá trị thiếu do lỗi này được điền bằng kết quả AutoGluon(HQ).
- Các framework tuân thủ thời gian chạy rất khác nhau; một số dừng sớm hơn nhiều so với ngân sách.

## 6. Hạn chế và hướng tương lai (mục 7)
1. Chỉ cấu hình mặc định/preset; kết quả với siêu-siêu tham số hoặc không gian tìm kiếm tùy chỉnh có thể khác.
2. Phần cứng khiêm tốn (8 CPU, **không GPU**), nên framework song song tốt hoặc dùng GPU có thể khác; ngược lại ngân sách ngắn hơn cũng đáng nghiên cứu (human-in-the-loop, giảm carbon).
3. Chưa có dữ liệu văn bản tự do, dữ liệu không i.i.d., học bán giám sát, tối ưu đa mục tiêu (công bằng, thời gian suy luận).
4. Kết quả chỉ đúng với phiên bản tháng 6/2023.
5. Mất cân bằng lớp chỉ xuất hiện như nguyên nhân gây lỗi và biến trong cây BT, không được điều khiển như một biến thí nghiệm; không đánh giá hiệu chuẩn hay nhãn nhiễu.

## 7. Liên hệ với bài Nature và đề tài
- Bài Nature (bài 1) dùng AMLB (cùng TabArena sau này) để so TabPFNv2 với AutoGluon và các baseline; quy ước thước đo (AUC/log-loss/RMSE) và việc điền kết quả thiếu bằng baseline được kế thừa trong TabArena (bài 5, điền bằng Random Forest mặc định).
- **AutoGluon (best quality) là mốc chuẩn mà mọi TFM phải đánh bại**; TabArena và TabICLv2 đều dùng nó làm pipeline tham chiếu. Kết quả bài này giải thích vì sao: ensemble nhiều tầng rất mạnh nhưng suy luận chậm, nên lợi thế của TFM (không huấn luyện) cần được đặt cạnh chi phí suy luận.
- Với thí nghiệm final-term: nên theo đúng các nguyên tắc ở đây (cùng thước đo tối ưu và đánh giá, ngân sách tương đương, ghi và xử lý lỗi, báo cáo kèm thời gian).
- Khoảng trống: mất cân bằng lớp được ghi nhận như nguyên nhân thất bại của framework AutoML (lớp thiểu số 5 mẫu) nhưng không được nghiên cứu có hệ thống; với TFM (cần cả ngữ cảnh huấn luyện) điều này càng đáng kiểm tra.

## 8. Từ khóa cần nhớ
- **71 phân loại + 33 hồi quy**; **9 framework**; **1 giờ và 4 giờ**; **10 fold**; 8 vCPU, 32 GB, không GPU.
- **AUC / log loss / RMSE**; điền thiếu bằng **constant predictor**; điểm chuẩn hóa **RF = 0, tốt nhất = 1**.
- **AutoGluon(B)** hạng đầu nhưng suy luận chậm; **Tuned Random Forest** là baseline mạnh; **cây Bradley-Terry**.
