# Bài 21. A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025 (theo bảng tổng hợp đề tài); bản PDF là arXiv 2502.17361v2 (11/6/2025), đề đầu trang ghi "Preprint. Under review" |
| Tác giả | Han-Jia Ye, Si-Yang Liu (Đại học Nam Kinh, LAMDA), Wei-Lun Chao (Ohio State) |
| Vị trí trong đề tài | Bài phân tích TabPFN v2 từ bên ngoài: đo trên hơn 300 bộ dữ liệu, giải thích vì sao nó hoạt động, và đề xuất cách khắc phục giới hạn mà không huấn luyện lại. |

## 1. Bài này làm gì trong một đoạn

Bài **không đề xuất kiến trúc mới**. Nó mở hộp đen TabPFN v2 bằng ba đóng góp: (1) cho thấy TabPFN v2 xử lý được dữ liệu bảng không đồng nhất vì nó **học quan hệ giữa các đặc trưng ngay trong lúc suy luận**, dù token đặc trưng được lấy ngẫu nhiên mỗi lần; (2) biến TabPFN v2 thành **bộ trích đặc trưng** (embedding) bằng chiến lược "leave-one-fold-out", cho thấy một bộ phân loại tuyến tính trên embedding đạt hiệu năng ngang bộ học trong ngữ cảnh gốc; (3) khắc phục ba giới hạn (**nhiều chiều, nhiều lớp, quy mô lớn**) bằng các chiến lược **chia để trị lúc kiểm tra (test-time)**, không cần huấn luyện lại.

## 2. Vấn đề và động cơ

- Dữ liệu bảng không đồng nhất (số cột, ý nghĩa cột khác nhau giữa các bảng) nên khó có mô hình nền tảng như ở ảnh/văn bản. Nhiều bảng nhỏ (ví dụ dữ liệu y tế) nên huấn luyện từ đầu rất nhạy siêu tham số.
- TabPFN v2 (bài 1) là bước tiến lớn nhưng: (a) cơ chế vì sao nó xử lý được tính không đồng nhất chưa được giải thích (chi tiết **token đặc trưng ngẫu nhiên** chỉ nằm trong phụ lục và mã nguồn); (b) có **giới hạn khuyến nghị** ≤ 10.000 mẫu, ≤ 500 chiều, ≤ 10 lớp; (c) dữ liệu và công thức huấn luyện không công bố, nên bài chỉ phân tích mô hình đã phát hành.

## 3. Nền tảng: TabPFN v1 và v2 theo cách bài mô tả

- **TabPFN v1:** đệm số 0 các đặc trưng tới chiều cố định (ví dụ 100), chiếu tuyến tính x_i và y_i lên chiều k, tạo ngữ cảnh gồm N + 1 token, qua Transformer, token kiểm tra qua MLP ra logit 10 lớp.
- **TabPFN v2:** mỗi đặc trưng của mẫu được nhúng vào không gian k chiều **cộng nhiễu ngẫu nhiên** để phân biệt đặc trưng; cùng với nhúng nhãn, mỗi mẫu thành d + 1 token. Với mẫu kiểm tra dùng nhãn giả (ví dụ nhãn trung bình của tập huấn luyện). Toàn bộ đầu vào là tensor (N + 1) × (d + 1) × k. Hai loại tự chú ý luân phiên: **giữa các mẫu** (inter-sample) và **giữa các đặc trưng** (inter-feature). Token của nhãn giả ở mẫu kiểm tra qua MLP ra logit 10 lớp (phân loại) hoặc một logit (hồi quy). Trọng số tiền huấn luyện trên dữ liệu tổng hợp từ SCM, **checkpoint được chọn theo dữ liệu thật**.

## 4. Đánh giá toàn diện TabPFN v2 (mục 4)

### Thiết lập
- Dùng benchmark của Ye et al. (tham chiếu [76] trong bài, tức bộ TALENT; gồm **120 phân loại nhị phân, 80 phân loại đa lớp, 100 hồi quy = 300 bộ**). Loại **27 bộ** từng dùng để chọn checkpoint của TabPFN v2 ⇒ còn **273 bộ** (tránh thiên lệch).
- Chia **64% huấn luyện / 16% validation / 20% kiểm tra**. TabPFN v2 dự đoán trực tiếp bằng ICL, không tinh chỉnh. **26 phương pháp** baseline (cây và học sâu) được tinh chỉnh bằng **Optuna, 100 lần thử** trên tập huấn luyện, dừng sớm theo validation. Mỗi phương pháp chạy **15 hạt giống**, lấy trung bình. Thước đo: độ chính xác (phân loại), RMSE (hồi quy). Với > 10 lớp dùng ECOC có sẵn của TabPFN v2.
- Kiểm định: Wilcoxon-Holm mức 0,05; và **PAMA** (xác suất đạt độ chính xác cao nhất / RMSE thấp nhất trên một bộ dữ liệu).

### Điểm mạnh
- Sơ đồ khác biệt tới hạn (Hình 1 phải): TabPFN v2 **vượt có ý nghĩa thống kê** cả cây (CatBoost, LightGBM, XGBoost, RF) lẫn mô hình sâu mạnh nhất (RealMLP, ModernNCA, TabM, TabR, FT-Transformer).
- PAMA (Hình 2): TabPFN v2 đạt kết quả tốt nhất trên **26,02%** bộ dữ liệu; ModernNCA **11,79%**; TabM **9,78%**.

### Điểm yếu (mục 4.3, Bảng 1)
Ba nhóm dữ liệu thử thêm: **18 bộ nhiều chiều** (d ≥ 2.000), **18 bộ quy mô lớn** (N × d > 1.000.000, dùng siêu tham số mặc định cho mọi phương pháp vì tinh chỉnh quá tốn), **12 bộ có hơn 10 lớp**.
- Trên **quy mô lớn**: TabPFN v2 xếp **dưới CatBoost và RealMLP**.
- Trên **nhiều chiều**: TabPFN v2 **thậm chí thua hồi quy logistic đơn giản** (độ chính xác trung bình 85,25% so với 89,36%, Bảng 10).
- Trên **> 10 lớp**: chiến lược ECOC của TabPFN v2 **kém hiệu quả**.
- Giả thuyết của tác giả: TabPFN v2 chỉ được tiền huấn luyện trên bảng tổng hợp nhỏ-vừa, dưới 10 lớp, nên lệch phân phối với dữ liệu lớn/phức tạp hơn.

## 5. TabPFN v2 xử lý tính không đồng nhất bằng cách nào (mục 5)

- Cách làm cũ: token hóa mỗi đặc trưng thành một token k chiều, dùng embedding ngữ nghĩa có sẵn (vector từ của tên cột) hoặc **embedding riêng theo bộ dữ liệu** (tốn kém, không dùng lại được).
- **Cách của TabPFN v2:** học một vector chung u ∈ R^k để nâng từng giá trị x_i^j lên k chiều, rồi cộng nhiễu ngẫu nhiên r_j = W p_j, với p_j là vector ngẫu nhiên và W là ma trận chiếu học được. Một mẫu thành chuỗi [x_i^1·u + r_1, …, x_i^d·u + r_d, nhãn]. Tức token đặc trưng **được lấy ngẫu nhiên lại mỗi lần suy luận**.
- **Phát hiện chính:** dù ngẫu nhiên, thông qua ICL mô hình **suy ra được quan hệ giữa các thuộc tính** (bài gọi là "nội hóa việc học token thuộc tính vào suy luận"). Bằng chứng, trên hai bộ **churn** và **bank** (Hình 3):
  1. **PCA của token**: ban đầu các token của các thuộc tính rải ngẫu nhiên; qua các lớp 3, 6, 9, 12 chúng dần tạo cụm có cấu trúc (ví dụ ở bank, "job", "education", "balance" tụ thành nhóm có nghĩa).
  2. **Bản đồ chú ý giữa thuộc tính**: lớp đầu, thuộc tính chủ yếu chú ý vào token nhãn (nhận tín hiệu tác vụ); lớp giữa, chú ý phân bố đều hơn (trao đổi thông tin); lớp sâu, chú ý tập trung vào các thuộc tính liên quan về mặt ngữ nghĩa.
  3. **Độ ổn định**: qua 10 lần chạy với khởi tạo ngẫu nhiên khác nhau, độ tương đồng cosine của bản đồ chú ý vẫn cao ngoại trừ lớp đầu tiên.
- Diễn giải: nhiễu ngẫu nhiên chỉ có chức năng phân biệt các thuộc tính, giống mã hóa one-hot; mô hình học các mẫu dự đoán (đồng xuất hiện, phân phối giá trị, độ lớn tương đối) trực tiếp từ cấu trúc thống kê của từng bảng. Nhờ vậy **hợp nhất học biểu diễn và dự đoán trong một lần chạy xuôi**.

## 6. Biến TabPFN v2 thành bộ trích đặc trưng (mục 6)

- **Cách ngây thơ thất bại:** lấy token ra tương ứng nhãn của mẫu huấn luyện làm embedding cho mẫu huấn luyện, token ra của nhãn giả làm embedding cho mẫu kiểm tra. Kết quả (Hình 4b) phân phối embedding huấn luyện và kiểm tra **lệch hẳn nhau** (nhãn thật so với nhãn giả khiến vai trò khác nhau), nên bộ phân loại tuyến tính trên embedding huấn luyện kém trên kiểm tra (độ chính xác trên churn 0,5650 so với 0,8600 ở đặc trưng thô; bank 0,6105 so với 0,8876).
- **Leave-one-fold-out:** chia tập huấn luyện thành nhiều phần (ví dụ 10). Mỗi vòng, một phần đóng vai **tập truy vấn Q** (nhãn giả) để lấy embedding, các phần còn lại là **tập hỗ trợ S** (nhãn thật). Giữ S lớn để ICL hiệu quả mà vẫn lấy được embedding cho mọi mẫu huấn luyện.
- **Kết quả (Hình 4c–f):** embedding trích theo cách này phản ánh đúng cấu trúc dữ liệu; sau các lớp giữa dữ liệu trở nên **gần như tách tuyến tính**. Hồi quy logistic trên embedding đạt, ví dụ trên churn: lớp 3 → 0,8590, lớp 6 → 0,9070, lớp 9 → 0,9330, lớp 12 → 0,9590; trên bank: 0,8828, 0,9046, 0,9047, 0,9085 (lớp 3 đến 12).
- **Xác nhận định lượng (Bảng 2, 29 bộ phân loại thuộc benchmark "tiny" của bài [76]):** hạng trung bình (thấp là tốt): TabPFN v2 gốc **2,69**; embedding vanilla **5,97**; lớp 6 **4,28**; lớp 9 **4,00**; lớp 12 **2,12**; **kết hợp** (nối embedding của tối đa 3 lớp chọn theo validation) **1,94**. Tức bộ phân loại tuyến tính trên embedding lớp 12 hoặc kết hợp **ngang hoặc hơn** bộ học ICL gốc.
- Hàm ý: dùng để trực quan hóa, phân tích lỗi, chẩn đoán dữ liệu. Phụ lục C.2 còn thử hai cách trích embedding không giám sát (S: thêm cột nhãn giả toàn 0; P: hoán vị từng cột làm nhãn) — Bảng 13 cho độ chính xác từng bộ.

## 7. Chia để trị lúc kiểm tra (mục 7)

Ý tưởng lấy cảm hứng từ chain-of-thought / test-time scaling của LLM: chia bài toán khó thành các bài toán con TabPFN v2 giải được.

### 7.1. Nhiều chiều → TabPFN v2* (bagging đặc trưng)
Lấy m tập con, mỗi tập d′ < d đặc trưng ngẫu nhiên (giống random forest), chạy TabPFN v2 trên từng tập, **gộp bằng trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại)**. Chọn **d′ = 500, m = ⌈d/d′⌉**. Kết quả trên 18 bộ nhiều chiều (Bảng 10, độ chính xác trung bình): **TabPFN v2* 89,73%** (cao nhất), hồi quy logistic 89,36%, RealMLP 88,83%, CatBoost 87,48%, PCA+TabPFN v2 87,29%, MLP 87,11%, RF 86,87%, ProtoGate 86,37%, TabPFN v2 gốc **85,25%**, XGBoost 84,76%, ModernNCA 82,86%, KNN 77,29%. Biến thể dùng PCA giảm chiều cho cải thiện một phần.

### 7.2. Hơn 10 lớp → mã hóa thập phân (decimal encoding)
Với C > 10 lớp: mã hóa nhãn thành **t = ⌈log₁₀ C⌉ chữ số thập phân**, huấn luyện (thực ra chạy) **một TabPFN v2 cho mỗi chữ số**, ghép lại ra nhãn. Cách này cũng có trong TabDPT (bài tương ứng, ký hiệu TabPFN v2-DPT). Vì ánh xạ nhãn–chữ số tạo tương quan nhân tạo, bản **TabPFN v2*** **hoán vị ngẫu nhiên ánh xạ lớp→chữ số √C lần** rồi gộp dự đoán. Kết quả 12 bộ > 10 lớp (Bảng 11, độ chính xác trung bình): RealMLP **72,51%**, **TabPFN v2* 70,93% (hạng nhì)**, ModernNCA 68,00%, CatBoost 67,47%, TabPFN v2-ECOC 66,69%, XGBoost 66,47%, TabPFN v2-DPT 65,67%, RF 64,01%, MLP 62,16%, KNN 61,94%, hồi quy logistic 51,40%.

### 7.3. Quy mô lớn → hai chiến lược
- **TabPFN v2\*-SQ (support/query):** lấy ngẫu nhiên **10.000** mẫu huấn luyện làm tập hỗ trợ, phần còn lại của tập huấn luyện + kiểm tra làm tập truy vấn; trích **embedding** của tất cả, rồi huấn luyện **hồi quy logistic** trên embedding để dự đoán tập kiểm tra. Lặp 4 lần và gộp dự đoán.
- **TabPFN v2\*-DF (decision-forest):** lấy **32** tập con, mỗi tập chiếm **60%** dữ liệu (không hoàn lại); mỗi tập huấn luyện một **cây nông** (số mẫu tối thiểu để chia nút = 10.000) để chia dữ liệu thành các nhóm nhỏ, mỗi lá một TabPFN v2; lúc suy luận, mẫu đi qua 32 cây tới lá tương ứng rồi gộp 32 dự đoán. (Chiến lược tương tự được nhắc ở bài Nature, nhưng mục đích khác.)
- Có thêm hai biến thể đối chứng: **-B** (lấy 10.000 mẫu ngẫu nhiên 4 lần và gộp), **-K** (chọn 10.000 mẫu đại diện theo gần các nguyên mẫu của KMeans). Kết quả (Hình 5 phải, hạng trung bình trên 18 bộ quy mô lớn): **mọi biến thể đều cải thiện TabPFN v2 gốc**, trong đó **-DF và -SQ cải thiện nhiều nhất**. Bảng 12 cho số liệu từng bộ (Higgs, covertype, Fashion-MNIST, Airlines_DepDelay, UJIndoorLoc, microsoft, yahoo...).

## 8. Kết luận của tác giả
Cơ chế cốt lõi của TabPFN v2 là suy ra quan hệ thuộc tính ngay trong suy luận từ token ngẫu nhiên, không cần ngữ nghĩa định sẵn hay embedding riêng theo bộ dữ liệu; nó là bộ trích đặc trưng mạnh; và giới hạn của nó có thể giải quyết sau khi huấn luyện bằng chia để trị.

## 9. Hạn chế (tác giả nêu hoặc suy ra)
- Bản PDF trong thư mục là bản arXiv v2 (còn ghi "under review"); không có mã sinh dữ liệu huấn luyện của TabPFN v2 nên chỉ phân tích được mô hình đã phát hành (hộp xám).
- Các phương pháp mở rộng là **chắp vá lúc suy luận**, tăng chi phí tính (nhiều lần chạy TabPFN, nhiều cây, nhiều hoán vị), không giải quyết nguyên nhân gốc là kiến trúc và dữ liệu huấn luyện.
- Trên bộ > 10 lớp RealMLP vẫn tốt hơn TabPFN v2*; trên bộ nhiều chiều hồi quy logistic chỉ kém TabPFN v2* 0,37 điểm phần trăm.
- Đánh giá chỉ về độ chính xác/RMSE; **không đo hiệu chuẩn, độ bền với nhiễu nhãn hay mất cân bằng lớp**.

## 10. Liên hệ với bài Nature và đề tài
- Bổ sung cho bài 1: kiểm chứng ngoài phòng thí nghiệm tác giả (hơn 300 bộ) và **xác nhận ba giới hạn** (nhiều chiều, quy mô lớn, > 10 lớp) mà Nature đã thừa nhận.
- Đáng chú ý cho TabICLv2: các hướng "chia để trị lúc kiểm tra" và "embedding + bộ phân loại tuyến tính" là những công cụ có thể áp dụng cho bất kỳ mô hình nền tảng bảng nào, không riêng TabPFN.
- Đối chiếu khi cần số cụ thể: Bảng 10–13 trong phụ lục E của PDF (kết quả từng bộ dữ liệu).

## 11. Từ khóa cần nhớ
- **Token thuộc tính ngẫu nhiên:** nhiễu r_j = W p_j cộng vào mỗi đặc trưng, lấy lại mỗi lần suy luận.
- **Leave-one-fold-out:** chia huấn luyện thành các phần, lần lượt làm tập truy vấn để lấy embedding so sánh được.
- **PAMA:** xác suất đạt kết quả tốt nhất trên một bộ.
- **Decimal encoding:** mã hóa nhãn thành chữ số thập phân để vượt giới hạn 10 lớp.
- **TabPFN v2\*-SQ / -DF:** hai chiến lược chia để trị cho dữ liệu lớn.
