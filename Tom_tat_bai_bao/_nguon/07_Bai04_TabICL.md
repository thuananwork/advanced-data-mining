# Bài 4. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2025 (PMLR 267); arXiv 2502.05564 (v2, 24/5/2025) |
| Tác giả | Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan (Inria Saclay, Inria Paris, ENS-PSL) |
| Mã nguồn | github.com/soda-inria/tabicl (mã tiền huấn luyện, suy luận và mô hình đã huấn luyện) |
| Vị trí trong đề tài | Tiền thân trực tiếp của TabICLv2 (bài 7). Muốn hiểu v2 thì phải hiểu bài này trước. |

## 1. Bài này làm gì trong một đoạn

TabPFNv2 rất mạnh trên bảng tới 10.000 mẫu nhưng chú ý luân phiên cột–hàng khiến chi phí quá lớn khi bảng to. TabICL hỏi: ICL có mở rộng được cho bảng lớn không? Câu trả lời là một kiến trúc **hai giai đoạn**: (1) **nén** mỗi dòng thành một vector cố định bằng chú ý theo cột rồi chú ý theo dòng; (2) chạy ICL trên các vector dòng đó bằng một Transformer thông thường. Mô hình chỉ **phân loại**, được tiền huấn luyện trên bảng tổng hợp tới **60.000 mẫu**, xử lý được **500.000 mẫu, 500 đặc trưng** với khoảng 20 GB GPU. Trên 200 bộ phân loại của **TALENT**, TabICL **ngang TabPFNv2 nhưng nhanh hơn tới 10 lần** và vượt xa mọi phương pháp khác; trên 53 bộ có hơn 10.000 mẫu, TabICL **vượt cả TabPFNv2 và CatBoost**.

## 2. Vấn đề và động cơ

- GBDT thống trị dữ liệu bảng, nay bị thách thức bởi mô hình nền tảng bảng (TFM) học trong ngữ cảnh. LLM được tinh chỉnh trên bảng đã được serialize thành văn bản bị giới hạn bởi cửa sổ ngữ cảnh (ví dụ 32–64 mẫu trong Gardner 2024), khó xử lý số, và có nguy cơ rò dữ liệu đánh giá do đã thấy các bộ dữ liệu phổ biến.
- TabPFN (v1) chỉ xử lý được bảng tới 1.000 mẫu / 100 đặc trưng; TabPFNv2 tới 10.000 mẫu / 500 đặc trưng, nhưng độ phức tạp **O(m²n + n²m)**: dự đoán 100.000 mẫu rất tốn bộ nhớ và thời gian. Dữ liệu công nghiệp có thể có hàng triệu mẫu.
- Các biến thể mở rộng TabPFN trước đó phần lớn là "prompt engineering", không cải thiện cấu trúc: chưng cất dữ liệu huấn luyện thành ngữ cảnh gọn (TuneTables), chọn tập con huấn luyện liên quan cho từng mẫu (TabDPT, retrieval), thay chú ý bậc hai bằng tuyến tính, hypernetwork sinh mạng nhỏ (MotherNet). TabForestPFN thêm dữ liệu cây vào prior; TabDPT dùng dữ liệu thật.
- Khó khăn của dữ liệu bảng so với văn bản: giá trị ô thiếu ngữ nghĩa nếu không có tên cột và kiểu dữ liệu; mỗi đặc trưng có phân phối khác nhau, các phương pháp cũ dùng mô-đun nhúng riêng cho từng cột nên hạn chế chuyển giao giữa các bảng.

## 3. Ý tưởng cốt lõi

"Nhúng rồi ICL": một mô-đun nhúng bảng (không dùng nhãn) mã hóa mỗi dòng thành vector, **nhãn chỉ được dùng ở mô-đun ICL cuối cùng**. Chiều cột bị nén ở giai đoạn 1 nên giai đoạn ICL chỉ cần chú ý bậc hai theo số dòng, không còn nhân với số cột. Nhờ cấu trúc hàng/cột như một thiên kiến quy nạp, mô hình thực hiện chú ý thưa hiệu quả trên mọi ô.

## 4. Kiến trúc (ba Transformer, Hình 1)

Bài toán: phân loại với X ∈ R^m, nhãn trong [1, …, C]; cho D_train và x_test, dự đoán p(· | x_test, D_train).

### 4.1. Nhúng cột theo phân phối: TF_col (mục 3.2)
- Thay vì mô-đun nhúng riêng cho từng cột, **mọi cột dùng chung một Set Transformer** (Lee 2019) coi cột c_j ∈ R^n là một **tập** giá trị ô bất biến hoán vị. TF_col đóng vai một **hypernetwork**: nhận cả cột và sinh ra trọng số W và độ lệch B **riêng cho từng ô** (W, B = TF_col(c_j); e_j = W ⊙ c_j + B ∈ R^{n×d}). Do đó phép nhúng "nhận biết phân phối" của cột.
- Chi tiết: chiếu cột lên d = 128 chiều bằng lớp tuyến tính (U = Lin(c)), qua **ISAB** (induced self-attention block) gồm hai khối chú ý đa đầu MAB₁ và MAB₂: M = MAB₁(V_I, U_train, U_train) ∈ R^{k×d}, V = MAB₂(U, M, M), rồi W, B = Lin(V). Dùng **k = 128 vector cảm ứng**, 4 đầu chú ý, **3 khối ISAB**. Quan trọng: **chỉ mẫu huấn luyện làm key/value ở MAB₁**, nên đầu ra phụ thuộc duy nhất dữ liệu huấn luyện, **tránh rò rỉ dữ liệu kiểm tra**. Độ phức tạp ISAB giảm xuống tuyến tính theo n.
- Phân tích (Hình 3): PCA tóm tắt M cho 40.000 đặc trưng tổng hợp cho thấy các cột có độ lệch (skewness) hoặc độ nhọn (kurtosis) giống nhau tụ lại gần nhau; tức TF_col mã hóa tính chất phân phối của cột (min, max, mean, mode...), dùng như **định danh đặc trưng** thay cho tên cột có ngữ nghĩa (CARTE) hoặc vector định danh học được (Kossen).

### 4.2. Tương tác đặc trưng theo dòng: TF_row (mục 3.3)
- Với E = [e_1, …, e_m] ∈ R^{n×m×d}, một Transformer **3 lớp, 8 đầu** xử lý từng dòng; thêm **4 token [CLS]** học được vào mỗi dòng, nối đầu ra của chúng thành vector dòng H ∈ R^{n×4d} (4 × 128 = 512). Chiều nhỏ d = 128 ở TF_col và TF_row nhằm tiết kiệm bộ nhớ.
- **Sụp đổ biểu diễn (representation collapse):** vì các đặc trưng được nhận diện qua phân phối, các đặc trưng cùng phân phối khó phân biệt; ở cực đoan nếu mọi đặc trưng cùng phân phối, chú ý bất biến hoán vị không phân biệt được một mẫu với phiên bản hoán vị cột của nó, nên các mẫu khác nhau có biểu diễn gần như trùng nhau. Ví dụ bộ **balance-scale** (4 đặc trưng cùng một phân phối rời rạc 5 giá trị, Hình 4).
- **Giải pháp: RoPE** (rotary positional embedding, như trong LLM) trong TF_row để phá đối xứng giữa các đặc trưng cùng phân phối. Hệ số co giãn tần số rất lớn **100.000** (theo Xiong 2023) để tổng quát hóa lên nhiều đặc trưng hơn lúc huấn luyện (tối đa 100). RoPE làm mất bất biến thứ tự cột, nên TabICL **khôi phục xấp xỉ bằng ensemble trên nhiều hoán vị cột**, cách thông thường của các mô hình giống TabPFN. (TabPFNv2 xử lý sụp đổ bằng vector định danh đặc trưng ngẫu nhiên và nhúng nhóm đặc trưng.)

### 4.3. ICL ở mức bộ dữ liệu: TF_icl (mục 3.4)
- Nhãn huấn luyện mã hóa one-hot, ánh xạ vào cùng không gian với H rồi **cộng** với embedding dòng huấn luyện để được H_train. Transformer **12 lớp, 4 đầu**: embedding huấn luyện chú ý lẫn nhau, **embedding kiểm tra chỉ chú ý tới huấn luyện**. MLP 2 lớp ra xác suất lớp.

### 4.4. Độ phức tạp (mục 3.5)
TF_col: O(n·k·m); TF_row: O(m²n) (m ≪ n thường); TF_icl: O(n²) (chỉ tính một lần vì chiều đặc trưng đã bị nén). Tổng **O(m²n + n²)**, so với O(m²n + n²m) của TabPFNv2.

## 5. Tiền huấn luyện và suy luận

### 5.1. Dữ liệu tổng hợp (mục 4.1, Phụ lục C)
Dùng SCM như TabPFN, với hai mở rộng:
- **SCM dựa trên cây:** hàm f được cài bằng **mô hình XGBoost hồi quy**: ở mỗi lớp của đồ thị, huấn luyện XGBoost trên nhãn giả lấy từ nhiễu Gauss, đầu vào là giá trị các biến cha; dự đoán thành giá trị biến con. Lý do: kế thừa thiên kiến quy nạp của cây (Grinsztajn 2024). Tỷ lệ trộn **70% SCM thường + 30% SCM dựa trên cây** (Hình 9 cho thấy có lợi, thử bằng tiền huấn luyện 20K bước giai đoạn 1).
- **Đa dạng hàm kích hoạt:** TabPFN v1 chỉ có Identity, Tanh, Leaky ReLU, ELU; TabICL thêm **15 hàm** (có hàm không đơn điệu, gián đoạn) và cả hàm kích hoạt **lấy mẫu từ Gaussian process với kernel ngẫu nhiên**; cho phép khác nhau giữa các lớp; chuẩn hóa rồi co giãn ngẫu nhiên trước mỗi hàm kích hoạt.
- Số đặc trưng ≤ 100, số lớp ≤ 10.

### 5.2. Học theo chương trình (curriculum learning, mục 4.2, Phụ lục E.1)
Tăng dần kích thước bảng tổng hợp, mỗi bước 512 bảng, Adam, cắt chuẩn gradient 1:
1. **Giai đoạn 1:** micro-batch NB = 4, cỡ cố định 1.024, **160K bước** (cosine decay có restart).
2. **Giai đoạn 2:** NB = 1, cỡ log-uniform 1K–40K, **2K bước** (suy giảm đa thức từ 2e-5 xuống 5e-6); bật activation checkpointing với bảng > 10K mẫu và giảm số đặc trưng để tránh hết bộ nhớ.
3. **Giai đoạn 3:** NB = 1, cỡ đều 40K–60K, **50 bước**, **chỉ huấn luyện TF_icl** (đóng băng các phần khác), tốc độ học hằng.
- Tổng cộng khoảng **82 triệu bảng**; **20 ngày trên 3 GPU A100 40GB** (16, 3, 1 ngày cho giai đoạn 1, 2, 3).
- Hiệu quả: hạng trung bình trên TALENT cải thiện **11,4 (hạng 9) → 7,46 (hạng 2) → 6,95 (hạng 1)** qua ba giai đoạn. Có sụt nhẹ ở một số bộ nhỏ trong quá trình (Hình 10).

### 5.3. Mở rộng cho > 10 lớp (mục 4.3)
**Phân loại phân cấp:** chia đệ quy và đều các lớp thành nhóm ≤ 10 lớp, tạo cây có độ sâu r = ⌈log₁₀ k⌉; mỗi nút trong dự đoán xác suất các nhóm con, mỗi lá dự đoán xác suất các lớp; xác suất của một lớp = tích xác suất dọc đường từ gốc tới lá. Vì nhãn chỉ dùng ở TF_icl, **mọi bài toán con dùng chung embedding dòng H và chung TF_icl**, nên rất hiệu quả.

### 5.4. Suy luận tiết kiệm bộ nhớ (mục 4.4, Phụ lục E.2)
Dùng FlashAttention; bộ nhớ đỉnh được mô hình hóa bằng hồi quy đa thức α₁·batch + α₂·len + α₃·batch·len + α₄ cho từng Transformer; từ đó điều chỉnh cỡ batch theo độ dài chuỗi và bộ nhớ GPU còn trống (batch là số đặc trưng cho TF_col, số mẫu cho TF_row, số bộ dữ liệu cho TF_icl); đẩy các kích hoạt trung gian ra CPU và đĩa khi cần. Kết quả: bảng **100.000 mẫu × 500 đặc trưng** chỉ cần **5 GB GPU và khoảng 25–32 GB RAM** (Hình E.3).

## 6. Thiết lập thí nghiệm (mục 5.1)

- **Benchmark: TALENT** (Ye 2025): **200 bộ phân loại** (120 nhị phân, 80 đa lớp), bảng tới 150K mẫu, có kết quả của hơn 30 baseline. Loại **15 bộ** TabPFNv2 dùng để chọn siêu tham số/mô hình; chủ yếu xét **171 bộ có ≤ 10 lớp** (TabICL và TabPFNv2 xử lý natively); 12 bộ > 10 lớp xét riêng. Chia 64% huấn luyện / 16% validation / 20% kiểm tra.
- Với TabICL và TabPFNv2 **không dùng validation** (huấn luyện chỉ trên tập huấn luyện; hơi bất lợi), nhưng cả hai dùng **ensemble 32 dự đoán** (hoán vị cột, hoán vị lớp, nhiều tiền xử lý khác nhau), trong khi các mô hình học sâu khác không dùng ensemble (so sánh chưa hoàn toàn công bằng, đã nêu rõ trong bài). TabICL tiền xử lý bằng z-normalization có hoặc không biến đổi lũy thừa. Với TabPFNv2, **lấy mẫu con huấn luyện tối đa 30K** để tránh tràn bộ nhớ; TabICL dự đoán mọi bộ không cần lấy mẫu con.
- Tắt mixed precision lúc suy luận để tái lập và đo thời gian nhất quán. Thời gian của baseline = thời gian huấn luyện với siêu tham số tốt nhất × 100 (xấp xỉ 100 lần tinh chỉnh). Các so sánh gồm: TabPFNv2, ModernNCA, RealMLP, LoCalPFN, CatBoost, TabR, TabM, LightGBM, XGBoost, FT-Transformer, ResNet, TabPFN (v1), MLP-PLR, TuneTables, DCNv2, AutoInt, Random Forest, ExcelFormer, SNN, TANGOS, TabCaps, PTaRL, DANets, NODE, TabTransformer, LogReg, GrowNet, KNN, SwitchTab, TabNet.

## 7. Kết quả

- **Độ chính xác (Hình 5, Hình I.1):** TabICL có **độ chính xác tương đối trung vị so với MLP đã tinh chỉnh tốt nhất** trong số mọi mô hình, đồng thời nhanh hơn nhiều: **trung bình hình học thời gian huấn luyện + suy luận trên 1.000 mẫu = 1,1 giây**, trong khi tinh chỉnh CatBoost trên CPU mất khoảng **3 phút**, RealMLP và ModernNCA trên GPU khoảng **7 phút**. Sơ đồ khác biệt tới hạn: TabICL và TabPFNv2 vượt đối thủ khác với biên độ lớn, **khác biệt giữa hai mô hình không có ý nghĩa thống kê**. Hạng (Phụ lục I, bộ ≤ 10 lớp): TabPFNv2 6,94; TabR 9,32; ModernNCA 9,72; RealMLP 9,76; TabM 9,96; CatBoost 10,16; LoCalPFN 10,54... (số trích từ sơ đồ trong PDF; hãy đối chiếu hình để lấy vị trí của TabICL).
- **Tốc độ so với TabPFNv2 (Hình 6, B.1):** nhanh hơn **1,5 lần ở bảng nhỏ và 3–10 lần ở bảng lớn**; ví dụ 10.000 mẫu × 100 đặc trưng: TabICL ≈ 20 giây so với TabPFNv2 ≈ 1 phút 40 giây; 1.000 mẫu × 10 đặc trưng: 1 giây so với 2 giây. Mô hình hóa thời gian time = α + β(nm(n + m))^0,8: TabPFNv2 y = 2,65 + 10⁻⁷·10,76·x^0,8; TabICL y = 1,87 + 10⁻⁷·2,12·x^0,8; tăng tốc xấp xỉ **5 lần** ở bảng lớn (1,5 lần ở bảng nhỏ).
- **Bảng lớn (Hình 7):** TabPFNv2 chỉ được tiền huấn luyện đến 2.048 mẫu và có thể lỗi trên bảng > 30K; **TabICL vẫn mạnh ở bảng lớn**. Trên **53 bộ > 10K mẫu**, TabICL vượt TabPFNv2 và CatBoost. Gợi ý của tác giả: ngay với dữ liệu lớn, tiền huấn luyện vẫn tạo ra prior ngầm giúp ICL cạnh tranh.
- **Xác suất đáng tin (log loss):** TabICL và TabPFNv2 đều **vượt các baseline được tinh chỉnh theo accuracy** về log loss (khác biệt giữa hai mô hình không có ý nghĩa), tức ước lượng xác suất đáng tin hơn.
- **Hơn 10 lớp (Hình 8, 12 bộ):** với phân cấp, TabICL đạt kết quả tốt thứ hai theo độ chính xác chuẩn hóa trung bình; TabPFNv2 không xử lý được natively.
- **Metafeature (Phụ lục B):** TabICL tệ hơn ở các bộ có **3 lớp** (không rõ do số lớp hay đặc điểm khác của các bộ này trong benchmark), không suy giảm ở số lớp lớn hơn; hoạt động tốt với **nhiều đặc trưng** dù nén cả dòng trước khi thấy nhãn; cả hai mô hình suy giảm nhẹ khi **tỷ lệ biến phân loại cao**, nhưng TabICL hơi tốt hơn TabPFNv2 ở đó dù prior của TabPFNv2 sinh biến phân loại tinh vi hơn.
- **Tiện ích random forest của TabPFNv2 (Phụ lục G):** định tuyến mẫu kiểm tra qua cây rồi khớp TFM tại mỗi lá (có cắt tỉa thích nghi theo validation). Với bộ > 10K mẫu và ensemble 4: **cải thiện đáng kể cả TabPFNv2 và TabICL** (hạng 7,73 cho TabICL-RF, 9,17 cho TabPFNv2-RF trong sơ đồ). (Ở TabICLv2, tiện ích này lại làm hại, xem bài 7.)
- **Kích thước ensemble (Phụ lục H):** TabPFNv2 hưởng lợi rõ ngay với E ≤ 4; TabICL cần E ≥ 8 mới thấy lợi (lợi ích tương đối vài phần mười %, ví dụ ở E = 32 khoảng 0,7–0,75%). Lý do suy đoán: TabPFNv2 tiền xử lý tinh vi hơn, và vì không nén cột nên hoán vị cột ảnh hưởng đầu ra nhiều hơn, tức các thành viên ensemble ít tương quan hơn.
- **Ablation (mục 5.3):** SCM dựa trên cây cải thiện hiệu năng (Hình 9); curriculum learning cải thiện, nhất là trên bảng lớn (Hình 10).

## 8. Hạn chế (tác giả nêu)
1. Suy luận **chậm** như các mô hình nền tảng nói chung (TabPFNv2 cho thấy caching có thể giảm).
2. **Chỉ phân loại** (hồi quy có thể làm tương tự như Hollmann 2025).
3. **Không bất biến hoán vị cột** vì dùng RoPE.
4. Đánh giá thừa hưởng điểm mạnh/yếu của TALENT: TALENT huấn luyện một mô hình duy nhất tinh chỉnh bằng holdout, trong khi ensemble với CV có thể tốt hơn; TabICL không cần tập validation.
5. TALENT điền khuyết bằng **trung bình**; có thể cho TabPFNv2 tự xử lý giá trị thiếu sẽ tốt hơn (tỷ lệ thiếu trong TALENT thấp).
6. Không cung cấp thông tin biến phân loại cho TabPFNv2 (dù nó tự suy ra), nên có thể bất lợi cho nó.
7. So sánh ensemble không đối xứng: các baseline học sâu không dùng ensemble.

## 9. Liên hệ với đề tài
- Hai điểm thiết kế bị TabICLv2 thay đổi: (a) **nhãn nhúng sớm** (v2) so với **muộn** (v1: nhãn chỉ ở giai đoạn ICL); (b) RoPE chống sụp đổ (v1) so với **gom đặc trưng lặp lại** (v2). Cải tiến v2 còn gồm QASSMax, prior mới, Muon, hồi quy phân vị.
- Điểm yếu còn lại làm cơ sở cho khoảng trống: chỉ phân loại, điền trung bình giá trị thiếu, chưa đánh giá mất cân bằng lớp hay nhãn nhiễu, không có ngữ nghĩa cột.

## 10. Từ khóa cần nhớ
- **Nhúng rồi ICL:** nén dòng trước, ICL sau; O(m²n + n²).
- **Set Transformer / ISAB:** nhúng cột bất biến hoán vị, hypernetwork sinh W, B cho từng ô.
- **RoPE:** phá đối xứng giữa đặc trưng cùng phân phối, chống sụp đổ biểu diễn.
- **SCM dựa trên cây (XGBoost):** 30% prior.
- **Curriculum learning:** 1.024 → 40K → 60K mẫu.
- **Phân loại phân cấp:** vượt giới hạn 10 lớp.
