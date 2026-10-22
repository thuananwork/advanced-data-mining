# Bài 8. TabDPT: Scaling Tabular Foundation Models on Real Data

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025; arXiv 2410.18164 (v3, 17/1/2026) |
| Tác giả | Junwei Ma*, Valentin Thomas* (đồng tác giả chính), Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari, Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, Maksims Volkovs (Layer 6 AI, Toronto) |
| Mã nguồn | Mã suy luận kèm trọng số, mã huấn luyện, và danh sách bộ dữ liệu tiền huấn luyện đều công khai |
| Vị trí trong đề tài | Một trong 4 SOTA cần so sánh (TabPFNv2, TabICLv2, **TabDPT**, LoCalPFN). Đại diện cho hướng **tiền huấn luyện trên dữ liệu thật** thay cho dữ liệu tổng hợp. |

## 1. Bài này làm gì trong một đoạn

TabDPT (Tabular Discriminative Pre-trained Transformer) là mô hình nền tảng bảng cho cả **phân loại và hồi quy**, được tiền huấn luyện **hoàn toàn trên dữ liệu bảng thật** (123 bộ từ OpenML, 32 triệu dòng, 2 tỷ ô) thay vì dữ liệu tổng hợp. Điểm đột phá là kết hợp hai kỹ thuật: (i) **học tự giám sát (SSL) bằng che cột**: chọn ngẫu nhiên một cột làm nhãn giả để dự đoán từ các cột còn lại, biến mỗi bảng thành vô số bài toán; (ii) **truy hồi ngữ cảnh (retrieval)** cả khi huấn luyện lẫn suy luận: chỉ lấy K dòng gần nhất làm ngữ cảnh. Trên OpenML-CC18 (phân loại) và OpenML-CTR23 (hồi quy), TabDPT khớp hoặc vượt TabPFN v2 và các mô hình riêng lẻ đã tinh chỉnh, và cho thấy **quy luật lũy thừa (scaling laws)** theo cả kích thước mô hình lẫn lượng dữ liệu.

## 2. Vấn đề và động cơ

- Xây TFM tổng quát hóa qua sự đa dạng khổng lồ của bảng dữ liệu là bài toán chính; cách truyền thống (huấn luyện riêng cho mỗi tác vụ, chọn mô hình và chỉnh siêu tham số) tốn kém, nhất là với học sâu.
- **ICL** tránh được tinh chỉnh theo từng bộ dữ liệu, tiện cho việc thử nhanh và thích ứng khi phân phối dịch chuyển (đưa dữ liệu mới vào ngữ cảnh).
- Dùng **LLM** cho ICL bảng gặp trở ngại: tokenization số rất kém hiệu quả (bảng vừa phải đã tràn cửa sổ ngữ cảnh), kết quả phụ thuộc định dạng prompt và **thứ tự ví dụ**, trong khi dữ liệu bảng không có thứ tự. Tabula-8B (Llama 3-8B tinh chỉnh) chỉ nhận được 32 mẫu.
- Mô hình bảng chuyên dụng như TabPFN hoàn toàn dựa vào bộ sinh dữ liệu tổng hợp, mà việc làm cho nó phản ánh sự đa dạng của dữ liệu thật rất khó; TabPFN v2 không công bố mã sinh dữ liệu và mã tiền huấn luyện; TabICL cần quy trình ba giai đoạn phức tạp.
- **Giả thuyết của bài:** dữ liệu bảng thật chứa nhiều thông tin hơn các bộ sinh tổng hợp được thiết kế kỹ, nên có thể cải thiện bằng cách tăng dữ liệu và mô hình.
- Vấn đề với học có giám sát trên dữ liệu thật: số bộ công khai chất lượng cao ít, mỗi bộ chỉ có một cột nhãn, nên huấn luyện giám sát nhanh bão hòa và quá khớp. Giải pháp: SSL.

## 3. Phương pháp (mục 3)

### 3.1. Bộ mã hóa Transformer theo dòng
- **Mỗi dòng của bảng là một token**, như TabPFN v1. Chuẩn hóa số chiều đặc trưng về F_max bằng đệm số 0 (F < F_max) hoặc giảm chiều (F > F_max), rồi nhúng mỗi dòng thành vector d chiều. Các dòng chú ý lẫn nhau qua các lớp Transformer.
- Lý do chọn mã hóa theo dòng: ô hoặc token văn bản làm số token là O(N × F × ⟨N_tok⟩) vượt giới hạn ngữ cảnh; theo dòng chỉ còn N token, nhờ vậy chạy được nhiều dòng hơn với bộ nhớ thấp. Không dùng mã hóa vị trí nên **bất biến thứ tự dòng**; bất biến thứ tự cột chỉ xấp xỉ bằng hoán vị cột lúc huấn luyện và ensemble lúc suy luận (nhiều nghiên cứu khác cũng thấy bất biến cột nghiêm ngặt gây mất ổn định khi huấn luyện).
- **Một backbone, hai đầu MLP**: một đầu phân loại (tối đa C_max = 10 lớp), một đầu hồi quy, chia sẻ tham số giữa hai loại tác vụ.

### 3.2. Học tự giám sát trên bảng (mục 3.2, Phụ lục D)
- Với mỗi bảng D = {X, y}: ngẫu nhiên chọn **tác vụ là hồi quy hoặc phân loại**, chọn một **cột c có đủ giá trị khác nhau** làm nhãn; bỏ cột c khỏi bảng, chuẩn hóa giá trị (hồi quy) hoặc **chia thùng thành lớp** (phân loại). Mô hình phải dự đoán y = c từ X\c.
- **Xáo và bỏ ngẫu nhiên** các cột còn lại (giữ ngẫu nhiên từ F/2 đến F cột) để buộc mô hình học từ các tổ hợp đặc trưng khác nhau. Không có các tăng cường này, số tác vụ chỉ tăng tuyến tính theo số đặc trưng; với chúng, số tác vụ tăng **tổ hợp**, tạo tín hiệu huấn luyện giàu hơn và đóng vai trò điều chuẩn.
- Mã giả (Code Block 1): nếu nhãn có hơn 10 giá trị khác nhau thì giữ làm hồi quy với xác suất 70% (30% còn lại chia ngưỡng thành 2–9 lớp); nếu từ 10 giá trị trở xuống thì làm phân loại sau khi mã hóa nhãn và xáo.

### 3.3. Tiền huấn luyện dựa trên truy hồi (mục 3.3)
- ICL thuần dùng toàn bộ bảng làm ngữ cảnh nhanh chóng quá lớn; truy hồi K dòng giống nhất làm ngữ cảnh đã được chứng minh cải thiện độ chính xác và khả năng mở rộng khi suy luận. Bài đẩy xa hơn: **căn chỉnh huấn luyện với suy luận bằng cách cũng dùng truy hồi khi xây batch huấn luyện**.
- Thủ tục: sau khi có y = c và X\c, lấy K dòng "gần nhau" trong không gian đặc trưng (dùng **FAISS**, truy hồi trong không gian đặc trưng thô sau tiền xử lý), chia ngẫu nhiên đều thành **ngữ cảnh** {X_ctx, y_ctx} và **truy vấn** {X_qy, y_qy}. Dự đoán ŷ_qy = Transformer[φ_x(X_ctx) ⊕ φ_y(y_ctx), φ_x(X_qy)], với ⊕ là cộng theo phần tử; **truy vấn chú ý tới mọi ngữ cảnh, ngữ cảnh chú ý lẫn nhau nhưng không chú ý truy vấn** (chỉ ngữ cảnh làm key).
- Mã giả (Algorithm 1): chọn B bảng; mỗi bảng ngẫu nhiên chọn tác vụ, sinh nhãn từ cột ngẫu nhiên, lấy K dòng gần, chia ngữ cảnh/truy vấn, xáo/bỏ cột, dự đoán, tính mất mát (cross-entropy cho phân loại, MSE cho hồi quy) và cập nhật. Vị trí chia eval_pos lấy đều ngẫu nhiên từ 10 đến độ dài.
- Kích thước ngữ cảnh không còn tăng theo kích thước bảng, giữ batch hiệu quả; huấn luyện với truy hồi **hội tụ nhanh hơn và chính xác hơn**.

### 3.4. Suy luận (mục 3.4)
- Với bộ dữ liệu mới: truy hồi **top-K dòng gần nhất** của mỗi dòng truy vấn từ tập huấn luyện làm ngữ cảnh, rồi chỉ chạy các lượt xuôi. Không tinh chỉnh theo bộ dữ liệu. Truy hồi thêm độ trễ nhưng FAISS trả lời trong mili-giây và mở rộng tới hàng tỷ dòng.
- **Nhiều hơn 10 lớp:** biểu diễn C theo cơ số C_max và phân loại **từng chữ số cơ số C_max** như một bài toán riêng; số lượt xuôi là ⌈log_{C_max}(C)⌉ (thay vì C lượt với one-vs-all).
- **Nhiều hơn F_max đặc trưng:** giảm chiều bằng **PCA** về F_max.
- **Tiền xử lý tối thiểu (Phụ lục C.1):** cột không phải số được mã hóa thành số nguyên bằng LabelEncoder; chuẩn hóa trung bình 0, phương sai 1; cắt ngoại lệ vượt ±10; sau truy hồi chuẩn hóa lại ngữ cảnh (và nhãn hồi quy). **Giá trị thiếu: đặt bằng 0 sau chuẩn hóa tức thay bằng trung bình** (thử thêm mặt nạ nhị phân "thiếu" chỉ cải thiện tối thiểu mà gần gấp đôi chi phí).

## 4. Cấu hình mô hình và huấn luyện (Phụ lục C)

- Mô hình chính **78 triệu tham số, 16 lớp Transformer** (kích thước nhúng tới 768), 4 đầu chú ý, hệ số FFN 2, tối đa 10 lớp và 100 đặc trưng, chuẩn hóa trước (pre-norm), dropout 0. Các cỡ khác dùng cho quy luật lũy thừa: 3, 4, 5, 6, 10, 12, 16 lớp với chiều 32, 64, 96, 256, 384, 512, 768, tổng từ **33 nghìn đến 78 triệu tham số**.
- Huấn luyện **600K bước** trên GPU A100 40GB. Bộ tối ưu **Schedule-Free AdamW** (Defazio), tốc độ học 5×10⁻⁴, weight decay 5×10⁻², **label smoothing 0,1**, batch 256, độ dài ngữ cảnh và truy vấn 1.024, tham số bfloat16. (TabICLv2 lưu ý không thấy Schedule-Free hơn AdamW cosine.)
- **Dữ liệu huấn luyện:** thu thập từ OpenML (giấy phép CC-BY), gồm các bộ trong Grinsztajn (bài 29), TabZilla (bài 28) và AMLB, cộng các bộ tìm thêm. Tổng cộng **123 bộ, 32 triệu dòng, 2 tỷ ô**, thuộc nhiều lĩnh vực (hành vi con người 17, tài chính/nhân khẩu 15, công nghiệp 11, y tế/cảm biến 11, sinh học/sinh thái 11, mô phỏng 9, thị giác/âm thanh/văn bản 7, vật lý 6, ... "khác" 28). Quy mô ngang Tabula-8B.

## 5. Thiết lập thí nghiệm (mục 4)

- **Dữ liệu đánh giá (tổng 107 bộ):** **OpenML-CC18** (72 bộ phân loại, 500–100.000 mẫu, dưới 5.000 đặc trưng, bỏ dữ liệu tổng hợp và bộ mà thuật toán đơn giản đạt 100% accuracy) và **OpenML-CTR23** (35 bộ hồi quy, cùng giới hạn nhưng yêu cầu mô hình tuyến tính không đạt R² = 1).
- **Baseline:** XGBoost, LightGBM, CatBoost (lấy kết quả từ TabZilla, bổ sung các bộ thiếu theo giao thức TabZilla bằng mã của TabR), TabR, TabM, MLP-PLR, MLP (30 vòng tìm kiếm siêu tham số, chọn tốt nhất cho từng fold theo validation); các mô hình ICL: **Tabula-8B** (LLM), **TabPFN v2**, **TabPFN (kNN)** (TabPFN có truy hồi lúc suy luận), TabPFN v1. Truy hồi với mô hình theo ô như TabPFN v2 sẽ quá tốn. Chạy ít nhất hai chia dữ liệu, khoảng tin cậy 95% bằng bootstrap.
- TabDPT: ngữ cảnh 2.048, 8 thành viên ensemble (hoán vị đặc trưng và lớp). **Gộp tập validation vào tập huấn luyện** (không cần validation), trong khi một số baseline tinh chỉnh trên validation riêng; để công bằng có biến thể **TabDPT (no val.)** chỉ dùng tập huấn luyện.

## 6. Kết quả

- **Bảng 1 (trung bình các thước đo, khoảng tin cậy 95%):** CC18 AUC / Accuracy; CTR23 Correlation / R².
  - **TabDPT:** AUC 0,933; Acc 0,884; Corr 0,872 (cột kế tiếp theo thứ tự trích là 0,837/0,742 cho Correlation/R² ở dòng đầu; hãy đối chiếu PDF Bảng 1 cho từng ô).
  - **TabPFN v2:** AUC 0,932 (khoảng 0,928–0,936); các số còn lại gần TabDPT.
  - **TabPFN (kNN):** AUC 0,918; **TabPFN v1:** AUC 0,898; **TabDPT (no val.):** AUC 0,928.
  - Baseline riêng lẻ: TabR AUC 0,925; XGBoost 0,926; CatBoost 0,926; LightGBM 0,924; TabM 0,917; MLP-PLR 0,912; MLP 0,872.
  - (Cột Corr và R² bị lệch trong bản trích văn bản; nhận định chắc chắn từ bài: **TabDPT tốt nhất tổng thể**, cạnh tranh với TabPFN v2 trên mọi thước đo và **vượt rõ về accuracy trên CC18**; "no val." chỉ thua TabM trên accuracy CC18 dù không dùng validation.)
- **Tỷ lệ thắng đôi (Hình 3a):** trung bình accuracy/R², TabDPT thắng TabPFN v2 ở **58%** bộ (TabPFN v2 thắng 42%), thắng TabR ở 69%, MLP-PLR 70%, XGBoost 71%, LightGBM 75%, CatBoost 76%, TabPFN (kNN) 81%, MLP 88%; **Tabula-8B (32-shot) hầu như không thắng ai** (thắng tối đa 23% so với MLP), cho thấy LLM chưa thích hợp với bảng. Thêm xếp hạng Elo và Glicko2 (Phụ lục E): nhất quán, TabDPT đầu tiên, TabPFN v2 thứ hai.
- **Độ bền theo đặc tính dữ liệu (Phụ lục F.1):** TabDPT ổn định khi thay đổi kích thước bộ, số đặc trưng, tỷ lệ biến phân loại và tỷ lệ giá trị thiếu.
- **TabArena (cập nhật 14/1/2026, theo bảng xếp hạng bên ngoài):** TabDPT có Elo cao nhất trong các TFM **mã nguồn mở hoàn toàn**, đứng thứ ba tổng thể (sau RealMLP và Real TabPFN-2.5; RealMLP không phải TFM, Real TabPFN-2.5 đóng nguồn tiền huấn luyện), và có **hiệu năng mặc định hồi quy cao nhất** trong mọi mô hình, ngang Real TabPFN-2.5 sau khi tinh chỉnh và ensemble.
- **Tốc độ (mục 4.7, Hình 3b):** kể cả cấu hình lớn nhất (ngữ cảnh 2.048) vẫn **nhanh hơn ít nhất một bậc** so với cây và học sâu đặc thù từng bộ (tính cả tìm siêu tham số); ngang TabPFN v2 nhưng chính xác hơn. Chi phí tiền huấn luyện không tính (một lần). Nhưng nếu không tính chi phí huấn luyện, pipeline dựa trên cây có thể nhanh hơn khoảng một bậc cho riêng suy luận (≈ 0,02–0,10 s/1.000 dòng).
- **Few-shot bán giám sát (mục 4.6, Bảng 2):** giao thức STUNT, **10 mẫu có nhãn cho mỗi lớp**, phần còn lại không nhãn, trên 7 bộ CC18 (cmc, karhunen, optdigit, diabetes, semeion, pixel, dna), 50 hạt giống. Cách làm **TabDPT (semi)**: dùng 10 mẫu có nhãn làm ngữ cảnh dự đoán xác suất lớp cho tập huấn luyện không nhãn, lấy top-1.000 điểm có xác suất cao nhất cùng nhãn giả làm ngữ cảnh. Độ chính xác trung bình: **TabDPT (semi) 77,56%**, STUNT 76,76%, CACTUs 75,16%, VIME+LR 74,55%, kNN (STUNT) 74,11%, ICT 74,01%. Thắng STUNT ở **5/7 bộ** (thua ở dna: 61,85 so với 80,96, và diabetes: 72,01 so với 72,82) và chỉ cần các lượt xuôi.

### Ablation (mục 4.4, Hình 4)
- **Bỏ SSL** (chỉ dùng nhãn gốc): **mất hiệu năng nhiều nhất**; không có SSL mô hình bắt đầu quá khớp sau ~50 epoch, còn có SSL tiếp tục cải thiện sau 500 epoch.
- **Bỏ truy hồi khi huấn luyện** (lấy mẫu con ngẫu nhiên): giảm nhất quán cả phân loại lẫn hồi quy, nhỏ hơn bỏ SSL.
- **Dữ liệu thật + SSL so với tổng hợp (dùng bộ sinh của TabPFN):** thật + SSL cho mất mát kiểm tra thấp hơn ở mọi epoch trong cùng ngân sách tính toán; cả hai bão hòa ở 500 epoch (xác nhận bằng 1.024 epoch).
- **Suy luận:** dùng lấy mẫu con thay truy hồi giảm hiệu năng đáng kể; ngữ cảnh nhỏ 256 giảm hiệu năng nhưng ít hơn nhiều.

### Quy luật lũy thừa (mục 4.5, Hình 1)
Phân tích đầu tiên về scaling laws cho TFM không giới hạn miền. Thay đổi cả kích thước mô hình (33K–78M tham số) và dữ liệu (52M ô = 104K dòng tới 2B ô = 32M dòng); đo kích thước dữ liệu bằng **số ô** vì không phải dòng nào cũng đóng góp như nhau. Mô hình chung (Hoffmann): ℓ̂(P, D) = A·P^(−α) + B·D^(−β) + E, ước lượng theo Besiroglu: **α = 0,42, β = 0,39**. Mất mát = trung bình của cross-entropy (phân loại) và 1 − ρ (hồi quy, ρ là tương quan dự đoán–mục tiêu, tương đương MSE cho vector chuẩn hóa). Cải thiện nhất quán khi tăng cả mô hình lẫn dữ liệu ⇒ thông tin trong dữ liệu bảng thật có thể khai thác bằng SSL.

## 7. Hạn chế (tác giả nêu và suy ra)
1. Embedding tên đặc trưng thử nghiệm sơ bộ gây **quá khớp**; cần thêm bảng có văn bản tự do.
2. Thiết kế cho bảng chữ nhật với dòng i.i.d.; **không mô hình hóa phụ thuộc thời gian, dịch chuyển phân phối, cấu trúc phân cấp**.
3. Chưa tích hợp mô hình sinh và ước lượng mật độ.
4. Giá trị thiếu: **thay bằng trung bình** (không có mặt nạ).
5. Giới hạn cứng: **10 lớp** (vượt qua bằng mã hóa cơ số), **100 đặc trưng** (vượt bằng PCA, có thể mất thông tin).
6. Suy luận có phụ thuộc truy hồi (cần chỉ mục FAISS), bất biến cột chỉ xấp xỉ.
7. Dữ liệu huấn luyện thật có nguy cơ **rò rỉ/chồng lấn với benchmark** (bài lấy từ OpenML, các bộ CC18 và CTR23 cũng thuộc OpenML; bài dùng bộ trong Grinsztajn, TabZilla, AMLB để huấn luyện, khác tập đánh giá, nhưng độc giả cần kiểm tra danh sách ở Phụ lục B để biết chắc không trùng).
8. Không đánh giá **nhãn nhiễu, mất cân bằng lớp, hiệu chuẩn** (chỉ AUC, accuracy, correlation, R²).

## 8. Liên hệ với đề tài
- Khác biệt gốc với TabPFN v2/TabICLv2: dữ liệu **thật** thay vì tổng hợp. TabICLv2 trích dẫn TabDPT ("quy luật scaling có thể vỡ nếu bộ sinh tổng hợp yếu") và Real-TabPFN (tiếp tục tiền huấn luyện trên dữ liệu thật); trên TabArena TabICLv2 vượt TabDPT.
- Kiến trúc theo dòng (như TabPFN v1) giải thích vì sao TabDPT có chi phí thấp nhưng không nhận biết cấu trúc cột nội bộ như TabICL.
- Hướng so sánh "truy hồi lúc suy luận" liên quan trực tiếp tới **LoCalPFN** (bài 10/TabPFN-kNN) và là một ứng viên cải tiến cho đề tài cuối kỳ.

## 9. Từ khóa cần nhớ
- **SSL che cột:** chọn một cột làm nhãn giả, xáo và bỏ cột còn lại, số tác vụ tăng tổ hợp.
- **Retrieval-based pretraining:** K dòng gần nhất làm ngữ cảnh khi cả huấn luyện lẫn suy luận (FAISS).
- **Scaling law:** α = 0,42 (tham số), β = 0,39 (ô dữ liệu).
- **Dữ liệu huấn luyện:** 123 bộ OpenML, 32M dòng, 2B ô.
- **Mã hóa cơ số C_max:** ⌈log_{10} C⌉ chữ số để xử lý hơn 10 lớp.
