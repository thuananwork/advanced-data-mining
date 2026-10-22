# Bài 9. Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025 (39th Conference on Neural Information Processing Systems); arXiv:2510.21204v1 [cs.LG], 24/10/2025; 39 trang kể cả phụ lục |
| Tác giả | Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, Bernie Wang (Amazon; Akoglu và Faloutsos kiêm CMU) |
| Mã nguồn / trọng số | Trọng số công khai trên HuggingFace: autogluon/mitra-classifier và autogluon/mitra-regressor |
| Lưu ý về PDF | Các bảng Table 1, 6, 9–11, 13–18 bị lệch cột khi trích chữ. Mình chỉ báo cáo con số nào đối chiếu được với văn bản hoặc có nhãn hàng rõ ràng, và ghi chú chỗ nào chỉ đọc được thứ tự. Các hình (quyết định biên 2D, critical difference) chỉ được tóm lược qua chú thích. |
| Vị trí trong đề tài | Bài **tổng hợp prior** cho TFM: trả lời câu hỏi "thiết kế dữ liệu tổng hợp thế nào là tốt". Là phần mở rộng/đối thủ của TabPFNv2 và TabICL trong cùng họ TFM. |

## 1. Bài này làm gì trong một đoạn

Hầu hết các công trình về mô hình nền cho dữ liệu bảng (TFM) tập trung vào kiến trúc, còn dữ liệu tổng hợp để tiền huấn luyện (prior) thì chọn theo kinh nghiệm: "đa dạng là tốt". Bài này cho rằng trọng tâm nên chuyển từ kiến trúc sang **thiết kế prior** và là bài đầu tiên nghiên cứu có hệ thống chuyện đó. Nhóm tác giả đề xuất ba tiêu chí để đánh giá một prior: **hiệu năng** (một TFM chỉ học từ prior đó chạy tốt đến đâu trên dữ liệu thật), **tính đa dạng** (mô hình khó quá khớp vào chính prior đó) và **tính khác biệt** (dữ liệu từ prior này khó bị mô hình học từ prior khác dự đoán). Từ đó họ dựng một hỗn hợp gồm mô hình nhân quả cấu trúc (SCM) và năm prior dựa trên cây (decision tree, extra tree, gradient boosting, random forest, và một biến thể "random forest lấy mẫu trực tiếp"), huấn luyện **Mitra** (kiến trúc 2D attention giống TabPFNv2) trên hỗn hợp đó. Mitra đứng đầu trên ba benchmark phân loại (TabRepo, TabZilla, AMLB) và benchmark hồi quy TabRepo, hiệu quả mẫu tốt hơn TabPFNv2 và TabICL, và các prior còn giúp cả kiến trúc 1D (TabPFN). Điểm cần nhớ: **ICL thuần của Mitra chỉ ngang TabPFNv2**; vị trí số một đạt được khi có thêm ensemble và fine-tuning.

## 2. Bối cảnh và động cơ

- Cây tăng cường (XGBoost, LightGBM, CatBoost) thống trị dữ liệu bảng nhưng mỗi tập phải huấn luyện lại, ít chuyển giao giữa các phân phối. TabPFN mở ra hướng TFM dựa trên **học trong ngữ cảnh (ICL)**: tiền huấn luyện trên dữ liệu tổng hợp sinh tức thời, rồi dự đoán bằng cách đưa mẫu huấn luyện vào làm ngữ cảnh.
- Các TFM ICL (TabPFN, TabPFNv2, TabICL, TabForestPFN, Attic, TabDPT) đều dùng prior cố định hoặc theo kinh nghiệm. TabForestPFN và Attic trộn SCM với cây quyết định (50/50), TabICL trộn SCM với XGBoost (70/30). Chưa ai giải thích **vì sao** thêm prior cây có lợi, hay nên trộn bao nhiêu và trộn những gì.
- Hai họ TFM khác được nhắc: nhóm dùng mô hình ngôn ngữ (TabLLM, GTL, Tabula, TP-BERTa, CARTE...) phụ thuộc siêu dữ liệu ngữ nghĩa và chi phí suy luận LLM; nhóm tiền huấn luyện chéo bảng trên dữ liệu thật (XTab, v.v.).

## 3. Phương pháp

### 3.1. Khung ICL và hàm mục tiêu
Một bộ sinh dữ liệu (prior) G_i nhận siêu tham số lấy ngẫu nhiên (số đặc trưng, số mẫu, số lớp, số đặc trưng hạng mục) và trả về bảng D⁽ⁱ⁾. Mô hình f_θ nhận tập hỗ trợ (s dòng có nhãn) và q dòng truy vấn, dự đoán nhãn truy vấn trong **một lượt chạy xuôi** mà không cập nhật tham số. Hàm mục tiêu là cực đại log-likelihood của nhãn truy vấn bị che: L = E_D[log p_θ(y_qry | x_sup, y_sup, x_qry)]. Với phân loại thì tương đương cross-entropy (phân phối categorical); với hồi quy tương đương MSE (phân phối Gaussian). Giả định các dòng truy vấn độc lập có điều kiện cho trước ngữ cảnh. Mỗi bước huấn luyện lấy generator theo trọng số w_i của hỗn hợp, sinh bảng, chia ngẫu nhiên thành hỗ trợ và truy vấn (Thuật toán 1).

### 3.2. Các prior được dùng
- **SCM** (như TabPFN): dựng DAG ngẫu nhiên rồi sinh đặc trưng và nhãn theo thứ tự phụ thuộc; mô hình hóa cả phụ thuộc giữa các cột lẫn p(y|x).
- **Prior dựa trên cây (TBP)**: tập trung vào p(y|x) với ranh giới theo ngưỡng. Gồm:
  - bốn prior **lấy mẫu gián tiếp**: DT (decision tree), ET (extra tree), GB (gradient boosting), RF (random forest). Quy trình: sinh một tập huấn luyện tổng hợp cỡ B, **fit** bộ phân loại/hồi quy của scikit-learn lên đó, rồi lấy đặc trưng mới X₂ từ phân phối đơn giản (chuẩn đa biến) và nhãn là đầu ra dự đoán của mô hình đã fit (Thuật toán 2). Khác với TabForestPFN/Attic ở chỗ fit thẳng bộ **phân loại** và sinh nhãn bằng thủ tục đa thức trực tiếp, nên bỏ được bước quantile transform để rời rạc hóa.
  - một prior **lấy mẫu trực tiếp** tự đề xuất, **DSRF** (directly sampled random forest): không cần fit; dựng ngẫu nhiên cây bằng cách chọn chỉ số tách và ngưỡng tách ngẫu nhiên (Thuật toán 4, có độ sâu tối thiểu/tối đa và xác suất nút lá), sinh nhãn lá theo vòng modulo số lớp (phân loại) hoặc Gaussian (hồi quy), rồi bỏ phiếu đa số hoặc lấy trung bình qua N_e cây (Thuật toán 3, 5).
- **Sinh đặc trưng:** N_cont đặc trưng liên tục (nhiễu Gaussian i.i.d.) và N_cat = ⌊p_cat(d+1)⌋ đặc trưng hạng mục; số lớp của đặc trưng hạng mục lấy từ phân phối hình học, giá trị theo phân phối đa thức.

### 3.3. Ba tiêu chí chọn prior
Với M prior ứng viên, huấn luyện M mô hình (mỗi mô hình chỉ trên một prior), rồi lập hai đối tượng:
- **Ma trận khả năng tổng quát hóa G ∈ R^{M×M}:** G_ij là chỉ số (AUC) của mô hình huấn luyện trên G_i và kiểm tra trên 100 bảng sinh từ G_j (mỗi bảng 1000 mẫu: 800 hỗ trợ, 200 truy vấn).
- **Vector hiệu năng P ∈ R^M:** P_i là chỉ số của mô hình huấn luyện trên G_i và đánh giá trên **dữ liệu thật** (TabRepo).

Ba tiêu chí:
1. **Hiệu năng:** P_i cao.
2. **Đa dạng:** giá trị trên **đường chéo** G_ii thấp, tức khó quá khớp vào chính prior của nó (một prior "dễ" dự đoán thì ít thông tin).
3. **Khác biệt:** giá trị **ngoài đường chéo** G_ij thấp, tức mô hình học từ prior khác dự đoán dữ liệu của G_j kém. Cụ thể, với hỗn hợp hiện có G', độ khác biệt của G_j là max_{i∈G'} G_ij; chọn thêm prior có giá trị max này nhỏ nhất (min_j max_i G_ij) để mở rộng độ phủ.

Ví dụ trong Bảng 1 (AUC): đường chéo của DSRF là 0,984, của ET là 0,871; mô hình học từ SCM dự đoán dữ liệu DSRF được 0,960 nhưng dữ liệu ET chỉ 0,751. Vì vậy ET có "độ khác biệt" cao hơn dù DSRF là prior đứng hạng hai về hiệu năng độc lập. Với RF, đường chéo thấp nhất (0,761), ngoài đường chéo trong khoảng [0,708; 0,768], nhưng hiệu năng thật là **tệ nhất** trong các prior.

### 3.4. Trộn prior, mô hình và huấn luyện
- Hỗn hợp cuối: **SCM + ET + GB + DT + RF + DSRF**, mặc định **50% SCM, 50% TBP** chia đều giữa các prior cây. Hỗ trợ cả phân loại và hồi quy.
- **Mitra (2D):** Transformer 12 lớp, embedding 512, 4 đầu attention, mỗi lớp có attention theo hàng và theo cột (FlashAttention); **72 triệu tham số**. **Mitra 1D:** attention theo hàng, 37 triệu tham số. Chuẩn hóa đặc trưng bằng uniform quantile transform theo tập hỗ trợ rồi chuẩn hóa trung bình/độ lệch chuẩn; với hồi quy còn min-max cho cột đích.
- **Giới hạn tiền huấn luyện:** tối đa **16 đặc trưng** và **640 dòng** (bằng khoảng 1/10 số đặc trưng và 1/3 số dòng tối đa của TabPFNv2).
- Huấn luyện trên **45 triệu bảng tổng hợp**, 8 GPU A100 40GB, khoảng **60 giờ**. Đường cong huấn luyện hội tụ quanh 2000 bước và dao động.

## 4. Thiết lập thí nghiệm

- **Phân loại:** ba benchmark 10-fold, gộp lại còn **137 bộ không trùng**: TabRepo (66 bộ), TabZilla (75 bộ), AMLB (29 bộ). TabRepo và TabZilla dùng bộ nhỏ (tối đa 3000 dòng, 100 đặc trưng, theo TabPFN); AMLB lớn hơn (tối đa 10.000 dòng, 500 đặc trưng, theo TabPFNv2). TabArena được đánh giá thêm ở phụ lục.
- **Hồi quy:** TabRepo (10 bộ, tối đa 3000 dòng/100 đặc trưng); phụ lục thêm AMLB + OpenML-CTR23 (28 bộ, tối đa 10.000 dòng/500 đặc trưng).
- **Đối chứng:** TFM (TabPFNv2, TabICL, Attic, TabPFN, TabForestPFN) và mô hình cổ điển/nơ-ron cần tinh chỉnh theo tập (RealMLP, LightGBM, XGBoost, CatBoost, MLP, AutoGluon); các mô hình cổ điển ở dạng bagging của AutoGluon 1.3 với siêu tham số mặc định.
- **Thước đo:** phân loại AUC-ROC (đa lớp dùng một-chống-một OvO), accuracy, cross-entropy; hồi quy R², RMSE, MAE; cộng thêm các thước đo theo hạng (vì thước đo gộp bị ảnh hưởng bởi vài tập cực đoan): hạng trung bình, **Elo**, winrate, rescaled accuracy (RAcc), champion delta (C∆); đơn vị đánh giá là từng fold.
- **Ba chế độ cho TFM:** ICL thuần; **+e** (ensemble: xáo đặc trưng, xáo thứ tự lớp, biến đổi đặc trưng ngẫu nhiên; số estimator mặc định: TabPFNv2 4 cho phân loại và 8 cho hồi quy, TabICL 32, TabPFN 3, trên TabZilla và AMLB chỉnh TabPFNv2 lên 8 vì tốt hơn); **+f** (fine-tune 50 epoch trên tập huấn luyện, thường kích hoạt dừng sớm).

## 5. Kết quả

### 5.1. Phân loại (Bảng 2, 137 bộ gộp)

| Mô hình | Hạng TB | Elo | Winrate | AUC | ACC | CE |
|---|---|---|---|---|---|---|
| **Mitra (+ef)** | **7,2** | **1136** | 0,69 | **0,905** | **0,858** | **0,328** |
| Attic (+ef) | 7,4 | 1128 | 0,68 | 0,903 | 0,857 | 0,332 |
| TabPFNv2 (+e) | 8,0 | 1107 | 0,65 | 0,901 | 0,856 | 0,338 |
| TabPFNv2 (+ef) | 8,6 | 1085 | 0,62 | 0,897 | 0,846 | 0,363 |
| TabICL (+e) | 9,5 | 1053 | 0,58 | 0,889 | 0,836 | 0,367 |
| Mitra (+e) | 9,7 | 1046 | 0,57 | 0,896 | 0,847 | 0,360 |
| TabPFNv2 (ICL) | 9,8 | 1043 | 0,56 | 0,891 | 0,846 | 0,352 |
| TabICL (ICL) | 10,6 | 1015 | 0,52 | 0,884 | 0,832 | 0,374 |
| Mitra (ICL) | 10,6 | 1015 | 0,52 | 0,891 | 0,841 | 0,368 |
| CatBoost | 11,3 | 988 | 0,48 | 0,888 | 0,837 | 0,375 |
| RealMLP | 12,2 | 958 | 0,44 | 0,878 | 0,827 | 0,412 |
| XGBoost | 13,1 | 926 | 0,40 | 0,883 | 0,833 | 0,388 |
| LightGBM | 13,4 | 917 | 0,38 | 0,876 | 0,829 | 0,393 |
| Random Forest | 13,7 | 903 | 0,36 | 0,874 | 0,822 | 0,471 |

(Chỉ liệt kê các hàng chính; mô hình MLP, Mitra 1D, TabForestPFN và các biến thể khác nằm ở phần cuối bảng gốc. Bảng gốc còn có RAcc và C∆: Mitra (+ef) 0,82 và 20,1.)

- Mitra thắng trong kết quả gộp và **trong cả ba benchmark riêng** (TabRepo, TabZilla, AMLB, Bảng 13–15), khi có ensemble và fine-tune.
- **ICL thuần của Mitra ngang TabPFNv2** (hạng 10,6 so với 9,8; Elo 1015 so với 1043; AUC 0,891 hai bên), dù tiền huấn luyện chỉ với tối đa 16 đặc trưng. Tác giả diễn giải đây là bằng chứng prior tổng quát hóa tốt.
- Khoảng cách giữa mô hình tốt nhất và cây tăng cường khá rõ: Mitra (+ef) hơn CatBoost khoảng 148 điểm Elo ở bảng này (1136 so với 988); phụ lục C.7 nói hơn 138 điểm Elo trên TabRepo.

### 5.2. Hồi quy TabRepo (Bảng 4, 10 bộ)
Mitra (+ef): hạng 4,3, Elo 1140 (±20), winrate 0,70, R² 0,636. Tiếp theo TabPFNv2 (+e) hạng 5,1, Elo 1090, R² 0,615; RealMLP và CatBoost cùng hạng 5,8, Elo 1044, R² lần lượt 0,627 và 0,629 (R² trung bình của CatBoost cao hơn TabPFNv2 (+e), nhưng thứ hạng thấp hơn). Hạng cuối: Random Forest 9,4 (Elo 804). TabPFN và TabICL không hỗ trợ hồi quy; Attic bị bất ổn khi huấn luyện hồi quy và kém XGBoost nên không đưa vào.

### 5.3. Prior có không phụ thuộc kiến trúc không? (Bảng 3, mô hình 1D)
Cùng hỗn hợp prior trên kiến trúc attention-theo-hàng: Mitra 1D (+f) hạng 3,0 / Elo 1057 / AUC 0,886; TabForestPFN (+f) 3,2 / 1038 / 0,878; TabPFN (+e) 3,4 / 1012; **Mitra 1D (ICL) 3,7 / 972 so với TabPFN (ICL) 3,8 / 970 và TabForestPFN (ICL) 3,9 / 951**. Tức lợi thế của prior ở chế độ ICL thuần rất nhỏ (2 điểm Elo so với TabPFN); lợi thế rõ khi có fine-tune.

### 5.4. Hiệu quả mẫu (Bảng 5 và 11)
Giảm số ví dụ ICL của TabRepo xuống 75%, 50%, 25%, 10% (ds). Mitra luôn đứng đầu trong ba mô hình (có ensemble/fine-tune): ví dụ ds=1,0 hạng 4,2 / Elo 1234; TabPFNv2 4,5 / 1217; TabICL 5,6 / 1144. Ở ds=0,1: Mitra 12,1 / Elo 740 / AUC 0,808; TabPFNv2 12,4 / 719 / 0,801; TabICL 12,7 / 689 / 0,777. Khoảng cách với TabPFNv2 khá nhỏ (hàng chục điểm Elo), với TabICL lớn hơn. So với chính các biến thể ít prior hơn (Mitra-Mix2 = SCM+ET+GB, và Attic ≈ SCM+DT), chênh lệch Elo **tăng dần khi dữ liệu càng ít**, lớn nhất ở ds=0,1. Đây là lập luận của tác giả rằng đa dạng prior giúp khi ít dữ liệu.

### 5.5. Ensemble nâng cao (mục 4.5, Hình 3)
**Mitra (bagging):** fine-tune riêng một bản Mitra cho mỗi fold trong CV phân tầng 8-fold rồi lấy trung bình đều (theo Krogh & Vedelsby). Tác giả nói đây là lần đầu dùng cross-validation ensemble cho TFM được fine-tune. So với TabPFNv2 PHE (Post-Hoc Ensemble) và AutoGluon 1.3 best-quality ở các ngân sách 300, 600, 900, 3600 giây mỗi bộ trên TabRepo 1-fold, Mitra (bagging) đều tốt hơn trên các thước đo; chi tiết ở Bảng 16.

### 5.6. Fine-tune (mục 4.6, Hình 4)
Tăng số estimator 1–2–4–8: Mitra fine-tune vượt TabPFNv2, TabICL, Attic ở mọi cỡ ensemble; fine-tune và ensemble của TabPFNv2 "hầu như không cải thiện thêm" so với ensemble đơn (tác giả đã xác minh quy trình fine-tune TabPFNv2 qua trao đổi với nhóm tác giả của nó). Giả thuyết: Mitra tiền huấn luyện chỉ với 16 đặc trưng nên khi thích nghi sang bảng nhiều đặc trưng thì còn nhiều dư địa; và prior đa dạng giúp mô hình thích nghi tốt hơn.

### 5.7. Ablation prior (mục 4.7, Bảng 6, 12)
- **Từng prior riêng:** SCM tốt nhất (Elo 1000, hạng 11,1); RF kém nhất (Elo 815, hạng 15,9); DSRF đứng thứ hai về hiệu năng riêng (Elo 899). Kết quả này khớp với vector hiệu năng P.
- **Thêm lần lượt (forward):** thứ tự {SCM, ET, GB, DT, RF, DSRF}. **SCM + ET tăng Elo thêm 63** so với SCM đơn (≈ 1000 → 1063); thêm GB tiếp tục cải thiện; DT, DSRF, RF cho nhiều cấu hình tương đương. Hỗn hợp đầy đủ (Mitra) Elo 1062 (hạng 9,3, winrate 0,59, RAcc 0,77, AUC 0,868). Bảng 6 gốc cho thấy cấu hình 4–5 prior xấp xỉ bằng hoặc nhỉnh hơn hỗn hợp đầy đủ trên vài chỉ số; tác giả vẫn giữ đủ prior để "đại diện đầy đủ họ prior cây" và vì hiệu quả mẫu tốt hơn (Bảng 11).
- **Giải thích qua G và P:** SCM có đóng góp lớn nhất vì hiệu năng cao và đường chéo thấp. DSRF đóng góp ít nhất khi thêm vào dù đứng hạng hai riêng lẻ: đường chéo 0,984 và ngoài đường chéo > 0,96 (chồng lấn mạnh với các prior khác). RF khác biệt tốt nhưng hiệu năng thật tệ.
- **Tỉ lệ SCM/TBP (Bảng 12):** p = 0,5, 0,4, 0,6 cùng hạng 3,2; p=0,5 Elo 1040 so với ~1030; **SCM thuần (p=1) hạng 3,7 tốt hơn TBP thuần (p=0) hạng 4,4**. Hỗn hợp chỉ-cây thiếu "khác biệt": các mô hình học từ một prior cây dự đoán prior cây khác tốt hơn so với mô hình SCM, và bỏ SCM là bỏ prior tốt nhất.

### 5.8. Quy luật mở rộng (mục 4.8, Hình 5)
Thử độ sâu 4, 8, 12, 16, 20, 24 lớp, mỗi cấu hình dùng 45 triệu mẫu: mô hình lớn hơn tốt hơn từ sớm và hội tụ cao hơn nhưng **bão hòa sau khoảng 12 lớp**. Theo lượng dữ liệu: hiệu năng tăng nhanh rồi phẳng, lợi ích giảm sau khoảng **18.000 bước** (mỗi bước 2048 bảng mới), tức bão hòa sau khoảng **37 triệu bảng**.

### 5.9. Kết quả bổ sung ở phụ lục
- **TabArena (Bảng 17):** Mitra là **mô hình đơn mạnh nhất** (single), vượt mọi phương pháp kể cả đã tinh chỉnh siêu tham số 200 vòng; chỉ bị vượt khi **ensemble các cấu hình HPO của TabPFNv2** (chậm hơn nhiều). Mitra nằm trên biên Pareto về thời gian huấn luyện và suy luận. Chưa dựng không gian tìm kiếm HPO cho Mitra (để dành tương lai). (Các Elo cụ thể của bảng này bị lệch cột nên không ghi lại.)
- **Thời gian (C.7):** Mitra (+ef) hơn CatBoost 138 Elo và nhanh hơn khoảng 3,5 lần trên 8 GPU A100; trên **một** GPU Mitra (+ef) tốn khoảng 89 giây, gần bằng CatBoost (83 giây).
- **Hồi quy lớn (Bảng 18, AMLB + OpenML-CTR23):** Mitra (+ef) và TabPFNv2 (+ef) ngang nhau, **TabPFNv2 (+e) tốt nhất**. Tác giả giải thích: lúc tiền huấn luyện Mitra chỉ thấy tối đa 16 đặc trưng và 640 dòng; vẫn vượt TabPFNv2 ở bộ tới 100 đặc trưng/3000 dòng và một số bộ tới 500/10k.
- **Hình quyết định biên 2D (C.8):** 1000 mẫu mỗi tập (10% hỗ trợ, 90% truy vấn), nhiều dạng (checkerboard, Gaussian process, Julia, spiral, star, Swiss roll...). Mitra cho biên đều và ít vỡ vụn hơn TabPFNv2; ví dụ checkerboard: Mitra acc 83,44%, ROC 91,57% so với random forest 69,22%/77,31% và TabPFN (v1) 50,22%/51,67%.

## 6. Hạn chế

Tác giả nêu (Phụ lục D):
- Hỗn hợp có thể cải thiện bằng cách **tối ưu siêu tham số trọng số hỗn hợp** theo từng miền/tác vụ; chưa làm.
- Dự kiến thêm prior liên tục như **Gaussian Process** để sinh biên trơn, hướng tới tác vụ ngoài dữ liệu bảng (chuỗi thời gian).
- Không luôn vượt TabPFNv2 trên **hồi quy nhiều đặc trưng**; cần mở rộng tiền huấn luyện lên nhiều dòng/đặc trưng hơn.

Nhận xét thêm của mình (từ chính số liệu):
1. **ICL thuần chỉ ngang TabPFNv2**; vị trí số một phụ thuộc ensemble và fine-tune, vốn tốn thêm tính toán. Biên độ ở chế độ ICL gần như bằng không (Bảng 3, hai mô hình 1D chênh 2 điểm Elo).
2. **Khoảng cách với đối thủ gần** (Attic +ef: 1128 so với 1136) nhỏ so với độ không chắc chắn nếu tính cả biến động theo fold, trong khi bài chỉ báo khoảng tin cậy Elo ±4.
3. Bộ benchmark chủ yếu là **bảng nhỏ** (≤3000 dòng hoặc ≤10.000 dòng), phù hợp "dữ liệu nhỏ" nhưng chưa phản ánh dữ liệu lớn; giới hạn tiền huấn luyện 16 đặc trưng/640 dòng.
4. Thang đo là AUC/ACC/CE trên dữ liệu **sạch và cân bằng vừa phải**; không có thí nghiệm nhãn nhiễu, mất cân bằng nặng hay hiệu chuẩn xác suất. Bài đề cao đa dạng prior nhưng không kiểm chứng độ bền với nhiễu.
5. Ma trận G được tính trên **dữ liệu tổng hợp** (100 bảng × 1000 mẫu); các giá trị G và P chỉ tin được khi dữ liệu thật đại diện, và kết luận "DSRF thừa" chỉ áp dụng trong khung TabRepo.
6. Mitra do nhóm Amazon/AutoGluon tạo ra, benchmark chính (TabRepo) cũng của nhóm này; cần kiểm chứng độc lập (TabArena trong phụ lục có bên thứ ba).

## 7. Liên hệ với bài Nature và đề tài

- **Quan hệ với TabPFNv2 (bài 1):** cùng kiến trúc 2D attention và cùng họ prior SCM; Mitra tấn công đúng phần mà TabPFNv2 ít giải thích, đó là prior. Câu chuyện "TFM chỉ cần dữ liệu tổng hợp" của bài Nature được mở rộng thành "thiết kế dữ liệu tổng hợp theo nguyên lý".
- **Quan hệ với TabICL / TabICLv2 (bài 4, 7):** TabICL dùng SCM 70% + XGBoost 30%; Mitra chỉ ra lựa chọn đó có thể chưa tối ưu và gợi ý ET + GB + DT + RF. TabICLv2 (bài 7) có thể thử hỗn hợp này như một hướng cải tiến prior trong phần thí nghiệm sau.
- **Ứng viên SOTA cho ma trận:** Mitra có trọng số công khai nên chạy lại được; nhưng thầy yêu cầu 3–4 mô hình chính (TabPFNv2, TabICLv2, TabDPT, LoCalPFN), nên Mitra là mô hình tham chiếu thêm.
- **Khoảng trống nghiên cứu:** Mitra không có thí nghiệm nhãn nhiễu/mất cân bằng/hiệu chuẩn. Một câu hỏi tự nhiên: prior nhiều cây (có ranh giới quyết định "vuông") có làm Mitra **nhạy hơn hay bền hơn** với nhãn nhiễu so với TabPFNv2 thuần SCM? Đây là điểm có thể kiểm chứng bằng các mức nhiễu có kiểm soát.

## 8. Từ khóa cần nhớ

- **Ba tiêu chí prior:** hiệu năng (vector P), đa dạng (đường chéo G_ii thấp), khác biệt (ngoài đường chéo G_ij thấp).
- **Hỗn hợp:** SCM 50% + {ET, GB, DT, RF, DSRF} chia đều; prior cây **gián tiếp** (fit sklearn rồi lấy nhãn dự đoán) và **trực tiếp** (DSRF).
- **Mitra:** 2D attention, 12 lớp, 512, 72M tham số; tiền huấn luyện ≤16 đặc trưng, ≤640 dòng, 45 triệu bảng, ~60 giờ A100×8.
- **Kết quả:** 137 bộ phân loại, Elo 1136 (+ef); ICL thuần ≈ TabPFNv2; SCM+ET tăng 63 Elo; bão hòa sau ~12 lớp và ~37 triệu bảng; mô hình đơn mạnh nhất TabArena.
- **Hạn chế:** hồi quy nhiều đặc trưng chưa hơn TabPFNv2; chưa có nhiễu/mất cân bằng/hiệu chuẩn.
