# Bài 19. TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025; arXiv:2505.18125v2 [cs.LG], 29/10/2025 |
| Tác giả | Alan Arazi, Eilam Shapira, Roi Reichart (Technion – IIT) |
| Mã nguồn | https://github.com/alanarazi7/TabSTAR; công bố nhiều biến thể trọng số, mỗi biến thể loại một số tập khỏi tiền huấn luyện |
| Lưu ý về PDF | Các bảng chi phí (Bảng 3), kết quả theo từng tập (Bảng 19–22) và Bảng 4, 29 bị lệch cột khi trích chữ. Mình chỉ báo cáo số có nhãn rõ hoặc được văn bản xác nhận. Phần phụ lục danh sách 253 + 93 tập tiền huấn luyện chỉ tóm tắt. |
| Vị trí trong đề tài | TFM **chuyên cho bảng có cột văn bản**, tiền huấn luyện có giám sát trên 350 tập thật. Cùng nhóm ConTextTab (bài 16): phía "dữ liệu thật + ngữ nghĩa". |

## 1. Bài này làm gì trong một đoạn

Hầu hết TFM (TabPFNv2, TabICL, TabDPT) xử lý cột văn bản bằng một embedding có sẵn, cố định, không biết biến mục tiêu là gì; bảng "benchmark kinh điển" thì ít cột văn bản (một thống kê được dẫn: một nửa số tập trong các benchmark phổ biến đã hơn 20 năm tuổi). TabSTAR đề xuất hai ý chính. Thứ nhất, **mở khóa (fine-tune) bộ mã hóa văn bản** (e5-small-v2) trong lúc huấn luyện, để biểu diễn văn bản được tối ưu cho tác vụ dự đoán. Thứ hai, **target-aware tokens**: các giá trị của biến mục tiêu được "nói thành lời" như một đặc trưng rồi đưa vào đầu vào, nên mô hình **không cần tham số riêng cho từng tập** (đầu dự đoán chung cho mọi số lớp). Tiền huấn luyện trên 350 tập thật (253 phân loại, 97 hồi quy; bài ghi 350, bảng phụ lục ghi 253 + 93) dưới 48 giờ trên một GPU A40. Trên 50 tập có văn bản, TabSTAR **đạt SOTA ở phân loại** (cả ≤10K mẫu và không giới hạn kích thước), vượt TabPFN-v2, TabM và cây tăng cường tinh chỉnh 4 giờ; **ở hồi quy thì chưa SOTA**, GBDT vẫn thắng.

## 2. Bài toán và bối cảnh

- GBDT thống trị bảng nhờ thiên kiến quy nạp tốt (bền với phân phối lệch, tự chọn đặc trưng) và hiệu quả tính toán, nhưng **không tiền huấn luyện được**, và phải dùng embedding ngoài cho văn bản/ảnh (biểu diễn cố định, không tinh chỉnh theo tác vụ). Điều này gây hại khi dữ liệu ít (ví dụ y tế).
- Dữ liệu thực tế có **cột hạng mục cardinality cao** và **văn bản tự do** (ví dụ báo cáo bệnh án khi dự đoán "xuất viện hay nằm lại" ở Bảng 1 mẫu của bài). Embedding **không phụ thuộc mục tiêu** dùng chung cho mọi biến đích, trong khi cùng cột "Report" cần ý nghĩa khác nhau khi dự đoán quyết định xuất viện hay chi phí điều trị. Tên cột thường bị bỏ qua.
- **Các hướng liên quan:**
  - Học sâu trên một tập đơn (TabNet, TabTransformer, FT-Transformer, SAINT, NPT, TabR, TabM, RealMLP): chưa thắng GBDT một cách thuyết phục và không xử lý văn bản giàu.
  - Chuyển giao chéo tập (TransTab, XTab) và TFM (TabPFN, TabPFN-v2, TabICL, TabDPT): TabPFN-v2 tiền huấn luyện 130 triệu tập tổng hợp, ICL nhận tối đa 10.000 mẫu; không dùng văn bản linh hoạt hơn GBDT.
  - TFM ngữ nghĩa: CM2 (tiền huấn luyện trên hơn 2000 tập nhưng dùng word embedding tĩnh, không fine-tune), CARTE (đồ thị hình sao, tốt cho cardinality cao nhưng chưa thấy văn bản dài, chủ yếu hữu ích cho tập nhỏ), TP-BERTa (RoBERTa + token hóa độ lớn tương đối cho số; có lớp đầu ra riêng từng tập nên mất khả năng chia sẻ, phải huấn luyện hai mô hình riêng cho phân loại và hồi quy). **Không phương pháp nào fine-tune biểu diễn ngữ nghĩa lúc huấn luyện tác vụ.**
  - LLM cho bảng (TabLLM, TabuLa-8B): tốn tài nguyên, kém với số, có nguy cơ nhiễm benchmark do LLM đã thấy dữ liệu.
  - AutoML đa phương thức (AutoGluon-Multimodal): là hệ thống tổ hợp nhiều thủ thuật chứ không phải một mô hình nền.

## 3. Phương pháp

Huấn luyện hai giai đoạn: (1) **tiền huấn luyện đa tác vụ** trên nhiều tập (trộn phân loại và hồi quy), (2) **fine-tune bằng LoRA** cho một tác vụ cụ thể. Học có giám sát trên biến mục tiêu ở cả hai giai đoạn. Kiến trúc có 5 mô-đun:

1. **Verbalization (nói thành lời):** mỗi đặc trưng thành "Tên cột: giá trị"; cột số tạo **hai đầu vào**: (a) giá trị số chuẩn hóa z-score cắt ngoại lai ở ±3σ (thiếu thì 0); (b) chuỗi văn bản có **rời rạc hóa theo 10 phân vị**, ví dụ "Age: 40–50 (Quantile 50–60%)", để giảm mất chính xác khi mô hình ngôn ngữ xử lý số. Cột ngữ nghĩa chỉ nối tên và giá trị. Giá trị thiếu thành "Unknown Value"; văn bản quá 512 token cắt thẳng. Cột ngày được tách bằng DatetimeEncoder của skrub (ví dụ thứ trong tuần là ngữ nghĩa, giây Unix là số).
   - **Biến mục tiêu cũng được đưa vào đầu vào:** phân loại có C giá trị thì có C phần tử "Target. Decision: Hospitalized", "Target. Decision: Released", hằng số với mọi mẫu (nhãn thật bị ẩn). Hồi quy có đúng một phần tử chỉ mang tên biến đích.
2. **Encoding:** e5-small-v2 mã hóa từng phần tử độc lập (attention chỉ trong token của phần tử); **mở khóa nửa trên các lớp** (6 lớp, tối ưu cho tác vụ). Phần số đi qua MLP hai lớp (1 → 2d → d).
3. **Fusion:** một lớp Transformer encoder (2 đầu attention) chỉ cho hai vector (văn bản, số) của **cùng một phần tử** tương tác, lấy trung bình hai đầu ra.
4. **Interaction:** Transformer encoder **6 lớp**, d = 384, 6 đầu attention, pre-norm; token đặc trưng và token mục tiêu cùng attend nhau; **không dùng mã hóa vị trí** nên bất biến với thứ tự cột.
5. **Prediction:** hai đầu MLP (một lớp ẩn 4d); với phân loại, mỗi token mục tiêu đi qua **cùng một đầu** cho ra logit, softmax qua C giá trị: hỗ trợ **số lớp bất kỳ** và không cần tham số riêng cho tập. Hồi quy: một token mục tiêu → số thực (z-score đã cắt).
- **Tổng tham số:** khoảng 47,26 triệu, phần lớn ở bộ mã hóa văn bản. Mở khóa 6 lớp: khoảng 24,7 triệu tham số tinh chỉnh, phần còn lại (11,92M embedding + 10,65M lớp) đóng băng.

### Huấn luyện
- **Tiền huấn luyện:** mỗi tập subsample tối đa 300.000 mẫu (phân tầng theo nhãn) và 200 đặc trưng; chia train/val 95%/5%, val tối đa 1000 mẫu; mỗi epoch lấy tối đa 2048 mẫu từ mỗi tập, mini-batch 32 (mỗi batch chỉ từ một tập, nên không cần padding), tích lũy gradient 4 bước (batch toàn cục 128); loss cross-entropy (phân loại) và MSE (hồi quy); 50 epoch, OneCycleLR, khởi động 5 epoch, cosine, lr tối đa 5e-5, weight decay 0,001, dừng sớm sau 3 epoch; mixed-precision. Mô hình đầy đủ (390 tập) huấn luyện **dưới 48 giờ trên một GPU A40 48GB**.
- **Fine-tune:** LoRA rank 32, α = 64, dropout 0,1, lr 1e-3, chia 90/10 và lấy 10% làm validation; chỉnh **1.597.440 trên 47.263.874 tham số (3,4%)**. Siêu tham số chọn bằng grid search trên 25 tác vụ **tách rời** khỏi 50 tập đánh giá; hiệu năng khá bền trên không gian tìm kiếm. Bản "không tiền huấn luyện" phải fine-tune toàn bộ vì LoRA kém trên trọng số ngẫu nhiên.
- **Kho tiền huấn luyện:** 350 tập chọn tay (253 phân loại, 97 hồi quy) từ OpenML và Kaggle sau khi khử trùng lặp (tác giả phát hiện nhiều trùng lặp trong các kho phổ biến), tập trung dữ liệu giàu ngữ nghĩa.

## 4. Thiết lập thí nghiệm

- **Benchmark:** gộp 3 nguồn (AutoML Multimodal Benchmark, bài CARTE, phân tích đặc trưng văn bản/cardinality cao) sau khử trùng, được **50 tập** (36 hồi quy, 14+ phân loại). Hai hạn chế do tác giả nêu: lệch về hồi quy (36 tập), và 29/36 tập hồi quy đến từ CARTE (thiên về cardinality cao hơn là văn bản dài). Vẫn đánh giá toàn bộ 50 tập.
- **Chia dữ liệu:** 10 lần chia ngẫu nhiên train/test (90/10) trên mỗi tập → 500 lần chạy mỗi mô hình. 30 tập có hơn 10.000 mẫu.
- **Hai điều kiện:** **10K** (mọi mô hình huấn luyện tối đa 10.000 mẫu) và **Unlimited** (TabSTAR và các baseline mở rộng được chạy trên bản đầy đủ của 30 tập lớn; cắt kỹ thuật ở 100.000 mẫu).
- **Đối chứng:** GBDT (Random Forest, LightGBM, XGBoost, CatBoost), MLP (RealMLP, TabM), TFM (CARTE, TabDPT, TabPFN-v2 qua API đóng vì bản công khai không hỗ trợ văn bản, TabICL chỉ phân loại). Mô hình mặc định và bản tinh chỉnh bằng random search + CV 5-fold với ngân sách 4 giờ trên 8 lõi CPU; mô hình không hỗ trợ văn bản dùng e5-small-v2 đóng băng qua skrub TextEncoder (rút còn 30 chiều).
- **Thước đo:** AUROC (phân loại), R² (hồi quy), chuẩn hóa về [0,1] theo mỗi lần chia (mô hình tốt nhất = 1, tệ nhất = 0) theo cách TabPFN-v2, trung bình có khoảng tin cậy 95%.
- **Mô hình bị loại vì rò rỉ:** TP-BERTa và CM2 bị loại do tiền huấn luyện đã chứa các tập có trong benchmark (rò rỉ trực tiếp). TabDPT bị loại khỏi hồi quy vì hiệu năng thất thường trên vài tập; TabPFN-v2 không chạy được 4 tập (C12 vì hơn 10 lớp; C02, C14, R01 vì vượt giới hạn 500.000 ô); CARTE không chạy 15 tập do lỗi `PowerTransformation`.
- **Huấn luyện chống rò rỉ:** tác giả tạo **5 biến thể TabSTAR**, mỗi biến thể tiền huấn luyện trên 350 tập + 40 tập benchmark và để dành 10 tập chỉ để kiểm tra; mỗi tập được đánh giá bằng đúng biến thể chưa từng thấy nó.

## 5. Kết quả

### 5.1. Phân loại (Hình 2)
- Điều kiện **10K**: TabSTAR **0,83**, tiếp theo TabM-Tuned 0,73, CatBoost-Tuned 0,69, TabPFN-v2 0,67. Đối đầu trực tiếp: thắng TabPFN-v2 ở 8/11 tập, TabM-Tuned 10/14, CatBoost-Tuned 12/14.
- Điều kiện **Unlimited**: TabSTAR **0,84**, TabM-Tuned 0,79, còn lại thấp hơn nhiều. Biến thể không giới hạn tốt hơn hẳn bản 10K, nhấn mạnh giá trị của mô hình không bị giới hạn kích thước.

### 5.2. Hồi quy (Hình 3)
Ở điều kiện 10K, TabSTAR **cạnh tranh nhưng không phải SOTA**; TabPFN-v2 hơn nhưng vẫn thua GBDT rõ rệt (GBDT thống trị ở đây). Điều này cho thấy xử lý văn bản trong bảng là bài toán chưa giải, ngay cả với TabPFN-v2 vốn xuất sắc ở dữ liệu số. Ở điều kiện Unlimited, TabSTAR mở rộng tốt và vượt các TFM không mở rộng được, nhưng khoảng cách với GBDT vẫn lớn.

### 5.3. Chi phí (Bảng 3, 50 tập ≤10K mẫu)
- Suy luận của TabSTAR **cỡ GBDT chạy GPU** vì thời gian bị chi phối bởi nhúng văn bản; CatBoost chỉ-CPU chậm gần gấp mười.
- TabICL và TabDPT chậm hơn **vài bậc độ lớn** và tốn bộ nhớ nhiều hơn; đây là hạn chế mở rộng của TFM dựa ICL.
- Huấn luyện TabSTAR **chậm hơn nhiều** so với GBDT, nhưng ưu thế của GBDT nhỏ lại khi phải tinh chỉnh siêu tham số hoặc không có GPU cho embedding.
- Phần cứng đánh giá: một GPU A100 40GB và 8 lõi AMD EPYC 7742; TabPFN-v2 (API) và CARTE bị loại khỏi phân tích chi phí.

## 6. Phân tích (3 câu hỏi nghiên cứu, trên 256 tập tiền huấn luyện và 20 tập kiểm tra: 12 hồi quy, 8 phân loại)

- **Q1 – Mở khóa bộ mã hóa văn bản:** mở khóa **chỉ một lớp** đã vượt hẳn embedding tĩnh; càng mở nhiều càng tốt, **tốt nhất ở 6 lớp**; 9 lớp kém hơn (có thể đổi khác khi thêm dữ liệu). Biến thể đóng băng hoàn toàn "kém hơn mọi baseline". Ví dụ ở Bảng 28 (AUROC hoặc R², mở khóa 0/1/3/6/9 lớp): C01 87,8 → 90,4 → 90,8 → 91,0 → 90,8; C07 82,1 → 89,1 → 90,6 → 91,3 → 91,0; R13 37,6 → 45,7 → 51,6 → 51,4 → 52,1.
- **Q2 – Tiền huấn luyện và số tập:** mô hình tiền huấn luyện hơn hẳn mô hình huấn luyện từ đầu; tăng số tập tiền huấn luyện (0, 16, 64, 256) tăng đều cả phân loại và hồi quy. Điểm chuẩn hóa tăng từ khoảng 0,35 (không tiền huấn luyện) lên khoảng 0,79 và 0,81 (256 tập) ở hai nhóm tác vụ. Mức tăng rõ ở hồi quy gợi ý với nhiều dữ liệu hơn, hồi quy có thể đạt SOTA ("quy luật mở rộng theo số tập").
- **Q3 – Cách nói thành lời số:** ba kiểu: chỉ **Name**; **Name + Bin**; **TabSTAR (Name + Bin + Quantile)**. Điểm chuẩn hóa (phân loại / hồi quy): Name 0,386 / 0,386; Name+Bin 0,544 / 0,584; TabSTAR 0,593 / 0,596. Có thông tin số trong lời nói cải thiện đáng kể; thêm phân vị cho lợi ích nhỏ. (Mình ghép bảng theo thứ tự hàng/cột.) Không thử giá trị số chính xác vì làm tăng số chuỗi duy nhất và tốn bộ nhớ.
- **Các lựa chọn kiến trúc (phụ lục A.3, tiền huấn luyện thu nhỏ: 25 epoch, 128 tập):**
  - Fusion bằng attention tốt hơn nối (concatenation) và nhân (multiplication).
  - Số lớp Interaction: 3 lớp hơi kém 6 lớp; 9 lớp kém rõ rệt.
  - Thêm **row-level attention kiểu SAINT** không lợi ích đáng kể mà nhạy với batch size và tăng độ phức tạp suy luận.
  - Verbalization: có tên cột tốt hơn chỉ giá trị; **nêu rõ tất cả giá trị mục tiêu** tốt hơn không nêu; thêm mô tả tập dữ liệu làm tiền tố không có lợi; mô hình vẫn tốt khi **không có siêu dữ liệu ngữ nghĩa** (là điểm mạnh).
  - Tiền huấn luyện: batch nhỏ hơn (16, 8) cho kết quả tốt hơn nhưng chậm hơn; các cách gán trọng số loss theo tác vụ hầu như không đổi hiệu năng; learning rate khác nhau theo lớp (0,5 hoặc 0,75 cho bộ mã hóa) **làm hại**.

## 7. Hạn chế (tác giả nêu)

1. **Hồi quy chưa SOTA**; cần mở rộng dữ liệu/mô hình, regression-via-classification (như TabPFN-v2, TabForestPFN), thêm thống kê phân phối từng đặc trưng cho bộ mã hóa số (như TabICL).
2. **Tốc độ huấn luyện** chậm hơn GBDT mặc định và TFM ICL; có thể gặp **nghẽn bộ nhớ** với hàng trăm đặc trưng (kiến trúc bị cắt ở 200 đặc trưng khi tiền huấn luyện).
3. **Chưa đánh giá kỹ** ở few-shot và ở dữ liệu thuần số.
4. Tiền huấn luyện **khiêm tốn** (350 tập, bộ mã hóa nhỏ); nguy cơ rò rỉ khi dùng LLM và khi tập đánh giá trùng tập huấn luyện; benchmark lệch hồi quy, lệch cardinality cao.
5. Khuyến nghị cho cộng đồng: công bố nhiều biến thể mô hình mỗi biến thể loại một số tập, để mỗi tập luôn có mô hình chưa từng thấy nó.

Nhận xét thêm của mình:
- Đánh giá trên **50 tập** với 10 lần chia là khá nhiều, nhưng chỉ còn 14 tập phân loại ≤10K và 10 tập >10K; kết luận "SOTA phân loại" dựa trên số tập khá ít và còn phụ thuộc vào tập CARTE/AutoML Multimodal.
- TabPFN-v2 phải dùng **API đóng** với xử lý văn bản không rõ; việc so sánh này khó tái lập.
- Không có thí nghiệm nhiễu nhãn hay mất cân bằng; chỉ AUROC/R².

## 8. Liên hệ với bài Nature và đề tài

- **So với TabPFNv2 (bài 1):** TabPFN-v2 không tận dụng văn bản; TabSTAR cho thấy khi có văn bản thì mô hình chuyên biệt thắng rõ trên phân loại. Nhưng với bảng thuần số (phạm vi bài Nature) TabSTAR chưa được kiểm tra.
- **So với ConTextTab (bài 16):** cùng hướng ngữ nghĩa và dữ liệu thật; ConTextTab theo ICL (cần ngữ cảnh), TabSTAR theo fine-tune LoRA (không ngữ cảnh), nên suy luận nhanh hơn nhiều ở dữ liệu lớn nhưng cần bước huấn luyện.
- **Đề tài:** nằm ngoài 4 SOTA thầy yêu cầu (TabPFNv2, TabICLv2, TabDPT, LoCalPFN) và nếu dữ liệu thí nghiệm thuần số thì không áp dụng; giá trị chính là làm rõ giới hạn "TFM tổng hợp không dùng văn bản" và đặt vấn đề **rò rỉ dữ liệu** khi đánh giá TFM, một điểm cần lưu ý trong ma trận so sánh.

## 9. Từ khóa cần nhớ

- **Target-aware tokens:** giá trị mục tiêu thành token đầu vào; một đầu dự đoán chung, không tham số riêng cho từng tập, số lớp tùy ý.
- **Mở khóa bộ mã hóa e5-small-v2** (6 lớp là tốt nhất); fusion theo từng phần tử; Interaction 6 lớp d=384; 47,26M tham số; LoRA r=32 (3,4% tham số).
- **Số**: z-score cắt ±3 + 10 phân vị trong câu chữ; 350 tập tiền huấn luyện, dưới 48 giờ trên A40.
- **Kết quả:** phân loại 0,83 (10K) và 0,84 (không giới hạn); hồi quy chưa SOTA; chi phí suy luận ngang GBDT GPU.
- **Rò rỉ:** loại TP-BERTa và CM2; 5 biến thể loại 10 tập mỗi biến thể.
