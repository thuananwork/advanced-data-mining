# Bài 10. Retrieval & Fine-Tuning for In-Context Tabular Models (LoCalPFN)

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2024 (main) theo bảng tổng hợp đề tài; PDF là arXiv 2406.05207 (v1, 7/6/2024) |
| Tác giả | Valentin Thomas*, Junwei Ma*, Rasa Hosseinzadeh, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, Anthony Caterini (Layer 6 AI, Toronto) |
| Mã nguồn / dữ liệu | Dựa trên TabPFN v1; thư viện faiss cho kNN; 95 bộ dữ liệu TabZilla chọn lọc |
| Vị trí trong đề tài | Một trong 4 SOTA cần so sánh (TabPFNv2, TabICLv2, TabDPT, **LoCalPFN**). Đại diện cho hướng **địa phương hóa ngữ cảnh + tinh chỉnh theo bài toán**. |

## 1. Bài này làm gì trong một đoạn

Mô hình học trong ngữ cảnh (ICL) như TabPFN v1 gặp hai giới hạn: không mở rộng theo **kích thước dữ liệu** (bộ nhớ bậc hai theo độ dài ngữ cảnh, nên phải lấy mẫu con khi bảng lớn) và **thiếu khớp (underfit)** khi bảng phức tạp, ngay cả khi toàn bộ bảng đã nằm trong ngữ cảnh. Bài đề xuất kết hợp **truy hồi (retrieval)** và **tinh chỉnh (fine-tuning)**: với mỗi mẫu truy vấn, lấy **k láng giềng gần nhất** từ tập huấn luyện làm ngữ cảnh riêng (**TabPFN-kNN**, không cần huấn luyện), rồi **tinh chỉnh toàn bộ mạng end-to-end cho từng bài toán** với ngữ cảnh địa phương đó (**LoCalPFN**, "Locally-Calibrated PFN"). Trên 95 bộ dữ liệu của TabZilla, LoCalPFN đạt SOTA mới, vượt cả cây tăng cường đã tinh chỉnh (XGBoost, CatBoost, LightGBM, Random Forest).

## 2. Vấn đề và động cơ

- Dữ liệu bảng phổ biến nhất trong thực tế nhưng đa dạng và không đồng nhất, nên mạng sâu thường cần huấn luyện và tinh chỉnh siêu tham số tốn kém; cây (XGBoost, CatBoost) bền hơn và vẫn thống trị.
- TabPFN (được huấn luyện trên hàng triệu quá trình sinh dữ liệu tổng hợp) xử lý bộ dữ liệu mới trong một lượt chạy xuôi, nhưng:
  1. **Bộ nhớ bậc hai theo cỡ ngữ cảnh**, nên khi N lớn không nhồi hết dữ liệu vào ngữ cảnh. Lấy mẫu con ngẫu nhiên không mở rộng được (Ma et al. đã chỉ ra hiệu năng giảm khi bộ dữ liệu lớn).
  2. **Underfit khi độ phức tạp tăng**: Hình 1 cho thấy TabPFN dù dùng cả N = 1.000 mẫu làm ngữ cảnh vẫn không phân loại được ba cặp vòng tròn đồng tâm của hai lớp (và hiệu năng AUC giảm dần khi tăng số cặp vòng tròn).
- Cách chọn ngữ cảnh trước đây cho TabPFN: tóm tắt tập dữ liệu bằng tâm cụm k-means hoặc tối ưu trực tiếp ngữ cảnh (TuneTables, In-Context Distillation của Ma et al.). Hạn chế chung: **một ngữ cảnh duy nhất dùng cho mọi điểm truy vấn**, thiếu linh hoạt.
- Các nghiên cứu trên LLM (RAG, truy hồi kết hợp tinh chỉnh) gợi ý hướng đi.

## 3. Ý tưởng cốt lõi: ngữ cảnh địa phương (mục 2.2–2.3)

**Giả thuyết:** thông tin quan trọng nhất để phân loại điểm truy vấn x_qy nằm trong lân cận của nó. Bằng chứng gián tiếp: (McElfresh 2023) một bộ phân loại kNN đơn giản cạnh tranh được với các mạng sâu hiện đại như TabNet và VIME.

Thay công thức TabPFN p_θ(y | x_qy, D_train) = softmax(f_θ(x_qy, D_train))[y] (xác suất dùng toàn bộ tập huấn luyện làm ngữ cảnh) bằng p_θ(y | x_qy, D_train) = softmax(f_θ(x_qy, kNN(x_qy)))[y], với kNN(x_qy) là k láng giềng gần nhất của truy vấn. Có một **liên tục** từ TabPFN dùng toàn bộ ngữ cảnh (k = N) tới TabPFN-kNN (k nhỏ).

**Lợi ích 1: biểu đạt tốt hơn.** Như kNN cổ điển, k điều khiển đánh đổi bias/phương sai: k lớn thì quá trơn (bias cao, underfit), k nhỏ cho biên quyết định phức tạp nhưng phương sai cao. Hình 1: tăng độ phức tạp (số cặp vòng tròn đồng tâm), lấy k ∈ {10, 30, 100, 300} láng giềng cho cải thiện AUC lớn so với TabPFN (k = 1.000 ≡ TabPFN gốc), mỗi điểm lấy trung bình 25 hạt giống. Điều đáng ngạc nhiên là mô hình 25 triệu tham số không tự học được vài vòng tròn chỉ bằng ICL; ngữ cảnh địa phương giúp khớp mẫu phức tạp giống như hồi quy tuyến tính cục bộ tạo ra biên phi tuyến.

**Lợi ích 2: mở rộng theo kích thước dữ liệu.** TabPFN phải lấy mẫu con nên không hưởng lợi từ N lớn. TabPFN-kNN tìm kiếm trên toàn bộ tập huấn luyện nên vẫn cải thiện khi N tăng, kể cả khi k ≪ N (Hình 2 trên adult-census, electricity, eeg-eye-state; mỗi điểm là trung bình 100 lần lấy lại mẫu). Khi k = N hai mô hình giống hệt nhau.

## 4. LoCalPFN: tinh chỉnh với truy hồi (mục 2.4)

- Ngoài truy hồi, tinh chỉnh toàn bộ mạng **cho từng bộ dữ liệu**, theo kiểu RAG. Tinh chỉnh ngây thơ không hiệu quả: TabPFN dùng một ngữ cảnh chung cho mọi truy vấn (batch B = 1, độ dài ngữ cảnh L_ctx, độ dài truy vấn L_qy = N_qy), mọi điểm ngữ cảnh chú ý lẫn nhau còn truy vấn chỉ chú ý ngữ cảnh nên tính song song được. Với láng giềng chính xác, ngữ cảnh không còn dùng chung, nên phải dùng chiều batch cho truy vấn (B = N_qy, L_ctx = k, L_qy = 1), kém hiệu quả nhiều về bộ nhớ lẫn song song. Nút thắt lúc này là lượt **xuôi và ngược**, không phải truy hồi, nên các kNN xấp xỉ thông thường không giúp.
- **Ngữ cảnh địa phương xấp xỉ (Hình 3b):** để truyền gradient cho N_qy ví dụ cùng lúc, chỉ dùng **B ngữ cảnh khác nhau**, mỗi ngữ cảnh phân loại N_qy/B điểm: (1) lấy mẫu B điểm huấn luyện; (2) với mỗi điểm tìm kNN với k = L_ctx + L_qy (bỏ chính điểm gốc để tránh trùng); (3) xáo và chia k mẫu thành vector ngữ cảnh dài L_ctx và vector truy vấn dài L_qy. B nhỏ ⇒ chia sẻ ngữ cảnh nhiều ⇒ xấp xỉ kNN (vì "láng giềng của láng giềng" không nhất thiết là láng giềng), nhưng mỗi chuỗi vẫn chỉ gồm các ví dụ trong vùng lân cận. Thực nghiệm: không giảm hiệu năng đáng kể mà huấn luyện nhanh hơn nhiều (Phụ lục A.5.5, Bảng 12 so sánh với phiên bản dùng láng giềng chính xác).
- **Chi tiết huấn luyện (Phụ lục A.2.2):** số láng giềng = min(10·√N_train, 1.000) (cố định, thích nghi theo cỡ bộ dữ liệu); batch suy luận 512 dùng faiss; AdamW tốc độ học 0,01, weight decay 0,01, không warmup và không scheduler; **1.000 điểm truy vấn lấy từ tập huấn luyện, batch 2**; dùng **one-hot** cho truy hồi và suy luận ở kết quả báo cáo; đánh giá mỗi 30 bước gradient và **dừng sớm theo AUC trên validation của từng fold**. Chạy được trên một GPU NVIDIA RTX 6000 Ada. Khi chưa tinh chỉnh (bước 0) LoCalPFN ≡ TabPFN-kNN.

## 5. Thiết lập thí nghiệm (mục 4.1)

- **Dữ liệu:** 95 trong 176 bộ của **TabZilla** (từ OpenML), lọc theo ràng buộc của TabPFN: tối đa **100 đặc trưng**, tối đa **10 lớp**, **không NaN**, mỗi lớp có ít nhất một mẫu trong mỗi phần chia. Chia thành **47 bộ nhỏ** (< 2.000 mẫu; gần như trong phân phối huấn luyện của TabPFN vốn tối đa 1.024 mẫu ngữ cảnh) và **48 bộ trung bình/lớn** (≥ 2.000 mẫu, tới 130.064; ngoài phân phối).
- **Giao thức:** chia TabZilla 80/10/10 huấn luyện/validation/kiểm tra; **10-fold** cho mọi phương pháp; baseline tinh chỉnh **30 vòng** siêu tham số (như McElfresh 2023) và chọn cấu hình tốt nhất mỗi fold theo AUC validation; TabPFN báo cáo không ensemble hay biến đổi trừ khi nói rõ (mục 4.4 so với TabPFN có biến đổi).
- **Baseline:** kNN, TabPFN, TabPFN 3k (ngữ cảnh 3.000), LightGBM, Random Forest, CatBoost, XGBoost; so với học sâu (VIME, rtdl_MLP, TabNet, STG, rtdl_ResNet, rtdl_FTTransformer) trên tập con 71 bộ mà các mô hình sâu chạy được (Bảng 4–5).
- **Thước đo:** AUC; vì có nhiều bộ dữ liệu nên báo cáo **IQM** (trung bình của 50% điểm giữa) và trung bình thường, khoảng tin cậy 95% bằng bootstrap phân tầng (thư viện rliable của Agarwal et al.). Thêm accuracy và F1 (Bảng 6, 7).

## 6. Kết quả

### 6.1. Kết quả chính (Bảng 1, AUC)
IQM AUC / mean AUC (95 bộ; bộ nhỏ; bộ trung bình-lớn):

| Phương pháp | Tất cả 95 bộ | 47 bộ nhỏ | 48 bộ trung bình/lớn |
|---|---|---|---|
| kNN | 0,843 / 0,812 | 0,807 / 0,781 | 0,882 / 0,848 |
| TabPFN | 0,917 / 0,867 | 0,898 / 0,849 | 0,927 / 0,884 |
| TabPFN 3k | 0,924 / 0,873 | 0,903 / 0,852 | 0,938 / 0,893 |
| LightGBM | 0,940 / 0,885 | 0,884 / 0,838 | 0,966 / 0,931 |
| Random Forest | 0,936 / 0,886 | 0,895 / 0,848 | 0,955 / 0,920 |
| CatBoost | 0,942 / 0,891 | 0,907 / 0,856 | 0,961 / 0,926 |
| XGBoost | 0,943 / 0,892 | 0,907 / 0,861 | 0,965 / 0,931 |
| **TabPFN-kNN** | 0,943 / 0,891 | 0,922 / 0,864 | 0,955 / 0,916 |
| **LoCalPFN** | **0,958 / 0,908** | **0,937 / 0,882** | **0,968 / 0,934** |

(Ghép cột từ bảng trích văn bản; giá trị khớp với Bảng 8 và 11 ở phụ lục nên đáng tin.)

- Trên **bộ nhỏ**, TabPFN vốn đã cạnh tranh (đúng như McElfresh); LoCalPFN vẫn cải thiện và là phương pháp tốt nhất.
- Trên **bộ trung bình/lớn**, TabPFN **thua cây một khoảng rất lớn**; chỉ cần thêm kNN (TabPFN-kNN) đã tăng đột ngột (IQM 0,927 → 0,955); LoCalPFN cải thiện thêm (0,968) và ngang hoặc vượt mọi phương pháp, kể cả XGBoost (0,965).
- **Accuracy và F1** (Bảng 6, 7) cho cùng xu hướng (ví dụ IQM accuracy tất cả: TabPFN 0,856, XGBoost 0,889, TabPFN-kNN 0,877, LoCalPFN 0,902).
- **So với học sâu** (71 bộ, Bảng 5): IQM AUC VIME 0,771; rtdl_MLP 0,855; TabNet 0,881; STG 0,877; ResNet 0,917; FT-Transformer 0,919; TabPFN 0,929; Fine-Tune 0,936; TabPFN-kNN 0,948; LoCalPFN 0,959: LoCalPFN tốt nhất. (Các mô hình học sâu tốn nhiều thời gian nên không chạy hết 95 bộ.)

### 6.2. Mở rộng theo kích thước và độ phức tạp (mục 4.3, Hình 4)
- **Theo kích thước N:** so với Random Forest, hiệu năng TabPFN **giảm mạnh khi N vượt 3.000**; các phương pháp khác mở rộng tốt; LoCalPFN mở rộng thuận lợi và **vượt cả XGBoost** ở bộ lớn.
- **Theo độ phức tạp:** vì không có thước đo chung, bài dùng thước đo xấp xỉ: **chênh lệch AUC giữa phương pháp tốt nhất và tệ nhất** trong số các phương pháp ở Bảng 1 trên mỗi bộ (như McElfresh); chia 5 phân vị từ dễ tới khó. TabPFN mở rộng kém khi độ phức tạp tăng; **LoCalPFN vẫn vượt mọi phương pháp ở các phân vị khó nhất**, tức cải thiện không chỉ ở bộ "dễ".

### 6.3. Ablation (mục 4.4, Phụ lục A.5)
- **Tinh chỉnh kết hợp truy hồi là then chốt (Bảng 8):** tinh chỉnh TabPFN thuần với ngữ cảnh ngẫu nhiên cải thiện so với TabPFN gốc (IQM 0,917 → 0,934), nhưng thêm kNN lên mô hình đã tinh chỉnh **không cải thiện, còn giảm** (0,938) so với TabPFN-kNN (0,943); trong khi tinh chỉnh **cùng truy hồi (LoCalPFN)** cho 0,958. Ví dụ Medium/Large: Fine-Tune 0,953, Fine-Tune+kNN 0,946 (giảm), TabPFN-kNN 0,955, LoCalPFN 0,968.
- **Chọn embedding để truy hồi (Bảng 9):** dùng đặc trưng chuẩn hóa thô đã rất cạnh tranh; one-hot cho cải thiện vừa phải (chủ yếu bộ nhỏ); đầu ra của encoder TabPFN (tính lại chỉ mục mỗi 30 bước) không tốt hơn. Giải thích: đặc trưng bảng thường mang ý nghĩa riêng, nên khoảng cách tách thành tổng theo từng đặc trưng d(x, x′) = Σ_i d_i(x_i, x′_i) là thiên kiến quy nạp tốt khi đã chuẩn hóa, điều không đúng với tín hiệu tự nhiên (ảnh, âm thanh).
- **Ngữ cảnh địa phương hơn ngữ cảnh toàn cục (Hình 5, Bảng 10):** thử TabPFN-32ens (ensemble 32 mô hình hoán vị đặc trưng và lớp), TabPFN-3k-32ens (ngữ cảnh 3.000), TabPFN-32ens-3k-int ("tích phân" xác suất trên các khối 3.000 của D_train, tốn kém O(N)), và In-Context Distillation (TabPFN-ICD; tối ưu trực tiếp ngữ cảnh, đã có tinh chỉnh nên so sánh được với LoCalPFN): tất cả cải thiện TabPFN nhưng **không bằng TabPFN-kNN**, chưa nói LoCalPFN.
- **Số láng giềng tối đa (Hình 6, Bảng 11):** TabPFN-kNN ít nhạy với k miễn là ≥ 100; LoCalPFN cải thiện TabPFN-kNN ở mọi cỡ ngữ cảnh; chỉ với **tối đa 50 láng giềng LoCalPFN đã vượt Random Forest** và với **500 láng giềng vượt XGBoost**. Ví dụ AUC trung bình (tất cả 95): TabPFN-kNN từ 0,874 (tối đa 20) tới 0,891 (500+); LoCalPFN từ 0,898 (20) tới 0,908 (1000).
- **Chi phí (Phụ lục A.5.6, Hình 11):** **TabPFN-kNN chạy rất nhanh**, ngang cây, AUC chỉ thấp hơn XGBoost một chút; **LoCalPFN đạt AUC cao nhất nhưng chậm hơn cây**; vẫn nhanh hơn các phương pháp học sâu khác.

## 7. Hạn chế (tác giả nêu và suy ra)
1. Chỉ chứng minh trên **TabPFN v1** (mô hình ICL bảng duy nhất đã được chứng minh lúc đó); chưa chắc chuyển sang mô hình nền tảng khác, dù ý tưởng thành công ở LLM (RAG ban đầu cũng chỉ trên BART).
2. Thừa hưởng giới hạn TabPFN v1: **tối đa 100 đặc trưng, 10 lớp, không có hồi quy**, không NaN (bộ có NaN bị loại khỏi 95 bộ).
3. LoCalPFN **chậm hơn cây**; cần tinh chỉnh riêng cho mỗi bộ dữ liệu (mất lợi thế "không huấn luyện" của ICL).
4. Không đánh giá **hiệu chuẩn, nhãn nhiễu, mất cân bằng lớp** (chỉ AUC, accuracy, F1).
5. Bản PDF đang ghi là "preprint, under review"; không tinh chỉnh siêu tham số lớn nhưng cần tập validation cho dừng sớm.
6. Dựa vào **kNN trong không gian đặc trưng thô**, dễ kém khi nhiều đặc trưng vô ích hoặc số chiều lớn (lời nguyền chiều).

## 8. Liên hệ với bài Nature và đề tài
- Bài Nature (bài 1) trích dẫn đây là ví dụ "retrieval & fine-tuning cho mô hình bảng ICL" (tham chiếu [50]) trong hướng tương lai; về lý thuyết, bài 12 (Nagler) giải thích vì sao **địa phương hóa** làm bias của PFN giảm (Localized TabPFN dùng kNN rất giống TabPFN-kNN ở đây).
- TabDPT (bài 8) cũng dùng truy hồi kNN cả lúc huấn luyện, nhưng tiền huấn luyện trên dữ liệu thật; TabICLv2 (bài 7) không cần truy hồi nhờ QASSMax và tiền huấn luyện chuỗi dài, thậm chí RF-routing còn làm hại.
- Khoảng trống tiềm năng: chưa ai đánh giá LoCalPFN (hay biến thể truy hồi) trên dữ liệu **nhãn nhiễu**; kNN rất nhạy với nhãn sai ở láng giềng, nên đây là điểm cần kiểm tra khi lập ma trận 4 SOTA.

## 9. Từ khóa cần nhớ
- **TabPFN-kNN:** ngữ cảnh = k láng giềng gần nhất của từng truy vấn, không cần huấn luyện.
- **LoCalPFN:** TabPFN-kNN + tinh chỉnh toàn bộ mạng end-to-end trên ngữ cảnh địa phương xấp xỉ (B ngữ cảnh dùng chung).
- **k = min(10·√N, 1.000)**; AdamW lr 0,01; 1.000 truy vấn, batch 2; dừng sớm theo AUC validation.
- **IQM AUC:** trung bình 50% giữa; kết quả chính LoCalPFN 0,958 so với XGBoost 0,943 (95 bộ).
