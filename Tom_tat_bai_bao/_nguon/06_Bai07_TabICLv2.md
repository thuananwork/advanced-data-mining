# Bài 7. TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2026 (PMLR 306); arXiv 2602.11139 (v2, 16/9/2026) |
| Tác giả | Jingang Qu, David Holzmüller (đồng tác giả chính), Gaël Varoquaux, Marine Le Morvan (SODA team, Inria Saclay; Probabl) |
| Mã nguồn | github.com/soda-inria/tabicl (mã suy luận, tiền huấn luyện và sinh dữ liệu); bản cài đặt giáo dục github.com/soda-inria/nanotabicl |
| Vị trí trong đề tài | **Phương pháp bạn sẽ chạy trực tiếp** cho bài giữa kỳ (TabICLv2). Đây cũng là một trong 4 SOTA cần so sánh. |

## 1. Bài này làm gì trong một đoạn

TabICLv2 là mô hình nền tảng cho dữ liệu bảng (TFM) mới của nhóm Inria, kế thừa TabICL (bài 4), cho cả **phân loại và hồi quy**, dựa trên ba trụ cột: (1) một **bộ sinh dữ liệu tổng hợp mới** (prior) có độ đa dạng cao; (2) các cải tiến kiến trúc, đáng chú ý là **Query-Aware Scalable Softmax (QASSMax)** giúp tổng quát hóa sang tập dữ liệu lớn hơn lúc huấn luyện mà không cần tiền huấn luyện chuỗi dài; (3) quy trình tiền huấn luyện được tối ưu, nổi bật là dùng bộ tối ưu **Muon** thay cho AdamW. Không tinh chỉnh gì, TabICLv2 **vượt RealTabPFN-2.5** (đã tinh chỉnh siêu tham số, ensemble, và tinh chỉnh trên dữ liệu thật) trên **TabArena** và **TALENT**, nhanh hơn rõ rệt (10,6 lần trên GPU H100 ở 50.000 mẫu), và xử lý được bảng cỡ triệu mẫu dưới 50 GB bộ nhớ GPU. Mô hình khoảng **28 triệu tham số**, mã nguồn đầy đủ (kể cả sinh dữ liệu và tiền huấn luyện) được công bố.

## 2. Bối cảnh và động cơ

- TFM kiểu PFN (Müller 2022): với prior p(D) các tập dữ liệu, huấn luyện q_θ để tối thiểu hóa L(θ) = E_{D~p}[−log q_θ(y_test | x_test, D_train)]; khi dùng thật, một lần chạy xuôi cho ICL, không cập nhật gradient.
- TFM đã "lật đổ" cây tăng cường trên các benchmark dự đoán. Kiến trúc đã có: **TabPFN** (mỗi dòng là một token), **TabPFNv2** (mỗi ô một token, chú ý luân phiên hàng/cột, độ phức tạp O(n²m + nm²)), **TabPFN-2.5** (sâu hơn), **TabICL** (hai giai đoạn: nhúng cột rồi tương tác dòng, nén thành vector dòng rồi mới làm ICL, độ phức tạp O(n² + nm²)), cùng Mitra, LimiX, Orion-MSP.
- Cần có TFM **mã nguồn mở hoàn toàn** ngang tầm các mô hình đóng (TabPFN-2.5 chỉ mở trọng số, không công bố dữ liệu và công thức huấn luyện) để dân chủ hóa và làm rõ "công thức" của các TFM mạnh nhất.
- Vấn đề kỹ thuật chính: **suy giảm chú ý (attention fading)**. Mẫu số của softmax tăng theo độ dài ngữ cảnh n làm phân phối chú ý phẳng ra, mô hình huấn luyện trên chuỗi ngắn không còn chú ý sắc nét khi suy luận trên chuỗi dài. Giải pháp tồn tại như softmax thay thế (Entmax, sigmoid attention) không tương thích FlashAttention, nên bài chọn hướng nhẹ hơn: **co giãn nhiệt độ (temperature scaling)** như Scalable Softmax (SSMax: nhân query với s·log n), ASEntmax, Selective Attention, YaRN.

## 3. Kiến trúc

TabICLv2 giữ chuỗi ba bước của TabICL với độ phức tạp thời gian **O(n² + n·m²)** cho bảng n dòng, m cột, rồi thêm các cải tiến (đánh dấu ▶ trong bài) mà không tăng kích thước mô hình (~28M tham số). Có **hai checkpoint riêng** cho phân loại và hồi quy.

**Ba tầng chính:**
1. **Nhúng theo cột** TF_col: Set Transformer (induced self-attention với k vector cảm ứng). Giai đoạn 1 gom thông tin đầu vào vào các vector cảm ứng, giai đoạn 2 phát lại cho đầu vào, giảm độ phức tạp từ O(n²) xuống O(nk). Cải tiến: dùng thẳng đầu ra TF_col làm embedding đặc trưng (TabICL còn có thêm biến đổi), áp **QASSMax** ở giai đoạn 1, và nhờ nhúng nhãn sớm TF_col thực chất là một bộ học ICL trong từng cột.
2. **Tương tác theo dòng** TF_row: thêm 4 token [CLS] học được vào mỗi dòng, Transformer có RoPE; nối đầu ra 4 token [CLS] thành vector dòng 4d chiều, qua đó nén chiều đặc trưng.
3. **ICL ở mức bộ dữ liệu** TF_icl: nối embedding dòng huấn luyện với embedding nhãn; mẫu kiểm tra chỉ chú ý tới mẫu huấn luyện; MLP hai lớp cho dự đoán. Áp **QASSMax**.

**Các cải tiến mới:**
- ▶ **Repeated feature grouping (gom đặc trưng lặp lại):** TabICL nhúng từng đặc trưng độc lập nên dễ **sụp đổ biểu diễn** khi các đặc trưng có phân phối giống nhau. TabPFNv2 gom nhiều cột vào một token (cũng giảm số đặc trưng hiệu dụng nhưng có thể mất chi tiết). TabICLv2 tạo **m nhóm** bằng dịch vòng tròn: nhóm j chứa các cột ở vị trí (j, j+1, j+3) mod m, mã hóa bằng một lớp tuyến tính chung Lin: R³ → R^d. Với bảng có ít nhất 7 cột, không cặp cột nào xuất hiện cùng nhau trong quá một nhóm. Thử nhóm lớn hơn không cho cải thiện nhất quán.
- ▶ **Target-aware embedding (nhúng có nhận biết nhãn):** cộng embedding nhãn **trực tiếp vào mọi token đặc trưng** của mẫu huấn luyện, E₂[i,j] = E₁[i,j] + Embed_TAE(y_i) (bảng tra cứu cho phân loại, lớp tuyến tính cho hồi quy), thay vì thêm nhãn như một cột phụ (TabPFNv2) hay chỉ đưa nhãn vào sau khi nén dòng (TabICL). Nhờ vậy mô hình biết đặc trưng nào liên quan trước khi nén, và giảm sụp đổ biểu diễn (hai đặc trưng cùng phân phối vẫn có quan hệ nhãn khác nhau).
- ▶ **QASSMax:** với query q_h của đầu h và cỡ tập huấn luyện n, SSMax nhân query với một vô hướng học được s_h·log n. QASSMax mở rộng: q̃_hi = q_hi · MLP_base(log n)_hi · (1 + tanh(MLP_gate(q_h)_i)), tức **co giãn cơ sở theo từng phần tử** (hàm của log n) nhân với **cổng phụ thuộc query** nằm trong (0, 2). Cả hai MLP có 2 lớp, 64 nơ-ron ẩn, GELU; lớp cuối của MLP_gate khởi tạo bằng 0 để ban đầu là phép đồng nhất. Lý do thiết kế: log n cần thiết để triệt tiêu mẫu số tăng tuyến tính; co giãn từng phần tử tăng biểu đạt; cổng theo query theo Selective Attention; cổng giống Gated Attention (Qiu 2025).
  - **Thí nghiệm "kim trong đống cỏ"** (Hình 3): một mẫu mỏ neo giữa 4 cụm âm tính tăng dần số mẫu âm; không dùng SSMax thì entropy chú ý tăng và độ chính xác giảm, sụp đổ ở 15K âm tính; **QASSMax giữ 100% với entropy thấp ngay cả 15K âm tính**, tốt hơn SSMax.
- ▶ **Mixed-radix ensembling (nhiều hơn 10 lớp):** mô hình tiền huấn luyện với tối đa 10 lớp. TabICL dùng **hierarchical classification** ở giai đoạn ICL, nhưng nhúng nhãn sớm đưa nhãn vào trước khi phân cấp. Giải pháp: với C > 10, tính các cơ số cân bằng [k₀, …, k_{D−1}] mỗi k_i ≤ 10, tích ≥ C, D nhỏ nhất; mã hóa mỗi nhãn y thành D "chữ số" y^(i) = ⌊y / Π_{j>i} k_j⌋ mod k_i. Ví dụ 16 lớp với [4, 4]: view 0 chia lớp theo khối liên tiếp, view 1 theo phần dư; không view đơn lẻ nào phân biệt hết 16 lớp nhưng cặp chữ số là song ánh. Chạy TF_col một lần cho mỗi view rồi lấy trung bình O_avg = (1/D) Σ TF_col(E₁ + Embed_TAE(y^(i))). Làm việc ở mức embedding và dùng mã k-phân (k ≤ 10), khác ECOC (mã nhị phân, nhiều bộ phân loại riêng). Kết hợp với phân cấp ở TF_icl cho số lớp bất kỳ, không cần huấn luyện lại.
- ▶ **Dự đoán phân vị cho hồi quy:** TabPFNv2 rời rạc hóa nhãn thành các thùng và dùng cross-entropy; Mitra, TabDPT dự đoán điểm bằng MSE. TabICLv2 dự đoán **999 phân vị** α ∈ {0,001; 0,002; …; 0,999} với **pinball loss** cộng trên mọi phân vị. Thử nghiệm sơ bộ với đánh giá RMSE: tốt hơn MSE và cách dùng thùng của TabPFNv2. Suy luận: điểm dự đoán = **trung bình các phân vị** (nhanh và hiệu quả); phân phối đầy đủ = sắp xếp để đảm bảo đơn điệu (mặc định) hoặc hồi quy đẳng hướng, ngoại suy đuôi bằng mô hình mũ tham số, có PDF, CDF, moment dạng đóng (Phụ lục I).

**Cấu hình (Bảng A.1, Phụ lục A.4).** TF_col: 3 khối induced self-attention, 128 vector cảm ứng, d = 128, 8 đầu. TF_row: 3 lớp, d = 128, 8 đầu, 4 token [CLS]. TF_icl: **12 lớp, d = 512, 8 đầu**. Pre-norm, GELU, hệ số mở rộng FFN 2×. Đầu dự đoán MLP 2 lớp, ẩn 1024, ra 10 (phân loại) hoặc 999 (hồi quy). Hồi quy dùng layer norm không bias.

## 4. Tiền huấn luyện và suy luận

- **Ba giai đoạn** (tăng dần cỡ dữ liệu, tối đa 100 đặc trưng ở cả ba), batch 64 bảng, tổng khoảng **35 triệu bảng** (TabICL ≈ 83M, TabPFNv2 ≈ 130M):
  - Giai đoạn 1: **500K bước**, mỗi bảng 1.024 mẫu (30–90% dùng để huấn luyện), tốc độ học tối đa 8e-4.
  - Giai đoạn 2: 40K bước, 400–10.240 mẫu (log-uniform), 80% huấn luyện, 1e-4.
  - Giai đoạn 3: 10K bước, 400–60.000 mẫu (log-uniform), 80% huấn luyện, 2e-5.
  - Phụ lục B.1: mỗi giai đoạn cải thiện đều trên TALENT. Hạng trung bình (thấp tốt): 9,94 (giai đoạn 1) → 5,69 (giai đoạn 2) → 5,41 (giai đoạn 3). Trên bảng lớn (> 10K mẫu): 14,91 → 5,50 → 4,71 (vượt RealTabPFN-2.5 ở hạng 6,35). Tức tiếp xúc với bảng lớn trong tiền huấn luyện là then chốt.
- **Bộ tối ưu Muon** thay AdamW: trực giao hóa cập nhật gradient bằng lặp Newton–Schulz (xuống dốc nhất theo chuẩn phổ), nhân tốc độ học mỗi tham số W ∈ R^{n×m} với 0,2·√max{n, m} (theo Moonlight). Dùng tốc độ học cao hơn (8e-4 so với 1e-4 của AdamW ở TabICL), weight decay 0,01, cắt gradient 10 (thay vì 1) ở giai đoạn 1 và 2, lấy mẫu cỡ huấn luyện/kiểm tra khác nhau cho mỗi micro-batch, lịch cosine.
- **Chi phí:** trên H100 80GB: giai đoạn 1 ≈ **20 GPU-ngày**, giai đoạn 2 ≈ 2,5, giai đoạn 3 ≈ 2; tổng **24,5 GPU-ngày** mỗi mô hình (một giờ H100 ≈ 2 giờ A100 nên rẻ hơn TabICL 60 A100-ngày).
- **Tối ưu suy luận:** offloading ra đĩa (Phụ lục H.2) → xử lý bảng **1 triệu mẫu × 500 đặc trưng trong 450 giây** với dưới 24 GB CPU và 50 GB GPU; chỉ tính chiếu Q/K/V cần thiết (khối cuối TF_row chỉ tính truy vấn cho 4 token [CLS]; TF_icl chỉ tính K, V cho mẫu huấn luyện); tái sử dụng layer norm.
- **Ensemble khi suy luận:** hoán vị cột/lớp ngẫu nhiên và các bộ tiền xử lý khác nhau như TabICL. Mặc định **8 estimator**; TALENT dùng 32.

## 5. Prior dữ liệu tổng hợp (đóng góp lớn nhất theo ablation)

Giữ khung SCM của TabPFN, hấp thu ý tưởng từ prior của TabICL và TabPFNv2, thêm nhiều lựa chọn mới. Phát triển chủ yếu dựa trên **nguyên tắc thiết kế** (tối đa hóa đa dạng, mã hóa thiên kiến quy nạp hữu ích, hiệu quả tính toán), không dựa trên phản hồi thực nghiệm vì ablation chi tiết dễ khớp quá tập kiểm định.
- **Cấu trúc:** lấy mẫu thuộc tính toàn cục (số đặc trưng số/phân loại, cỡ bộ dữ liệu), một DAG, vector ngẫu nhiên tại gốc rồi truyền qua đồ thị, mỗi đặc trưng đọc từ một nút được gán ngẫu nhiên, **chỉ dùng một tập con chiều của nút** (các chiều còn lại không quan sát được, tạo nhiễu). **Không cộng nhiễu Gauss ở cấp nút** (khác các prior trước). Đặc trưng số lấy từ một chiều; đặc trưng phân loại lấy nhiều chiều rồi rời rạc hóa bằng nearest-neighbor hoặc softmax.
- **Tám loại hàm ngẫu nhiên** trên cạnh: MLP, tập hợp cây đối xứng kiểu CatBoost, rời rạc hóa theo láng giềng gần nhất, hàm Gaussian process đa biến, tuyến tính, bậc hai, hàm có đoạn bằng phẳng kiểu cụm EM, và tích các hàm khác. Ba cái đầu lấy từ TabPFNv2, năm cái còn lại là mới; chúng phủ nhiều mức độ trơn (có chứng minh cho GP) và nhiều thiên kiến (đoạn bằng phẳng, song song trục). Nhiều cha → hoặc nối tất cả rồi áp một hàm, hoặc áp hàm riêng rồi gộp bằng tổng, tích, max, logsumexp.
- **Cơ chế lấy mẫu đồ thị mới "random Cauchy graph"** (prior TabPFNv2 chỉ sinh đồ thị dạng cây): mô hình hóa độ liên kết toàn cục và cục bộ khác nhau (Phụ lục E.4).
- **Hậu xử lý** giống TabICL: bỏ cột và bộ dữ liệu có vấn đề, hoán vị cột và nhãn, tiền xử lý đặc trưng/nhãn. **Chuẩn hóa rồi co giãn cột ngẫu nhiên** để mô phỏng độ quan trọng đặc trưng khác nhau; warping (Kumaraswamy); nhân dữ liệu nút với một vô hướng ngẫu nhiên ("độ quan trọng nút").
- **Lọc dữ liệu:** loại bộ mà một mô hình **ExtraTrees** không vượt được baseline hằng số theo kiểm định bootstrap, và loại đồ thị mà các nút đặc trưng không có tổ tiên chung với nút nhãn (nhãn độc lập với đặc trưng). Ở giai đoạn 1 lọc khoảng **35% bộ phân loại và 25% bộ hồi quy**.
- **Lấy mẫu vô hướng tương quan:** nhiều siêu tham số cùng loại (ví dụ số lớp mỗi cột phân loại) được lấy chung tham số Beta, để tái tạo hiện tượng "nhiều cột phân loại có cùng số giá trị" ở dữ liệu thật.
- Prior mới là **yếu tố then chốt**: tiền huấn luyện TabICLv2 bằng prior của TabICL cho hiệu năng thấp hơn đáng kể.

## 6. Thiết lập thí nghiệm

- **TabArena** (Erickson 2025, "living benchmark"): **51 bộ** (38 phân loại ≤ 10 lớp, 13 hồi quy), đánh giá bằng CV lặp, thước đo ROC AUC (nhị phân), log-loss (đa lớp), RMSE (hồi quy). Dùng 8 estimator để khớp RealTabPFN-2.5.
- **TALENT** (Ye 2024): **300 bộ** (120 nhị phân, 80 đa lớp, 100 hồi quy), chia 64/16/20; siêu tham số chọn trên validation theo accuracy / RMSE. Dùng 32 estimator cho TabICLv2 và TabPFN-2.5 / RealTabPFN-2.5. Còn có "TALENT extension" với bảng rất lớn (Covertype 581K mẫu, Data Science Good Kiva 671K mẫu).
- **Thước đo tổng hợp chính: improvability** (độ có thể cải thiện) = trung bình theo bộ dữ liệu của (err − err_tốt_nhất) / err × 100%, nhạy với biên độ chênh lệch (khác Elo, vốn chỉ đếm thắng–thua). Ngoài ra Elo (1.000 ứng với RandomForest mặc định, bootstrap 200 lần) và hạng trung bình (Phụ lục J, K). Với TabICL (v1) dùng checkpoint TabICLv1.1 theo TabArena và TALENT.
- Ba chế độ: Default (siêu tham số mặc định), Tuned (tốt nhất trong 200 cấu hình ngẫu nhiên theo validation), Tuned + Ens. (ensemble hậu kỳ có trọng số trên mọi cấu hình).

## 7. Kết quả

- **TabICLv2 đạt SOTA trên cả hai benchmark.** Trên mặt Pareto của improvability theo thời gian (Hình 1, 5) TabICLv2 thống trị. Không tinh chỉnh, nó **vượt RealTabPFN-2.5 (tuned + ensemble)**. Kiểm định nhị thức trực tiếp trên TALENT: win rate của TabICLv2 so với RealTabPFN-2.5 lớn hơn 50% có ý nghĩa với accuracy (p = 0,0018), AUC (6,4e-9), log-loss (1,4e-8), R² (0,0028); so với TabPFN-2.5 thấp hơn nữa (p tới 1,0e-10). Biểu đồ khác biệt tới hạn không cho thấy khác biệt có ý nghĩa giữa hai mô hình (có thể thiếu độ mạnh kiểm định do hiệu chỉnh so sánh bội), nên cần kiểm định trực tiếp bổ sung.
- **TabArena (Phụ lục J):** hạng trung bình của TabICLv2 (mặc định) **4,82**, tốt hơn AutoGluon 1.4 (extreme, 4 giờ) **5,24** và RealTabPFN-2.5 (T+E) **5,88**. Tuy nhiên **AutoGluon 1.5 (extreme, 4 giờ, có TabDPT) đạt hạng 3,88 (tốt hơn)**; nhưng ma trận thắng thua đôi cho TabICLv2 thắng AutoGluon 1.5 ở **57%** bộ và thắng RealTabPFN-2.5 (T+E) ở **59%** bộ. Giải thích của tác giả: win rate chỉ đếm thắng/thua; hạng trung bình phạt các bộ mà mô hình kém hẳn; nên TabICLv2 có thể **kém ở một số bộ nằm ngoài phân phối tiền huấn luyện**. Cũng trên TabArena: hạng của các mô hình khác gồm RealMLP 9,04; TabDPT 9,82; LimiX 10,24; TabM 11,27; LightGBM 11,84; CatBoost 12,27; TabPFNv2 12,30; Mitra 12,74; TabICL 13,20.
- **Tốc độ:** TabICLv2 luôn nhanh hơn TabPFN-2.5 (Hình 6, 100 đặc trưng, 8 estimator, phân loại với 500 mẫu kiểm tra): **10,6 lần trên GPU H100 NVL tại 50K mẫu** và **11,8 lần trên CPU chỉ ở 10K mẫu**.
- **Nhiều lớp (Hình 7):** trên 12 bộ TALENT có hơn 10 lớp, TabICLv2 (cả với ECOC wrapper của TabPFNv2 lẫn mixed-radix gốc) **vượt rõ mọi baseline**; ECOC hơi tốt hơn nhưng chậm hơn khoảng 3 lần.
- **Mở rộng quy mô (Hình 8, 9):** giữ hạng cao ở mọi cỡ bộ dữ liệu từ 10³ đến 10⁵, vượt RealTabPFN-2.5 trên bộ lớn (> 20K); trên bộ rất lớn (≈ 600K mẫu) vẫn mạnh (Covertype, Kiva) trong khi **TabPFN-2.5 hết bộ nhớ**.
- **Tiếp cận ít mẫu (Phụ lục L.3):** thử với đúng 16 mẫu huấn luyện (101 nhị phân, 119 hồi quy; chưa từng thấy bảng dưới 300 mẫu lúc tiền huấn luyện). Hạng AUC phân loại: RealTabPFN-2.5 **4,00**, TabICLv2 **4,11**, ExtraTrees 4,49, RandomForest 4,91, RealMLP 6,00...; hạng RMSE hồi quy: RealTabPFN-2.5 4,18, TabICLv2 5,02, RealMLP 5,21, ExtraTrees 5,52... TabICLv2 đứng thứ hai sau RealTabPFN-2.5 và vượt mọi phương pháp truyền thống.
- **So với LLM (L.4):** trên 9 bộ phân loại TALENT ≤ 2.048 mẫu, với 8/16/64 mẫu: Claude Opus 4.6 cho 0,536 / 0,541 / 0,606; Qwen3.5-112B cho 0,511 / 0,531 / 0,574; **TabICLv2 cho 0,515 / 0,586 / 0,659**. LLM chỉ cạnh tranh được ở 8 mẫu; từ 16 mẫu TabICLv2 thắng, và hiệu quả hơn nhiều bậc.
- **Kích thước ensemble (L.1):** lợi ích tăng dần nhưng giảm dần; E = 8 đã lấy được phần lớn (tăng độ chính xác phân loại +0,20% và RMSE +1,22% so với E = 1; E = 32 cho +0,26% và +1,46%; thời gian suy luận 0,9 giây ở E = 8, 2,6 giây ở E = 32).
- **Tiện ích RF của TabPFN (L.5):** gán mẫu kiểm tra qua random forest rồi khớp TFM trên tập con mỗi lá (giúp TabPFNv2 và TabICL) **làm hại TabICLv2** trên bộ > 10K mẫu (accuracy 0,857 → 0,848, AUC 0,897 → 0,891, log-loss 0,306 → 0,327; R² 0,720 → 0,687), vì chia dữ liệu làm vỡ ngữ cảnh hữu ích; QASSMax đã đủ xử lý bộ lớn.

### 7.1. Khả năng chịu nhãn nhiễu (Phụ lục L.6, rất liên quan đến khoảng trống nghiên cứu)
- Thiết lập: chọn **6 bộ nhị phân ít nhiễu** từ TabArena (theo AUC tốt nhất trong bài TabArena: hazelnut-spread-contaminant-detection, NATICUSdroid, online_shoppers_intention, polish_companies_bankruptcy, qsar-biodeg, taiwanese_bankruptcy_prediction), **thay 0%–50% nhãn huấn luyện bằng nhãn ngẫu nhiên**, 5 lần chạy với chia ngoài ngẫu nhiên (40% dữ liệu để kiểm tra), đo AUROC trên tập kiểm tra sạch. So với CatBoost (tuned default của pytabkit, ensemble CV 8 lần, dừng sớm theo AUROC), ExtraTrees, RandomForest, RealTabPFN-2.5.
- Kết quả: **TabICLv2 và RealTabPFN-2.5 suy giảm ít hơn các mô hình dựa trên cây**.
- Phạm vi hẹp: chỉ nhị phân, chỉ nhiễu đối xứng ngẫu nhiên đồng đều, chỉ 6 bộ, không có nhiễu phụ thuộc lớp/đặc trưng, không đề cập mất cân bằng lớp. **Bài này không đo hiệu chuẩn, không đo mất cân bằng lớp.**

## 8. Ablation (mục 7 và Phụ lục C)

Kết quả đo trên 60 bộ validation dùng để phát triển TabPFNv2 (tối đa 2.048 mẫu huấn luyện/bộ, 2 phép chia), mô hình tham chiếu huấn luyện 280K bước (không QASSMax, 4 thay vì 8 đầu chú ý).
- **Tương tác mạnh giữa kiến trúc và prior:** huấn luyện kiến trúc TabICLv2 với **prior của TabICL thất bại** (hiệu năng dưới TabICL, mất mát kiểm định tăng ở nửa sau); còn kiến trúc TabICL với prior TabICLv2 chỉ ngang TabICL: kiến trúc cũ ít khai thác được độ đa dạng prior cao hơn. Gợi ý: kiến trúc mới cần prior đa dạng hơn để tổng quát hóa.
- **Prior có tác động lớn nhất.** Ba thành phần cho cải thiện tương đương và đáng kể (≈ **100 điểm Elo, win rate 64%**): **nhúng nhãn sớm**, **Muon thay AdamW**, **QASSMax**. **Gom đặc trưng lặp** và **lọc prior** đóng góp nhỏ hơn. Thứ tự này nhất quán qua accuracy chuẩn hóa, Elo, log-loss.
- **Tăng độ sâu** (4 lớp TF_col/TF_row, 18 lớp TF_icl): không rõ cải thiện theo log-loss, chỉ hơi tốt hơn ở cuối quá trình theo accuracy và Elo; tác giả không theo hướng này vì mục tiêu là đổi mới có nguyên tắc chứ không phải tăng kích thước.
- **Thêm nhiễu Gauss vào prior** (như TabPFNv2): tác động không đáng kể.
- **Những thứ đã thử nhưng không dùng (Phụ lục D):** schedule-free AdamW, AdEMAMix (kém hơn Muon), label smoothing có lịch, MLP thay lớp tuyến tính cho nhúng x, thêm log n cạnh x, chú ý cột thường thay ISAB (kém hơn), cột riêng cho y (kém hơn), thay chú ý dòng đầy đủ bằng induced self-attention (kém hẳn), định danh đặc trưng ngẫu nhiên kiểu LimiX (không rõ), vị trí chuẩn hóa khác nhau, ISQF thay hồi quy phân vị (tương đương nhưng nặng hơn), hàm tích chập kiểu LimiX trong prior (không lợi), kiến trúc kiểu TabPFNv2 (không kết luận rõ nhưng bị loại vì chậm hơn). Lưu ý các lần chạy tiền huấn luyện khá nhiễu.

## 9. Hạn chế (tác giả nêu)
1. Không khai thác **ngữ nghĩa tên cột** hay đặc trưng văn bản (đã chứng minh hữu ích, ví dụ ConTextTab); tác giả cho rằng khả năng chịu nhiều đặc trưng cho phép ghép với mô hình nhúng văn bản.
2. Bảng hàng **triệu mẫu vẫn khó**, dù cải thiện.
3. Chưa hỗ trợ **hồi quy đa đầu ra** và **dịch chuyển phân phối** (Drift-Resilient TabPFN).
4. **Hồi quy phân phối** chưa được đánh giá ngoài bộ đồ chơi (do thiếu benchmark).
5. **Giá trị thiếu** hiện được **điền bằng trung bình**; thêm chỉ báo thiếu hoặc đưa tính thiếu vào tiền huấn luyện chưa được thử.
6. Chưa khám phá **tinh chỉnh siêu tham số hoặc fine-tuning** (Rubachev 2025), vốn có thể cải thiện thêm.
7. Từ phân tích: hạng trung bình kém hơn AutoGluon 1.5 trên TabArena; tác giả suy đoán TabICLv2 có thể yếu trên bộ ngoài phân phối tiền huấn luyện. Chưa kiểm tra mất cân bằng lớp. Chỉ nhãn ngẫu nhiên đối xứng ở thí nghiệm nhiễu.

## 10. Liên hệ và gợi ý cho đề tài

- **Cách chạy:** dùng mã chính thức (`tabicl`), có cả phân loại và hồi quy; mặc định 8 estimator. Cần lưu ý: bài đặt chuẩn đối sánh là TabArena (51 bộ) và TALENT (300 bộ); báo cáo giữa kỳ có thể chạy lại một tập con.
- **Điểm so sánh với TabPFNv2 (bài 1):** cùng ý tưởng ICL và prior SCM, nhưng TabICLv2 có kiến trúc hai giai đoạn (nén dòng) nên chi phí thấp hơn, prior mở và đa dạng hơn, dự đoán phân vị thay vì thùng, xử lý > 10 lớp bằng mixed-radix, mở rộng tới triệu mẫu.
- **Điều chỉnh quan sát trước:** trước đây ghi nhận "không bài nào đánh giá TFM với nhãn nhiễu". Bài này **có** một thí nghiệm (L.6) nhưng hẹp (nhãn ngẫu nhiên đồng đều, chỉ nhị phân, 6 bộ). Vẫn **chưa có đánh giá hệ thống** về (a) nhãn nhiễu phụ thuộc lớp/đặc trưng, (b) mất cân bằng lớp, (c) hiệu chuẩn xác suất dưới các điều kiện đó, (d) so với TabPFNv2/TabDPT/LoCalPFN. Đây vẫn là ứng viên khoảng trống, cần kiểm lại khi lập ma trận 4 SOTA.
- Điểm yếu có thể tận dụng làm "niche": missing value chỉ điền trung bình; chưa xét mất cân bằng lớp; chưa dùng ngữ nghĩa cột.

## 11. Từ khóa cần nhớ
- **QASSMax:** softmax co giãn nhận biết query, chống suy giảm chú ý khi n lớn.
- **Target-aware embedding:** cộng embedding nhãn vào mọi token đặc trưng huấn luyện từ đầu.
- **Repeated feature grouping:** nhóm (j, j+1, j+3) mod m.
- **Mixed-radix ensembling:** mã hóa nhãn nhiều lớp bằng "chữ số" cơ số ≤ 10 và lấy trung bình embedding.
- **Muon:** bộ tối ưu trực giao hóa cập nhật, thay AdamW.
- **Improvability:** khoảng cách sai số tương đối so với phương pháp tốt nhất trên từng bộ, lấy trung bình.
