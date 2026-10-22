# Bài 22. TabFlex: Scaling Tabular Learning to Millions with Linear Attention

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2025 (PMLR 267); arXiv:2506.05584v1, 5/6/2025 |
| Tác giả | Yuchen Zeng*, Tuan Dinh*, Wonjun Kang, Andreas C. Müller (Microsoft Gray Systems Lab, UW-Madison, UCSF, Furiosa AI, SNU) |
| Mã nguồn | github.com/microsoft/ticl |
| Lưu ý về PDF | Các bảng số (Bảng 1, 2, 3, 14) bị lệch cột khi trích chữ. Mình chỉ dùng số xác nhận được qua văn bản hoặc thứ tự hàng rõ ràng; các chỗ khác ghi theo nhận định định tính của tác giả. |
| Vị trí trong đề tài | Nhánh **mở rộng quy mô của TabPFN v1** (hiệu quả tính toán). Bổ sung cho TabICL/TabICLv2 (bài 4, 7) và LoCalPFN (bài 10): cùng giải bài toán "ngữ cảnh dài", nhưng bằng cách thay attention. |

## 1. Bài này làm gì trong một đoạn

TabPFN v1 xử lý mỗi mẫu như một token và phân loại toàn bộ tập kiểm tra trong một lượt truyền xuôi, nhưng attention softmax có độ phức tạp bậc hai theo độ dài chuỗi, nên chỉ dùng được cho dữ liệu nhỏ (≤ 1000 mẫu, ≤ 100 đặc trưng, ≤ 10 lớp). TabFlex thay attention softmax bằng **linear attention không nhân quả (non-causal)**, sau khi phân tích hai hướng thay thế: **mô hình không gian trạng thái (SSM, Mamba)** và linear attention. Kết luận sau các thí nghiệm kiểm soát: SSM mang tính nhân quả nên kém trong ICL; linear attention không nhân quả giữ độ chính xác ngang softmax mà nhanh hơn rõ. Bài huấn luyện **ba mô hình** (S100, L100, H1K) cho ba chế độ dữ liệu và chọn mô hình theo kích thước/số chiều bằng một thuật toán điều kiện đơn giản. Kết quả: trên dữ liệu nhỏ gần bằng TabPFN nhưng nhanh hơn hơn 2 lần; trên dữ liệu lớn/nhiều chiều xử lý được hàng triệu mẫu (poker-hand hơn 1 triệu mẫu trong khoảng 4,88 giây), hàng nghìn đặc trưng, hàng trăm lớp.

## 2. Bối cảnh và công trình liên quan

- Transformer cho dữ liệu bảng: TabNet, TabTransformer, FT-Transformer, SAINT, LIFT (biến mẫu thành câu cho LLM), TabR (kNN kết hợp), BiSHop, XTab. Các mô hình này phải huấn luyện riêng từng tập, nên độ trễ lớn so với GBDT.
- Các hướng mở rộng TabPFN đồng thời: **MixturePFN** (định tuyến mẫu kiểm tra tới nhiều prompter, ICLR 2025), **LoCalPFN** (truy xuất tập con cục bộ rồi tinh chỉnh), **in-context data distillation** (Ma et al. 2024), **TuneTables** (prefix tuning cho từng tập), **TabPFNv2** (cải thiện độ chính xác dưới 10.000 mẫu). TabFlex nhấn vào **tốc độ và quy mô**, bổ sung chứ không cạnh tranh với TabPFNv2.
- Attention thay thế: RNN (tuyến tính nhưng huấn luyện khó song song), linear attention (Katharopoulos 2020, công thức lại attention dưới dạng tích vô hướng ánh xạ đặc trưng ϕ), SSM (S4, Mamba, Mamba-2, RWKV, RetNet), FlashAttention (tối ưu truy cập bộ nhớ HBM nhưng độ phức tạp vẫn bậc hai). Có công trình dùng SSM cho bảng (MambaTab, Mambular) nhưng phải huấn luyện từng tập và chậm hơn GBDT. Linear attention từng bị báo cáo kém trong mô hình ngôn ngữ, nhưng lý thuyết gần đây cho thấy nó có thể mô phỏng gradient descent trong ICL (Ahn et al. 2023).

## 3. Kiến thức nền: ICL của TabPFN, SSM và linear attention

- **TabPFN:** mỗi mẫu là một token; mẫu huấn luyện nhúng (x, y), mẫu kiểm tra chỉ nhúng x, bằng MLP; 12 lớp transformer; attention ở vị trí huấn luyện nhìn tất cả mẫu huấn luyện (nên kết quả bất biến thứ tự huấn luyện); ở vị trí kiểm tra chỉ nhìn các mẫu huấn luyện (nên các mẫu kiểm tra độc lập); MLP cuối cho phân phối lớp. Tương đương ICL chuẩn nhưng chỉ một lượt thay vì m lượt cho m mẫu kiểm tra.
- **SSM:** trạng thái ẩn h_t = A h_{t−1} + B x_t, đầu ra y_t = C h_t; có thể viết dạng tích chập để huấn luyện song song. Đầu ra mỗi token chỉ phụ thuộc trạng thái và token hiện tại, nên **mang tính nhân quả**.
- **Linear attention:** thay exp(qᵢᵀkⱼ) bằng ϕ(qᵢ)ᵀϕ(kⱼ). Hai đại lượng Σⱼ ϕ(kⱼ)vⱼ và Σⱼ ϕ(kⱼ) chỉ tính một lần, mỗi vị trí chỉ nhân với ϕ(qᵢ), nên tổng độ phức tạp tuyến tính theo độ dài chuỗi (bài dùng ϕ = elu + 1). Ở phiên bản nhân quả, tổng chạy đến vị trí i, trở thành một RNN tuyến tính (một dạng SSM).

## 4. Khảo sát kiến trúc

Hai thí nghiệm kiểm soát trong cùng thiết lập huấn luyện TabPFN (sinh dữ liệu tổng hợp từ prior):

1. **Nhân quả so với không nhân quả (Hình 2a).** Sinh 20 tập tổng hợp với 1000 mẫu kiểm tra cố định và số mẫu huấn luyện thay đổi (đến 3000). Mô hình không nhân quả tiếp tục cải thiện khi nhiều mẫu hơn; mô hình nhân quả chỉ cải thiện ở khoảng mẫu rất nhỏ rồi **bão hòa và suy giảm**. Gợi ý: attention nhân quả không tận dụng được ngữ cảnh dài, phù hợp các nghiên cứu về LLM (Ding 2024, Gong 2023).
2. **Mamba so với Transformer (Hình 2b).** Thay lớp transformer bằng Mamba-II, huấn luyện cùng quy trình, đánh giá trên 150 tập kiểm chứng của TabPFN. Mamba có **loss huấn luyện cao hơn rõ rệt** và **AUC trung bình thấp hơn nhiều**.
3. **Softmax so với linear (Hình 2c, Hình 8).** Thay attention bằng linear attention rồi so sánh với TabPFN trên 57 tập thật (Bảng 2 của McElfresh 2023). Chênh lệch độ chính xác trung bình rất nhỏ (khoảng −0,003 theo biểu đồ mật độ), trong khi thời gian chạy giảm đáng kể.
4. **So với sliding window attention (Phụ lục B.1, Bảng 4):** linear attention vượt rõ trên ba tập lớn (poker-hand, airlines, higgs); ví dụ trên higgs sliding window chỉ đạt 0,39 so với 0,76 của linear (đọc theo thứ tự cột của bảng).

**Phát hiện 1:** tính nhân quả của SSM cản trở ICL. **Phát hiện 2:** linear attention không nhân quả giữ chất lượng và tăng tốc.

## 5. TabFlex

### 5.1. Phân tích tính toán
Dao et al. (2022) cho thấy tốc độ attention bị chi phối bởi số lần đọc/ghi giữa HBM và SRAM của GPU. Bài phân tích cùng tiêu chí cho **FlashLinearAttention (nhân quả)** và linear attention không nhân quả:
- Cả hai cần **5ND truy cập HBM**, **4ND bộ nhớ HBM** và **O(ND²) FLOPs** (Bổ đề 2, 3, 4), với N là độ dài chuỗi, D là chiều embedding.
- Cài đặt PyTorch hai dòng (hai einsum: kv = kᵀv rồi q·kv) tốn thêm 2D² truy cập và D² bộ nhớ, tức vẫn O(ND) (Định lý 1). Kết luận: **không cần tối ưu thêm**, dùng cài đặt PyTorch đơn giản. Đo thực nghiệm (số đầu 32–256, số head 2–16, độ dài 2⁴–2¹⁵) cho thấy thời gian và bộ nhớ CUDA tăng tuyến tính theo độ dài.

### 5.2. Ba mô hình và thuật toán chọn mô hình

| Mô hình | Độ dài prompt | Số đặc trưng | Số lớp | Dùng cho |
|---|---|---|---|---|
| TabFlex-S100 | 1152 (như TabPFN) | 100 | 10 | Dữ liệu nhỏ, thấp chiều |
| TabFlex-L100 | 50K | 100 | 10 | Dữ liệu lớn, thấp chiều |
| TabFlex-H1K | 50K | 1000 | 100 | Dữ liệu lớn/nhiều chiều |

Lý do tách thành ba mô hình: mô hình huấn luyện trên dữ liệu lớn nhiều chiều thường khó tối ưu ở vùng nhỏ.

**Thuật toán 1 (chọn mô hình):**
- n ≥ 3K và d ≤ 100 → L100.
- d > 100, hoặc (d/n ≥ 0,2 và n ≥ 3K) → H1K; nếu d > 1000 thì **chiếu ngẫu nhiên** xuống 1000 đặc trưng trước.
- còn lại → S100.

Phụ lục C.4 (Bảng 8) cho thấy hiệu năng **không nhạy với ngưỡng**: các biến thể đều tốt trên dữ liệu đơn giản và dữ liệu lớn thấp chiều.

### 5.3. Huấn luyện (Phụ lục C.2, Bảng 6)
Embedding 512 (như TabPFN); mở rộng số nơ-ron lớp chiếu đầu để nhận nhiều đặc trưng; learning rate 3e-5; batch 1210/110/1410 (S100/L100/H1K), 8/4/4 epoch, 8192/8192/1024 bước mỗi epoch; huấn luyện trên một GPU A100 80GB. **TabFlex-H1K cần hơn ba tuần huấn luyện.** Tập kiểm chứng gồm 40 tập từ OpenML AutoML Benchmark, mỗi khoảng kích thước [0,1K, 1K), [1K, 10K), [10K, 100K) gồm 10 tập (Hình 3 trong bài trình bày kết quả trên các tập này: TabFlex vừa nhanh hơn vừa AUC cao hơn TabPFN).

## 6. Thiết lập thí nghiệm

- **Bộ dữ liệu:** 115 tập OpenML. Bao gồm 98 tập đơn giản (Bảng 1 McElfresh 2023) và 57 tập nhỏ (Bảng 2 McElfresh 2023, kích thước ≤ 1250), cùng **TabZilla hard** (36 tập khó, trong đó 11 tập nhiều chiều 100–2000 đặc trưng và ≥ 50K mẫu).
- **Baselines (25):** 5 kinh điển (RF, SVM, mô hình tuyến tính, KNN, cây quyết định), 3 GBDT (XGBoost, CatBoost, LightGBM), 10 mạng nơ-ron không-Transformer (SAINT, ResNet, DANet, NODE, MLP, MLP-rtdl, DeepFM, STG, VIME, NAM), 4 Transformer (TabPFN, FT-Transformer, TabNet, TabTransformer), cộng TuneTables và HyperFast. Kết quả baseline (trừ TabPFN) lấy từ McElfresh 2023 (V100); TabFlex chạy trên A100, nên có thể lệch nhẹ.
- **Cách xử lý để TabPFN chạy được:** nếu hơn 100 đặc trưng thì chọn ngẫu nhiên đặc trưng; nếu hơn 10 lớp thì chỉ đánh giá 9 lớp phổ biến nhất và coi các lớp còn lại là "other" (bị tính sai). TabPFN còn chỉ dùng 3000 mẫu huấn luyện theo McElfresh.
- **Chuẩn đo:** AUC chuẩn hóa; 10 lần chia train/test; trung bình, trung vị; thời gian tổng (huấn luyện + kiểm tra) chia cho tổng số mẫu ("thời gian/1000 mẫu").

## 7. Kết quả

### 7.1. Tập nhỏ đơn giản (Bảng 1, 57 tập; Bảng 11, 98 tập)
Xếp hạng theo trung vị AUC chuẩn hóa (đọc từ bảng): TabPFN 0,97; **TabFlex 0,96**; CatBoost 0,95; ResNet 0,93; SAINT 0,93; Random Forest 0,92; XGBoost 0,91; HyperFast 0,91; ... Cây quyết định, mô hình tuyến tính, KNN nằm thấp hơn nhiều. Kết luận: TabFlex **đạt top-2** với hiệu năng gần như TabPFN và **nhanh hơn khoảng 2 lần**; so với các phương pháp nhanh hơn (Decision Tree, Linear Model, LightGBM, KNN) thì hiệu năng của chúng thấp rõ rệt. Trên 98 tập đơn giản TabFlex top-3.

### 7.2. Tập khó TabZilla hard (Hình 4, Bảng 2)
- Chỉ TabFlex, TabPFN và XGBoost chạy được trên **toàn bộ** các tập; nhiều phương pháp khác lỗi do không scale. Trong nhóm xử lý dưới 0,5 giây/1000 mẫu, TabFlex nhanh hơn TabPFN (≈ 2×) và nhanh hơn XGBoost, chỉ kém XGBoost một chút về AUC trung vị.
- Trên 11 tập nhiều chiều và lớn, TabFlex **vượt hẳn TabPFN** (vì dùng toàn bộ dữ liệu huấn luyện thay vì 3000 mẫu) và cạnh tranh với "hạng 5" của các baseline (báo cáo 5th-best AUC và 5th-best thời gian vì đa số baseline không có kết quả đầy đủ).
- Với poker-hand (hơn 1 triệu mẫu): TabFlex phân loại toàn bộ trong **4,88 giây**, trong khi phương pháp nhanh thứ năm cần hơn 500 giây (504,52 giây trong bảng).

### 7.3. Mở rộng sang hồi quy (Bảng 3)
Rời rạc hóa biến mục tiêu thành 10 hoặc 100 khoảng đều (chọn cấu hình tốt hơn), áp dụng trên 18 tập hồi quy số của Grinsztajn (cpu_act, pol, elevators, wine_quality, houses, diamonds, ...) và so với hồi quy tuyến tính và XGBoost Regressor (100 cây, độ sâu 6, mặc định scikit-learn). Tác giả mô tả "hiệu năng hợp lý" dù không phải mục tiêu chính của bài; mình không dùng được con số từng ô vì bảng bị lệch cột.

### 7.4. Ảnh (Phụ lục D.6, Bảng 14)
Coi ảnh là bảng: MNIST, Fashion-MNIST, CIFAR-10 (làm phẳng 3072 chiều) và CIFAR-10 (embedding ResNet-18 pretrained 384 chiều). So với MLP hai/ba lớp (10 nơ-ron ẩn, 70 epoch) và ResNet 2 khối: TabFlex nhanh hơn MLP khoảng 30 lần và ResNet khoảng 400 lần với hiệu năng cạnh tranh. CIFAR-10 làm phẳng đạt AUC khoảng 0,791 trong vài giây; giảm chiều còn 30% bằng PCA làm độ trễ giảm 4 lần mà AUC giữ nguyên; dùng embedding đưa AUC lên **0,922** và độ trễ giảm 6 lần.

## 8. Ablation

- **Đối chiếu tinh với XGBoost trên dữ liệu tổng hợp (Hình 5).** XGBoost cấu hình độ sâu 3, 20 cây; số đặc trưng 100–1000; số mẫu 1000–12000; trung bình 20 tập. TabFlex tốt hơn cả về độ chính xác lẫn độ trễ khi số đặc trưng **dưới 800**; khoảng cách thu hẹp dần, và **tại 800 đặc trưng XGBoost bắt đầu vượt**. Dù vậy tradeoff tổng thể của TabFlex vẫn tốt ở đa số cấu hình.
- **Giảm chiều (Hình 6).** PCA, SVD, chiếu ngẫu nhiên trên 8 tập nhiều chiều (dna, musk, scene, jasmine, semeion, SpeedDating, hill-valley, mfeat-factors): có thể **giảm đến 10% số chiều**, AUC giữ nguyên hoặc tốt hơn, độ trễ giảm hơn 2 lần.
- **Lấy mẫu dữ liệu (Hình 7).** K-centers, K-medoids, uncertainty sampling, ngẫu nhiên trên 9 tập (cmc, kc1, car, yeast, ...) với n > 1000 và d < 100: chỉ cần khoảng **20% dữ liệu huấn luyện** để giữ hiệu năng, độ trễ giảm đáng kể.
- **Số mẫu (Hình 11).** Độ chính xác tăng đều theo số mẫu, thời gian chạy tăng tuyến tính, với 800 hoặc 1000 đặc trưng.
- **So với TuneTables (Phụ lục D.5, Bảng 13):** kết quả TuneTables trong bài chính dùng bản mặc định; bài còn đối chiếu số từ bài gốc (30 vòng tìm siêu tham số cho cả hai). TabFlex vẫn cạnh tranh, nhất là khi ít mẫu, còn TuneTables tốt hơn khi nhiều mẫu nhờ cập nhật tham số nhưng chậm hơn nhiều.

## 9. Hạn chế

Tác giả nêu:
- Mới tập trung **phân loại**; hồi quy chỉ là cách rời rạc hóa thô, cần cách có nguyên tắc hơn (huấn luyện trên dữ liệu hồi quy tổng hợp).
- Cần mở rộng cho không gian đặc trưng lớn hơn hai nghìn chiều; thử nghiệm ảnh mới ở quy mô nhỏ; âm thanh/video cần chiến lược sinh dữ liệu tổng hợp mới.
- Chưa kiểm tra tác động khi áp dụng linear attention cho TabPFNv2.

Nhận xét thêm của mình:
1. Dựa trên **TabPFN v1**, trong khi TabPFNv2 và TabICL đã vượt xa về độ chính xác; so sánh trực tiếp với hai mô hình này không có.
2. TabFlex-H1K cần hơn **ba tuần** huấn luyện trên một A100, và hiệu năng tốt trên dữ liệu lớn nhiều chiều phụ thuộc vào một thuật toán chọn mô hình thủ công.
3. Baseline lấy từ McElfresh 2023 (V100) còn TabFlex chạy trên A100, nên so sánh thời gian chưa hoàn toàn đồng nhất.
4. Cách đánh giá TabPFN với hơn 10 lớp (gộp thành "other" và coi là sai) có thể làm TabPFN bị bất lợi giả tạo.
5. Không có thí nghiệm về nhiễu nhãn, mất cân bằng lớp hay hiệu chỉnh xác suất.

## 10. Liên hệ với bài Nature và đề tài

- Đây là một cách giải **điểm yếu "không scale"** mà bài Nature nêu (TabPFNv2 giới hạn khoảng 10.000 mẫu), khác hướng TabICL (kiến trúc hai giai đoạn) và LoCalPFN (truy xuất cục bộ). Có thể xếp ba hướng: đổi attention (TabFlex), đổi kiến trúc (TabICL), đổi ngữ cảnh (LoCalPFN, TabDPT).
- Bằng chứng gián tiếp cho thấy **phần lớn lợi thế của TFM là từ ICL không nhân quả trên toàn tập**, điều TabICLv2 cũng dựa vào.
- Với đề tài nhiễu/mất cân bằng: TabFlex dùng toàn bộ dữ liệu huấn luyện làm ngữ cảnh, nên nhiễu nhãn trong dữ liệu lớn sẽ đi thẳng vào ngữ cảnh. Chưa có thí nghiệm nào kiểm tra.

## 11. Từ khóa cần nhớ

- **Linear attention không nhân quả** (ϕ = elu + 1) thay softmax; O(N) theo độ dài chuỗi.
- **Phát hiện:** SSM/Mamba nhân quả kém ICL; non-causal tốt hơn khi nhiều mẫu.
- **Ba mô hình:** S100 (1152, 100 đặc trưng, 10 lớp), L100 (50K), H1K (50K, 1K đặc trưng, 100 lớp); chọn theo n, d.
- **Số liệu:** nhanh hơn TabPFN trên 2×, nhanh hơn XGBoost 1,5×; poker-hand 1M mẫu trong 4,88 giây; 115 tập OpenML, 25 baseline; XGBoost vượt tại khoảng 800 đặc trưng; 20% dữ liệu hoặc 10% chiều vẫn giữ hiệu năng; H1K huấn luyện hơn 3 tuần.
