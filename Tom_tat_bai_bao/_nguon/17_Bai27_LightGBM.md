# Bài 27. LightGBM: A Highly Efficient Gradient Boosting Decision Tree

| Mục | Thông tin |
|---|---|
| Nơi đăng | NIPS 2017 (31st Conference on Neural Information Processing Systems, Long Beach, CA, USA) |
| Tác giả | Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (Microsoft Research; Peking University; Microsoft Redmond) |
| Mã nguồn | github.com/Microsoft/LightGBM |
| Lưu ý về PDF | Bản PDF chỉ có phần thân bài (8 trang). Chứng minh Định lý 3.2, Mệnh đề 2.1, chi tiết tham số và các đường cong huấn luyện còn lại nằm ở tài liệu bổ sung (supplementary), **không có trong PDF**, nên file này không tóm tắt chúng. |
| Vị trí trong đề tài | Một trong ba GBDT chuẩn (XGBoost, CatBoost, LightGBM) mà các bài TFM dùng làm baseline. Bài này nhắm vào **tốc độ và quy mô**, không nhắm vào độ chính xác trên dữ liệu nhỏ. |

## 1. Bài này làm gì trong một đoạn

Các cài đặt GBDT như XGBoost và pGBRT đã tối ưu nhiều về kỹ thuật, nhưng khi số đặc trưng lớn và dữ liệu nhiều thì vẫn chậm, vì với **mỗi đặc trưng** thuật toán phải quét **mọi mẫu** để ước lượng information gain của mọi điểm tách. Bài đề xuất hai kỹ thuật: **GOSS** (Gradient-based One-Side Sampling), bỏ bớt các mẫu có gradient nhỏ và chỉ dùng phần còn lại để ước lượng gain, và **EFB** (Exclusive Feature Bundling), gom các đặc trưng gần như loại trừ nhau (hiếm khi cùng khác 0) thành một đặc trưng để giảm số đặc trưng. GBDT cài đặt GOSS + EFB gọi là **LightGBM**; thí nghiệm cho thấy nó **nhanh hơn GBDT thông thường tới hơn 20 lần** mà độ chính xác gần như không đổi.

## 2. Vấn đề và động cơ

- GBDT là mô hình ensemble các cây quyết định huấn luyện tuần tự; mỗi vòng, cây mới khớp **gradient âm** (residual). Phần tốn kém nhất là tìm điểm tách tốt nhất.
- Hai thuật toán tìm điểm tách phổ biến: **pre-sorted** (liệt kê mọi điểm tách trên giá trị đặc trưng đã sắp; đơn giản, tìm được tách tối ưu nhưng chậm và tốn bộ nhớ) và **histogram-based** (gom giá trị liên tục vào các bin rời rạc, dựng histogram đặc trưng; nhanh và tiết kiệm hơn). Bài xây dựng trên histogram-based.
- Thuật toán 1 (histogram): dựng histogram tốn O(#data × #feature), tìm điểm tách tốn O(#bin × #feature). Vì #bin ≪ #data nên **dựng histogram chiếm phần lớn chi phí**; muốn nhanh phải giảm #data hoặc #feature.
- Vì sao khó giảm:
  - **Giảm mẫu:** các phương pháp lấy mẫu theo trọng số (lọc mẫu có trọng số dưới ngưỡng, điều chỉnh tỉ lệ lấy mẫu động) đều dựa trên AdaBoost, mà GBDT **không có trọng số mẫu** nên không áp dụng trực tiếp. SGB (Stochastic Gradient Boosting, dùng một tập con ngẫu nhiên mỗi vòng) áp dụng được nhưng thường làm giảm độ chính xác.
  - **Giảm đặc trưng:** lọc đặc trưng yếu (PCA, projection pursuit) giả định đặc trưng dư thừa nhiều, điều không phải lúc nào cũng đúng, vì mỗi đặc trưng thường được thiết kế để đóng góp riêng.
  - **Thưa:** GBDT pre-sorted tận dụng được tính thưa bằng cách bỏ qua giá trị 0 (như XGBoost), nhưng histogram-based phải lấy bin của mỗi mẫu cho dù giá trị bằng 0 hay không, nên chưa có cách khai thác tính thưa hiệu quả.

## 3. GOSS: lấy mẫu một phía theo gradient (mục 3)

### 3.1. Thuật toán
- Quan sát: mẫu có gradient (trị tuyệt đối) nhỏ là mẫu đã được huấn luyện tốt; mẫu gradient lớn (under-trained) **đóng góp nhiều hơn vào information gain**. Bỏ thẳng mẫu gradient nhỏ làm đổi phân phối dữ liệu và hại độ chính xác, nên cần cách bù.
- Các bước (Thuật toán 2):
  1. Sắp mẫu theo |gradient| giảm dần.
  2. Giữ **top a × 100%** mẫu gradient lớn (tập A).
  3. Từ phần còn lại A^c (gồm (1 − a) × 100% mẫu), **lấy ngẫu nhiên b × |A^c|** mẫu, được tập B (b là tỉ lệ so với toàn bộ dữ liệu trong cài đặt thuật toán: randN = b × len(I)).
  4. Khi tính information gain, **nhân trọng số hằng (1 − a)/b** cho các mẫu gradient nhỏ được chọn, để đưa tổng gradient trên B về cỡ của A^c, nhờ đó phân phối gốc ít bị thay đổi.
  5. Huấn luyện cây mới trên A ∪ B với các trọng số tương ứng.

### 3.2. Phân tích lý thuyết
- **Variance gain** (Định nghĩa 3.1) của việc tách đặc trưng j tại d trên nút O:
  V_{j|O}(d) = (1/n_O) · [ (Σ_{x_i∈O: x_ij ≤ d} g_i)² / n^j_{l|O}(d) + (Σ_{x_i∈O: x_ij > d} g_i)² / n^j_{r|O}(d) ].
- **Ước lượng GOSS:** Ṽ_j(d) = (1/n) · [ (Σ_{A_l} g_i + ((1−a)/b) Σ_{B_l} g_i)² / n^j_l(d) + (tương tự cho phía phải) ], trong đó A_l, A_r, B_l, B_r là phần trái/phải của A và B.
- **Định lý 3.2** (chứng minh ở supplementary): với xác suất ít nhất 1 − δ, sai số xấp xỉ E(d) = |Ṽ_j(d) − V_j(d)| thỏa
  E(d) ≤ C²_{a,b} ln(1/δ) · max{1/n^j_l(d), 1/n^j_r(d)} + 2D·C_{a,b}·√(ln(1/δ)/n),
  với C_{a,b} = ((1−a)/√b) · max_{x_i∈A^c}|g_i| và D = max(ḡ^j_l(d), ḡ^j_r(d)).
- Nhận xét của tác giả:
  1. Sai số tiệm cận là O(1/n^j_l(d) + 1/n^j_r(d) + 1/√n). Nếu phép tách không quá mất cân bằng (n^j_l(d), n^j_r(d) ≥ O(√n)) thì **số hạng thứ hai chi phối** và sai số giảm về 0 với tốc độ O(1/√n) khi n lớn: xấp xỉ rất chính xác khi dữ liệu nhiều.
  2. **Lấy mẫu ngẫu nhiên là trường hợp riêng của GOSS với a = 0.** GOSS tốt hơn lấy mẫu ngẫu nhiên khi C_{0,β} > C_{a,β−a}, tương đương α_a/√β > √(1−a)/√(β−a), với α_a = max_{x_i∈A∪A^c}|g_i| / max_{x_i∈A^c}|g_i|.
- **Tổng quát hóa:** sai số tổng quát hóa của GOSS ≤ sai số xấp xỉ GOSS + sai số tổng quát hóa khi dùng toàn bộ dữ liệu, nên nếu xấp xỉ chính xác thì khả năng tổng quát hóa gần như không đổi; lấy mẫu còn có thể tăng độ đa dạng của các base learner, có lợi cho tổng quát hóa.

## 4. EFB: gom đặc trưng loại trừ nhau (mục 4)

- Dữ liệu cao chiều thực tế rất **thưa**; nhiều đặc trưng gần như **loại trừ nhau** (không bao giờ khác 0 cùng lúc), ví dụ các đặc trưng one-hot. Có thể gom chúng thành **một đặc trưng** mà vẫn dựng được cùng histogram như từ các đặc trưng riêng lẻ; độ phức tạp dựng histogram giảm từ O(#data × #feature) xuống O(#data × #bundle), với #bundle ≪ #feature.
- Hai vấn đề: (1) đặc trưng nào nên gom chung; (2) gom như thế nào.
- **Định lý 4.1:** chia đặc trưng thành **số bundle loại trừ nhỏ nhất là NP-khó** (quy về bài toán tô màu đồ thị: mỗi hàng ma trận kề của đồ thị là một đặc trưng, một bundle loại trừ tương ứng với tập đỉnh cùng màu). Vì thế dùng thuật toán tham lam xấp xỉ.
- **Thuật toán 3 (Greedy Bundling):** dựng đồ thị có trọng số cạnh (trọng số = tổng số xung đột giữa hai đặc trưng; có cạnh nếu hai đặc trưng không loại trừ hoàn toàn nhau); sắp đặc trưng theo bậc giảm dần; duyệt từng đặc trưng, gán vào bundle hiện có nếu số xung đột không vượt K (điều khiển bởi tỉ lệ xung đột tối đa **γ**), nếu không thì tạo bundle mới. Độ phức tạp O(#feature²), chạy một lần trước huấn luyện. Khi số đặc trưng rất lớn, thay bậc bằng **số giá trị khác 0** (nhiều giá trị khác 0 thì dễ xung đột hơn), không cần dựng đồ thị.
- **Cho phép xung đột nhỏ:** nhiều đặc trưng không hoàn toàn loại trừ nhưng hiếm khi khác 0 cùng lúc. Việc "làm ô nhiễm ngẫu nhiên" một phần nhỏ giá trị ảnh hưởng độ chính xác huấn luyện tối đa **O([(1 − γ)n]^(−2/3))** (Mệnh đề 2.1, supplementary), với γ là tỉ lệ xung đột tối đa mỗi bundle. Chọn γ nhỏ thì cân bằng được giữa độ chính xác và hiệu quả.
- **Thuật toán 4 (Merge Exclusive Features):** vì histogram lưu bin rời rạc, có thể gộp bằng cách **cộng offset** để các đặc trưng nằm ở các dải bin khác nhau. Ví dụ A ∈ [0, 10), B ∈ [0, 20): cộng 10 vào B để B ∈ [10, 30); bundle nhận giá trị trong [0, 30]. Giá trị gốc vẫn nhận ra được từ bundle.
- **Tối ưu thưa cơ bản:** có thể thêm bảng ghi các mẫu khác 0 cho mỗi đặc trưng để chi phí dựng histogram giảm từ O(#data) xuống O(#non_zero_data), nhưng phải tốn thêm bộ nhớ và tính toán để duy trì các bảng này suốt quá trình mọc cây. LightGBM cài đặt tối ưu này như một chức năng cơ bản, **không mâu thuẫn với EFB** (vẫn dùng được khi bundle còn thưa).

## 5. Thực nghiệm (mục 5)

### 5.1. Thiết lập
- Máy chủ Linux, 2 CPU E5-2670 v3 (24 lõi), RAM 256 GB, đa luồng, **16 luồng**.
- **5 bộ dữ liệu công khai (Bảng 1):**

| Bộ | #data | #feature | Kiểu | Tác vụ | Metric |
|---|---|---|---|---|---|
| Allstate Insurance Claim | 12 M | 4.228 | Thưa | Phân loại nhị phân | AUC |
| Flight Delay | 10 M | 700 | Thưa | Phân loại nhị phân | AUC |
| LETOR (Microsoft Learning to Rank; 30K truy vấn web) | 2 M | 136 | Dày | Xếp hạng | NDCG |
| KDD CUP 2010 | 19 M | 29 M | Thưa | Phân loại nhị phân | AUC |
| KDD CUP 2012 | 119 M | 54 M | Thưa | Phân loại nhị phân | AUC |

  LETOR gồm chủ yếu đặc trưng số; Allstate và Flight Delay có nhiều đặc trưng one-hot; hai bộ KDD dùng đặc trưng của giải thắng cuộc (NTU), gồm cả dày và thưa, rất lớn.
- **Baseline:** `xgb_exa` (XGBoost pre-sorted), `xgb_his` (XGBoost histogram), `lgb_baseline` (LightGBM không GOSS và EFB), `SGB` (lgb_baseline với Stochastic Gradient Boosting, cùng tỉ lệ lấy mẫu với LightGBM), `EFB_only` (lgb_baseline + EFB). Cây mọc **leaf-wise** cho xgb_his, lgb_baseline và LightGBM; xgb_exa chỉ hỗ trợ layer-wise nên được chỉnh tham số để tạo cây tương tự.
- **Tham số:** a = b = 0,05 cho Allstate, KDD10 và KDD12; a = b = 0,1 cho Flight Delay và LETOR; γ = 0 trong EFB. Tất cả thuật toán chạy số vòng cố định; lấy độ chính xác ở vòng có điểm tốt nhất. Chi tiết tham số nằm ở supplementary.

### 5.2. Kết quả chính

**Thời gian trung bình huấn luyện một vòng (giây) — Bảng 2:**

| Bộ | xgb_exa | xgb_his | lgb_baseline | EFB_only | LightGBM |
|---|---|---|---|---|---|
| Allstate | 10,85 | 2,63 | 6,07 | 0,71 | **0,28** |
| Flight Delay | 5,94 | 1,05 | 1,39 | 0,27 | **0,22** |
| LETOR | 5,55 | 0,63 | 0,49 | 0,46 | **0,31** |
| KDD10 | 108,27 | OOM | 39,85 | 6,33 | **2,85** |
| KDD12 | 191,99 | OOM | 168,26 | 20,23 | **12,67** |

**Độ chính xác trên tập kiểm tra (AUC; riêng LETOR là NDCG@10) — Bảng 3:**

| Bộ | xgb_exa | xgb_his | lgb_baseline | SGB | LightGBM |
|---|---|---|---|---|---|
| Allstate | 0,6070 | 0,6089 | 0,6093 | 0,6064 ± 7e-4 | 0,6093 ± 9e-5 |
| Flight Delay | 0,7601 | 0,7840 | 0,7847 | 0,7780 ± 8e-4 | 0,7846 ± 4e-5 |
| LETOR | 0,4977 | 0,4982 | 0,5277 | 0,5239 ± 6e-4 | 0,5275 ± 5e-4 |
| KDD10 | 0,7796 | OOM | 0,78735 | 0,7759 ± 3e-4 | 0,78732 ± 1e-4 |
| KDD12 | 0,7029 | OOM | 0,7049 | 0,6989 ± 8e-4 | 0,7051 ± 5e-5 |

- LightGBM nhanh nhất và độ chính xác gần như bằng baseline: **so với lgb_baseline, tăng tốc 21×, 6×, 1,6×, 14× và 13×** lần lượt trên Allstate, Flight Delay, LETOR, KDD10, KDD12. So với xgb_his (không chạy được trên KDD10 và KDD12 vì hết bộ nhớ, OOM), LightGBM vẫn nhanh hơn, tối đa **9×** trên Allstate. Tăng tốc tính theo thời gian mỗi vòng vì các thuật toán hội tụ sau số vòng tương tự. Hình 1 và 2 vẽ đường cong thời gian–AUC (Flight Delay) và thời gian–NDCG (LETOR).
- Đáng chú ý trên LETOR: LightGBM và lgb_baseline đạt khoảng 0,527–0,528, cao hơn rõ rệt hai bản XGBoost (khoảng 0,498); bài không giải thích chênh lệch này (có thể do cách mọc cây hoặc xử lý ranking, bài không nêu).

### 5.3. Phân tích GOSS và EFB
- **GOSS (so với EFB_only trong Bảng 2):** riêng GOSS mang lại khoảng **2× tăng tốc** khi chỉ dùng 10–20% dữ liệu. Không tăng tốc tuyến tính theo tỉ lệ mẫu vì vẫn phải dự đoán và tính gradient trên toàn bộ dữ liệu.
- **GOSS so với SGB (Bảng 4, LETOR, cùng tỉ lệ lấy mẫu, chạy đến hội tụ với early stopping):** GOSS luôn tốt hơn SGB:

| Tỉ lệ mẫu | 0,1 | 0,15 | 0,2 | 0,25 | 0,3 | 0,35 | 0,4 |
|---|---|---|---|---|---|---|---|
| SGB | 0,5182 | 0,5216 | 0,5239 | 0,5249 | 0,5252 | 0,5263 | 0,5267 |
| GOSS | 0,5224 | 0,5256 | 0,5275 | 0,5284 | 0,5289 | 0,5293 | 0,5296 |

  Phù hợp với phân tích ở mục 3.2. Cài đặt a, b cho từng mức nằm ở supplementary; độ lệch chuẩn nhỏ.
- **EFB (γ = 0, so lgb_baseline với EFB_only):** tăng tốc lớn trên các bộ quy mô lớn, **dù lgb_baseline đã tối ưu cho đặc trưng thưa**. Lý do: EFB gộp nhiều đặc trưng thưa (one-hot lẫn các đặc trưng loại trừ ngầm) thành ít đặc trưng hơn nhiều, không tốn chi phí duy trì bảng giá trị khác 0 trong lúc mọc cây, và vì nhiều đặc trưng được gộp nên **tăng tính cục bộ không gian, tăng tỉ lệ cache hit**. Ví dụ: KDD10 từ 39,85 giây xuống 6,33 giây mỗi vòng; Allstate từ 6,07 xuống 0,71; KDD12 từ 168,26 xuống 20,23. Nghiên cứu chi tiết cho γ nằm ở supplementary.

## 6. Kết luận của tác giả
LightGBM với GOSS và EFB vượt XGBoost và SGB rõ rệt về **tốc độ tính toán và bộ nhớ**, kết quả thực nghiệm khớp với lý thuyết. Hướng tương lai: nghiên cứu cách chọn tối ưu a và b trong GOSS, và cải tiến EFB để xử lý số đặc trưng rất lớn, cho dù thưa hay không.

## 7. Hạn chế (tác giả nêu và suy ra)
1. Mục tiêu chính là **hiệu quả tính toán**, nhắm vào dữ liệu lớn (hàng triệu mẫu, hàng nghìn đến hàng chục triệu đặc trưng). Bài **không đánh giá dữ liệu nhỏ**, nơi TFM cạnh tranh; trên dữ liệu nhỏ GOSS và EFB ít có ý nghĩa.
2. Chỉ **5 bộ dữ liệu**, toàn bộ cực lớn; không có hồi quy, không có đa lớp, không có bộ dữ liệu bảng cỡ nhỏ chuẩn; chỉ so với XGBoost (và biến thể của chính mình), không so với CatBoost hay mô hình khác.
3. Chỉ báo một lần chạy cho nhiều ô bảng (không có độ lệch chuẩn cho các baseline XGBoost); kết quả "bằng độ chính xác" là sai khác rất nhỏ chứ không phải kiểm định thống kê.
4. a và b phải chọn thủ công theo bộ dữ liệu (0,05 hoặc 0,1), tác giả thừa nhận cách chọn tối ưu là việc tương lai.
5. EFB: đồ thị O(#feature²) và bài toán bundle tối ưu là NP-khó (chỉ có xấp xỉ tham lam); với hàng triệu đặc trưng phải dùng cách sắp theo số giá trị khác 0.
6. Đánh giá tăng tốc theo **thời gian mỗi vòng**, không phải thời gian đến hội tụ; không đo hiệu năng dự đoán trên dữ liệu có nhiễu nhãn, mất cân bằng hay hiệu chuẩn.
7. Cách xử lý đặc trưng hạng mục của LightGBM (không thuộc nội dung PDF này) được bài CatBoost (bài 26) phê bình là tốn thời gian và bộ nhớ, và gom các hạng mục đuôi.

## 8. Liên hệ với bài Nature và đề tài
- LightGBM là baseline GBDT nhanh nhất; trong thí nghiệm final-term nên dùng khi cần so sánh thời gian, nhưng **bài này không chứng minh độ chính xác cao hơn trên dữ liệu nhỏ**: ưu điểm của nó là tốc độ.
- Hữu ích cho phần thảo luận chi phí: TFM có chi phí suy luận cao nhưng không cần huấn luyện; GBDT huấn luyện nhanh nhờ các kỹ thuật như GOSS/EFB, tạo nên trục so sánh thời gian huấn luyện – độ chính xác (như AMLB bài 2 và TabArena bài 5).
- Mở rộng ý bài 26: các GBDT (XGBoost, LightGBM, CatBoost) khác nhau ở cách giải quyết tốc độ, thống kê (prediction shift) và hạng mục, nên kết quả "GBDT" trong bài TFM tùy thuộc phiên bản được chọn.

## 9. Từ khóa cần nhớ
- **GOSS:** giữ top a mẫu gradient lớn, lấy ngẫu nhiên b mẫu còn lại, bù trọng số **(1 − a)/b**; random sampling là trường hợp a = 0; sai số xấp xỉ **O(1/√n)**.
- **EFB:** đặc trưng loại trừ → bundle bằng offset bin; bài toán bundle tối ưu **NP-khó** (quy về tô màu đồ thị); tham lam, xung đột tối đa γ, ảnh hưởng độ chính xác ≤ O([(1−γ)n]^(−2/3)).
- Histogram-based, leaf-wise; tăng tốc **21× / 6× / 1,6× / 14× / 13×**; xgb_his OOM trên KDD10/KDD12.
