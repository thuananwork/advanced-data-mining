# Bài 23. Random Forests

| Mục | Thông tin |
|---|---|
| Nơi đăng | Machine Learning, 45(1), 5–32 (2001); PDF là bản thảo đề ngày tháng 1/2001 |
| Tác giả | Leo Breiman (Statistics Department, University of California, Berkeley) |
| Mã nguồn / dữ liệu | Không có mã; dữ liệu là 13 bộ nhỏ + 3 bộ lớn trong UCI, 4 bộ tổng hợp (mô tả ở Breiman 1996, Schapire et al. 1997) cho phân loại; 8 bộ cho hồi quy |
| Vị trí trong đề tài | Nền tảng của mọi phương pháp cây ensemble (Random Forest là baseline trong TabArena, TabZilla, Grinsztajn, AMLB). Cũng là bằng chứng sớm rằng ensemble ngẫu nhiên **bền với nhãn nhiễu hơn boosting**. |

## 1. Bài này làm gì trong một đoạn

Bài định nghĩa **random forest**: một tập các cây quyết định h(x, Θ_k) với Θ_k là các vector ngẫu nhiên độc lập, cùng phân phối; mỗi cây bỏ một phiếu cho lớp phổ biến nhất tại x. Breiman (1) chứng minh sai số tổng quát hóa **hội tụ gần như chắc chắn** khi số cây tăng nên rừng **không quá khớp khi thêm cây**; (2) chặn trên sai số tổng quát hóa theo **độ mạnh (strength) s của từng cây** và **tương quan trung bình ρ** giữa các cây: PE\* ≤ ρ̄(1 − s²)/s²; (3) đề xuất hai biến thể thực dụng dùng **chọn đặc trưng ngẫu nhiên tại mỗi nút** (Forest-RI) và **tổ hợp tuyến tính ngẫu nhiên của đầu vào** (Forest-RC), kết hợp với bagging và ước lượng **out-of-bag (OOB)**; (4) cho thấy chúng tương đương độ chính xác Adaboost nhưng **bền với nhiễu nhãn** hơn nhiều, nhanh hơn, và cho các ước lượng nội bộ (lỗi, strength, tương quan, tầm quan trọng biến).

## 2. Bối cảnh và động cơ (mục 1)

- Gộp nhiều cây rồi biểu quyết cải thiện độ chính xác phân loại. Các cách tạo ngẫu nhiên trước đó: **bagging** (Breiman 1996; chọn mẫu huấn luyện ngẫu nhiên), **random split selection** (Dietterich 1998; chọn tách ngẫu nhiên trong K tách tốt nhất), **random output** (Breiman 1998: nhiễu hóa nhãn), trọng số ngẫu nhiên trên mẫu, **random subspace** (Ho 1998: chọn tập con đặc trưng), và công trình nhận dạng ký tự của Amit & Geman (1997): tìm kiếm điểm tách trên một lựa chọn ngẫu nhiên của số lớn đặc trưng hình học, là công trình ảnh hưởng nhiều nhất đến tác giả.
- Điểm chung: với cây thứ k, sinh Θ_k độc lập với Θ_1..Θ_{k−1} nhưng cùng phân phối, rồi mọc cây từ tập huấn luyện và Θ_k. Trong bagging, Θ là số lần mỗi mẫu được chọn (ném N phi tiêu vào N ô).
- **Định nghĩa 1.1:** random forest là bộ phân loại gồm tập các cây {h(x, Θ_k)} với Θ_k i.i.d., mỗi cây bỏ một phiếu cho lớp phổ biến nhất tại x.
- Các rừng cũ (bagging, random split, random output) đều kém Adaboost và các thuật toán "arcing" (tái trọng số thích nghi). Câu hỏi của bài: làm sao chèn ngẫu nhiên để đạt độ chính xác ngang Adaboost.

## 3. Lý thuyết đặc trưng độ chính xác (mục 2)

- **Margin** của tập phân loại: mg(X,Y) = avg_k I(h_k(X) = Y) − max_{j≠Y} avg_k I(h_k(X) = j); sai số tổng quát hóa PE\* = P_{X,Y}(mg(X,Y) < 0).
- **Định lý 1.2** (chứng minh ở Phụ lục I, dùng Luật số lớn): khi số cây tăng, với hầu như mọi chuỗi Θ_1, Θ_2, ..., PE\* hội tụ về P_{X,Y}(P_Θ(h(X,Θ) = Y) − max_{j≠Y} P_Θ(h(X,Θ) = j) < 0). Giải thích **vì sao rừng ngẫu nhiên không quá khớp** khi thêm cây: chỉ tiến tới một giá trị giới hạn. Ý chứng minh: với tập huấn luyện và Θ cố định, tập {x : h(Θ,x) = j} là hợp các hyper-rectangle; có hữu hạn K hợp như vậy, rồi áp dụng Luật số lớn cho từng tần suất N_k/N.
- **Strength và tương quan (Định nghĩa 2.1–2.4):**
  - margin của rừng: mr(X,Y) = P_Θ(h(X,Θ)=Y) − max_{j≠Y} P_Θ(h(X,Θ)=j); **strength** s = E_{X,Y} mr(X,Y).
  - Nếu s ≥ 0, bất đẳng thức Chebyshev cho PE\* ≤ var(mr)/s².
  - Đặt ĵ(X,Y) = argmax_{j≠Y} P_Θ(h(X,Θ)=j) và **raw margin** rmg(Θ,X,Y) = I(h(X,Θ)=Y) − I(h(X,Θ)=ĵ(X,Y)); mr là kỳ vọng của rmg theo Θ.
  - Dùng đẳng thức [E_Θ f(Θ)]² = E_{Θ,Θ'} f(Θ)f(Θ') với Θ, Θ' độc lập cùng phân phối, suy ra var(mr) = E_{Θ,Θ'}(ρ(Θ,Θ')·sd(Θ)·sd(Θ')) = ρ̄ (E_Θ sd(Θ))² ≤ ρ̄ E_Θ var(Θ), với ρ̄ là tương quan trung bình có trọng số giữa các raw margin.
  - Vì E_Θ var(Θ) ≤ 1 − s², ta được:
  - **Định lý 2.3:** **PE\* ≤ ρ̄ (1 − s²) / s².**
  - **Tỉ số c/s² = ρ̄/s²** là chỉ dẫn hữu ích: càng nhỏ càng tốt. Tác giả thừa nhận chặn "có thể lỏng" nhưng có vai trò gợi ý như các chặn kiểu VC.
- Hai thành phần cốt lõi: **độ mạnh của cây riêng lẻ** và **mức tương quan giữa các cây**. Ngẫu nhiên được chèn vào phải **giảm ρ̄ mà vẫn giữ strength**.
- **Hai lớp:** margin là mr = 2·P_Θ(h(X,Θ)=Y) − 1; yêu cầu s > 0 tương tự điều kiện weak learning E P_Θ(h = Y) > 0,5; ρ̄ là tương quan giữa hai cây khác nhau.
- **Nhiều lớp:** strength theo định nghĩa (3) phụ thuộc vào cả rừng. Cách khác: PE\* ≤ Σ_j P(P_Θ(h=Y) − P_Θ(h=j) < 0), định nghĩa strength theo lớp s_j (không phụ thuộc rừng), rồi chặn bằng Σ_j var(·)/s_j² (phương trình 9). Tác giả không ước lượng được các đại lượng này trong thực nghiệm và cho rằng sẽ thú vị với bài đa lớp.

## 4. Chèn đặc trưng ngẫu nhiên (mục 3) và OOB

- Rừng bàn tới gồm: chọn **đầu vào hoặc tổ hợp đầu vào ngẫu nhiên** ở mỗi nút để mọc cây. Ưu điểm tác giả nêu: (i) chính xác ngang hoặc hơn Adaboost; (ii) **tương đối bền với outlier và nhiễu**; (iii) nhanh hơn bagging hay boosting; (iv) cho ước lượng nội bộ của lỗi, strength, tương quan, tầm quan trọng biến; (v) đơn giản, dễ song song hóa.
- **Dùng bagging cùng lúc** với chọn đặc trưng ngẫu nhiên: mỗi tập huấn luyện mới lấy có hoàn lại từ tập gốc; cây mọc không cắt tỉa. Hai lý do: bagging tăng độ chính xác khi dùng đặc trưng ngẫu nhiên, và cho phép ước lượng liên tục sai số tổng quát hóa, strength, tương quan.
- **Ước lượng out-of-bag:** với mỗi mẫu (y,x) của tập huấn luyện, chỉ gộp phiếu của các cây mà tập bootstrap T_k **không chứa** mẫu đó; sai số của "bộ phân loại OOB" trên tập huấn luyện là ước lượng OOB của sai số tổng quát hóa. Các nghiên cứu trước (Tibshirani 1996; Wolpert & Macready 1996; Breiman 1996b) cho thấy OOB chính xác như một tập kiểm tra cùng cỡ tập huấn luyện, nên **không cần giữ tập kiểm tra riêng**. Khoảng **một phần ba** mẫu bị bỏ khỏi mỗi bootstrap, nên ước lượng OOB chỉ dùng khoảng 1/3 số cây, vì thế có xu hướng **đánh giá cao (overestimate)** lỗi hiện hành; để không chệch cần chạy quá điểm hội tụ của sai số tập kiểm tra. Khác với CV, OOB không chệch (trong sự khác biệt này tác giả nói CV có chệch mức độ không biết).
- **Phụ lục II – ước lượng OOB cho strength và tương quan:** đặt Q(x,j) = Σ_k I(h(x,Θ_k)=j; (y,x) ∉ T_{k,B}) / Σ_k I((y,x) ∉ T_{k,B}), tỉ lệ phiếu OOB cho lớp j tại x (ước lượng P_Θ(h(x,Θ)=j)). Strength ước lượng = trung bình trên tập huấn luyện của Q(x,y) − max_{j≠y} Q(x,j). var(mr) ước lượng bằng trung bình của (Q(x,y) − max_{j≠y}Q(x,j))² trừ s² (phương trình A1); sd(Θ) = [p₁ + p₂ + (p₁ − p₂)²]^{1/2} (A2), với p₁ = trung bình I(h(x,Θ_k)=y) và p₂ = trung bình I(h(x,Θ_k)=ĵ) trên các mẫu ngoài túi thứ k, rồi lấy trung bình trên k; từ đó ρ̄ = var(mr)/(E_Θ sd(Θ))².

## 5. Forest-RI: chọn đầu vào ngẫu nhiên (mục 4)

- **Forest-RI:** ở mỗi nút chọn ngẫu nhiên nhóm F biến đầu vào để tách; cây mọc theo CART đến kích thước tối đa, **không cắt tỉa**. Thử hai giá trị: F = 1 và F = số nguyên đầu tiên nhỏ hơn log₂M + 1 (M là số đầu vào).
- **Dữ liệu (Bảng 1):** 13 bộ nhỏ UCI (glass 214, breast cancer 699, diabetes 768, sonar 208, vowel 990, ionosphere 351, vehicle 846, soybean 685, German credit 1000, image 2310, ecoli 336, votes 435, liver 345; số đầu vào từ 6 đến 60; số lớp 2 đến 19), 3 bộ lớn có train/test tách sẵn (letters 15000/5000, 16 đầu vào, 26 lớp; sat-images 4435/2000, 36 đầu vào, 6 lớp; zip-code 7291/2007, 256 đầu vào, 10 lớp) và 4 bộ tổng hợp (waveform 3 lớp, twonorm, threenorm, ringnorm; mỗi bộ 300 mẫu huấn luyện, 3000 kiểm tra, 20–21 đầu vào).
- **Giao thức cho 13 bộ nhỏ:** giữ ngẫu nhiên 10% làm kiểm tra; chạy rừng 2 lần (F = 1 và F = int(log₂M + 1)) với 100 cây; chọn lỗi kiểm tra ứng với ước lượng OOB thấp hơn; lặp 100 lần và lấy trung bình. Adaboost kết hợp 50 cây (100 cây rừng vì OOB chỉ dùng 1/3 số cây; mọc 100 cây rừng vẫn nhanh hơn 50 cây Adaboost). Bộ lớn: rừng 100 cây (zip-code 200), Adaboost 50 cây (zip-code 100). Bộ tổng hợp: 50 lần chạy, mỗi lần sinh tập mới 300/3000, 100 cây (rừng) và 50 cây (Adaboost).

**Bảng 2 – Sai số kiểm tra (%), Adaboost vs Forest-RI (Selection = chọn F theo OOB; Single = F = 1; One tree = một cây đơn):**

| Bộ | Adaboost | Selection | Single | One tree |
|---|---|---|---|---|
| glass | 22,0 | 20,6 | 21,2 | 36,9 |
| breast cancer | 3,2 | 2,9 | 2,7 | 6,3 |
| diabetes | 26,6 | 24,2 | 24,3 | 33,1 |
| sonar | 15,6 | 15,9 | 18,0 | 31,7 |
| vowel | 4,1 | 3,4 | 3,3 | 30,4 |
| ionosphere | 6,4 | 7,1 | 7,5 | 12,7 |
| vehicle | 23,2 | 25,8 | 26,4 | 33,1 |
| German credit | 23,5 | 24,4 | 26,2 | 33,3 |
| image | 1,6 | 2,1 | 2,7 | 6,4 |
| ecoli | 14,8 | 12,8 | 13,0 | 24,5 |
| votes | 4,8 | 4,1 | 4,6 | 7,4 |
| liver | 30,7 | 25,1 | 24,7 | 40,6 |
| letters | 3,4 | 3,5 | 4,7 | 19,8 |
| sat-images | 8,8 | 8,6 | 10,5 | 17,2 |
| zip-code | 6,2 | 6,3 | 7,8 | 20,6 |
| waveform | 17,8 | 17,2 | 17,3 | 34,0 |
| twonorm | 4,9 | 3,9 | 3,9 | 24,7 |
| threenorm | 18,8 | 17,5 | 17,5 | 38,4 |
| ringnorm | 6,9 | 4,9 | 4,9 | 25,7 |

- Kết luận: sai số dùng chọn đầu vào ngẫu nhiên **ngang Adaboost** (thắng ở nhiều bộ, thua ở vehicle, German credit, image, ionosphere). Không nhạy với F: chênh sai số giữa F = 1 và giá trị F cao hơn trung bình dưới 1%, rõ nhất ở 3 bộ lớn. Điều đáng ngạc nhiên: **chỉ chọn một biến ngẫu nhiên để tách ở mỗi nút vẫn cho độ chính xác tốt**. (Lưu ý: khi hai giá trị F cho sai số gần nhau, chọn theo OOB gần như ngẫu nhiên, nên ở vài ô Selection có thể kém hơn Single.)
- **Tốc độ:** tỉ lệ thời gian Forest-RI so với mọc cây không cắt tỉa dùng mọi biến = F·log₂(N)/M. Zip-code với F = 1: tỉ lệ 0,025, tức **nhanh gấp 40 lần**; thực tế 100 cây Forest-RI (F = 1) chạy **4,0 phút** trên Macintosh 250 MHz, so với **gần ba giờ** cho Adaboost.

## 6. Forest-RC: tổ hợp tuyến tính ngẫu nhiên (mục 5)

- Khi có ít đầu vào (M nhỏ), tăng F lên gần M làm tăng strength nhưng cũng tăng tương quan. Cách khác: tạo thêm đặc trưng bằng **tổ hợp tuyến tính ngẫu nhiên**: tại mỗi nút chọn ngẫu nhiên L biến, cộng lại với hệ số ngẫu nhiên đều trên [−1, 1]; sinh F tổ hợp rồi tìm tách tốt nhất. Dùng **L = 3 và F = 2 hoặc 8** (chọn bằng OOB); L = 3 vì có O(M³) tổ hợp nên F lớn không làm tăng tương quan nhiều trong khi tăng strength. Đầu vào không cùng thang đo được chuẩn hóa theo trung bình và độ lệch chuẩn của tập huấn luyện.

**Bảng 3 – Sai số kiểm tra (%), Forest-RC (Selection = F theo OOB; Two features = F = 2):**

| Bộ | Adaboost | Selection | Two features | One tree |
|---|---|---|---|---|
| glass | 22,0 | 24,4 | 23,5 | 42,4 |
| breast cancer | 3,2 | 3,1 | 2,9 | 5,8 |
| diabetes | 26,6 | 23,0 | 23,1 | 32,1 |
| sonar | 15,6 | 13,6 | 13,8 | 31,7 |
| vowel | 4,1 | 3,3 | 3,3 | 30,4 |
| ionosphere | 6,4 | 5,5 | 5,7 | 14,2 |
| vehicle | 23,2 | 23,1 | 22,8 | 39,1 |
| German credit | 23,5 | 22,8 | 23,8 | 32,6 |
| image | 1,6 | 1,6 | 1,8 | 6,0 |
| ecoli | 14,8 | 12,9 | 12,4 | 25,3 |
| votes | 4,8 | 4,1 | 4,0 | 8,6 |
| liver | 30,7 | 27,3 | 27,2 | 40,3 |
| letters | 3,4 | 3,4 | 4,1 | 23,8 |
| sat-images | 8,8 | 9,1 | 10,2 | 17,3 |
| zip-code | 6,2 | 6,2 | 7,2 | 22,7 |
| waveform | 17,8 | 16,0 | 16,1 | 33,2 |
| twonorm | 4,9 | 3,8 | 3,9 | 20,9 |
| threenorm | 18,8 | 16,8 | 16,9 | 34,8 |
| ringnorm | 6,9 | 4,8 | 4,6 | 24,6 |

- F = 8 chỉ cần cho 3 bộ lớn; với các bộ khác F = 2 gần đạt cực tiểu. Forest-RC **đặc biệt tốt trên bộ tổng hợp** và nhìn chung so sánh tốt hơn với Adaboost so với Forest-RI.
- **Cải thiện cho bộ lớn:** giả thuyết strength vẫn tăng trên bộ lớn còn tương quan sớm bão hòa. Chạy F = 100 (100, 100, 200 cây): sat-images giảm xuống **8,5%**, letters **3,0%**, zip-code không giảm; thử F = 25 với Forest-RI trên zip-code được **5,8%**. Đây là mức thấp nhất của ensemble cây khi đó trên ba bộ này.

### Biến hạng mục (5.1)
- Mỗi lần biến hạng mục được chọn để tách tại nút, chọn một **tập con ngẫu nhiên các hạng mục** và định nghĩa biến thay thế bằng 1 nếu giá trị thuộc tập con, 0 nếu ngoài. Biến hạng mục I giá trị mã thành I − 1 biến giả 0/1 nên được cho xác suất chọn lớn hơn biến số I − 1 lần. Khi nhiều biến hạng mục, F thấp cho tương quan thấp nhưng strength thấp; phải tăng F lên khoảng 2–3 lần int(log₂M + 1).
- Ví dụ: **DNA** (60 biến hạng mục 4 giá trị, 2000 mẫu huấn luyện, 1186 kiểm tra): Forest-RI F = 20 cho sai số **3,6%** (Adaboost 4,2%); **soybean** (685 mẫu, 35 biến, 19 lớp, 15 biến hạng mục): Forest-RI F = 12 cho 5,3% (Adaboost 5,8%), Forest-RC với tổ hợp 3 và F = 8 cho 5,5%.
- Ưu điểm: tránh được bài toán biến hạng mục có nhiều giá trị. Với hai lớp có thể dùng thủ thuật của Breiman et al. (1985) giảm tìm kiếm tách xuống O(I); với nhiều lớp tìm kiếm tách tốt nhất là O(2^{I−1}). Cài đặt rừng chỉ chọn một tập con ngẫu nhiên hạng mục.

## 7. Strength và tương quan trong thực nghiệm (mục 6)

- Chạy Forest-RI trên **sonar** (60 đầu vào, 208 mẫu) với F từ 1 đến 50; mỗi F lấy 100 cây; lặp 80 lần, mỗi lần giữ 10% làm kiểm tra; tổng cộng 400.000 cây (Hình 1).
  - Quá F ≈ 4, **strength gần như hằng số**: thêm đầu vào không giúp; **tương quan vẫn tăng**.
  - Sai số kiểm tra và OOB đều giảm nhẹ từ F = 1 đến F ≈ 4–8, rồi tăng dần, trùng khớp với vùng strength hằng. OOB ổn định hơn.
- **Breast** với tổ hợp 3 đầu vào, số đặc trưng 1..25 (Hình 2): tương quan tăng chậm, strength hầu như không đổi, sai số nhỏ nhất ở F = 1.
- **Satellite** (tổ hợp tổng của 2 đầu vào, 1..25 đặc trưng, Hình 3): khác bộ nhỏ — cả strength và tương quan **đều tăng nhẹ và đều**, sai số giảm nhẹ. Giả thuyết: với bộ lớn và phức tạp, strength tiếp tục tăng lâu hơn trước khi bão hòa.
- Kết luận: rừng tốt có **tương quan thấp hơn và strength cao hơn**; ngẫu nhiên khi mọc cây phải nhắm đến ρ thấp mà vẫn giữ strength hợp lý. Khớp với Dietterich (1998) (ensemble chính xác hơn có độ phân tán lớn hơn), Freund (cho rằng Adaboost mỗi bước cố gắng tách rời bộ phân loại kế tiếp khỏi bộ hiện tại) và Amit et al. (1999) (Adaboost giữ hiệp phương sai giữa các bộ phân loại nhỏ).

## 8. Giả thuyết: Adaboost là một random forest (mục 7)

- Định nghĩa rừng với K tập trọng số không âm, tổng bằng 1 trên mẫu huấn luyện w^(1), ..., w^(K), xác suất p^(k); Θ rút theo p; nếu Θ = k thì mọc phân loại với trọng số w^(k).
- Thí nghiệm: chạy Adaboost 75 lần trên một bộ dữ liệu, thu các tập trọng số (bỏ 25 tập đầu); đặt xác suất của tập trọng số thứ k tỉ lệ với Q(w_k) = log[(1 − error(k))/error(k)] (error(k) là sai số có trọng số w^(k) của phân loại thứ k); chạy rừng 250 lần. Lặp 100 lần trên vài bộ, mỗi lần giữ 10% làm kiểm tra. Kết quả: sai số Adaboost **rất gần** sai số rừng; ví dụ Wisconsin Breast Cancer: Adaboost **2,91%**, rừng **2,94%**.
- **Giả thuyết ergodic:** Adaboost có w^(k+1) = φ(w^(k)); với toán tử Tf(w) = f(φ(w)) giả định T ergodic với độ đo bất biến π(dw), phiếu bầu chuẩn hóa hội tụ về E_{Q_π}[I(h(x,w) = j)], với Q_π(dw) ∝ Q(w)π(dw). Nếu đúng thì Adaboost tương đương rừng với trọng số rút từ Q_π, giải thích **vì sao Adaboost không quá khớp khi thêm cây** (kinh nghiệm của Breiman: chạy đến 1000 cây, sai số kiểm tra hội tụ về giá trị tiệm cận; Grove & Schuurmans 1998 báo cáo quá khớp nếu chạy hàng nghìn vòng nhưng với base classifier rất đơn giản). Giả thuyết **không** giải thích được vì sao Adaboost chọn được các phân phối trọng số tốt; và phân phối trọng số của Adaboost phụ thuộc tập huấn luyện, khác với rừng thông thường.

## 9. Ảnh hưởng của nhiễu nhãn (mục 8)

- Dietterich (1998): khi một phần nhãn huấn luyện bị đổi ngẫu nhiên, độ chính xác Adaboost suy giảm còn bagging và random split selection bền hơn. Thí nghiệm theo Dietterich: giữ 10% làm kiểm tra; hai lần chạy trên phần còn lại, một trên tập sạch, một trên tập bị nhiễu: **5% nhãn bị đổi** ngẫu nhiên sang nhãn khác (chọn đều trong các nhãn còn lại). Lặp 50 lần với Adaboost (bản tất định), Forest-RI và Forest-RC; tính **phần trăm tăng sai số**. Chỉ dùng 9 bộ nhỏ nhất vì chạy lâu.

**Bảng 4 – Tăng sai số do nhiễu nhãn 5% (%):**

| Bộ | Adaboost | Forest-RI | Forest-RC |
|---|---|---|---|
| glass | 1,6 | 0,4 | −0,4 |
| breast cancer | 43,2 | 1,8 | 11,1 |
| diabetes | 6,8 | 1,7 | 2,8 |
| sonar | 15,1 | −6,6 | 4,2 |
| ionosphere | 27,7 | 3,8 | 5,7 |
| soybean | 26,9 | 3,2 | 8,5 |
| ecoli | 7,5 | 7,9 | 7,8 |
| votes | 48,9 | 6,3 | 4,6 |
| liver | 10,3 | −0,2 | 4,8 |

- Adaboost suy giảm rõ (đến +43,2% ở breast cancer, +48,9% ở votes), còn các rừng ngẫu nhiên thường thay đổi nhỏ. Hai bộ đa lớp (glass, ecoli) và diabetes ít chịu ảnh hưởng ở Adaboost; ở ecoli cả ba tương đương (+7,5, +7,9, +7,8).
- **Giải thích của tác giả:** Adaboost tăng dần trọng số cho các mẫu vừa bị phân loại sai; mẫu có nhãn sai sẽ liên tục bị phân loại sai nên Adaboost tập trung trọng số ngày càng lớn vào các mẫu nhiễu và bị "méo". Rừng ngẫu nhiên không tập trung trọng số vào bất kỳ tập con nào nên hiệu ứng nhiễu nhỏ hơn.

## 10. Dữ liệu có nhiều đầu vào yếu (mục 9)

- Bộ dữ liệu mô phỏng 10 lớp, 1000 đầu vào nhị phân (xác suất p(j,k) = 0,2·rnd + 0,01, cộng thêm 0,4·rnd tại một số vị trí ngẫu nhiên, tối đa ~400 vị trí mỗi lớp), N = 1000 mẫu huấn luyện và 4000 mẫu kiểm tra. Tổng vị trí có xác suất cao ~2000 nên các lớp chồng lấn đáng kể. **Sai số Bayes = 1,0%**; **Naive Bayes** (tối ưu vì các đầu vào độc lập) có sai số 6,2% (tác giả lưu ý không phải ủng hộ Naive Bayes).

| Cấu hình | Sai số kiểm tra | Strength | Tương quan | c/s² | Sai số cây đơn trung bình |
|---|---|---|---|---|---|
| Forest-RI, F = 1 (dừng ở 2500 vòng, chưa hội tụ) | 10,7% | 0,069 | 0,012 | 2,5 | 80% |
| Forest-RI, F = 10 (hội tụ sau 2000 vòng) | 3,0% | 0,22 | 0,045 | 0,91 | 65% |
| Forest-RI, F = 25 (dừng ở 2000 vòng) | 2,8% | 0,28 | 0,065 | 0,83 | 60% |

- Rừng đạt sai số **không xa Bayes** dù từng cây rất yếu: rừng làm việc được với phân loại yếu miễn là **tương quan thấp**. Adaboost **không chạy được** trên dữ liệu này vì base classifier quá yếu.

## 11. Cơ chế rừng: tầm quan trọng biến (mục 10)

- **Permutation importance (trên OOB):** sau khi mọc mỗi cây, hoán vị ngẫu nhiên giá trị biến thứ m trong các mẫu OOB và cho chúng chạy qua cây đó; lặp cho m = 1..M. Cuối cùng so sánh phiếu đa số OOB (khi biến m bị "nhiễu hóa") với nhãn thật để tính tỉ lệ phân loại sai; đầu ra là **phần trăm tăng** so với tỉ lệ sai OOB khi giữ nguyên mọi biến. Một lần chạy rừng 1000 cây, không cần tập kiểm tra.
- **Diabetes** (F = 1): biến 2 quan trọng nhất, tiếp theo biến 8 và 6. Chạy lại chỉ với biến 2: sai số **29,7%** (tất cả biến: 23,1%); thêm biến 8: 29,4%; thêm biến 6: **26,4%**. Giải thích: biến 8 mang cùng thông tin với biến 2 nên khi đã có biến 2 thì biến 8 không giảm sai số thêm; khi hai biến x₁, x₂ giống hệt nhau và đều có thông tin, mỗi biến bị chọn với tần suất như nhau nên hoán vị từng biến cho cùng mức tăng sai số, nhưng thêm biến thứ hai sau biến thứ nhất không giảm sai số. Thứ tự tương đối ổn định khi dùng tổ hợp 3 đầu vào với F = 2 (Hình 5).
- **Votes** (435 nghị sĩ, 16 phiếu yes/no; F = 5 với đầu vào đơn, 1000 cây): biến 4 nổi bật, **sai số tăng gấp ba** khi biến 4 bị nhiễu hóa. Chạy lại chỉ với biến 4 cho sai số **4,3%**, tương đương dùng cả 16 biến. Tác giả nói đây mới là bước khởi đầu.

## 12. Random forest cho hồi quy (mục 11–12)

- Cây h(x, Θ) nhận giá trị số; sai số bình phương trung bình tổng quát hóa E_{X,Y}(Y − h(X))². Rừng dự đoán trung bình các cây.
- **Định lý 11.1:** khi số cây → ∞, hầu chắc chắn E_{X,Y}(Y − avg_k h(X,Θ_k))² → E_{X,Y}(Y − E_Θ h(X,Θ))² =: PE\*(forest). Định nghĩa PE\*(tree) = E_Θ E_{X,Y}(Y − h(X,Θ))².
- **Định lý 11.2:** nếu với mọi Θ, EY = E_X h(X,Θ), thì **PE\*(forest) ≤ ρ̄·PE\*(tree)**, với ρ̄ là tương quan có trọng số giữa các phần dư Y − h(X,Θ) và Y − h(X,Θ'). Chứng minh qua E_Θ E_Θ' (ρ·sd·sd) / (E_Θ sd(Θ))². Hệ quả: rừng giảm sai số trung bình của cây theo hệ số ρ̄; cần **phần dư tương quan thấp** và **cây có sai số thấp**.
- **Dữ liệu hồi quy (Bảng 5):** Boston Housing (12 đầu vào, 506 mẫu), Ozone (8, 330), Servo (4, 167), Abalone (8, 4177, 25% làm kiểm tra), Robot Arm (12, 15000 huấn luyện, 5000 kiểm tra), Friedman#1 (10 đầu vào), #2 (4) và #3 (4) — mỗi bộ Friedman 200 mẫu huấn luyện và 2000 kiểm tra. Bộ Peak20 bị loại vì bất thường. Với 3 bộ vừa: giữ 10% kiểm tra, lặp 100 lần; Abalone: lặp 10 lần với 25%.
- **Thiết lập:** bagging + chọn đặc trưng ngẫu nhiên, mỗi đặc trưng là tổ hợp tuyến tính ngẫu nhiên của **2 đầu vào**, dùng **25 đặc trưng** mỗi nút, 100 cây, quy tắc không tách nếu nút nhỏ hơn 5 mẫu.

**Bảng 6 – MSE kiểm tra:**

| Bộ | Bagging | Adaptive bagging | Forest |
|---|---|---|---|
| Boston Housing | 11,4 | 9,7 | 10,2 |
| Ozone | 17,8 | 17,8 | 16,3 |
| Servo ×10⁻² | 24,5 | 25,1 | 24,6 |
| Abalone | 4,9 | 4,9 | 4,6 |
| Robot Arm ×10⁻² | 4,7 | 2,8 | 4,2 |
| Friedman #1 | 6,3 | 4,1 | 5,7 |
| Friedman #2 ×10³ | 21,5 | 21,5 | 19,6 |
| Friedman #3 ×10⁻³ | 24,8 | 24,8 | 21,6 |

- Rừng luôn tốt hơn hoặc bằng bagging (trừ Servo, chênh nhẹ: 24,6 vs 24,5); khi adaptive bagging cải thiện mạnh (Boston, Robot Arm, Friedman#1) thì rừng cải thiện ít hơn; khi adaptive bagging không cải thiện (Ozone, Abalone, Friedman#2/#3) thì rừng cải thiện.
- Khác với phân loại, ở hồi quy tương quan tăng **khá chậm** khi tăng số đặc trưng; tác động chính là giảm PE\*(tree), nên cần số đặc trưng tương đối lớn. Nếu quá ít, PE\*(tree) quá cao; quá nhiều, tương quan cao; khoảng ở giữa thường rộng.

**Bảng 7 – Lỗi và ước lượng OOB (Lỗi kiểm tra, lỗi OOB, PE\*(tree) OOB, tương quan):**

| Bộ | Test | OOB | PE\*(tree) | Tương quan |
|---|---|---|---|---|
| Boston Housing | 10,2 | 11,6 | 26,3 | 0,45 |
| Ozone | 16,3 | 17,6 | 32,5 | 0,55 |
| Servo ×10⁻² | 24,6 | 27,9 | 56,4 | 0,56 |
| Abalone | 4,6 | 4,6 | 8,3 | 0,56 |
| Robot Arm ×10⁻² | 4,2 | 3,7 | 9,1 | 0,41 |
| Friedman #1 | 5,7 | 6,3 | 15,3 | 0,41 |
| Friedman #2 ×10³ | 19,6 | 20,4 | 40,7 | 0,51 |
| Friedman #3 ×10⁻³ | 21,6 | 22,9 | 48,3 | 0,49 |

  Lỗi OOB luôn cao hơn lỗi kiểm tra như dự kiến, ngoại trừ Robot Arm (thấp hơn), mà tác giả cho là artifact do tập huấn luyện/kiểm tra tách riêng.
- **Thay bagging bằng nhiễu hóa đầu ra** (cộng nhiễu Gauss kỳ vọng 0, độ lệch chuẩn bằng độ lệch chuẩn của đầu ra):

| Bộ | Với bagging | Với nhiễu đầu ra |
|---|---|---|
| Boston Housing | 10,2 | 9,1 |
| Ozone | 17,8 | 16,3 |
| Servo ×10⁻² | 24,6 | 23,2 |
| Abalone | 4,6 | 4,7 |
| Robot Arm ×10⁻² | 4,2 | 3,9 |
| Friedman #1 | 5,7 | 5,1 |
| Friedman #2 ×10³ | 19,6 | 20,4 |
| Friedman #3 ×10⁻³ | 21,6 | 19,8 |

  Lỗi trên Boston và Ozone là thấp nhất khi đó; nhìn chung nhiễu đầu ra + đặc trưng ngẫu nhiên tốt hơn bagging. (Lưu ý: ô "Ozone, với bagging" là 17,8 ở bảng này nhưng 16,3 ở Bảng 6/7: có thể là lỗi in của bản thảo; file này giữ nguyên số theo PDF.)

## 13. Kết luận của tác giả (mục 13)

Rừng ngẫu nhiên là công cụ dự đoán hiệu quả: không quá khớp nhờ Luật số lớn, chính xác khi chèn đúng loại ngẫu nhiên; khung strength–tương quan giúp hiểu cơ chế; OOB cụ thể hóa các đại lượng lý thuyết. Trước đây người ta cho rằng rừng không đấu được với thuật toán arcing; kết quả bác bỏ điều đó, dẫn đến câu hỏi mới: boosting và arcing giảm được **cả bias lẫn variance**; rừng **cũng có vẻ giảm bias** nhưng cơ chế chưa rõ (tác giả đùa rằng nếu nhìn rừng như quy trình Bayes giải thích được thì "tôi có thể trở thành Bayesian hơn"). Chỉ dùng hai kiểu ngẫu nhiên (bagging và đặc trưng ngẫu nhiên); có thể các kiểu khác (tổ hợp Boolean ngẫu nhiên do một phản biện đề xuất) tốt hơn. Kết hợp đặc trưng ngẫu nhiên với boosting có thể cho kết quả tốt hơn: sai số thấp tới **5,1%** zip-code, **2,2%** letters, **7,9%** satellite. Breiman (2000) chứng minh trong không gian phân phối với hai lớp, rừng tương đương một kernel tác động lên true margin (ngẫu nhiên/tương quan thấp giữ tính đối xứng của kernel, strength tạo độ lệch mong muốn ở biên cong).

## 14. Hạn chế (tác giả nêu và suy ra)
1. Chặn PE\* ≤ ρ̄(1 − s²)/s² **có thể lỏng** (tác giả nói thế); chỉ có giá trị gợi ý.
2. Ước lượng OOB **chệch về phía lỗi cao** vì chỉ dùng ~1/3 số cây; để không chệch phải chạy quá điểm hội tụ.
3. Strength cho đa lớp phụ thuộc vào rừng; các ước lượng theo lớp (phương trình 9) **không được ước lượng thực nghiệm**.
4. Thử nghiệm đã cũ (dữ liệu UCI nhỏ, 100 cây, máy 250 MHz); không có bộ lớn cỡ hàng trăm nghìn mẫu hay đa chiều rất cao như bảng hiện đại; kết quả tốt hơn Adaboost không đồng đều (thua ở vehicle, German credit, image, ionosphere với Forest-RI).
5. Chọn giá trị F bằng OOB khi hai giá trị gần nhau gần như ngẫu nhiên; một số ô Selection kém hơn Single.
6. Tầm quan trọng biến **chỉ là bước đầu**: biến tương quan nhau chia sẻ độ quan trọng (diabetes biến 6, 8), không thể diễn giải riêng; tác giả nói cần nghiên cứu thêm.
7. **Nhiễu nhãn** chỉ thử một mức duy nhất (5%) trên 9 bộ nhỏ, chỉ nhiễu đối xứng ngẫu nhiên; không thử mất cân bằng lớp hay mức nhiễu cao hơn.
8. Hồi quy: kết quả hỗn hợp so với adaptive bagging; rừng ít hiệu quả hơn ở hồi quy so với phân loại; mức ngẫu nhiên đã dùng (tổ hợp 2 biến) chưa chứng tỏ là tối ưu.
9. Giả thuyết Adaboost = random forest vẫn chỉ là **giả thuyết** (T ergodic chưa được chứng minh).
10. Không bàn đến cây trên dữ liệu bảng hạng mục cao cardinality như CatBoost (bài 26).

## 15. Liên hệ với bài Nature và đề tài
- Random Forest là baseline mặc định trong TabArena và là thuật toán bị bài Nature so sánh; cây ensemble dựa vào ý tưởng bagging + đặc trưng ngẫu nhiên của bài này. Lý thuyết **strength × tương quan** đối nghịch với TFM: TabPFN thực chất là "ensemble" qua các hoán vị đặc trưng/lớp và bootstrap (như TabPFN v2 dùng 8–16 ensemble member) nên có thể diễn giải theo cùng khung giảm tương quan.
- **Bài này cung cấp bằng chứng sớm duy nhất về robustness với nhãn nhiễu (5% nhãn đổi, mục 8 và Bảng 4)** trong nhóm bài đã đọc: Random Forest bền hơn Adaboost, vì Adaboost tập trung trọng số vào mẫu bị gán nhãn sai. Đây là gợi ý trực tiếp cho thí nghiệm nhãn nhiễu ở final-term: cần so sánh TFM với XGBoost/CatBoost (cũng là boosting, dễ nhạy nhiễu hơn bagging) ở các mức nhiễu đa dạng. Bài chỉ thử một mức nhiễu, nhưng cũng cho thấy sự khác biệt boosting vs bagging là quan trọng.
- Permutation importance (mục 10) là nền tảng cho giải thích mô hình khi cần ở phần phân tích (feature importance của TFM có thể tính bằng cách tương tự).

## 16. Từ khóa cần nhớ
- **PE\* ≤ ρ̄(1 − s²)/s²**; strength s, tương quan ρ̄, tỉ số **c/s²**; rừng **hội tụ**, không quá khớp.
- **Forest-RI** (F = 1 hoặc int(log₂M + 1)) và **Forest-RC** (L = 3 biến, F = 2 hoặc 8, hệ số đều [−1,1]); bagging + OOB (khoảng 1/3 mẫu bị bỏ).
- Nhiễu nhãn 5%: Adaboost +43,2% (breast cancer), +48,9% (votes); rừng thường < 12%.
- Sai số zip-code **5,8%** (F = 25); bộ 1000 biến yếu: **2,8%** vs Bayes 1,0%, Naive Bayes 6,2%.
- Hồi quy: **PE\*(forest) ≤ ρ̄·PE\*(tree)**.
- **Permutation importance** trên OOB.
