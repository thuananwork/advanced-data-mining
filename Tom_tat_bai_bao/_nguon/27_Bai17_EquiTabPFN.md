# Bài 17. EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Network

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2025; arXiv:2502.06684v4 [cs.LG], 7/1/2026 |
| Tác giả | Michael Arbel (INRIA), David Salinas (Univ. Freiburg, ELLIS Institute Tübingen), Frank Hutter (Univ. Freiburg, ELLIS Institute Tübingen, PriorLabs); Arbel và Salinas đồng tác giả chính |
| Mã nguồn | Có công bố (đường dẫn bị cắt trong bản PDF) |
| Lưu ý về PDF | Các bảng kết quả tổng hợp (Bảng 4, 5, 6) và phần lớn số liệu trong hình (Hình 4, 5) bị lệch cột hoặc chỉ có ở dạng hình; mình dùng các số xuất hiện trong văn bản. Phần phụ lục B (siêu tham số huấn luyện) không thấy trong bản trích. Bảng 1–2 chỉ liệt kê mã tác vụ TabZilla. |
| Vị trí trong đề tài | Bài **kiến trúc** về một tính chất đối xứng của TabPFN (bất biến theo hoán vị nhãn lớp), giúp xử lý số lớp tùy ý. Liên quan tới việc TabPFNv2, TabICL xử lý hơn 10 lớp bằng ECOC/phân cấp. |

## 1. Bài này làm gì trong một đoạn

Các PFN (TabPFN và họ hàng) dự đoán xác suất lớp bằng một MLP đầu ra có **kích thước cố định** (ví dụ tối đa 10 lớp) và mã hóa nhãn bằng vector one-hot với **thứ tự lớp cụ thể**, nên dự đoán **thay đổi khi hoán vị thứ tự lớp**, dù về mặt ngữ nghĩa việc gọi lớp nào là "lớp 0" hay "lớp 1" là tùy ý. Tác giả gọi đây là thiếu **tính tương đẳng theo mục tiêu (target equivariance)**, và chứng minh rằng điều này gây ra một **"khe hở tương đẳng" (equivariance gap)** không thể triệt tiêu: sai số bổ sung vào hàm mục tiêu tiền huấn luyện, cùng sự bất ổn dự đoán. TabPFN giảm nhẹ bằng ensemble nhiều hoán vị lớp (tốn O(q!) nếu muốn đầy đủ). EquiTabPFN thiết kế **kiến trúc tương đẳng ngay từ đầu** với ba thành phần: bi-attention (attention xen kẽ giữa các thành phần đặc trưng/mục tiêu và giữa các mẫu), **token dự đoán** thay cho nhãn kiểm tra chưa biết, và **bộ giải mã phi tham số** (trung bình có trọng số các nhãn huấn luyện cộng hiệu chỉnh dư). Kết quả: trên dữ liệu có **số lớp lớn hơn số lớp thấy khi tiền huấn luyện**, EquiTabPFN ngang hoặc vượt các phương pháp hiện có với chi phí tính toán thấp hơn; trên dữ liệu ≤10 lớp thì **ngang TabPFNv2** dù không dùng prior cải tiến của v2 (vì mã prior v2 không công khai).

## 2. Bài toán và bối cảnh

- PFN huấn luyện transformer dự đoán nhãn kiểm tra từ đặc trưng kiểm tra và các cặp (đặc trưng, nhãn) huấn luyện, tất cả lấy từ một mô hình sinh. Attention **theo hàng** giúp bất biến theo thứ tự mẫu. Các nghiên cứu Müller et al. (2025) và Hollmann et al. (2025, TabPFNv2) đã xử lý **số đặc trưng bất kỳ** nhờ bi-attention (xen kẽ attention theo hàng/mẫu và theo cột/đặc trưng) và bất biến theo thứ tự đặc trưng; nhưng **kích thước mục tiêu vẫn cố định** (ví dụ 10 lớp).
- Cách xử lý hiện tại với hơn 10 lớp: chia bài toán nhỏ rồi tổng hợp bằng **ECOC** (TabPFNv2) hoặc **phân cấp** (TabICL), tốn chi phí tính toán vì phải chạy nhiều lần.
- Các hướng liên quan: PFN cho tối ưu Bayes, dự báo, ngoại suy learning curve, phân tích công bằng; MotherNet (tạo trọng số MLP bằng ICL); TabICL (kiến trúc hai giai đoạn); sửa đầu ra TabPFN bằng mô hình cộng tổng quát để dễ giải thích; kết hợp mô hình phi tham số lên TabPFN (cần huấn luyện lúc suy luận); công trình đồng thời Koshil et al. (tương đẳng nhãn nhưng ưu tiên khả năng giải thích, tránh đầu ra phi tuyến); Wu & Bergman (số lớp tùy ý nhưng không tương đẳng, dùng huấn luyện đối kháng). Khái niệm tương đẳng đã dùng trong CNN, mạng cầu, tập, đồ thị, và gần đây cho thứ tự lựa chọn trong LLM.

## 3. Kiến thức nền: TabPFN gốc (v1)

Mô hình f_{X,Y}(X*) nhận bộ huấn luyện (X, Y) và truy vấn X*. Ba mô-đun:
- **Bộ mã hóa tuyến tính:** token huấn luyện e_n = U x_n + V y_n (U: d×p, V: d×q); token kiểm tra e*_m = U x*_m (không có nhãn). Nhãn nhỏ hơn q thì đệm không; lớn hơn q thì không biểu diễn được.
- **Backbone transformer:** self-attention với mặt nạ để mọi token chỉ nhìn token huấn luyện (token kiểm tra xử lý độc lập).
- **Bộ giải mã MLP:** một lớp ẩn, ánh xạ token kiểm tra thành vector dự đoán độ dài q cố định.
Bất biến với thứ tự mẫu nhưng **không tương đẳng theo thành phần mục tiêu**.

## 4. EquiTabPFN

Cho tensor E kích thước (B, N+M, q+1, d): với mỗi mẫu, một token cho vector đặc trưng (qua phép chiếu U, d×p) và q token cho từng thành phần mục tiêu.

1. **Bộ mã hóa tương đẳng theo mục tiêu:** mỗi thành phần (y_n)_j được nhân với **cùng một vector nhúng V** (kích thước d) cho mọi j; tương đương **tích chập 1×1** dọc theo các thành phần mục tiêu, nên hoán vị thành phần chỉ hoán vị token tương ứng. Nhãn của mẫu kiểm tra (không biết) được thay bằng **token dự đoán** có thể học W_pred, lặp M×q lần để tạo ước lượng ban đầu Ỹ⁰.
2. **Hai loại self-attention xen kẽ** với kết nối dư và layer norm, cộng MLP điểm theo từng embedding:
   - *Attention giữa các thành phần* (SelfAtt_c): áp dụng độc lập cho từng mẫu, giữa q+1 token (1 đặc trưng + q mục tiêu) của cùng mẫu; có **mặt nạ**: token mục tiêu chỉ attend vào token đặc trưng, còn token đặc trưng attend vào mọi token (tác giả nói thấy hữu ích bằng thực nghiệm).
   - *Attention giữa các mẫu* (SelfAtt_b): áp dụng độc lập cho từng chiều đầu vào p và q; như TabPFN, token huấn luyện và kiểm tra chỉ attend vào token huấn luyện.
   - Độ phức tạp tuyến tính theo số lớp như TabPFN, nhưng hệ số lớn hơn vì backbone phụ thuộc số lớp (TabPFN ánh xạ về không gian cố định rồi quay lại, backbone độc lập số lớp).
3. **Bộ giải mã phi tham số tương đẳng:** hai bước. Một mô-đun attention tính dự đoán trung gian ỹ_m là **trung bình có trọng số của các nhãn huấn luyện** (trọng số lấy từ tích vô hướng giữa embedding của token huấn luyện và token kiểm tra, cộng trên mọi thành phần i và chiều u, chia (1+q)d); sau đó một MLP điểm cho từng thành phần j **cộng hiệu chỉnh dư**, vẫn tương đẳng. Nếu bỏ hiệu chỉnh dư và gỡ sự phụ thuộc của khóa/truy vấn vào Y thì bộ giải mã trở thành **ước lượng hồi quy phi tham số tuyến tính** (tổng quát hóa của Nadaraya–Watson). Vì ước lượng tuyến tính kém tối ưu hơn phi tuyến, tác giả thêm phụ thuộc phi tuyến vào Y và quan sát cải thiện rõ.
- **Hệ quả:** số lớp bất kỳ ở suy luận; Định lý 5.1 nói kiến trúc này tương đẳng theo mục tiêu.

## 5. Lý thuyết

- **Định nghĩa 5.1 (tương đẳng hoán vị mục tiêu):** f_{X,σ(Y)}(X*) = σ(f_{X,Y}(X*)) với mọi hoán vị σ của các thành phần mục tiêu. Ví dụ: mã hóa "đỏ" hay "xanh" là lớp thứ nhất không được ảnh hưởng dự đoán.
- **Định nghĩa 5.2 (khe hở tương đẳng):** E_equi(f) = L(f) − L(f_equi), trong đó f_equi là hàm đối xứng hóa bằng trung bình trên mọi hoán vị. Khe hở bằng 0 khi và chỉ khi f tương đẳng.
- **Giả thiết:** (A) hàm mất mát điểm ℓ lồi chặt theo đối số đầu và bất biến theo hoán vị (cross-entropy, bình phương thỏa); (B) phân phối dữ liệu bất biến khi hoán vị Y và Y* (đúng với bảng i.i.d. không có thứ tự ưu tiên).
- **Mệnh đề 5.3:** dưới (A), (B), khe hở **không âm** và chỉ bằng 0 nếu f tương đẳng; nếu f* là cực tiểu của L trên mọi hàm đo được thì f* **phải tương đẳng**. Phác chứng minh: viết lại L(f) = E_p E_σ ℓ(σ⁻¹ f_{X,σ(Y)}(X*), Y*), rồi bất đẳng thức Jensen do tính lồi; nếu f* không tương đẳng thì f*_equi cho L thấp hơn, mâu thuẫn tối ưu.
- **Ý nghĩa:** mô hình không tương đẳng phải "dùng một phần khả năng biểu diễn chỉ để học tính tương đẳng", gây lãng phí.

## 6. Thực nghiệm

### 6.1. Bất ổn của PFN do thứ tự lớp
- **Ví dụ đồ chơi:** 9 điểm huấn luyện trên lưới 2D, mỗi điểm một lớp riêng (lấy cảm hứng từ McCarter); dự đoán trên lưới 40² điểm với 3 thứ tự lớp khác nhau. TabPFN và TabPFN-v2 cho **biên quyết định thay đổi rõ rệt** khi đổi thứ tự (TabPFN-v2 đặc biệt nhiễu), còn EquiTabPFN cho kết quả không đổi (Hình 2).
- **Khe hở tương đẳng trong lúc huấn luyện TabPFN (Hình 3):** đo bằng tỉ lệ vi phạm (khác lớp dự đoán khi hoán vị), trên 512 tập lấy từ prior: sai số tương đẳng **giảm rất chậm** theo bước huấn luyện.
- **Ensemble tốn kém:** TabPFN trung bình nhiều hoán vị lớp; đầy đủ cần O(q!) lần gọi mô hình; ngẫu nhiên hóa hội tụ theo 1/√N_ens nhưng phương sai phụ thuộc chiều và khó định lượng; càng nhiều lớp càng cần nhiều thành viên. EquiTabPFN không cần.

### 6.2. Thiết lập
- **Huấn luyện:** như TabPFNv1 (mở rộng mã của Müller et al. 2022), chỉ trên tác vụ **dưới 10 lớp**; khoảng **4 ngày trên một GPU A100 80GB**; khoảng **25 triệu tham số** (khớp baseline).
- **Dữ liệu đánh giá:** TabZilla. **76 tác vụ đa lớp với ≤10 lớp** và **10 tác vụ với >10 lớp** (11 đến 100 lớp; ví dụ audiology 23 lớp, vowel 11, isolet 26, one-hundred-plants-texture 100...).
- **Đối chứng:** baseline TabZilla; **TabPFNv2** (mã trọng số mở, prior cải tiến nhưng **mã prior và mã huấn luyện không công khai**); và **TabPFNv2\*** (cùng kiến trúc v2 nhưng huấn luyện trên prior công khai và mã huấn luyện của bài, để so sánh công bằng).
- Trừ Hình 5 (trái), EquiTabPFN được báo cáo **không ensemble**.

### 6.3. Kết quả chính
1. **Số lớp chưa thấy (Hình 4 phải, 10 tập >10 lớp):** EquiTabPFN có **trung vị độ chính xác tương đối (so với KNN) tốt nhất**; **vượt mạnh TabPFNv2**, vốn kém hơn cả mô hình tuyến tính và random forest về trung vị khi dùng ECOC với số ensemble tối thiểu. Bảng 5, 6 cho thấy cải thiện nhất quán với AUC, accuracy, F1.
2. **Số lớp ≤10 (Hình 4 trái, 76 tập):** EquiTabPFN **ngang TabPFNv2** dù không dùng prior cải tiến; **vượt TabPFNv2\*** ở cả hai nhóm. Tác giả suy ra nó có thể hưởng lợi thêm nếu dùng prior/quy trình của v2.
3. **Tốc độ so với TabPFNv2 khi nhiều lớp:** TabPFNv2 cần ECOC (nhiều mô hình/ensemble) nên chậm đáng kể; với ≤10 lớp thì thời gian hai bên tương đương (không cần ensemble).
4. **Đánh đổi hiệu năng–chi phí (Hình 5):** khi tăng số thành viên ensemble cho cả TabPFNv2 và EquiTabPFN, ensemble chỉ cải thiện **nhẹ** cho TabPFNv2 so với EquiTabPFN không ensemble mà tốn nhiều thời gian; trong các mô hình đứng đầu, EquiTabPFN có đánh đổi tốt nhất.
5. **Ablation (Bảng 3, giảm lỗi tương đối so với TabPFN):** EquiTabPFN đầy đủ **+1,50%**; chỉ thay bộ giải mã tương đẳng (giữ backbone TabPFN) **+0,94%**; chỉ dùng backbone bi-attention với MLP decoder **−0,12%**. Sự kết hợp mới cho cải thiện lớn nhất. Thử backbone TabPFNv2 sửa thành tương đẳng cũng cải thiện so với TabPFNv2\*, nhưng backbone **TabPFNv1 cho kết quả tốt nhất**.
6. **Chi phí tính toán (phụ lục C.2):**
   - ≤10 lớp: EquiTabPFN **chậm hơn khoảng 5 lần** TabPFNv1; >10 lớp: chỉ chậm khoảng 1,3 lần (vì TabPFNv1 cần ensemble).
   - FLOPS (A100, 2000 mẫu, 100 đặc trưng, 10 lớp): EquiTabPFN **566 GFLOPS** so với **76 GFLOPS** của TabPFN (khoảng 7,45 lần); với 15 lớp: 820 so với 456 GFLOPS.
   - Bộ nhớ: tăng, nên dùng batch nhỏ hơn, nhưng độ dài ngữ cảnh tối đa trong thí nghiệm không bị ảnh hưởng.
7. **Biên quyết định nhị phân (Hình 6):** trên 3 tập, biên của EquiTabPFN (không ensemble) **ổn định và trơn**, khác TabPFN.

## 7. Hạn chế

Tác giả nêu:
- **Dữ liệu không tương đẳng:** dữ liệu thứ bậc (ordinal) mà thứ tự lớp có nghĩa; chỉ gặp **5/86** tập đã khảo sát. Muốn xử lý cần sửa prior (hiện tương đẳng) và thêm mã hóa vị trí hoặc chỉ số cột.
- **Chi phí bậc hai theo số chiều mục tiêu** do self-attention giữa các thành phần; vấn đề khi số lớp rất lớn; có thể dùng attention hiệu quả trong tương lai.

Nhận xét thêm của mình:
1. Dựa trên **TabPFNv1** (prior công khai, 25M tham số) nên so sánh với TabPFNv2 (130 triệu tập tổng hợp trong bài Nature) không trực tiếp; tác giả có đưa TabPFNv2\* nhưng chỉ để kiểm soát prior, còn chưa chứng minh lợi thế trên kiến trúc/prior v2 đầy đủ.
2. Thí nghiệm chủ yếu **TabZilla**, nhóm >10 lớp chỉ **10 tập**, nên kết luận "vượt TabPFNv2 khi nhiều lớp" dựa trên ít tập.
3. Hồi quy và các bài toán nhãn đa chiều không được thử; không có thí nghiệm nhiễu nhãn/mất cân bằng (mặc dù tương đẳng theo lớp có ý nghĩa với nhãn bị hoán đổi).

## 8. Liên hệ với bài Nature và đề tài

- Giải thích **một nguồn bất ổn và chi phí** trong TabPFNv2: ensemble hoán vị nhãn (bài Nature dùng ensemble nhiều hoán vị đặc trưng và nhãn) là cách vá cho một khiếm khuyết kiến trúc.
- **Với TabICLv2 (bài 7)**: nếu thí nghiệm có nhiều lớp (>10), TabICL xử lý bằng phân cấp; EquiTabPFN gợi ý hướng khác (kiến trúc tương đẳng). Lưu ý chọn dữ liệu cho final-term: nếu phần lớn dữ liệu ≤10 lớp thì bài này ít tác động.
- Liên hệ nhẹ với **mất cân bằng lớp**: hoán vị nhãn không đổi dự đoán là điều kiện cần cho đánh giá công bằng theo từng lớp; các thước đo như macro-F1 sẽ không bị ảnh hưởng bởi thứ tự lớp ở EquiTabPFN.

## 9. Từ khóa cần nhớ

- **Target equivariance** (tương đẳng hoán vị thành phần mục tiêu); **equivariance gap** E_equi ≥ 0, bằng 0 iff tương đẳng; nghiệm tối ưu của mục tiêu tiền huấn luyện **phải** tương đẳng.
- **Kiến trúc:** tích chập 1×1 cho mục tiêu, token dự đoán, bi-attention (giữa thành phần và giữa mẫu, có mặt nạ), bộ giải mã phi tham số (trung bình trọng số nhãn + hiệu chỉnh dư).
- **Số lớp bất kỳ** ở suy luận dù huấn luyện chỉ <10 lớp; tránh ECOC/phân cấp.
- **Số liệu:** 25M tham số, 4 ngày A100; 76 tập ≤10 lớp, 10 tập >10 lớp; giảm lỗi +1,50%; chậm 5× (≤10 lớp) và 1,3× (>10 lớp) so với TabPFNv1; 566 so với 76 GFLOPS.
- **Hạn chế:** dữ liệu thứ bậc, chi phí bậc hai theo số lớp, dựa trên v1.
