# Bài 34. TransTab: Learning Transferable Tabular Transformers Across Tables

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2022 (arXiv:2205.09328v2) |
| Tác giả | Zifeng Wang, Jimeng Sun (UIUC) |
| Mã nguồn | github.com/RyanWangZf/transtab |
| Lưu ý về PDF | Bảng 2–6 và 8–12 đọc rõ. Hình 4–6 (số phân vùng, độ chồng cột) chỉ có đồ thị, không có số cụ thể nên mình chỉ nêu xu hướng. |
| Vị trí trong đề tài | Một trong những mô hình **"nền tảng cho bảng" đầu tiên theo hướng ngữ nghĩa cột**: biến mỗi dòng thành chuỗi token có tên cột, nên học được trên nhiều bảng có cột khác nhau. Cùng họ ý tưởng với ConTextTab, TabSTAR, CARTE. |

## 1. Bài này làm gì trong một đoạn

Mô hình học máy cho bảng thường giả định cấu trúc cột cố định ở lúc huấn luyện và lúc kiểm thử. Thực tế nhiều bảng chỉ chồng lấn một phần cột, nên người ta phải gộp bảng, bỏ cột không khớp và bỏ mẫu, gây lãng phí dữ liệu. TransTab giải bài toán này bằng cách **biến mỗi ô thành token có ngữ cảnh cột** (tên cột + giá trị) rồi đưa vào Transformer có cổng (gated transformer), nên nhận được bảng có số cột bất kỳ. Nhờ đó có thể: học chéo bảng, học tăng dần khi có thêm cột, tiền huấn luyện rồi tinh chỉnh, và thậm chí suy luận zero-shot trên bảng mới. Tác giả thêm phương pháp tiền huấn luyện **tương phản theo phân vùng dọc (VPCL)**. Trên 5 bộ thử nghiệm lâm sàng và 8 bộ dữ liệu công khai, TransTab xếp hạng trung bình 1,00 / 1,00 / 1,78 trên 12 phương pháp ở ba kịch bản (có giám sát, thêm cột dần, chuyển giao); tiền huấn luyện nâng AUC trung bình 2,3% so với học có giám sát thuần.

## 2. Bối cảnh

- **Vấn đề:** các bảng thật thường dùng quy ước khác nhau (tên cột và mã giá trị khác nhau). Trước đây, phải làm sạch, loại cột/mẫu không khớp trước khi huấn luyện.
- **Hạn chế của phương pháp trước:** các mô hình sâu cho bảng (TabNet, TabTransformer, FT-Transformer) và SSL cho bảng (VIME, SCARF, SubTab) đều đòi hỏi cùng cấu trúc cột ở huấn luyện và kiểm thử, nên không chuyển giao được giữa các bảng. Tác giả nói rõ: tiền huấn luyện kiểu BERT cho bảng chưa thành công vì thiếu "đơn vị cơ bản" chung (như pixel hay token).
- **Ý tưởng trung tâm:** ý nghĩa của ô do **mô tả cột** quyết định (60 ở cột "weight" là 60 kg, ở cột "age" là 60 tuổi). Thay vì mã hóa "man → 0 theo codebook", đưa vào chuỗi "gender is man". Cách này khái quát hóa giữa các bảng (ví dụ "previously smoked" và "smoking history" giống nghĩa).

### Bốn kịch bản ứng dụng (Hình 1)
Lấy ví dụ dự đoán hiệu quả thuốc từ nhiều bảng thử nghiệm lâm sàng ung thư:
1. **Chuyển giao chéo bảng:** nhiều bảng cùng thuốc, cột chồng một phần; học cho một thử nghiệm bằng cách tận dụng tất cả.
2. **Học tăng dần theo cột:** các giai đoạn thử nghiệm thêm cột mới theo thời gian.
3. **Tiền huấn luyện + tinh chỉnh:** nhiều bảng không có nhãn dùng để tiền huấn luyện, sau đó tinh chỉnh trên bảng đích có nhãn.
4. **Zero-shot:** suy luận trên bảng bệnh nhân không có chung cột nào với bảng huấn luyện.

## 3. Phương pháp

### 3.1. Bộ xử lý đầu vào (Input processor)
Mỗi ô được xếp vào một trong ba loại, rồi chuyển thành embedding token:
- **Phân loại/văn bản (cat):** nối tên cột với giá trị thành câu, tokenize, tra ma trận embedding token (Eᶜ ∈ ℝ^{n_c×d}).
- **Nhị phân (bin):** chỉ khi giá trị = 1 thì tokenize tên cột (Eᵇ); nếu = 0 thì **bỏ qua** hoàn toàn. Vì thế rất tiết kiệm với dữ liệu one-hot thưa, nhiều chiều.
- **Số (num):** không nối tên cột với giá trị (vì tokenization ghi nhận kém khi phân biệt số). Thay vào đó, embedding của tên cột được **nhân với giá trị số** đã chuẩn hóa: Eᵘ = xᵘ × Eᵘ,col. Tác giả thấy cách đơn giản này tốt hơn các kỹ thuật embedding số phức tạp.
- Cả ba qua layer norm và cùng một lớp tuyến tính để căn chỉnh vào chung một không gian, rồi nối với embedding [cls].

### 3.2. Gated Transformer
Gồm multi-head self-attention và lớp feed-forward có cổng:
- Attention: Z^att = MultiHeadAttn(Z^l).
- Cổng theo từng token: g^l = σ(Z^att · w_G), g ∈ [0,1]^n điều chỉnh độ lớn mỗi token.
- Đầu ra: Z^{l+1} = Linear(g^l ⊙ Z^att) ⊕ Linear(Z^att).
Cơ chế cổng giúp mô hình tập trung vào đặc trưng quan trọng bằng cách phân phối lại sự chú ý trên token. Embedding [cls] ở lớp L cuối dùng cho dự đoán.

### 3.3. Tiền huấn luyện: VPCL (vertical-partition contrastive learning)
Các phương pháp SSL cho bảng hiện nay làm việc trên toàn bộ cột, tốn tính toán và dễ quá khớp. TransTab **chia dọc các cột thành K phân vùng** (các phân vùng kề nhau có thể chồng lấn một phần) và coi chúng như các "góc nhìn" của cùng một mẫu, với giả thuyết biểu diễn tốt phải bất biến theo góc nhìn. Thao tác này cũng thân thiện với cơ sở dữ liệu hướng cột (truy vấn tập con cột nhanh). Khi K > 2 chọn ngẫu nhiên hai phân vùng.
- **Self-VPCL (không nhãn):** cặp dương = các phân vùng của cùng một mẫu, cặp âm = phân vùng của mẫu khác. Loss InfoNCE với độ tương tự cosine ψ trên embedding [cls] đã chiếu (ẑ), cộng qua batch B và các phân vùng.
- **VPCL (có nhãn):** cặp dương = các phân vùng của các mẫu **cùng lớp** (như supervised contrastive learning), cặp âm = phân vùng của mẫu khác lớp. 
- **Lý do không dùng loss có giám sát thông thường để tiền huấn luyện:** các bảng khác nhau về kích thước, định nghĩa tác vụ và phân phối lớp, nên đầu dự đoán riêng cho từng bảng làm bộ mã hóa thiên về tác vụ/lớp chiếm ưu thế, và siêu tham số phù hợp lại khác nhau giữa các bảng (cùng bộ siêu tham số có thể quá khớp ở bảng này, thiếu khớp ở bảng kia). Học tương phản có giám sát được cho là bền hơn với nhiễu và siêu tham số, và không cần nhiều đầu dự đoán.

## 4. Thiết lập thí nghiệm

- **Câu hỏi nghiên cứu:** Q1 học có giám sát; Q2 thêm cột dần; Q3 học chuyển giao nhiều bảng cùng miền; Q4 zero-shot; Q5 VPCL so với tiền huấn luyện có giám sát và self-supervised.
- **Dữ liệu 1: 5 bộ thử nghiệm lâm sàng** (Project Data Sphere), dự đoán tử vong nhị phân (kiểm duyệt = sống; sự kiện khác = tử vong). Mỗi bộ có nhóm bệnh nhân và cột riêng:

| Bộ | Số mẫu | Phân loại | Nhị phân | Số | Tỉ lệ dương |
|---|---|---|---|---|---|
| NCT00041119 | 3871 | 5 | 8 | 2 | 0,07 |
| NCT00174655 | 994 | 3 | 31 | 15 | 0,02 |
| NCT00312208 | 1651 | 5 | 12 | 6 | 0,19 |
| NCT00079274 | 2968 | 5 | 8 | 3 | 0,12 |
| NCT00694382 | 1604 | 1 | 29 | 11 | 0,45 |

- **Dữ liệu 2: 8 bộ công khai** (đều nhị phân): credit-g (1000), credit-approval (690), dresses-sales (500), adult (48.842), cylinder-bands (540), blastchar (7043), insurance-co (5822), 1995-income (32.561).
- **Tiền xử lý:** cho baseline, đặc trưng phân loại mã hóa thứ tự (ordinal) hoặc one-hot, số chuẩn hóa min-max về [0,1]; riêng TransTab ánh xạ chỉ số lớp về mô tả gốc (ví dụ "1" ở "gender" thành "female").
- **Cấu hình TransTab:** 2 lớp gated transformer, embedding 128, ẩn 256, 8 head, ReLU, không dropout; Adam, lr ∈ {2e-5, 5e-5, 1e-4}, batch ∈ {16, 64, 128}; tiền huấn luyện tối đa 50 epoch, có giám sát tối đa 100 epoch với early stopping kiên nhẫn 10. Một GPU RTX3070.
- **Baseline (11):** LR, XGBoost, MLP, SNN (SeLU-MLP), TabNet, DCN, AutoInt, TabTransformer, FT-Transformer, VIME, SCARF (siêu tham số baseline tìm trên lưới nhỏ, xem phụ lục B của bài).
- **Đánh giá:** AUROC trung bình 10 seed; cột Rank là xếp hạng trung bình trên các bộ dữ liệu.
- **Cách chia tập con** (cố định seed, khác nhau giữa các thí nghiệm, số mẫu bằng nhau): thêm cột dần chia thành v1, v2, v3 với set1 = v1, set2 = v1+v2, set3 = v1+v2+v3; chuyển giao chia hai phần cột v1, v2 chồng 50%; zero-shot chia ba phần cột rời nhau, set1 = v1, set2 = v2, set3 = v3.

## 5. Kết quả

### 5.1. Q1. Học có giám sát (không tiền huấn luyện), 5 bộ lâm sàng

| Mô hình | 00041119 | 00174655 | 00312208 | 00079274 | 00694382 | Rank |
|---|---|---|---|---|---|---|
| LR | 0,6364 | 0,8543 | 0,7382 | 0,7067 | 0,7360 | 5,40 |
| XGBoost | 0,5937 | 0,5000 | 0,6911 | 0,6784 | 0,7440 | 9,60 |
| MLP | 0,6340 | 0,6189 | 0,7427 | 0,6967 | 0,7063 | 8,00 |
| SNN | 0,6335 | 0,9130 | 0,7469 | 0,6948 | 0,7246 | 5,80 |
| TabNet | 0,5856 | 0,5401 | 0,6910 | 0,6031 | 0,7113 | 11,40 |
| DCN | 0,6349 | 0,7577 | 0,7431 | 0,6952 | 0,7458 | 5,60 |
| AutoInt | 0,6327 | 0,7502 | 0,7479 | 0,6958 | 0,7411 | 6,20 |
| TabTransformer | 0,6187 | 0,9035 | 0,7069 | 0,7178 | 0,7229 | 7,20 |
| FT-Transformer | 0,6372 | 0,9073 | 0,7586 | 0,7090 | 0,7231 | 4,20 |
| VIME | 0,6397 | 0,8533 | 0,7227 | 0,6790 | 0,7232 | 7,00 |
| SCARF | 0,6248 | 0,9310 | 0,7267 | 0,7176 | 0,7103 | 6,60 |
| **TransTab** | **0,6408** | **0,9428** | **0,7770** | **0,7281** | **0,7648** | **1,00** |

- TransTab thắng trên cả 5 bộ. Điều đáng chú ý: **LR thắng hơn một nửa số baseline**; ngoài TransTab chỉ FT-Transformer hơn LR rõ rệt.
- Trên 8 bộ công khai (Bảng 8), TransTab xếp hạng 3,00 (FT-Transformer 3,94; DCN 4,12; AutoInt 4,62; XGBoost 5,12), tức **ngang các mô hình tốt nhất**, không vượt trội. Baseline từ lĩnh vực dự đoán CTR (DCN, AutoInt) cạnh tranh tốt. Ví dụ AUROC TransTab: credit-g 0,768; credit-approval 0,881; dresses-sales 0,643; adult 0,907; cylinder-bands 0,851; blastchar 0,845; insurance-co 0,822; 1995-income 0,919 (XGBoost thắng adult 0,912, cylinder-bands 0,892, 1995-income 0,925).

### 5.2. Q2. Học tăng dần theo cột (Bảng 3, lâm sàng)
Baseline phải chọn: (a) học từ mọi dữ liệu chỉ với đặc trưng set1, hoặc (b) chỉ dùng dữ liệu set3 (đủ cột); báo cáo kết quả tốt hơn trong hai cách. TransTab học từ cả ba tập.

| Mô hình | 00041119 | 00174655 | 00312208 | 00079274 | 00694382 | Rank |
|---|---|---|---|---|---|---|
| LR | 0,6213 | 0,8485 | 0,6801 | 0,6258 | 0,7236 | 4,6 |
| XGBoost | 0,5735 | 0,7890 | 0,6760 | 0,6038 | 0,6463 | 8,8 |
| FT-Transformer | 0,5552 | 0,8045 | 0,7148 | 0,6471 | 0,6815 | 5,8 |
| SCARF | 0,5996 | 0,6261 | 0,7072 | 0,6535 | 0,6957 | 5,2 |
| **TransTab** | **0,6797** | **0,8545** | **0,7617** | **0,6857** | **0,7795** | **1,0** |

(Các baseline khác như MLP, SNN, TabNet, DCN, AutoInt, TabTransformer, VIME có rank 5,8–9,0; VIME rất thấp ở 00312208: 0,3705.) Trên 8 bộ công khai (Bảng 9) TransTab rank 1,12, vượt xa; ví dụ credit-g 0,741 so với 0,674 của DCN.

### 5.3. Q3. Chuyển giao chéo bảng (Bảng 4)
Mỗi bộ chia hai tập cột set1, set2 chồng 50%. Baseline huấn luyện và kiểm thử riêng trên set1 hoặc set2 (chỉ giám sát); TransTab tiền huấn luyện trên set1 rồi tinh chỉnh trên set2 (và ngược lại). TransTab rank **1,78** (FT-Transformer 4,22; LR 5,33; SCARF 5,56; AutoInt 5,89). Ví dụ AUROC set1 / set2: 00041119: 0,653 / 0,653; 00174655: 0,904 / 0,846; 00312208: 0,730 / 0,756; 00079274: 0,680 / 0,711; 00694382: 0,747 / 0,774. Ở bộ 00079274 set1, FT-Transformer (0,692) và DCN (0,682) cao hơn 0,680 của TransTab, nên TransTab không thắng tuyệt đối mọi ô. Trên dữ liệu công khai (Bảng 10) rank 2,33 (FT-Transformer 4,97; MLP 5,00).

### 5.4. Q4. Zero-shot (Bảng 5)
Chia bảng thành 3 tập cột rời nhau; mô hình học từ set1+set2 rồi kiểm thử trên set3 mà không huấn luyện thêm. Hai đối chứng: *Supervised* (học trên set3) và *Transfer* (học set1+set2 rồi tinh chỉnh trên set3).

| TransTab | 00041119 | 00174655 | 00312208 | 00079274 | 00694382 |
|---|---|---|---|---|---|
| Supervised | 0,5854 | 0,6484 | 0,7536 | 0,7087 | 0,6479 |
| Transfer | 0,6130 | 0,6909 | 0,7658 | 0,7163 | 0,6752 |
| Zero-shot | 0,5990 | 0,6752 | 0,7576 | 0,7036 | 0,6740 |

Zero-shot **tốt hơn học có giám sát trung bình** (thắng 4/5 bộ; thua nhẹ ở 00079274: 0,7036 so với 0,7087), vì tận dụng nhiều dữ liệu hơn (set1+set2 so với chỉ set3); Transfer tốt nhất. Trên dữ liệu công khai (Bảng 11), zero-shot vượt Supervised ở 6/8 bộ, ví dụ credit-g: 0,685 so với 0,581; cylinder-bands: 0,710 so với 0,733 thì thấp hơn; dresses-sales: 0,538 so với 0,571 thấp hơn. Hình 6 (thay đổi tỉ lệ chồng cột từ 0 đến 1): mô hình vẫn dự đoán hợp lý kể cả khi tập huấn luyện **không có cột chung nào** với tập kiểm thử.

### 5.5. Q5. VPCL so với tiền huấn luyện thường (Bảng 6, lâm sàng)

| TransTab | 00041119 | 00174655 | 00312208 | 00079274 | 00694382 |
|---|---|---|---|---|---|
| Supervised | 0,6313 | 0,8348 | 0,7444 | 0,6885 | 0,7293 |
| Transfer (tiền huấn luyện giám sát thường) | 0,6424 | 0,8183 | 0,7458 | 0,6928 | 0,7239 |
| Self-VPCL | 0,6412 | 0,8577 | 0,7486 | 0,7069 | 0,7348 |
| VPCL | 0,6405 | 0,8583 | 0,7517 | 0,7063 | 0,7392 |

Tiền huấn luyện giám sát thường **làm giảm hiệu quả ở 2/5 bộ** (00174655, 00694382); VPCL luôn cải thiện so với Supervised. Khi tăng số phân vùng K, VPCL tốt hơn Self-VPCL (Hình 4). Trên dữ liệu công khai (Bảng 12) tiền huấn luyện cho lợi ích ít, vì các bảng không liên quan định nghĩa cột và tác vụ khác hẳn; ví dụ cylinder-bands: Supervised 0,841, Transfer 0,819, VPCL 0,862. Tác giả nói cần nghiên cứu "kiểu hình bảng" (table phenotype) để biết những bảng nào có lợi cho nhau.

## 6. Hạn chế

Tác giả nêu (phụ lục A):
- Tác động tiêu cực tiềm ẩn: cần cân nhắc quyền riêng tư vì TransTab tốt hơn khi đặc trưng có tên mô tả bằng chữ (giải pháp: ánh xạ sang token máy đọc được bằng codebook riêng).
- Tốn tài nguyên hơn MLP và cây vì dùng full attention và tokenize khi tạo đặc trưng (giải pháp: thay bằng khối kiểu MLP như gated attention unit, hoặc tokenize trước).
- Tiền huấn luyện trên bảng không liên quan ít hiệu quả.

Nhận xét thêm của mình:
1. **Quy mô tiền huấn luyện rất nhỏ:** chỉ vài bộ cùng miền lâm sàng hoặc tách từ một bảng (chia cột), không phải corpus hàng nghìn bảng; "nền tảng" ở đây là hướng đi hơn là kết quả đã đạt.
2. Phần lớn bộ dữ liệu nhỏ (500–5000 mẫu), AUROC nhị phân; chưa có hồi quy hay đa lớp. Trên dữ liệu công khai TransTab chỉ ngang FT-Transformer/DCN, XGBoost còn thắng ba bộ.
3. Baseline của kịch bản "thêm cột/chuyển giao" bị đặt ở thế bất lợi (bỏ cột hoặc bỏ mẫu), nên mức vượt trội phần lớn do thiết kế so sánh; so với việc đơn giản là điền khuyết rồi dùng XGBoost thì chưa kiểm chứng.
4. Không so sánh với TabPFN hay các mô hình ICL (bài ra trước TabPFN v2); không có đánh giá nhiễu nhãn, mất cân bằng hay hiệu chỉnh xác suất. Các bộ lâm sàng lại mất cân bằng mạnh (tỉ lệ dương 0,02–0,07), nhưng chỉ báo cáo AUROC.
5. Cách xử lý số bằng "nhân embedding cột với giá trị" khá thô, về sau bị thay bằng embedding số tinh vi hơn (xem bài 35) hoặc mã hóa ngữ nghĩa của CARTE/TabSTAR.

## 7. Liên hệ với bài Nature và đề tài

- TabPFN v2 xử lý bảng mới bằng ICL **không dùng tên cột** (chỉ có phân phối giá trị); TransTab và các mô hình sau (CARTE, ConTextTab, TabSTAR) khai thác **ngữ nghĩa tên cột**. Đây là hướng bổ sung, và là điểm TabPFNv2 chưa dùng (ghi chú trong bài 11, 16, 19).
- Là tiền thân của khái niệm "mô hình nền tảng cho bảng" qua học chuyển giao giữa các bảng khác schema.
- Với thí nghiệm cuối kỳ: dữ liệu lâm sàng có tỉ lệ dương rất thấp là gợi ý tốt cho bài toán mất cân bằng, nhưng TransTab cần dữ liệu có tên cột để khai thác hết lợi thế.

## 8. Từ khóa cần nhớ

- **Ý chính:** ô = (tên cột, giá trị) → token; nhận bảng có số cột bất kỳ.
- **Bộ xử lý:** cat nối tên cột + giá trị; bin chỉ token hóa khi bằng 1; num = embedding cột × giá trị.
- **Gated transformer:** cổng sigmoid theo token; 2 lớp, d = 128, 8 head.
- **VPCL:** tương phản trên phân vùng dọc của cột; dương = cùng lớp; bền hơn loss giám sát thường (làm giảm hiệu năng ở 2/5 bộ).
- **Số liệu:** rank 1,00 / 1,00 / 1,78 trên 12 phương pháp; zero-shot vẫn tốt hơn học giám sát trên set3 trung bình; trên bộ công khai chỉ ngang FT-Transformer.
