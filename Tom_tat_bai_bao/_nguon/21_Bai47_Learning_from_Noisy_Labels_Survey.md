# Bài 47. Learning from Noisy Labels with Deep Neural Networks: A Survey

| Mục | Thông tin |
|---|---|
| Nơi đăng | IEEE Transactions on Neural Networks and Learning Systems (bản arXiv 2007.08199v7, 10/3/2022) |
| Tác giả | Hwanjun Song (NAVER AI Lab), Minseok Kim, Dongmin Park, Yooju Shin, Jae-Gil Lee (KAIST) |
| Mã nguồn / tài nguyên | github.com/songhwanjun/Awesome-Noisy-Labels (danh sách bài, mã) |
| Lưu ý về PDF | Bảng II và III dùng ký hiệu đánh dấu (hoàn toàn/một phần/không hỗ trợ) bị mất khi trích chữ, nên file này **không** liệt kê dấu tích của từng phương pháp; chỉ nêu các nhận xét bằng lời của tác giả. Bài không có bảng kết quả thực nghiệm mới (chỉ có Hình 1 minh họa). |
| Vị trí trong đề tài | Bài tổng quan về **nhãn nhiễu**, một trong hai chủ đề của khoảng trống nghiên cứu (cùng với mất cân bằng lớp, bài 49). Là cơ sở định nghĩa các loại nhiễu (đối xứng, bất đối xứng, phụ thuộc mẫu) cho thí nghiệm nhãn nhiễu của final-term. **Quan trọng: toàn bộ bài viết cho ảnh và mạng sâu; không có dữ liệu bảng.** |

## 1. Bài này làm gì trong một đoạn

Mạng nơ-ron sâu rất dễ **ghi nhớ nhãn sai** nên khả năng tổng quát hóa giảm mạnh khi nhãn huấn luyện bị nhiễu, và các kỹ thuật điều chuẩn phổ biến (tăng cường dữ liệu, weight decay, dropout, batch norm) không giải quyết triệt để. Bài tổng quan **62 phương pháp huấn luyện bền với nhãn nhiễu**, chia thành **năm nhóm**: kiến trúc bền, điều chuẩn bền, hàm mất mát bền, điều chỉnh mất mát, và chọn mẫu; sau đó so sánh theo **sáu thuộc tính** (linh hoạt, không tiền huấn luyện, khai thác đầy đủ dữ liệu, không giám sát, chịu nhiễu nặng, chịu nhiễu phức tạp), phân tích **ước lượng tỉ lệ nhiễu**, tổng kết **quy trình đánh giá** (bộ dữ liệu, thước đo) và nêu **hướng nghiên cứu tương lai**. Kết luận: **không phương pháp nào đáp ứng cả sáu thuộc tính**.

## 2. Vấn đề và động cơ (mục I)

- Thành công của DNN cần dữ liệu lớn có nhãn sạch, đắt đỏ và tốn thời gian. Nguồn rẻ (Amazon Mechanical Turk, văn bản xung quanh dữ liệu thu thập) thường cho nhãn không đáng tin; chuyên gia cũng gán sai với nhãn phức tạp; nhãn có thể bị tấn công lật nhãn (label-flipping). **Tỉ lệ nhãn hỏng trong dữ liệu thực được báo cáo từ 8,0% đến 38,5%.**
- DNN có rất nhiều tham số nên có thể khớp mọi hàm phức tạp, kể cả nhãn hỏng. Zhang et al. (2017) cho thấy DNN **dễ khớp toàn bộ tập huấn luyện với bất kỳ tỉ lệ nhãn hỏng nào**, dẫn đến tổng quát hóa kém. Hình 1 (WideResNet-16-8 trên CIFAR-100 với nhiễu đối xứng 40%): khoảng cách độ chính xác kiểm tra giữa mô hình huấn luyện trên dữ liệu sạch và dữ liệu nhiễu **vẫn lớn** dù đã bật điều chuẩn. Nhãn nhiễu còn được cho là **gây hại nhiều hơn nhiễu đầu vào**.
- Phạm vi: chỉ **nhãn nhiễu**. Hai loại dữ liệu lỗi khác nằm ngoài phạm vi: học đối kháng (nhiễu loạn nhỏ trên đầu vào) và điền giá trị thiếu (data imputation).
- Các khảo sát liên quan: Frénay & Verleysen (nhãn nhiễu với phương pháp cổ điển như naive Bayes, SVM; định nghĩa và nguồn nhiễu); Zhang et al. (nhãn crowdsourcing, thuật toán EM); Nigam et al. (chỉ hai nhóm); Han et al. (phân loại theo góc nhìn rộng hơn). Bài này là **khảo sát đầu tiên so sánh có hệ thống về phương pháp** (Bảng II, III).

## 3. Nền tảng và phân loại nhiễu (mục II)

- **Bài toán:** phân loại c lớp, DNN với softmax. Dữ liệu sạch D = {(x_i, y_i)}; ta chỉ có dữ liệu nhiễu D̃ = {(x_i, ỹ_i)}; huấn luyện chuẩn trên mini-batch nhiễu: Θ_{t+1} = Θ_t − η∇(1/|B_t|)Σℓ(f(x;Θ_t), ỹ). Khi tối thiểu hóa rủi ro dùng nhãn nhiễu thì không còn bền với nhiễu, và DNN ghi nhớ nhãn hỏng.
- **Nhiễu độc lập với mẫu (instance-independent):** nhãn bị đổi theo **ma trận chuyển nhiễu T** với T_ij = p(ỹ = j | y = i).
  - *Đối xứng (đều)* với tỉ lệ τ: T_ii = 1 − τ, T_ij = τ/(c−1) với i ≠ j (nhãn bị đổi đều sang các lớp khác).
  - *Bất đối xứng (phụ thuộc nhãn):* nhãn dễ nhầm sang một số lớp nhất định ("chó" dễ nhầm "mèo" hơn "cá").
  - *Nhiễu cặp (pair noise):* nhãn chỉ bị đổi sang một lớp duy nhất.
- **Nhiễu phụ thuộc mẫu (instance-dependent):** xác suất hỏng phụ thuộc cả đặc trưng, ρ_ij(x) = p(ỹ = j | y = i, x); thực tế hơn nhưng khó mô hình hóa nên ít được nghiên cứu.
- **Các cách tiếp cận không dùng học sâu** (bốn nhóm): làm sạch dữ liệu (loại mẫu nghi sai, ví dụ bagging/boosting lọc mẫu có trọng số cao, kNN, phát hiện ngoại lai; nhược điểm loại cả mẫu đúng), hàm mất mát thay thế (xấp xỉ 0-1 loss; chỉ cho nhị phân), phương pháp xác suất (ước lượng độ tin cậy nhãn bằng gom cụm, Bayes; dễ quá khớp), mô hình chuyên biệt (SVM phạt mẫu phân loại sai, cây với tiêu chí tách mới; không áp dụng được cho học sâu). Học sâu **nhạy với nhiễu nhãn hơn** học máy cổ điển do khả năng biểu diễn lớn.
- **Hồi quy với nhãn nhiễu:** nhiễu cộng ỹ = y + ε (ε độc lập với x) và nhiễu phụ thuộc mẫu ỹ = ρ(x); nhiều phương pháp phân loại mở rộng được sang hồi quy với chỉnh sửa nhỏ, nên bài tập trung vào phân loại.

## 4. Năm nhóm phương pháp (mục III)

### 4.1. Kiến trúc bền (robust architecture)
- **Lớp thích nghi nhiễu (noise adaptation layer):** thêm lớp trên softmax mô hình hóa ma trận chuyển nhãn, p(ỹ = j | x) = Σ_i T_ij p(y = i | x); lớp này được bỏ khi dự đoán. Ví dụ: Webly learning (huấn luyện trên mẫu dễ từ công cụ tìm kiếm rồi khởi tạo W bằng ma trận nhầm lẫn), noise model (khởi tạo đơn vị, ép khuếch tán), dropout noise model, s-model và c-model (c-model mô hình hóa nhiễu phụ thuộc mẫu), NLNN (EM). *Hạn chế:* không phân biệt được mẫu sai nên coi mọi mẫu như nhau; sai số ước lượng ma trận lớn khi chỉ có dữ liệu nhiễu hoặc nhiễu nặng; EM dễ kẹt cực trị cục bộ, tốn tính toán.
- **Kiến trúc chuyên biệt:** mô hình nhiễu xác suất (hai mạng riêng, tiền huấn luyện trên ít dữ liệu sạch), masking (tri thức con người về chuyển nhãn không hợp lệ, dùng biến thể GAN), contrastive-additive noise network (nhúng chất lượng của nhãn), RoG (bộ phân loại sinh trên DNN đã tiền huấn luyện). *Hạn chế:* chịu được nhiều loại nhiễu hơn nhưng khó mở rộng sang kiến trúc khác.

### 4.2. Điều chuẩn bền
- Điều chuẩn kinh điển (augmentation, weight decay, dropout, batch norm) tốt với nhiễu vừa nhưng không đủ khi nhiễu nặng.
- **Tường minh:** bilevel learning (dùng tập validation sạch điều chỉnh trọng số mini-batch), annotator confusion (EM có điều chuẩn cho nhiều người gán nhãn), **tiền huấn luyện** (fine-tune từ mô hình tiền huấn luyện bền hơn huấn luyện từ đầu), PHuber (gradient clipping), robust early-learning (chỉ phạt tham số "không quan trọng"), ODLN (dữ liệu mở có nhãn ngẫu nhiên). *Hạn chế:* nhạy với siêu tham số phụ thuộc mô hình, có thể cần mạng sâu hơn để bù dung lượng.
- **Ngầm:** huấn luyện đối kháng, **label smoothing** (ȳ(i) = (1 − α)·[ỹ = i] + α/c), **mixup** (nội suy tuyến tính hai mẫu x_mix = λx_i + (1 − λ)x_j, y_mix = λỹ_i + (1 − λ)ỹ_j). *Ưu điểm:* không làm giảm dung lượng, ít siêu tham số phụ thuộc mô hình; nhược điểm: không gian đặc trưng/nhãn mở rộng làm hội tụ chậm.

### 4.3. Hàm mất mát bền
- Cơ sở lý thuyết: nếu ℓ là Lipschitz và classification-calibrated, sai số vượt quá Bayes bị chặn bởi sai số xấp xỉ + ước lượng (Rademacher complexity). **Điều kiện đủ cho hàm mất mát chịu nhiễu đối xứng** (nhiều lớp): τ < (c−1)/c và Σ_{j=1}^{c} ℓ(f(x;Θ), y = j) = C (hằng số) với mọi x, f.
- Hàm mất mát: **MAE** (thỏa điều kiện nhưng tổng quát hóa kém khi dữ liệu phức tạp), **GCE** (Generalized Cross Entropy; bao gồm MAE và CCE), **bi-tempered loss**, **SCE** (Symmetric CE = CCE + reverse CE), **curriculum loss** (cận trên chặt của 0-1 loss), **APL** (Active Passive Loss: kết hợp hàm tích cực tối đa hóa xác suất lớp đã cho và hàm thụ động tối thiểu hóa xác suất lớp khác). Khi dùng categorical cross entropy (mất mát phổ biến) thì kém bền.
- *Hạn chế:* lý thuyết tốt nhưng chỉ hoạt động khi bài toán dễ hoặc ít lớp; làm tăng thời gian hội tụ.

### 4.4. Điều chỉnh mất mát (loss adjustment)
- **Hiệu chỉnh mất mát (loss correction):** nhân ma trận chuyển ước lượng. *Backward correction* (nhân nghịch đảo T̂⁻¹ với vector mất mát của mọi nhãn), *forward correction* (nhân T̂ với đầu ra softmax trước khi tính mất mát), gold loss correction (dùng dữ liệu sạch/anchor points), T-Revision (không cần anchor points), Dual T (tách ma trận thành hai ma trận dễ ước lượng), và các mô hình cho nhiễu phụ thuộc mẫu. *Hạn chế:* phụ thuộc độ chính xác của T̂, thường cần tri thức tiền nghiệm (anchor point, tập sạch).
- **Gán lại trọng số mất mát (reweighting):** w(x, ỹ) nhỏ cho mẫu nghi sai: importance reweighting (tỉ số P_D(x,ỹ)/P_D̃(x,ỹ)), active bias (dùng phương sai dự đoán), DualGraph (GNN). *Hạn chế:* phải chỉ định hàm trọng số và siêu tham số thủ công, phụ thuộc loại nhiễu và dữ liệu.
- **Làm mới nhãn (label refurbishment):** y_refurb = α·ỹ + (1 − α)·ŷ. Bootstrapping, dynamic bootstrapping (mô hình trộn beta trên phân phối mất mát), self-adaptive training (trung bình động), D2L (chiều nội tại cục bộ), SELFIE (chỉ sửa mẫu "có thể làm mới" với dự đoán nhất quán), AdaCorr (có cận sai số), SEAL (trung bình softmax suốt quá trình huấn luyện). *Hạn chế:* nếu có nhiều lớp dễ nhầm, DNN có thể khớp nhãn sửa sai.
- **Học meta (meta learning):** học quy tắc điều chỉnh tự động bằng một tập sạch nhỏ: L2LWS, CWS, automatic reweighting, **meta-weight-net** (hàm trọng số là MLP một lớp ẩn), data coefficients (tập sạch chỉ khoảng 0,2% tập huấn luyện), knowledge distillation, MLC. *Hạn chế:* cần tập validation sạch không chệch, điều thường không có.

### 4.5. Chọn mẫu (sample selection)
- Cập nhật chỉ trên tập mẫu được nhận là sạch C_t ⊆ B_t. Cơ sở: **hiệu ứng ghi nhớ** — DNN học các mẫu đơn giản, tổng quát trước, rồi mới khớp dần nhãn nhiễu; Li et al. chứng minh trọng số lệch khỏi khởi tạo chỉ khi quá khớp nhãn hỏng. Do đó **mẹo "small-loss"**: mẫu có mất mát nhỏ coi như nhãn đúng (Hình 5). *Rủi ro:* sai số chọn tích lũy, nhất là khi có nhiều lớp mơ hồ.
- **Học nhiều mạng:** Decouple (cập nhật khi hai mạng bất đồng), MentorNet (mạng thầy chọn mẫu cho mạng trò), **Co-teaching / Co-teaching+** (hai mạng chọn mẫu small-loss cho nhau), JoCoR (co-regularization). Giảm confirmation bias nhưng huấn luyện kém hiệu quả; small-loss kém khi phân phối mất mát của mẫu đúng và sai chồng lấn nhiều (như nhiễu bất đối xứng, Hình 5b).
- **Học nhiều vòng:** ITLM (trimmed loss), INCV (cross-validation), O2U-Net (tốc độ học tuần hoàn), iterative detection (local outlier factor), MORPH, TopoFilter, NGC (đồ thị láng giềng gần nhất). Chi phí tăng tuyến tính theo số vòng.
- **Hybrid:** kết hợp chọn mẫu với học bán giám sát: SELF (mean-teacher), **DivideMix** (hai thành phần GMM chia mẫu thành có nhãn/không nhãn rồi MixMatch), RoCL; hoặc SELFIE (chọn mẫu + hiệu chỉnh). Bền hơn nhiều nhưng thêm siêu tham số và chi phí tính toán.

## 5. So sánh phương pháp (mục IV, Bảng II–III)

Sáu thuộc tính: (P1) **Flexibility** (dùng với kiến trúc bất kỳ); (P2) **No pre-training**; (P3) **Full exploration** (dùng mọi mẫu, không loại bỏ mẫu khó nhưng có ích); (P4) **No supervision** (không cần tập sạch hay tỉ lệ nhiễu biết trước); (P5) **Heavy noise**; (P6) **Complex noise** (cả nhiễu độc lập lẫn phụ thuộc mẫu). Kết luận: **không phương pháp nào hỗ trợ đủ cả sáu**; các phương pháp cùng (nhóm con) hỗ trợ tập thuộc tính tương tự vì chung triết lý. 62 phương pháp được liệt kê kèm tình trạng mã nguồn (chính thức/không chính thức, framework Caffe/Keras/PyTorch/TensorFlow, hoặc "N/A").

## 6. Ước lượng tỉ lệ nhiễu (mục V)

Cần cho nhóm điều chỉnh mất mát và chọn mẫu (xác định bao nhiêu mẫu giữ lại), nhưng độ chính xác của ước lượng ít được phân tích. Ba cách:
1. **Ma trận chuyển nhiễu và anchor points:** mẫu x nhãn i là anchor point nếu p(y = i|x) = 1; khi đó T̂_ij = (1/|A_i|)Σ_{x∈A_i} p(ỹ = j | x; Θ) và tỉ lệ nhiễu (dữ liệu cân bằng) τ̂ = (1/c)Σ_iΣ_{j≠i} T̂_ij. Anchor points thường không biết nên được tìm bằng lý thuyết hoặc heuristic; T-Revision, Dual-T, VolMinNet bỏ yêu cầu này.
2. **Mô hình trộn Gauss (GMM) hai thành phần một chiều** trên phân phối mất mát: τ̂ = E[p(g | ℓ)] với g là thành phần có mất mát lớn (Hình 7). Pleiss et al. chỉ ra mất mát khó tách dần theo thời gian, nên dùng **diện tích dưới đường mất mát (AUL)**, tổng mất mát qua các epoch trước.
3. **Cross-validation:** cần tập sạch; Chen et al. chia ngẫu nhiên hai tập nhiễu và dùng độ chính xác kiểm tra: Test Acc = (1 − τ̂)² + τ̂²/(c−1) (đối xứng) hoặc (1 − τ̂)² + τ̂² (bất đối xứng) để giải ngược τ̂.

## 7. Thiết kế thực nghiệm chuẩn (mục VI, Bảng IV)

- **Dữ liệu sạch (7 bộ, nhiễu nhân tạo phải tự thêm):** MNIST (60K huấn luyện, 10 lớp), Fashion-MNIST (60K, 10), CIFAR-10 (50K, 10), CIFAR-100 (50K, 100), SVHN (73K, 10), Tiny-ImageNet (100K, 200), ImageNet (1,3M, 1000). Nhiễu tổng hợp: đối xứng và bất đối xứng.
- **Dữ liệu nhiễu thực (6 bộ):** ANIMAL-10N (50K, 10 lớp, nhiễu khoảng 8,0%), CIFAR-10N (nhiễu khoảng 9,0/18,0/40,2%), CIFAR-100N (khoảng 25,6/40,2%), Food-101N (310K, 101 lớp, khoảng 18,4%), Clothing1M (1M, 14 lớp, khoảng 38,5%), WebVision (2,4M, 1000 lớp, khoảng 20,0%). Phần lớn có tập validation sạch riêng và tỉ lệ nhiễu ước lượng.
- **Thước đo:** độ chính xác kiểm tra (chính), độ chính xác validation; **label precision** và **label recall** (cho nhóm chọn mẫu: tỉ lệ mẫu được chọn có nhãn đúng; tỉ lệ mẫu nhãn đúng được chọn); **correction error** (cho nhóm làm mới nhãn: tỉ lệ mẫu được sửa nhãn sai).

## 8. Hướng nghiên cứu tương lai (mục VII)

1. **Nhiễu phụ thuộc mẫu:** giả định độc lập với đặc trưng của phần lớn lý thuyết và hàm mất mát không đúng thực tế; Chen et al. kiểm định giả thuyết trên Clothing1M và bác bỏ nhiễu độc lập mẫu (xác suất dưới 10^−21250). Chọn mẫu hoạt động ổn nhưng hiệu năng giảm nhiều với nhiễu phụ thuộc mẫu vì mất mát của mẫu đúng và sai chồng lấn (Hình 5b); các nghiên cứu hiện có chủ yếu nhị phân hoặc mô hình nhỏ (hồi quy logistic).
2. **Dữ liệu đa nhãn có nhiễu:** phần lớn phương pháp cho đơn nhãn nhiều lớp; tập đa nhãn hàng triệu mẫu có hơn 26,6% nhãn dương sai và nhiều nhãn bị bỏ sót.
3. **Mất cân bằng lớp kết hợp nhãn nhiễu (liên hệ trực tiếp với đề tài):** phần lớn phương pháp bền giả định phân phối lớp đều. Khi mất cân bằng, mô hình hội tụ cho lớp đa số nhanh hơn nên **hầu hết mẫu lớp đa số có mất mát nhỏ ("ghi nhớ sớm")**, và mẹo small-loss có nguy cơ **loại bỏ hầu hết mẫu lớp thiểu số**. Ngoài ra bài toán mất cân bằng ưa mẫu mất mát lớn còn bài toán nhãn nhiễu ưa mẫu mất mát nhỏ: mâu thuẫn về khái niệm. Cần xét đồng thời hai vấn đề.
4. **Huấn luyện bền và công bằng:** bền với nhãn nhiễu và công bằng (thiên lệch dữ liệu) thường được nghiên cứu riêng; nhóm bảo vệ nhiễu làm các tiêu chí công bằng theo nhóm không áp dụng trực tiếp; mới có vài nghiên cứu sau 2020 cho nhị phân.
5. **Liên hệ với nhiễu đầu vào:** huấn luyện đối kháng làm DNN bền với nhãn nhiễu (Damodaran: Wasserstein adversarial regularization; Zhu: số bước PGD làm tiêu chí chọn mẫu).
6. **Hiệu quả của quy trình học:** nhiều mạng hoặc nhiều vòng làm giảm hiệu quả; phần lớn công trình **không báo cáo độ phức tạp thời gian**.

## 9. Kết luận của tác giả
DNN dễ quá khớp nhãn sai vì dung lượng lớn, điều chuẩn kinh điển không đủ, và khó khăn gán nhãn thực tế làm tình hình tệ hơn. Bài tổng hợp các phương pháp thành năm nhóm kèm điểm yếu, so sánh bằng sáu thuộc tính (không ai đủ cả sáu), và hướng dẫn thực nghiệm.

## 10. Hạn chế (tác giả nêu và suy ra)
1. **Toàn bộ thiên về ảnh và mạng sâu.** Các bộ dữ liệu chuẩn đều là ảnh (MNIST, CIFAR, ImageNet, Clothing1M...), không có bộ dữ liệu bảng; không bàn về cây tăng cường hay mô hình bảng. Việc áp dụng sang dữ liệu bảng và TFM **không được khảo sát**.
2. **Không có thực nghiệm so sánh trực tiếp:** chỉ so sánh định tính theo sáu thuộc tính; không có bảng độ chính xác chung cho 62 phương pháp (Hình 1 chỉ là minh họa).
3. Tác giả nói các nghiên cứu có thể chưa đầy đủ ("có thể thiếu một số công trình vì số bài học sâu tăng bậc hai").
4. Ước lượng tỉ lệ nhiễu "chưa được phân tích đúng mức" dù nhiều phương pháp phụ thuộc vào nó.
5. Phần lớn lý thuyết dựa trên giả định nhiễu độc lập mẫu, đối xứng; nhiễu thực (phụ thuộc mẫu) chỉ có ít nghiên cứu, thường nhị phân.
6. Hồi quy chỉ được nêu ngắn, không đi sâu.

## 11. Liên hệ với bài Nature và đề tài
- Cung cấp **từ vựng và quy ước chuẩn** cho thí nghiệm nhãn nhiễu: nhiễu đối xứng với tỉ lệ τ (nhãn bị đổi đều sang các lớp khác, T_ij = τ/(c−1)), nhiễu bất đối xứng, nhiễu cặp, nhiễu phụ thuộc mẫu; mức nhiễu thực tế 8–38,5%. Có thể dùng các mức τ như 5%, 10%, 20%, 40% (bài 23 dùng 5%).
- **Bằng chứng chính cho khoảng trống nghiên cứu:** khảo sát nhãn nhiễu không có dữ liệu bảng và không có mô hình nền tảng; bài Nature (TabPFNv2) và các bài TFM khác (TabICLv2) không đánh giá nhãn nhiễu có kiểm soát. Kiểm tra lại khi lập ma trận 4 SOTA.
- Gợi ý lý thuyết cho TFM: vì TabPFN thực hiện suy luận trong ngữ cảnh mà không cập nhật trọng số, **hiệu ứng ghi nhớ nhãn sai của huấn luyện SGD có thể không xảy ra theo cùng cơ chế**, nhưng nhãn sai trong ngữ cảnh vẫn ảnh hưởng suy luận; đây là giả thuyết cần kiểm chứng, không phải kết luận của bài.
- Mục VII-C (mất cân bằng + nhiễu) là lý do thiết kế thí nghiệm kết hợp hai yếu tố; liên kết với bài 49.

## 12. Từ khóa cần nhớ
- 62 phương pháp, **5 nhóm** (robust architecture, regularization, loss function, loss adjustment, sample selection), **6 thuộc tính** (P1–P6), nhãn hỏng thực tế **8,0–38,5%**.
- Loại nhiễu: **đối xứng, bất đối xứng, cặp, phụ thuộc mẫu**; ma trận chuyển nhiễu T; điều kiện hàm mất mát chịu nhiễu: τ < (c−1)/c và Σℓ = hằng số.
- **Small-loss trick**, hiệu ứng ghi nhớ, Co-teaching, DivideMix, forward/backward correction; ước lượng nhiễu qua anchor points, GMM, cross-validation.
- Mất cân bằng lớp + nhãn nhiễu: small-loss có nguy cơ loại mẫu thiểu số.
