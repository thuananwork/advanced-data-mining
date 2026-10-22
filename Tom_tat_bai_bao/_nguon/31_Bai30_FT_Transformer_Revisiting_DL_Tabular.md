# Bài 30. Revisiting Deep Learning Models for Tabular Data (ResNet và FT-Transformer)

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2021; arXiv:2106.11959v5, 26/10/2023 |
| Tác giả | Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko (Yandex) |
| Mã nguồn | github.com/yandex-research/tabular-dl-revisiting-models |
| Lưu ý về PDF | Văn bản và Bảng 2, 3, 4, 5, 6, 10 đọc sạch. Bảng 1 và Bảng 7 (mô tả dữ liệu) bị lệch dòng nhưng ghép lại được từ hai bảng; Bảng 8–9 (độ lệch chuẩn từng mô hình) bị lệch nhiều nên không dùng. Công thức chính và siêu tham số đọc được. |
| Vị trí trong đề tài | Bài **nền tảng của học sâu trên bảng**: định nghĩa baseline mạnh (ResNet, FT-Transformer) mà hầu như mọi TFM và survey sau đó đều dùng làm mốc so sánh, kể cả TabArena, TabM, TabDPT. |

## 1. Bài này làm gì trong một đoạn

Văn liệu học sâu cho dữ liệu bảng lúc đó đề xuất hàng loạt kiến trúc mới nhưng mỗi bài dùng một bộ dữ liệu và một quy trình khác nhau, nên không ai biết mô hình nào thực sự tốt và GBDT có bị vượt hay không. Bài này làm ba việc: (1) so sánh công bằng các họ kiến trúc chính trên cùng 11 bộ dữ liệu với cùng quy trình huấn luyện và tinh chỉnh; (2) chỉ ra rằng một **ResNet đơn giản** (bị các bài trước bỏ qua) đã là baseline rất mạnh, mà không đối thủ nào vượt ổn định; (3) đề xuất **FT-Transformer (Feature Tokenizer + Transformer)**, chuyển mọi đặc trưng thành embedding rồi dùng transformer, đạt kết quả tốt nhất trong số mạng nơ-ron ở phần lớn bộ dữ liệu. Khi so với GBDT, kết luận là **vẫn không có giải pháp nào vượt trội toàn diện**: GBDT còn thắng ở một số bộ dữ liệu sau khi tinh chỉnh.

## 2. Bối cảnh

- GBDT (XGBoost, LightGBM, CatBoost) là "state of the art nông", lựa chọn hàng đầu trong các cuộc thi. Mạng nơ-ron hấp dẫn vì hứa hẹn hiệu năng cao hơn và nhất là cho phép xây pipeline đa phương thức huấn luyện đầu cuối.
- Ba họ mô hình sâu cho bảng: **cây khả vi** (NODE, Tree Ensemble Layer, Neural Decision Forest: làm "mềm" hàm chia nút để lan truyền gradient), **mô hình dựa trên attention** (TabNet, TabTransformer, AutoInt), và **mô hình tương tác nhân** (DCN V2, PNN... từ lĩnh vực hệ gợi ý). Ngoài ra còn SNN (SELU) và GrowNet (boosting các MLP yếu).
- Thiếu một benchmark chung kiểu ImageNet hay GLUE, và MLP vẫn là baseline đơn giản nhưng không phải lúc nào cũng khó đánh bại. Giả thuyết: các khối kiến trúc đã được nghiên cứu kỹ ở lĩnh vực khác có thể chưa được khai thác tốt cho bảng.

## 3. Các mô hình

**Ký hiệu:** tập dữ liệu D = {(xᵢ, yᵢ)} với đặc trưng số và hạng mục; tách thành train/val/test không giao nhau (val dùng cho early stopping và tinh chỉnh); ba loại tác vụ: nhị phân, đa lớp, hồi quy.

- **MLP:** các khối Dropout(ReLU(Linear(x))) xếp chồng và một lớp Linear cuối.
- **ResNet** (bản đơn giản): ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x)))))); chuẩn hóa chỉ đặt trong nhánh phần dư, giữ một "đường thẳng" từ đầu vào đến đầu ra (có lợi cho tối ưu); đầu dự đoán Linear(ReLU(BatchNorm(x))). Kỳ vọng tốt hơn MLP khi biểu diễn sâu có ích.
- **FT-Transformer:**
  - **Feature Tokenizer:** mỗi đặc trưng xⱼ thành embedding Tⱼ = bⱼ + fⱼ(xⱼ) ∈ ℝᵈ, với bⱼ là bias riêng của đặc trưng j. Đặc trưng số: fⱼ = xⱼ · Wⱼ (nhân từng phần tử với vector học được); đặc trưng hạng mục: bảng tra Wⱼ ∈ ℝ^{Sⱼ×d} (tương đương one-hot nhân ma trận). Kết quả T ∈ ℝ^{k×d}.
  - **Transformer:** thêm token [CLS] vào đầu chuỗi rồi áp dụng L lớp transformer (PreNorm; bỏ chuẩn hóa đầu tiên ở lớp đầu để huấn luyện tốt). Mỗi lớp = Multi-Head Self-Attention + FFN với phần dư.
  - **Dự đoán:** ŷ = Linear(ReLU(LayerNorm(T_L[CLS]))). Dùng ReGLU (hoặc ReLU), 8 đầu attention không tinh chỉnh.
  - **Hạn chế đã nêu:** cần tài nguyên hơn ResNet; không scale tốt khi số đặc trưng rất lớn do độ phức tạp bậc hai của attention theo số đặc trưng (có thể giảm bằng xấp xỉ MHSA); dùng rộng rãi có thể làm tăng phát thải CO₂; có thể chưng cất sang kiến trúc đơn giản hơn.
- **Các mô hình khác so sánh:** SNN, NODE, TabNet, GrowNet, DCN V2, AutoInt (mạng); XGBoost, CatBoost (GBDT).

## 4. Thiết lập thí nghiệm

- **Phạm vi:** chỉ so sánh **kiến trúc**; không dùng pretraining, hàm mất mát phụ, tăng cường dữ liệu, chưng cất, warmup hay giảm learning rate, để đo tác động của thiên kiến quy nạp.
- **11 bộ dữ liệu (một chia train/val/test cố định cho mọi thuật toán):**

| Bộ | Viết tắt | #mẫu | #đặc trưng số | #hạng mục | Thước đo | #lớp |
|---|---|---|---|---|---|---|
| California Housing | CA | 20.640 | 8 | 0 | RMSE | – |
| Adult | AD | 48.842 | 6 | 8 | Acc. | 2 |
| Helena | HE | 65.196 | 27 | 0 | Acc. | 100 |
| Jannis | JA | 83.733 | 54 | 0 | Acc. | 4 |
| Higgs (98K) | HI | 98.050 | 28 | 0 | Acc. | 2 |
| ALOI | AL | 108.000 | 128 | 0 | Acc. | 1000 |
| Epsilon | EP | 500.000 | 2000 | 0 | Acc. | 2 |
| Year | YE | 515.345 | 90 | 0 | RMSE | – |
| Covertype | CO | 581.012 | 54 | 0 | Acc. | 7 |
| Yahoo | YA | 709.877 | 699 | 0 | RMSE | – |
| Microsoft | MI | 1.200.192 | 136 | 0 | RMSE | – |

  (Yahoo và Microsoft là bài toán xếp hạng, xử lý theo cách pointwise như hồi quy. Kích thước train/val/test: ví dụ CA 13.209/3.303/4.128; MI 723.412/241.521/235.259.)
- **Tiền xử lý:** biến đổi **quantile** (scikit-learn) cho mọi mô hình sâu, chuẩn hóa thường cho Helena và ALOI, dữ liệu thô cho Epsilon; thêm nhiễu N(0, 1e−3) vào đặc trưng số khi tính phân vị để xử lý đặc trưng ít giá trị; chuẩn hóa nhãn hồi quy. GBDT không tiền xử lý.
- **Tinh chỉnh:** Optuna (TPE, tối ưu Bayes) trên tập validation; 100 vòng cho A = {CA, AD, HE, JA, HI} và 50 vòng cho B = {AL, YE, CO, MI}; Epsilon dùng vài cấu hình mặc định theo quy tắc co giãn do tinh chỉnh toàn phần quá tốn; Yahoo dùng cấu hình mặc định.
- **Đánh giá:** mỗi cấu hình chạy 15 seed; ensemble tạo bằng cách chia 15 mô hình thành 3 nhóm 5 và lấy trung bình dự đoán; mức "top" xác định bằng kiểm định Wilcoxon một phía (p = 0,01).
- **Huấn luyện:** cross-entropy / MSE; AdamW (Adam cho TabNet và GrowNet); không lịch learning rate; patience = 16; batch size theo bộ dữ liệu.
- **FT-Transformer mặc định (Bảng 12):** 3 lớp, embedding 192, 8 đầu, FFN ReGLU hệ số 4/3, dropout attention 0,2, FFN 0,1, residual 0; AdamW lr 1e−4, weight decay 1e−5 (không áp dụng cho Feature Tokenizer, LayerNorm, bias); khoảng 929K tham số cho 100 đặc trưng số. Đây là một "educated guess", không tinh chỉnh nhiều.

## 5. Kết quả

### 5.1. Các mô hình sâu (Bảng 2, trung bình 15 seed)

| Mô hình | CA↓ | AD↑ | HE↑ | JA↑ | HI↑ | AL↑ | EP↑ | YE↓ | CO↑ | YA↓ | MI↓ | Hạng TB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TabNet | 0,510 | 0,850 | 0,378 | 0,723 | 0,719 | 0,954 | 0,8896 | 8,909 | 0,957 | 0,823 | 0,751 | 7,5 |
| SNN | 0,493 | 0,854 | 0,373 | 0,719 | 0,722 | 0,954 | 0,8975 | 8,895 | 0,961 | 0,761 | 0,751 | 6,4 |
| AutoInt | 0,474 | 0,859 | 0,372 | 0,721 | 0,725 | 0,945 | 0,8949 | 8,882 | 0,934 | 0,768 | 0,750 | 5,7 |
| GrowNet | 0,487 | 0,857 | – | – | 0,722 | – | 0,8970 | 8,827 | – | 0,765 | 0,751 | 5,7 |
| MLP | 0,499 | 0,852 | 0,383 | 0,719 | 0,723 | 0,954 | 0,8977 | 8,853 | 0,962 | 0,757 | 0,747 | 4,8 |
| DCN2 | 0,484 | 0,853 | 0,385 | 0,716 | 0,723 | 0,955 | 0,8977 | 8,890 | 0,965 | 0,757 | 0,749 | 4,7 |
| NODE | 0,464 | 0,858 | 0,359 | 0,727 | 0,726 | 0,918 | 0,8958 | 8,784 | 0,958 | 0,753 | 0,745 | 3,9 |
| ResNet | 0,486 | 0,854 | 0,396 | 0,728 | 0,727 | 0,963 | 0,8969 | 8,846 | 0,964 | 0,757 | 0,748 | 3,3 |
| **FT-Transformer** | **0,459** | 0,859 | 0,391 | 0,732 | 0,729 | 0,960 | 0,8982 | 8,855 | **0,970** | 0,756 | 0,746 | **1,8** |

(Ghi chú của bảng gốc: đây là bảng thứ tự; hạng trung bình tính trên hạng từng bộ dữ liệu.) Kết luận:
- MLP vẫn là kiểm tra sức khỏe (sanity check) tốt; **tinh chỉnh làm MLP và ResNet cạnh tranh**, nên baseline cần được tinh chỉnh.
- **ResNet** là baseline mà không đối thủ nào vượt ổn định.
- **FT-Transformer** tốt nhất ở đa số bộ dữ liệu, hạng trung bình 1,8 so với 3,3 của ResNet.
- **NODE** là mô hình khác duy nhất mạnh ở nhiều bộ dữ liệu, nhưng thua ResNet ở sáu bộ (Helena, Jannis, Higgs, ALOI, Epsilon, Covertype), phức tạp hơn, và thực chất đã là một ensemble.

### 5.2. Ensemble của mô hình sâu (Bảng 3)
FT-Transformer và ResNet **hưởng lợi nhiều hơn từ ensemble**: khi so ensemble thì FT-Transformer vượt NODE và khoảng cách giữa ResNet và NODE thu hẹp mạnh. Ví dụ ensemble FT-Transformer: CA 0,448, HE 0,398, JA 0,739, AL 0,967, CO 0,973, YA 0,747, MI 0,743.

### 5.3. So với GBDT (Bảng 4, ensemble ba nhóm; không xét tốc độ và phần cứng)

| Cấu hình | CA↓ | AD↑ | HE↑ | JA↑ | HI↑ | AL↑ | EP↑ | YE↓ | CO↑ | YA↓ | MI↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Mặc định* XGBoost | 0,462 | 0,874 | 0,348 | 0,711 | 0,717 | 0,924 | 0,8799 | 9,192 | 0,964 | 0,761 | 0,751 |
| CatBoost | 0,428 | 0,873 | 0,386 | 0,724 | 0,728 | 0,948 | 0,8893 | 8,885 | 0,910 | 0,749 | 0,744 |
| FT-Transformer | 0,454 | 0,860 | 0,395 | 0,734 | 0,731 | 0,966 | 0,8969 | 8,727 | 0,973 | 0,747 | 0,742 |
| *Tinh chỉnh* XGBoost | 0,431 | 0,872 | 0,377 | 0,724 | 0,728 | – | 0,8861 | 8,819 | 0,969 | 0,732 | 0,742 |
| CatBoost | 0,423 | 0,874 | 0,388 | 0,727 | 0,729 | – | 0,8898 | 8,837 | 0,968 | 0,740 | 0,741 |
| ResNet | 0,478 | 0,857 | 0,398 | 0,734 | 0,731 | 0,966 | 0,8976 | 8,770 | 0,967 | 0,751 | 0,745 |
| FT-Transformer | 0,448 | 0,860 | 0,398 | 0,739 | 0,731 | 0,967 | 0,8984 | 8,751 | 0,973 | 0,747 | 0,743 |

- **Mặc định:** ensemble FT-Transformer mặc định thắng GBDT ở hầu hết bộ dữ liệu, trừ California Housing và Adult. Ensemble FT-T mặc định gần ngang ensemble FT-T tinh chỉnh ("hiệu năng mạnh ngay ra hộp").
- **Tinh chỉnh:** GBDT **thắng ở California Housing, Adult, Yahoo**, và khoảng cách đủ lớn để kết luận mạng sâu không vượt toàn diện. Tác giả cảnh báo: việc mạng sâu thắng ở đa số bộ dữ liệu **không có nghĩa là tốt hơn** mà chỉ cho thấy benchmark hơi thiên về bài toán "thân thiện với DL".
- GBDT không phù hợp cho đa lớp với số lớp lớn: kém ở Helena, và **không tinh chỉnh nổi ở ALOI** (1000 lớp) vì quá chậm.
- Khuyến nghị: nghiên cứu DL muốn vượt GBDT nên tập trung vào các bộ dữ liệu GBDT đang thắng, nhưng vẫn giữ các bài "thân thiện DL" để không thoái hóa.

### 5.4. Tính chất "đa năng" của FT-Transformer (4.6)
FT-Transformer thu lợi thế so với ResNet **đúng ở những bộ mà GBDT thắng ResNet** (CA, AD, CO, YA, MI) và ngang ResNet ở các bộ còn lại. Tức FT-Transformer cạnh tranh trên mọi tác vụ, còn GBDT và ResNet chỉ tốt ở một số nhóm. Hiện tượng này không do ensemble (thấy cả ở mô hình đơn).

## 6. Phân tích

### 6.1. Khi nào FT-Transformer tốt hơn ResNet (Hình 3, phụ lục G.1)
Tạo các tác vụ hồi quy tổng hợp nội suy giữa hai hàm: y = α·f_GBDT(x) + (1 − α)·f_DL(x), với x ~ N(0, I₁₀₀) (50 đặc trưng đầu dùng sinh nhãn, 50 còn lại là nhiễu), f_GBDT là trung bình 30 cây quyết định ngẫu nhiên (độ sâu tối đa 10, 100 nút), f_DL là MLP ba lớp ẩn kích thước 256 khởi tạo ngẫu nhiên và cố định. Train/val/test = 500.000/50.000/100.000. Kết quả: ResNet và FT-Transformer ngang nhau và thắng CatBoost ở nhiệm vụ "kiểu MLP" (α thấp); khi nhiệm vụ nghiêng về cây (α cao), **hiệu năng ResNet giảm rõ**, còn FT-Transformer giữ cạnh tranh trên cả dải. Tức FT-Transformer xấp xỉ tốt hơn các hàm kiểu cây.

### 6.2. Ablation (Bảng 5)
So FT-T với AutoInt (cũng dùng embedding + self-attention nhưng khác: không có bias đặc trưng, backbone không phải transformer chuẩn, không dùng [CLS]) và với FT-T không có bias đặc trưng:

| | CA↓ | HE↑ | JA↑ | HI↑ | AL↑ | YE↓ | CO↑ | MI↓ |
|---|---|---|---|---|---|---|---|---|
| AutoInt | 0,474 | 0,372 | 0,721 | 0,725 | 0,945 | 8,882 | 0,934 | 0,750 |
| FT-T không bias | 0,470 | 0,381 | 0,724 | 0,727 | 0,958 | 8,843 | 0,964 | 0,751 |
| FT-T | 0,459 | 0,391 | 0,732 | 0,729 | 0,960 | 8,855 | 0,970 | 0,746 |

Cho thấy backbone transformer tốt hơn AutoInt và **bias theo đặc trưng là cần thiết**.

### 6.3. Tầm quan trọng đặc trưng từ attention (Bảng 6)
Trung bình bản đồ attention của [CLS] qua các head và lớp làm "độ quan trọng đặc trưng", nhanh vì chỉ cần một lượt truyền xuôi. Đối chiếu với kiểm định hoán vị (Breiman) bằng tương quan hạng: AM đạt 0,81 (CA), 0,77 (HE), 0,78 (JA), 0,91 (HI), 0,84 (AL), 0,92 (YE), 0,84 (CO), 0,86 (MI); Integrated Gradients (IG) đạt 0,84, 0,74, 0,75, 0,72, 0,89, 0,50, 0,90, 0,56. AM tương đương IG nhưng rẻ hơn rất nhiều, nên được xem là lựa chọn hiệu quả chi phí (không có nghĩa trùng với IG).

## 7. Phụ lục: chi tiết đáng nhớ

- **Thời gian huấn luyện (Bảng 10):** FT-Transformer tốn gấp 0,9–3,5 lần ResNet ở hầu hết bộ dữ liệu, đỉnh **13,8 lần ở Yahoo** (699 đặc trưng; 12.712 giây so với 923 giây); trung bình khoảng 2,3 lần ở nhiều bộ khác. Đây là hệ quả bậc hai theo số đặc trưng.
- **Ảnh hưởng ngân sách tinh chỉnh (Bảng 11):** với 15 phút đến 6 giờ trên California Housing, Adult, Higgs Small: FT-T đạt hiệu năng tốt chỉ sau vài cấu hình ngẫu nhiên; kéo dài tinh chỉnh cho mô hình khác không cải thiện đáng kể, nên lợi thế của FT-T không do ngân sách tinh chỉnh dài hơn.
- **Không gian siêu tham số FT-T (Bảng 13):** 1–4 lớp (nhóm A) hoặc 1–6 (nhóm B), embedding 64–512, dropout attention/FFN U[0, 0,5], hệ số FFN U[2/3, 8/3], lr log-uniform [1e−5, 1e−3], weight decay [1e−6, 1e−3]; 100 hoặc 50 vòng.
- **XGBoost:** max depth 3–10 (nhóm A) hoặc 6–10, min child weight, subsample, learning rate log-uniform [1e−5, 1], v.v.; 100 vòng. **CatBoost:** early stopping 50, od-pval 0,001, tối đa 2000 vòng; tinh chỉnh trên GPU, đánh giá cuối trên CPU vì CPU cho kết quả tốt hơn với phiên bản thư viện đó.
- Cho ResNet: kiểm chứng nhiều biến thể, "đường chính rõ ràng" tốt hơn và dễ huấn luyện sâu hơn.

## 8. Hạn chế

Tác giả nêu: FT-Transformer tốn tài nguyên và khó scale với rất nhiều đặc trưng; benchmark có thể hơi thiên về bài toán thân thiện với DL; các thực hành chung (pretraining, tăng cường dữ liệu, ...) cố tình không dùng.

Nhận xét thêm của mình:
1. Chỉ **11 bộ dữ liệu** lớn (hầu hết hơn 20.000 mẫu) và **một chia cố định** cho mỗi bộ; không có bộ dữ liệu nhỏ, trong khi TFM nhắm đến chính vùng đó (các bài sau như bài 33 McElfresh, bài 29 Grinsztajn chỉ ra kết luận phụ thuộc vào kích thước dữ liệu).
2. Chênh lệch giữa FT-T, ResNet và GBDT nhiều chỗ chỉ ở chữ số thứ ba; các kết luận về "top" dựa trên Wilcoxon một phía p = 0,01 trên 15 seed.
3. Không xét tốc độ suy luận, hiệu chỉnh xác suất, nhiễu nhãn hay mất cân bằng.
4. Đây là mô hình huấn luyện **riêng cho từng bộ dữ liệu** (nhiều giờ tinh chỉnh), đối lập hoàn toàn với ICL của TabPFN (không huấn luyện).

## 9. Liên hệ với bài Nature và đề tài

- **Baseline học sâu:** trong bài Nature và các TFM sau này (TabICLv2, TabDPT), ResNet/FT-Transformer/MLP là nhóm so sánh "học sâu thường". Bài này cho biết cách tinh chỉnh và công bằng của chúng.
- **Ý tưởng Feature Tokenizer** (mỗi cột thành một token) được TabPFNv2 tiếp nối ở mức "token theo ô", TabICL ở mức embedding cột; hiểu bài này giúp hiểu vì sao attention theo cột/hàng là cách làm chuẩn.
- Cho thí nghiệm cuối kỳ: có thể dùng FT-Transformer (hoặc ResNet) làm đại diện "mạng sâu huấn luyện riêng" bên cạnh XGBoost/CatBoost và TabPFN.

## 10. Từ khóa cần nhớ

- **Feature Tokenizer:** đặc trưng số = x·W + b theo cột; hạng mục = bảng tra + bias theo cột; token [CLS] cuối cùng cho dự đoán.
- **ResNet đơn giản:** BatchNorm trong nhánh phần dư; baseline mạnh bị bỏ qua.
- **Kết luận chính:** FT-T tốt nhất trong các mạng sâu (hạng 1,8), nhưng **không có người thắng tuyệt đối giữa DL và GBDT**; GBDT còn thắng CA, AD, Yahoo sau tinh chỉnh.
- **Số liệu:** 11 bộ dữ liệu, 15 seed, Optuna 100/50 vòng; FT-T mặc định 3 lớp, 192 chiều, ~929K tham số; chậm gấp tới 13,8 lần ResNet ở Yahoo.
- **Phát hiện phụ:** lợi thế của FT-T so với ResNet đúng ở các bộ GBDT mạnh; bias theo đặc trưng cần thiết; attention map ≈ độ quan trọng đặc trưng.
