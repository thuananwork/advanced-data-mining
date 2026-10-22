# Bài 49. Survey on deep learning with class imbalance

| Mục | Thông tin |
|---|---|
| Nơi đăng | Journal of Big Data, 6:27 (2019), truy cập mở (CC BY 4.0), 54 trang |
| Tác giả | Justin M. Johnson, Taghi M. Khoshgoftaar (Florida Atlantic University, Boca Raton, Mỹ) |
| Mã nguồn | Không (khảo sát) |
| Lưu ý về PDF | Phần "Deep learning background" (giới thiệu MLP, CNN, RNN, autoencoder, DBN, big data) được lược bỏ vì là kiến thức nền, không liên quan đến dữ liệu bảng. Một số bảng lớn (Bảng 13, 14, 16 về CelebA và DOS) bị lệch cột khi trích chữ nên chỉ báo cáo các con số được văn bản xác nhận. |
| Vị trí trong đề tài | Bài tổng quan về **mất cân bằng lớp** cho học sâu; cùng bài 47 (nhãn nhiễu) làm bằng chứng cho khoảng trống nghiên cứu. **Gần như toàn bộ là dữ liệu ảnh và CNN.** |

## 1. Bài này làm gì trong một đoạn

Mất cân bằng lớp (class imbalance) đã được nghiên cứu kỹ hai thập niên với học máy truyền thống, nhưng rất ít công trình thực nghiệm đánh giá nó với học sâu. Tác giả tìm trên Google Scholar và IEEE Xplore (từ khóa như "class imbalance", "class rarity", "skewed data", "deep learning"), loại các bài không có mạng từ hai tầng ẩn trở lên, rồi theo dõi trích dẫn xuôi ngược; sau đó nới lỏng các tiêu chí để có đủ bài (loại các bài chỉ thử mức mất cân bằng thấp, không so với phương pháp mất cân bằng khác, hoặc chỉ dùng một bộ dữ liệu), và chốt **15 công trình (2015–2018)**, gồm bài báo tạp chí, hội nghị và luận văn sinh viên. Các công trình được phân thành **cấp dữ liệu, cấp thuật toán, và lai (hybrid)**. Kết luận: các kỹ thuật truyền thống (lấy mẫu, nhạy cảm chi phí) **chuyển sang học sâu được**, nhưng gần như mọi nghiên cứu tập trung vào thị giác máy tính với CNN, hiếm xét dữ liệu lớn và hiếm xét dữ liệu không phải ảnh, và **không đủ bằng chứng thống kê** để chỉ ra phương pháp nào tốt nhất.

## 2. Nền tảng về mất cân bằng lớp (mục "Class imbalance background")

- **Định nghĩa:** trong phân loại nhị phân, mất cân bằng xảy ra khi lớp thiểu số (thường là lớp quan tâm, lớp dương) ít mẫu hơn đáng kể so với lớp đa số. Khi đó bộ học thường phân loại thiên về lớp đa số vì xác suất tiên nghiệm lớn hơn, nên mẫu thiểu số bị phân loại sai nhiều hơn. Độ chính xác (accuracy) gây hiểu lầm: dữ liệu có lớp dương 1% thì bộ phân loại luôn trả "âm" đạt 99%.
- **Nguồn gốc:** *nội tại* (intrinsic: tần suất tự nhiên như chẩn đoán bệnh, phát hiện gian lận) và *ngoại sinh* (extrinsic: do cách thu thập, lưu trữ).
- Các điểm quan trọng nêu ra:
  - Nếu hai nhóm đều được biểu diễn tốt và không chồng lấn, kết quả tốt bất kể chênh lệch (Krawczyk). Japkowicz: độ nhạy với mất cân bằng **tăng theo độ phức tạp bài toán**; bài tuyến tính tách được không bị ảnh hưởng ở mọi mức mất cân bằng.
  - **Số lượng tuyệt đối mẫu thiểu số quan trọng hơn tỉ lệ**: lớp thiểu số 1% của 1 triệu mẫu vẫn có 10.000 mẫu; còn khi lớp thiểu số hiếm (class rarity, nhỏ tới 0,1%) thì hiệu năng bị tổn hại nhiều hơn.
  - Để so sánh giữa các nghiên cứu, bài dùng **tỉ lệ mất cân bằng cực đại ρ = max_i|C_i| / min_i|C_i|**. Ví dụ lớp lớn nhất 100 mẫu, nhỏ nhất 10 mẫu thì ρ = 10.
- **Thước đo:** từ ma trận nhầm lẫn: Accuracy/Error Rate (không đủ); **Precision** (nhạy mất cân bằng), **Recall/TPR** (không nhạy mất cân bằng nhưng bỏ qua âm giả), **Selectivity/TNR**; **F-Measure** (trung bình điều hòa, hệ số β), **G-Mean = √(TPR × TNR)**, **Balanced Accuracy = (TPR + TNR)/2** (nhạy với lớp thiểu số, vẫn chưa hoàn hảo); **ROC/AUC** (có thể quá lạc quan trên dữ liệu rất lệch, theo Davis & Goadrich nên dùng **đường Precision–Recall**), weighted-AUC. Seliya et al. (22 thước đo, 35 bộ dữ liệu) khuyên dùng **bộ thước đo bổ sung nhau**, ví dụ AUC + Brier Inaccuracy + accuracy.
- **Kỹ thuật học máy truyền thống** (ba nhóm):
  - *Cấp dữ liệu:* random under-sampling (RUS, bỏ ngẫu nhiên mẫu đa số) và random over-sampling (ROS, nhân bản mẫu thiểu số). RUS bỏ thông tin; ROS tăng thời gian huấn luyện và dễ quá khớp. Lấy mẫu có hiểu biết: Near-Miss (kNN), one-sided selection (Tomek links), Wilson's editing; **SMOTE** (nội suy giữa mẫu thiểu số và láng giềng thiểu số gần nhất; biến thể Borderline-SMOTE, Safe-Level-SMOTE); cluster-based over-sampling (xử lý "small disjuncts"). Van Hulse et al. (7 kỹ thuật, 11 thuật toán, 35 bộ, 6 thước đo): kết quả phụ thuộc cả bộ học và thước đo; **RUS tốt nhìn chung**, nhưng không có kỹ thuật nào tốt nhất ở mọi miền.
  - *Cấp thuật toán:* **học nhạy cảm chi phí** (ma trận chi phí c_ij; trực tiếp hoặc meta-learning), **dịch ngưỡng (threshold moving)** với p\* = c10/(c10 + c01). Khó nhất là xác định ma trận chi phí (chuyên gia, kinh nghiệm, hoặc quét trên validation set; tốn kém nếu dữ liệu lớn).
  - *Lai:* lấy mẫu rồi nhạy cảm chi phí/ngưỡng; EasyEnsemble và BalanceCascade (Liu et al.), SMOTEBoost, DataBoost-IM, JOUS-Boost; AdaC1/AdaC2/AdaC3 (Sun, đưa chi phí vào cập nhật trọng số AdaBoost).
- **Vấn đề gradient trong mạng nơ-ron:** Anand et al. (thập niên 1990): với mất cân bằng, thành phần gradient của lớp thiểu số **ngắn hơn nhiều** so với lớp đa số, nên lớp đa số áp đảo gradient cập nhật trọng số; sai số lớp đa số giảm rất nhanh ở những vòng đầu nhưng sai số lớp thiểu số có thể tăng, và mạng kẹt ở chế độ hội tụ chậm.

## 3. Các phương pháp cấp dữ liệu (4 công trình)

### 3.1. ROS trên CIFAR-10 (Hensman & Masko, 2015)
- Tạo 11 phân phối mất cân bằng từ CIFAR-10 (mỗi lớp 6.000 ảnh), kích thước mỗi lớp từ 6% đến 15%; **ρ tối đa chỉ 2,3** (mức rất nhẹ). CNN biến thể AlexNet, huấn luyện 10 epoch, 3 lần chạy trung bình.
- Không lấy mẫu: phân phối nhẹ (Dist. 6, 7) vẫn như cân bằng (Total 0,73), nhưng Dist. 5 và 9 (một lớp áp đảo: airplane 14,5% hoặc truck) cho **mô hình vô dụng (Total 0,10)**, chỉ đoán lớp áp đảo (airplane 1,00, các lớp khác 0,00).
- Với ROS (nhân bản lớp thiểu số đến cân bằng): Total hầu hết 0,72–0,73 trên mọi phân phối, **phục hồi gần hoàn toàn** hiệu năng (Dist. 5: 0,10 → 0,73; Dist. 9: 0,10 → 0,72).
- Hạn chế (tác giả của khảo sát): ρ quá thấp so với thực tế; chỉ 10 epoch nên chưa loại trừ hội tụ chậm.

### 3.2. Học hai pha (Lee et al., WHOI-Plankton)
- 3,4 triệu ảnh plankton, 103 lớp, 5 lớp lớn chiếm 90% dữ liệu; **ρ > 650**; nhiều lớp < 0,1%. Học hai pha: tiền huấn luyện CNN trên dữ liệu bị cắt ngưỡng (RUS các lớp lớn xuống N = 5000), rồi tinh chỉnh với toàn bộ dữ liệu.
- Kết quả (F1 trung bình, Bảng 5; L5 = 5 lớp lớn nhất, Rest = các lớp còn lại):

| Phương pháp | Tất cả lớp | L5 | Rest |
|---|---|---|---|
| (A) Full (CNN trên toàn dữ liệu) | 0,1773 | 0,7773 | 0,1548 |
| (B) Noise (nhân bản bằng nhiễu) | 0,2465 | 0,5409 | 0,3599 |
| (C) Aug (xoay, tỉ lệ, dịch, lật) | 0,2726 | 0,5776 | 0,3700 |
| (D) Thresh (RUS tới 5000) | 0,3086 | 0,6510 | 0,4044 |
| (E) Noise + full | 0,3038 | 0,7531 | 0,2971 |
| (F) Aug + full | 0,3212 | 0,7668 | 0,3156 |
| (G) Thresh + full (đề xuất) | **0,3339** | **0,7791** | 0,3262 |

- Hai pha tăng F1 của nhóm thiểu số từ 0,1548 lên 0,3262 mà giữ F1 nhóm đa số (0,7791). Hạn chế: không nêu số epoch tiền huấn luyện và tiêu chí dừng.

### 3.3. Lấy mẫu động (Pouyanfar et al.)
- Ảnh camera mạng (hơn 10.000 ảnh, 19 khái niệm như intersection, forest, farm; ρ tới 500; 70/20/10 train/val/test). Ý tưởng: lấy mẫu **nhiều hơn** các lớp có F1 thấp và **ít hơn** các lớp có F1 cao ở vòng sau: kích thước mẫu lớp j ∝ (1 − f1_{i,j}) / Σ_k (1 − f1_{i,k}) × N\* (N\* là cỡ lớp trung bình). Kết hợp augmentation thời gian thực + học chuyển giao Inception-V3 + một mô hình thứ hai không lấy mẫu.
- Kết quả (Bảng 6): Basic CNN: Acc 0,649, Avg F1 0,254, WAvg F1 0,630; Deep CNN features + SVM: 0,746/0,528/0,747; TL không augmentation: 0,765/0,432/0,755; TL + augmentation thường: 0,792/0,502/0,779; TL + augmentation cân bằng: 0,759/0,553/0,766; **đề xuất: 0,802/0,599/0,794**. Hạn chế: cần validation set để tính F1 theo lớp (khó khi lớp hiếm); không so với RUS/ROS thuần.

### 3.4. ROS, RUS và học hai pha (Buda et al.)
- MNIST (ρ ∈ [10, 5000]), CIFAR-10 (ρ ∈ [2, 50]), ImageNet (100 lớp, ρ ∈ [10, 100]); gần **23.000 CNN** được huấn luyện; thước đo ROC AUC đa lớp (trung bình AUC một-chống-tất-cả).
- Kết luận của họ: ROS tốt nhất gần như mọi nơi và không bao giờ kém baseline; RUS kém và chỉ ngang ROS khi số lớp thiểu số rất cao (80–90%); học hai pha (ROS hoặc RUS) kém hơn ROS và RUS thuần; dịch ngưỡng (chia đầu ra cho xác suất tiên nghiệm của lớp) cải thiện độ chính xác, **tốt hơn ROS và RUS trên MNIST**, còn trên CIFAR-10 có vài trường hợp ROS thắng; **ROS + dịch ngưỡng** gần như luôn tốt nhất.
- Phê bình của khảo sát: AUC trung bình che giấu cân bằng theo lớp; kết luận "ROS đến cân bằng hoàn toàn" **chưa chắc đúng với dữ liệu lớn hoặc cực kỳ lệch** (nhân bản khối lượng lớn tốn tài nguyên).
- **Tổng kết cấp dữ liệu:** hai nghiên cứu cho thấy ROS đưa lại cải thiện lớn; hai pha có kết quả mâu thuẫn giữa Lee (có lợi) và Buda (kém hơn), vì mức cân bằng tiền huấn luyện và thước đo khác nhau; ROS có thể không phù hợp dữ liệu lớn (RUS giảm thời gian huấn luyện, nên RUS loại bỏ dư thừa/nhiễu lớp có thể có ích). Tất cả đều thử trên ảnh với CNN.

## 4. Các phương pháp cấp thuật toán (9 nghiên cứu)

1. **MFE và MSFE loss (Wang et al.)**: tách MSE thành sai số dương giả trung bình FPE và sai số âm giả trung bình FNE (FPE = (1/N)Σ½(d−y)² trên mẫu âm, FNE tương tự trên mẫu dương); **MFE = FPE + FNE**, **MSFE = FPE² + FNE²** (= ½((FPE+FNE)² + (FPE−FNE)²), nên tối thiểu hóa chênh lệch FPE–FNE). Dữ liệu: 8 bộ nhị phân nhỏ từ CIFAR-100 (3 bộ) và 20 Newsgroup (5 bộ), thường < 2000 mẫu, ρ = 5–20 (mức 20%, 10%, 5% lớp dương). MLP sâu. MFE/MSFE vượt MSE hầu hết cả F-measure lẫn AUC, nổi bật nhất ở mức 5%: ví dụ Household F-measure 0,1143 → 0,2353 (MSFE) ở 5%; nhưng lợi ích nhỏ: AUC tăng trung bình chỉ 0,025 (trung vị 0,008) trên dữ liệu ảnh; không rõ số vòng chạy.
2. **Focal loss (Lin et al.)**: FL(p_t) = −α_t (1 − p_t)^γ log(p_t), hệ số điều biến giảm trọng số mẫu dễ (p_t → 1). Cho phát hiện đối tượng một giai đoạn (RetinaNet, ResNet-FPN), nơi mẫu nền áp đảo (ρ ≈ 1000 hoặc hơn). Huấn luyện bằng CE chuẩn **phân kỳ**; khởi tạo prior π = 0,01 cải thiện (AP 30,2); chọn γ = 2,0, α = 0,25. Trên COCO, RetinaNet (ResNet-101-FPN) AP 39,1 và (ResNeXt-101-FPN) 40,8, vượt một giai đoạn tốt nhất (DSSD513, 33,2) và hai giai đoạn tốt nhất (Faster R-CNN w TDM, 36,8) tương ứng khoảng 7,6 và 4,0 điểm AP (theo văn bản bài; con số tuyệt đối theo Bảng 9 hiển thị); vượt OHEM (AP 32,8 → 36,0). Nemoto et al. dùng focal loss cho phát hiện thay đổi tòa nhà (203.358 ảnh, 200.000 ảnh "không đổi"; ρ tới 900), kết quả **trái ngược**: độ chính xác cao nhất ở γ = 0 (tức CE) trên hai trong ba lớp.
3. **CSDNN (Wang et al.)**: dự đoán tái nhập viện (Barnes-Jewish Hospital; GHW 2565 bản ghi, ρ = 5,3 và 3,8; ORP 700 bản ghi); embedding hạng mục thay one-hot + CNN trích đặc trưng từ chuỗi thời gian + DNN hai tầng ẩn (128, 64); CE có ma trận chi phí cố định (chi phí âm giả gấp 2× với GHW, 1,5× với ORP). Thắng 5 baseline (cây quyết định, SVM, ANN, RF với RUS) trên mọi thước đo trừ NPV; GHW AUC 0,70 so với 0,62 của đối thủ gần nhất; đã triển khai thực tế. Hạn chế: không tách riêng đóng góp của hàm mất mát nhạy chi phí so với embedding và CNN.
4. **CoSen CNN (Khan et al.)**: học đồng thời trọng số mạng và ma trận chi phí (cập nhật theo sai số hiện tại, sai số tổng và độ tách lớp C2C). Trên 6 bộ (MNIST, CIFAR-100, Caltech-101, MIT-67 với 10% các lớp lẻ; DIL ρ = 13; MLC ρ = 76) dùng VGG-16 tiền huấn luyện. Độ chính xác (Bảng 10) CoSen CNN vs baseline CNN: MNIST 98,6 vs 97,6; CIFAR-100 60,1 vs 55,0; Caltech-101 83,2 vs 77,4; MIT-67 56,9 vs 50,4; DIL 72,6 vs 69,5; MLC 68,6 vs 66,1; các phương pháp lấy mẫu (SMOTE, RUS, SMOTE-RSB) kém hơn rõ. Bài chỉ dùng accuracy cho bảng chính (gây hiểu lầm); F1 và G-Mean cũng cải thiện (ví dụ Caltech-101 F1 0,389 → 0,416). Nhận xét: baseline CNN tiền huấn luyện đã mạnh hơn lấy mẫu truyền thống, nhờ trích đặc trưng được học từ hơn 1,3 triệu ảnh.
5. **CSDBN-DE (Zhang et al.)**: DBN + tiến hóa vi phân tìm ma trận chi phí; 42 bộ KEEL cấu trúc (171–3339 mẫu, 6–13 thuộc tính, ρ từ 9,28 đến 128,21); thắng ELM trên **28/42** bộ (66%), 20 lần lặp. Hạn chế: một baseline kiến trúc khác hẳn và chỉ dùng accuracy.
6. **Dịch ngưỡng đầu ra (Buda et al.)**: chia đầu ra mỗi lớp cho xác suất tiên nghiệm ước lượng; chỉ dùng khi suy luận, không tốn thời gian huấn luyện, nhưng không cải thiện khả năng phân biệt của mô hình.
7. **Category centers (Zhang et al.)**: tính tâm lớp trong không gian đặc trưng sâu (VGG-16 tiền huấn luyện), phân loại theo tâm gần nhất. CIFAR-10 (Dist. A ρ = 10, B ρ = 20, C ρ ∈ [1, 20]); mean precision (Bảng 11): baseline CNN 0,779/0,747/0,857; CC trên tầng tích chập cuối 0,824/0,775/0,859; CC trên tầng FC cuối 0,826/0,772/0,865; baseline + over-sampling 0,787/0,770/0,850; CC + over-sampling 0,830–0,831 (A), 0,792–0,796 (B), 0,861–0,862 (C). Tác giả tuyên bố tâm lớp ổn định hơn biên quyết định nhưng không có bằng chứng; phụ thuộc đặc trưng tiền huấn luyện tốt; chỉ một thước đo.
8. **Mạng rất sâu (Ding et al.)**: 11 bài nhị phân nhận dạng đơn vị hành động khuôn mặt (EmotioNet, 450.000 ảnh huấn luyện, 40.000 validation; ρ tới 10.000, AU20 chỉ 0,01%). F1 trung bình (Bảng 12): CNN 6 tầng 0,29; Plain34 0,48; Res10 0,48; Res18 0,49; Res34 0,49; Res50 0,50; AU20 luôn 0,00 ở mọi cấu hình. Mạng 6 tầng không bắt được các lớp lệch nặng; từ 10 tầng trở đi chênh lệch nhỏ; mạng sâu hơn hội tụ nhanh hơn (18 tầng so với 6 tầng). Tốn bộ nhớ và phép nhân ma trận.

**Tổng kết cấp thuật toán:** không đổi dữ liệu, không cần tiền xử lý, ít ảnh hưởng thời gian huấn luyện (so với ROS) nên có thể phù hợp hơn cho dữ liệu lớn; ít cần tinh chỉnh ngoại trừ định nghĩa chi phí; hai phương pháp học chi phí tự động (CoSen, CSDBN-DE). Thiếu so sánh đúng đắn giữa các phương pháp học sâu và với kỹ thuật truyền thống (baseline yếu, thước đo thiếu, thí nghiệm gắn với bài toán cụ thể).

## 5. Các phương pháp lai (3 công trình)

1. **LMLE (Huang et al.)**: lấy mẫu "quintuplet" (anchor + 4 mẫu theo khoảng cách cụm trong lớp và giữa lớp) + **triple-header hinge loss** để học biểu diễn bảo toàn lân cận lớp và tăng phân biệt giữa lớp, rồi phân loại bằng kNN theo cụm. CelebA (160.000 ảnh huấn luyện, 40 thuộc tính nhị phân, ρ tới 49); Balanced Accuracy: LMLE-kNN bằng hoặc tốt hơn Triplet-kNN, PANDA, ANet ở mọi thuộc tính; lợi thế tăng khi mất cân bằng tăng (ví dụ thuộc tính Bald từ 75 lên 90 so với đối thủ gần nhất). Nhược: phức tạp, mỗi lượt đi xuôi gọi 5 CNN, cần gom cụm trước.
2. **DOS (Ando & Huang)**: over-sampling trong **không gian đặc trưng sâu**: lớp dưới học embedding (micro-cluster loss kéo đặc trưng về trung bình láng giềng cùng lớp), lớp trên học phân biệt bằng tổng có trọng số CE của k láng giềng cùng lớp; số k thay đổi theo lớp (k > 0 cho lớp thiểu số). Năm bộ ảnh (MNIST, MNIST-back-rot, SVHN, CIFAR-10, STL-10); với tỉ lệ giảm lớp 90%, 95%, 99% (ρ tới 100) DOS vượt CNN và CNN-CL (kNN trên đặc trưng sâu) và ưu thế rõ rệt hơn khi mất cân bằng tăng (lớp thiểu số MNIST-back-rot F1 0,43 → 0,66 và SVHN 0,42 → 0,64 ở mức giảm 99%). Thí nghiệm với k: k = 10 bắt đầu giảm hiệu năng lớp thiểu số. Còn tốt hơn baseline cả trên dữ liệu cân bằng.
3. **CRL (Dong et al.)**: hàm mất mát điều chỉnh lớp (class rectification loss) kết hợp khai thác mẫu khó từ các lớp thiểu số (lớp thiểu số = các lớp nhỏ nhất cộng lại tối đa nửa dữ liệu; mini-batch ≤ 50% lớp thiểu số): L_bln = α L_crl + (1 − α) L_ce, α = η·Δimb tỉ lệ thuận với mức mất cân bằng. Ba bộ: CelebA (ρ = 49), X-Domain (245.467 ảnh thời trang, ρ > 4000), CIFAR-100 (ρ tới 20). Mean sensitivity trên CelebA: CRL hơn DeepID2 (không xử lý mất cân bằng) 6% và hơn LMLE 3%; ROS và học nhạy chi phí ngang nhau, hơn dịch ngưỡng 3% và RUS 4%; RUS thường kém nhất ở thuộc tính ρ ≥ 6; LMLE tốt hơn CRL ở mức mất cân bằng thấp (ví dụ thuộc tính Attractive hơn 7%). Trên X-Domain, CRL hơn LMLE gần 5%, hơn 6% so với học nhạy chi phí, dịch ngưỡng và ROS. Trên CIFAR-100 cân bằng, CRL tăng mean sensitivity 3,6% (CifarNet), 1,2% (ResNet32), 0,8% (DenseNet). Huấn luyện **27,2 giờ so với 166,6 giờ** của LMLE (nhanh hơn khoảng 6×).

**Tổng kết lai:** hiệu quả nhưng phức tạp, khó thích nghi bài toán mới. Kết quả CelebA cho thấy các phương pháp khác nhau tốt ở những mức mất cân bằng khác nhau, củng cố yêu cầu đánh giá qua nhiều mức mất cân bằng và độ phức tạp.

## 6. Thảo luận của tác giả (mục "Discussion of surveyed works")

- Bảng 17: tóm tắt phương pháp (ROS, RUS, hai pha, lấy mẫu động; MFE/MSFE, focal, CSDNN, CoSen, CSDBN-DE, dịch ngưỡng, category centers, mạng rất sâu; LMLE, DOS, CRL). Bảng 18: tóm tắt dữ liệu, kích thước lớp nhỏ nhất/lớn nhất và ρ (ví dụ MNIST của Buda: ρ = 5000; EmotioNet: ρ = 10.000; COCO: ρ = 10.000).
- Nhiều tác giả khuyên **ROS** (ROS hơn RUS gần mọi trường hợp), nhưng chưa xét dữ liệu lớn; ngược lại Bauder et al. thấy **RUS hiệu quả hơn ROS** khi phát hiện gian lận trong dữ liệu lớn với học máy truyền thống.
- **80% công trình dùng CNN cho ảnh**; gần một nửa dùng **chỉ một bộ dữ liệu**; chỉ **một phần ba** nêu số lần lặp mỗi thí nghiệm (nghĩa là còn lại không lặp hoặc không nêu, có thể trình bày kết quả đẹp nhất); chưa đến một nửa dùng bộ lớn hơn 100.000 mẫu; chỉ Lee et al. dùng hơn một triệu mẫu.
- **Không thể so sánh trực tiếp** các phương pháp: bộ dữ liệu, mức mất cân bằng và thước đo khác nhau, đôi khi kết quả mâu thuẫn. **Thiếu bằng chứng cho thấy một phương pháp học sâu nào vượt trội.**

## 7. Kết luận và hướng tương lai
Các kỹ thuật truyền thống chuyển sang học sâu thành công; nghiên cứu hầu như chỉ ở thị giác máy tính. Cần: (1) áp dụng các phương pháp mới lên nhiều bộ dữ liệu và mức mất cân bằng; (2) so sánh bằng nhiều thước đo bổ sung và báo cáo bằng chứng thống kê; (3) thử trong bối cảnh dữ liệu lớn và lớp hiếm; (4) kiểm tra trên mạng không tích chập (MLP, RNN); (5) mở rộng sang dữ liệu không phải ảnh.

## 8. Hạn chế (tác giả nêu và suy ra)
1. Chỉ **15 công trình**, tìm kiếm theo từ khóa và hai cơ sở dữ liệu; tác giả đã nới lỏng tiêu chí chọn vì quá ít bài đạt chuẩn nên chất lượng bằng chứng không đồng đều (có cả luận văn sinh viên).
2. **Hầu hết ảnh và CNN**, gần như không có dữ liệu bảng. Chỉ có CSDNN (hai bộ bệnh viện, MLP) và CSDBN-DE (42 bộ KEEL, DBN) là cấu trúc/bảng, với thiết kế thí nghiệm yếu (một baseline khác kiến trúc, chủ yếu accuracy).
3. Khảo sát **dừng ở 2018, không có nội dung về cây tăng cường** (XGBoost/LightGBM/CatBoost) hay **tabular foundation model**: không biết mô hình bảng tiền huấn luyện phản ứng thế nào với mất cân bằng.
4. Đa số thí nghiệm gốc có thước đo đơn lẻ (accuracy hoặc AUC), một lần chạy, ρ thấp (ví dụ Hensman ρ = 2,3).
5. Bài không bàn **kết hợp mất cân bằng với nhãn nhiễu** (đó là mục VII-C của bài 47).
6. Học sâu "nền" ở đây là mạng huấn luyện bằng SGD; cách xử lý mất cân bằng như thay đổi hàm mất mát không áp dụng trực tiếp cho suy luận trong ngữ cảnh của TFM (TabPFN không cập nhật trọng số khi dùng), nên cần thử nghiệm riêng (ý kiến của mình, bài không nói).

## 9. Liên hệ với bài Nature và đề tài
- **Bằng chứng thứ hai cho khoảng trống nghiên cứu** (cùng bài 47): hai khảo sát về nhãn nhiễu và mất cân bằng lớp đều dựa trên ảnh/CNN; không có công trình nào trong đó đánh giá TFM. Benchmark TFM chính (OpenML-CC18, Grinsztajn, TabArena) hoặc loại các bộ lệch nặng hoặc cân bằng lớp hoặc không kiểm soát nhiễu (đã ghi ở bài 36, 29, 5).
- **Chỉ tiêu cần dùng trong thí nghiệm final-term:** không dùng accuracy khi lệch lớp; dùng **AUC, F1, balanced accuracy, G-Mean, PR-AUC** và báo cáo cùng lúc nhiều thước đo (khuyến nghị của Seliya, Davis & Goadrich). Khai báo mức mất cân bằng bằng **ρ** và số mẫu thiểu số tuyệt đối.
- **Chiến lược đối chứng khả thi cho cây:** SMOTE/RUS/ROS, `scale_pos_weight` của XGBoost, `class_weights` của CatBoost, dịch ngưỡng quyết định (rất rẻ và không cần huấn luyện lại; Buda et al.: dịch ngưỡng + ROS tốt nhất trên ảnh). Với TFM có thể thử ROS/RUS trên tập ngữ cảnh và dịch ngưỡng đầu ra.
- Bài 49 và bài 47 cùng gợi ý một câu hỏi nghiên cứu: TFM có **bền hơn** cây tăng cường khi nhãn nhiễu và/hoặc mất cân bằng lớp, đặc biệt trên dữ liệu nhỏ không? Cần thiết lập thí nghiệm với các mức τ (nhiễu nhãn) và ρ (mất cân bằng) có kiểm soát.

## 10. Từ khóa cần nhớ
- **ρ = max|C_i| / min|C_i|**; nội tại vs ngoại sinh; class rarity; số mẫu thiểu số tuyệt đối quan trọng hơn tỉ lệ.
- Ba nhóm: **cấp dữ liệu** (ROS, RUS, hai pha, lấy mẫu động), **cấp thuật toán** (MFE/MSFE, focal loss, nhạy cảm chi phí, dịch ngưỡng, category centers, mạng sâu), **lai** (LMLE, DOS, CRL).
- Thước đo: F-Measure, G-Mean, Balanced Accuracy, AUC, PR curve; **ROS nhìn chung tốt nhất trên ảnh** (Buda: ~23.000 CNN), **ROS + dịch ngưỡng** mạnh nhất; focal loss γ = 2, α = 0,25; CRL nhanh hơn LMLE 6×.
- 80% công trình dùng CNN + ảnh; gần một nửa chỉ một bộ dữ liệu; một phần ba nêu số lần lặp.
