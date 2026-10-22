# Bài 11. Position: Why Tabular Foundation Models Should Be a Research Priority

| Mục | Thông tin |
|---|---|
| Nơi đăng | ICML 2024 (41st ICML, Vienna, PMLR 235); arXiv:2405.01147v2, 2/6/2024 |
| Tác giả | Boris van Breugel (Cambridge), Mihaela van der Schaar (Cambridge, Alan Turing Institute) |
| Loại bài | **Position paper** (bài lập trường): không có thí nghiệm hay bảng kết quả, chỉ lập luận, tổng quan và kêu gọi nghiên cứu |
| Lưu ý về PDF | Văn bản trích sạch; Hình 1 (số bài báo theo modality) là biểu đồ, chỉ lấy các con số đếm ghi trong Phụ lục A. |
| Vị trí trong đề tài | Bài **động lực / khung khái niệm** cho cả mảng TFM: định nghĩa "Large Tabular Model (LTM)", bốn tiêu chí thiết kế D1–D4 và danh sách thách thức. Nên đọc đầu tiên khi viết phần giới thiệu của báo cáo. |

## 1. Bài này làm gì trong một đoạn

Bài lập luận rằng nghiên cứu mô hình nền (foundation model) đang dồn gần hết vào **văn bản và ảnh**, còn **dữ liệu bảng**, vốn là dạng dữ liệu thống trị trong y tế, tài chính, khoa học và chính sách, gần như bị bỏ quên. Tác giả đề xuất thuật ngữ **Large Tabular Model (LTM)** cho mô hình nền cho dữ liệu bảng, nêu bốn tiêu chí thiết kế (D1–D4), điểm lại các công trình hiện có (chưa công trình nào thỏa mãn đầy đủ), đưa ra các ứng dụng (AI có trách nhiệm, khoa học, trợ lý cho nhà khoa học dữ liệu), thách thức (xây mô hình sinh, quy mô, đa dạng dữ liệu, thiên kiến, đánh giá) và so sánh tác động LTM với LLM trên năm khía cạnh. Mục đích là "dịch chuyển ưu tiên của cộng đồng ML một chút", thuyết phục thêm nhà nghiên cứu theo hướng LTM.

## 2. Luận điểm động lực (Phần 1)

### 2.1. Vì sao nên quan tâm đến bảng
1. **Bảng có mặt khắp nơi:** hồ sơ bệnh án điện tử, điều tra dân số, an ninh mạng, chấm điểm tín dụng, tài chính, khoa học tự nhiên; nghiên cứu định lượng dựa trên các bộ dữ liệu bảng, từ đó ảnh hưởng đến tri thức và chính sách. Các cuộc thi (Kaggle, KDD Cup) chủ yếu là bảng.
2. **Còn nhiều bài toán mở:** hai thập kỷ qua học máy trên bảng không tiến nhanh bằng các modality khác. Các benchmark (Gorishniy 2021, Borisov 2022, Shwartz-Ziv & Armon 2022, Grinsztajn 2022) đều thấy **XGBoost vẫn thuộc nhóm đầu**. Lý do được các tác giả đó nêu: dữ liệu bảng không liên tục, đặc trưng không đồng nhất và nhiều đặc trưng vô ích, không có bất biến không gian để làm prior như CNN cho ảnh. Nhưng bài van Breugel chỉ ra thêm một lý do rõ hơn: **quy mô dữ liệu, mô hình và độ phức tạp tác vụ khác xa** văn bản và ảnh. Ví dụ trong Grinsztajn các tập lớn nhất chỉ 50.000 mẫu, và với các tập đó khoảng cách giữa transformer và cây thu hẹp; trong Borisov bộ lớn nhất 11 triệu mẫu (27 đặc trưng) thì SAINT thắng cây; McElfresh 2023 thấy mạng nơ-ron thắng khi số mẫu tăng, Gorishniy 2021 khi số lớp tăng, Kadra 2021 khi dùng regularization hiện đại. Vì vậy lợi ích của scaling thấy ở modality khác **có thể vẫn đúng ở bảng nhưng chưa từng được thử** vì benchmark quá nhỏ.
3. **Con người không giỏi đọc bảng:** khác văn bản và ảnh, nơi mô hình nền cố gắng đạt trình độ con người, ở bảng mô hình có thể vượt xa năng lực tự nhiên của chúng ta. Ví dụ: mô hình huấn luyện trên bảng có đặc trưng (A, B) và (B, C) có thể suy ra quan hệ A–C.
4. **Chi phí:** phát triển LTM tiên tiến vẫn **trong tầm tính toán** của nhiều nhóm nghiên cứu, khác với chi phí huấn luyện LLM.

### 2.2. Vì sao LTM bị bỏ quên (ba giả thuyết của tác giả)
- **Dữ liệu:** trước đây thiếu các bộ siêu dữ liệu bảng lớn; dữ liệu lộn xộn, không gắn nhãn cho tác vụ cụ thể, cần kiến thức chuyên môn; một số lĩnh vực (y tế) bị loại do riêng tư/sở hữu.
- **Độ khó của ML bảng:** baseline mạnh nên phương pháp mới khó vượt đều, có thể cản trở xuất bản; kèm các thách thức riêng (xem D1–D4).
- **Nhận thức của con người:** kết quả văn bản/ảnh đánh giá được bằng mắt, đẹp, gây ấn tượng với công chúng; bảng thì khô khan, khó kiểm tra bằng mắt, nên ít thu hút (kể cả các chỉ số FID/IS của ảnh còn có vấn đề).

## 3. Định nghĩa LTM và tiêu chí thiết kế (Phần 2)

- **Foundation model** theo Bommasani et al.: mô hình huấn luyện trên dữ liệu rộng (thường tự giám sát, quy mô lớn) và thích nghi được cho nhiều tác vụ hạ nguồn. Báo cáo gốc hơn 200 trang không đề cập mô hình nền cho bảng. **LTM = mô hình nền cho bảng**, định nghĩa lỏng, không ràng buộc loại mô hình.
- Phạm vi: bài chỉ xét dạng đơn giản nhất, **các bảng độc lập đơn** (không xét cơ sở dữ liệu quan hệ hay chuỗi thời gian).

**Bốn tiêu chí (desiderata) cho LTM:**

| Mã | Tiêu chí | Ý nghĩa |
|---|---|---|
| D1 | Cột kiểu hỗn hợp | Xử lý số, hạng mục, thời gian, giá trị thiếu. |
| D2 | Mô hình hóa liên bộ dữ liệu | Một LTM huấn luyện trên nhiều bộ dữ liệu, tức xử lý **không gian đặc trưng không đồng nhất** (số cột và ý nghĩa khác nhau). |
| D3 | Ngữ cảnh văn bản | Tận dụng mô tả bộ dữ liệu, tên cột, tên hạng mục. |
| D4 | Bất biến/tương đẳng với thứ tự cột | Thứ tự cột thường tùy ý, nên f(T(x)) = f(x) (bất biến) hoặc f(T(x)) = T(f(x)) (tương đẳng). |

- **Kiến trúc:** transformer với attention là chủ đạo vì xử lý được số đặc trưng thay đổi và tự thỏa D4 khi không dùng mã hóa vị trí; nhưng các benchmark gợi ý cần cân nhắc kiến trúc khác. Mô hình sinh có điều kiện (cho phép sinh theo một tập con biến và siêu dữ liệu) có thể quan trọng, cho dự đoán ít mẫu, điền khuyết và sinh có điều kiện; với học biểu diễn, mô hình chỉ encoder có thể phù hợp.
- **Dữ liệu (2.2):** các bộ siêu dữ liệu đã tăng quy mô: **WebTables** (10 triệu bảng HTML), bảng Wikipedia (Bhagavatula 2015, có chọn lọc), và **TabLib** (Eggert 2023) với **627 triệu bảng và 827 tỉ token ngữ cảnh**, cỡ so được với dữ liệu ảnh/văn bản lớn, nên dữ liệu không còn là rào cản về mặt quy mô.
- **Benchmark cho LTM (2.3, Phụ lục B):** cần nhiều tác vụ và bộ dữ liệu như MTEB (8 tác vụ, 56 bộ dữ liệu, 112 ngôn ngữ).

## 4. Hiện trạng các công trình gần LTM (Phần 3)

| Nhóm | Công trình | Điểm chính | Hạn chế theo bài |
|---|---|---|---|
| Học biểu diễn | TaBERT (Yin 2020), TaPas/TURL (Deng 2020) | Biến bảng thành "câu", dùng BERT, huấn luyện tái tạo che | Cửa sổ ngữ cảnh 128 token; chủ yếu hiểu bảng, không dự đoán |
| | XTab (Zhu 2023b) | Pretrain liên bảng, số là một token; embedding riêng cho từng hạng mục | Chỉ 100 bộ dữ liệu; bảng tra hạng mục cản trở D2 quy mô lớn |
| | CT-BERT (Ye 2023) | Nhúng token từng đặc trưng bằng LLM đã pretrain, pooling | Chỉ dự đoán; 17k bộ dữ liệu; tối đa 100 đặc trưng |
| Có giám sát | LIFT (Dinh 2022) | Tinh chỉnh LLM trên dữ liệu bảng viết thành câu | Hiệu năng ngang baseline cổ điển |
| | TabFM (Zhang 2023) | Biểu diễn hàng khác, thêm thông tin tác vụ, ICL ít mẫu | |
| | **TabPFN** (Hollmann 2023a) | Transformer pretrain trên dữ liệu **tổng hợp**; được cho là vượt XGBoost trên dữ liệu thật | Gợi ý bảng có nhiều cấu trúc hơn Grinsztajn 2022 giả định |
| Sinh | Yoon 2018 (GAN liên miền) | Dịch giữa các miền qua không gian ẩn chung | Mỗi miền một cặp encoder-decoder, không scale, không thỏa D1, D3, D4 |
| | GReaT (Borisov 2023), Solatorio & Dupriez 2023, Zhao 2023 | Tinh chỉnh LLM để sinh bảng | Chi phí cao, xem dưới |

### Ba nhược điểm của LLM khi sinh dữ liệu bảng (3.1)
1. **Đắt:** tự hồi quy, mỗi hàng hàng trăm đặc trưng nhiều token (số được token hóa ngây thơ) nên huấn luyện và suy luận chậm; giới hạn cửa sổ ngữ cảnh (GReaT không xử lý hơn 100 mẫu ngữ cảnh chiều thấp). CT-BERT chỉ dùng LLM để mã hóa trường rồi dùng transformer nhỏ, giảm độ dài chuỗi về số đặc trưng và tăng tốc nhờ cache.
2. **Kém với biến liên tục:** số bị chia thành chuỗi token (1,89 → 1 → . → 89) nên mất tiên nghiệm liên tục (1,89 và 1,90 gần nhau, 1,89 và 90,89 chỉ khác một token). Hình 2 minh họa sinh một biến Gaussian chuẩn bằng LLM cần phân phối xác suất token rất phức tạp ở từng bước (từ vựng chỉ 102 token: "−", ".", "00".."99"). Cải thiện bằng tokenization số (nhúng số thành vector cố định) lại biến số thành ước lượng điểm. Mô hình khuếch tán hỗ trợ biến liên tục tự nhiên hơn.
3. **Hiệu chỉnh kém:** LLM sinh kém từ phân phối đơn giản (ví dụ phân phối đều; Renda 2023), phù hợp với việc hiệu chỉnh kém đã biết.

**Tóm tắt (takeaway):** có làn sóng nghiên cứu huấn luyện liên bảng, nhiều công trình dựa vào LLM nhưng không hiệu quả với cột số; xây **LTM sinh** là thách thức lớn nhất.

## 5. Ứng dụng và tác động (Phần 4)

### 5.1. Hai kiểu thích nghi
- **Thích nghi trực tiếp:** tinh chỉnh LTM hoặc đưa dữ liệu làm ngữ cảnh ICL; tiện, rẻ, đo được qua hiệu năng hạ nguồn.
- **Thích nghi gián tiếp:** dùng LTM sinh bộ dữ liệu tổng hợp để tăng cường dữ liệu thật rồi huấn luyện mô hình hạ nguồn; phức tạp hơn nhưng có thể cải thiện dữ liệu có chủ đích (thêm mẫu cho nhóm thiểu số), chia sẻ được, và người dùng hạ nguồn không cần LTM. Ví dụ minh họa dùng chính **phân loại trên tập nhỏ mất cân bằng**. Lưu ý chú thích: cách gián tiếp khá riêng của bảng, ở văn bản thì vô nghĩa.

### 5.2. LTM cho AI có trách nhiệm
- **Hòa nhập và đại diện:** nhóm thiểu số thiếu dữ liệu; LTM few-shot và sinh dữ liệu tổng hợp cho nhóm thiểu số (giống SMOTE hay mô hình sinh sâu nhưng thực tế hơn nhờ tri thức nền).
- **Độ bền qua mô phỏng ngoài miền:** sinh dữ liệu mô phỏng cho kịch bản hiếm/lệch phân phối để kiểm thử và giám sát sau triển khai; các phương pháp hiện nay bị hạn chế bởi không sinh được dữ liệu ngoài phân phối và cần nhiều dữ liệu.
- **Riêng tư, dân chủ hóa dữ liệu, tái lập:** LTM pretrain trên dữ liệu mở cần ít thông tin từ dữ liệu riêng hơn nên giảm ngân sách riêng tư; dữ liệu tổng hợp có trade-off riêng tư–tiện ích tốt hơn; giúp công bố nhiều bộ dữ liệu hơn (y tế).

### 5.3. LTM cho khoa học
- **Meta-analysis:** tự động hóa hợp nhất dữ liệu nhiều nghiên cứu (đồng bộ định dạng, tên cột, hạng mục).
- **Cầu nối các bộ dữ liệu:** tìm và ghép các bảng không khớp trực tiếp, như SQL join tự động dựa trên toàn bộ cơ sở dữ liệu huấn luyện.
- **Trợ lý cho nhà khoa học dữ liệu:** làm sạch, phân tích khám phá, tìm dữ liệu liên quan, truy vấn SQL, thống kê tự động, trực quan hóa.
- **Cơ sở tri thức:** LTM trở thành kho tri thức, trả lời câu hỏi dựa trên dữ liệu, và có thể **chưng cất prior Bayes**.

### 5.4. LTM và dữ liệu phi bảng
Làm nền cho chuỗi thời gian và dữ liệu quan hệ; đa phương thức (D3 gợi ý dùng bộ mã hóa LLM để lấy ngữ cảnh; LTM có thể kết hợp với mô hình đa phương thức mà hiện nay phần lớn bỏ qua bảng).

## 6. Thách thức (Phần 5)

- **Xây LTM sinh:** mô hình hóa phân phối qua bộ dữ liệu (D2), kiểu (D1), ngữ cảnh (D3) vừa khó về khái niệm vừa về kiến trúc.
- **Quy mô:** các mô hình hiện nay chỉ huấn luyện trên ít bộ dữ liệu (khoảng 100) và đánh giá trên một tác vụ giám sát nên chưa biết khái quát hóa sang bộ/tác vụ mới. Mở rộng có thể khó hơn modality khác vì dữ liệu "bẩn" (định dạng khác nhau, thiếu, nhiễu, đã tiền xử lý).
- **Đa dạng và chất lượng dữ liệu:** TabLib rất lớn nhưng chưa biết đủ đa dạng. Ở bảng, kém đại diện đáng lo hơn vì (i) hầu hết bảng không liên quan tới nhau (không gian đặc trưng và ngữ cảnh khác nhau), (ii) nhiều dữ liệu độc quyền/riêng tư (y tế) không có trong TabLib. Hy vọng: LLM nhỏ huấn luyện trên dữ liệu chọn lọc (Gunasekar 2023, "Textbooks are all you need") vẫn tốt, nên LTM chuyên biệt cho miền nhỏ có kho dữ liệu (bảng Wikipedia, biểu hiện gen) khả thi.
- **Đánh giá:** phân biệt nội tại và ngoại tại (Bommasani); đánh giá ngoại tại có thể không đại diện, nội tại khó vì mô hình nền được định nghĩa qua tính thích nghi; **kiểm tra bằng mắt vô ích với bảng**. Cho đến khi có chỉ số nội tại tốt, đánh giá chủ yếu là ngoại tại trên nhiều tác vụ hạ nguồn. Còn phải đo ghi nhớ (rò rỉ dữ liệu riêng/bản quyền) với ý tưởng từ chỉ số khái quát hóa của mô hình sinh và riêng tư vi phân.
- **Thiên kiến:** LTM có thể sao chép/khuếch đại thiên kiến trong dữ liệu (ví dụ Boston Housing có đặc trưng phân biệt đối xử; thiên kiến công bố); cần nghiên cứu song song và có biện pháp bảo vệ khi công bố.

## 7. So sánh tác động LTM và LLM (Phần 6)

| Khía cạnh | LTM so với LLM |
|---|---|
| 1. Sử dụng công chúng/thương mại | LTM quan trọng trong ngành dùng nhiều dữ liệu (tài chính, giáo dục, chính phủ) nhưng nhỏ hơn LLM |
| 2. Sử dụng khoa học | Bảng có lẽ là modality lớn nhất trong khoa học; LLM hiện gặp vấn đề hiệu quả và số học với bảng |
| 3. Lợi ích công | LLM: giáo dục, y tế cá nhân, luật; LTM: dữ liệu hòa nhập, mô hình bền, riêng tư, tái lập, tri thức đa ngành. Cả hai lớn nhưng ở lĩnh vực khác nhau |
| 4. Rủi ro lạm dụng | LTM thấp hơn: đóng góp dữ liệu bảng thường cần xác minh danh tính và truy được nguồn; bịa dữ liệu thủ công vốn đã dễ. Rủi ro không chủ ý (thiên kiến, rò rỉ) tương tự |
| 5. Tác động trên mỗi nhà nghiên cứu | LTM có thể cao hơn: ít người làm, nhiều câu hỏi mở, chưa cần tính toán khổng lồ |

Kết luận: không phải phán xét modality nào tốt hơn ("so táo với lê") mà là mỗi lớp mô hình xứng đáng được quan tâm.

## 8. Phụ lục

- **Phụ lục A (Hình 1):** đếm bài được chấp nhận theo từ khóa trong tiêu đề/tóm tắt ở ICML 2023 (1828 bài), NeurIPS 2023 (3218), ICLR 2024 (2250). Nhóm "Table" (vừa "foundation model" vừa "table"): **2 bài ở ICML 2023, 6 ở NeurIPS 2023, 7 ở ICLR 2024**, và còn gồm cả các công trình LLM đánh giá trên bảng (như Table QA). Nghĩa là số bài về mô hình nền cho bảng rất ít so với LLM.
- **Phụ lục B (benchmark):** tác vụ gồm học giám sát (qua biểu diễn hàng hoặc trực tiếp), sinh dữ liệu tổng hợp (train-on-synthetic-test-on-real, fidelity/diversity, ϵ-identifiability), điền khuyết (đơn/đa, cơ chế MCAR/MAR/MNAR), LTM cho khoa học (giảm chiều, làm sạch, phân cụm, tác vụ liên bộ dữ liệu). Thiết lập thực nghiệm: few-shot (có/không tinh chỉnh; đo cả độ bền), zero-shot (cần khái quát hóa thật), in-distribution (đo khoảng cách khái quát hóa). Cảnh báo: mô hình nền công bố thường không mô tả chính xác dữ liệu huấn luyện nên dễ **rò rỉ dữ liệu**, nhất là khi tưởng đang đo few/zero-shot.

## 9. Hạn chế và nhận xét

Tác giả thừa nhận LTM chỉ được định nghĩa lỏng, bài chỉ xét bảng đơn; so sánh LLM/LTM là "so táo với lê".

Nhận xét thêm của mình:
1. Bài viết **trước** TabPFNv2 (Nature 2025), TabICL, TabDPT, Mitra...: tiêu chí D1–D4 cho phép đối chiếu các mô hình đó. Gần như D1 và D4 đã được TabPFNv2/TabICL giải quyết; D2 một phần; D3 chỉ có ở ConTextTab, TabSTAR (bài 16, 19).
2. Lập luận "bảng chưa thấy scaling vì benchmark nhỏ" khá thuyết phục nhưng chỉ là giả thuyết; TabPFNv2 đạt tốt với tiền huấn luyện trên dữ liệu **tổng hợp** (không phải dữ liệu thật quy mô lớn), một hướng mà bài này chưa đề xuất.
3. Bài nhấn mạnh ứng dụng sinh dữ liệu, nhưng các TFM thành công hiện nay là mô hình **dự đoán** ICL; hướng sinh dữ liệu vẫn chưa trưởng thành.
4. Khi bàn về nhóm thiểu số/dữ liệu ít, bài chỉ nêu kỳ vọng few-shot và SMOTE, **không đề cập đến độ bền với nhiễu nhãn hay mất cân bằng của LTM**; đây đúng là khoảng trống mà đề tài của An nhắm tới.

## 10. Liên hệ với bài Nature và đề tài

- Có thể dùng làm **lập luận mở đầu**: bảng quan trọng, chưa có mô hình nền; sau đó bài Nature TabPFNv2 và TabICLv2/TabDPT/LoCalPFN là những lời đáp.
- **Khung đối chiếu 4 SOTA:** lập bảng D1–D4 (cột kiểu hỗn hợp, liên bộ dữ liệu, ngữ cảnh văn bản, bất biến thứ tự cột) cho TabPFNv2, TabICLv2, TabDPT, LoCalPFN. Đây là cách đặt "niche" rõ ràng cho giảng viên.
- **Đánh giá:** phần Challenges về đánh giá và rò rỉ dữ liệu (Phụ lục B) hỗ trợ lập luận cần benchmark mới (ví dụ nhiễu, mất cân bằng, hiệu chỉnh).

## 11. Từ khóa cần nhớ

- **LTM (Large Tabular Model):** mô hình nền cho dữ liệu bảng.
- **D1–D4:** cột kiểu hỗn hợp; liên bộ dữ liệu; ngữ cảnh văn bản; bất biến/tương đẳng thứ tự cột.
- **Hai kiểu thích nghi:** trực tiếp (tinh chỉnh/ICL) và gián tiếp (dữ liệu tổng hợp).
- **Ba nhược điểm LLM cho bảng:** đắt, kém biến liên tục, hiệu chỉnh kém.
- **Số liệu:** TabLib 627 triệu bảng, 827 tỉ token; nhóm "Table" chỉ 2 + 6 + 7 bài ở ICML23, NeurIPS23, ICLR24.
- **Thách thức:** mô hình sinh, quy mô, đa dạng dữ liệu, đánh giá (khó kiểm tra bằng mắt), thiên kiến.
