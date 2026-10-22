# Bài 14. ForecastPFN: Synthetically-Trained Zero-Shot Forecasting

| Mục | Thông tin |
|---|---|
| Nơi đăng | NeurIPS 2023 (arXiv:2311.01933v1) |
| Tác giả | Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha Naidu, Colin White (Abacus.AI, Caltech) |
| Mã nguồn | github.com/abacusai/forecastpfn (kèm mô hình đã huấn luyện) |
| Lưu ý về PDF | Bảng 1 (MSE theo từng bộ dữ liệu và baseline) bị rời cột khi trích chữ, nên mình không chép số từng ô mà dùng kết luận của tác giả và các số trong phần chữ/phụ lục. Phần mô tả huấn luyện có chỗ không khớp giữa thân bài và phụ lục (xem mục 3.3). |
| Vị trí trong đề tài | Bài **PFN cho chuỗi thời gian**, ngoài dữ liệu bảng. Cho thấy khung "huấn luyện một lần trên dữ liệu **hoàn toàn tổng hợp** rồi dự đoán zero-shot trong một lượt chạy xuôi" áp dụng cho miền khác, cùng nguyên tắc với TabPFN. Là bằng chứng rằng **thiết kế prior tổng hợp** quyết định chất lượng. |

## 1. Bài này làm gì trong một đoạn

Hầu hết phương pháp dự báo chuỗi thời gian cần nhiều điểm lịch sử để học, trong khi nhiều ứng dụng thực tế chỉ có rất ít quan sát (khoảng 40 hoặc ít hơn). Các phương pháp "zero-shot" trước đó (Meta-N-BEATS) có hiệu năng không ổn định vì phụ thuộc bộ dữ liệu thật dùng để tiền huấn luyện. ForecastPFN là mô hình dự báo zero-shot **đầu tiên được huấn luyện hoàn toàn trên dữ liệu tổng hợp**: một PFN (Transformer) học xấp xỉ suy luận Bayes trên một phân phối chuỗi thời gian tổng hợp gồm xu hướng toàn cục (tuyến tính + mũ), tính mùa vụ đa thang (tuần, tháng, năm) và nhiễu Weibull nhân. Sau huấn luyện một lần (~30 giờ trên 1 GPU V100), mô hình dự báo trên chuỗi mới chỉ bằng một lượt chạy xuôi (~0,2 giây). Trên 7 bộ dữ liệu thật, zero-shot với đầu vào chỉ 36 điểm **chính xác hơn và nhanh hơn** các phương pháp tiên tiến, kể cả khi chúng được huấn luyện thêm hàng trăm điểm lịch sử cùng chuỗi.

## 2. Bối cảnh

- **Bài toán:** dự báo chuỗi thời gian đơn biến, ứng dụng ở y tế, kinh tế, khí hậu, năng lượng tái tạo, kinh doanh (nhu cầu sản phẩm, rời bỏ khách hàng).
- **Phương pháp truyền thống:** ARIMA (1970), Prophet (xu hướng phi tuyến + mùa vụ đa thang). **Học sâu:** RNN; Transformer (Informer, Autoformer, FEDformer — FEDformer kết hợp biến đổi Fourier và phân rã mùa vụ-xu hướng). Hầu hết các Transformer dự báo được cấu hình để dự đoán N bước kế tiếp với độ dài dự đoán cố định.
- **Zero-shot trước đó:** Oreshkin 2020 (Meta-N-BEATS: huấn luyện trên một bộ dữ liệu thật như M4, kiểm thử trên bộ khác; kết quả khác hẳn tùy bộ dùng để huấn luyện); một phương pháp nền RNN chỉ so với phương pháp truyền thống và độ dài dự đoán cố định.
- **Công trình đồng thời:** dùng LLM (GPT-3) với cách token hóa mới cho zero-shot (tái dụng mô hình ngôn ngữ), và **LC-PFN** (Adriaensen 2023), PFN ngoại suy đường cong học, cho thấy tăng tốc AutoML tới 6 lần nhưng prior hoàn toàn khác.
- **Lý do chọn PFN:** PFN (Müller 2022) học offline trên dữ liệu tổng hợp từ một prior để xấp xỉ suy luận Bayes; TabPFN là bước đột phá cho dữ liệu bảng. Khó khăn ở dự báo: xây dựng phân phối chuỗi thời gian đủ chung và đủ học được, kèm kiến trúc/lược đồ huấn luyện phù hợp.

## 3. Phương pháp

### 3.1. PFN cho dự báo
Chuỗi D = {(t, y_t)}. Mỗi giả thuyết φ ∈ Φ là một cơ chế sinh chuỗi, y_t = ψ(t)·z_t với z_t ~ μ (nhiễu). Phân phối dự đoán hậu nghiệm p(y | T, D) ∝ ∫ p(y|T,φ) p(D|φ) p(φ) dφ. PFN q_θ xấp xỉ PPD qua "prior-fitting": lấy φ ~ p(φ), sinh D ~ p(D|φ), tách D_in = {(t,y_t)} (t = 1..ℓ) và D_out (t = ℓ..T), huấn luyện q_θ để dự đoán D_out từ D_in. Khác phân loại (cross-entropy), **dự báo là hồi quy nên dùng MSE**: L = E[Σ_t (y_t − q_θ(t, D_in))²], dẫn tới dự đoán điểm bằng kỳ vọng hậu nghiệm (khác PFNs4BO và TabPFN hồi quy dùng phân phối Riemann). Bước prior-fitting chỉ làm một lần; khi suy luận, trọng số không cập nhật.

### 3.2. Prior tổng hợp cho chuỗi thời gian
Giả thuyết cốt lõi: chuỗi = thành phần nền ψ nhân với nhiễu z_t; ψ = xu hướng × mùa vụ. Có ba thang thời gian độc lập: ngày, tháng, năm (chi tiết tháng/năm ở phụ lục D).
- **Nhiễu:** z_t = 1 + m_noise(z − z̄) với z ~ Weibull(1, k), z̄ = (ln 2)^{1/k} (kỳ vọng nhiễu bằng 1 để không ảnh hưởng xu hướng/mùa vụ). **Nhiễu nhân** để cân bằng tỉ số tín hiệu/nhiễu giữa các chuỗi (nhiễu cộng sẽ làm tỉ số này khác nhau nhiều theo thang của thành phần xu hướng). Phân phối Weibull là cách tham số hóa đơn giản nằm giữa Gauss và mũ — hai dạng thường gặp trong chuỗi thật.
- **Xu hướng:** trend(t) = (1 + m_lin·t + c_lin)·(m_exp·c_exp^t): kết hợp tuyến tính và mũ, hệ số m_lin, m_exp ~ N(μ_m, σ_m), c_lin ~ N(0, σ_lin), c_exp ~ N(1, σ_exp).
- **Mùa vụ:** tích của ba chuỗi Fourier theo tuần (p = 7), tháng (30,5), năm (365,25): seasonal_ν(t) = 1 + m_ν Σ_{f=1}^{⌊p_ν/2⌋} [c_{f,ν} sin(2πf t/p_ν) + d_{f,ν} cos(2πf t/p_ν)]; c_{f,ν}, d_{f,ν} ~ N(0, 1/f) (hệ số tỉ lệ nghịch với bậc hài), rồi chuẩn hóa để tổng bình phương bằng 1.
- Toàn bộ prior p(φ) là tích phân phối trên các tham số ξ; có **6 siêu tham số** của chính phân phối tổng hợp: μ_m, σ_m, σ_lin, σ_exp (xu hướng), m_noise, k (nhiễu). Chọn trong giai đoạn phát triển sao cho phân phối **đa dạng nhất có thể** nhưng mô hình vẫn huấn luyện được (nhiễu quá lớn thì mô hình không hội tụ). Toàn bộ phát triển diễn ra **trước** khi xem 7 bộ dữ liệu thật và chỉ có một mô hình duy nhất được dùng cho mọi thí nghiệm.

### 3.3. Kiến trúc và huấn luyện
- **Kiến trúc:** Transformer chỉ có encoder, mô tả trong thân bài là một lớp multi-head attention và hai lớp feed-forward; phụ lục E.2 nói **hai encoder**, mỗi encoder có attention 4 head và hai lớp feed-forward (lớp đầu gấp 32 lần chiều embedding, lớp sau gấp 8 lần).
- **Token đầu vào (t, y_t):** nhãn thời gian được biểu diễn bằng các đặc trưng lịch (năm, tháng, ngày, thứ trong tuần, ngày trong năm); giá trị chuỗi đã chuẩn hóa. **Truy vấn tùy ý:** đầu vào là một tập token cùng với một truy vấn là ngày tương lai bất kỳ, mô hình trả về dự đoán tại điểm đó; đầu vào không cần độ dài cố định hay liên tục. Khác các Transformer dự báo chỉ dự đoán N bước kế tiếp, ForecastPFN đạt zero-shot cho **độ dài dự đoán bất kỳ**.
- **Chuẩn hóa mạnh (robust scaling):** thang giá trị chuỗi rất lớn do xu hướng mũ và nhiễu nhân, nên (1) che giá trị thiếu, (2) chuẩn hóa theo các điểm không phải ngoại lai (định nghĩa là ngoài 2σ), (3) cắt các ngoại lai 3σ. Loss MSE còn được chia theo giá trị tối đa của đầu vào đã chuẩn hóa: nằm giữa MSE (trọng số quá lớn cho chuỗi biên độ cao) và MSPE (quá lớn cho chuỗi biên độ nhỏ).
- **Loại nhiễu khỏi nhãn huấn luyện:** vì dữ liệu tổng hợp có thành phần nền và nhiễu tách biệt, loss huấn luyện được tính trên **giá trị không nhiễu**; mô hình không thể học nhiễu vốn độc lập thời gian, nên hội tụ nhanh hơn. Đây là lợi thế chỉ có với dữ liệu tổng hợp.
- **Thông số huấn luyện (thân bài):** độ dài đầu vào ℓ = 100, dự đoán tối đa 10 bước (thực tế dự đoán xa hơn nhiều); 300.000 chuỗi tổng hợp, mỗi chuỗi dài 200, cửa sổ trượt 100 → 101 tác vụ mỗi chuỗi; mỗi epoch 1.024.000 tác vụ, 600 epoch, Adam, 1 GPU V100 16GB, **~30 giờ**. (Phụ lục E.3 nêu cấu hình hơi khác: 100.000 chuỗi cho mỗi tần suất ngày/tuần/tháng, 1000 bước × 1024 tác vụ mỗi epoch, 300 epoch, lr 1e-4, dự đoán xa nhất bằng 10 lần tần suất, vẫn khoảng 30 giờ; mình ghi cả hai vì không thể xác định bản nào là bản cuối.) Tổng ngân sách tính toán cả dự án khoảng 500 giờ GPU (chủ yếu thiết kế kiến trúc và phân phối tổng hợp).

## 4. Thiết lập thí nghiệm

- **Baseline:** ba cách đơn giản (Last, Mean, SeasonalNaive); hai phương pháp truyền thống (ARIMA qua pmdarima, Prophet); ba Transformer dự báo (FEDformer, Autoformer, Informer) và Transformer thuần; và một phương pháp zero-shot (Meta-N-BEATS, huấn luyện trên M4; do độ dài dự đoán cố định nên huấn luyện một mô hình cho mỗi độ dài). Tất cả dùng siêu tham số mặc định và mã chính thức.
- **Dữ liệu:** 7 bộ dữ liệu chuẩn: ECL (điện, 26.304 điểm hàng giờ), ETT1/ETT2 (nhiệt độ dầu máy biến áp), Exchange (tỉ giá, 7588 điểm hàng ngày), Illness (cúm, 966 điểm hàng tuần), Traffic (tắc nghẽn cao tốc California), Weather (khí hậu Đức, 3 năm). Các bộ ETT, ECL, Weather có độ phân giải dưới ngày nên được gộp thành ngày bằng tổng.
- **Giao thức:** hai mô hình zero-shot chỉ thấy **đầu vào 36 điểm** rồi dự đoán; các phương pháp còn lại được huấn luyện trên phần trước của chuỗi (lấy từ t = 500−x đến 500, 10% ngân sách dữ liệu dành cho validation), sau đó cùng kiểm thử trên đầu vào t = 501..536 và dự đoán từ 537 đến 537+ℓ. So sánh này **không công bằng cho ForecastPFN** vì các phương pháp khác được thấy dữ liệu lịch sử của chính chuỗi.
- **Hai trục ràng buộc:** (i) ngân sách dữ liệu x ∈ {50, 100, 150, 200, 250, 300, 500}; (ii) ngân sách thời gian huấn luyện 1–120 giây (chỉ tính vòng huấn luyện); độ dài dự đoán 6 đến 48 (theo FEDformer); mỗi cấu hình 5 seed (với phương pháp tất định như ForecastPFN, ARIMA, Prophet, baseline đơn giản chỉ một seed). Phương pháp không xuất được dự đoán trong ngân sách thì đặt đầu ra bằng 0 (Bảng 3 phụ lục E.6 thống kê tỉ lệ cấu hình theo từng ngân sách thời gian; chữ trong bài nói ARIMA là mô hình vất vả nhất ở ngân sách thấp, với 9,68% ở mức 1 giây so với 66–89% của các mô hình khác. Chú thích bảng ghi là "thất bại" nhưng các số tăng dần tới 100% khi ngân sách tăng, nên chiều của số liệu mâu thuẫn với chú thích; mình chỉ giữ nhận xét định tính của tác giả).
- **Thước đo:** MSE chính; kèm MAE, MAPE, MSPE ở phụ lục; tổng hợp bằng **số lần thắng (rank 1)** và xếp hạng MSE trung bình.

## 5. Kết quả

- **Dưới ngân sách thấp (Hình 3):** ForecastPFN **thắng nhiều nhất** (số lần thắng MSE) với ngân sách thời gian 1–30 giây và ngân sách dữ liệu 50–250 điểm; ở ngân sách cao hơn vẫn cạnh tranh với phương pháp tiên tiến. Xu hướng giữ nguyên theo từng độ dài dự đoán (Hình 4): ForecastPFN thắng gần như mọi độ dài ở ngân sách 50, 100, 150, và thắng hoặc rất cạnh tranh ở 200–300.
- **Bảng 1 (MSE trung bình theo bộ dữ liệu):** ở ngân sách dữ liệu 50, ForecastPFN có MSE thấp nhất trên nhiều bộ dữ liệu hơn bất kỳ thuật toán nào; ở ngân sách 500 vẫn cạnh tranh (hòa tốt nhất). Ví dụ hiếm số đọc chắc ở ngân sách 50: Illness và ETTh1/ETTh2 tốt hơn nhiều baseline học sâu, và ở các bộ như Traffic, ECL thì baseline đơn giản (SeasonalNaive) lại thắng; vì bảng bị lệch nên không chép thêm.
- **Baseline đơn giản đứng thứ hai:** SeasonalNaive và Last thường đứng thứ hai sau ForecastPFN ở ngân sách thấp nhất, minh họa độ khó khi dữ liệu và thời gian huấn luyện rất ít.
- **So với Meta-N-BEATS (so sánh công bằng duy nhất):** ForecastPFN **vượt chặt chẽ trong mọi thiết lập**.
- **Tốc độ:** dự đoán chỉ cần một lượt chạy xuôi, ~0,2 giây; các Transformer dự báo khác cần **hơn 100 lần** thời gian để đạt cùng độ chính xác.
- **Diễn giải của tác giả:** prior mã hóa mùa vụ đa thang + xu hướng + nhiễu nhiều tham số khiến mô hình đã biết "mẫu chung" của chuỗi thời gian; khoảng 250 điểm của một chuỗi thật vẫn đủ phức tạp và nhiễu để các phương pháp học từ đầu gặp khó, còn ForecastPFN chỉ cần 36 điểm.

### Ablation (chỉ hai, vì mỗi lần huấn luyện tốn 30 giờ GPU)
- **Chuẩn hóa mạnh so với min-max:** robust scaler cho validation loss thấp hơn trong suốt quá trình huấn luyện (Hình 5).
- **Mức nhiễu của dữ liệu tổng hợp:** huấn luyện với m_noise = 1 (gốc), 2/3 (nhiễu thấp), 1/6 (nhiễu thấp nhất) (không đổi nhiễu của dữ liệu validation); mức chọn cho loss tối thiểu bền vững (Bảng 2).
- **Bỏ nhiễu khỏi nhãn huấn luyện (E.5):** MSE validation trên dữ liệu tổng hợp **22,692** (loại nhiễu) so với **33,089** (không loại), nên dùng loại nhiễu; lưu ý lợi thế này chỉ có với dữ liệu tổng hợp.
- Cố gắng đạt zero-shot với kiến trúc FEDformer thất bại ("không đạt hiệu năng không suy biến sau nhiều nỗ lực"), nên chọn Transformer đơn giản.

## 6. Hạn chế

Tác giả nêu:
- Transformer thuần nên giới hạn ở chuỗi **dưới 1000 điểm** (không phải giới hạn cố hữu: có thể dùng Transformer mở rộng ngữ cảnh).
- Chỉ **đơn biến**; chưa có đa biến hay đặc trưng ngoại sinh.
- Prior tập trung vào chuỗi kiểu "con người/trái đất" (chu kỳ ngày, tuần, năm), có thể kém với chu kỳ lạ (41, 89, 97).
- Chỉ **dự đoán điểm** (xấp xỉ kỳ vọng), chưa dự đoán xác suất.
- Hướng tương lai: tận dụng khả năng xử lý dữ liệu thiếu (khoảng trống lớn) và tần suất thay đổi trong cùng chuỗi.

Nhận xét thêm của mình:
1. Số bộ dữ liệu thử (7) ít và là các bộ chuẩn có mùa vụ rõ rệt, **đúng loại mà prior được thiết kế để khớp**; chưa rõ hiệu năng trên chuỗi không mùa vụ hoặc có thay đổi chế độ (regime shift).
2. Tác giả thừa nhận mô hình được phát triển trước khi xem 7 bộ thật, nhưng siêu tham số prior vẫn được chọn "sao cho đa dạng nhất mà vẫn huấn luyện được", tức vẫn có kiến thức tiên nghiệm về dạng chuỗi chuẩn.
3. Giao thức đặt baseline học sâu vào ngân sách rất hạn chế (dữ liệu ≤ 500 điểm, thời gian ≤ 120 giây, đa số thất bại ở ngân sách thấp nhất), nên phần "thắng" ở ngân sách thấp một phần phản ánh việc baseline chưa kịp huấn luyện; khi dữ liệu dồi dào thì baseline học sâu bắt kịp hoặc vượt (ở ngân sách 500 chỉ "cạnh tranh"). Kết luận tương tự bài Nature: mô hình nền tảng thắng mạnh nhất khi dữ liệu nhỏ.
4. Mô tả huấn luyện và kiến trúc có chỗ mâu thuẫn giữa thân bài và phụ lục (số epoch, số encoder, cách sinh dữ liệu), nên khó tái lập chính xác từ bản này.
5. Bài đã được thay thế/vượt qua bởi các mô hình nền tảng chuỗi thời gian sau đó (Chronos, TimesFM, Moirai…); ở đây chỉ là mốc lịch sử cho khung PFN.

## 7. Liên hệ với bài Nature và đề tài

- **Khẳng định mạnh khung "prior tổng hợp + ICL":** cùng nguyên tắc TabPFN v2 (bài Nature) — chất lượng quyết định bởi prior; ForecastPFN thiết kế tay prior (xu hướng, mùa vụ, nhiễu Weibull), còn TabPFN v2 dùng prior SCM + cây.
- **Hai điểm thiết kế có thể mượn khi nghĩ về dữ liệu nhiễu:** (a) nhiễu nhân phân phối Weibull để tín hiệu/nhiễu cân bằng giữa các tập; (b) **loại nhiễu khỏi nhãn** khi huấn luyện trên dữ liệu tổng hợp giúp hội tụ nhanh. TabPFN v2 chưa rõ có mô hình hóa nhiễu nhãn trong prior hay không (ghi chú cho thí nghiệm cuối kỳ về nhiễu nhãn).
- **Thiếu hiệu chỉnh:** ForecastPFN chỉ dự đoán điểm (MSE) chứ không phân phối, khác PFNs4BO/TabPFN dùng đầu Riemann; nhắc ta nhớ rằng độ bất định là phần phải kiểm tra riêng.
- Khẳng định rằng "một mô hình, một lượt chạy xuôi, không huấn luyện lại" là lợi thế tốc độ lớn (0,2 giây so với hàng chục giây huấn luyện), điểm đáng so sánh với XGBoost/CatBoost cần tuning ở thí nghiệm cuối kỳ.

## 8. Từ khóa cần nhớ

- **ForecastPFN:** PFN dự báo zero-shot, huấn luyện 100% trên dữ liệu tổng hợp, đầu vào 36 điểm, truy vấn thời điểm bất kỳ.
- **Prior:** xu hướng (tuyến tính × mũ) × mùa vụ (tuần/tháng/năm, chuỗi Fourier hệ số N(0,1/f)) × nhiễu Weibull nhân (kỳ vọng 1); 6 siêu tham số của chính prior.
- **Mẹo huấn luyện:** MSE (không phải cross-entropy), robust scaling 2σ/3σ, loại nhiễu khỏi nhãn (MSE validation 22,692 so với 33,089).
- **Kết quả:** thắng nhiều nhất ở ngân sách dữ liệu 50–250 và thời gian 1–30 giây, nhanh hơn >100 lần; vượt chặt Meta-N-BEATS ở mọi thiết lập.
- **Hạn chế:** đơn biến, < 1000 điểm, dự đoán điểm, prior "human-like".
