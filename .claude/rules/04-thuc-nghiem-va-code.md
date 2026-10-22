# 04. Thực nghiệm và code

## Chia dữ liệu và chống rò rỉ
1. **Tập kiểm tra tách rời hoàn toàn.** Không dùng để huấn luyện, chọn tham số, chọn mô hình hay chọn tiền xử lý, dù chỉ một lần.
2. Mọi bước học từ dữ liệu (chuẩn hóa, mã hóa, điền thiếu, chọn đặc trưng) chỉ khớp trên tập huấn luyện rồi áp lên tập khác.
3. Chọn tham số bằng tập xác thực hoặc kiểm định chéo 5 lần **bên trong tập huấn luyện**.
4. **Khóa cấu hình trước khi nhìn tập kiểm tra.** Chạy tập kiểm tra một lần cho cấu hình đã chốt. Không chỉnh lại sau khi thấy kết quả để cho đẹp.
5. Có kiểm thử tự động: không dòng nào của tập kiểm tra nằm trong tập huấn luyện.
6. Cỡ mẫu "dữ liệu nhỏ" (D3): n = 20, 50, 100, 200, 500, 1.000. Lấy mẫu con có phân tầng theo lớp.

## So sánh công bằng
7. Mọi mô hình dùng cùng cách chia, cùng mẫu huấn luyện, cùng tập kiểm tra. Cùng ngân sách chỉnh tham số.
8. Có mô hình nền (đoán lớp đa số, hồi quy logistic, rừng ngẫu nhiên, XGBoost và CatBoost mặc định) trước khi nói mô hình nào tốt hơn.
9. Lặp nhiều seed (mục tiêu 10 lần như bài Nature), báo cáo trung bình và khoảng tin cậy 95%. Dùng kiểm định (Wilcoxon theo cặp) khi kết luận hơn kém.
10. Chỉ số: độ chính xác, ROC-AUC, Log-Loss; khi làm ngách thêm ECE và biểu đồ độ tin cậy. Đo riêng thời gian tải mô hình, chuẩn bị và dự đoán.
11. Kết quả chênh vài mẫu trên tập kiểm tra nhỏ (ví dụ 2/114) không được gọi là "tốt hơn". Nói rõ chênh lệch nằm trong dao động.

## Code
12. Viết lại script chính, có chú thích tiếng Việt cho từng bước để An đọc hiểu.
13. Ghi cố định seed, phiên bản thư viện (`requirements.txt` có số phiên bản), thiết bị chạy (CPU hay Colab GPU), phiên bản trọng số mô hình.
14. Kết quả ghi ra file CSV riêng cho từng lần chạy, có cột seed, n, mô hình, chỉ số. **Không sửa tay file kết quả** sau khi đã so sánh.
15. Code đặt trong thư mục riêng của dự án (ví dụ `code/`), tách khỏi thư mục anh bạn. Không để token, khóa API hay mật khẩu trong code.
16. Chạy được lại từ đầu bằng một lệnh hướng dẫn trong README.
17. Phần cứng của An: CPU Intel, không có GPU NVIDIA, RAM khoảng 11,9 GB. Thử nghiệm nhỏ chạy CPU, phần nặng chạy Google Colab. Nếu một lượt chạy lâu hoặc tốn RAM thì báo trước.
18. Trọng số TabPFN có thể cần tải và chấp nhận điều khoản: kiểm tra sớm, không giả định chạy được.

## Phần mở rộng (ngách D2)
19. Ngách: độ bền với nhãn nhiễu (đổi nhãn ngẫu nhiên 5%, 10%, 20%) và mất cân bằng lớp, đo kèm độ tin cậy xác suất. Nhiễu chỉ thêm vào **tập huấn luyện**, tập kiểm tra luôn sạch.
20. Phải chỉ ra được bài nào trong 4 SOTA chưa đo điều này và có câu hoặc số làm bằng chứng, trước khi gọi là "khoảng trống".
