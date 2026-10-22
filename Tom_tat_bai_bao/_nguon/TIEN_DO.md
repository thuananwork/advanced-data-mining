# TIẾN ĐỘ VIẾT LẠI 41 FILE TÓM TẮT (lưu 22/10/2026 08:30)

## Đã xong: 41/41 (md trong _nguon/, docx trong Tom_tat_bai_bao/)
01 Bai06 | 02 Bai03 | 03 Bai01 | 04 Bai21 | 05 Bai12 | 06 Bai07 | 07 Bai04 | 08 Bai08 | 09 Bai10
10 Bai33 | 11 Bai29 | 12 Bai05 | 13 Bai02 | 14 Bai36 | 15 Bai28 XGBoost | 16 Bai26 CatBoost (5 trang)
17 Bai27 LightGBM | 18 Bai23 Random Forests | 19 Bai25 Deep Learning Not All You Need | 20 Bai24 DNN Survey | 21 Bai47 Noisy Labels Survey | 22 Bai49 Class Imbalance Survey | 23 Bai09 Mitra | 24 Bai16 ConTextTab | 25 Bai19 TabSTAR | 26 Bai15 BETA | 27 Bai17 EquiTabPFN | 28 Bai20 APT | 29 Bai22 TabFlex | 30 Bai11 Position | 31 Bai30 FT-Transformer | 32 Bai32 TabM | 33 Bai31 TabNet | 34 Bai35 Embeddings | 35 Bai38 NODE | 36 Bai34 TransTab | 37 Bai39 CARTE | 38 Bai40 TP-BERTa | 39 Bai13 PFNs4BO | 40 Bai14 ForecastPFN | 41 Bai18 Do-PFN

## Trạng thái: HOÀN THÀNH 41/41 (22/10/2026)
Đã cập nhật 00_THU_TU_DOC.md và docs/KE_HOACH_THUC_HIEN.md. Việc tiếp theo: An đọc, rồi cùng chọn bài quan trọng để làm TabICLv2.

## Cách tiếp tục
1. PDF nằm ở Bai_bao_Survey/60_paper/*.pdf. Nếu /tmp/p/*.txt mất: `pdftotext -layout <pdf> /tmp/p/<số>.txt`.
2. Đọc kỹ cả phụ lục, viết md theo mẫu các file trước (bảng thông tin, tóm tắt 1 đoạn, vấn đề, phương pháp, thiết lập, kết quả có số đã kiểm, hạn chế, liên hệ đề tài, từ khóa).
3. `cd _nguon && python3 build_docx.py <file.md>` (docx ra thư mục cha). Gửi tin nhắn báo cho An sau mỗi nhóm.
4. Cuối cùng: tạo lại cột "File" của 00_THU_TU_DOC.md theo tên mới, kiểm tra số trang, ghi log vào docs/KE_HOACH_THUC_HIEN.md.

## Lưu ý đã phát hiện
- Bài 23 (RF) là bằng chứng sớm duy nhất về nhiễu nhãn (5%): boosting nhạy hơn bagging.
- Bài 23: ô Ozone "với bagging" ở Bảng 8 (17,8) lệch Bảng 6/7 (16,3), giữ nguyên số theo PDF.
- Bài 27 PDF không có phần supplementary.
