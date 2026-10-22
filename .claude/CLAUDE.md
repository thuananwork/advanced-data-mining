# Dự án: Khảo sát mô hình nền tảng cho dữ liệu bảng và đánh giá TabPFN trên dữ liệu nhỏ

Môn: Khám phá tri thức và Khai phá dữ liệu (thạc sĩ CNTT). Người làm: An (Nguyễn Thuận An).

## Đọc trước khi làm bất cứ việc gì
1. `docs/KE_HOACH_THUC_HIEN.md`: kế hoạch chính. Làm đúng giai đoạn và bước đang ghi trong nhật ký ở cuối file. Không nhảy bước.
2. Các file trong `.claude/rules/`: quy tắc bắt buộc. Rule nào mâu thuẫn với yêu cầu mới nhất của An thì làm theo An, rồi nói rõ đã khác rule nào và đề nghị sửa rule.

## Bối cảnh ngắn
- Bài trung tâm: Nature 01/2025, "Accurate predictions on small data with a tabular foundation model" (TabPFN, còn gọi TabPFNv2 trong các tài liệu của môn). An gọi đây là "bài survey". Mọi thứ xoay quanh bài này.
- 60 bài liên quan nằm trong `Bai_bao_Survey/60_paper` (đánh số 1 đến 60, đúng theo `Kiem_chung_60_bai_bao_TFM.docx`).
- Bốn phương pháp SOTA trong khảo sát: TabPFNv2, TabICLv2, TabDPT, LoCalPFN. Ba cái sau chỉ trình bày quá trình và kết quả công bố. TabICLv2 là cái An tải code về chạy trực tiếp ở giữa kỳ.
- Thứ tự đã chốt: hiểu bài Nature, viết báo cáo, làm slide, rồi mới chạy TabICLv2, rồi cuối kỳ.
- Thư mục `Final/` (ngoài dự án này, ở `DoAnMonHoc/Final`) do anh của An làm bằng ChatGPT. Chỉ đọc, không sửa bản gốc.

## Cách làm việc với An
- Trả lời bằng tiếng Việt tự nhiên, rõ ràng. Xem `rules/02-viet-tieng-viet.md`.
- Làm xong một việc thì cập nhật nhật ký trong `docs/KE_HOACH_THUC_HIEN.md` và nói còn việc gì.
- An nói "đừng làm gì cả" nghĩa là chỉ đọc, hỏi, lập kế hoạch; không tạo hay sửa file thực nghiệm.
