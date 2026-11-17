# Hướng dẫn làm việc (nhánh A1)

Nhiệm vụ nằm trong `TASK.md`; thông tin dữ liệu trong `DATASET_CARD.md`. Đọc cả hai trước khi bắt đầu.

## Nguyên tắc
1. **Kiểm định theo thời gian.** Mọi đánh giá dùng tách theo `WEEK_NUM` (quá khứ → tương lai), không xáo trộn ngẫu nhiên. Nếu cần early stopping hoặc chọn tham số, tách thêm các tuần cuối của Train; không dùng Valid để fit.
2. **Leakage là lỗi nghiêm trọng nhất.** Với mỗi nhóm cột, tự hỏi: thông tin này có sẵn tại `date_decision` không? Có thể chứa thông tin sau ngày quyết định, hoặc gián tiếp mã hóa thời gian (năm, mùa, vị trí của hồ sơ trên trục thời gian) không? Kiểm tra bằng dữ liệu (phân phối theo tuần, quan hệ với `WEEK_NUM`), ghi kết luận vào `REPORT.md`.
3. **Ổn định theo thời gian quan trọng ngang Gini.** Theo dõi Gini từng tuần và xu hướng của nó; một feature mạnh nhưng trôi theo thời gian có thể làm mô hình kém ở các tuần sau.
4. **Mọi bước fit chỉ trên dữ liệu huấn luyện**: mã hóa danh mục, chọn feature, chuẩn hóa, tinh chỉnh.
5. **Tái lập được:** cố định seed; lưu kết quả quan trọng ra file; mã nguồn chạy lại từ đầu ra đúng kết quả.

## Cách làm việc
- Ghi mỗi thí nghiệm vào `EXPERIMENTS.md`: thay đổi gì, kết quả `score.py` (gini_overall, stability), quyết định giữ/bỏ.
- So sánh hai cấu hình bằng cùng thiết lập và nhiều seed khi chênh lệch nhỏ.
- Bộ nhớ có hạn (~10–12 GB dùng được): ưu tiên polars lazy, float32, giải phóng đối tượng lớn.
- Trước khi kết thúc: dùng subagent `reviewer` (nếu có) để rà soát leakage và tính hợp lệ của sản phẩm; chạy `predict.py` thử trên một phần `features.parquet` đã bỏ cột `target`; chạy `score.py` trên `preds_valid.csv`.
