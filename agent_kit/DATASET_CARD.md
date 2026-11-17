# Dataset card — features.parquet (chỉ dành cho nhánh A1)

Thông tin dưới đây là **sự thật về dữ liệu**, không phải hướng dẫn cách làm.

## Nguồn và cách tạo
- Bộ dữ liệu Kaggle "Home Credit – Credit Risk Model Stability". Mỗi dòng là một hồ sơ vay (`case_id`) tại ngày quyết định `date_decision`.
- Dữ liệu gốc gồm nhiều bảng: depth 0 (thông tin hồ sơ, 1 dòng/hồ sơ), depth 1 và depth 2 (lịch sử hồ sơ cũ, tín dụng ở các tổ chức khác — credit bureau, thông tin người liên quan, thuế, thẻ, tiền gửi; nhiều dòng/hồ sơ). Các bảng depth 1/2 đã được tổng hợp về 1 dòng/hồ sơ; tên cột tổng hợp có dạng `<bảng>__<cột gốc>__<phép tổng hợp>` (ví dụ `__max`, `__mean`, `__sum`, `__min`, `__count`, `__nuniq`).
- Tập `features.parquet` ở đây chỉ chứa tuần `WEEK_NUM` 0–51 (01/2019 – 12/2019). Các tuần sau đó do người tổ chức giữ để chấm `predict.py`.

## Thời điểm của thông tin
- Dữ liệu được **trích xuất khoảng 14 ngày sau `date_decision`**, không phải đúng tại ngày quyết định.
- Cột ngày (hậu tố `D`) đã được đổi thành **số ngày so với `date_decision`** (âm = trước ngày quyết định). Ở bước tổng hợp, một phần thông tin sau ngày quyết định đã được xử lý; phần còn lại chưa được kiểm tra hết.
- Cột hậu tố `T` có thể chứa năm/tháng lịch.

## Giá trị thiếu và kiểu dữ liệu
- Cột danh mục `M` gốc mã hóa giá trị thiếu bằng chuỗi `"a55475b1"`; trong file này đã đổi thành null.
- Các giá trị danh mục đã được ẩn danh (ví dụ `P127_74_114`).
- Hậu tố: `P` số ngày quá hạn, `M` danh mục ẩn danh, `A` số tiền, `D` ngày, `T`/`L` khác.

## Thay đổi theo thời gian
- Tỷ lệ vỡ nợ và số hồ sơ thay đổi theo tuần. Nguồn dữ liệu (bảng nào có mặt, tỷ lệ có giá trị của cột) có thể thay đổi giữa các giai đoạn; một số cột chỉ bắt đầu có dữ liệu từ giữa năm 2019.
- Các tuần sau tập Valid (người tổ chức giữ) bao gồm giai đoạn đầu năm 2020.

## Thuộc tính nhạy cảm
- Bộ dữ liệu có thông tin giới tính của khách hàng; theo `TASK.md`, thuộc tính này không được làm đầu vào của mô hình.

## Chia tập và chấm điểm
- Train: tuần 0–43. Valid: tuần 44–51. Metric và bộ chấm: xem `TASK.md` và `score.py`.
