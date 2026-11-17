# Đề bài: Mô hình dự đoán rủi ro vỡ nợ — Home Credit (Credit Risk Model Stability)

## 1. Mục tiêu

Xây dựng mô hình dự đoán xác suất **vỡ nợ** (`target = 1`) cho từng hồ sơ vay tại ngày quyết định cho vay, sao cho **khả năng phân biệt cao nhất và ổn định theo thời gian** trên tập Valid. Mô hình bạn nộp sau đó sẽ được người tổ chức chấm thêm trên các tuần **sau** tập Valid mà bạn không được thấy.

Bạn tự quyết định toàn bộ cách làm: xử lý dữ liệu, chọn và tạo feature, chọn mô hình, tinh chỉnh tham số. Không hỏi lại; khi có điểm chưa rõ, tự chọn cách hợp lý nhất và ghi lại lựa chọn đó trong `REPORT.md`.

## 2. Dữ liệu

- File duy nhất: `features.parquet` trong thư mục này — 1,065,921 hồ sơ, tuần `WEEK_NUM` 0–51.
- Mỗi dòng là một hồ sơ (`case_id`), đã được tổng hợp từ nhiều bảng nguồn của bộ dữ liệu Home Credit – Credit Risk Model Stability (Kaggle).
- Cột tổ chức: `case_id`, `date_decision` (ngày quyết định), `MONTH`, `WEEK_NUM` (tuần, tăng dần theo thời gian), `target` (nhãn). Các cột còn lại là thông tin về hồ sơ.
- Hậu tố tên cột cho biết loại biến: `P` (số ngày quá hạn), `M` (danh mục đã ẩn danh), `A` (số tiền), `D` (ngày), `T`, `L` (khác).

## 3. Chia tập (bắt buộc)

| Tập | WEEK_NUM | Dùng để |
|---|---|---|
| Train | 0–43 | Huấn luyện (bạn được chia nhỏ tiếp bên trong, ví dụ để early stopping) |
| Valid | 44–51 | **Chỉ** để đánh giá và chọn mô hình |

- Mô hình tạo ra `preds_valid.csv` **chỉ được fit trên các dòng Train** (tuần 0–43). Nhãn Valid không được dùng để fit bất kỳ bước nào (mô hình, mã hóa, chọn ngưỡng theo nhãn...).
- Mô hình dùng trong `predict.py` (để người tổ chức chấm trên các tuần sau): bạn chọn giữ nguyên mô hình trên, **hoặc** huấn luyện lại trên tuần 0–51 theo cùng quy trình — ghi rõ lựa chọn trong `REPORT.md`.

## 4. Metric

Dùng bộ chấm có sẵn: `python score.py preds_valid.csv` (đọc nhãn từ `features.parquet`, chấm tuần 44–51).

- `gini_overall` = 2·AUC − 1 trên toàn bộ Valid.
- Metric ổn định:
  - `gini_w` = 2·AUC − 1 tính riêng từng `WEEK_NUM`;
  - hồi quy tuyến tính `gini_w = a·WEEK_NUM + b`, `residuals` = phần dư;
  - `stability = mean(gini_w) + 88.0·min(0, a) − 0.5·std(residuals)`.

Mô hình được đánh giá theo **cả hai** chỉ số. Bản `score.py` chính thức do người tổ chức giữ; sửa file trong thư mục này không thay đổi điểm được công nhận.

## 5. Sản phẩm phải nộp (trong thư mục này)

1. **`preds_valid.csv`** — đúng 2 cột `case_id`, `score`; một dòng cho mỗi hồ sơ Valid (tuần 44–51); `score` càng cao = rủi ro càng cao. Phải chạy `score.py` thành công.
2. **`predict.py`** — dùng được như sau:
   ```
   python predict.py --input <file.parquet> --output <file.csv>
   ```
   - Đầu vào: parquet **cùng cấu trúc cột** với `features.parquet` nhưng **không có cột `target`**, chứa các tuần bất kỳ (kể cả tuần > 51).
   - Đầu ra: CSV `case_id, score` cho mọi dòng đầu vào.
   - Chỉ dùng các file đã lưu trong thư mục `artifacts/` (mô hình, bộ mã hóa...); không huấn luyện lại, không cần mạng.
3. **`artifacts/`** — mọi thứ `predict.py` cần.
4. **Mã nguồn** tái lập được toàn bộ quy trình từ `features.parquet` đến `preds_valid.csv` và `artifacts/` (script `.py` hoặc notebook), có cố định seed.
5. **`REPORT.md`** gồm:
   - Tóm tắt cách làm (xử lý dữ liệu, feature, mô hình, tinh chỉnh).
   - Các cột/feature đã loại bỏ hoặc biến đổi và **lý do**.
   - Cửa sổ dữ liệu dùng để huấn luyện; mô hình trong `predict.py` fit trên tuần nào.
   - Kết quả `score.py` trên Valid (dán nguyên output).
   - Thời gian chạy toàn bộ quy trình và các hạn chế đã biết.

## 6. Ràng buộc

- Chỉ làm việc trong thư mục này; không đọc/ghi ngoài thư mục.
- Không truy cập internet; không cài thêm thư viện. Môi trường Python 3.12 có sẵn (chính): polars 2.0.0, pandas 3.0.6, numpy 2.5.4, pyarrow 25.0.1, scikit-learn 1.9.1, scipy 1.18.1, statsmodels 0.15.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, optbinning 1.0.0, optuna 5.0.0, shap 0.54.1, joblib 1.6.0, matplotlib 3.11.2, seaborn 0.13.2.
- Phần cứng: CPU Intel Core i7-9850H (6 nhân / 12 luồng), RAM 16 GB (khoảng 10–12 GB dùng được cho quy trình). **Chỉ dùng CPU.** Quy trình phải chạy được trong giới hạn này.
- Thời gian tối đa: 8 giờ tính từ lúc bắt đầu.
- Không được dùng thuộc tính giới tính của khách hàng làm đầu vào của mô hình (yêu cầu pháp lý/đạo đức trong chấm điểm tín dụng).
