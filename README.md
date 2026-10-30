# Đồ án 2 – Dự đoán rủi ro tín dụng theo khung Champion–Challenger

Đồ án 2 học phần **Phân tích dữ liệu và học máy** (Đại học Bách khoa Hà Nội).
Bộ dữ liệu: [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (Kaggle).

## Bài toán

Dự đoán xác suất một hồ sơ vay **vỡ nợ** (`target = 1`) tại ngày ra quyết định cho vay, sao cho mô hình vừa **phân biệt tốt** vừa **ổn định theo thời gian**, dưới các ràng buộc chống rò rỉ dữ liệu (leakage).

Đồ án trả lời hai câu hỏi:

1. **Mô hình tốt nhất và ổn định nhất đạt được đến đâu?** So sánh Champion (scorecard Logistic/WOE, baseline truyền thống) với Challenger (mô hình cây và stacking, có các thành phần AI/AutoML).
2. **Code agent so với con người thế nào?** Cùng một đề bài, so ba cách làm: người viết (H), agent với một prompt duy nhất (A0), agent được hướng dẫn chi tiết (A1).

## Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Số hồ sơ | 1,526,659 |
| Thời gian | 01/01/2019 – 05/10/2020 (`WEEK_NUM` 0–91) |
| Tỷ lệ vỡ nợ | 3.14% |
| Số bảng nguồn | 17 bảng logic, 3 mức độ sâu (depth 0, 1, 2) |
| Feature sau xử lý | 627 |

### Chia tập theo thời gian

| Tập | `WEEK_NUM` | Số hồ sơ | Tỷ lệ vỡ nợ | Vai trò |
|---|---|---|---|---|
| train | 0–43 | 841,500 | 2.98% | Huấn luyện, mọi bước fit |
| valid | 44–51 | 224,421 | 3.52% | Chọn mô hình |
| calib | 52–55 | 82,387 | 3.85% | Dự phòng |
| oot1 | 56–61 | 102,273 | 4.83% | Đánh giá cuối (mở một lần) |
| oot2 | 62–91 | 276,078 | 2.52% | Đánh giá cuối (mở một lần) |

## Metric

- **Gini** = 2·AUC − 1.
- **Metric ổn định của cuộc thi**, tính trên Gini theo từng tuần:

  `stability = mean(gini) + 88.0 · min(0, a) − 0.5 · std(residuals)`

  với `a` là hệ số góc khi hồi quy tuyến tính Gini tuần theo `WEEK_NUM`, `residuals` là phần dư của hồi quy đó. Mô hình bị phạt nếu Gini giảm dần theo thời gian hoặc dao động mạnh.

## Pipeline

```
Dữ liệu thô (Kaggle, 17 bảng)
  │  depth 0 / depth 1 / depth 2: xử lý leakage cột ngày, aggregate theo case_id
  ▼
features.parquet (726 feature)
  │  src/features/prepare.py: quy tắc R1–R4, R7
  ▼
model_table.parquet (627 feature)
  ├─► Champion: optbinning → WOE → Logistic Regression (cửa sổ Train tuần 20–43)
  └─► Challenger: LightGBM / XGBoost / CatBoost + Optuna → stacking (cửa sổ Train tuần 0–43)
  ▼
Mô phỏng Champion–Challenger trên OOT: Champion chạy song song với Challenger-H, A0, A1
```

## Quy tắc tiền xử lý

| Quy tắc | Nội dung | Áp dụng |
|---|---|---|
| R1 | Loại feature ≥ 99.9% missing trên Train hoặc Valid, hoặc chỉ một giá trị khác null trên Train | Mọi mô hình |
| R2 | Loại cột năm lịch (`dateyear`) và tháng trong năm (`datemonth`) | Mọi mô hình |
| R3 | Loại proxy thời gian: `firstclxcampaign_1125D`, `datefirstoffer_1144D`, `credit_bureau_a_2__pmts_year_1139T__{min_m, n_months, n_contracts}` | Mọi mô hình |
| R4 | Gộp ba nguồn thuế (a, b, c) thành nhóm `tax__*` | Mọi mô hình |
| R5 | Cửa sổ Train: Challenger tuần 0–43, Champion tuần 20–43 | Theo họ mô hình |
| R6 | Loại feature có PSI (Valid so với cửa sổ Train) > 0.25 | Chỉ Champion |
| R7 | Không dùng giới tính (`person_1__app__sex_738L`) làm feature | Mọi mô hình |

Các nguyên tắc chống leakage khác: cột ngày bị ghi sau ngày quyết định được loại hoặc đổi thành null; giá trị missing mã hóa `a55475b1` được đổi thành null; mọi ngưỡng, binning và mã hóa chỉ fit trên Train.

## Kết quả hiện tại

| Mô hình | Valid Gini | Ghi chú |
|---|---|---|
| LightGBM chẩn đoán, cửa sổ tuần 0–43 | 0.6549 ± 0.0009 (3 seed) | Bảng 643 feature cũ; tham số mặc định |
| LightGBM chẩn đoán, cửa sổ tuần 20–43 | 0.6437 ± 0.0009 (3 seed) | Thua cửa sổ 0–43 ở 24/24 cặp (seed, tuần) |
| Champion, cửa sổ tuần 20–43, 61 feature, không giới tính | 0.6067 | Champion chính thức |

Một số phát hiện chính:

- **Cửa sổ Train tối ưu khác nhau theo họ mô hình.** LightGBM tốt hơn với tuần 0–43; scorecard tốt hơn với tuần 20–43 (ΔGini 0.0072, CI95 bootstrap [0.0052; 0.0099]).
- **Hiệu ứng độ phủ tuần 20.** Bảng `credit_bureau_a_1` tăng độ phủ đột ngột quanh tuần 20, gây phần lớn drift giữa Train và Valid.
- **Proxy thời gian và hiệu ứng mùa.** Các feature `pmts_year_1139T` trôi theo thời gian; các feature "số năm trước quyết định" giảm đều trong năm (Spearman ≈ −0.997). Loại chúng không làm giảm Valid Gini.

## Cấu trúc repo

```
da2-credit-risk/
├── configs/
│   └── config.yaml            # seed, chia tập, tham số mô hình
├── src/
│   ├── config.py              # load_config()
│   └── features/
│       ├── prepare.py         # quy tắc R1–R4, R7; META, AUDIT_COLS
│       └── unify_tax.py       # gộp nguồn thuế (R4)
├── notebooks/
│   ├── kaggle/                # notebook xử lý dữ liệu thô trên Kaggle
│   ├── 01_check_depth0.ipynb
│   ├── 02_merge_features.ipynb
│   ├── 03_eda_train.ipynb
│   ├── 04_prepare_model_table.ipynb
│   ├── 05_train_window_ab.ipynb   # cửa sổ Train, drift, proxy thời gian, ablation
│   └── 06_champion.ipynb          # Champion: optbinning, WOE, Logistic
├── reports/                   # bảng kết quả (csv, json) và hình
├── data/                      # không đưa lên GitHub
└── models/                    # không đưa lên GitHub
```

## Cài đặt và chạy

Yêu cầu: Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install polars numpy scikit-learn lightgbm optbinning joblib pyyaml
```

Quy trình:

1. **Trên Kaggle:** chạy các notebook trong `notebooks/kaggle/` để tạo `depth0.parquet`, `depth1_agg.parquet`, `depth2_agg.parquet`, rồi tải về `data/processed/`.
2. **Trên máy cá nhân:** chạy lần lượt notebook 02 → 04 để tạo `features.parquet` và `model_table.parquet`.
3. Chạy các notebook mô hình (05 trở đi).

Thư mục `data/` và `models/` không có trong repo vì dung lượng lớn.

## Tiến độ

- [x] Xử lý dữ liệu thô và leakage (depth 0, 1, 2)
- [x] EDA, phân tích drift và chọn cửa sổ Train
- [x] Champion (scorecard Logistic/WOE)
- [ ] LightGBM baseline trên bảng 627 feature
- [ ] Challenger: sinh feature, LLM feature engineering (kiểu CAAFE), CatBoost/XGBoost/LightGBM + Optuna, stacking
- [ ] AutoGluon benchmark
- [ ] Thí nghiệm agent: Challenger-A0, Challenger-A1
- [ ] Mô phỏng Champion–Challenger trên OOT

## Tác giả

Nguyễn Xuân Cường – Toán Tin, Đại học Bách khoa Hà Nội.
