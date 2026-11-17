# Đồ án 2 – Dự đoán rủi ro tín dụng theo khung Champion–Challenger

Đồ án 2 chủ đề **Phân tích dữ liệu và học máy** (Đại học Bách khoa Hà Nội).
Bộ dữ liệu: [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (Kaggle).

## Bài toán

Dự đoán xác suất một hồ sơ vay **vỡ nợ** (`target = 1`) tại ngày ra quyết định cho vay, sao cho mô hình vừa **phân biệt tốt** vừa **ổn định theo thời gian**, dưới các ràng buộc chống rò rỉ dữ liệu (leakage).

Đồ án trả lời hai câu hỏi:

1. **Mô hình tốt nhất và ổn định nhất đạt được đến đâu?** So sánh Champion (scorecard Logistic/WOE, baseline truyền thống) với Challenger (mô hình cây, có các thành phần AI/AutoML).
2. **Code agent so với con người thế nào?** Cùng một đề bài, so ba cách làm: người thiết kế và ra quyết định từng bước, có trợ lý chat hỗ trợ viết code (H); agent với một prompt duy nhất (A0); agent được hướng dẫn chi tiết (A1).

## Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Số hồ sơ | 1,526,659 |
| Thời gian | 01/01/2019 – 05/10/2020 (`WEEK_NUM` 0–91) |
| Tỷ lệ vỡ nợ | 3.14% |
| Số bảng nguồn | 17 bảng logic, 3 mức độ sâu (depth 0, 1, 2) |
| Feature sau xử lý | 626 (Challenger, sau R1–R4, R7, R8) |

### Chia tập theo thời gian

| Tập | `WEEK_NUM` | Số hồ sơ | Tỷ lệ vỡ nợ | Vai trò |
|---|---|---|---|---|
| train | 0–43 | 841,500 | 2.98% | Huấn luyện, mọi bước fit (Challenger: fit 0–39, early stopping 40–43) |
| valid | 44–51 | 224,421 | 3.52% | Chọn mô hình |
| calib | 52–55 | 82,387 | 3.85% | Dự phòng |
| oot1 | 56–61 | 102,273 | 4.83% | Đánh giá cuối (mở một lần) |
| oot2 | 62–91 | 276,078 | 2.52% | Đánh giá cuối (mở một lần) |

## Metric

- **Gini** = 2·AUC − 1.
- **Metric ổn định của cuộc thi**, tính trên Gini theo từng tuần:

  `stability = mean(gini) + 88.0 · min(0, a) − 0.5 · std(residuals)`

  với `a` là hệ số góc khi hồi quy tuyến tính Gini tuần theo `WEEK_NUM`, `residuals` là phần dư của hồi quy đó. Mô hình bị phạt nếu Gini giảm dần theo thời gian hoặc dao động mạnh.

**Quy tắc nhận một biến thể trên Valid** (đặt trước khi chạy): trung bình dự đoán 3 seed có ΔValid Gini > 0 **và** thắng ở ≥ 6/8 tuần Valid so với cấu hình tham chiếu. Metric ổn định trên 8 tuần Valid rất nhiễu (hệ số góc từ 8 điểm) nên chỉ dùng tham khảo; nó có ý nghĩa thực sự trên OOT (36 tuần).

## Pipeline

```
Dữ liệu thô (Kaggle, 17 bảng)
  │  depth 0 / depth 1 / depth 2: xử lý leakage cột ngày, aggregate theo case_id
  ▼
features.parquet (726 feature)
  │  src/features/prepare.py: quy tắc R1–R4, R7
  ▼
model_table.parquet
  ├─► Champion: optbinning → WOE → Logistic Regression (cửa sổ Train tuần 20–43)
  └─► Challenger-H: src/features/encode.py (R8) → LightGBM, tham số Optuna, nhiều seed (cửa sổ Train tuần 0–43)
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
| R7 | Không dùng giới tính làm feature: `person_1__app__sex_738L` và `person_1__app__gender_992L` (null 99.16%, khi có giá trị khớp `sex_738L` 100%) | Mọi mô hình |
| R8 | Bỏ 7 cột danh mục nhiều giá trị (zipcode/district, 198–3,341 danh mục) ở dạng gốc, thay bằng tần suất danh mục `{cột}__freq` tính trên tập huấn luyện | Challenger (mô hình cây) |

Các nguyên tắc chống leakage khác: cột ngày bị ghi sau ngày quyết định được loại hoặc đổi thành null; giá trị missing mã hóa `a55475b1` được đổi thành null; mọi ngưỡng, binning và mã hóa chỉ fit trên dữ liệu huấn luyện.

## Kết quả hiện tại (Valid, tuần 44–51)

| Mô hình | Valid Gini | Metric ổn định | Ghi chú |
|---|---|---|---|
| Champion: Logistic/WOE, 61 feature, tuần 20–43 | 0.6067 | 0.5049 | Champion chính thức (`models/champion_final_61.joblib`) |
| LightGBM baseline, 33 cột danh mục gốc | 0.6524 | 0.6172 | Trung bình 3 seed; overfit ở cột zipcode/district |
| LightGBM `ref` (R8, 626 feature) | 0.6961 | 0.6804 | Trung bình 3 seed |
| **LightGBM `opt7` (tham số Optuna)** | **0.6986** | **0.6822** | Trung bình 3 seed; chênh Gini fit–Valid 0.073 (`ref`: 0.129) |

Một số phát hiện chính:

- **Cột danh mục nhiều giá trị gây overfit nặng.** Để LightGBM xử lý 7 cột zipcode/district như biến danh mục gốc chiếm ~40% gain và làm Valid Gini thấp hơn 0.043; bỏ chúng và thay bằng tần suất danh mục (R8) sửa được hoàn toàn. Target encoding out-of-fold không cho thêm cải thiện.
- **Stacking/blend không hiệu quả trên dữ liệu này.** XGBoost trên cùng feature xếp hạng gần như LightGBM (Spearman 0.992); blend LightGBM + XGBoost bằng đúng LightGBM nhiều seed hơn. Champion khác LightGBM hơn (Spearman 0.868) nhưng ghép 10–20% Champion làm Valid Gini giảm 0.002–0.006 và thua cả 8/8 tuần. Challenger-H vì vậy là LightGBM nhiều seed.
- **Optuna tìm được cấu hình regularize mạnh** (cây 22 lá, 1,332 mẫu/lá, mỗi cây dùng ~26% cột, extra trees): Valid Gini tăng 0.0025 và chênh fit–Valid giảm gần một nửa. Tổng 45 trial (30 trial tìm rộng + 15 trial tìm hẹp quanh cấu hình tốt nhất); bộ cắt sớm theo trung vị loại oan các cấu hình học chậm, nên lần tìm hẹp tắt cắt sớm trong 1,000 vòng đầu.
- **Chi phí của việc bỏ giới tính (R7):** khoảng 0.01 Valid Gini với cả Champion và LightGBM; riêng `gender_992L` (chỉ phủ 0.84% hồ sơ) là 0.0015.
- **Cửa sổ Train tối ưu khác nhau theo họ mô hình.** LightGBM tốt hơn với tuần 0–43 (kiểm tra lại sau khi sửa R8: vẫn thắng 8/8 tuần); scorecard tốt hơn với tuần 20–43 (ΔGini 0.0072, CI95 bootstrap [0.0052; 0.0099]).
- **Hiệu ứng độ phủ tuần 20.** Bảng `credit_bureau_a_1` tăng độ phủ đột ngột quanh tuần 20, gây phần lớn drift giữa Train và Valid.
- **Proxy thời gian và hiệu ứng mùa.** Các feature `pmts_year_1139T` trôi theo thời gian; các feature "số năm trước quyết định" giảm đều trong năm (Spearman ≈ −0.997). Loại chúng không làm giảm Valid Gini.

## Thí nghiệm code agent

Ba nhánh cùng đề bài (`agent_kit/TASK.md`) và cùng bộ chấm cố định (`agent_kit/score.py`):

| Nhánh | Cách làm |
|---|---|
| H | Người thiết kế và ra quyết định từng bước, có trợ lý chat Claude hỗ trợ viết code |
| A0 | Claude Code, **một prompt duy nhất**, không file chỉ dẫn, không can thiệp |
| A1 | Claude Code + hướng dẫn chi tiết: `CLAUDE.md`, dataset card, hook kiểm tra leakage, subagent reviewer |

- Agent làm việc trong thư mục riêng ngoài repo, chỉ có `features.parquet` (tuần 0–51) và tài liệu đề bài; không truy cập internet (chặn trong cấu hình quyền); cùng model và phiên bản Claude Code cho mọi lần chạy; tối đa 8 giờ mỗi lần; mỗi nhánh chạy 3 lần.
- Agent nộp kèm `predict.py` để người tổ chức chấm trên OOT; OOT mở một lần cho cả bốn mô hình.
- "Hiệu quả" đo theo ba nghĩa: mô hình tạo ra, code (thời gian chạy, RAM đỉnh, tái lập), công sức (giờ công, số lần can thiệp).

## Cấu trúc repo

```
da2-credit-risk/
├── configs/
│   └── config.yaml            # seed, chia tập, tham số mô hình
├── src/
│   ├── config.py              # load_config()
│   └── features/
│       ├── prepare.py         # quy tắc R1–R4, R7; META, AUDIT_COLS
│       ├── unify_tax.py       # gộp nguồn thuế (R4)
│       └── encode.py          # mã hóa cho Challenger (R8)
├── notebooks/
│   ├── kaggle/                # notebook xử lý dữ liệu thô trên Kaggle
│   ├── 01_check_depth0.ipynb
│   ├── 02_merge_features.ipynb
│   ├── 03_eda_train.ipynb
│   ├── 04_prepare_model_table.ipynb
│   ├── 05_train_window_ab.ipynb   # cửa sổ Train, drift, proxy thời gian, ablation
│   ├── 06_champion.ipynb          # Champion: optbinning, WOE, Logistic
│   ├── 06b_champion_final.ipynb   # lưu Champion chính thức 61 feature
│   ├── 07_lgbm_baseline.ipynb     # LightGBM baseline, ablation cột danh mục (R8)
│   └── 08_challenger_h.ipynb      # Challenger-H: target encoding, XGBoost, blend, Optuna
├── agent_kit/                 # gói thí nghiệm agent: đề bài, bộ chấm, cấu hình, tài liệu A1
├── reports/                   # bảng kết quả (csv, json)
├── data/                      # không đưa lên GitHub
└── models/                    # không đưa lên GitHub
```

## Cài đặt và chạy

Yêu cầu: Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Quy trình:

1. **Trên Kaggle:** chạy các notebook trong `notebooks/kaggle/` để tạo `depth0.parquet`, `depth1_agg.parquet`, `depth2_agg.parquet`, rồi tải về `data/processed/`.
2. **Trên máy cá nhân:** chạy lần lượt notebook 02 → 04 để tạo `features.parquet` và `model_table.parquet`.
3. Chạy các notebook mô hình (05 trở đi). Toàn bộ chạy trên CPU; LightGBM đặt `deterministic=True` nên kết quả tái lập chính xác.

Thư mục `data/` và `models/` không có trong repo vì dung lượng lớn.

## Tiến độ

- [x] Xử lý dữ liệu thô và leakage (depth 0, 1, 2)
- [x] EDA, phân tích drift và chọn cửa sổ Train
- [x] Champion (scorecard Logistic/WOE), lưu bản chính thức
- [x] LightGBM baseline; phát hiện và xử lý overfit do cột danh mục nhiều giá trị (R8)
- [x] Kiểm chứng target encoding, XGBoost, stacking/blend (không hiệu quả)
- [x] Optuna cho LightGBM
- [x] Gói thí nghiệm agent (`agent_kit/`)
- [ ] Feature: tổng hợp lại depth 1 theo số tháng chính xác, sinh feature tự động, LLM feature engineering (kiểu CAAFE)
- [ ] AutoGluon benchmark
- [ ] Thí nghiệm agent: Challenger-A0, Challenger-A1 (3 lần mỗi nhánh)
- [ ] Mô phỏng Champion–Challenger trên OOT

## Tác giả

Nguyễn Xuân Cường – Toán Tin, Đại học Bách khoa Hà Nội.
