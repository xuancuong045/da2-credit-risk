"""Tiền xử lý bảng feature cho mô hình: quy tắc R1–R4 và R7 (R5–R6 ở bước chọn cửa sổ và chọn biến).

Sửa 28/10/2026 (kết quả notebook 05–06):
- R2: cột năm lịch (dateyear) bị loại hẳn thay vì đổi sang *_yrs_ago, vì *_yrs_ago có hiệu ứng mùa
  (chỉ có năm, trung bình giảm đều trong năm và nhảy ở tháng 1); bản số tháng chính xác làm lại ở depth 1 (W7).
- R3 mở rộng: thêm 3 proxy thời gian credit_bureau_a_2__pmts_year_1139T__* (lịch sử bureau bị cắt ở mốc lịch cố định).
- R7: thuộc tính giới tính không đưa vào mô hình; giữ trong bảng (AUDIT_COLS) để kiểm tra công bằng.
"""
import re
import polars as pl

from src.features.unify_tax import unify_tax

# R7: giữ trong bảng để kiểm tra công bằng, KHÔNG đưa vào mô hình
AUDIT_COLS = ['person_1__app__sex_738L']
# Cột không phải feature: cột tổ chức + cột kiểm tra công bằng
META = ['case_id', 'date_decision', 'MONTH', 'WEEK_NUM', 'target', 'split'] + AUDIT_COLS

# R1: trường ngừng thu thập (dữ liệu kết thúc tuần 34–40, Valid 100% missing)
DISCONTINUED = ['pmtaverage_3A', 'birthdate_574D', 'pmtssum_45A', 'pmtscount_423L',
                'assignmentdate_238D', 'pmtcount_693L']
# R3: feature trôi theo lịch (proxy thời gian)
TIME_PROXY = ['firstclxcampaign_1125D', 'datefirstoffer_1144D',
              'credit_bureau_a_2__pmts_year_1139T__min_m',
              'credit_bureau_a_2__pmts_year_1139T__n_months',
              'credit_bureau_a_2__pmts_year_1139T__n_contracts']
# R2: năm lịch và tháng trong năm → loại
RE_YEAR = re.compile(r'dateyear_\d+T__(max|mean)$')
RE_MONTH = re.compile(r'datemonth_\d+T__(max|mean)$')


def apply_rules(lf: pl.LazyFrame) -> pl.LazyFrame:
    """R2, R3, R4 và phần cố định của R1. Chỉ biến đổi trên từng hồ sơ, không fit."""
    names = lf.collect_schema().names()
    year_cols = [c for c in names if RE_YEAR.search(c)]
    month_cols = [c for c in names if RE_MONTH.search(c)]

    lf = unify_tax(lf)                                                    # R4
    drop = year_cols + month_cols + [c for c in DISCONTINUED + TIME_PROXY if c in names]
    return lf.drop(drop)


def find_uninformative(lf: pl.LazyFrame, feats: list[str],
                       splits=('train', 'valid'), max_miss: float = 0.999) -> dict[str, list[str]]:
    """R1: feature ≥ 99.9% missing trên Train hoặc Valid; feature chỉ 1 giá trị khác null trên Train."""
    out = {}
    for s in splits:
        sub = lf.filter(pl.col('split') == s)
        n = sub.select(pl.len()).collect().item()
        miss = sub.select([pl.col(c).null_count() for c in feats]).collect().row(0, named=True)
        out[f'empty_{s}'] = [c for c in feats if miss[c] / n >= max_miss]
    tr = lf.filter(pl.col('split') == 'train')
    nu = tr.select([pl.col(c).drop_nulls().n_unique() for c in feats]).collect().row(0, named=True)
    out['constant_train'] = [c for c in feats if nu[c] <= 1]
    return out