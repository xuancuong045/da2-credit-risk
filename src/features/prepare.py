"""Tiền xử lý bảng feature cho mô hình: quy tắc R1–R4 (R5–R6 ở bước chọn cửa sổ và chọn biến)."""
import re
import polars as pl

from src.features.unify_tax import unify_tax

META = ['case_id', 'date_decision', 'MONTH', 'WEEK_NUM', 'target', 'split']

# R1: trường ngừng thu thập (dữ liệu kết thúc tuần 34–40, Valid 100% missing)
DISCONTINUED = ['pmtaverage_3A', 'birthdate_574D', 'pmtssum_45A', 'pmtscount_423L',
                'assignmentdate_238D', 'pmtcount_693L']
# R3: số ngày tính từ một sự kiện cấp công ty → trôi theo lịch
TIME_PROXY = ['firstclxcampaign_1125D', 'datefirstoffer_1144D']
# R2: năm lịch → số năm trước quyết định; tháng trong năm → loại
RE_YEAR = re.compile(r'dateyear_\d+T__(max|mean)$')
RE_MONTH = re.compile(r'datemonth_\d+T__(max|mean)$')

def apply_rules(lf: pl.LazyFrame) -> pl.LazyFrame:
    """R2, R3, R4 và phần cố định của R1. Chỉ biến đổi trên từng hồ sơ, không fit."""
    names = lf.collect_schema().names()
    year_cols = [c for c in names if RE_YEAR.search(c)]
    month_cols = [c for c in names if RE_MONTH.search(c)]

    lf = unify_tax(lf)                                                    # R4
    dec_year = pl.col('date_decision').dt.year().cast(pl.Float32)
    lf = lf.with_columns([
        pl.when(dec_year - pl.col(c).cast(pl.Float32) >= 0)
          .then(dec_year - pl.col(c).cast(pl.Float32))
          .alias(f'{c}_yrs_ago')                                          # R2: âm (sau quyết định) → null
        for c in year_cols])
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