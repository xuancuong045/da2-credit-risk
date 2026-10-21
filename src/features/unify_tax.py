"""Gộp 3 nguồn thuế thay nhau theo thời gian thành nhóm tax__*.

tax_registry_c_1: WEEK_NUM 1–33 · tax_registry_a_1: 34–~63 · tax_registry_b_1: ~64–91.
c và a cùng thang số tiền; b lớn gấp ~9 lần (không quy đổi vì b chỉ có ở OOT-2),
nên số tiền tuyệt đối chỉ gộp c → a. Cột ngày không gộp (nguồn a không có cột ngày).
"""
import polars as pl

C, A, B = 'tax_registry_c_1', 'tax_registry_a_1', 'tax_registry_b_1'
AMT = {C: 'pmtamount_36A', A: 'amount_4527230A', B: 'amount_4917619A'}
EMP = {C: 'employername_160M', A: 'name_4527232M', B: 'name_4917606M'}
TAX_SOURCES = (C, A, B)


def _col(src: str, rest: str) -> pl.Expr:
    return pl.col(f'{src}__{rest}')


def unify_tax(lf: pl.LazyFrame, drop_sources: bool = True) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    exprs = [
        pl.coalesce(*[_col(s, 'count') for s in TAX_SOURCES]).alias('tax__count'),
        pl.coalesce(*[_col(s, f'{EMP[s]}__nuniq') for s in TAX_SOURCES]).alias('tax__employer_nuniq'),
    ]
    for op in ('max', 'mean', 'sum'):
        exprs.append(pl.coalesce(_col(C, f'{AMT[C]}__{op}'), _col(A, f'{AMT[A]}__{op}'))
                     .cast(pl.Float32).alias(f'tax__amount_{op}'))
    ratios = [pl.when(_col(s, f'{AMT[s]}__mean') > 0)
                .then(_col(s, f'{AMT[s]}__max') / _col(s, f'{AMT[s]}__mean'))
              for s in TAX_SOURCES]
    exprs.append(pl.coalesce(*ratios).cast(pl.Float32).alias('tax__amount_max_to_mean'))

    lf = lf.with_columns(exprs)
    if drop_sources:
        lf = lf.drop([c for c in names if c.startswith(TAX_SOURCES)])
    return lf