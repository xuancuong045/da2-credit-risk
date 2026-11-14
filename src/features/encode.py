"""Mã hóa feature cho Challenger (mô hình cây) — cấu hình d_freq (notebook 07, 14/11).

R8 (đề xuất): bỏ 7 cột categorical nhiều danh mục (HC_COLS) ở dạng gốc, thay bằng
{c}__freq = tỷ lệ danh mục trong TẬP HUẤN LUYỆN (null/chưa gặp -> NaN).
Các cột String còn lại: mã số nguyên theo danh mục của tập huấn luyện
(chưa gặp -> NaN), đưa vào LightGBM dưới dạng categorical_feature.
Mọi feature ép về Float32.

Chỉ gọi fit_encoder trên dòng của tập huấn luyện (vd tuần 0–39). Khi huấn luyện
lại trên cửa sổ khác (0–43, 0–51) phải fit lại encoder trên đúng cửa sổ đó.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

HC_COLS = [
    'person_1__app__registaddr_zipcode_184M',
    'person_1__app__contaddr_zipcode_807M',
    'person_1__app__empladdr_zipcode_114M',
    'person_1__app__registaddr_district_1083M',
    'person_1__app__contaddr_district_15M',
    'previouscontdistrict_112M',
    'person_1__app__empladdr_district_926M',
]


def fit_encoder(train: pl.DataFrame, feats: list[str]) -> dict:
    """Học danh mục (cột String thường) và tỷ lệ danh mục (cột HC) trên tập huấn luyện.

    train: các dòng huấn luyện; cần có ít nhất mọi cột String thuộc feats.
           Cột không có trong train được coi là cột số.
    feats: danh sách feature đầu vào theo đúng thứ tự (vd FEATS 627 cột).
    """
    schema = train.schema
    str_cols = [c for c in feats if schema.get(c) == pl.String]
    hc = [c for c in HC_COLS if c in feats]
    cat_cols = [c for c in str_cols if c not in hc]
    n = train.height

    cat_maps = {c: train[c].drop_nulls().unique().sort().to_list() for c in cat_cols}
    freq_maps = {}
    for c in hc:
        vc = train[c].drop_nulls().value_counts()
        freq_maps[c] = dict(zip(vc[c].to_list(), (vc['count'] / n).to_list()))

    base = [c for c in feats if c not in hc]
    return {
        'feats_in': list(feats),
        'base_feats': base,
        'hc_cols': hc,
        'cat_cols': cat_cols,
        'cat_maps': cat_maps,
        'freq_maps': freq_maps,
        'n_train': n,
        'out_feats': base + [f'{c}__freq' for c in hc],
    }


def transform_exprs(enc: dict) -> list[pl.Expr]:
    """Biểu thức Polars theo đúng thứ tự enc['out_feats']."""
    cat = set(enc['cat_cols'])
    exprs = []
    for c in enc['base_feats']:
        if c in cat:
            cats = enc['cat_maps'][c]
            exprs.append(pl.col(c).replace_strict(cats, list(range(len(cats))),
                                                  default=None, return_dtype=pl.Float32).alias(c))
        else:
            exprs.append(pl.col(c).cast(pl.Float32).alias(c))
    for c in enc['hc_cols']:
        fm = enc['freq_maps'][c]
        exprs.append(pl.col(c).replace_strict(list(fm.keys()), list(fm.values()),
                                              default=None, return_dtype=pl.Float32)
                     .alias(f'{c}__freq'))
    return exprs


def transform(df: pl.DataFrame, enc: dict) -> np.ndarray:
    """Ma trận Float32 (n_dòng × len(enc['out_feats'])), null -> NaN."""
    X = df.select(transform_exprs(enc)).to_numpy()
    assert X.dtype == np.float32 and X.shape[1] == len(enc['out_feats'])
    return X


def save_encoder(enc: dict, path: str | Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(enc, f, ensure_ascii=False)


def load_encoder(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)
