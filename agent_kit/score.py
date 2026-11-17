"""Bộ chấm cố định — Đồ án 2 (Home Credit – Credit Risk Model Stability).

Dùng chung cho cả ba nhánh H, A0, A1. Bản chính thức do người tổ chức giữ:
sửa file này trong thư mục làm việc KHÔNG thay đổi điểm được công nhận.

Cách dùng:
    python score.py preds_valid.csv
    python score.py preds_valid.csv --data features.parquet --weeks 44 51 --json score_valid.json

preds_valid.csv: CSV có đúng 2 cột `case_id`, `score`
    (score càng cao = khả năng vỡ nợ càng cao; mỗi case_id một dòng).

Metric (metric ổn định của cuộc thi, tính trên các tuần được chấm):
    gini_w   = 2·AUC_w − 1                 (theo từng WEEK_NUM)
    y = a·x + b                            (hồi quy tuyến tính gini_w theo x = WEEK_NUM)
    residuals = gini_w − (a·x + b)
    stability = mean(gini_w) + 88.0·min(0, a) − 0.5·std(residuals)
Ngoài ra báo cáo Gini trên toàn tập (gini_overall).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def gini_stability(week: np.ndarray, y: np.ndarray, s: np.ndarray) -> dict:
    weeks = np.unique(week)
    ginis = np.array([2 * roc_auc_score(y[week == w], s[week == w]) - 1 for w in weeks])
    a, b = np.polyfit(weeks.astype(float), ginis, 1)
    residuals = ginis - (a * weeks + b)
    return {
        'gini_overall': float(2 * roc_auc_score(y, s) - 1),
        'mean_gini': float(ginis.mean()),
        'slope_a': float(a),
        'std_resid': float(residuals.std()),
        'stability': float(ginis.mean() + 88.0 * min(0.0, a) - 0.5 * residuals.std()),
        'weekly_gini': {int(w): round(float(g), 6) for w, g in zip(weeks, ginis)},
    }


def fail(msg: str) -> None:
    print(f'LỖI: {msg}', file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description='Bộ chấm cố định — Valid Gini + metric ổn định')
    ap.add_argument('preds', help='CSV gồm cột case_id, score')
    ap.add_argument('--data', default='features.parquet', help='parquet có case_id, WEEK_NUM, target')
    ap.add_argument('--weeks', nargs=2, type=int, default=[44, 51], metavar=('LO', 'HI'),
                    help='khoảng WEEK_NUM được chấm (mặc định Valid 44–51)')
    ap.add_argument('--json', default=None, help='ghi kết quả ra file json')
    args = ap.parse_args()
    lo, hi = args.weeks

    preds = pl.read_csv(args.preds)
    if set(preds.columns) != {'case_id', 'score'}:
        fail(f'file dự đoán phải có đúng 2 cột case_id, score — đang có {preds.columns}')
    preds = preds.with_columns(pl.col('case_id').cast(pl.Int64), pl.col('score').cast(pl.Float64))
    if preds['case_id'].is_duplicated().any():
        fail(f"case_id bị trùng: {int(preds['case_id'].is_duplicated().sum())} dòng")
    bad = preds.filter(pl.col('score').is_null() | pl.col('score').is_nan() | pl.col('score').is_infinite())
    if bad.height:
        fail(f'score có {bad.height} giá trị null/NaN/inf')

    labels = (pl.scan_parquet(args.data)
                .filter(pl.col('WEEK_NUM').is_between(lo, hi))
                .select(pl.col('case_id').cast(pl.Int64), 'WEEK_NUM', pl.col('target').cast(pl.Int8))
                .collect())
    if labels.height == 0:
        fail(f'không có dòng nào với WEEK_NUM trong [{lo}, {hi}] ở {args.data}')

    j = labels.join(preds, on='case_id', how='left')
    n_missing = j['score'].null_count()
    if n_missing:
        fail(f'thiếu dự đoán cho {n_missing} / {labels.height} hồ sơ trong tuần {lo}–{hi}')
    n_extra = preds.height - preds.join(labels.select('case_id'), on='case_id', how='semi').height
    if n_extra:
        print(f'Cảnh báo: {n_extra} dòng dự đoán không thuộc tuần {lo}–{hi} (bị bỏ qua)')

    r = gini_stability(j['WEEK_NUM'].to_numpy(), j['target'].to_numpy(), j['score'].to_numpy())
    r.update({'weeks': [lo, hi], 'n': int(labels.height), 'default_rate': float(labels['target'].mean())})

    print(f"Tuần {lo}–{hi} | n = {r['n']:,} | default rate {r['default_rate']:.4f}")
    print(f"gini_overall : {r['gini_overall']:.4f}")
    print(f"stability    : {r['stability']:.4f}   "
          f"(mean_gini {r['mean_gini']:.4f}, slope_a {r['slope_a']:+.5f}, std_resid {r['std_resid']:.4f})")
    print('Gini theo tuần:', {w: round(g, 4) for w, g in r['weekly_gini'].items()})
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(r, f, indent=2)


if __name__ == '__main__':
    main()
