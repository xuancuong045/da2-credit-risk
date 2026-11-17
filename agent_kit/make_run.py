"""Tạo thư mục chạy cho một lần thử agent (A0 hoặc A1).

Cách dùng (từ thư mục gốc repo, venv đang bật):
    python agent_kit/make_run.py --arm A0 --run 1 --src data/processed/features.parquet --root C:/Users/HOME/agent_runs

Kết quả: <root>/<arm>_run<run>/ gồm
    features.parquet   (CHỈ tuần 0–51, chỉ cột tổ chức + feature)
    TASK.md, score.py
    .claude/settings.json   (quyền: sửa file + chạy python; chặn web)
    RUN_LOG.md         (mẫu nhật ký do NGƯỜI tổ chức điền)
    + với A1: CLAUDE.md, DATASET_CARD.md
Thư mục chạy phải nằm NGOÀI repo (agent không được thấy src/, notebook, README của nhánh H).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl

KIT = Path(__file__).resolve().parent
KEEP_META = ['case_id', 'date_decision', 'MONTH', 'WEEK_NUM', 'target']
DROP_ALWAYS = ['split']          # cột tổ chức nội bộ của nhánh H, không đưa cho agent
MAX_WEEK = 51


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A0', 'A1'])
    ap.add_argument('--run', required=True, type=int)
    ap.add_argument('--src', required=True, help='features.parquet đầy đủ (notebook 02)')
    ap.add_argument('--root', required=True, help='thư mục chứa các lần chạy, NGOÀI repo')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    repo = KIT.parent.resolve()
    if root == repo or repo in root.parents:
        raise SystemExit(f'--root phải nằm ngoài repo ({repo})')
    run_dir = root / f'{args.arm}_run{args.run}'
    if run_dir.exists():
        raise SystemExit(f'{run_dir} đã tồn tại — không ghi đè')

    lf = pl.scan_parquet(args.src)
    cols = lf.collect_schema().names()
    missing = [c for c in KEEP_META if c not in cols]
    if missing:
        raise SystemExit(f'{args.src} thiếu cột {missing}')
    keep = KEEP_META + [c for c in cols if c not in KEEP_META and c not in DROP_ALWAYS]

    run_dir.mkdir(parents=True)
    out = run_dir / 'features.parquet'
    lf.filter(pl.col('WEEK_NUM') <= MAX_WEEK).select(keep).sink_parquet(out)

    chk = pl.scan_parquet(out).select(pl.len(), pl.col('WEEK_NUM').min().alias('wmin'),
                                      pl.col('WEEK_NUM').max().alias('wmax'),
                                      pl.col('target').mean().alias('dr')).collect().row(0, named=True)
    assert chk['wmax'] <= MAX_WEEK, chk

    for f in ['TASK.md', 'score.py']:
        shutil.copy2(KIT / f, run_dir / f)
    (run_dir / '.claude').mkdir()
    shutil.copy2(KIT / 'settings.json', run_dir / '.claude' / 'settings.json')
    if args.arm == 'A1':
        for f in ['CLAUDE.md', 'DATASET_CARD.md']:
            shutil.copy2(KIT / f, run_dir / f)
        a1 = KIT / 'a1_claude'           # hook + subagent reviewer (thêm sau)
        if a1.exists():
            shutil.copytree(a1, run_dir / '.claude', dirs_exist_ok=True)

    log = run_dir / 'RUN_LOG.md'
    log.write_text(
        f"# Nhật ký lần chạy {args.arm}_run{args.run}\n\n"
        f"- Tạo thư mục: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"- features.parquet: {chk['len']:,} dòng, {len(keep)} cột, tuần {chk['wmin']}–{chk['wmax']}, "
        f"dr {chk['dr']:.4f}, sha256 {sha256(out)[:16]}\n"
        f"- Model (/model): \n- Phiên bản Claude Code (`claude --version`): \n"
        f"- Bắt đầu: \n- Kết thúc: \n- Prompt duy nhất: \n\n"
        "## Checklist trước khi gõ prompt\n"
        "- [ ] Không có notebook/kernel/tác vụ nặng nào khác đang chạy; RAM trống ≥ 12 GB\n"
        "- [ ] Mở `claude` từ PowerShell NGOÀI VS Code, venv đã bật\n"
        "- [ ] Ô nhập không có dòng `[In ...]` (không kết nối IDE)\n"
        "- [ ] Dòng dưới cùng: `accept edits on`\n"
        "- [ ] `/model` = Opus 5.5, high effort; `claude --version` đã ghi ở trên\n"
        + ("- [ ] Không có CLAUDE.md ở thư mục cha và ~/.claude/CLAUDE.md\n" if args.arm == 'A0' else "")
        + "\n"
        "## Can thiệp / câu hỏi của agent\n| Giờ | Agent hỏi gì | Trả lời |\n|---|---|---|\n\n"
        "## Gián đoạn do hạn mức gói Pro\n| Từ | Đến | Ghi chú |\n|---|---|---|\n\n"
        "## Kết quả (người tổ chức chấm bằng score.py bản gốc)\n- Valid gini_overall: \n- Valid stability: \n"
        "- Thời gian chạy quy trình cuối: \n- RAM đỉnh: \n- predict.py chạy được: \n- Checklist hợp lệ: \n"
        "- Rà transcript (urllib / requests / http / đường dẫn ngoài thư mục): \n",
        encoding='utf-8')

    print(f'Đã tạo {run_dir}')
    print(f"features.parquet: {chk['len']:,} dòng | {len(keep)} cột | tuần {chk['wmin']}–{chk['wmax']} | dr {chk['dr']:.4f}")
    print('Files:', sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob('*') if p.is_file()))


if __name__ == '__main__':
    main()
