"""Hook PostToolUse (Write/Edit) cho nhánh A1 — rà soát tĩnh file .py/.ipynb vừa ghi.

Không chặn công việc: chỉ khi phát hiện dấu hiệu đáng ngờ mới trả phản hồi cho agent
(exit code 2 + thông điệp ở stderr). Đây là nhắc nhở theo luật trong TASK.md,
không phải lời giải; agent tự đánh giá và quyết định.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHECKS = [
    (r'\b(import|from)\s+(requests|urllib|httpx|http\.client|socket|aiohttp)\b',
     'Có import thư viện mạng — TASK.md cấm truy cập internet.'),
    (r'pip\s+install|subprocess\.[a-z_]+\([^)]*pip',
     'Có lệnh cài thư viện — TASK.md cấm cài thêm thư viện.'),
    (r'''['"](\.\.[/\\]|[A-Za-z]:[/\\]|/(home|Users|mnt|tmp)/)''',
     'Có đường dẫn ra ngoài thư mục làm việc — chỉ được đọc/ghi trong thư mục này.'),
    (r'WEEK_NUM[^\n]{0,40}(>=?\s*(5[2-9]|[6-9]\d)|<=?\s*(5[2-9]|[6-9]\d))',
     'Có tham chiếu WEEK_NUM > 51 — features.parquet chỉ có tuần 0–51; kiểm tra lại logic.'),
]

FIT_PAT = re.compile(r'\.(fit|fit_transform|train)\s*\(|lgb\.train|xgb\.train|cv\s*\(')
VALID_PAT = re.compile(r'WEEK_NUM[^\n]{0,40}(4[4-9]|5[01])|split\s*==\s*[\'"]valid')
TARGET_AGG_PAT = re.compile(r'target[^\n]{0,60}(mean|sum|agg|groupby|group_by)|'
                            r'(groupby|group_by)[^\n]{0,80}target')


def code_of(path: Path) -> str:
    text = path.read_text(encoding='utf-8', errors='ignore')
    if path.suffix == '.ipynb':
        try:
            nb = json.loads(text)
            return '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', [])
                             if c.get('cell_type') == 'code')
        except json.JSONDecodeError:
            return text
    return text


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    fp = (data.get('tool_input') or {}).get('file_path') or ''
    path = Path(fp)
    if path.suffix not in {'.py', '.ipynb'} or not path.exists():
        sys.exit(0)
    code = code_of(path)

    msgs = [m for pat, m in CHECKS if re.search(pat, code, flags=re.IGNORECASE)]
    if FIT_PAT.search(code) and VALID_PAT.search(code):
        msgs.append('File vừa fit/huấn luyện vừa nhắc tới tuần Valid (44–51). Xác nhận mọi bước fit '
                    '(mô hình, mã hóa, chọn feature, early stopping) chỉ dùng dòng Train cho mô hình tạo preds_valid.csv.')
    if TARGET_AGG_PAT.search(code):
        msgs.append('Có tổng hợp theo target (vd target encoding / thống kê theo nhóm). Xác nhận phép tổng hợp '
                    'chỉ tính trên dữ liệu huấn luyện và không để mỗi dòng "thấy" nhãn của chính nó.')

    if msgs:
        print(f'[leak_check] {path.name}:\n- ' + '\n- '.join(msgs) +
              '\n(Nhắc nhở tự động; nếu đã đúng, ghi lý do vào EXPERIMENTS.md và tiếp tục.)', file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == '__main__':
    main()
