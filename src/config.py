"""Đọc config dùng chung cho toàn dự án.

Dùng trong notebook:
    import sys; sys.path.append('..')
    from src.config import load_config, path
    cfg = load_config()
    df = pl.read_parquet(path(cfg['paths']['depth0']))
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'config.yaml'


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Đọc configs/config.yaml thành dict."""
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def path(relative: str) -> Path:
    """Đổi đường dẫn tương đối trong config thành đường dẫn tuyệt đối từ thư mục gốc repo."""
    return ROOT / relative
