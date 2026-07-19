#!/usr/bin/env python3
# Auto-generated script — Xóa label rỗng
from pathlib import Path
EMPTY_LABELS = []
TARGET_DIRS = ["datasets/processed/train/labels", "datasets/processed/val/labels", "datasets/processed/test/labels"]
removed = 0
for d in TARGET_DIRS:
    dp = Path(d)
    if not dp.exists():
        continue
    for f in dp.iterdir():
        if f.name in EMPTY_LABELS:
            print(f"Xóa: {f}")
            # f.unlink()
            removed += 1
print(f"Tổng: {removed} label rỗng")
