#!/usr/bin/env python3
# Auto-generated script — Xóa duplicate images
from pathlib import Path
DUP_FILES = []
TARGET_DIRS = ["datasets/processed/train/images", "datasets/processed/val/images", "datasets/processed/test/images"]
removed = 0
for d in TARGET_DIRS:
    dp = Path(d)
    if not dp.exists():
        continue
    for f in dp.iterdir():
        for dup in DUP_FILES:
            if dup in f.name:
                print(f"Xóa: {f}")
                # f.unlink()
                removed += 1
                break
print(f"Tổng: {removed} duplicate")
