"""
analysis.py — IoU computation, geometry statistics, resolution analysis,
class distribution, split ratios.
"""

from __future__ import annotations

import sys

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Tính IoU giữa 2 box YOLO (cx, cy, w, h)."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    b1_x1, b1_y1 = cx1 - w1 / 2, cy1 - h1 / 2
    b1_x2, b1_y2 = cx1 + w1 / 2, cy1 + h1 / 2
    b2_x1, b2_y1 = cx2 - w2 / 2, cy2 - h2 / 2
    b2_x2, b2_y2 = cx2 + w2 / 2, cy2 + h2 / 2
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_advanced_stats(values: List[float]) -> Dict[str, float]:
    """Tính thống kê nâng cao: mean, std, median, P1-P99, variance, CV, IQR, mode, skewness, kurtosis."""
    if not values:
        return {k: 0.0 for k in [
            "mean", "std", "median", "p1", "p5", "p25", "p50", "p75", "p95", "p99",
            "variance", "cv", "iqr", "mode", "skewness", "kurtosis", "min", "max",
        ]}
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if n > 1 else 0.0
    median_val = float(np.median(arr))
    sorted_arr = np.sort(arr)
    def _pct(p: float) -> float:
        return float(sorted_arr[min(int(p / 100.0 * (n - 1)), n - 1)])
    variance = float(np.var(arr)) if n > 1 else 0.0
    cv = std_val / mean_val if mean_val != 0 else 0.0
    iqr = _pct(75) - _pct(25)
    counter = Counter(arr.tolist())
    mode_val = counter.most_common(1)[0][0] if counter else 0.0
    skewness = float(np.mean(((arr - mean_val) / std_val) ** 3)) if n > 2 and std_val > 0 else 0.0
    kurtosis = (float(np.mean(((arr - mean_val) / std_val) ** 4)) - 3.0) if n > 3 and std_val > 0 else 0.0
    return {
        "mean": mean_val, "std": std_val, "median": median_val,
        "p1": _pct(1), "p5": _pct(5), "p25": _pct(25), "p50": _pct(50), "p75": _pct(75),
        "p95": _pct(95), "p99": _pct(99), "variance": variance, "cv": cv, "iqr": iqr,
        "mode": float(mode_val), "skewness": skewness, "kurtosis": kurtosis,
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def analyze_resolution(img_width: int, img_height: int) -> Dict[str, Any]:
    """Phân tích resolution: orientation, aspect_ratio, bucket."""
    ar = img_width / img_height if img_height > 0 else 0.0
    if abs(img_width - img_height) < min(img_width, img_height) * 0.05:
        orientation = "square"
    elif img_width > img_height:
        orientation = "landscape"
    else:
        orientation = "portrait"
    mp = (img_width * img_height) / 1_000_000.0
    bucket = "tiny" if mp < 0.1 else "small" if mp < 0.5 else "medium" if mp < 2.0 else "large" if mp < 5.0 else "very_large"
    return {"orientation": orientation, "aspect_ratio": ar, "megapixels": mp, "resolution_bucket": bucket}


def analyze_class_distribution(class_dist: Dict[str, int]) -> Dict[str, Any]:
    """Phân tích phân bố class: imbalance, rare, missing, gini, entropy."""
    if not class_dist:
        return {"imbalance_ratio": 0.0, "rare_classes": [], "missing_classes": [], "gini_index": 0.0, "entropy": 0.0}
    total = sum(class_dist.values())
    counts = list(class_dist.values())
    max_c, min_c = max(counts), min(counts)
    imbalance = max_c / min_c if min_c > 0 else float("inf")
    threshold = total * 0.01
    rare = [cls for cls, cnt in class_dist.items() if cnt < threshold]
    from config import CLASS_NAMES
    all_classes = set(CLASS_NAMES) if isinstance(CLASS_NAMES, (list, dict)) else set()
    if isinstance(CLASS_NAMES, dict):
        all_classes = set(CLASS_NAMES.values())
    elif isinstance(CLASS_NAMES, list):
        all_classes = set(CLASS_NAMES)
    missing = list(all_classes - set(class_dist.keys()))
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    gini = sum((2 * (i + 1) - n - 1) * c for i, c in enumerate(sorted_counts)) / (n * sum(sorted_counts)) if sum(sorted_counts) > 0 else 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts if c > 0)
    return {"imbalance_ratio": round(imbalance, 2), "rare_classes": rare, "missing_classes": missing, "gini_index": round(gini, 4), "entropy": round(entropy, 4)}


def analyze_split_ratios(train_n: int, val_n: int, test_n: int) -> Dict[str, Any]:
    """Phân tích tỷ lệ split train/val/test."""
    total = train_n + val_n + test_n
    if total == 0:
        return {"ratios": {}, "warnings": ["Không có dữ liệu"]}
    tr, vr, ter = train_n / total, val_n / total, test_n / total
    warnings = []
    if tr < 0.5:
        warnings.append(f"Train ratio quá thấp ({tr:.1%})")
    if vr < 0.05:
        warnings.append(f"Val ratio quá thấp ({vr:.1%})")
    if ter < 0.05:
        warnings.append(f"Test ratio quá thấp ({ter:.1%})")
    if tr > 0.9:
        warnings.append(f"Train ratio quá cao ({tr:.1%})")
    return {"ratios": {"train": round(tr, 4), "val": round(vr, 4), "test": round(ter, 4)}, "warnings": warnings}
