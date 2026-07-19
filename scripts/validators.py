"""
validators.py — Annotation validation worker, DatasetValidator class.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import CLASS_NAMES, IMAGE_EXTENSIONS, LABEL_EXTENSION, NUM_CLASSES, get_project_paths
from models import DatasetValidationSummary, ImageQualityMetrics, Severity, ValidationError, ValidatorConfig
from image_processing import analyze_image_quality
from analysis import compute_iou
from hash_utils import hamming_distance

logger = logging.getLogger("dataset_validator")


def validate_labels_for_split(args: Tuple[str, str, List[str]]) -> Tuple[List[dict], List[Tuple[str, int, int, float, float, float, float]], Dict[str, int]]:
    """Worker — validate tất cả label trong split."""
    labels_dir_str, split_name, image_stems_list = args
    labels_dir = Path(labels_dir_str)
    image_stems = set(image_stems_list)
    errors: List[dict] = []
    all_annotations: List[Tuple[str, int, int, float, float, float, float]] = []
    class_dist: Dict[str, int] = {}
    now = datetime.now(timezone.utc).isoformat()
    label_files = {p.stem: p for p in labels_dir.glob(f"*{LABEL_EXTENSION}")}
    for stem in label_files:
        if stem not in image_stems:
            errors.append({"split": split_name, "label_name": f"{stem}{LABEL_EXTENSION}", "error_type": "MISSING_IMAGE", "severity": Severity.WARNING, "description": "Label không có ảnh", "line_number": 0, "timestamp": now})
    seen_anns: Set[Tuple[int, float, float, float, float]] = set()
    for stem, lbl_path in label_files.items():
        try:
            content = lbl_path.read_text(encoding="utf-8").strip()
        except UnicodeError as exc:
            errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "ENCODING_ERROR", "severity": Severity.ERROR, "description": f"Encoding: {exc}", "line_number": 0, "timestamp": now})
            continue
        except (IOError, PermissionError) as exc:
            errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "IO_ERROR", "severity": Severity.ERROR, "description": f"IO: {exc}", "line_number": 0, "timestamp": now})
            continue
        if not content:
            errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "EMPTY_LABEL", "severity": Severity.INFO, "description": "Label rỗng", "line_number": 0, "timestamp": now})
            continue
        for line_idx, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "INVALID_FORMAT", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: cần 5 giá trị, có {len(parts)}", "line_number": line_idx, "timestamp": now})
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "INVALID_CLASS_ID", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: class_id '{parts[0]}' sai", "line_number": line_idx, "timestamp": now})
                continue
            if class_id < 0 or class_id >= NUM_CLASSES:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "UNKNOWN_CLASS", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: class_id={class_id} ngoài [0,{NUM_CLASSES-1}]", "line_number": line_idx, "timestamp": now})
                continue
            coords = []
            valid = True
            for _, vs in enumerate(parts[1:], 1):
                try:
                    v = float(vs)
                except ValueError:
                    errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "INVALID_FLOAT", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: '{vs}' sai", "line_number": line_idx, "timestamp": now})
                    valid = False
                    break
                if math.isnan(v):
                    errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "NaN_VALUE", "severity": Severity.CRITICAL, "description": f"Dòng {line_idx}: NaN", "line_number": line_idx, "timestamp": now})
                    valid = False
                    break
                if math.isinf(v):
                    errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "INFINITY_VALUE", "severity": Severity.CRITICAL, "description": f"Dòng {line_idx}: Infinity", "line_number": line_idx, "timestamp": now})
                    valid = False
                    break
                coords.append(v)
            if not valid or len(coords) != 4:
                continue
            cx, cy, w, h = coords
            if w < 0:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "NEGATIVE_WIDTH", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: w={w}<0", "line_number": line_idx, "timestamp": now})
                continue
            if h < 0:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "NEGATIVE_HEIGHT", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: h={h}<0", "line_number": line_idx, "timestamp": now})
                continue
            if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "COORDS_OUTSIDE_IMAGE", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: ({cx},{cy}) ngoài [0,1]", "line_number": line_idx, "timestamp": now})
                continue
            if w <= 0 or h <= 0:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "ZERO_AREA_BBOX", "severity": Severity.ERROR, "description": f"Dòng {line_idx}: diện tích 0", "line_number": line_idx, "timestamp": now})
                continue
            if (cx - w / 2) < 0 or (cx + w / 2) > 1 or (cy - h / 2) < 0 or (cy + h / 2) > 1:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "BBOX_OVERFLOW", "severity": Severity.WARNING, "description": f"Dòng {line_idx}: bbox tràn", "line_number": line_idx, "timestamp": now})
            cls_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            class_dist[cls_name] = class_dist.get(cls_name, 0) + 1
            all_annotations.append((lbl_path.name, line_idx, class_id, cx, cy, w, h))
            ann_key = (class_id, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
            if ann_key in seen_anns:
                errors.append({"split": split_name, "label_name": lbl_path.name, "error_type": "DUPLICATE_ANNOTATION", "severity": Severity.WARNING, "description": f"Dòng {line_idx}: trùng", "line_number": line_idx, "timestamp": now})
            else:
                seen_anns.add(ann_key)
    return errors, all_annotations, class_dist


class DatasetValidator:
    """Validator chính — kiểm tra toàn diện dataset YOLO."""

    def __init__(self, root_dir: Path, cache_dir: Path = Path(".validator_cache"), workers: int = 4, config: Optional[ValidatorConfig] = None) -> None:
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "fingerprint_cache.json"
        self.workers = workers
        self.config = config or ValidatorConfig()

    def generate_fingerprint(self, file_paths: List[Path]) -> str:
        hasher = hashlib.sha256()
        for p in sorted(file_paths)[:self.config.max_fingerprint_files]:
            try:
                hasher.update(p.name.encode())
                hasher.update(str(p.stat().st_size).encode())
            except (OSError, PermissionError):
                continue
        return hasher.hexdigest()

    def load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save_cache(self, data: dict) -> None:
        try:
            self.cache_file.write_text(json.dumps(data, indent=4), encoding="utf-8")
        except (IOError, PermissionError) as exc:
            logger.warning("Không thể ghi cache: %s", exc)

    def analyze_split(self, split_name: str, img_dir: Path, lbl_dir: Path):
        from models import ValidationResult
        res = ValidationResult(split_name=split_name)
        img_files: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            img_files.extend(img_dir.glob(f"*{ext}"))
        img_files = sorted(set(img_files))
        res.total_images = len(img_files)
        res.total_labels = len(list(lbl_dir.glob(f"*{LABEL_EXTENSION}")))
        logger.info("Split '%s': %d ảnh, %d label", split_name, res.total_images, res.total_labels)
        tasks = [(str(p), split_name) for p in img_files]
        quality_results: Dict[str, ImageQualityMetrics] = {}
        img_stems: List[str] = [p.stem for p in img_files]
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            future_to_stem = {executor.submit(analyze_image_quality, task): img_files[i].stem for i, task in enumerate(tasks)}
            for future in as_completed(future_to_stem):
                stem = future_to_stem[future]
                try:
                    metrics, img_errors = future.result()
                    for e in img_errors:
                        res.errors.append(ValidationError(**e))
                    if metrics is not None:
                        quality_results[stem] = metrics
                        res.valid_images += 1
                        if metrics.blur_score < self.config.blur_threshold:
                            res.blurry_images += 1
                        if metrics.contrast < self.config.contrast_threshold:
                            res.low_contrast_images += 1
                    else:
                        res.invalid_images += 1
                except Exception as exc:
                    res.invalid_images += 1
                    res.errors.append(ValidationError(split=split_name, image_name=stem, error_type="WORKER_ERROR", severity=Severity.ERROR, description=f"Worker: {exc}"))
        res.quality_metrics = quality_results
        label_args = (str(lbl_dir), split_name, img_stems)
        lbl_errors, all_annotations, class_dist = validate_labels_for_split(label_args)
        for e in lbl_errors:
            res.errors.append(ValidationError(**e))
        res.class_distribution = class_dist
        lbl_stems = {p.stem for p in lbl_dir.glob(f"*{LABEL_EXTENSION}")}
        for stem in img_stems:
            if stem not in lbl_stems:
                res.errors.append(ValidationError(split=split_name, image_name=f"{stem}.*", error_type="MISSING_LABEL", severity=Severity.WARNING, description=f"Ảnh {stem} thiếu label"))
        iou_errors, overlapping = self._analyze_iou(all_annotations, split_name)
        for e in iou_errors:
            res.errors.append(ValidationError(**e))
        res.overlapping_bboxes = overlapping
        return res

    def _analyze_iou(self, annotations, split_name):
        errors: List[dict] = []
        overlapping = 0
        by_file: Dict[str, List] = defaultdict(list)
        for ln, lid, _, cx, cy, w, h in annotations:
            by_file[ln].append((lid, cx, cy, w, h))
        for ln, bboxes in by_file.items():
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    iou_val = compute_iou([bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]], [bboxes[j][1], bboxes[j][2], bboxes[j][3], bboxes[j][4]])
                    if iou_val > self.config.iou_threshold:
                        overlapping += 1
                        errors.append({"split": split_name, "label_name": ln, "error_type": "HIGH_IOU_OVERLAP", "severity": Severity.WARNING, "description": f"IoU={iou_val:.2f} dòng {bboxes[i][0]}-{bboxes[j][0]}", "line_number": bboxes[i][0]})
        return errors, overlapping

    def detect_leakage(self, summary: DatasetValidationSummary) -> List[ValidationError]:
        leakage: List[ValidationError] = []
        all_hashes: Dict[str, Tuple[str, str, str]] = {}
        for sn, result in [("train", summary.train), ("val", summary.val), ("test", summary.test)]:
            if result is None:
                continue
            for stem, m in result.quality_metrics.items():
                for hv, ht in [(m.phash, "phash"), (m.ahash, "ahash"), (m.md5, "md5")]:
                    if not hv:
                        continue
                    for eh, (es, ens, _) in all_hashes.items():
                        if es == sn:
                            continue
                        dist = hamming_distance(hv, eh) if ht != "md5" else (0 if hv == eh else 999)
                        is_match = (ht == "md5" and hv == eh) or (ht != "md5" and dist <= self.config.leakage_hamming_threshold)
                        if is_match:
                            sev = Severity.CRITICAL if es != sn else Severity.WARNING
                            et = "CROSS_SPLIT_LEAKAGE" if es != sn else "DUPLICATE_CONTENT"
                            leakage.append(ValidationError(split=sn, image_name=f"{stem}.*", error_type=et, severity=sev, description=f"Trùng {ht} với {ens} [{es.upper()}]"))
                            break
                    else:
                        all_hashes[hv] = (sn, stem, ht)
        return leakage

    def process_all(self, splits: Dict[str, Tuple[Path, Path]]) -> DatasetValidationSummary:
        all_files: List[Path] = []
        for s_name, (img_d, lbl_d) in splits.items():
            if img_d.exists():
                all_files.extend(list(img_d.iterdir()))
            if lbl_d.exists():
                all_files.extend(list(lbl_d.iterdir()))
        current_fp = self.generate_fingerprint(all_files)
        del all_files
        cache_data = self.load_cache()
        if cache_data.get("fingerprint") == current_fp and "summary" in cache_data:
            logger.info("Dataset không đổi — khôi phục từ cache.")
            cached = cache_data["summary"]
            summary = DatasetValidationSummary(fingerprint=current_fp, metadata=cached.get("metadata", {}))
            for sn in ["train", "val", "test"]:
                if sn in cached.get("splits", {}):
                    sd = cached["splits"][sn]
                    from models import ValidationResult
                    res = ValidationResult(split_name=sn)
                    for k, v in sd.items():
                        if hasattr(res, k) and k not in ("quality_metrics", "errors"):
                            setattr(res, k, v)
                    setattr(summary, sn, res)
            return summary
        summary = DatasetValidationSummary(fingerprint=current_fp)
        start_time = time.time()
        for s_name, (img_d, lbl_d) in splits.items():
            logger.info("Phân tích split '%s'...", s_name.upper())
            res = self.analyze_split(s_name, img_d, lbl_d)
            if s_name == "train":
                summary.train = res
            elif s_name == "val":
                summary.val = res
            elif s_name == "test":
                summary.test = res
        logger.info("Kiểm tra data leakage...")
        summary.leakage_errors = self.detect_leakage(summary)
        elapsed = time.time() - start_time
        summary.metadata["execution_time"] = datetime.now(timezone.utc).isoformat()
        summary.metadata["elapsed_seconds"] = round(elapsed, 2)
        cache_payload = {"fingerprint": current_fp, "summary": {"splits": {}, "metadata": summary.metadata}}
        for sn in ["train", "val", "test"]:
            r = getattr(summary, sn)
            if r is not None:
                cache_payload["summary"]["splits"][sn] = {"split_name": r.split_name, "total_images": r.total_images, "total_labels": r.total_labels, "valid_images": r.valid_images, "invalid_images": r.invalid_images, "overlapping_bboxes": r.overlapping_bboxes, "blurry_images": r.blurry_images, "low_contrast_images": r.low_contrast_images, "class_distribution": r.class_distribution}
        self.save_cache(cache_payload)
        return summary
