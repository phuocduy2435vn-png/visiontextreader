"""
dataset_validator.py — Dataset validation for YOLO training.

Checks: missing files, invalid annotations, image quality, geometry stats,
class distribution, IoU overlap, dataset leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("visiontextreader.dataset_validator")


# ===========================================================
# Data Classes
# ===========================================================

class Severity:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationFinding:
    severity: str
    category: str
    message: str
    file_path: str = ""
    line_number: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class ImageStats:
    path: str
    width: int = 0
    height: int = 0
    channels: int = 0
    dtype: str = ""
    file_size: int = 0
    is_rgba: bool = False
    is_16bit: bool = False
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0


@dataclass
class AnnotationStats:
    image_path: str
    num_boxes: int = 0
    class_ids: list[int] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    aspect_ratios: list[float] = field(default_factory=list)
    cx_list: list[float] = field(default_factory=list)
    cy_list: list[float] = field(default_factory=list)


@dataclass
class DatasetReport:
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    findings: list[ValidationFinding] = field(default_factory=list)
    image_stats: list[ImageStats] = field(default_factory=list)
    annotation_stats: list[AnnotationStats] = field(default_factory=list)
    class_distribution: dict[int, int] = field(default_factory=dict)
    resolution_distribution: dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def health_score(self) -> float:
        if self.total_images == 0:
            return 0.0
        penalty = self.critical_count * 5 + self.error_count * 2 + self.warning_count * 0.5
        return max(0.0, min(100.0, 100.0 - penalty / max(1, self.total_images) * 100))


# ===========================================================
# Dataset Validator
# ===========================================================

class DatasetValidator:
    """Comprehensive YOLO dataset validator."""

    BLUR_THRESHOLD = 50.0
    BRIGHTNESS_LOW = 20.0
    BRIGHTNESS_HIGH = 235.0
    CONTRAST_LOW = 15.0
    IOU_THRESHOLD = 0.5

    def __init__(
        self,
        dataset_dir: Path,
        class_names: dict[int, str] | None = None,
        num_classes: int = 1,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names or {0: "text"}
        self.num_classes = num_classes

    def validate(self, splits: list[str] | None = None) -> DatasetReport:
        """Run full validation on dataset."""
        if splits is None:
            splits = ["train", "val", "test"]

        report = DatasetReport()
        start_time = time.time()

        for split in splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="structure",
                    message=f"Split directory not found: {split}",
                ))
                continue
            self._validate_split(split, split_dir, report)

        self._check_leakage(splits, report)
        report.processing_time = time.time() - start_time
        logger.info(
            "Validation complete: %d images, %d findings, %.2fs",
            report.total_images, len(report.findings), report.processing_time,
        )
        return report

    def _validate_split(self, split: str, split_dir: Path, report: DatasetReport) -> None:
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            report.findings.append(ValidationFinding(
                severity=Severity.ERROR, category="structure",
                message=f"Images directory not found: {img_dir}",
            ))
            return

        image_files = self._collect_images(img_dir)
        label_files = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}

        report.total_images += len(image_files)
        report.total_labels += len(label_files)

        for img_path in image_files:
            stem = img_path.stem
            if stem not in label_files:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="missing_label",
                    message=f"Missing label for image: {img_path.name}", file_path=str(img_path),
                ))
                continue

            img_stats = self._validate_image(img_path, report)
            if img_stats:
                report.image_stats.append(img_stats)
                report.valid_images += 1
                res_key = f"{img_stats.width}x{img_stats.height}"
                report.resolution_distribution[res_key] = report.resolution_distribution.get(res_key, 0) + 1
            else:
                report.invalid_images += 1

            lbl_path = label_files[stem]
            ann_stats = self._validate_label(lbl_path, img_stats, report)
            if ann_stats:
                report.annotation_stats.append(ann_stats)
                report.total_annotations += ann_stats.num_boxes
                for cls_id in ann_stats.class_ids:
                    report.class_distribution[cls_id] = report.class_distribution.get(cls_id, 0) + 1

        for stem, lbl_path in label_files.items():
            if stem not in {p.stem for p in image_files}:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="missing_image",
                    message=f"Label without image: {lbl_path.name}", file_path=str(lbl_path),
                ))

    def _collect_images(self, img_dir: Path) -> list[Path]:
        images: list[Path] = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
            images.extend(img_dir.glob(f"*{ext}"))
            images.extend(img_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _validate_image(self, img_path: Path, report: DatasetReport) -> ImageStats | None:
        stats = ImageStats(path=str(img_path))
        try:
            stats.file_size = img_path.stat().st_size
            if stats.file_size == 0:
                report.findings.append(ValidationFinding(
                    severity=Severity.ERROR, category="corrupted",
                    message=f"Empty image file: {img_path.name}", file_path=str(img_path),
                ))
                return None

            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                report.findings.append(ValidationFinding(
                    severity=Severity.ERROR, category="corrupted",
                    message=f"Cannot read image: {img_path.name}", file_path=str(img_path),
                ))
                return None

            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            stats.width = w
            stats.height = h
            stats.channels = channels
            stats.dtype = str(img.dtype)

            if w <= 0 or h <= 0:
                report.findings.append(ValidationFinding(
                    severity=Severity.ERROR, category="invalid_dimensions",
                    message=f"Invalid dimensions: {w}x{h}", file_path=str(img_path),
                ))
                return None

            if channels == 4:
                stats.is_rgba = True
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="rgba",
                    message=f"RGBA image (4 channels): {img_path.name}", file_path=str(img_path),
                ))

            if img.dtype == np.uint16:
                stats.is_16bit = True
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="16bit",
                    message=f"16-bit image: {img_path.name}", file_path=str(img_path),
                ))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels >= 3 else img

            stats.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if stats.blur_score < self.BLUR_THRESHOLD:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="blur",
                    message=f"Blurry image (score={stats.blur_score:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))

            stats.brightness = float(np.mean(gray))
            if stats.brightness < self.BRIGHTNESS_LOW:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="brightness",
                    message=f"Very dark image (brightness={stats.brightness:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))
            elif stats.brightness > self.BRIGHTNESS_HIGH:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="brightness",
                    message=f"Very bright image (brightness={stats.brightness:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))

            stats.contrast = float(np.std(gray))
            if stats.contrast < self.CONTRAST_LOW:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="contrast",
                    message=f"Low contrast (contrast={stats.contrast:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))

            return stats

        except Exception as exc:
            report.findings.append(ValidationFinding(
                severity=Severity.ERROR, category="exception",
                message=f"Error validating image {img_path.name}: {exc}", file_path=str(img_path),
            ))
            return None

    def _validate_label(
        self, lbl_path: Path, img_stats: ImageStats | None, report: DatasetReport,
    ) -> AnnotationStats | None:
        ann_stats = AnnotationStats(image_path=str(lbl_path))
        try:
            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                return ann_stats

            for line_idx, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR, category="invalid_format",
                        message=f"Line {line_idx}: expected 5 values, got {len(parts)}",
                        file_path=str(lbl_path), line_number=line_idx,
                    ))
                    continue

                try:
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError as exc:
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR, category="invalid_format",
                        message=f"Line {line_idx}: parse error: {exc}",
                        file_path=str(lbl_path), line_number=line_idx,
                    ))
                    continue

                if class_id < 0 or class_id >= self.num_classes:
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR, category="invalid_class",
                        message=f"Line {line_idx}: class_id {class_id} outside [0, {self.num_classes - 1}]",
                        file_path=str(lbl_path), line_number=line_idx,
                    ))

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR, category="invalid_bbox",
                        message=f"Line {line_idx}: center ({cx:.4f}, {cy:.4f}) outside [0,1]",
                        file_path=str(lbl_path), line_number=line_idx,
                    ))

                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR, category="invalid_bbox",
                        message=f"Line {line_idx}: size ({w:.4f}, {h:.4f}) invalid",
                        file_path=str(lbl_path), line_number=line_idx,
                    ))

                ann_stats.num_boxes += 1
                ann_stats.class_ids.append(class_id)
                ann_stats.widths.append(w)
                ann_stats.heights.append(h)
                ann_stats.areas.append(w * h)
                ann_stats.aspect_ratios.append(w / h if h > 0 else 0)
                ann_stats.cx_list.append(cx)
                ann_stats.cy_list.append(cy)

            self._check_duplicate_boxes(ann_stats, lbl_path, report)
            self._check_iou_overlap(ann_stats, lbl_path, report)

        except Exception as exc:
            report.findings.append(ValidationFinding(
                severity=Severity.ERROR, category="exception",
                message=f"Error reading label {lbl_path.name}: {exc}", file_path=str(lbl_path),
            ))

        return ann_stats

    def _check_duplicate_boxes(
        self, ann_stats: AnnotationStats, lbl_path: Path, report: DatasetReport,
    ) -> None:
        seen: set[tuple] = set()
        for i in range(ann_stats.num_boxes):
            key = (
                ann_stats.class_ids[i],
                round(ann_stats.cx_list[i], 6),
                round(ann_stats.cy_list[i], 6),
                round(ann_stats.widths[i], 6),
                round(ann_stats.heights[i], 6),
            )
            if key in seen:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING, category="duplicate_bbox",
                    message=f"Duplicate bounding box at index {i}", file_path=str(lbl_path),
                ))
            seen.add(key)

    def _check_iou_overlap(
        self, ann_stats: AnnotationStats, lbl_path: Path, report: DatasetReport,
    ) -> None:
        n = ann_stats.num_boxes
        if n < 2:
            return
        for i in range(n):
            for j in range(i + 1, n):
                x1_i = ann_stats.cx_list[i] - ann_stats.widths[i] / 2
                y1_i = ann_stats.cy_list[i] - ann_stats.heights[i] / 2
                x2_i = ann_stats.cx_list[i] + ann_stats.widths[i] / 2
                y2_i = ann_stats.cy_list[i] + ann_stats.heights[i] / 2

                x1_j = ann_stats.cx_list[j] - ann_stats.widths[j] / 2
                y1_j = ann_stats.cy_list[j] - ann_stats.heights[j] / 2
                x2_j = ann_stats.cx_list[j] + ann_stats.widths[j] / 2
                y2_j = ann_stats.cy_list[j] + ann_stats.heights[j] / 2

                ix1 = max(x1_i, x1_j)
                iy1 = max(y1_i, y1_j)
                ix2 = min(x2_i, x2_j)
                iy2 = min(y2_i, y2_j)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_i = ann_stats.widths[i] * ann_stats.heights[i]
                area_j = ann_stats.widths[j] * ann_stats.heights[j]
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0.0

                if iou > self.IOU_THRESHOLD:
                    report.findings.append(ValidationFinding(
                        severity=Severity.WARNING, category="iou_overlap",
                        message=f"High IoU ({iou:.2f}) between boxes {i} and {j}",
                        file_path=str(lbl_path),
                    ))

    def _check_leakage(self, splits: list[str], report: DatasetReport) -> None:
        split_hashes: dict[str, set[str]] = defaultdict(set)
        for split in splits:
            img_dir = self.dataset_dir / split / "images"
            if not img_dir.exists():
                continue
            for img_path in self._collect_images(img_dir):
                try:
                    with open(img_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    split_hashes[split].add(file_hash)
                except Exception:
                    pass

        split_names = list(split_hashes.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = split_hashes[s1] & split_hashes[s2]
                if overlap:
                    report.findings.append(ValidationFinding(
                        severity=Severity.CRITICAL, category="leakage",
                        message=f"Data leakage: {len(overlap)} duplicate images between {s1} and {s2}",
                        details={"overlap_count": len(overlap)},
                    ))

    def print_report(self, report: DatasetReport) -> None:
        """Print validation report to console."""
        print("\n" + "=" * 70)
        print("  DATASET VALIDATION REPORT")
        print("=" * 70)
        print(f"  Total Images:     {report.total_images}")
        print(f"  Total Labels:     {report.total_labels}")
        print(f"  Total Annotations:{report.total_annotations}")
        print(f"  Valid Images:     {report.valid_images}")
        print(f"  Invalid Images:   {report.invalid_images}")
        print(f"  Health Score:     {report.health_score:.1f}/100")
        print(f"  Processing Time:  {report.processing_time:.2f}s")
        print("-" * 70)

        severity_counts = Counter(f.severity for f in report.findings)
        print("  Findings by Severity:")
        for sev in [Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                print(f"    {sev}: {count}")

        cat_counts = Counter(f.category for f in report.findings)
        if cat_counts:
            print("\n  Findings by Category:")
            for cat, count in cat_counts.most_common(10):
                print(f"    {cat}: {count}")

        if report.class_distribution:
            print("\n  Class Distribution:")
            for cls_id, count in sorted(report.class_distribution.items()):
                name = self.class_names.get(cls_id, f"class_{cls_id}")
                print(f"    {name} (id={cls_id}): {count}")

        print("=" * 70)


# ===========================================================
# Report Export
# ===========================================================

class ReportExporter:
    """Export validation reports to JSON and CSV."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, report: DatasetReport, filename: str = "validation_report.json") -> Path:
        data = {
            "total_images": report.total_images,
            "total_labels": report.total_labels,
            "total_annotations": report.total_annotations,
            "valid_images": report.valid_images,
            "invalid_images": report.invalid_images,
            "health_score": report.health_score,
            "processing_time": report.processing_time,
            "class_distribution": report.class_distribution,
            "resolution_distribution": report.resolution_distribution,
            "findings": [
                {"severity": f.severity, "category": f.category, "message": f.message,
                 "file_path": f.file_path, "line_number": f.line_number}
                for f in report.findings
            ],
        }
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_csv(self, report: DatasetReport, filename: str = "error_log.csv") -> Path:
        path = self.output_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Severity", "Category", "Message", "File", "Line"])
            for finding in report.findings:
                writer.writerow([
                    finding.severity, finding.category, finding.message,
                    finding.file_path, finding.line_number,
                ])
        return path


# ===========================================================
# CLI
# ===========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Dataset Validator")
    parser.add_argument("dataset_dir", type=str, help="Path to YOLO dataset")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="Splits to validate")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--output", "-o", type=str, default="output/validation",
                        help="Output directory")
    parser.add_argument("--export", type=str, nargs="+", default=["json", "csv"],
                        help="Export formats")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    validator = DatasetValidator(
        dataset_dir=Path(args.dataset_dir),
        num_classes=args.num_classes,
    )

    report = validator.validate(splits=args.splits)
    validator.print_report(report)

    exporter = ReportExporter(Path(args.output))
    if "json" in args.export:
        path = exporter.export_json(report)
        print(f"JSON report: {path}")
    if "csv" in args.export:
        path = exporter.export_csv(report)
        print(f"CSV report: {path}")


if __name__ == "__main__":
    main()
