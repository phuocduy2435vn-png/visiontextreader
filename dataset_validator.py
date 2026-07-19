"""
dataset_validator.py — Comprehensive dataset validation for YOLO training.

Checks: missing files, invalid annotations, image quality, geometry stats,
class distribution, IoU overlap, dataset leakage, and more.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("visiontextreader.dataset_validator")


# ===========================================================
# Severity Levels
# ===========================================================

class Severity:
    """Severity levels for validation findings."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ===========================================================
# Data Classes
# ===========================================================

@dataclass
class ValidationFinding:
    """A single validation finding."""
    severity: str
    category: str
    message: str
    file_path: str = ""
    line_number: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageStats:
    """Statistics for a single image."""
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
    """Statistics for annotations in an image."""
    image_path: str
    num_boxes: int = 0
    class_ids: List[int] = field(default_factory=list)
    widths: List[float] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)
    cx_list: List[float] = field(default_factory=list)
    cy_list: List[float] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Complete validation report."""
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    findings: List[ValidationFinding] = field(default_factory=list)
    image_stats: List[ImageStats] = field(default_factory=list)
    annotation_stats: List[AnnotationStats] = field(default_factory=list)
    class_distribution: Dict[int, int] = field(default_factory=dict)
    resolution_distribution: Dict[str, int] = field(default_factory=dict)
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
        """Compute health score 0-100."""
        if self.total_images == 0:
            return 0.0
        penalty = self.critical_count * 5 + self.error_count * 2 + self.warning_count * 0.5
        return max(0.0, min(100.0, 100.0 - penalty / max(1, self.total_images) * 100))


# ===========================================================
# Dataset Validator
# ===========================================================

class DatasetValidator:
    """Comprehensive YOLO dataset validator.

    Checks:
        - Missing images/labels
        - Invalid annotations (format, class, bbox)
        - Image quality (blur, brightness, contrast)
        - Geometry statistics
        - IoU overlap
        - Dataset leakage (duplicates)
    """

    def __init__(
        self,
        dataset_dir: Path,
        class_names: Optional[Dict[int, str]] = None,
        num_classes: int = 1,
    ):
        """Initialize validator.

        Args:
            dataset_dir: Root directory of YOLO dataset.
            class_names: Mapping of class_id to name.
            num_classes: Expected number of classes.
        """
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names or {0: "text"}
        self.num_classes = num_classes

    def validate(self, splits: Optional[List[str]] = None) -> DatasetReport:
        """Run full validation on dataset.

        Args:
            splits: List of splits to validate (default: train, val, test).

        Returns:
            Complete validation report.
        """
        if splits is None:
            splits = ["train", "val", "test"]

        report = DatasetReport()
        start_time = time.time()

        for split in splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="structure",
                    message=f"Split directory not found: {split}",
                ))
                continue

            self._validate_split(split, split_dir, report)

        # Check for leakage across splits
        self._check_leakage(splits, report)

        report.processing_time = time.time() - start_time
        logger.info(
            "Validation complete: %d images, %d findings, %.2fs",
            report.total_images, len(report.findings), report.processing_time,
        )

        return report

    def _validate_split(self, split: str, split_dir: Path, report: DatasetReport) -> None:
        """Validate a single split."""
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            report.findings.append(ValidationFinding(
                severity=Severity.ERROR,
                category="structure",
                message=f"Images directory not found: {img_dir}",
            ))
            return

        # Collect files
        image_files = self._collect_images(img_dir)
        label_files = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}

        report.total_images += len(image_files)
        report.total_labels += len(label_files)

        # Check missing labels
        for img_path in image_files:
            stem = img_path.stem
            if stem not in label_files:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="missing_label",
                    message=f"Missing label for image: {img_path.name}",
                    file_path=str(img_path),
                ))
                continue

            # Validate image
            img_stats = self._validate_image(img_path, report)
            if img_stats:
                report.image_stats.append(img_stats)
                report.valid_images += 1

                # Resolution distribution
                res_key = f"{img_stats.width}x{img_stats.height}"
                report.resolution_distribution[res_key] = \
                    report.resolution_distribution.get(res_key, 0) + 1
            else:
                report.invalid_images += 1

            # Validate label
            lbl_path = label_files[stem]
            ann_stats = self._validate_label(lbl_path, img_stats, report)
            if ann_stats:
                report.annotation_stats.append(ann_stats)
                report.total_annotations += ann_stats.num_boxes

                # Class distribution
                for cls_id in ann_stats.class_ids:
                    report.class_distribution[cls_id] = \
                        report.class_distribution.get(cls_id, 0) + 1

        # Check orphan labels
        for stem, lbl_path in label_files.items():
            if stem not in {p.stem for p in image_files}:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="missing_image",
                    message=f"Label without image: {lbl_path.name}",
                    file_path=str(lbl_path),
                ))

    def _collect_images(self, img_dir: Path) -> List[Path]:
        """Collect all image files from directory."""
        images = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
            images.extend(img_dir.glob(f"*{ext}"))
            images.extend(img_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _validate_image(self, img_path: Path, report: DatasetReport) -> Optional[ImageStats]:
        """Validate a single image file."""
        stats = ImageStats(path=str(img_path))

        try:
            # Check file size
            stats.file_size = img_path.stat().st_size
            if stats.file_size == 0:
                report.findings.append(ValidationFinding(
                    severity=Severity.ERROR,
                    category="corrupted",
                    message=f"Empty image file: {img_path.name}",
                    file_path=str(img_path),
                ))
                return None

            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                report.findings.append(ValidationFinding(
                    severity=Severity.ERROR,
                    category="corrupted",
                    message=f"Cannot read image: {img_path.name}",
                    file_path=str(img_path),
                ))
                return None

            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            stats.width = w
            stats.height = h
            stats.channels = channels
            stats.dtype = str(img.dtype)

            # Check dimensions
            if w <= 0 or h <= 0:
                report.findings.append(ValidationFinding(
                    severity=Severity.ERROR,
                    category="invalid_dimensions",
                    message=f"Invalid dimensions: {w}x{h}",
                    file_path=str(img_path),
                ))
                return None

            # Check RGBA
            if channels == 4:
                stats.is_rgba = True
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="rgba",
                    message=f"RGBA image (4 channels): {img_path.name}",
                    file_path=str(img_path),
                ))

            # Check 16-bit
            if img.dtype == np.uint16:
                stats.is_16bit = True
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="16bit",
                    message=f"16-bit image: {img_path.name}",
                    file_path=str(img_path),
                ))

            # Image quality checks
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels >= 3 else img

            # Blur score (Variance of Laplacian)
            stats.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if stats.blur_score < 50.0:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="blur",
                    message=f"Blurry image (score={stats.blur_score:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))

            # Brightness
            stats.brightness = float(np.mean(gray))
            if stats.brightness < 20.0:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="brightness",
                    message=f"Very dark image (brightness={stats.brightness:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))
            elif stats.brightness > 235.0:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="brightness",
                    message=f"Very bright image (brightness={stats.brightness:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))

            # Contrast
            stats.contrast = float(np.std(gray))
            if stats.contrast < 15.0:
                report.findings.append(ValidationFinding(
                    severity=Severity.WARNING,
                    category="contrast",
                    message=f"Low contrast (contrast={stats.contrast:.1f}): {img_path.name}",
                    file_path=str(img_path),
                ))

            return stats

        except Exception as exc:
            report.findings.append(ValidationFinding(
                severity=Severity.ERROR,
                category="exception",
                message=f"Error validating image {img_path.name}: {exc}",
                file_path=str(img_path),
            ))
            return None

    def _validate_label(
        self,
        lbl_path: Path,
        img_stats: Optional[ImageStats],
        report: DatasetReport,
    ) -> Optional[AnnotationStats]:
        """Validate a YOLO label file."""
        ann_stats = AnnotationStats(image_path=str(lbl_path))

        try:
            content = lbl_path.read_text(encoding="utf-8").strip()

            # Empty label
            if not content:
                return ann_stats

            for line_idx, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR,
                        category="invalid_format",
                        message=f"Line {line_idx}: expected 5 values, got {len(parts)}",
                        file_path=str(lbl_path),
                        line_number=line_idx,
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
                        severity=Severity.ERROR,
                        category="invalid_format",
                        message=f"Line {line_idx}: parse error: {exc}",
                        file_path=str(lbl_path),
                        line_number=line_idx,
                    ))
                    continue

                # Validate class_id
                if class_id < 0 or class_id >= self.num_classes:
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR,
                        category="invalid_class",
                        message=f"Line {line_idx}: class_id {class_id} outside [0, {self.num_classes-1}]",
                        file_path=str(lbl_path),
                        line_number=line_idx,
                    ))

                # Validate bbox coordinates
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR,
                        category="invalid_bbox",
                        message=f"Line {line_idx}: center ({cx:.4f}, {cy:.4f}) outside [0,1]",
                        file_path=str(lbl_path),
                        line_number=line_idx,
                    ))

                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    report.findings.append(ValidationFinding(
                        severity=Severity.ERROR,
                        category="invalid_bbox",
                        message=f"Line {line_idx}: size ({w:.4f}, {h:.4f}) invalid",
                        file_path=str(lbl_path),
                        line_number=line_idx,
                    ))

                # Collect stats
                ann_stats.num_boxes += 1
                ann_stats.class_ids.append(class_id)
                ann_stats.widths.append(w)
                ann_stats.heights.append(h)
                ann_stats.areas.append(w * h)
                ann_stats.aspect_ratios.append(w / h if h > 0 else 0)
                ann_stats.cx_list.append(cx)
                ann_stats.cy_list.append(cy)

            # Check for duplicate boxes
            self._check_duplicate_boxes(ann_stats, lbl_path, report)

            # Check IoU overlap
            self._check_iou_overlap(ann_stats, lbl_path, report)

        except Exception as exc:
            report.findings.append(ValidationFinding(
                severity=Severity.ERROR,
                category="exception",
                message=f"Error reading label {lbl_path.name}: {exc}",
                file_path=str(lbl_path),
            ))

        return ann_stats

    def _check_duplicate_boxes(
        self,
        ann_stats: AnnotationStats,
        lbl_path: Path,
        report: DatasetReport,
    ) -> None:
        """Check for duplicate bounding boxes."""
        seen = set()
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
                    severity=Severity.WARNING,
                    category="duplicate_bbox",
                    message=f"Duplicate bounding box at index {i}",
                    file_path=str(lbl_path),
                ))
            seen.add(key)

    def _check_iou_overlap(
        self,
        ann_stats: AnnotationStats,
        lbl_path: Path,
        report: DatasetReport,
    ) -> None:
        """Check for high IoU overlap between boxes."""
        n = ann_stats.num_boxes
        if n < 2:
            return

        for i in range(n):
            for j in range(i + 1, n):
                # Convert to xyxy for IoU
                x1_i = ann_stats.cx_list[i] - ann_stats.widths[i] / 2
                y1_i = ann_stats.cy_list[i] - ann_stats.heights[i] / 2
                x2_i = ann_stats.cx_list[i] + ann_stats.widths[i] / 2
                y2_i = ann_stats.cy_list[i] + ann_stats.heights[i] / 2

                x1_j = ann_stats.cx_list[j] - ann_stats.widths[j] / 2
                y1_j = ann_stats.cy_list[j] - ann_stats.heights[j] / 2
                x2_j = ann_stats.cx_list[j] + ann_stats.widths[j] / 2
                y2_j = ann_stats.cy_list[j] + ann_stats.heights[j] / 2

                # Compute IoU
                ix1 = max(x1_i, x1_j)
                iy1 = max(y1_i, y1_j)
                ix2 = min(x2_i, x2_j)
                iy2 = min(y2_i, y2_j)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_i = ann_stats.widths[i] * ann_stats.heights[i]
                area_j = ann_stats.widths[j] * ann_stats.heights[j]
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0.0

                if iou > 0.5:
                    report.findings.append(ValidationFinding(
                        severity=Severity.WARNING,
                        category="iou_overlap",
                        message=f"High IoU ({iou:.2f}) between boxes {i} and {j}",
                        file_path=str(lbl_path),
                    ))

    def _check_leakage(self, splits: List[str], report: DatasetReport) -> None:
        """Check for duplicate images across splits (data leakage)."""
        split_hashes: Dict[str, Set[str]] = defaultdict(set)

        for split in splits:
            img_dir = self.dataset_dir / split / "images"
            if not img_dir.exists():
                continue

            for img_path in self._collect_images(img_dir):
                try:
                    # Compute file hash
                    with open(img_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    split_hashes[split].add(file_hash)
                except Exception:
                    pass

        # Check for overlap
        split_names = list(split_hashes.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = split_hashes[s1] & split_hashes[s2]
                if overlap:
                    report.findings.append(ValidationFinding(
                        severity=Severity.CRITICAL,
                        category="leakage",
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

        # Summary by severity
        severity_counts = Counter(f.severity for f in report.findings)
        print("  Findings by Severity:")
        for sev in [Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                print(f"    {sev}: {count}")

        # Summary by category
        cat_counts = Counter(f.category for f in report.findings)
        if cat_counts:
            print("\n  Findings by Category:")
            for cat, count in cat_counts.most_common(10):
                print(f"    {cat}: {count}")

        # Class distribution
        if report.class_distribution:
            print("\n  Class Distribution:")
            for cls_id, count in sorted(report.class_distribution.items()):
                name = self.class_names.get(cls_id, f"class_{cls_id}")
                print(f"    {name} (id={cls_id}): {count}")

        print("=" * 70)


# ===========================================================
# Report Exporters
# ===========================================================

class ReportExporter:
    """Export validation reports to various formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, report: DatasetReport, filename: str = "validation_report.json") -> Path:
        """Export report as JSON."""
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
                {
                    "severity": f.severity,
                    "category": f.category,
                    "message": f.message,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                }
                for f in report.findings
            ],
        }
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_csv(self, report: DatasetReport, filename: str = "error_log.csv") -> Path:
        """Export findings as CSV."""
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

    def export_html(self, report: DatasetReport, filename: str = "validation_report.html") -> Path:
        """Export report as HTML dashboard."""
        # Group findings by category
        by_category = defaultdict(list)
        for f in report.findings:
            by_category[f.category].append(f)

        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>VisionTextReader — Dataset Validation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .health-badge {{ font-size: 2rem; padding: 15px 30px; border-radius: 15px; color: white; font-weight: bold; }}
        .health-good {{ background: linear-gradient(135deg, #4CAF50, #45a049); }}
        .health-warn {{ background: linear-gradient(135deg, #FF9800, #F57C00); }}
        .health-bad {{ background: linear-gradient(135deg, #f44336, #d32f2f); }}
        .card {{ border: none; border-radius: 12px; }}
        .metric-card {{ text-align: center; padding: 20px; }}
        .metric-value {{ font-size: 1.8rem; font-weight: bold; }}
    </style>
</head>
<body class="bg-light">
    <div class="container py-4">
        <h1 class="text-center mb-4">Dataset Validation Report</h1>

        <!-- Health Score -->
        <div class="text-center mb-4">
            <span class="health-badge {'health-good' if report.health_score >= 80 else 'health-warn' if report.health_score >= 50 else 'health-bad'}">
                Health Score: {report.health_score:.1f}/100
            </span>
        </div>

        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3"><div class="card metric-card shadow-sm"><div class="metric-value">{report.total_images:,}</div><div>Total Images</div></div></div>
            <div class="col-md-3"><div class="card metric-card shadow-sm"><div class="metric-value text-success">{report.valid_images:,}</div><div>Valid</div></div></div>
            <div class="col-md-3"><div class="card metric-card shadow-sm"><div class="metric-value text-danger">{report.invalid_images:,}</div><div>Invalid</div></div></div>
            <div class="col-md-3"><div class="card metric-card shadow-sm"><div class="metric-value text-warning">{len(report.findings):,}</div><div>Findings</div></div></div>
        </div>

        <!-- Charts -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card p-3 shadow-sm">
                    <h5>Findings by Severity</h5>
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card p-3 shadow-sm">
                    <h5>Class Distribution</h5>
                    <canvas id="classChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Findings Table -->
        <div class="card shadow-sm">
            <div class="card-header"><h5>Detailed Findings</h5></div>
            <div class="card-body">
                <table class="table table-striped">
                    <thead><tr><th>Severity</th><th>Category</th><th>Message</th><th>File</th></tr></thead>
                    <tbody>
"""
        for f in report.findings[:100]:  # Limit to 100
            sev_class = {"CRITICAL": "danger", "ERROR": "danger", "WARNING": "warning", "INFO": "info"}.get(f.severity, "secondary")
            html += f"""                        <tr><td><span class="badge bg-{sev_class}">{f.severity}</span></td>
                            <td>{f.category}</td><td>{f.message}</td><td>{Path(f.file_path).name if f.file_path else '-'}</td></tr>\n"""

        html += f"""                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Severity Chart
        const severityData = {json.dumps(dict(Counter(f.severity for f in report.findings)))};
        new Chart(document.getElementById('severityChart'), {{
            type: 'pie',
            data: {{
                labels: Object.keys(severityData),
                datasets: [{{ data: Object.values(severityData), backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#17a2b8'] }}]
            }}
        }});

        // Class Distribution Chart
        const classData = {json.dumps(report.class_distribution)};
        new Chart(document.getElementById('classChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(classData).map(k => 'Class ' + k),
                datasets: [{{ label: 'Count', data: Object.values(classData), backgroundColor: 'rgba(54,162,235,0.7)' }}]
            }},
            options: {{ responsive: true }}
        }});
    </script>
</body>
</html>"""

        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        return path


# ===========================================================
# CLI
# ===========================================================

def main() -> None:
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(description="YOLO Dataset Validator")
    parser.add_argument("dataset_dir", type=str, help="Path to YOLO dataset")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="Splits to validate")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--output", "-o", type=str, default="outputs/validation",
                        help="Output directory")
    parser.add_argument("--export", type=str, nargs="+", default=["json", "csv", "html"],
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

    # Export
    exporter = ReportExporter(Path(args.output))
    if "json" in args.export:
        path = exporter.export_json(report)
        print(f"JSON report: {path}")
    if "csv" in args.export:
        path = exporter.export_csv(report)
        print(f"CSV report: {path}")
    if "html" in args.export:
        path = exporter.export_html(report)
        print(f"HTML report: {path}")


if __name__ == "__main__":
    main()
