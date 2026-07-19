"""
verify_dataset.py — Dataset integrity verification for VisionTextReader.

Performs comprehensive checks on the YOLO-format processed dataset:
    - Missing images / labels (orphan detection)
    - Corrupt or empty files
    - Invalid bounding box coordinates
    - Class ID validation
    - Image readability
    - Duplicate images (MD5 Hash check)
    - Duplicate label files / lines
    - Bounding boxes outside image dimensions
    - Detailed statistics report

Usage:
    python scripts/verify_dataset.py                      # verify all splits
    python scripts/verify_dataset.py --split train        # verify one split
    python scripts/verify_dataset.py --fix                # remove broken pairs/duplicates
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

try:
    import send2trash
    HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSION,
    NUM_CLASSES,
    ProjectPaths,
    get_project_paths,
)

logger = logging.getLogger("visiontextreader.verify")


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class FileInfo:
    """Metadata about a single file for reporting."""
    path: Path
    size_bytes: int = 0
    is_empty: bool = False
    issue: str = ""


@dataclass
class SplitReport:
    """Verification report for a single train/val/test split."""
    split_name: str
    images_total: int = 0
    labels_total: int = 0
    images_without_labels: List[FileInfo] = field(default_factory=list)
    labels_without_images: List[FileInfo] = field(default_factory=list)
    empty_images: List[FileInfo] = field(default_factory=list)
    empty_labels: List[FileInfo] = field(default_factory=list)
    invalid_bboxes: List[Tuple[Path, str]] = field(default_factory=list)
    invalid_classes: List[Tuple[Path, int]] = field(default_factory=list)
    unreadable_images: List[FileInfo] = field(default_factory=list)
    duplicate_images: List[Tuple[Path, Path]] = field(default_factory=list)  # (duplicate, original)
    duplicate_labels: List[Path] = field(default_factory=list)
    bbox_outside_image: List[Tuple[Path, str]] = field(default_factory=list)
    total_annotations: int = 0
    class_distribution: Dict[int, int] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """
        Đã cập nhật theo chat: 
        Không coi empty_labels (ảnh background) và duplicate_labels là lỗi nghiêm trọng.
        """
        return not any([
            self.images_without_labels,
            self.labels_without_images,
            self.empty_images,
            self.invalid_bboxes,
            self.invalid_classes,
            self.duplicate_images,
            self.bbox_outside_image,
        ])

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"\n{'=' * 60}",
            f"  SPLIT: {self.split_name.upper()}",
            f"{'=' * 60}",
            f"  Images:              {self.images_total}",
            f"  Labels:              {self.labels_total}",
            f"  Total annotations:   {self.total_annotations}",
            f"  Healthy:             {'YES' if self.is_healthy else 'NO'}",
            "---",
            f"  Images w/o labels:   {len(self.images_without_labels)}",
            f"  Labels w/o images:   {len(self.labels_without_images)}",
            f"  Empty images:        {len(self.empty_images)}",
            f"  Empty labels (No Anno): {len(self.empty_labels)}",
            f"  Duplicate images:    {len(self.duplicate_images)}",
            f"  Duplicate labels:    {len(self.duplicate_labels)}",
            f"  Invalid bboxes:      {len(self.invalid_bboxes)}",
            f"  Invalid classes:     {len(self.invalid_classes)}",
            f"  BBox outside bounds: {len(self.bbox_outside_image)}",
            f"  Unreadable images:   {len(self.unreadable_images)}",
        ]

        if self.class_distribution:
            lines.append("---")
            lines.append("  Class distribution:")
            for cls_id in sorted(self.class_distribution.keys()):
                lines.append(f"    class {cls_id}: {self.class_distribution[cls_id]} annotations")

        for label, items in [
            ("Images without labels", self.images_without_labels[:5]),
            ("Labels without images", self.labels_without_images[:5]),
            ("Duplicate images (Dup -> Orig)", self.duplicate_images[:5]),
            ("BBox outside bounds", self.bbox_outside_image[:5]),
            ("Invalid bboxes", self.invalid_bboxes[:5]),
        ]:
            if items:
                lines.append(f"---  {label} (showing up to 5):")
                for item in items:
                    if isinstance(item, FileInfo):
                        lines.append(f"    {item.path.name}")
                    elif isinstance(item, tuple):
                        if isinstance(item[1], Path):
                            lines.append(f"    {item[0].name} is duplicate of {item[1].name}")
                        else:
                            lines.append(f"    {item[0].name}: {item[1]}")
                    elif isinstance(item, Path):
                        lines.append(f"    {item.name}")

        return "\n".join(lines)


# =========================================================================
# Verification engine
# =========================================================================

class DatasetVerifier:
    """Verifies integrity of a YOLO-format dataset."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.reports: List[SplitReport] = []

    def verify_split(self, split_name: str) -> SplitReport:
        """Run all checks on a single split."""
        images_dir = getattr(self.paths, f"{split_name}_images")
        labels_dir = getattr(self.paths, f"{split_name}_labels")

        report = SplitReport(split_name=split_name)

        if not images_dir.exists() or not labels_dir.exists():
            logger.warning("Directories do not exist for split: %s", split_name)
            return report

        logger.info("Verifying split '%s' ...", split_name)

        image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        label_paths = [p for p in labels_dir.iterdir() if p.suffix.lower() == LABEL_EXTENSION]

        report.images_total = len(image_paths)
        report.labels_total = len(label_paths)

        image_stems = {img.stem: img for img in image_paths}
        label_stems = {lbl.stem: lbl for lbl in label_paths}

        # 1. Orphan Checks
        for stem, img_path in image_stems.items():
            if stem not in label_stems:
                report.images_without_labels.append(FileInfo(path=img_path, size_bytes=img_path.stat().st_size))

        for stem, lbl_path in label_stems.items():
            if stem not in image_stems:
                report.labels_without_images.append(FileInfo(path=lbl_path, size_bytes=lbl_path.stat().st_size))

        # 2. Duplicate Content Checks
        self._check_duplicate_images(image_paths, report)
        self._check_duplicate_labels(label_paths, report)

        # 3. Deep File Validations
        common_stems = sorted(set(image_stems.keys()) & set(label_stems.keys()))
        for stem in common_stems:
            img_path = image_stems[stem]
            lbl_path = label_stems[stem]

            if img_path.stat().st_size == 0:
                report.empty_images.append(FileInfo(path=img_path, is_empty=True))
                continue

            img_dims = self._get_image_dimensions(img_path)
            if img_dims is None:
                report.unreadable_images.append(FileInfo(path=img_path, issue="Cannot decode image structure"))
                continue

            if self._check_empty_annotation(lbl_path, report):
                continue

            try:
                content = lbl_path.read_text(encoding="utf-8").strip()
            except IOError as exc:
                report.invalid_bboxes.append((lbl_path, f"Read error: {exc}"))
                continue

            seen_lines = set()
            for line_idx, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue

                if line in seen_lines:
                    continue
                seen_lines.add(line)

                parts = line.split()
                if len(parts) < 5:
                    report.invalid_bboxes.append((lbl_path, f"Line {line_idx}: expected 5 fields, got {len(parts)}"))
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError as exc:
                    report.invalid_bboxes.append((lbl_path, f"Line {line_idx}: parse error: {exc}"))
                    continue

                if class_id < 0 or class_id >= NUM_CLASSES:
                    report.invalid_classes.append((lbl_path, class_id))

                self._check_bbox_outside_image(lbl_path, line_idx, cx, cy, w, h, report)

                report.total_annotations += 1
                report.class_distribution[class_id] = report.class_distribution.get(class_id, 0) + 1

        return report

    def verify_all(self) -> List[SplitReport]:
        self.reports = []
        for split in ("train", "val", "test"):
            self.reports.append(self.verify_split(split))
        return self.reports

    def _check_duplicate_images(self, image_paths: List[Path], report: SplitReport) -> None:
        """Đọc file theo khối tuần tự để tránh hash nhầm."""
        hashes: Dict[str, Path] = {}
        for img_path in image_paths:
            if img_path.stat().st_size == 0:
                continue
            try:
                hasher = hashlib.md5()
                with open(img_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                img_hash = hasher.hexdigest()
                if img_hash in hashes:
                    report.duplicate_images.append((img_path, hashes[img_hash]))
                else:
                    hashes[img_hash] = img_path
            except IOError:
                pass

    def _check_duplicate_labels(self, label_paths: List[Path], report: SplitReport) -> None:
        label_hashes: Dict[str, Path] = {}
        for lbl_path in label_paths:
            if lbl_path.stat().st_size == 0:
                continue
            try:
                content_hash = hashlib.md5(lbl_path.read_bytes()).hexdigest()
                if content_hash in label_hashes:
                    report.duplicate_labels.append(lbl_path)
                else:
                    label_hashes[content_hash] = lbl_path
            except IOError:
                pass

    def _check_empty_annotation(self, label_path: Path, report: SplitReport) -> bool:
        if label_path.stat().st_size == 0:
            report.empty_labels.append(FileInfo(path=label_path, is_empty=True))
            return True
        try:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                report.empty_labels.append(FileInfo(path=label_path, is_empty=True))
                return True
        except IOError:
            pass
        return False

    def _check_bbox_outside_image(self, lbl_path: Path, line_idx: int, cx: float, cy: float, w: float, h: float, report: SplitReport) -> None:
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        issues = []
        if x1 < -0.01 or x2 > 1.01:
            issues.append(f"X-bounds [{x1:.2f}, {x2:.2f}] outside [0,1]")
        if y1 < -0.01 or y2 > 1.01:
            issues.append(f"Y-bounds [{y1:.2f}, {y2:.2f}] outside [0,1]")

        if issues:
            report.bbox_outside_image.append((lbl_path, f"Line {line_idx}: {' & '.join(issues)}"))

    @staticmethod
    def _get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None


# =========================================================================
# Execution & Action Logic
# =========================================================================

def print_full_report(reports: List[SplitReport]) -> None:
    print("\n" + "=" * 70)
    print("  DATASET VERIFICATION REPORT (ENHANCED)")
    print("  VisionTextReader — YOLO Format Dataset Integrity")
    print("=" * 70)

    total_images, total_labels, total_annotations = 0, 0, 0
    all_healthy = True

    for report in reports:
        print(report.summary())
        total_images += report.images_total
        total_labels += report.labels_total
        total_annotations += report.total_annotations
        if not report.is_healthy:
            all_healthy = False

    print("\n" + "-" * 70)
    print("  GRAND TOTAL SUMMARY")
    print("-" * 70)
    print(f"  Total images across dataset:   {total_images}")
    print(f"  Total labels across dataset:   {total_labels}")
    print(f"  Total annotations across data: {total_annotations}")
    print(f"  Overall status validation:     {'ALL HEALTHY' if all_healthy else 'ISSUES FOUND'}")
    print("=" * 70)


def remove_broken_pairs(reports: List[SplitReport], paths: ProjectPaths) -> int:
    """
    Sử dụng send2trash (nếu có) để đưa file vào Thùng rác, hoặc xóa trực tiếp.
    """
    removed = 0
    for report in reports:
        # 1. Xử lý các file mồ côi hoặc ảnh rỗng lỗi
        for info in (report.images_without_labels + report.labels_without_images + report.empty_images):
            if info.path.exists():
                try:
                    if HAS_SEND2TRASH:
                        send2trash.send2trash(str(info.path))
                    else:
                        info.path.unlink()
                    removed += 1
                except Exception as e:
                    logger.error("Không thể xóa %s: %s", info.path.name, e)

        # 2. Xử lý ảnh trùng lặp nội dung và tệp nhãn đồng bộ của nó
        for dup_path, _ in report.duplicate_images:
            if dup_path.exists():
                try:
                    if HAS_SEND2TRASH:
                        send2trash.send2trash(str(dup_path))
                    else:
                        dup_path.unlink()
                    removed += 1

                    lbl_sync = getattr(paths, f"{report.split_name}_labels") / f"{dup_path.stem}.txt"
                    if lbl_sync.exists():
                        if HAS_SEND2TRASH:
                            send2trash.send2trash(str(lbl_sync))
                        else:
                            lbl_sync.unlink()
                        removed += 1
                except Exception as e:
                    logger.error("Lỗi xử lý file trùng lặp %s: %s", dup_path.name, e)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify YOLO-format dataset integrity.")
    parser.add_argument("--split", "-s", type=str, choices=["train", "val", "test"], help="Verify only one split.")
    parser.add_argument("--fix", "-f", action="store_true", help="Clean up broken/duplicate files automatically.")
    args = parser.parse_args()

    paths = get_project_paths()
    verifier = DatasetVerifier(paths)

    reports = [verifier.verify_split(args.split)] if args.split else verifier.verify_all()
    print_full_report(reports)

    if args.fix:
        removed = remove_broken_pairs(reports, paths)
        print(f"\n[FIX] Đã chuyển {removed} tệp lỗi/mồ côi/trùng lặp vào Thùng rác hệ thống.")
        if removed > 0:
            print("\nĐang xác minh lại cấu trúc thư mục sau khi sửa đổi...")
            print_full_report([verifier.verify_split(args.split)] if args.split else verifier.verify_all())


if __name__ == "__main__":
    main()
