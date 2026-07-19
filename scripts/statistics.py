"""
statistics.py — Thống kê toàn bộ dataset theo định dạng YOLO.

Đọc dữ liệu từ datasets/processed/ (train, val, test), tính toán
các chỉ số thống kê về ảnh, label, bounding box và xuất báo cáo
ra file TXT, JSON, CSV trong thư mục reports/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Thêm project root vào sys.path để import config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    LABEL_EXTENSION,
    NUM_CLASSES,
    ProjectPaths,
    get_project_paths,
    logger as root_logger,
)

# Logger cho module này
logger = logging.getLogger("visiontextreader.statistics")


# ===========================================================
# Cấu trúc dữ liệu
# ===========================================================

@dataclass
class BBoxInfo:
    """Thông tin một bounding box YOLO."""
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Diện tích bbox (đã chuẩn hoá)."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Tỷ lệaspect ratio (width / height). Trả về 0 nếu height = 0."""
        if self.height == 0.0:
            return 0.0
        return self.width / self.height


@dataclass
class ImageInfo:
    """Thông tin một ảnh và các annotation của nó."""
    path: Path
    width: int = 0
    height: int = 0
    file_size: int = 0
    label_path: Optional[Path] = None
    label_exists: bool = False
    label_empty: bool = False
    boxes: List[BBoxInfo] = field(default_factory=list)

    @property
    def annotation_count(self) -> int:
        """Số lượng annotation trong ảnh."""
        return len(self.boxes)

    @property
    def resolution(self) -> Tuple[int, int]:
        """Kích thước ảnh (width, height)."""
        return (self.width, self.height)

    @property
    def megapixels(self) -> float:
        """Độ phân giải tính bằng megapixel."""
        return (self.width * self.height) / 1_000_000.0


@dataclass
class SplitStatistics:
    """Kết quả thống kê cho một split (train/val/test)."""
    split_name: str
    images: List[ImageInfo] = field(default_factory=list)

    # --- Tổng quan ---
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    empty_labels: int = 0
    images_without_labels: int = 0
    labels_without_images: int = 0

    # --- Phân bố annotation trên mỗi ảnh ---
    avg_annotations_per_image: float = 0.0
    max_annotations: int = 0
    min_annotations: int = 0

    # --- Phân bố class ---
    class_distribution: Dict[int, int] = field(default_factory=dict)

    # --- Kích thước ảnh ---
    max_image_width: int = 0
    max_image_height: int = 0
    min_image_width: int = 0
    min_image_height: int = 0
    avg_image_width: float = 0.0
    avg_image_height: float = 0.0
    resolution_distribution: Dict[str, int] = field(default_factory=dict)

    # --- Dung lượng file ---
    total_dataset_size_bytes: int = 0
    avg_image_size_bytes: float = 0.0
    largest_image_path: str = ""
    largest_image_size: int = 0
    smallest_image_path: str = ""
    smallest_image_size: int = 0

    # --- Bounding Box ---
    avg_bbox_width: float = 0.0
    avg_bbox_height: float = 0.0
    avg_bbox_area: float = 0.0
    avg_bbox_aspect_ratio: float = 0.0
    bbox_width_distribution: Dict[str, int] = field(default_factory=dict)
    bbox_height_distribution: Dict[str, int] = field(default_factory=dict)
    bbox_area_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetStatistics:
    """Kết quả thống kê toàn bộ dataset."""
    splits: Dict[str, SplitStatistics] = field(default_factory=dict)
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    avg_bbox_per_image: float = 0.0
    avg_resolution: Tuple[float, float] = (0.0, 0.0)
    total_dataset_size_bytes: int = 0
    total_bbox_width: float = 0.0
    total_bbox_height: float = 0.0
    total_bbox_area: float = 0.0
    total_bbox_aspect_ratio: float = 0.0


# ===========================================================
# Quét và đọc dữ liệu
# ===========================================================

def scan_split(
    paths: ProjectPaths,
    split_name: str,
) -> SplitStatistics:
    """Quét toàn bộ ảnh và label trong một split.

    Đọc ảnh từ processed/<split>/images/ và label từ processed/<split>/labels/.
    Trả về SplitStatistics với danh sách ImageInfo đã được điền đầy đủ.

    Args:
        paths: Cấu trúc thư mục dự án.
        Tên split cần quét ('train', 'val', 'test').

    Returns:
        SplitStatistics chứa thông tin chi tiết của split.
    """
    stats = SplitStatistics(split_name=split_name)

    # Lấy đường dẫn thư mục ảnh và label
    images_dir: Path = getattr(paths, f"{split_name}_images")
    labels_dir: Path = getattr(paths, f"{split_name}_labels")

    # Kiểm tra thư mục ảnh tồn tại
    if not images_dir.exists():
        logger.warning("Thư mục ảnh không tồn tại: %s", images_dir)
        return stats

    # Quét toàn bộ file ảnh trong thư mục
    image_files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*{ext}"))
    image_files = sorted(set(image_files))

    stats.total_images = len(image_files)
    logger.info("Split '%s': tìm thấy %d ảnh", split_name, len(image_files))

    # Duyệt qua từng ảnh để thu thập thông tin
    annotation_counts: List[int] = []
    widths: List[int] = []
    heights: List[int] = []
    file_sizes: List[int] = []
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []
    bbox_areas: List[float] = []
    bbox_aspect_ratios: List[float] = []
    class_counter: Counter = Counter()

    largest_size: int = 0
    largest_path: str = ""
    smallest_size: int = 0
    smallest_path: str = ""

    for img_path in image_files:
        img_info = _read_image_info(img_path, labels_dir)

        # Thêm vào danh sách
        stats.images.append(img_info)

        # Đếm label tồn tại
        if img_info.label_exists:
            stats.total_labels += 1
        else:
            stats.images_without_labels += 1

        # Đếm label rỗng
        if img_info.label_exists and img_info.label_empty:
            stats.empty_labels += 1

        # Đếm annotation
        annotation_counts.append(img_info.annotation_count)
        stats.total_annotations += img_info.annotation_count

        # Thu thập thông tin kích thước ảnh (chỉ khi ảnh đọc được)
        if img_info.width > 0 and img_info.height > 0:
            widths.append(img_info.width)
            heights.append(img_info.height)

            # Phân bố độ phân giải theo nhóm
            res_key = _resolution_bucket(img_info.width, img_info.height)
            stats.resolution_distribution[res_key] = (
                stats.resolution_distribution.get(res_key, 0) + 1
            )

        # Thu thập thông tin dung lượng file
        file_sizes.append(img_info.file_size)
        stats.total_dataset_size_bytes += img_info.file_size

        # Xác định ảnh lớn nhất và nhỏ nhất
        if img_info.file_size > largest_size:
            largest_size = img_info.file_size
            largest_path = img_path.name
        if smallest_size == 0 or img_info.file_size < smallest_size:
            smallest_size = img_info.file_size
            smallest_path = img_path.name

        # Thu thập thông tin bounding box
        for box in img_info.boxes:
            bbox_widths.append(box.width)
            bbox_heights.append(box.height)
            bbox_areas.append(box.area)
            if box.aspect_ratio > 0:
                bbox_aspect_ratios.append(box.aspect_ratio)
            class_counter[box.class_id] += 1

    # --- Tính toán các chỉ số thống kê tổng quát ---

    # Annotation
    if annotation_counts:
        stats.max_annotations = max(annotation_counts)
        stats.min_annotations = min(annotation_counts)
    else:
        stats.max_annotations = 0
        stats.min_annotations = 0

    if stats.total_images > 0:
        stats.avg_annotations_per_image = stats.total_annotations / stats.total_images

    # Phân bố class
    stats.class_distribution = dict(class_counter)

    # Kích thước ảnh
    if widths:
        stats.max_image_width = max(widths)
        stats.max_image_height = max(heights)
        stats.min_image_width = min(widths)
        stats.min_image_height = min(heights)
        stats.avg_image_width = sum(widths) / len(widths)
        stats.avg_image_height = sum(heights) / len(heights)

    # Dung lượng file
    if file_sizes:
        stats.avg_image_size_bytes = sum(file_sizes) / len(file_sizes)
    stats.largest_image_path = largest_path
    stats.largest_image_size = largest_size
    stats.smallest_image_path = smallest_path
    stats.smallest_image_size = smallest_size

    # Bounding Box
    if bbox_widths:
        stats.avg_bbox_width = sum(bbox_widths) / len(bbox_widths)
        stats.avg_bbox_height = sum(bbox_heights) / len(bbox_heights)
        stats.avg_bbox_area = sum(bbox_areas) / len(bbox_areas)
    if bbox_aspect_ratios:
        stats.avg_bbox_aspect_ratio = sum(bbox_aspect_ratios) / len(bbox_aspect_ratios)

    # Phân bố bounding box theo khoảng
    stats.bbox_width_distribution = _compute_distribution(bbox_widths, "width")
    stats.bbox_height_distribution = _compute_distribution(bbox_heights, "height")
    stats.bbox_area_distribution = _compute_distribution(bbox_areas, "area")

    # Đếm label không có ảnh
    if labels_dir.exists():
        label_files = list(labels_dir.glob(f"*{LABEL_EXTENSION}"))
        image_stems = {p.stem for p in image_files}
        for lbl in label_files:
            if lbl.stem not in image_stems:
                stats.labels_without_images += 1

    return stats


def _read_image_info(img_path: Path, labels_dir: Path) -> ImageInfo:
    """Đọc thông tin một ảnh và file label tương ứng.

    Args:
        img_path: Đường dẫn file ảnh.
        labels_dir: Thư mục chứa các file label.

    Returns:
        ImageInfo với thông tin đã được điền đầy đủ.
    """
    info = ImageInfo(path=img_path)

    # Đọc kích thước ảnh bằng Pillow
    try:
        with Image.open(img_path) as img:
            info.width, info.height = img.size
    except Exception as exc:
        logger.warning("Không thể đọc ảnh %s: %s", img_path.name, exc)
        info.width = 0
        info.height = 0

    # Lấy dung lượng file
    try:
        info.file_size = img_path.stat().st_size
    except OSError:
        info.file_size = 0

    # Tìm file label tương ứng
    lbl_path = labels_dir / f"{img_path.stem}{LABEL_EXTENSION}"
    info.label_path = lbl_path

    if lbl_path.exists():
        info.label_exists = True
        # Đọc nội dung label
        try:
            content = lbl_path.read_text(encoding="utf-8").strip()
            if content:
                info.boxes = _parse_yolo_labels(content)
            else:
                info.label_empty = True
        except IOError as exc:
            logger.warning("Không thể đọc label %s: %s", lbl_path.name, exc)
            info.label_empty = True

    return info


def _parse_yolo_labels(content: str) -> List[BBoxInfo]:
    """Phân tích nội dung file label YOLO.

    Mỗi dòng có định dạng: class_id cx cy width height

    Args:
        content: Nội dung file label.

    Returns:
        Danh sách BBoxInfo.
    """
    boxes: List[BBoxInfo] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append(BBoxInfo(
                class_id=class_id,
                cx=cx,
                cy=cy,
                width=w,
                height=h,
            ))
        except (ValueError, IndexError):
            continue
    return boxes


def _resolution_bucket(width: int, height: int) -> str:
    """Phân loại kích thước ảnh thành các nhóm.

    Args:
        width: Chiều rộng ảnh.
        height: Chiều cao ảnh.

    Returns:
        Chuỗi mô tả nhóm kích thước.
    """
    megapixels = (width * height) / 1_000_000.0
    if megapixels < 0.1:
        return "tiny (<0.1 MP)"
    elif megapixels < 0.5:
        return "small (0.1-0.5 MP)"
    elif megapixels < 2.0:
        return "medium (0.5-2 MP)"
    elif megapixels < 5.0:
        return "large (2-5 MP)"
    else:
        return "very large (>5 MP)"


def _compute_distribution(values: List[float], metric: str) -> Dict[str, int]:
    """Tính phân bố của danh sách giá trị theo khoảng.

    Args:
        values: Danh sách giá trị.
        metric: Loại metric ('width', 'height', 'area').

    Returns:
        Dict ánh xạ tên khoảng → số lượng.
    """
    distribution: Dict[str, int] = {}
    if not values:
        return distribution

    # Xác định khoảng phân bố dựa trên loại metric
    if metric == "width":
        bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        labels = [
            "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5",
            "0.5-0.7", "0.7-1.0",
        ]
    elif metric == "height":
        bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        labels = [
            "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5",
            "0.5-0.7", "0.7-1.0",
        ]
    else:  # area
        bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        labels = [
            "0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2",
            "0.2-0.5", "0.5-1.0",
        ]

    # Khởi tạo các khoảng với giá trị 0
    for label in labels:
        distribution[label] = 0

    # Phân loại từng giá trị vào khoảng phù hợp
    for val in values:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i + 1]:
                distribution[labels[i]] += 1
                placed = True
                break
        # Nếu giá trị bằng bins[-1], cho vào khoảng cuối cùng
        if not placed and val == bins[-1]:
            distribution[labels[-1]] += 1

    return distribution


# ===========================================================
# Tính toán thống kê toàn bộ dataset
# ===========================================================

def compute_statistics(
    paths: ProjectPaths,
    splits: Optional[List[str]] = None,
) -> DatasetStatistics:
    """Tính toán thống kê toàn bộ dataset.

    Quét từng split, tổng hợp kết quả thành DatasetStatistics.

    Args:
        paths: Cấu trúc thư mục dự án.
        Danh sách split cần thống kê. Nếu None, thống kê tất cả.

    Returns:
        DatasetStatistics chứa kết quả tổng hợp.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    result = DatasetStatistics()
    all_bbox_widths: List[float] = []
    all_bbox_heights: List[float] = []
    all_bbox_areas: List[float] = []
    all_widths: List[int] = []
    all_heights: List[int] = []

    for split_name in splits:
        logger.info("Đang quét split '%s'...", split_name)
        split_stats = scan_split(paths, split_name)
        result.splits[split_name] = split_stats

        # Tổng hợp
        result.total_images += split_stats.total_images
        result.total_labels += split_stats.total_labels
        result.total_annotations += split_stats.total_annotations
        result.total_dataset_size_bytes += split_stats.total_dataset_size_bytes

        # Thu thập bounding box để tính trung bình toàn cục
        for img in split_stats.images:
            for box in img.boxes:
                all_bbox_widths.append(box.width)
                all_bbox_heights.append(box.height)
                all_bbox_areas.append(box.area)
            if img.width > 0 and img.height > 0:
                all_widths.append(img.width)
                all_heights.append(img.height)

    # Tính các chỉ số tổng hợp
    if result.total_images > 0:
        result.avg_bbox_per_image = result.total_annotations / result.total_images

    if all_widths and all_heights:
        result.avg_resolution = (
            sum(all_widths) / len(all_widths),
            sum(all_heights) / len(all_heights),
        )

    if all_bbox_widths:
        result.total_bbox_width = sum(all_bbox_widths) / len(all_bbox_widths)
        result.total_bbox_height = sum(all_bbox_heights) / len(all_bbox_heights)
    if all_bbox_areas:
        result.total_bbox_area = sum(all_bbox_areas) / len(all_bbox_areas)
    if all_bbox_widths and all_bbox_heights:
        ratios = [
            w / h if h > 0 else 0.0
            for w, h in zip(all_bbox_widths, all_bbox_heights)
            if h > 0
        ]
        if ratios:
            result.total_bbox_aspect_ratio = sum(ratios) / len(ratios)

    return result


# ===========================================================
# Xuất báo cáo
# ===========================================================

class StatisticsExporter:
    """Xuất báo cáo thống kê ra nhiều định dạng khác nhau."""

    def __init__(self, output_dir: Path) -> None:
        """Khởi tạo exporter.

        Args:
            output_dir: Thư mục xuất báo cáo.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_text(
        self,
        dataset_stats: DatasetStatistics,
        filename: str = "statistics.txt",
    ) -> Path:
        """Xuất báo cáo dưới dạng file văn bản.

        Args:
            dataset_stats: Kết quả thống kê.
            Tên file xuất.

        Returns:
            Đường dẫn file đã tạo.
        """
        output_path = self.output_dir / filename
        lines: List[str] = []

        # Tiêu đề báo cáo
        lines.append("=" * 70)
        lines.append("  VISIONTEXTREADER DATASET REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Thống kê từng split
        for split_name in ["train", "val", "test"]:
            split = dataset_stats.splits.get(split_name)
            if split is None:
                continue

            lines.append(f"  {split_name.upper()}")
            lines.append(f"  {'-' * 40}")
            lines.append(f"    Tổng số ảnh:              {split.total_images}")
            lines.append(f"    Tổng số label:             {split.total_labels}")
            lines.append(f"    Tổng số annotation:        {split.total_annotations}")
            lines.append(f"    Annotation trung bình/ảnh: {split.avg_annotations_per_image:.2f}")
            lines.append(f"    Annotation nhiều nhất:     {split.max_annotations}")
            lines.append(f"    Annotation ít nhất:        {split.min_annotations}")
            lines.append(f"    Label rỗng:                {split.empty_labels}")
            lines.append(f"    Ảnh không có label:        {split.images_without_labels}")
            lines.append(f"    Label không có ảnh:        {split.labels_without_images}")
            lines.append("")

            # Kích thước ảnh
            lines.append(f"    Kích thước ảnh:")
            lines.append(f"      Lớn nhất:       {split.max_image_width}x{split.max_image_height}")
            lines.append(f"      Nhỏ nhất:       {split.min_image_width}x{split.min_image_height}")
            lines.append(f"      Trung bình:     {split.avg_image_width:.0f}x{split.avg_image_height:.0f}")
            lines.append("")

            # Phân bố độ phân giải
            if split.resolution_distribution:
                lines.append(f"    Phân bố độ phân giải:")
                for res_key, count in sorted(split.resolution_distribution.items()):
                    pct = (count / split.total_images * 100) if split.total_images > 0 else 0
                    lines.append(f"      {res_key:<25s} {count:>6d} ({pct:5.1f}%)")
                lines.append("")

            # Dung lượng file
            avg_mb = split.avg_image_size_bytes / (1024 * 1024)
            total_mb = split.total_dataset_size_bytes / (1024 * 1024)
            lines.append(f"    Dung lượng:")
            lines.append(f"      Tổng:           {total_mb:.2f} MB")
            lines.append(f"      Trung bình:     {avg_mb:.4f} MB")
            lines.append(f"      Ảnh lớn nhất:   {split.largest_image_path} ({split.largest_image_size / 1024:.1f} KB)")
            lines.append(f"      Ảnh nhỏ nhất:   {split.smallest_image_path} ({split.smallest_image_size / 1024:.1f} KB)")
            lines.append("")

            # Phân bố class
            if split.class_distribution:
                lines.append(f"    Phân bố class:")
                for cls_id in sorted(split.class_distribution.keys()):
                    cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    count = split.class_distribution[cls_id]
                    pct = (count / split.total_annotations * 100) if split.total_annotations > 0 else 0
                    lines.append(f"      class {cls_id} ({cls_name}): {count} ({pct:.1f}%)")
                lines.append("")

            # Bounding Box
            lines.append(f"    Bounding Box:")
            lines.append(f"      Width trung bình:       {split.avg_bbox_width:.4f}")
            lines.append(f"      Height trung bình:      {split.avg_bbox_height:.4f}")
            lines.append(f"      Area trung bình:        {split.avg_bbox_area:.6f}")
            lines.append(f"      Aspect Ratio trung bình:{split.avg_bbox_aspect_ratio:.4f}")
            lines.append("")

            # Phân bố Bounding Box theo width
            if split.bbox_width_distribution:
                lines.append(f"      Phân bố Width:")
                for bucket, count in sorted(split.bbox_width_distribution.items()):
                    lines.append(f"        {bucket:<15s} {count:>6d}")
                lines.append("")

            # Phân bố Bounding Box theo height
            if split.bbox_height_distribution:
                lines.append(f"      Phân bố Height:")
                for bucket, count in sorted(split.bbox_height_distribution.items()):
                    lines.append(f"        {bucket:<15s} {count:>6d}")
                lines.append("")

            # Phân bố Bounding Box theo area
            if split.bbox_area_distribution:
                lines.append(f"      Phân bố Area:")
                for bucket, count in sorted(split.bbox_area_distribution.items()):
                    lines.append(f"        {bucket:<15s} {count:>6d}")
                lines.append("")

            lines.append("")

        # Tổng quan toàn bộ dataset
        lines.append("=" * 70)
        lines.append("  TỔNG QUAN DATASET")
        lines.append("=" * 70)
        lines.append(f"    Tổng số ảnh:          {dataset_stats.total_images}")
        lines.append(f"    Tổng số label:         {dataset_stats.total_labels}")
        lines.append(f"    Tổng số annotation:    {dataset_stats.total_annotations}")
        lines.append(f"    Average BBox/Image:     {dataset_stats.avg_bbox_per_image:.2f}")
        lines.append(
            f"    Average Resolution:    "
            f"{dataset_stats.avg_resolution[0]:.0f}x{dataset_stats.avg_resolution[1]:.0f}"
        )
        lines.append(
            f"    Dataset Size:          "
            f"{dataset_stats.total_dataset_size_bytes / (1024 * 1024):.2f} MB"
        )
        lines.append("")
        lines.append(f"    Bounding Box (toàn bộ):")
        lines.append(f"      Width trung bình:      {dataset_stats.total_bbox_width:.4f}")
        lines.append(f"      Height trung bình:     {dataset_stats.total_bbox_height:.4f}")
        lines.append(f"      Area trung bình:       {dataset_stats.total_bbox_area:.6f}")
        lines.append(f"      Aspect Ratio TB:       {dataset_stats.total_bbox_aspect_ratio:.4f}")
        lines.append("=" * 70)

        # Ghi file
        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Đã xuất báo cáo TXT: %s", output_path)
        return output_path

    def export_json(
        self,
        dataset_stats: DatasetStatistics,
        filename: str = "statistics.json",
    ) -> Path:
        """Xuất báo cáo dưới dạng file JSON.

        Args:
            dataset_stats: Kết quả thống kê.
            Tên file xuất.

        Returns:
            Đường dẫn file đã tạo.
        """
        output_path = self.output_dir / filename

        # Chuyển đổi dataclass thành dict để serialise
        data: Dict[str, Any] = {
            "total_images": dataset_stats.total_images,
            "total_labels": dataset_stats.total_labels,
            "total_annotations": dataset_stats.total_annotations,
            "avg_bbox_per_image": dataset_stats.avg_bbox_per_image,
            "avg_resolution": {
                "width": dataset_stats.avg_resolution[0],
                "height": dataset_stats.avg_resolution[1],
            },
            "total_dataset_size_mb": (
                dataset_stats.total_dataset_size_bytes / (1024 * 1024)
            ),
            "bbox_statistics": {
                "avg_width": dataset_stats.total_bbox_width,
                "avg_height": dataset_stats.total_bbox_height,
                "avg_area": dataset_stats.total_bbox_area,
                "avg_aspect_ratio": dataset_stats.total_bbox_aspect_ratio,
            },
            "splits": {},
        }

        # Thêm thông tin từng split
        for split_name, split in dataset_stats.splits.items():
            split_data: Dict[str, Any] = {
                "total_images": split.total_images,
                "total_labels": split.total_labels,
                "total_annotations": split.total_annotations,
                "avg_annotations_per_image": split.avg_annotations_per_image,
                "max_annotations": split.max_annotations,
                "min_annotations": split.min_annotations,
                "empty_labels": split.empty_labels,
                "images_without_labels": split.images_without_labels,
                "labels_without_images": split.labels_without_images,
                "class_distribution": {
                    str(k): v for k, v in split.class_distribution.items()
                },
                "image_dimensions": {
                    "max_width": split.max_image_width,
                    "max_height": split.max_image_height,
                    "min_width": split.min_image_width,
                    "min_height": split.min_image_height,
                    "avg_width": split.avg_image_width,
                    "avg_height": split.avg_image_height,
                },
                "resolution_distribution": split.resolution_distribution,
                "file_size": {
                    "total_bytes": split.total_dataset_size_bytes,
                    "total_mb": split.total_dataset_size_bytes / (1024 * 1024),
                    "avg_bytes": split.avg_image_size_bytes,
                    "avg_mb": split.avg_image_size_bytes / (1024 * 1024),
                    "largest": {
                        "path": split.largest_image_path,
                        "size_bytes": split.largest_image_size,
                    },
                    "smallest": {
                        "path": split.smallest_image_path,
                        "size_bytes": split.smallest_image_size,
                    },
                },
                "bbox_statistics": {
                    "avg_width": split.avg_bbox_width,
                    "avg_height": split.avg_bbox_height,
                    "avg_area": split.avg_bbox_area,
                    "avg_aspect_ratio": split.avg_bbox_aspect_ratio,
                    "width_distribution": split.bbox_width_distribution,
                    "height_distribution": split.bbox_height_distribution,
                    "area_distribution": split.bbox_area_distribution,
                },
            }
            data["splits"][split_name] = split_data

        # Ghi file JSON
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Đã xuất báo cáo JSON: %s", output_path)
        return output_path

    def export_csv(
        self,
        dataset_stats: DatasetStatistics,
        filename: str = "statistics.csv",
    ) -> Path:
        """Xuất báo cáo dưới dạng file CSV.

        File CSV chứa 2 phần:
        1. Tổng quan các split
        2. Phân bố class

        Args:
            dataset_stats: Kết quả thống kê.
            Tên file xuất.

        Returns:
            Đường dẫn file đã tạo.
        """
        output_path = self.output_dir / filename

        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            # Phần 1: Tổng quan split
            writer.writerow(["=== TỔNG QUAN SPLIT ==="])
            writer.writerow([
                "Split",
                "Tổng ảnh",
                "Tổng label",
                "Tổng annotation",
                "TB annotation/ảnh",
                "Max annotation",
                "Min annotation",
                "Label rỗng",
                "Ảnh không có label",
                "Label không có ảnh",
                "TB Width ảnh",
                "TB Height ảnh",
                "TB Width bbox",
                "TB Height bbox",
                "TB Area bbox",
                "TB Aspect Ratio",
                "Tổng dung lượng (MB)",
                "TB dung lượng (MB)",
            ])

            for split_name in ["train", "val", "test"]:
                split = dataset_stats.splits.get(split_name)
                if split is None:
                    continue
                writer.writerow([
                    split_name,
                    split.total_images,
                    split.total_labels,
                    split.total_annotations,
                    f"{split.avg_annotations_per_image:.2f}",
                    split.max_annotations,
                    split.min_annotations,
                    split.empty_labels,
                    split.images_without_labels,
                    split.labels_without_images,
                    f"{split.avg_image_width:.0f}",
                    f"{split.avg_image_height:.0f}",
                    f"{split.avg_bbox_width:.4f}",
                    f"{split.avg_bbox_height:.4f}",
                    f"{split.avg_bbox_area:.6f}",
                    f"{split.avg_bbox_aspect_ratio:.4f}",
                    f"{split.total_dataset_size_bytes / (1024 * 1024):.2f}",
                    f"{split.avg_image_size_bytes / (1024 * 1024):.4f}",
                ])

            # Dòng tổng
            writer.writerow([
                "TOTAL",
                dataset_stats.total_images,
                dataset_stats.total_labels,
                dataset_stats.total_annotations,
                f"{dataset_stats.avg_bbox_per_image:.2f}",
                "",
                "",
                "",
                "",
                "",
                f"{dataset_stats.avg_resolution[0]:.0f}",
                f"{dataset_stats.avg_resolution[1]:.0f}",
                f"{dataset_stats.total_bbox_width:.4f}",
                f"{dataset_stats.total_bbox_height:.4f}",
                f"{dataset_stats.total_bbox_area:.6f}",
                f"{dataset_stats.total_bbox_aspect_ratio:.4f}",
                f"{dataset_stats.total_dataset_size_bytes / (1024 * 1024):.2f}",
                "",
            ])

            writer.writerow([])
            writer.writerow([])

            # Phần 2: Phân bố class
            writer.writerow(["=== PHÂN BỐ CLASS ==="])
            writer.writerow(["Split", "Class ID", "Class Name", "Số lượng", "Tỷ lệ (%)"])

            for split_name in ["train", "val", "test"]:
                split = dataset_stats.splits.get(split_name)
                if split is None:
                    continue
                for cls_id in sorted(split.class_distribution.keys()):
                    cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    count = split.class_distribution[cls_id]
                    pct = (
                        (count / split.total_annotations * 100)
                        if split.total_annotations > 0
                        else 0.0
                    )
                    writer.writerow([
                        split_name,
                        cls_id,
                        cls_name,
                        count,
                        f"{pct:.1f}",
                    ])

            writer.writerow([])
            writer.writerow([])

            # Phần 3: Phân bố độ phân giải
            writer.writerow(["=== PHÂN BỐ ĐỘ PHÂN GIẢI ==="])
            writer.writerow(["Split", "Nhóm", "Số lượng", "Tỷ lệ (%)"])

            for split_name in ["train", "val", "test"]:
                split = dataset_stats.splits.get(split_name)
                if split is None:
                    continue
                for res_key in sorted(split.resolution_distribution.keys()):
                    count = split.resolution_distribution[res_key]
                    pct = (
                        (count / split.total_images * 100)
                        if split.total_images > 0
                        else 0.0
                    )
                    writer.writerow([split_name, res_key, count, f"{pct:.1f}"])

            writer.writerow([])
            writer.writerow([])

            # Phần 4: Phân bố Bounding Box Width
            writer.writerow(["=== PHÂN BỐ BOUNDING BOX WIDTH ==="])
            writer.writerow(["Split", "Khoảng", "Số lượng"])

            for split_name in ["train", "val", "test"]:
                split = dataset_stats.splits.get(split_name)
                if split is None:
                    continue
                for bucket in sorted(split.bbox_width_distribution.keys()):
                    writer.writerow([
                        split_name,
                        bucket,
                        split.bbox_width_distribution[bucket],
                    ])

            writer.writerow([])
            writer.writerow([])

            # Phần 5: Phân bố Bounding Box Height
            writer.writerow(["=== PHÂN BỐ BOUNDING BOX HEIGHT ==="])
            writer.writerow(["Split", "Khoảng", "Số lượng"])

            for split_name in ["train", "val", "test"]:
                split = dataset_stats.splits.get(split_name)
                if split is None:
                    continue
                for bucket in sorted(split.bbox_height_distribution.keys()):
                    writer.writerow([
                        split_name,
                        bucket,
                        split.bbox_height_distribution[bucket],
                    ])

            writer.writerow([])
            writer.writerow([])

            # Phần 6: Phân bố Bounding Box Area
            writer.writerow(["=== PHÂN BỐ BOUNDING BOX AREA ==="])
            writer.writerow(["Split", "Khoảng", "Số lượng"])

            for split_name in ["train", "val", "test"]:
                split = dataset_stats.splits.get(split_name)
                if split is None:
                    continue
                for bucket in sorted(split.bbox_area_distribution.keys()):
                    writer.writerow([
                        split_name,
                        bucket,
                        split.bbox_area_distribution[bucket],
                    ])

        logger.info("Đã xuất báo cáo CSV: %s", output_path)
        return output_path


# ===========================================================
# In báo cáo ra màn hình
# ===========================================================

def print_summary(dataset_stats: DatasetStatistics) -> None:
    """In báo cáo tổng quan ra màn hình.

    Hiển thị bảng tóm tắt với các chỉ số chính của từng split
    và tổng quan toàn bộ dataset.

    Args:
        dataset_stats: Kết quả thống kê cần hiển thị.
    """
    sep = "=" * 70
    line = "-" * 70

    print(f"\n{sep}")
    print("  VISIONTEXTREADER DATASET REPORT")
    print(sep)

    # Bảng tóm tắt từng split
    header = f"  {'':>25s} {'Train':>12s} {'Validation':>12s} {'Test':>12s}"
    print(header)
    print(f"  {line}")

    # Tổng số ảnh
    train = dataset_stats.splits.get("train")
    val = dataset_stats.splits.get("val")
    test = dataset_stats.splits.get("test")

    def _val(split: Optional[SplitStatistics], attr: str) -> str:
        """Lấy giá trị từ split, trả về chuỗi hiển thị."""
        if split is None:
            return "N/A"
        value = getattr(split, attr, 0)
        if isinstance(value, float):
            return f"{value:,.2f}"
        return f"{value:,}"

    print(f"  {'Total Images':>25s} {_val(train, 'total_images'):>12s} {_val(val, 'total_images'):>12s} {_val(test, 'total_images'):>12s}")
    print(f"  {'Total Labels':>25s} {_val(train, 'total_labels'):>12s} {_val(val, 'total_labels'):>12s} {_val(test, 'total_labels'):>12s}")
    print(f"  {'Total Annotations':>25s} {_val(train, 'total_annotations'):>12s} {_val(val, 'total_annotations'):>12s} {_val(test, 'total_annotations'):>12s}")
    print(f"  {'Average BBox/Image':>25s} {_val(train, 'avg_annotations_per_image'):>12s} {_val(val, 'avg_annotations_per_image'):>12s} {_val(test, 'avg_annotations_per_image'):>12s}")

    # Average Resolution
    def _res(split: Optional[SplitStatistics]) -> str:
        if split is None or split.avg_image_width == 0:
            return "N/A"
        return f"{split.avg_image_width:.0f}x{split.avg_image_height:.0f}"

    print(f"  {'Average Resolution':>25s} {_res(train):>12s} {_res(val):>12s} {_res(test):>12s}")

    # Dataset Size
    def _size(split: Optional[SplitStatistics]) -> str:
        if split is None:
            return "N/A"
        mb = split.total_dataset_size_bytes / (1024 * 1024)
        return f"{mb:,.1f} MB"

    print(f"  {'Dataset Size':>25s} {_size(train):>12s} {_size(val):>12s} {_size(test):>12s}")

    print(f"  {line}")

    # Tổng quan
    total_mb = dataset_stats.total_dataset_size_bytes / (1024 * 1024)
    avg_w = dataset_stats.avg_resolution[0]
    avg_h = dataset_stats.avg_resolution[1]

    print(f"\n  {'TOTAL':>25s}")
    print(f"  {line}")
    print(f"  {'Total Images':>25s} {dataset_stats.total_images:>12,}")
    print(f"  {'Total Labels':>25s} {dataset_stats.total_labels:>12,}")
    print(f"  {'Total Annotations':>25s} {dataset_stats.total_annotations:>12,}")
    print(f"  {'Average BBox/Image':>25s} {dataset_stats.avg_bbox_per_image:>12.2f}")
    print(f"  {'Average Resolution':>25s} {avg_w:>6.0f}x{avg_h:<5.0f}")
    print(f"  {'Dataset Size':>25s} {total_mb:>10,.1f} MB")
    print(f"\n  {'BBox Width TB':>25s} {dataset_stats.total_bbox_width:>12.4f}")
    print(f"  {'BBox Height TB':>25s} {dataset_stats.total_bbox_height:>12.4f}")
    print(f"  {'BBox Area TB':>25s} {dataset_stats.total_bbox_area:>12.6f}")
    print(f"  {'BBox Aspect Ratio TB':>25s} {dataset_stats.total_bbox_aspect_ratio:>12.4f}")

    print(sep)
    print()


# ===========================================================
# Điểm vào chính
# ===========================================================

def main() -> None:
    """Điểm vào chính — CLI entry point.

    Hỗ trợ các tham số:
        --split: Chỉ thống kê một split cụ thể (train/val/test).
        --output: Thư mục xuất báo cáo (mặc định: reports/).
    """
    parser = argparse.ArgumentParser(
        description="Thống kê dataset YOLO cho VisionTextReader.",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["train", "val", "test"],
        help="Chỉ thống kê split cụ thể (mặc định: tất cả).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="reports",
        help="Thư mục xuất báo cáo (mặc định: reports/).",
    )
    args = parser.parse_args()

    # Lấy đường dẫn dự án
    paths = get_project_paths()

    # Xác định danh sách split cần thống kê
    splits_to_scan: Optional[List[str]] = None
    if args.split:
        splits_to_scan = [args.split]

    # Tính toán thống kê
    logger.info("Bắt đầu thống kê dataset...")
    dataset_stats = compute_statistics(paths, splits_to_scan)

    # In báo cáo ra màn hình
    print_summary(dataset_stats)

    # Xuất báo cáo ra file
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = paths.root / output_dir

    exporter = StatisticsExporter(output_dir)

    try:
        txt_path = exporter.export_text(dataset_stats)
        logger.info("Đã xuất file TXT: %s", txt_path)
    except Exception as exc:
        logger.error("Lỗi khi xuất file TXT: %s", exc)

    try:
        json_path = exporter.export_json(dataset_stats)
        logger.info("Đã xuất file JSON: %s", json_path)
    except Exception as exc:
        logger.error("Lỗi khi xuất file JSON: %s", exc)

    try:
        csv_path = exporter.export_csv(dataset_stats)
        logger.info("Đã xuất file CSV: %s", csv_path)
    except Exception as exc:
        logger.error("Lỗi khi xuất file CSV: %s", exc)

    logger.info("Hoàn tất thống kê dataset.")


if __name__ == "__main__":
    main()
