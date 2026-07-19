"""
dataset_statistics.py — Thống kê, trực quan hoá và xuất báo cáo dữ liệu YOLO.

Thu thập thống kê nâng cao từ datasets/processed/, sinh biểu đồ bằng matplotlib,
xuất báo cáo văn bản chi tiết và xuất dữ liệu sang định dạng Excel (.xlsx).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

# Thống kê thủ công tránh shadow hệ thống
def _mean(values: List[float] | List[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: List[float] | List[int]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return float(sorted_vals[mid])


def _stdev(values: List[float] | List[int]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("visiontextreader.dataset_statistics")


# ===========================================================
# Cấu trúc dữ liệu (Dataclass)
# ===========================================================

@dataclass
class ImageStatistics:
    """Thống kê cơ bản của một ảnh."""
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    bbox_count: int = 0


@dataclass
class BoundingBoxStatistics:
    """Thống kê của một bounding box."""
    class_id: int = 0
    area: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass
class SplitStatistics:
    """Thống kê tổng hợp toàn diện cho một split (train/val/test)."""
    split_name: str = ""
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0

    # Chỉ số phân phối kích thước ảnh (Bổ sung đầy đủ chỉ số nâng cao)
    average_width: float = 0.0
    median_width: float = 0.0
    std_width: float = 0.0
    min_width: int = 0
    max_width: int = 0

    average_height: float = 0.0
    median_height: float = 0.0
    std_height: float = 0.0
    min_height: int = 0
    max_height: int = 0

    # Chỉ số bounding box
    average_bbox_area: float = 0.0
    median_bbox_area: float = 0.0
    largest_bbox: Optional[BoundingBoxStatistics] = None
    smallest_bbox: Optional[BoundingBoxStatistics] = None

    class_distribution: Dict[int, int] = field(default_factory=dict)
    aspect_ratio_distribution: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class DatasetStatistics:
    """Thống kê tổng hợp cho toàn bộ dataset."""
    train: Optional[SplitStatistics] = None
    val: Optional[SplitStatistics] = None
    test: Optional[SplitStatistics] = None
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0


# ===========================================================
# DatasetStatisticsCollector — Thu thập dữ liệu
# ===========================================================

class DatasetStatisticsCollector:
    """Thu thập dữ liệu thống kê hình ảnh và nhãn định dạng YOLO."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.statistics = DatasetStatistics()

    def collect_image_statistics(self, image_path: Path) -> Optional[ImageStatistics]:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Không thể đọc ảnh: %s", image_path.name)
                return None
            height, width = image.shape[:2]
            del image
            if height == 0 or width == 0:
                return None
            return ImageStatistics(width=width, height=height, aspect_ratio=width / height)
        except Exception as exc:
            logger.warning("Lỗi khi đọc ảnh %s: %s", image_path.name, exc)
            return None

    def load_labels(self, label_path: Path) -> List[BoundingBoxStatistics]:
        boxes: List[BoundingBoxStatistics] = []
        if not label_path.exists():
            return boxes
        try:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                return boxes
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                if 0 <= class_id < NUM_CLASSES and w > 0 and h > 0:
                    boxes.append(BoundingBoxStatistics(class_id=class_id, area=w * h, width=w, height=h))
        except Exception:
            pass
        return boxes

    def collect_split(self, split_name: str) -> SplitStatistics:
        start_time = time.time()
        split_stats = SplitStatistics(split_name=split_name)
        images_dir = getattr(self.paths, f"{split_name}_images")
        labels_dir = getattr(self.paths, f"{split_name}_labels")

        if not images_dir.exists():
            return split_stats

        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(images_dir.glob(f"*{ext}"))
        image_files = sorted(set(image_files))
        split_stats.total_images = len(image_files)

        all_img_stats: List[ImageStatistics] = []
        all_bboxes: List[BoundingBoxStatistics] = []

        for img_path in image_files:
            img_stat = self.collect_image_statistics(img_path)
            if img_stat is None:
                continue

            label_path = labels_dir / f"{img_path.stem}{LABEL_EXTENSION}"
            bboxes = self.load_labels(label_path)
            img_stat.bbox_count = len(bboxes)

            all_img_stats.append(img_stat)
            all_bboxes.extend(bboxes)
            if label_path.exists():
                split_stats.total_labels += 1

        split_stats.total_annotations = len(all_bboxes)

        # Tính toán chi tiết phân phối chiều rộng/cao ảnh
        if all_img_stats:
            widths = [s.width for s in all_img_stats]
            heights = [s.height for s in all_img_stats]

            split_stats.average_width = _mean(widths)
            split_stats.median_width = _median(widths)
            split_stats.std_width = _stdev(widths)
            split_stats.min_width = min(widths)
            split_stats.max_width = max(widths)

            split_stats.average_height = _mean(heights)
            split_stats.median_height = _median(heights)
            split_stats.std_height = _stdev(heights)
            split_stats.min_height = min(heights)
            split_stats.max_height = max(heights)

        # Tính toán chi tiết bounding box
        if all_bboxes:
            areas = [b.area for b in all_bboxes]
            split_stats.average_bbox_area = _mean(areas)
            split_stats.median_bbox_area = _median(areas)
            split_stats.largest_bbox = max(all_bboxes, key=lambda b: b.area)
            split_stats.smallest_bbox = min(all_bboxes, key=lambda b: b.area)

            # Phân bố class
            class_counts = Counter([b.class_id for b in all_bboxes])
            split_stats.class_distribution = dict(class_counts)

            # Phân bố Aspect Ratio
            aspect_dist = {"very_wide (>3)": 0, "wide (1.5-3)": 0, "square (0.67-1.5)": 0, "tall (0.33-0.67)": 0, "very_tall (<0.33)": 0}
            for b in all_bboxes:
                ratio = b.width / b.height if b.height > 0 else 999.0
                if ratio > 3.0: aspect_dist["very_wide (>3)"] += 1
                elif ratio > 1.5: aspect_dist["wide (1.5-3)"] += 1
                elif ratio > 0.67: aspect_dist["square (0.67-1.5)"] += 1
                elif ratio > 0.33: aspect_dist["tall (0.33-0.67)"] += 1
                else: aspect_dist["very_tall (<0.33)"] += 1
            split_stats.aspect_ratio_distribution = aspect_dist

        split_stats.processing_time = time.time() - start_time
        return split_stats

    def collect_dataset(self, splits: Optional[List[str]] = None) -> DatasetStatistics:
        if splits is None:
            splits = ["train", "val", "test"]
        for name in splits:
            stat = self.collect_split(name)
            if name == "train": self.statistics.train = stat
            elif name == "val": self.statistics.val = stat
            elif name == "test": self.statistics.test = stat
            self.statistics.total_images += stat.total_images
            self.statistics.total_labels += stat.total_labels
            self.statistics.total_annotations += stat.total_annotations
        return self.statistics


# ===========================================================
# ChartGenerator — Sinh biểu đồ trực quan hóa
# ===========================================================

class ChartGenerator:
    """Sinh các đồ thị phân tích phân phối thuộc tính tập dữ liệu."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, stats: DatasetStatistics, collector_paths: ProjectPaths) -> None:
        # Gom dữ liệu toàn cục để vẽ biểu đồ tổng hợp
        all_widths, all_heights, all_areas = [], [], []
        class_counter: Counter = Counter()
        aspect_dist = {"very_wide (>3)": 0, "wide (1.5-3)": 0, "square (0.67-1.5)": 0, "tall (0.33-0.67)": 0, "very_tall (<0.33)": 0}

        for s in [stats.train, stats.val, stats.test]:
            if s is None or s.total_images == 0: continue
            images_dir = getattr(collector_paths, f"{s.split_name}_images")
            labels_dir = getattr(collector_paths, f"{s.split_name}_labels")

            for ext in IMAGE_EXTENSIONS:
                for img_path in images_dir.glob(f"*{ext}"):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        all_heights.append(img.shape[0])
                        all_widths.append(img.shape[1])
                    lbl_path = labels_dir / f"{img_path.stem}{LABEL_EXTENSION}"
                    if lbl_path.exists():
                        for box in DatasetStatisticsCollector(collector_paths).load_labels(lbl_path):
                            all_areas.append(box.area)
                            class_counter[box.class_id] += 1
                            ratio = box.width / box.height if box.height > 0 else 999.0
                            if ratio > 3.0: aspect_dist["very_wide (>3)"] += 1
                            elif ratio > 1.5: aspect_dist["wide (1.5-3)"] += 1
                            elif ratio > 0.67: aspect_dist["square (0.67-1.5)"] += 1
                            elif ratio > 0.33: aspect_dist["tall (0.33-0.67)"] += 1
                            else: aspect_dist["very_tall (<0.33)"] += 1

        if all_widths:
            self._plot_hist(all_widths, "Phân bố Chiều rộng Ảnh", "Chiều rộng (pixel)", "image_width.png", "#2196F3")
            self._plot_hist(all_heights, "Phân bố Chiều cao Ảnh", "Chiều cao (pixel)", "image_height.png", "#4CAF50")
        if all_areas:
            self._plot_hist(all_areas, "Phân bố Diện tích Bounding Box", "Diện tích (chuẩn hoá)", "bbox_area.png", "#9C27B0")
            self._plot_bars(class_counter, "Phân bố Class Toàn bộ Dataset", "class_distribution.png")
            self._plot_h_bars(aspect_dist, "Phân bố Aspect Ratio Bounding Box", "aspect_ratio.png")

    def _plot_hist(self, data: List[Any], title: str, xlabel: str, filename: str, color: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(_mean(data), color="#FF9800", linestyle="--", linewidth=2, label=f"Mean: {_mean(data):.1f}")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Số lượng")
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=120)
        plt.close(fig)

    def _plot_bars(self, counter: Counter, title: str, filename: str) -> None:
        if not counter: return
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = sorted(counter.keys())
        labels = [CLASS_NAMES.get(c, f"Class {c}") for c in classes]
        counts = [counter[c] for c in classes]
        bars = ax.bar(labels, counts, color="#2196F3", edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=120)
        plt.close(fig)

    def _plot_h_bars(self, dist: Dict[str, int], title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels, counts = list(dist.keys()), list(dist.values())
        bars = ax.barh(labels, counts, color="#FF9800", edgecolor="white")
        for bar in bars:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2.0, f" {int(bar.get_width()):,}", ha="left", va="center", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=120)
        plt.close(fig)


# ===========================================================
# ReportWriter — Xuất báo cáo Text & Bổ sung Xuất báo cáo Excel
# ===========================================================

class ReportWriter:
    """Xử lý định dạng đầu ra dữ liệu báo cáo sang tệp văn bản và bảng tính Excel."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, stats: DatasetStatistics) -> Path:
        """Xuất file thống kê dạng tường thuật .txt."""
        lines = ["=" * 65, "  VISIONTEXTREADER ADVANCED DATASET REPORT", "=" * 65, ""]

        for s in [stats.train, stats.val, stats.test]:
            if s is None or s.total_images == 0: continue
            lines.extend([
                f"  SPLIT: {s.split_name.upper()}",
                "-" * 65,
                f"    Tổng số ảnh:        {s.total_images:>10,}",
                f"    Tổng số file nhãn:   {s.total_labels:>10,}",
                f"    Tổng số BBox:       {s.total_annotations:>10,}",
                f"    Chiều rộng ảnh (px): Mean={s.average_width:.1f}, Median={s.median_width:.1f}, Std={s.std_width:.1f}, Min={s.min_width}, Max={s.max_width}",
                f"    Chiều cao ảnh (px):  Mean={s.average_height:.1f}, Median={s.median_height:.1f}, Std={s.std_height:.1f}, Min={s.min_height}, Max={s.max_height}",
                f"    Diện tích BBox:      Mean={s.average_bbox_area:.5f}, Median={s.median_bbox_area:.5f}",
                f"    Thời gian tính toán: {s.processing_time:.2f}s",
                ""
            ])

        lines.extend(["=" * 65, "  GRAND TOTAL DATASET SUMMARY", "=" * 65,
                      f"    Tổng số hình ảnh:    {stats.total_images:,}",
                      f"    Tổng số nhãn đính kèm: {stats.total_labels:,}",
                      f"    Tổng số đối tượng:    {stats.total_annotations:,}"])

        out_path = self.output_dir / "statistics_report.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def export_excel(self, stats: DatasetStatistics) -> Path:
        """Xuất báo cáo thống kê chuyên nghiệp sang tệp Excel (.xlsx)."""
        excel_path = self.output_dir / "statistics.xlsx"

        # 1. Bảng dữ liệu tổng hợp các split
        split_rows = []
        for s in [stats.train, stats.val, stats.test]:
            if s is None or s.total_images == 0: continue
            split_rows.append({
                "Split": s.split_name.upper(), "Images": s.total_images, "Labels": s.total_labels, "Annotations": s.total_annotations,
                "Mean Width": round(s.average_width, 1), "Median Width": round(s.median_width, 1), "Std Width": round(s.std_width, 1), "Min Width": s.min_width, "Max Width": s.max_width,
                "Mean Height": round(s.average_height, 1), "Median Height": round(s.median_height, 1), "Std Height": round(s.std_height, 1), "Min Height": s.min_height, "Max Height": s.max_height,
                "Mean BBox Area": round(s.average_bbox_area, 5), "Median BBox Area": round(s.median_bbox_area, 5)
            })
        df_splits = pd.DataFrame(split_rows)

        # 2. Bảng phân bố Class
        class_rows = []
        for s in [stats.train, stats.val, stats.test]:
            if s is None: continue
            for cid, count in s.class_distribution.items():
                class_rows.append({
                    "Split": s.split_name.upper(), "Class ID": cid, "Class Name": CLASS_NAMES.get(cid, f"Class {cid}"), "Count": count
                })
        df_classes = pd.DataFrame(class_rows)

        # Sử dụng ExcelWriter để ghi nhiều sheet
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_splits.to_excel(writer, sheet_name="Dataset Summary", index=False)
            df_classes.to_excel(writer, sheet_name="Class Distribution", index=False)

        logger.info("Đã xuất Excel báo cáo thành công tại: %s", excel_path)
        return excel_path


# ===========================================================
# Hàm hiển thị CLI Summary
# ===========================================================

def print_summary(stats: DatasetStatistics) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  VISIONTEXTREADER DATASET STATISTICS (UPDATED)\n{sep}")
    header = f"  {'Metric':<25s} {'Train':>13s} {'Val':>13s} {'Test':>13s}"
    print(header)
    print(f"  {'-' * 66}")

    def _get_metric(split: Optional[SplitStatistics], attr: str, fmt: str = ",.1f") -> str:
        if split is None or getattr(split, "total_images", 0) == 0: return "N/A"
        val = getattr(split, attr, 0)
        return f"{val:{fmt}}"

    t, v, te = stats.train, stats.val, stats.test
    print(f"  {'Total Images':<25s} {_get_metric(t, 'total_images', ',d'):>13s} {_get_metric(v, 'total_images', ',d'):>13s} {_get_metric(te, 'total_images', ',d'):>13s}")
    print(f"  {'Total Labels':<25s} {_get_metric(t, 'total_labels', ',d'):>13s} {_get_metric(v, 'total_labels', ',d'):>13s} {_get_metric(te, 'total_labels', ',d'):>13s}")
    print(f"  {'Total Annotations':<25s} {_get_metric(t, 'total_annotations', ',d'):>13s} {_get_metric(v, 'total_annotations', ',d'):>13s} {_get_metric(te, 'total_annotations', ',d'):>13s}")
    print(f"  {'-' * 66}")
    print(f"  {'Mean / Median Width':<25s} {(_get_metric(t,'average_width')+ ' / ' +_get_metric(t,'median_width')):>13s} {(_get_metric(v,'average_width')+ ' / ' +_get_metric(v,'median_width')):>13s} {(_get_metric(te,'average_width')+ ' / ' +_get_metric(te,'median_width')):>13s}")
    print(f"  {'Std Width (px)':<25s} {_get_metric(t, 'std_width'):>13s} {_get_metric(v, 'std_width'):>13s} {_get_metric(te, 'std_width'):>13s}")
    print(f"  {'Min / Max Width':<25s} {( _get_metric(t,'min_width',',d')+'-'+_get_metric(t,'max_width',',d')):>13s} {(_get_metric(v,'min_width',',d')+'-'+_get_metric(v,'max_width',',d')):>13s} {(_get_metric(te,'min_width',',d')+'-'+_get_metric(te,'max_width',',d')):>13s}")
    print(f"  {'-' * 66}")
    print(f"  {'Mean / Median Height':<25s} {(_get_metric(t,'average_height')+ ' / ' +_get_metric(t,'median_height')):>13s} {(_get_metric(v,'average_height')+ ' / ' +_get_metric(v,'median_height')):>13s} {(_get_metric(te,'average_height')+ ' / ' +_get_metric(te,'median_height')):>13s}")
    print(f"  {'Std Height (px)':<25s} {_get_metric(t, 'std_height'):>13s} {_get_metric(v, 'std_height'):>13s} {_get_metric(te, 'std_height'):>13s}")
    print(f"  {'Min / Max Height':<25s} {(_get_metric(t,'min_height',',d')+'-'+_get_metric(t,'max_height',',d')):>13s} {(_get_metric(v,'min_height',',d')+'-'+_get_metric(v,'max_height',',d')):>13s} {(_get_metric(te,'min_height',',d')+'-'+_get_metric(te,'max_height',',d')):>13s}")
    print(sep)


# ===========================================================
# Điểm vào chính (CLI Engine)
# ===========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Thống kê nâng cao tập dữ liệu VisionTextReader.")
    parser.add_argument("--split", "-s", type=str, choices=["train", "val", "test"])
    parser.add_argument("--output", "-o", type=str, default="outputs/statistics")
    parser.add_argument("--excel", "-e", action="store_true", help="Bắt buộc xuất file excel.")
    args = parser.parse_args()

    paths = get_project_paths()
    output_dir = Path(args.output) if Path(args.output).is_absolute() else paths.root / args.output

    splits_to_process = [args.split] if args.split else None

    logger.info("Bắt đầu phân tích dữ liệu...")
    collector = DatasetStatisticsCollector(paths)
    dataset_stats = collector.collect_dataset(splits_to_process)

    # 1. In báo cáo tóm tắt cấu trúc mới ra màn hình CLI
    print_summary(dataset_stats)

    # 2. Sinh toàn bộ hệ thống biểu đồ lưu trữ thành tệp hình ảnh
    chart_gen = ChartGenerator(output_dir)
    chart_gen.generate_all(dataset_stats, paths)

    # 3. Xuất file text (.txt) và file bảng tính Excel (.xlsx)
    writer = ReportWriter(output_dir)
    writer.write_report(dataset_stats)
    writer.export_excel(dataset_stats)

    logger.info("Hoàn tất quy trình xử lý thống kê nâng cao.")


if __name__ == "__main__":
    main()
