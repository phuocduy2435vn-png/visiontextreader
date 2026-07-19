"""
visualize_dataset.py — Trực quan hoá dataset YOLO cho VisionTextReader.

Đọc dataset từ datasets/processed/, vẽ bounding box lên ảnh,
xuất ảnh minh họa ra thư mục outputs/ với tên file đánh số thứ tự.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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
logger = logging.getLogger("visiontextreader.visualize")


# ===========================================================
# Bảng màu cố định cho từng class
# ===========================================================

# Bảng màu BGR — mỗi class một màu khác nhau
# Đủ 20 màu để mở rộng class sau này
_CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),      # Xanh lá
    (255, 0, 0),      # Xanh dương
    (0, 0, 255),      # Đỏ
    (255, 255, 0),    # Cyan
    (0, 255, 255),    # Vàng
    (255, 0, 255),    # Magenta
    (128, 255, 0),    # Xanh lá nhạt
    (0, 128, 255),    # Cam
    (255, 128, 0),    # Xanh dương nhạt
    (128, 0, 255),    # Tím
    (255, 128, 128),  # Hồng
    (128, 255, 128),  # Mint
    (128, 128, 255),  # Lavender
    (255, 255, 128),  # Kem
    (255, 128, 255),  # Phấn hồng
    (0, 255, 128),    # Ngọc
    (128, 255, 255),  # Sky blue
    (255, 0, 128),    # Crimson
    (64, 255, 64),    # Xanh cỏ
    (255, 64, 64),    # Đỏ cam
]


# ===========================================================
# Cấu trúc dữ liệu
# ===========================================================

@dataclass
class BBox:
    """Một bounding box YOLO đã chuyển sang tọa độ pixel."""
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    index: int = 0

    @property
    def width(self) -> int:
        """Chiều rộng bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Chiều cao bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Diện tích bounding box."""
        return self.width * self.height


@dataclass
class VisualizationReport:
    """Báo cáo kết quả trực quan hoá."""
    split_name: str
    total_images: int = 0
    processed_images: int = 0
    error_images: int = 0
    total_bboxes: int = 0
    elapsed_seconds: float = 0.0
    output_dir: str = ""

    def summary(self) -> str:
        """Trả về chuỗi tóm tắt báo cáo."""
        lines = [
            f"  Split '{self.split_name}':",
            f"    Ảnh đã xử lý:    {self.processed_images}/{self.total_images}",
            f"    Ảnh lỗi:          {self.error_images}",
            f"    Tổng Bounding Box: {self.total_bboxes}",
            f"    Thời gian:         {self.elapsed_seconds:.2f}s",
            f"    Thư mục output:    {self.output_dir}",
        ]
        return "\n".join(lines)


# ===========================================================
# BoundingBoxDrawer — vẽ bounding box và label lên ảnh
# ===========================================================

class BoundingBoxDrawer:
    """Vẽ bounding box, label, và thông tin thống kê lên ảnh OpenCV.

    Hỗ trợ:
        - Mỗi class một màu riêng
        - Nền semi-transparent cho label
        - Font dễ đọc, cỡ chữ phù hợp
        - Đánh số thứ tự bounding box
        - Hiển thị confidence
    """

    # Độ dày nét vẽ bounding box (pixel)
    THICKNESS: int = 2

    # Kích thước font chữ
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 1

    # Khoảng cách padding cho label nền
    LABEL_PADDING_X: int = 4
    LABEL_PADDING_Y: int = 2

    # Độ trong suốt của nền label (0 = trong suốt hoàn toàn, 1 = opaq)
    LABEL_ALPHA: float = 0.6

    def __init__(self, class_names: Optional[Dict[int, str]] = None) -> None:
        """Khởi tạo drawer.

        Args:
            class_names: Dict ánh xạ class_id → tên class. Nếu None, dùng từ config.
        """
        self.class_names = class_names if class_names is not None else CLASS_NAMES

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Lấy màu BGR cho class_id.

        Args:
            class_id: ID của class.

        Returns:
            Tuple (B, G, R) màu sắc.
        """
        return _CLASS_COLORS[class_id % len(_CLASS_COLORS)]

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: BBox,
    ) -> np.ndarray:
        """Vẽ một bounding box lên ảnh.

        Args:
            image: Ảnh OpenCV (BGR, uint8).
            bbox: Bounding box cần vẽ.

        Returns:
            Ảnh đã vẽ (tham chiếu đến ảnh gốc).
        """
        color = self.get_color(bbox.class_id)

        # Vẽ khung bounding box
        cv2.rectangle(
            image,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            self.THICKNESS,
        )

        # Tạo label text
        class_name = self.class_names.get(bbox.class_id, f"class_{bbox.class_id}")
        label_text = f"#{bbox.index} {class_name}"
        if bbox.confidence < 1.0:
            label_text += f" {bbox.confidence:.2f}"

        # Vẽ label với nền
        self._draw_label_with_bg(image, label_text, (bbox.x1, bbox.y1), color)

        return image

    def _draw_label_with_bg(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Vẽ text với nền semi-transparent.

        Args:
            image: Ảnh OpenCV.
            text: Chuỗi cần vẽ.
            position: Tọa độ (x, y) góc trên bên trái.
            color: Màu BGR của text và viền nền.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, self.FONT_SCALE, self.FONT_THICKNESS,
        )

        x, y = position

        # Tọa độ góc trên bên trái và dưới bên phải của nền
        bg_x1 = x
        bg_y1 = y - text_h - baseline - self.LABEL_PADDING_Y * 2
        bg_x2 = x + text_w + self.LABEL_PADDING_X * 2
        bg_y2 = y + self.LABEL_PADDING_Y

        # Đảm bảo tọa độ không âm
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)

        # Vẽ nền semi-transparent
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, self.LABEL_ALPHA, image, 1 - self.LABEL_ALPHA, 0, image)

        # Vẽ text lên nền
        text_x = x + self.LABEL_PADDING_X
        text_y = y - self.LABEL_PADDING_Y
        cv2.putText(
            image, text, (text_x, text_y),
            font, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS, cv2.LINE_AA,
        )

    def draw_statistics(
        self,
        image: np.ndarray,
        split_name: str,
        image_name: str,
        image_index: int,
        img_width: int,
        img_height: int,
        bbox_count: int,
    ) -> np.ndarray:
        """Vẽ thông tin thống kê ở góc trên cùng của ảnh.

        Hiển thị: tên split, tên file, kích thước ảnh, số lượng bbox.

        Args:
            image: Ảnh OpenCV.
            split_name: Tên split (train/val/test).
            image_name: Tên file ảnh.
            image_index: Số thứ tự ảnh.
            img_width: Chiều rộng ảnh.
            img_height: Chiều cao ảnh.
            bbox_count: Số lượng bounding box.

        Returns:
            Ảnh đã vẽ thông tin thống kê.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 22
        y_offset = 20

        # Dòng 1: Split và số thứ tự
        info_line1 = f"[{split_name.upper()}] #{image_index:06d}"
        self._draw_label_with_bg(image, info_line1, (5, y_offset), (64, 64, 64))
        y_offset += line_height

        # Dòng 2: Tên file
        info_line2 = image_name
        if len(info_line2) > 50:
            info_line2 = "..." + info_line2[-47:]
        self._draw_label_with_bg(image, info_line2, (5, y_offset), (64, 64, 64))
        y_offset += line_height

        # Dòng 3: Kích thước và số bbox
        info_line3 = f"{img_width}x{img_height} | {bbox_count} bbox"
        self._draw_label_with_bg(image, info_line3, (5, y_offset), (64, 64, 64))

        return image


# ===========================================================
# DatasetVisualizer — điều phối trực quan hoá toàn bộ dataset
# ===========================================================

class DatasetVisualizer:
    """Đọc dataset YOLO, trực quan hoá và xuất ảnh minh họa.

    Chịu trách nhiệm:
        - Quét ảnh và label trong dataset
        - Random chọn N ảnh để trực quan hoá
        - Vẽ bounding box bằng BoundingBoxDrawer
        - Xuất ảnh đã vẽ ra thư mục outputs/
    """

    def __init__(
        self,
        paths: ProjectPaths,
        output_root: Path,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """Khởi tạo visualizer.

        Args:
            paths: Cấu trúc thư mục dự án.
            output_root: Thư mục gốc chứa output (outputs/).
            class_names: Dict class_id → tên. Nếu None, dùng từ config.
        """
        self.paths = paths
        self.output_root = output_root
        self.drawer = BoundingBoxDrawer(class_names)
        self.reports: List[VisualizationReport] = []

    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Đọc ảnh từ đường dẫn bằng OpenCV.

        Args:
            image_path: Đường dẫn file ảnh.

        Returns:
            Ảnh numpy array (BGR) hoặc None nếu lỗi.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Không thể đọc ảnh: %s", image_path.name)
                return None
            return image
        except Exception as exc:
            logger.warning("Lỗi khi đọc ảnh %s: %s", image_path.name, exc)
            return None

    def load_labels(
        self,
        label_path: Path,
        img_width: int,
        img_height: int,
    ) -> List[BBox]:
        """Đọc file label YOLO và chuyển sang tọa độ pixel.

        Định dạng YOLO: class_id cx cy width height (tất cả float, [0,1]).

        Args:
            label_path: Đường dẫn file label.
            img_width: Chiều rộng ảnh (để chuyển tọa độ).
            img_height: Chiều cao ảnh.

        Returns:
            Danh sách BBox đã chuyển sang tọa độ pixel.
        """
        boxes: List[BBox] = []

        if not label_path.exists():
            return boxes

        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except IOError as exc:
            logger.warning("Không thể đọc label %s: %s", label_path.name, exc)
            return boxes

        if not content:
            return boxes

        for line_idx, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                logger.warning(
                    "Label %s dòng %d: định dạng không hợp lệ (cần 5 giá trị, có %d)",
                    label_path.name, line_idx, len(parts),
                )
                continue

            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except (ValueError, IndexError) as exc:
                logger.warning(
                    "Label %s dòng %d: lỗi parse — %s",
                    label_path.name, line_idx, exc,
                )
                continue

            # Kiểm tra giá trị nằm trong khoảng [0, 1]
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                logger.warning(
                    "Label %s dòng %d: giá trị ngoài khoảng [0,1] — cx=%.3f cy=%.3f w=%.3f h=%.3f",
                    label_path.name, line_idx, cx, cy, w, h,
                )
                continue

            # Chuyển từ YOLO centre format sang pixel corners
            abs_x1 = int((cx - w / 2.0) * img_width)
            abs_y1 = int((cy - h / 2.0) * img_height)
            abs_x2 = int((cx + w / 2.0) * img_width)
            abs_y2 = int((cy + h / 2.0) * img_height)

            # Clamp vào phạm vi ảnh
            abs_x1 = max(0, min(abs_x1, img_width - 1))
            abs_y1 = max(0, min(abs_y1, img_height - 1))
            abs_x2 = max(0, min(abs_x2, img_width - 1))
            abs_y2 = max(0, min(abs_y2, img_height - 1))

            # Đảm bảo bbox có diện tích > 0
            if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                logger.warning(
                    "Label %s dòng %d: bbox có diện tích 0 — bỏ qua",
                    label_path.name, line_idx,
                )
                continue

            boxes.append(BBox(
                class_id=class_id,
                x1=abs_x1,
                y1=abs_y1,
                x2=abs_x2,
                y2=abs_y2,
                confidence=1.0,
                index=0,
            ))

        return boxes

    def save_image(
        self,
        image: np.ndarray,
        output_path: Path,
    ) -> bool:
        """Lưu ảnh đã vẽ ra file.

        Args:
            image: Ảnh OpenCV cần lưu.
            output_path: Đường dẫn file output.

        Returns:
            True nếu lưu thành công, False nếu lỗi.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), image)
            if not success:
                logger.warning("cv2.imwrite trả về False cho %s", output_path)
                return False
            return True
        except Exception as exc:
            logger.warning("Lỗi khi lưu ảnh %s: %s", output_path.name, exc)
            return False

    def process_split(
        self,
        split_name: str,
        count: Optional[int] = None,
    ) -> VisualizationReport:
        """Xử lý trực quan hoá một split.

        Quét tất cả ảnh, random chọn count ảnh, vẽ bounding box
        và lưu ra thư mục outputs/<split>/.

        Args:
            split_name: Tên split ('train', 'val', 'test').
            count: Số ảnh cần trực quan hoá. None = tất cả.

        Returns:
            VisualizationReport chứa kết quả xử lý.
        """
        report = VisualizationReport(split_name=split_name)

        # Lấy đường dẫn thư mục ảnh và label
        images_dir: Path = getattr(self.paths, f"{split_name}_images")
        labels_dir: Path = getattr(self.paths, f"{split_name}_labels")

        # Kiểm tra thư mục tồn tại
        if not images_dir.exists():
            logger.warning("Thư mục ảnh không tồn tại: %s", images_dir)
            return report

        # Thu thập tất cả file ảnh
        image_files: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(images_dir.glob(f"*{ext}"))
        image_files = sorted(set(image_files))

        report.total_images = len(image_files)
        logger.info("Split '%s': tìm thấy %d ảnh", split_name, len(image_files))

        # Random chọn ảnh nếu cần
        if count is not None and count < len(image_files):
            random.shuffle(image_files)
            image_files = image_files[:count]
            logger.info("Split '%s': đã random chọn %d ảnh", split_name, count)

        # Tạo thư mục output
        output_dir = self.output_root / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        report.output_dir = str(output_dir)

        start_time = time.time()

        # Xử lý từng ảnh
        for idx, img_path in enumerate(image_files, start=1):
            try:
                # Đọc ảnh
                image = self.load_image(img_path)
                if image is None:
                    report.error_images += 1
                    continue

                img_height, img_width = image.shape[:2]

                # Đọc label
                label_path = labels_dir / f"{img_path.stem}{LABEL_EXTENSION}"
                boxes = self.load_labels(label_path, img_width, img_height)

                # Gán số thứ tự cho từng bbox
                for box_idx, box in enumerate(boxes, start=1):
                    box.index = box_idx

                # Tạo bản sao để vẽ (giữ nguyên ảnh gốc)
                vis_image = image.copy()

                # Vẽ tất cả bounding box
                for box in boxes:
                    vis_image = self.drawer.draw_bbox(vis_image, box)

                # Vẽ thông tin thống kê ở góc trên
                vis_image = self.drawer.draw_statistics(
                    vis_image,
                    split_name=split_name,
                    image_name=img_path.name,
                    image_index=idx,
                    img_width=img_width,
                    img_height=img_height,
                    bbox_count=len(boxes),
                )

                # Tên file output: 000001_visualized.jpg
                output_filename = f"{idx:06d}_visualized.jpg"
                output_path = output_dir / output_filename

                # Lưu ảnh
                if self.save_image(vis_image, output_path):
                    report.processed_images += 1
                    report.total_bboxes += len(boxes)
                else:
                    report.error_images += 1

                del image, vis_image

            except Exception as exc:
                logger.warning("Lỗi khi xử lý ảnh %s: %s", img_path.name, exc)
                report.error_images += 1

        report.elapsed_seconds = time.time() - start_time

        logger.info(
            "Split '%s': hoàn tất — %d/%d ảnh, %d bbox, %.2fs",
            split_name,
            report.processed_images,
            report.total_images,
            report.total_bboxes,
            report.elapsed_seconds,
        )

        return report

    def process_dataset(
        self,
        splits: Optional[List[str]] = None,
        count: Optional[int] = None,
    ) -> List[VisualizationReport]:
        """Xử lý trực quan hoá toàn bộ dataset.

        Args:
            splits: Danh sách split cần xử lý. None = tất cả.
            count: Số ảnh mỗi split cần trực quan hoá. None = tất cả.

        Returns:
            Danh sách VisualizationReport cho từng split.
        """
        if splits is None:
            splits = ["train", "val", "test"]

        self.reports = []

        for split_name in splits:
            logger.info("=" * 50)
            logger.info("Bắt đầu trực quan hoá split '%s'", split_name)
            logger.info("=" * 50)

            report = self.process_split(split_name, count)
            self.reports.append(report)
            logger.info("\n%s", report.summary())

        return self.reports


# ===========================================================
# In báo cáo tóm tắt
# ===========================================================

def print_summary(reports: List[VisualizationReport]) -> None:
    """In báo cáo tổng hợp ra màn hình.

    Hiển thị bảng tóm tắt với số ảnh xử lý, lỗi, bbox, và thời gian.

    Args:
        reports: Danh sách VisualizationReport.
    """
    sep = "=" * 60

    print(f"\n{sep}")
    print("  VISIONTEXTREADER — BÁO CÁO TRỰC QUAN HOÁ")
    print(sep)

    total_processed = 0
    total_errors = 0
    total_bboxes = 0
    total_time = 0.0

    for report in reports:
        print(report.summary())
        print()
        total_processed += report.processed_images
        total_errors += report.error_images
        total_bboxes += report.total_bboxes
        total_time += report.elapsed_seconds

    print("-" * 60)
    print(f"  TỔNG CỘNG:")
    print(f"    Ảnh đã xử lý:    {total_processed}")
    print(f"    Ảnh lỗi:          {total_errors}")
    print(f"    Tổng Bounding Box: {total_bboxes}")
    print(f"    Tổng thời gian:    {total_time:.2f}s")
    print(sep)
    print()


# ===========================================================
# Điểm vào chính
# ===========================================================

def main() -> None:
    """Điểm vào chính — CLI entry point.

    Hỗ trợ các tham số:
        --split: Chỉ xử lý một split cụ thể (train/val/test).
        --count: Số ảnh cần trực quan hoá mỗi split (mặc định: 100).
        --output: Thư mục gốc xuất ảnh (mặc định: outputs/).
    """
    parser = argparse.ArgumentParser(
        description="Trực quan hoá dataset YOLO cho VisionTextReader.",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["train", "val", "test"],
        help="Chỉ xử lý split cụ thể (mặc định: tất cả).",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=100,
        help="Số ảnh cần trực quan hoá mỗi split (mặc định: 100). Đặt 0 để xử lý tất cả.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Thư mục gốc xuất ảnh (mặc định: outputs/).",
    )
    args = parser.parse_args()

    # Lấy đường dẫn dự án
    paths = get_project_paths()

    # Xác định thư mục output
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = paths.root / output_root

    # Xác định splits cần xử lý
    splits_to_process: Optional[List[str]] = None
    if args.split:
        splits_to_process = [args.split]

    # Xác định số ảnh (0 = tất cả)
    count = args.count if args.count > 0 else None

    # Khởi tạo visualizer
    visualizer = DatasetVisualizer(paths, output_root)

    # Xử lý dataset
    logger.info("Bắt đầu trực quan hoá dataset...")
    reports = visualizer.process_dataset(splits_to_process, count)

    # In báo cáo tóm tắt
    print_summary(reports)

    logger.info("Hoàn tất trực quan hoá dataset.")


if __name__ == "__main__":
    main()
