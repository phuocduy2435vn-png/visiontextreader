"""
utils.py — Shared utilities for VisionTextReader detection pipeline.

Provides image I/O, BBox conversion, visualization, and common helpers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("visiontextreader.utils")


# ===========================================================
# Constants
# ===========================================================

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
LABEL_EXTENSION: str = ".txt"

# Color palette for visualization (BGR)
COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


# ===========================================================
# Image I/O
# ===========================================================

def load_image(image_path: str | Path) -> Optional[np.ndarray]:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        numpy array in BGR format, or None if loading fails.
    """
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Cannot read image: %s", image_path)
            return None
        return image
    except Exception as exc:
        logger.warning("Error loading image %s: %s", image_path, exc)
        return None


def save_image(image: np.ndarray, output_path: str | Path, quality: int = 95) -> bool:
    """Save an image to disk.

    Args:
        image: numpy array in BGR format.
        output_path: Destination path.
        quality: JPEG quality (1-100).

    Returns:
        True if saved successfully.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            params = []
        success = cv2.imwrite(str(output_path), image, params)
        if not success:
            logger.warning("Failed to save image: %s", output_path)
        return success
    except Exception as exc:
        logger.warning("Error saving image %s: %s", output_path, exc)
        return False


def get_image_info(image_path: str | Path) -> Optional[Dict[str, Any]]:
    """Get image metadata without loading the full image.

    Returns:
        Dict with width, height, channels, file_size, or None.
    """
    try:
        from typing import Any
        path = Path(image_path)
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        return {
            "width": w, "height": h, "channels": channels,
            "file_size": path.stat().st_size,
            "dtype": str(img.dtype),
        }
    except Exception:
        return None


# ===========================================================
# BBox Conversion
# ===========================================================

def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int,
                  img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert absolute (x1, y1, x2, y2) to YOLO normalized (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    return x1, y1, x2 - x1, y2 - y1


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: List[List[int]], scores: List[float], iou_threshold: float = 0.5) -> List[int]:
    """Non-Maximum Suppression.

    Args:
        boxes: List of [x1, y1, x2, y2].
        scores: Confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of kept boxes.
    """
    if not boxes:
        return []
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ===========================================================
# Visualization
# ===========================================================

def draw_detections(
    image: np.ndarray,
    boxes: List[List[int]],
    scores: List[float],
    class_names: Dict[int, str],
    class_ids: Optional[List[int]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes with labels and confidence on image.

    Args:
        image: Input image (BGR).
        boxes: List of [x1, y1, x2, y2].
        scores: Confidence scores.
        class_names: Mapping of class_id to name.
        class_ids: Class IDs (default: all 0).
        thickness: Box line thickness.
        font_scale: Font size.

    Returns:
        Annotated image.
    """
    vis = image.copy()
    if class_ids is None:
        class_ids = [0] * len(boxes)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls_id % len(COLORS)]
        label = f"{class_names.get(cls_id, f'class_{cls_id}')}: {score:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Background for text
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def draw_boxes_simple(
    image: np.ndarray,
    boxes: List[List[int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw simple bounding boxes without labels.

    Args:
        image: Input image (BGR).
        boxes: List of [x1, y1, x2, y2].
        color: Box color (BGR).
        thickness: Line thickness.

    Returns:
        Annotated image.
    """
    vis = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis
