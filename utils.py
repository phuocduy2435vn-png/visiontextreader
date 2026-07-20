"""
utils.py — Shared utilities for VisionTextReader.

Image I/O, bbox conversion, and visualization helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_EXTENSIONS

logger = logging.getLogger("visiontextreader.utils")


# ===========================================================
# Image I/O
# ===========================================================

def load_image(image_path: str | Path) -> np.ndarray | None:
    """Load an image from disk. Returns BGR numpy array or None."""
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Cannot read image: %s", image_path)
        return image
    except Exception as exc:
        logger.warning("Error loading image %s: %s", image_path, exc)
        return None


def save_image(image: np.ndarray, output_path: str | Path, quality: int = 95) -> bool:
    """Save an image to disk. Returns True if saved successfully."""
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


# ===========================================================
# BBox Conversion
# ===========================================================

def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w - 1, int((cx + w / 2) * img_w))
    y2 = min(img_h - 1, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


# ===========================================================
# Visualization
# ===========================================================

COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def draw_detections(
    image: np.ndarray,
    boxes: list[list[int]],
    scores: list[float],
    class_names: dict[int, str],
    class_ids: list[int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes with labels and confidence on image."""
    vis = image.copy()
    if class_ids is None:
        class_ids = [0] * len(boxes)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls_id % len(COLORS)]
        label = f"{class_names.get(cls_id, f'class_{cls_id}')}: {score:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )
        cv2.rectangle(vis, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis
