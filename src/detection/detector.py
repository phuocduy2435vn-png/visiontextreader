"""
detector.py — Text region detection using YOLO model.

Provides TextDetector for detecting text regions in images.
Returns plain Python objects (no tensors, no YOLO objects).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("visiontextreader.detection")


class TextDetector:
    """Text region detector using trained YOLO model.

    Wraps YOLO inference and returns plain Python dicts.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.class_names = class_names or {0: "text"}
        self._model = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(
            "TextDetector initialized — model=%s, conf=%.2f, iou=%.2f, imgsz=%d",
            self.model_path.name, self.conf_threshold, self.iou_threshold, self.imgsz,
        )

    def load_model(self) -> None:
        """Load the YOLO model (called automatically on first detect)."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required: pip install ultralytics")

        logger.info("Loading model: %s", self.model_path)
        self._model = YOLO(str(self.model_path))

        if hasattr(self._model, "names") and self._model.names:
            self.class_names = self._model.names

    def detect(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[dict]:
        """Detect text regions in an image.

        Returns:
            List of dicts with keys: bbox, confidence, class_id, class_name.
        """
        if self._model is None:
            self.load_model()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        img = self._load_image(image)
        if img is None:
            return []

        start_time = time.time()
        try:
            results = self._model.predict(
                source=img, conf=conf, iou=iou, imgsz=self.imgsz, verbose=False,
            )
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return []

        elapsed = time.time() - start_time
        detections = self._parse_results(results)

        logger.debug(
            "Detected %d regions in %.3fs (conf=%.2f, iou=%.2f)",
            len(detections), elapsed, conf, iou,
        )
        return detections

    def detect_batch(
        self,
        images: list[str | Path | np.ndarray],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[list[dict]]:
        """Detect text regions in multiple images."""
        return [self.detect(img, conf_threshold, iou_threshold) for img in images]

    def detect_with_timing(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> tuple[list[dict], float]:
        """Detect and return results with inference time."""
        start = time.time()
        detections = self.detect(image, conf_threshold, iou_threshold)
        return detections, time.time() - start

    def _load_image(self, image: str | Path | np.ndarray) -> np.ndarray | None:
        if isinstance(image, np.ndarray):
            return image if image.ndim >= 2 else None
        try:
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Cannot read image: %s", image)
            return img
        except Exception as exc:
            logger.warning("Error loading image %s: %s", image, exc)
            return None

    def _parse_results(self, results: list) -> list[dict]:
        detections: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name,
                })
        return detections


def draw_boxes(
    image: str | Path | np.ndarray,
    boxes: list[list[int]],
    scores: list[float] | None = None,
    class_names: dict[int, str] | None = None,
    class_ids: list[int] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on image (standalone convenience function)."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image}")
    else:
        img = image.copy()

    if not boxes:
        return img

    if scores is None:
        scores = [0.0] * len(boxes)
    if class_names is None:
        class_names = {0: "text"}
    if class_ids is None:
        class_ids = [0] * len(boxes)

    palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        box_color = palette[cls_id % len(palette)] if class_ids != [0] * len(boxes) else color

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

        label_parts = [class_names.get(cls_id, f"class_{cls_id}")]
        if show_confidence and score > 0:
            label_parts.append(f"{score:.2f}")
        label = " ".join(label_parts)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y1 - 5, th + baseline + 5)
        cv2.rectangle(img, (x1, label_y - th - baseline - 4), (x1 + tw + 4, label_y), box_color, -1)
        cv2.putText(
            img, label, (x1 + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img
