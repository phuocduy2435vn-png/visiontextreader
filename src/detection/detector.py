"""
detector.py — Text region detection using YOLO model.

Provides TextDetector class for detecting text regions in images.
Returns plain Python objects (no tensors, no YOLO objects).

Usage:
    from src.detection import TextDetector

    detector = TextDetector("weights/best.pt")
    boxes = detector.detect("image.jpg")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("visiontextreader.detection")


class TextDetector:
    """Text region detector using trained YOLO model.

    This class wraps YOLO inference and returns plain Python objects.
    No tensors or YOLO-specific objects are exposed to the caller.

    Attributes:
        model_path: Path to the YOLO model file.
        conf_threshold: Minimum confidence for detection.
        iou_threshold: IoU threshold for NMS.
        imgsz: Input image size for inference.
        device: Inference device ('0', 'cpu', etc.).
        class_names: Mapping of class_id to class name.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize TextDetector.

        Args:
            model_path: Path to trained YOLO model (.pt file).
            conf_threshold: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for Non-Maximum Suppression.
            imgsz: Input image size for model inference.
            device: Inference device ('0' for GPU, 'cpu' for CPU).
            class_names: Custom mapping of class_id to name.
                         If None, uses model's class names or defaults to {0: 'text'}.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.class_names = class_names or {0: "text"}
        self._model = None

        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(
            "TextDetector initialized — model=%s, conf=%.2f, iou=%.2f, imgsz=%d",
            self.model_path.name, self.conf_threshold, self.iou_threshold, self.imgsz,
        )

    def load_model(self) -> None:
        """Load the YOLO model from disk.

        This is called automatically on first detect() call.
        Can be called explicitly to pre-load the model.
        """
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )

        logger.info("Loading model: %s", self.model_path)
        self._model = YOLO(str(self.model_path))

        # Update class names from model if available
        if hasattr(self._model, "names") and self._model.names:
            self.class_names = self._model.names
            logger.info("Loaded class names from model: %s", self.class_names)

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect text regions in an image.

        Args:
            image: Input image as file path (str/Path) or numpy array (BGR).
            conf_threshold: Override confidence threshold for this call.
            iou_threshold: Override IoU threshold for this call.

        Returns:
            List of detection dictionaries, each containing:
                - bbox: List[int] — [x1, y1, x2, y2] in pixel coordinates
                - confidence: float — detection confidence score
                - class_id: int — class identifier
                - class_name: str — human-readable class name

            Returns empty list if no detections or on error.

        Example:
            >>> detector = TextDetector("model.pt")
            >>> results = detector.detect("photo.jpg")
            >>> for r in results:
            ...     print(r["bbox"], r["confidence"])
        """
        # Ensure model is loaded
        if self._model is None:
            self.load_model()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        # Load image if path provided
        img = self._load_image(image)
        if img is None:
            return []

        # Run YOLO inference
        start_time = time.time()
        try:
            results = self._model.predict(
                source=img,
                conf=conf,
                iou=iou,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return []
        inference_time = time.time() - start_time

        # Parse results into plain Python objects
        detections = self._parse_results(results)

        logger.debug(
            "Detected %d regions in %.3fs (conf=%.2f, iou=%.2f)",
            len(detections), inference_time, conf, iou,
        )

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Detect text regions in multiple images.

        Args:
            images: List of images (paths or numpy arrays).
            conf_threshold: Override confidence threshold.
            iou_threshold: Override IoU threshold.

        Returns:
            List of detection lists, one per input image.
        """
        return [
            self.detect(img, conf_threshold, iou_threshold)
            for img in images
        ]

    def detect_with_timing(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Detect and return results with inference time.

        Args:
            image: Input image.
            conf_threshold: Override confidence threshold.
            iou_threshold: Override IoU threshold.

        Returns:
            Tuple of (detections, inference_time_seconds).
        """
        start = time.time()
        detections = self.detect(image, conf_threshold, iou_threshold)
        elapsed = time.time() - start
        return detections, elapsed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from path or validate numpy array.

        Args:
            image: File path or BGR numpy array.

        Returns:
            BGR numpy array, or None if loading fails.
        """
        if isinstance(image, np.ndarray):
            if image.ndim < 2:
                logger.warning("Invalid image array: ndim=%d", image.ndim)
                return None
            return image

        try:
            path = Path(image)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Cannot read image: %s", path)
                return None
            return img
        except Exception as exc:
            logger.warning("Error loading image %s: %s", image, exc)
            return None

    def _parse_results(self, results: list) -> List[Dict[str, Any]]:
        """Parse YOLO results into plain Python dictionaries.

        Args:
            results: Raw YOLO prediction results.

        Returns:
            List of detection dictionaries.
        """
        detections: List[Dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Extract bounding box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int)

                # Extract confidence score
                confidence = float(box.conf[0])

                # Extract class ID
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name,
                })

        return detections


def crop(
    image: Union[str, Path, np.ndarray],
    boxes: List[List[int]],
    margin: int = 0,
    padding: int = 0,
) -> List[np.ndarray]:
    """Crop regions from image based on bounding boxes.

    This is a standalone function for quick cropping.
    For more control, use TextCropper from crop.py.

    Args:
        image: Input image (path or numpy array).
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        margin: Margin around bbox in pixels (negative = shrink).
        padding: Padding around bbox in pixels (positive = expand).

    Returns:
        List of cropped image arrays (BGR), in the same order as input boxes.
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            return []
    else:
        img = image

    h, w = img.shape[:2]
    crops: List[np.ndarray] = []

    for box in boxes:
        x1, y1, x2, y2 = box

        # Apply margin and padding
        x1c = max(0, x1 - margin + padding)
        y1c = max(0, y1 - margin + padding)
        x2c = min(w, x2 + margin - padding)
        y2c = min(h, y2 + margin - padding)

        # Skip invalid crops
        if x2c <= x1c or y2c <= y1c:
            continue

        crop_img = img[y1c:y2c, x1c:x2c]
        if crop_img.size > 0:
            crops.append(crop_img.copy())

    return crops


def draw_boxes(
    image: Union[str, Path, np.ndarray],
    boxes: List[List[int]],
    scores: Optional[List[float]] = None,
    class_names: Optional[Dict[int, str]] = None,
    class_ids: Optional[List[int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on image.

    This is a standalone function for quick visualization.

    Args:
        image: Input image (path or numpy array).
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        scores: Optional confidence scores for each box.
        class_names: Mapping of class_id to name (for labels).
        class_ids: Class IDs for color selection.
        color: Default box color (BGR) when class_ids not provided.
        thickness: Box line thickness in pixels.
        font_scale: Font size for labels.
        show_confidence: Whether to show confidence scores.

    Returns:
        Annotated image (numpy array, BGR).
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image}")
    else:
        img = image.copy()

    if not boxes:
        return img

    # Default values
    if scores is None:
        scores = [0.0] * len(boxes)
    if class_names is None:
        class_names = {0: "text"}
    if class_ids is None:
        class_ids = [0] * len(boxes)

    # Color palette for different classes
    palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box

        # Select color based on class
        box_color = palette[cls_id % len(palette)] if class_ids != [0] * len(boxes) else color

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

        # Build label text
        label_parts = [class_names.get(cls_id, f"class_{cls_id}")]
        if show_confidence and score > 0:
            label_parts.append(f"{score:.2f}")
        label = " ".join(label_parts)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y = max(y1 - 5, th + baseline + 5)
        cv2.rectangle(
            img, (x1, label_y - th - baseline - 4), (x1 + tw + 4, label_y),
            box_color, -1,
        )
        cv2.putText(
            img, label, (x1 + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img


def save_crops(
    crops: List[np.ndarray],
    folder: Union[str, Path],
    prefix: str = "crop",
    extension: str = "png",
) -> List[Path]:
    """Save cropped images to a folder.

    Args:
        crops: List of cropped image arrays (BGR).
        folder: Output directory (created if not exists).
        prefix: Filename prefix.
        extension: Image file extension (png, jpg, etc.).

    Returns:
        List of saved file paths.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for i, crop_img in enumerate(crops, start=1):
        filename = f"{prefix}_{i:03d}.{extension}"
        filepath = folder / filename

        try:
            cv2.imwrite(str(filepath), crop_img)
            saved.append(filepath)
        except Exception as exc:
            logger.warning("Failed to save crop %s: %s", filepath, exc)

    logger.info("Saved %d crops to %s", len(saved), folder)
    return saved
