"""
detect.py — Text detection using trained YOLO model.

Provides TextDetector class for detecting text regions in images.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    load_image, save_image, draw_detections,
    yolo_to_xyxy, nms, COLORS, IMAGE_EXTENSIONS,
)

logger = logging.getLogger("visiontextreader.detect")


class TextDetector:
    """Text region detector using YOLO model.

    Usage:
        detector = TextDetector("runs/detect/visiontextreader/weights/best.pt")
        results = detector.detect("image.jpg")
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        class_names: Optional[Dict[int, str]] = None,
    ):
        """Initialize detector.

        Args:
            model_path: Path to trained YOLO model (.pt).
            conf_threshold: Confidence threshold for detection.
            iou_threshold: IoU threshold for NMS.
            imgsz: Input image size for inference.
            device: Inference device ('0', 'cpu', etc.).
            class_names: Mapping of class_id to name.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.class_names = class_names or {0: "text"}
        self.model = None

    def load_model(self) -> None:
        """Load the YOLO model from disk."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        logger.info("Loaded model: %s", self.model_path)

        # Update class names from model if available
        if hasattr(self.model, "names") and self.model.names:
            self.class_names = self.model.names

    def detect(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect text regions in an image.

        Args:
            image: Image path, numpy array, or file path.
            conf_threshold: Override confidence threshold.
            iou_threshold: Override IoU threshold.

        Returns:
            List of detection dicts:
                [
                    {
                        "class_id": int,
                        "class_name": str,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2],
                    },
                    ...
                ]
        """
        if self.model is None:
            self.load_model()

        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        # Load image if path
        if isinstance(image, (str, Path)):
            img = load_image(image)
            if img is None:
                return []
        else:
            img = image

        # Run inference
        start_time = time.time()
        results = self.model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        inference_time = time.time() - start_time

        # Parse results
        detections: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # Get coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int)

                # Get confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        logger.debug(
            "Detected %d regions in %.3fs",
            len(detections), inference_time,
        )

        return detections

    def detect_batch(
        self,
        image_paths: List[str | Path],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Detect text regions in multiple images.

        Args:
            image_paths: List of image paths.
            conf_threshold: Override confidence threshold.
            iou_threshold: Override IoU threshold.

        Returns:
            Dict mapping image path to detection results.
        """
        results = {}
        for img_path in image_paths:
            detections = self.detect(
                img_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            results[str(img_path)] = detections
        return results

    def detect_and_annotate(
        self,
        image: str | Path | np.ndarray,
        output_path: Optional[str | Path] = None,
        conf_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """Detect and annotate image with bounding boxes.

        Args:
            image: Image path or numpy array.
            output_path: Path to save annotated image (optional).
            conf_threshold: Override confidence threshold.

        Returns:
            Tuple of (detections, annotated_image).
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = load_image(image)
            if img is None:
                return [], None
        else:
            img = image.copy()

        # Detect
        detections = self.detect(image, conf_threshold=conf_threshold)

        # Annotate
        if detections:
            boxes = [d["bbox"] for d in detections]
            scores = [d["confidence"] for d in detections]
            class_ids = [d["class_id"] for d in detections]
            annotated = draw_detections(img, boxes, scores, self.class_names, class_ids)
        else:
            annotated = img

        # Save if requested
        if output_path:
            save_image(annotated, output_path)

        return detections, annotated


def main() -> None:
    """CLI entry point for detection."""
    parser = argparse.ArgumentParser(description="Text Detection with YOLO")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--model", type=str, default="runs/detect/visiontextreader/weights/best.pt",
                        help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (0, cpu)")
    parser.add_argument("--output", "-o", type=str, default="outputs/detections",
                        help="Output directory")
    parser.add_argument("--save-json", action="store_true", help="Save results as JSON")
    parser.add_argument("--save-txt", action="store_true", help="Save results as TXT")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Create detector
    detector = TextDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    # Detect
    image_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
    detections, annotated = detector.detect_and_annotate(
        image_path,
        output_path=annotated_path,
    )

    # Print results
    print(f"\nDetected {len(detections)} text regions:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2f} "
              f"at {det['bbox']}")

    # Save JSON
    if args.save_json:
        json_path = output_dir / f"{image_path.stem}_predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_path}")

    # Save TXT (YOLO format)
    if args.save_txt:
        txt_path = output_dir / f"{image_path.stem}_predictions.txt"
        lines = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Convert to YOLO format (requires image dimensions)
            img = load_image(image_path)
            if img is not None:
                h, w = img.shape[:2]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved TXT: {txt_path}")

    print(f"\nAnnotated image saved: {annotated_path}")


if __name__ == "__main__":
    main()
