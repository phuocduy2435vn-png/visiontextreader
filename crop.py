"""
crop.py — Crop text regions from images based on detection results.

Provides TextCropper class for extracting text regions with margin/padding.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from utils import load_image, save_image

logger = logging.getLogger("visiontextreader.crop")


class TextCropper:
    """Crop text regions from images."""

    def __init__(
        self,
        margin: int = 5,
        padding: int = 10,
        min_size: int = 10,
        output_format: str = "png",
    ):
        self.margin = margin
        self.padding = padding
        self.min_size = min_size
        self.output_format = output_format

    def crop(
        self,
        image: str | Path | np.ndarray,
        boxes: list[list[int]],
        scores: list[float] | None = None,
    ) -> list[np.ndarray]:
        """Crop text regions from image. Returns list of cropped arrays."""
        if isinstance(image, (str, Path)):
            img = load_image(image)
            if img is None:
                return []
        else:
            img = image

        h, w = img.shape[:2]
        crops: list[np.ndarray] = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - self.margin + self.padding)
            y1 = max(0, y1 - self.margin + self.padding)
            x2 = min(w, x2 + self.margin - self.padding)
            y2 = min(h, y2 + self.margin - self.padding)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < self.min_size or crop_h < self.min_size:
                continue

            crop = img[y1:y2, x1:x2].copy()
            if crop.size > 0:
                crops.append(crop)

        logger.info("Cropped %d regions from image", len(crops))
        return crops

    def crop_to_files(
        self,
        image: str | Path | np.ndarray,
        boxes: list[list[int]],
        output_dir: str | Path,
        prefix: str = "crop",
        scores: list[float] | None = None,
    ) -> list[Path]:
        """Crop text regions and save to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        crops = self.crop(image, boxes, scores)
        saved_paths: list[Path] = []
        ext = f".{self.output_format}"

        for i, crop_img in enumerate(crops):
            output_path = output_dir / f"{prefix}_{i + 1:03d}{ext}"
            if save_image(crop_img, output_path):
                saved_paths.append(output_path)

        logger.info("Saved %d cropped images to %s", len(saved_paths), output_dir)
        return saved_paths

    def crop_with_metadata(
        self,
        image: str | Path | np.ndarray,
        boxes: list[list[int]],
        scores: list[float] | None = None,
    ) -> list[dict]:
        """Crop regions and return with metadata."""
        if isinstance(image, (str, Path)):
            img = load_image(image)
            if img is None:
                return []
        else:
            img = image

        h, w = img.shape[:2]
        results: list[dict] = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1c = max(0, x1 - self.margin + self.padding)
            y1c = max(0, y1 - self.margin + self.padding)
            x2c = min(w, x2 + self.margin - self.padding)
            y2c = min(h, y2 + self.margin - self.padding)

            crop_w = x2c - x1c
            crop_h = y2c - y1c
            if crop_w < self.min_size or crop_h < self.min_size:
                continue

            crop = img[y1c:y2c, x1c:x2c].copy()
            if crop.size == 0:
                continue

            score = scores[i] if scores and i < len(scores) else 0.0
            results.append({
                "crop": crop,
                "index": i,
                "original_bbox": box,
                "cropped_bbox": [x1c, y1c, x2c, y2c],
                "confidence": score,
                "width": crop_w,
                "height": crop_h,
            })

        return results

    def crop_from_detections(
        self,
        image: str | Path | np.ndarray,
        detections: list[dict],
        output_dir: str | Path,
        prefix: str = "crop",
    ) -> list[Path]:
        """Crop from TextDetector.detect() output."""
        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]
        return self.crop_to_files(image, boxes, output_dir, prefix, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Text Regions from Images")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--boxes", type=str, required=True, help="JSON file with bounding boxes")
    parser.add_argument("--output", "-o", type=str, default="outputs/crops", help="Output directory")
    parser.add_argument("--margin", type=int, default=5, help="Margin in pixels")
    parser.add_argument("--padding", type=int, default=10, help="Padding in pixels")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum crop size")
    parser.add_argument("--format", type=str, default="png", help="Output format")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    boxes_path = Path(args.boxes)
    if not boxes_path.exists():
        print(f"Error: boxes file not found: {boxes_path}")
        return

    with open(boxes_path, "r", encoding="utf-8") as f:
        boxes_data = json.load(f)

    if isinstance(boxes_data, list):
        if boxes_data and isinstance(boxes_data[0], dict):
            boxes = [d["bbox"] for d in boxes_data]
            scores = [d.get("confidence", 0.0) for d in boxes_data]
        else:
            boxes = boxes_data
            scores = None
    else:
        print("Error: boxes must be a list")
        return

    cropper = TextCropper(
        margin=args.margin,
        padding=args.padding,
        min_size=args.min_size,
        output_format=args.format,
    )

    saved_paths = cropper.crop_to_files(
        Path(args.image), boxes, Path(args.output),
        prefix=Path(args.image).stem, scores=scores,
    )

    print(f"\nCropped {len(saved_paths)} regions:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
