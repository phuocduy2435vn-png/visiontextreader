"""
detect.py — CLI for text detection using trained YOLO model.

Usage:
    python detect.py image.jpg
    python detect.py image.jpg --model weights/best.pt --conf 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.detection import TextDetector
from src.detection.detector import draw_boxes
from utils import load_image, save_image

logger = logging.getLogger("visiontextreader.detect")


def main() -> None:
    parser = argparse.ArgumentParser(description="Text Detection with YOLO")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (0, cpu)")
    parser.add_argument("--output", "-o", type=str, default="output/detections", help="Output directory")
    parser.add_argument("--save-json", action="store_true", help="Save results as JSON")
    parser.add_argument("--save-txt", action="store_true", help="Save results as TXT")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    detector = TextDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    image_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = detector.detect(image_path)

    # Annotate and save
    img = load_image(image_path)
    if img is not None and detections:
        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]
        class_ids = [d["class_id"] for d in detections]
        annotated = draw_boxes(img, boxes, scores, detector.class_names, class_ids)
        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        save_image(annotated, annotated_path)
        print(f"Annotated image saved: {annotated_path}")

    print(f"\nDetected {len(detections)} text regions:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")

    if args.save_json:
        json_path = output_dir / f"{image_path.stem}_predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_path}")

    if args.save_txt and img is not None:
        txt_path = output_dir / f"{image_path.stem}_predictions.txt"
        h, w = img.shape[:2]
        lines = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved TXT: {txt_path}")


if __name__ == "__main__":
    main()
