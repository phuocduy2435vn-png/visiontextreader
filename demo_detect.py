"""
demo_detect.py — Full detection pipeline demo.

Pipeline:
    Load model → Load image → Detect → Draw bbox → Crop → Save

Usage:
    python demo_detect.py image.jpg
    python demo_detect.py image.jpg --model weights/best.pt
    python demo_detect.py image.jpg --conf 0.3 --output outputs/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection import TextDetector
from src.detection.detector import crop, draw_boxes, save_crops

logger = logging.getLogger("visiontextreader.demo")


def run_demo(
    image_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path = "outputs/demo",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "",
    margin: int = 5,
    padding: int = 10,
) -> Dict[str, Any]:
    """Run complete detection pipeline demo.

    Args:
        image_path: Path to input image.
        model_path: Path to YOLO model.
        output_dir: Output directory for results.
        conf_threshold: Confidence threshold.
        iou_threshold: IoU threshold for NMS.
        imgsz: Input image size.
        device: Inference device.
        margin: Crop margin in pixels.
        padding: Crop padding in pixels.

    Returns:
        Dict with pipeline results and statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 1: Load model
    # ============================================================
    print("=" * 60)
    print("  VisionTextReader — Detection Pipeline Demo")
    print("=" * 60)
    print()

    t_start = time.time()
    detector = TextDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
    )
    detector.load_model()
    t_model = time.time() - t_start
    print(f"[1] Model loaded in {t_model * 1000:.1f} ms")

    # ============================================================
    # Step 2: Load image
    # ============================================================
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Cannot read image: {image_path}")
        return {"error": f"Cannot read image: {image_path}"}

    h, w = img.shape[:2]
    print(f"[2] Image loaded: {image_path.name} ({w}x{h})")

    # ============================================================
    # Step 3: Detect text regions
    # ============================================================
    t_detect_start = time.time()
    detections = detector.detect(img)
    t_detect = time.time() - t_detect_start

    print(f"[3] Detected {len(detections)} text regions in {t_detect * 1000:.1f} ms")

    # ============================================================
    # Step 4: Draw bounding boxes
    # ============================================================
    if detections:
        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]
        class_ids = [d["class_id"] for d in detections]

        annotated = draw_boxes(
            img, boxes, scores,
            class_names=detector.class_names,
            class_ids=class_ids,
            show_confidence=True,
        )

        # Save annotated image
        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[4] Annotated image saved: {annotated_path}")
    else:
        annotated = img.copy()
        print("[4] No detections to draw")

    # ============================================================
    # Step 5: Crop text regions
    # ============================================================
    if detections:
        boxes = [d["bbox"] for d in detections]
        crops = crop(img, boxes, margin=margin, padding=padding)

        # Save crops
        crops_dir = output_dir / "crops"
        saved_crops = save_crops(crops, crops_dir, prefix=image_path.stem)
        print(f"[5] Cropped {len(saved_crops)} regions → {crops_dir}")
    else:
        saved_crops = []
        print("[5] No regions to crop")

    # ============================================================
    # Step 6: Save detection results
    # ============================================================
    # Save as JSON
    json_path = output_dir / f"{image_path.stem}_predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"[6] Predictions saved: {json_path}")

    # Save as YOLO TXT
    txt_path = output_dir / f"{image_path.stem}_predictions.txt"
    yolo_lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")
    print(f"       YOLO format saved: {txt_path}")

    # ============================================================
    # Step 7: Print statistics
    # ============================================================
    print()
    print("-" * 60)
    print("  STATISTICS")
    print("-" * 60)

    if detections:
        confidences = [d["confidence"] for d in detections]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        print(f"  Detected objects : {len(detections)}")
        print(f"  Avg confidence   : {avg_conf:.4f}")
        print(f"  Min confidence   : {min_conf:.4f}")
        print(f"  Max confidence   : {max_conf:.4f}")
        print(f"  Inference time   : {t_detect * 1000:.1f} ms")
        print(f"  Model load time  : {t_model * 1000:.1f} ms")
        print(f"  Crop count       : {len(saved_crops)}")
    else:
        print("  No objects detected")

    print("-" * 60)
    print(f"  Output directory : {output_dir}")
    print("=" * 60)

    # Return results
    return {
        "detections": detections,
        "annotated_path": str(annotated_path) if detections else None,
        "crops": [str(p) for p in saved_crops],
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "stats": {
            "num_detections": len(detections),
            "avg_confidence": avg_conf if detections else 0,
            "inference_time_ms": t_detect * 1000,
            "model_load_time_ms": t_model * 1000,
            "image_size": [w, h],
        },
    }


def main() -> None:
    """CLI entry point for demo."""
    parser = argparse.ArgumentParser(
        description="VisionTextReader — Detection Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_detect.py image.jpg
  python demo_detect.py image.jpg --model weights/best.pt
  python demo_detect.py image.jpg --conf 0.3 --output outputs/demo
        """,
    )
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument(
        "--model", type=str,
        default="runs/detect/visiontextreader/weights/best.pt",
        help="Path to YOLO model (default: runs/detect/visiontextreader/weights/best.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (0, cpu)")
    parser.add_argument("--output", "-o", type=str, default="outputs/demo", help="Output directory")
    parser.add_argument("--margin", type=int, default=5, help="Crop margin")
    parser.add_argument("--padding", type=int, default=10, help="Crop padding")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Run demo
    results = run_demo(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        margin=args.margin,
        padding=args.padding,
    )

    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
