"""
benchmark.py — Benchmark YOLO detection performance.

Measures FPS, inference time (min/max/mean), and throughput
on a directory of test images.

Usage:
    python benchmark.py test/images --model weights/best.pt
    python benchmark.py test/images --model weights/best.pt --iterations 5
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection import TextDetector

logger = logging.getLogger("visiontextreader.benchmark")

# Image extensions to scan
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class BenchmarkRunner:
    """Benchmark YOLO detection performance.

    Measures:
        - FPS (frames per second)
        - Inference time: min, max, mean, median, std
        - Total time
        - Images processed
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
    ):
        """Initialize benchmark runner.

        Args:
            model_path: Path to YOLO model.
            conf_threshold: Confidence threshold.
            iou_threshold: IoU threshold.
            imgsz: Input image size.
            device: Inference device.
        """
        self.detector = TextDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            device=device,
        )

    def run(
        self,
        image_dir: str | Path,
        iterations: int = 1,
        warmup: int = 3,
        max_images: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run benchmark on image directory.

        Args:
            image_dir: Directory containing test images.
            iterations: Number of full passes over the dataset.
            warmup: Number of warmup iterations (not counted).
            max_images: Maximum number of images to use (None = all).

        Returns:
            Dict with benchmark results.
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Collect images
        image_files = self._collect_images(image_dir)
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        if max_images and max_images < len(image_files):
            image_files = image_files[:max_images]

        logger.info(
            "Benchmark: %d images, %d warmup, %d iterations",
            len(image_files), warmup, iterations,
        )

        # Pre-load all images into memory
        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

        if not images:
            raise ValueError("No valid images loaded")

        # Ensure model is loaded
        self.detector.load_model()

        # Warmup
        logger.info("Running %d warmup iterations...", warmup)
        for _ in range(warmup):
            for img in images:
                self.detector.detect(img)

        # Benchmark
        all_times: List[float] = []
        all_detections: List[int] = []
        total_start = time.time()

        for iteration in range(iterations):
            iter_start = time.time()
            for img in images:
                start = time.time()
                detections = self.detector.detect(img)
                elapsed = time.time() - start
                all_times.append(elapsed)
                all_detections.append(len(detections))
            iter_elapsed = time.time() - iter_start
            logger.info(
                "  Iteration %d/%d: %.2fs (%.1f FPS)",
                iteration + 1, iterations, iter_elapsed,
                len(images) / iter_elapsed if iter_elapsed > 0 else 0,
            )

        total_time = time.time() - total_start

        # Compute statistics
        times_ms = [t * 1000 for t in all_times]
        total_inferences = len(all_times)

        results = {
            "total_images": len(images),
            "total_inferences": total_inferences,
            "iterations": iterations,
            "warmup": warmup,
            "total_time_s": round(total_time, 3),
            "fps": round(total_inferences / total_time, 2) if total_time > 0 else 0,
            "inference_time_mean_ms": round(statistics.mean(times_ms), 2),
            "inference_time_median_ms": round(statistics.median(times_ms), 2),
            "inference_time_min_ms": round(min(times_ms), 2),
            "inference_time_max_ms": round(max(times_ms), 2),
            "inference_time_std_ms": round(statistics.stdev(times_ms), 2) if len(times_ms) > 1 else 0,
            "avg_detections_per_image": round(statistics.mean(all_detections), 2),
            "total_detections": sum(all_detections),
        }

        return results

    def _collect_images(self, directory: Path) -> List[Path]:
        """Collect all image files from directory."""
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    @staticmethod
    def print_results(results: Dict[str, float]) -> None:
        """Pretty-print benchmark results."""
        print()
        print("=" * 60)
        print("  BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Total images       : {results['total_images']}")
        print(f"  Total inferences   : {results['total_inferences']}")
        print(f"  Iterations         : {results['iterations']}")
        print(f"  Warmup             : {results['warmup']}")
        print("-" * 60)
        print(f"  FPS                : {results['fps']:.2f}")
        print(f"  Total time         : {results['total_time_s']:.3f} s")
        print("-" * 60)
        print(f"  Inference time:")
        print(f"    Mean             : {results['inference_time_mean_ms']:.2f} ms")
        print(f"    Median           : {results['inference_time_median_ms']:.2f} ms")
        print(f"    Min              : {results['inference_time_min_ms']:.2f} ms")
        print(f"    Max              : {results['inference_time_max_ms']:.2f} ms")
        print(f"    Std              : {results['inference_time_std_ms']:.2f} ms")
        print("-" * 60)
        print(f"  Detections:")
        print(f"    Total            : {results['total_detections']}")
        print(f"    Avg per image    : {results['avg_detections_per_image']:.2f}")
        print("=" * 60)
        print()


def main() -> None:
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO Detection Performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py test/images
  python benchmark.py test/images --model weights/best.pt
  python benchmark.py test/images --iterations 5 --warmup 3
        """,
    )
    parser.add_argument("image_dir", type=str, help="Directory with test images")
    parser.add_argument(
        "--model", type=str,
        default="runs/detect/visiontextreader/weights/best.pt",
        help="Path to YOLO model",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (0, cpu)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to use")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Run benchmark
    runner = BenchmarkRunner(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    try:
        results = runner.run(
            image_dir=args.image_dir,
            iterations=args.iterations,
            warmup=args.warmup,
            max_images=args.max_images,
        )
        runner.print_results(results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
