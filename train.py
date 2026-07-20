"""
train.py — YOLO training for text detection.

Supports YOLOv8/v11 with resume, early stopping, mixed precision.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("visiontextreader.train")


class YOLOTrainer:
    """YOLO training pipeline for text detection."""

    def __init__(
        self,
        data_yaml: str | Path,
        model_name: str = "yolov8n.pt",
        project_dir: str | Path = "runs/detect",
        experiment_name: str = "visiontextreader",
    ):
        self.data_yaml = str(Path(data_yaml).resolve())
        self.model_name = model_name
        self.project_dir = str(Path(project_dir).resolve())
        self.experiment_name = experiment_name

    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        workers: int = 8,
        patience: int = 50,
        resume: bool = False,
        amp: bool = True,
        cache: bool | str = True,
        device: str = "",
        exist_ok: bool = True,
    ) -> dict[str, float]:
        """Run YOLO training. Returns metrics dict."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        model = YOLO(self.model_name)
        logger.info("Loaded model: %s", self.model_name)

        train_args: dict = {
            "data": self.data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": warmup_momentum,
            "warmup_bias_lr": warmup_bias_lr,
            "workers": workers,
            "patience": patience,
            "resume": resume,
            "amp": amp,
            "cache": cache,
            "project": self.project_dir,
            "name": self.experiment_name,
            "exist_ok": exist_ok,
            "verbose": True,
            "plots": True,
            "save": True,
            "save_period": -1,
        }

        if device:
            train_args["device"] = device

        logger.info("Starting training")
        results = model.train(**train_args)

        metrics: dict[str, float] = {}
        if hasattr(results, "box"):
            metrics["mAP50"] = results.box.map50
            metrics["mAP50-95"] = results.box.map
            metrics["precision"] = results.box.mp
            metrics["recall"] = results.box.mr

        logger.info("Training completed — %s/%s", self.project_dir, self.experiment_name)
        return metrics

    def validate(
        self,
        data_yaml: str | None = None,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "",
        split: str = "val",
    ) -> dict[str, float]:
        """Run validation on trained model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        best_model = Path(self.project_dir) / self.experiment_name / "weights" / "best.pt"
        if not best_model.exists():
            best_model = Path(self.project_dir) / self.experiment_name / "weights" / "last.pt"
        if not best_model.exists():
            logger.error("No model weights found at %s", best_model)
            return {}

        model = YOLO(str(best_model))
        val_args: dict = {
            "data": data_yaml or self.data_yaml,
            "imgsz": imgsz,
            "batch": batch,
            "split": split,
            "verbose": True,
            "plots": True,
        }
        if device:
            val_args["device"] = device

        results = model.val(**val_args)

        metrics: dict[str, float] = {}
        if hasattr(results, "box"):
            metrics["mAP50"] = results.box.map50
            metrics["mAP50-95"] = results.box.map
            metrics["precision"] = results.box.mp
            metrics["recall"] = results.box.mr

        logger.info("Validation results: %s", metrics)
        return metrics

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
    ) -> Path:
        """Export trained model. Returns path to exported model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        best_model = Path(self.project_dir) / self.experiment_name / "weights" / "best.pt"
        if not best_model.exists():
            raise FileNotFoundError(f"No model weights: {best_model}")

        model = YOLO(str(best_model))
        result = model.export(format=format, imgsz=imgsz, half=half, dynamic=dynamic)
        logger.info("Exported model: %s", result)
        return Path(result)

    def get_latest_run(self) -> Path | None:
        """Get path to the latest training run directory."""
        project = Path(self.project_dir)
        if not project.exists():
            return None
        runs = sorted(project.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for run in runs:
            if run.is_dir() and (run / "weights").exists():
                return run
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Training for Text Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pre-trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device (0, cpu)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default="visiontextreader", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    trainer = YOLOTrainer(
        data_yaml=args.data,
        model_name=args.model,
        project_dir=args.project,
        experiment_name=args.name,
    )

    metrics = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )

    print("\n=== Training Complete ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
