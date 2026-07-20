"""
dataset_splitter.py — Split YOLO dataset into train/val/test sets.

Supports creating full and small (processed_small) splits.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger("visiontextreader.dataset_splitter")


class DatasetSplitter:
    """Split a YOLO dataset into train/val/test sets."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")

    def split(self, dry_run: bool = False) -> dict[str, dict[str, int]]:
        """Execute the split operation."""
        random.seed(self.seed)

        image_files = self._collect_images()
        if not image_files:
            logger.warning("No images found in %s", self.source_dir)
            return {}

        logger.info("Found %d images to split", len(image_files))

        shuffled = list(image_files)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        logger.info(
            "Split distribution: train=%d, val=%d, test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )

        stats: dict[str, dict[str, int]] = {}
        for split_name, files in splits.items():
            stats[split_name] = self._copy_split(split_name, files, dry_run)

        return stats

    def split_small(
        self,
        target_count: int = 12000,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, dict[str, int]]:
        """Create a smaller dataset subset (processed_small).

        Args:
            target_count: Target number of total images.
            output_dir: Output directory (defaults to processed_small).
            dry_run: If True, compute but don't copy.
        """
        out_dir = output_dir or self.output_dir.parent / "processed_small"
        out_dir = Path(out_dir)

        random.seed(self.seed)
        image_files = self._collect_images()
        if not image_files:
            logger.warning("No images found in %s", self.source_dir)
            return {}

        shuffled = list(image_files)
        random.shuffle(shuffled)

        subset = shuffled[:target_count]
        logger.info("Selected %d / %d images for small subset", len(subset), len(image_files))

        n = len(subset)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        splits = {
            "train": subset[:train_end],
            "val": subset[train_end:val_end],
            "test": subset[val_end:],
        }

        stats: dict[str, dict[str, int]] = {}
        for split_name, files in splits.items():
            img_dir = out_dir / split_name / "images"
            lbl_dir = out_dir / split_name / "labels"

            if not dry_run:
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)

            images_copied = 0
            labels_copied = 0

            for img_path in files:
                if not dry_run:
                    dst_img = img_dir / img_path.name
                    if not dst_img.exists():
                        shutil.copy2(img_path, dst_img)
                images_copied += 1

                lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    if not dry_run:
                        dst_lbl = lbl_dir / lbl_path.name
                        if not dst_lbl.exists():
                            shutil.copy2(lbl_path, dst_lbl)
                    labels_copied += 1

            stats[split_name] = {"images": images_copied, "labels": labels_copied}
            logger.info("Small split '%s': %d images, %d labels", split_name, images_copied, labels_copied)

        return stats

    def _collect_images(self) -> list[Path]:
        """Collect all image files from source directory."""
        images: list[Path] = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
            images.extend(self.source_dir.glob(f"*{ext}"))
            images.extend(self.source_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _copy_split(
        self, split_name: str, image_files: list[Path], dry_run: bool,
    ) -> dict[str, int]:
        """Copy images and labels for a split."""
        img_dir = self.output_dir / split_name / "images"
        lbl_dir = self.output_dir / split_name / "labels"

        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        images_copied = 0
        labels_copied = 0
        labels_missing = 0

        for img_path in image_files:
            if not dry_run:
                dst_img = img_path
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
            images_copied += 1

            lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if lbl_path.exists():
                if not dry_run:
                    dst_lbl = lbl_dir / lbl_path.name
                    if not dst_lbl.exists():
                        shutil.copy2(lbl_path, dst_lbl)
                labels_copied += 1
            else:
                labels_missing += 1

        stats = {"images": images_copied, "labels": labels_copied, "missing_labels": labels_missing}
        logger.info(
            "Split '%s': %d images, %d labels, %d missing",
            split_name, images_copied, labels_copied, labels_missing,
        )
        return stats
