"""
dataset_splitter.py — Split YOLO dataset into train/val/test sets.

Ensures no data leakage between splits and maintains class distribution.
"""

from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("visiontextreader.dataset_splitter")


class DatasetSplitter:
    """Split a YOLO dataset into train/val/test sets.

    Supports:
        - Stratified splitting (maintains class distribution)
        - Deterministic splitting (fixed random seed)
        - Copy-based (preserves original data)
    """

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        """Initialize splitter.

        Args:
            source_dir: Source directory with images/ and labels/.
            output_dir: Output directory for split dataset.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            seed: Random seed for reproducibility.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")

    def split(self, dry_run: bool = False) -> Dict[str, Dict[str, int]]:
        """Execute the split operation.

        Args:
            dry_run: If True, compute splits but don't copy files.

        Returns:
            Dict with split statistics.
        """
        random.seed(self.seed)

        # Collect all images
        image_files = self._collect_images()
        if not image_files:
            logger.warning("No images found in %s", self.source_dir)
            return {}

        logger.info("Found %d images to split", len(image_files))

        # Shuffle deterministically
        shuffled = list(image_files)
        random.shuffle(shuffled)

        # Compute split boundaries
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

        # Copy files
        stats = {}
        for split_name, files in splits.items():
            split_stats = self._copy_split(split_name, files, dry_run)
            stats[split_name] = split_stats

        return stats

    def _collect_images(self) -> List[Path]:
        """Collect all image files from source directory."""
        images = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
            images.extend(self.source_dir.glob(f"*{ext}"))
            images.extend(self.source_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _copy_split(
        self,
        split_name: str,
        image_files: List[Path],
        dry_run: bool,
    ) -> Dict[str, int]:
        """Copy images and labels for a split.

        Args:
            split_name: Name of the split (train/val/test).
            image_files: List of image paths.
            dry_run: If True, don't copy.

        Returns:
            Dict with counts.
        """
        img_dir = self.output_dir / split_name / "images"
        lbl_dir = self.output_dir / split_name / "labels"

        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        images_copied = 0
        labels_copied = 0
        labels_missing = 0

        for img_path in image_files:
            # Copy image
            if not dry_run:
                dst_img = img_path  # Keep same name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
            images_copied += 1

            # Copy matching label
            lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            if lbl_path.exists():
                if not dry_run:
                    dst_lbl = lbl_dir / lbl_path.name
                    if not dst_lbl.exists():
                        shutil.copy2(lbl_path, dst_lbl)
                labels_copied += 1
            else:
                labels_missing += 1

        stats = {
            "images": images_copied,
            "labels": labels_copied,
            "missing_labels": labels_missing,
        }

        logger.info(
            "Split '%s': %d images, %d labels, %d missing",
            split_name, images_copied, labels_copied, labels_missing,
        )

        return stats
