"""
split_dataset.py — Dataset splitting for VisionTextReader.

Splits the converted YOLO dataset from a single pool (datasets/processed/train/)
into train / val / test sets with configurable ratios.

Design principles:
    - Copy files (never move) to preserve the original pool.
    - Deterministic via configurable random seed.
    - No data loss — every image ends up in exactly one split.
    - Handles label files alongside images atomically.

Usage:
    python scripts/split_dataset.py                        # default 70/20/10
    python scripts/split_dataset.py --train 0.8 --val 0.1 --test 0.1
    python scripts/split_dataset.py --seed 123
    python scripts/split_dataset.py --dry-run              # preview without copying
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSION,
    ProjectPaths,
    get_project_paths,
    logger as root_logger,
)

logger = logging.getLogger("visiontextreader.split")


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class SplitStats:
    """Statistics for a single split operation."""
    split_name: str
    images_copied: int = 0
    labels_copied: int = 0
    labels_missing: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  {self.split_name:>5}: {self.images_copied} images, "
            f"{self.labels_copied} labels",
        ]
        if self.labels_missing:
            lines.append(f"         ({self.labels_missing} images had no label)")
        if self.errors:
            lines.append(f"         ({len(self.errors)} errors)")
        return "\n".join(lines)


@dataclass
class SplitResult:
    """Aggregate result of a full split operation."""
    total_images: int = 0
    splits: Dict[str, SplitStats] = field(default_factory=dict)
    seed: int = 42
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 60,
            "  DATASET SPLIT REPORT",
            "=" * 60,
            f"  Seed: {self.seed}",
            f"  Ratios: train={self.ratios[0]:.0%}  val={self.ratios[1]:.0%}  test={self.ratios[2]:.0%}",
            f"  Total images in pool: {self.total_images}",
            "-" * 60,
        ]
        for stats in self.splits.values():
            lines.append(stats.summary())
        lines.append("=" * 60)
        return "\n".join(lines)


# =========================================================================
# Splitting engine
# =========================================================================

class DatasetSplitter:
    """Splits a YOLO-format dataset into train/val/test.

    The splitter reads image files from the source pool directory,
    shuffles them deterministically, and copies each image + its
    matching label file to the appropriate output directory.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        """Initialise the splitter.

        Args:
            paths: Project directory layout.
            train_ratio: Fraction of data for training (default 70%).
            val_ratio: Fraction of data for validation (default 20%).
            test_ratio: Fraction of data for testing (default 10%).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If ratios don't sum to 1.0.
        """
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {ratio_sum:.4f} "
                f"({train_ratio} + {val_ratio} + {test_ratio})"
            )

        self.paths = paths
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, dry_run: bool = False) -> SplitResult:
        """Execute the split operation.

        Args:
            dry_run: If True, compute splits but don't copy files.

        Returns:
            SplitResult with statistics.
        """
        result = SplitResult(seed=self.seed, ratios=(self.train_ratio, self.val_ratio, self.test_ratio))

        # ------------------------------------------------------------------
        # Step 1: Find all images in the source pool
        # ------------------------------------------------------------------
        source_pool = self.paths.train_images
        source_labels = self.paths.train_labels

        if not source_pool.exists():
            logger.error("Source pool does not exist: %s", source_pool)
            return result

        image_files = self._collect_images(source_pool)
        result.total_images = len(image_files)

        if not image_files:
            logger.warning("No images found in source pool: %s", source_pool)
            return result

        logger.info("Found %d images in source pool", len(image_files))

        # ------------------------------------------------------------------
        # Step 2: Shuffle deterministically
        # ------------------------------------------------------------------
        rng = random.Random(self.seed)
        shuffled = list(image_files)
        rng.shuffle(shuffled)

        # ------------------------------------------------------------------
        # Step 3: Compute split boundaries
        # ------------------------------------------------------------------
        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        split_indices: Dict[str, List[Path]] = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        logger.info(
            "Split distribution: train=%d, val=%d, test=%d",
            len(split_indices["train"]),
            len(split_indices["val"]),
            len(split_indices["test"]),
        )

        # ------------------------------------------------------------------
        # Step 4: Copy files to split directories
        # ------------------------------------------------------------------
        for split_name, files in split_indices.items():
            stats = SplitStats(split_name=split_name)
            images_dir = getattr(self.paths, f"{split_name}_images")
            labels_dir = getattr(self.paths, f"{split_name}_labels")

            if not dry_run:
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                if dry_run:
                    stats.images_copied += 1
                    # Check label exists
                    lbl_src = source_labels / f"{img_path.stem}{LABEL_EXTENSION}"
                    if lbl_src.exists():
                        stats.labels_copied += 1
                    else:
                        stats.labels_missing += 1
                    continue

                # Copy image (skip if source == destination)
                img_dst = images_dir / img_path.name
                try:
                    if img_path.resolve() != img_dst.resolve():
                        shutil.copy2(img_path, img_dst)
                    stats.images_copied += 1
                except (OSError, shutil.Error) as exc:
                    stats.errors.append(f"Failed to copy {img_path.name}: {exc}")
                    continue

                # Copy matching label (skip if source == destination)
                lbl_src = source_labels / f"{img_path.stem}{LABEL_EXTENSION}"
                if lbl_src.exists():
                    lbl_dst = labels_dir / f"{img_path.stem}{LABEL_EXTENSION}"
                    try:
                        if lbl_src.resolve() != lbl_dst.resolve():
                            shutil.copy2(lbl_src, lbl_dst)
                        stats.labels_copied += 1
                    except (OSError, shutil.Error) as exc:
                        stats.errors.append(f"Failed to copy label {lbl_src.name}: {exc}")
                else:
                    stats.labels_missing += 1

            result.splits[split_name] = stats

        return result

    @staticmethod
    def _collect_images(directory: Path) -> List[Path]:
        """Collect all image files from a directory (non-recursive)."""
        images: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(set(images))


# =========================================================================
# CLI entry point
# =========================================================================

def main() -> None:
    """Command-line interface for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val/test sets.",
    )
    parser.add_argument(
        "--train", "-t",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val", "-v",
        type=float,
        default=0.20,
        help="Validation set ratio (default: 0.20)",
    )
    parser.add_argument(
        "--test", "-e",
        type=float,
        default=0.10,
        help="Test set ratio (default: 0.10)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview the split without copying any files.",
    )
    args = parser.parse_args()

    paths = get_project_paths()

    try:
        splitter = DatasetSplitter(
            paths=paths,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        sys.exit(1)

    result = splitter.split(dry_run=args.dry_run)
    print(result.summary())


if __name__ == "__main__":
    main()
