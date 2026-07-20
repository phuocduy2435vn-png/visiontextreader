"""
config.py — Central configuration for VisionTextReader.

Project paths, dataset registry, class mappings, and pipeline parameters.
All paths use pathlib for cross-platform / Kaggle compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("visiontextreader.config")


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

class TextClass(IntEnum):
    TEXT = 0


CLASS_NAMES: dict[int, str] = {cls.value: cls.name.lower() for cls in TextClass}
NUM_CLASSES: int = len(TextClass)
TEXT_CLASS: int = TextClass.TEXT


# ---------------------------------------------------------------------------
# Project directory layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectPaths:
    """Immutable directory tree rooted at the project base."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    datasets: Path = field(init=False)
    original: Path = field(init=False)
    processed: Path = field(init=False)
    processed_small: Path = field(init=False)
    train_images: Path = field(init=False)
    train_labels: Path = field(init=False)
    val_images: Path = field(init=False)
    val_labels: Path = field(init=False)
    test_images: Path = field(init=False)
    test_labels: Path = field(init=False)
    weights: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", self.root / "datasets")
        object.__setattr__(self, "original", self.datasets / "original")
        object.__setattr__(self, "processed", self.datasets / "processed")
        object.__setattr__(self, "processed_small", self.datasets / "processed_small")
        object.__setattr__(self, "train_images", self.processed / "train" / "images")
        object.__setattr__(self, "train_labels", self.processed / "train" / "labels")
        object.__setattr__(self, "val_images", self.processed / "val" / "images")
        object.__setattr__(self, "val_labels", self.processed / "val" / "labels")
        object.__setattr__(self, "test_images", self.processed / "test" / "images")
        object.__setattr__(self, "test_labels", self.processed / "test" / "labels")
        object.__setattr__(self, "weights", self.root / "weights")

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        dirs = [
            self.processed,
            self.train_images, self.train_labels,
            self.val_images, self.val_labels,
            self.test_images, self.test_labels,
            self.weights,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetInfo:
    """Metadata describing a single source dataset."""

    name: str
    slug: str
    format: str
    split_dirs: dict[str, str] | None = None
    has_bbox: bool = True
    has_polygon: bool = False
    image_glob: str = "**/*.jpg"
    label_glob: str | None = "**/*.txt"


DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="COCO-Text",
        slug="cocotext",
        format="coco_json",
        has_bbox=True,
        has_polygon=True,
        image_glob="**/*.jpg",
        label_glob=None,
    ),
    DatasetInfo(
        name="ICDAR 2013",
        slug="icdar2013",
        format="yolo_txt",
        split_dirs={"train": "train", "val": "valid", "test": "test"},
    ),
    DatasetInfo(
        name="Vietnamese Scene Text",
        slug="vietnamese_scene",
        format="vintext_json",
        has_bbox=True,
        has_polygon=True,
        image_glob="**/*.jpg",
        label_glob=None,
    ),
    DatasetInfo(
        name="UIT-HWDB",
        slug="uit_hwdb",
        format="hwdb_image",
        has_bbox=False,
        has_polygon=False,
        image_glob="**/*.{jpg,png}",
        label_glob=None,
    ),
    DatasetInfo(
        name="Vietnamese Handwriting",
        slug="vietnamese_handwriting",
        format="hwdb_image",
        has_bbox=False,
        has_polygon=False,
        image_glob="**/*.{jpg,png}",
        label_glob=None,
    ),
]

DATASET_MAP: dict[str, DatasetInfo] = {ds.slug: ds for ds in DATASETS}


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
LABEL_EXTENSION: str = ".txt"
ANNOTATION_EXTENSIONS: tuple[str, ...] = (".txt", ".xml", ".json")


TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.20
TEST_RATIO: float = 0.10
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_project_paths() -> ProjectPaths:
    return ProjectPaths()


def get_class_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"unknown_{class_id}")


def get_dataset_by_slug(slug: str) -> DatasetInfo | None:
    return DATASET_MAP.get(slug)


def list_datasets() -> list[DatasetInfo]:
    return list(DATASETS)


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------

_paths = get_project_paths()
_paths.ensure_dirs()
