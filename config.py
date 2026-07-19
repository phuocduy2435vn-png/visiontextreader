"""
config.py — Central configuration for VisionTextReader data engineering pipeline.

Defines project paths, dataset registry, class mappings, and pipeline parameters.
All paths use pathlib for cross-platform compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("visiontextreader.config")


# ---------------------------------------------------------------------------
# Class definitions — unified label schema for text detection
# ---------------------------------------------------------------------------

class TextClass(IntEnum):
    """Unified class IDs for text detection across all datasets.

    Every dataset is normalised to a single class: 'text'.
    Adding future classes (e.g. handwritten, sign, digital) is done here.
    """
    TEXT = 0


# Friendly name mapping for YOLO data.yaml
CLASS_NAMES: Dict[int, str] = {cls.value: cls.name.lower() for cls in TextClass}

NUM_CLASSES: int = len(TextClass)

# Convenience constant for the default class ID
TEXT_CLASS: int = TextClass.TEXT


# ---------------------------------------------------------------------------
# Project directory layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectPaths:
    """Immutable directory tree rooted at the project base.

    Attributes:
        root: Project root (VisionTextReader/)
        datasets: datasets/ — holds raw sources and processed output
        raw: datasets/raw/ — downloaded / extracted source datasets
        processed: datasets/processed/ — unified YOLO output after conversion
        train_images / train_labels: YOLO train split
        val_images / val_labels: YOLO validation split
        test_images / test_labels: YOLO test split
        scripts: scripts/ — pipeline scripts
    """

    root: Path = Path(__file__).resolve().parent
    datasets: Path = field(init=False)
    raw: Path = field(init=False)
    processed: Path = field(init=False)
    train_images: Path = field(init=False)
    train_labels: Path = field(init=False)
    val_images: Path = field(init=False)
    val_labels: Path = field(init=False)
    test_images: Path = field(init=False)
    test_labels: Path = field(init=False)
    scripts: Path = field(init=False)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "datasets", self.root / "datasets")
        object.__setattr__(self, "raw", self.datasets / "raw")
        object.__setattr__(self, "processed", self.datasets / "processed")
        object.__setattr__(self, "train_images", self.processed / "train" / "images")
        object.__setattr__(self, "train_labels", self.processed / "train" / "labels")
        object.__setattr__(self, "val_images", self.processed / "val" / "images")
        object.__setattr__(self, "val_labels", self.processed / "val" / "labels")
        object.__setattr__(self, "test_images", self.processed / "test" / "images")
        object.__setattr__(self, "test_labels", self.processed / "test" / "labels")
        object.__setattr__(self, "scripts", self.root / "scripts")

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        dirs = [
            self.processed,
            self.train_images, self.train_labels,
            self.val_images, self.val_labels,
            self.test_images, self.test_labels,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Output directories verified under %s", self.processed)


# ---------------------------------------------------------------------------
# Dataset registry — metadata for each supported source dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetInfo:
    """Metadata describing a single source dataset.

    Attributes:
        name: Human-readable name (e.g. "COCO-Text")
        slug: Directory name under datasets/raw/
        format: Annotation format key — used to select the right parser
        split_dirs: Mapping from split name to subdirectory within raw/<slug>/
                    If None, the dataset has no built-in split structure.
        has_bbox: True if annotations contain bounding boxes.
        has_polygon: True if annotations contain polygon coords (need bbox conversion).
        image_glob: Glob pattern to find images relative to raw/<slug>/.
        label_glob: Glob pattern to find label files relative to raw/<slug>/.
    """

    name: str
    slug: str
    format: str  # "coco_json", "yolo_txt", "vintext_json", "hwdb_image", "icdar_xml"
    split_dirs: Optional[Dict[str, str]] = None
    has_bbox: bool = True
    has_polygon: bool = False
    image_glob: str = "**/*.jpg"
    label_glob: str = "**/*.txt"


# Supported datasets — add new datasets here
DATASETS: List[DatasetInfo] = [
    DatasetInfo(
        name="COCO-Text",
        slug="cocotext",
        format="coco_json",
        split_dirs=None,  # COCO-Text has train/test splits embedded in JSON
        has_bbox=True,
        has_polygon=True,
        image_glob="**/*.jpg",
        label_glob=None,  # labels come from JSON, not separate files
    ),
    DatasetInfo(
        name="ICDAR 2013",
        slug="icdar2013",
        format="yolo_txt",
        split_dirs={
            "train": "train",
            "val": "valid",
            "test": "test",
        },
        has_bbox=True,
        has_polygon=False,
    ),
    DatasetInfo(
        name="Vietnamese Scene Text",
        slug="vietnamese_scene",
        format="vintext_json",
        split_dirs=None,  # splits defined in *_data.txt files
        has_bbox=True,
        has_polygon=True,
        image_glob="**/*.jpg",
        label_glob=None,
    ),
    DatasetInfo(
        name="UIT-HWDB",
        slug="uit_hwdb",
        format="hwdb_image",
        split_dirs=None,
        has_bbox=False,  # word/line crops — full-image bbox
        has_polygon=False,
        image_glob="**/*.{jpg,png}",
        label_glob=None,
    ),
    DatasetInfo(
        name="Vietnamese Handwriting",
        slug="vietnamese_handwriting",
        format="hwdb_image",
        split_dirs=None,
        has_bbox=False,
        has_polygon=False,
        image_glob="**/*.{jpg,png}",
        label_glob=None,
    ),
]

# Quick lookup by slug
DATASET_MAP: Dict[str, DatasetInfo] = {ds.slug: ds for ds in DATASETS}


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------

# Image extensions accepted by the pipeline
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# YOLO label file extension
LABEL_EXTENSION: str = ".txt"

# Supported annotation file extensions
ANNOTATION_EXTENSIONS: tuple[str, ...] = (".txt", ".xml", ".json")

# Supported coordinate formats
class CoordFormat:
    """Coordinate format constants."""
    ABSOLUTE = "absolute"      # pixels: [x, y, w, h]
    NORMALIZED = "normalized"  # YOLO: [cx, cy, w, h] in [0, 1]
    POLYGON = "polygon"        # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


# Split ratios for train/val/test
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.20
TEST_RATIO: float = 0.10
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_project_paths() -> ProjectPaths:
    """Return the singleton ProjectPaths instance."""
    return ProjectPaths()


def get_class_name(class_id: int) -> str:
    """Return the human-readable name for a class ID."""
    return CLASS_NAMES.get(class_id, f"unknown_{class_id}")


def get_dataset_by_slug(slug: str) -> Optional[DatasetInfo]:
    """Look up a DatasetInfo by its directory slug."""
    return DATASET_MAP.get(slug)


def list_datasets() -> List[DatasetInfo]:
    """Return all registered dataset descriptors."""
    return list(DATASETS)


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------

_paths = get_project_paths()
_paths.ensure_dirs()

logger.info(
    "Config loaded — %d classes, %d datasets registered, output root: %s",
    NUM_CLASSES,
    len(DATASETS),
    _paths.processed,
)
