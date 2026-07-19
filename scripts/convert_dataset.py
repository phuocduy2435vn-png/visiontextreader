"""
convert_dataset.py — Unified dataset converter for VisionTextReader.

Reads multiple source dataset formats (COCO JSON, ICDAR YOLO TXT,
Vietnamese Scene Text JSON, UIT-HWDB image crops) and converts them
into a single YOLO-format output under datasets/processed/.

Usage:
    python scripts/convert_dataset.py                     # convert all datasets
    python scripts/convert_dataset.py --dataset cocotext  # convert one dataset
    python scripts/convert_dataset.py --list              # list available datasets

Each dataset has its own converter class inheriting from BaseConverter.
Adding a new dataset = add a DatasetInfo in config.py + implement a converter.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Add project root to sys.path so config can be imported when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSION,
    TEXT_CLASS,
    DatasetInfo,
    ProjectPaths,
    get_project_paths,
    logger as root_logger,
)

# Module-level logger
logger = logging.getLogger("visiontextreader.convert")


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class BBox:
    """Bounding box in YOLO normalised format: (cx, cy, w, h) all in [0, 1].

    Attributes:
        cx: Centre x-coordinate, normalised.
        cy: Centre y-coordinate, normalised.
        width: Box width, normalised.
        height: Box height, normalised.
        class_id: Class label ID (default TEXT = 0).
    """
    cx: float
    cy: float
    width: float
    height: float
    class_id: int = TEXT_CLASS

    def to_yolo_line(self) -> str:
        """Return a single YOLO-format label line."""
        return (
            f"{self.class_id} "
            f"{self.cx:.6f} {self.cy:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    def is_valid(self) -> bool:
        """Check that all coordinates are within [0, 1]."""
        return all(0.0 <= v <= 1.0 for v in (self.cx, self.cy, self.width, self.height))


@dataclass
class ConversionResult:
    """Statistics produced by a single converter run."""
    dataset_name: str
    images_found: int = 0
    images_copied: int = 0
    labels_written: int = 0
    labels_skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== {self.dataset_name} ===",
            f"  Images found:     {self.images_found}",
            f"  Images copied:    {self.images_copied}",
            f"  Labels written:   {self.labels_written}",
            f"  Labels skipped:   {self.labels_skipped}",
        ]
        if self.errors:
            lines.append(f"  Errors:           {len(self.errors)}")
            for err in self.errors[:10]:  # show first 10
                lines.append(f"    - {err}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more")
        return "\n".join(lines)


# =========================================================================
# Coordinate conversion helpers
# =========================================================================

def polygon_to_bbox(
    points: Sequence[Tuple[float, float]],
    img_width: int,
    img_height: int,
) -> Optional[BBox]:
    """Convert a 4-point polygon to a normalised YOLO bounding box.

    Args:
        points: List of (x, y) tuples in absolute pixel coords.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        BBox in YOLO format, or None if conversion fails.
    """
    if len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Guard against zero-size or out-of-bounds boxes
    if x_max <= x_min or y_max <= y_min:
        return None
    if img_width <= 0 or img_height <= 0:
        return None

    # Centre and size in normalised coords
    cx = ((x_min + x_max) / 2.0) / img_width
    cy = ((y_min + y_max) / 2.0) / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return BBox(cx=cx, cy=cy, width=w, height=h)


def abs_bbox_to_yolo(
    x: float, y: float, w: float, h: float,
    img_width: int, img_height: int,
) -> Optional[BBox]:
    """Convert absolute (x, y, w, h) bbox to normalised YOLO format.

    Args:
        x, y: Top-left corner in pixels.
        w, h: Width and height in pixels.
        img_width, img_height: Image dimensions.

    Returns:
        Normalised BBox, or None on invalid input.
    """
    if img_width <= 0 or img_height <= 0:
        return None
    if w <= 0 or h <= 0:
        return None

    cx = ((x + w / 2.0) / img_width)
    cy = ((y + h / 2.0) / img_height)
    nw = w / img_width
    nh = h / img_height

    # Clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return BBox(cx=cx, cy=cy, width=nw, height=nh)


def full_image_bbox(img_width: int, img_height: int) -> BBox:
    """Create a bounding box covering the entire image (for word/line crops).

    Used for datasets like UIT-HWDB where each image IS the text region.
    """
    return BBox(cx=0.5, cy=0.5, width=1.0, height=1.0)


# =========================================================================
# Image utility helpers
# =========================================================================

def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Read image dimensions without heavy dependencies.

    Uses a lightweight check of JPEG/PNG headers to avoid requiring
    PIL or opencv just for dimension reading.

    Args:
        image_path: Path to the image file.

    Returns:
        (width, height) tuple.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If dimensions cannot be determined.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except ImportError:
        pass

    # Fallback: read PNG / JPEG header manually
    try:
        with open(image_path, "rb") as f:
            header = f.read(32)

            # PNG
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                import struct
                w, h = struct.unpack(">II", header[16:24])
                return w, h

            # JPEG — need to scan for SOF marker
            f.seek(0)
            data = f.read()
            i = 0
            while i < len(data) - 1:
                if data[i] == 0xFF:
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC1, 0xC2):
                        import struct
                        h = struct.unpack(">H", data[i + 5:i + 7])[0]
                        w = struct.unpack(">H", data[i + 7:i + 9])[0]
                        return w, h
                    elif marker == 0xD9:
                        break
                    elif marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0x01):
                        i += 2
                    else:
                        length = struct.unpack(">H", data[i + 2:i + 4])[0]
                        i += 2 + length
                else:
                    i += 1
    except Exception:
        pass

    raise ValueError(f"Cannot determine dimensions for: {image_path}")


def copy_image(src: Path, dst: Path) -> bool:
    """Copy an image file to the destination, avoiding overwrite.

    If dst already exists, it's assumed to be the same image (skip copy).

    Returns:
        True if copied, False if skipped (already exists).
    """
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


def write_yolo_label(label_path: Path, boxes: List[BBox]) -> None:
    """Write a YOLO-format label file.

    Each line: ``class_id cx cy w h`` (normalised, space-separated).
    """
    lines = [box.to_yolo_line() for box in boxes if box.is_valid()]
    label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def find_images(root: Path, glob_pattern: str = "**/*.jpg") -> List[Path]:
    """Find all image files under *root* matching common extensions.

    Tries the provided glob first, then falls back to IMAGE_EXTENSIONS.
    """
    results = list(root.glob(glob_pattern))
    if not results:
        # Fallback: search all known extensions
        for ext in IMAGE_EXTENSIONS:
            results.extend(root.glob(f"**/*{ext}"))
    return sorted(set(results))


# =========================================================================
# Base converter
# =========================================================================

class BaseConverter(ABC):
    """Abstract base for all dataset converters.

    Subclasses implement ``convert()`` which reads the source format
    and writes YOLO labels + copied images into ``self.out_images`` /
    ``self.out_labels``.
    """

    def __init__(
        self,
        dataset_info: DatasetInfo,
        paths: ProjectPaths,
        source_dir: Optional[Path] = None,
    ) -> None:
        self.info = dataset_info
        self.paths = paths
        self.source_dir = source_dir or (paths.raw / dataset_info.slug)
        self.result = ConversionResult(dataset_name=dataset_info.name)

    @abstractmethod
    def convert(self) -> ConversionResult:
        """Run the full conversion. Must be implemented by subclasses."""
        ...

    def _image_stem(self, image_path: Path) -> str:
        """Return a unique stem for an image, prefixed with dataset slug.

        Prevents filename collisions when merging multiple datasets.
        """
        return f"{self.info.slug}_{image_path.stem}"

    def _output_paths(self, stem: str) -> Tuple[Path, Path]:
        """Return (image_output_path, label_output_path) for a given stem."""
        img_dst = self.paths.train_images / f"{stem}{image_path_suffix(stem)}"
        lbl_dst = self.paths.train_labels / f"{stem}{LABEL_EXTENSION}"
        return img_dst, lbl_dst


def image_path_suffix(stem: str) -> str:
    """Infer a file extension from the stem or return .jpg as default."""
    # This is a heuristic — the actual extension comes from the source file.
    return ".jpg"


# =========================================================================
# COCO-Text converter
# =========================================================================

class COCOTextConverter(BaseConverter):
    """Convert COCO-Text dataset (JSON annotations) to YOLO format.

    COCO-Text structure:
        cocotext/
            cocotext.v2.json  — annotations + image metadata
            train2014/        — training images (COCO format)
            val2014/          — validation images (COCO format)

    The JSON contains:
        - ``anns``: annotation dict keyed by annotation ID
          Each annotation has: bbox [x,y,w,h], image_id, utf8_string, legibility
        - ``imgs``: image dict keyed by image ID
          Each image has: id, width, height, file_name
    """

    def convert(self) -> ConversionResult:
        json_path = self.source_dir / "cocotext.v2.json"
        if not json_path.exists():
            self.result.errors.append(f"COCO-Text JSON not found: {json_path}")
            return self.result

        logger.info("Loading COCO-Text annotations from %s", json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        anns = coco_data.get("anns", {})
        imgs = coco_data.get("imgs", {})

        # Build image-id → image-info lookup
        image_lookup: Dict[int, dict] = {}
        for img_id_str, img_info in imgs.items():
            img_id = int(img_id_str) if isinstance(img_id_str, str) else img_id_str
            image_lookup[img_id] = img_info

        # Group annotations by image_id
        anns_by_image: Dict[int, List[dict]] = {}
        for ann_id_str, ann in anns.items():
            img_id = ann.get("image_id")
            if img_id is None:
                continue
            # Only include legible text
            if ann.get("legibility") == "illegible":
                continue
            anns_by_image.setdefault(img_id, []).append(ann)

        self.result.images_found = len(anns_by_image)
        logger.info(
            "COCO-Text: %d images with annotations, %d total annotations",
            len(anns_by_image), len(anns),
        )

        for img_id, img_anns in anns_by_image.items():
            img_info = image_lookup.get(img_id)
            if img_info is None:
                self.result.errors.append(f"Image ID {img_id} not found in metadata")
                continue

            img_w = img_info.get("width", 0)
            img_h = img_info.get("height", 0)
            if img_w <= 0 or img_h <= 0:
                self.result.errors.append(
                    f"Invalid dimensions for image ID {img_id}: {img_w}x{img_h}"
                )
                continue

            # Locate the image file — try common COCO directory names
            file_name = img_info.get("file_name", "")
            image_path = self._find_coco_image(file_name, img_id)
            if image_path is None:
                self.result.errors.append(f"Image file not found for ID {img_id}: {file_name}")
                continue

            # Build YOLO bounding boxes
            boxes: List[BBox] = []
            for ann in img_anns:
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                yolo_box = abs_bbox_to_yolo(x, y, w, h, img_w, img_h)
                if yolo_box is not None:
                    boxes.append(yolo_box)

            if not boxes:
                self.result.labels_skipped += 1
                continue

            # Copy image and write label
            stem = self._image_stem(image_path)
            img_dst = self.paths.train_images / f"{stem}{image_path.suffix}"
            lbl_dst = self.paths.train_labels / f"{stem}{LABEL_EXTENSION}"

            if copy_image(image_path, img_dst):
                self.result.images_copied += 1
            write_yolo_label(lbl_dst, boxes)
            self.result.labels_written += 1

        return self.result

    def _find_coco_image(self, file_name: str, img_id: int) -> Optional[Path]:
        """Search for a COCO image file in common subdirectory layouts."""
        # Try direct path
        direct = self.source_dir / file_name
        if direct.exists():
            return direct

        # Try train2014 / val2014 subdirs
        for split_dir in ("train2014", "val2017", "val2014", "train2017"):
            candidate = self.source_dir / split_dir / file_name
            if candidate.exists():
                return candidate

        # Brute-force search by filename
        for ext in IMAGE_EXTENSIONS:
            matches = list(self.source_dir.rglob(f"*{Path(file_name).stem}{ext}"))
            if matches:
                return matches[0]

        return None


# =========================================================================
# ICDAR 2013 converter (Roboflow format — already YOLO)
# =========================================================================

class ICDAR2013Converter(BaseConverter):
    """Convert ICDAR 2013 dataset (Roboflow export, already YOLO format).

    The dataset ships in ``train/``, ``valid/``, ``test/`` subdirectories,
    each with ``images/`` and ``labels/`` already in YOLO format.

    This converter copies images and labels into the processed output,
    prefixing stems with the dataset slug to avoid collisions.
    """

    def convert(self) -> ConversionResult:
        if not self.source_dir.exists():
            self.result.errors.append(f"ICDAR2013 source not found: {self.source_dir}")
            return self.result

        split_dirs = self.info.split_dirs or {}
        for split_name, sub_dir in split_dirs.items():
            split_path = self.source_dir / sub_dir
            images_dir = split_path / "images"
            labels_dir = split_path / "labels"

            if not images_dir.exists():
                self.result.errors.append(f"Images dir not found: {images_dir}")
                continue

            logger.info("ICDAR2013: processing split '%s' from %s", split_name, split_path)

            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                self.result.images_found += 1

                # Find matching label file
                lbl_path = labels_dir / f"{img_path.stem}{LABEL_EXTENSION}"
                if not lbl_path.exists():
                    self.result.labels_skipped += 1
                    continue

                # Read existing YOLO labels (they're already normalised)
                boxes = self._read_yolo_labels(lbl_path)
                if not boxes:
                    self.result.labels_skipped += 1
                    continue

                # Copy to processed with slug prefix
                stem = self._image_stem(img_path)
                img_dst = self.paths.train_images / f"{stem}{img_path.suffix}"
                lbl_dst = self.paths.train_labels / f"{stem}{LABEL_EXTENSION}"

                if copy_image(img_path, img_dst):
                    self.result.images_copied += 1
                write_yolo_label(lbl_dst, boxes)
                self.result.labels_written += 1

        return self.result

    def _read_yolo_labels(self, label_path: Path) -> List[BBox]:
        """Read a YOLO-format label file and return list of BBox."""
        boxes: List[BBox] = []
        try:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                return boxes
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append(BBox(cx=cx, cy=cy, width=w, height=h, class_id=class_id))
        except (ValueError, IOError) as exc:
            self.result.errors.append(f"Error reading {label_path}: {exc}")
        return boxes


# =========================================================================
# Vietnamese Scene Text converter
# =========================================================================

class VintextConverter(BaseConverter):
    """Convert Vietnamese Scene Text dataset to YOLO format.

    Source format:
        vietnamese_scene/
            text_recognition_data/text_recognition_data/
                train_images/     — training images
                test_images/      — test images
                train_data.txt    — tab-separated: path<TAB>label
                test_data.txt
                val_data.txt
            test_vintext_label.txt — JSON annotations with polygons

    Each line of test_vintext_label.txt:
        <image_path>\t[{"transcription": "...", "points": [[x1,y1],...]}]
    """

    def convert(self) -> ConversionResult:
        if not self.source_dir.exists():
            self.result.errors.append(f"Vietnamese Scene source not found: {self.source_dir}")
            return self.result

        # Try to load the detailed annotation file (with polygons)
        label_file = self.source_dir / "test_vintext_label.txt"
        poly_annotations: Dict[str, List[dict]] = {}
        if label_file.exists():
            poly_annotations = self._load_polygon_labels(label_file)
            logger.info("Vintext: loaded polygon annotations for %d images", len(poly_annotations))

        # Process train/test/val data files
        data_dir = self.source_dir / "text_recognition_data" / "text_recognition_data"
        if not data_dir.exists():
            # Fallback: search deeper
            data_dirs = list(self.source_dir.rglob("train_images"))
            if data_dirs:
                data_dir = data_dirs[0].parent
            else:
                self.result.errors.append(f"Cannot find data directory under {self.source_dir}")
                return self.result

        data_files = {
            "train": data_dir / "train_data.txt",
            "test": data_dir / "test_data.txt",
            "val": data_dir / "val_data.txt",
        }

        for split_name, data_file in data_files.items():
            if not data_file.exists():
                logger.warning("Vintext: data file not found: %s", data_file)
                continue

            logger.info("Vintext: processing split '%s' from %s", split_name, data_file)
            entries = self._parse_data_file(data_file, data_dir)

            for img_rel_path, _text_label in entries:
                # Resolve full image path
                img_path = data_dir / img_rel_path
                if not img_path.exists():
                    self.result.errors.append(f"Image not found: {img_path}")
                    continue

                self.result.images_found += 1

                # Get image dimensions
                try:
                    img_w, img_h = get_image_dimensions(img_path)
                except (FileNotFoundError, ValueError) as exc:
                    self.result.errors.append(str(exc))
                    continue

                # Build bounding boxes
                boxes: List[BBox] = []

                # Check polygon annotations
                # Try various key formats (with/without path prefix)
                ann_key = img_rel_path.replace("\\", "/")
                ann_entries = poly_annotations.get(ann_key, [])

                # Also try just the filename
                if not ann_entries:
                    for key, val in poly_annotations.items():
                        if key.endswith(img_path.name):
                            ann_entries = val
                            break

                if ann_entries:
                    for entry in ann_entries:
                        points = entry.get("points", [])
                        if len(points) >= 2:
                            # Convert list of [x,y] to tuple list
                            pt_tuples = [(float(p[0]), float(p[1])) for p in points]
                            yolo_box = polygon_to_bbox(pt_tuples, img_w, img_h)
                            if yolo_box is not None:
                                boxes.append(yolo_box)

                # Fallback: if no polygon annotations, use full-image bbox
                if not boxes:
                    boxes.append(full_image_bbox(img_w, img_h))

                # Copy image and write label
                stem = self._image_stem(img_path)
                img_dst = self.paths.train_images / f"{stem}{img_path.suffix}"
                lbl_dst = self.paths.train_labels / f"{stem}{LABEL_EXTENSION}"

                if copy_image(img_path, img_dst):
                    self.result.images_copied += 1
                write_yolo_label(lbl_dst, boxes)
                self.result.labels_written += 1

        return self.result

    def _parse_data_file(
        self, data_file: Path, base_dir: Path,
    ) -> List[Tuple[str, str]]:
        """Parse a tab-separated data file (path<TAB>label)."""
        entries: List[Tuple[str, str]] = []
        try:
            for line in data_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    entries.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    entries.append((parts[0], ""))
        except IOError as exc:
            self.result.errors.append(f"Error reading {data_file}: {exc}")
        return entries

    def _load_polygon_labels(self, label_file: Path) -> Dict[str, List[dict]]:
        """Load the polygon annotation file.

        Returns:
            Dict mapping image path (relative) to list of annotation dicts.
        """
        annotations: Dict[str, List[dict]] = {}
        try:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    continue
                img_path = parts[0]
                try:
                    ann_list = json.loads(parts[1])
                    if isinstance(ann_list, list):
                        annotations[img_path] = ann_list
                except json.JSONDecodeError:
                    self.result.errors.append(f"Invalid JSON in label file for {img_path}")
        except IOError as exc:
            self.result.errors.append(f"Error reading {label_file}: {exc}")
        return annotations


# =========================================================================
# UIT-HWDB / Vietnamese Handwriting converter
# =========================================================================

class HWDBConverter(BaseConverter):
    """Convert UIT-HWDB (or similar handwriting crop datasets) to YOLO format.

    These datasets consist of individual word/line images without explicit
    bounding box annotations — each image IS the text region. We create
    a full-image bounding box (covering 100% of the image) for each.

    Expected structure (flexible):
        uit_hwdb/
            images/ or words/ or lines/ or paragraphs/
                *.jpg or *.png

    The converter searches recursively for image files.
    """

    def convert(self) -> ConversionResult:
        if not self.source_dir.exists():
            self.result.errors.append(f"HWDB source not found: {self.source_dir}")
            return self.result

        # Find all image files recursively
        image_files = find_images(self.source_dir)
        self.result.images_found = len(image_files)

        logger.info(
            "HWDB (%s): found %d images",
            self.info.slug, len(image_files),
        )

        for img_path in image_files:
            try:
                img_w, img_h = get_image_dimensions(img_path)
            except (FileNotFoundError, ValueError) as exc:
                self.result.errors.append(str(exc))
                continue

            # Full-image bounding box
            box = full_image_bbox(img_w, img_h)

            # Copy and label
            stem = self._image_stem(img_path)
            img_dst = self.paths.train_images / f"{stem}{img_path.suffix}"
            lbl_dst = self.paths.train_labels / f"{stem}{LABEL_EXTENSION}"

            if copy_image(img_path, img_dst):
                self.result.images_copied += 1
            write_yolo_label(lbl_dst, [box])
            self.result.labels_written += 1

        return self.result


# =========================================================================
# Converter registry
# =========================================================================

# Maps format string → converter class
CONVERTER_REGISTRY: Dict[str, type] = {
    "coco_json": COCOTextConverter,
    "yolo_txt": ICDAR2013Converter,
    "vintext_json": VintextConverter,
    "hwdb_image": HWDBConverter,
}


def get_converter(
    dataset_info: DatasetInfo,
    paths: ProjectPaths,
) -> BaseConverter:
    """Instantiate the appropriate converter for a dataset.

    Args:
        dataset_info: Metadata about the source dataset.
        paths: Project directory layout.

    Returns:
        A BaseConverter subclass instance.

    Raises:
        ValueError: If no converter is registered for the dataset format.
    """
    converter_cls = CONVERTER_REGISTRY.get(dataset_info.format)
    if converter_cls is None:
        raise ValueError(
            f"No converter registered for format '{dataset_info.format}' "
            f"(dataset: {dataset_info.name})"
        )
    return converter_cls(dataset_info=dataset_info, paths=paths)


# =========================================================================
# Main pipeline
# =========================================================================

def convert_all(datasets: Optional[List[str]] = None) -> List[ConversionResult]:
    """Run conversion for all (or selected) datasets.

    Args:
        datasets: Optional list of dataset slugs to convert.
                  If None, converts all registered datasets.

    Returns:
        List of ConversionResult, one per dataset.
    """
    from config import DATASETS

    paths = get_project_paths()
    results: List[ConversionResult] = []

    targets = DATASETS
    if datasets:
        targets = [ds for ds in DATASETS if ds.slug in datasets]
        if not targets:
            logger.error("No matching datasets found for: %s", datasets)
            return results

    for ds_info in targets:
        logger.info("=" * 60)
        logger.info("Converting dataset: %s (format: %s)", ds_info.name, ds_info.format)
        logger.info("=" * 60)

        try:
            converter = get_converter(ds_info, paths)
            result = converter.convert()
            results.append(result)
            logger.info("\n%s", result.summary())
        except Exception as exc:
            logger.exception("Failed to convert %s: %s", ds_info.name, exc)
            results.append(ConversionResult(
                dataset_name=ds_info.name,
                errors=[str(exc)],
            ))

    return results


def print_conversion_report(results: List[ConversionResult]) -> None:
    """Print a consolidated report of all conversion results."""
    print("\n" + "=" * 70)
    print("  DATASET CONVERSION REPORT")
    print("=" * 70)

    total_images = 0
    total_labels = 0
    total_errors = 0

    for r in results:
        print(r.summary())
        print()
        total_images += r.images_copied
        total_labels += r.labels_written
        total_errors += len(r.errors)

    print("-" * 70)
    print(f"  TOTAL: {total_images} images, {total_labels} labels, {total_errors} errors")
    print("=" * 70)


# =========================================================================
# CLI entry point
# =========================================================================

def main() -> None:
    """Command-line interface for the conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert raw datasets to YOLO format for VisionTextReader.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        nargs="*",
        help="Dataset slug(s) to convert (default: all). Use --list to see options.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all registered datasets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        from config import DATASETS
        print("\nRegistered datasets:")
        for ds in DATASETS:
            print(f"  {ds.slug:<25} {ds.name:<30} format={ds.format}")
        return

    results = convert_all(datasets=args.dataset)
    print_conversion_report(results)


if __name__ == "__main__":
    main()
