"""
dataset_converter.py — Convert raw datasets to YOLO format.

Supports COCO JSON, ICDAR TXT, Pascal VOC XML, and custom TXT formats.
Outputs unified YOLO format: class_id cx cy w h (normalized).
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger("visiontextreader.dataset_converter")


class DatasetConverter:
    """Convert various annotation formats to YOLO format.

    Supported input formats:
        - coco_json: COCO-style JSON annotations
        - yolo_txt: Existing YOLO format (copy through)
        - icdar_xml: Pascal VOC / ICDAR XML
        - txt_xywh: Class x y w h (absolute pixels)
        - txt_xyxy: Class x1 y1 x2 y2 (absolute pixels)
    """

    def __init__(self, output_dir: Path, class_names: Optional[Dict[int, str]] = None):
        """Initialize converter.

        Args:
            output_dir: Output directory for YOLO dataset.
            class_names: Mapping of class_id to class_name.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or {0: "text"}
        self._class_map: Dict[str, int] = {name: idx for idx, name in self.class_names.items()}

    def convert_coco_json(
        self,
        json_path: Path,
        image_dir: Path,
        output_split: str = "train",
    ) -> int:
        """Convert COCO JSON annotations to YOLO format.

        Args:
            json_path: Path to COCO JSON annotation file.
            image_dir: Directory containing images.
            output_split: Output split name (train/val/test).

        Returns:
            Number of annotations converted.
        """
        logger.info("Converting COCO JSON: %s", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build image lookup
        images = {img["id"]: img for img in coco.get("images", [])}

        # Group annotations by image
        ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id is not None:
                ann_by_image.setdefault(img_id, []).append(ann)

        out_img_dir = self.output_dir / output_split / "images"
        out_lbl_dir = self.output_dir / output_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_id, img_info in images.items():
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            img_w = img_info.get("width", 0)
            img_h = img_info.get("height", 0)
            if img_w <= 0 or img_h <= 0:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

            # Copy image
            dst_img = out_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Convert annotations
            lbl_lines = []
            for ann in ann_by_image.get(img_id, []):
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                # COCO category_id to class name
                cat_id = ann.get("category_id", 0)
                cat_name = "text"  # Default
                for cat in coco.get("categories", []):
                    if cat["id"] == cat_id:
                        cat_name = cat.get("name", "text")
                        break

                class_id = self._class_map.get(cat_name, 0)

                # Normalize to YOLO format
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                # Clamp
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                lbl_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                count += 1

            # Write label
            lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
            lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")

        logger.info("Converted %d annotations from COCO JSON", count)
        return count

    def convert_yolo_txt(
        self,
        source_images: Path,
        source_labels: Path,
        output_split: str = "train",
    ) -> int:
        """Copy through existing YOLO format dataset.

        Args:
            source_images: Directory with source images.
            source_labels: Directory with source YOLO labels.
            output_split: Output split name.

        Returns:
            Number of annotations converted.
        """
        logger.info("Copying YOLO format: %s", source_images)
        out_img_dir = self.output_dir / output_split / "images"
        out_lbl_dir = self.output_dir / output_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in source_images.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            lbl_path = source_labels / f"{img_path.stem}.txt"

            # Copy image
            dst_img = out_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy label if exists
            if lbl_path.exists():
                dst_lbl = out_lbl_dir / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)
                # Count annotations
                try:
                    content = lbl_path.read_text(encoding="utf-8").strip()
                    if content:
                        count += len(content.splitlines())
                except Exception:
                    pass

        logger.info("Copied %d annotations from YOLO format", count)
        return count

    def convert_icdar_xml(
        self,
        xml_dir: Path,
        image_dir: Path,
        output_split: str = "train",
    ) -> int:
        """Convert Pascal VOC / ICDAR XML annotations to YOLO format.

        Args:
            xml_dir: Directory with XML annotation files.
            image_dir: Directory with images.
            output_split: Output split name.

        Returns:
            Number of annotations converted.
        """
        logger.info("Converting ICDAR XML: %s", xml_dir)
        out_img_dir = self.output_dir / output_split / "images"
        out_lbl_dir = self.output_dir / output_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for xml_path in xml_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Get image info
                filename = root.findtext("filename", "")
                size = root.find("size")
                if size is None:
                    continue
                img_w = int(size.findtext("width", "0"))
                img_h = int(size.findtext("height", "0"))
                if img_w <= 0 or img_h <= 0:
                    continue

                # Copy image
                img_path = image_dir / filename
                if img_path.exists():
                    dst_img = out_img_dir / img_path.name
                    if not dst_img.exists():
                        shutil.copy2(img_path, dst_img)

                # Convert objects
                lbl_lines = []
                for obj in root.findall("object"):
                    name = obj.findtext("name", "text")
                    class_id = self._class_map.get(name, 0)

                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue
                    xmin = float(bndbox.findtext("xmin", "0"))
                    ymin = float(bndbox.findtext("ymin", "0"))
                    xmax = float(bndbox.findtext("xmax", "0"))
                    ymax = float(bndbox.findtext("ymax", "0"))

                    cx = ((xmin + xmax) / 2) / img_w
                    cy = ((ymin + ymax) / 2) / img_h
                    w = (xmax - xmin) / img_w
                    h = (ymax - ymin) / img_h

                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))

                    lbl_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    count += 1

                # Write label
                lbl_path = out_lbl_dir / f"{xml_path.stem}.txt"
                lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")

            except Exception as exc:
                logger.warning("Error processing XML %s: %s", xml_path, exc)

        logger.info("Converted %d annotations from ICDAR XML", count)
        return count

    def convert_txt_absolute(
        self,
        txt_path: Path,
        image_dir: Path,
        output_split: str = "train",
        format: str = "xywh",
    ) -> int:
        """Convert TXT file with absolute coordinates to YOLO format.

        Expected line format:
            class_name x y w h    (xywh format)
            class_name x1 y1 x2 y2 (xyxy format)

        Args:
            txt_path: Path to annotation TXT file.
            image_dir: Directory with images.
            output_split: Output split name.
            format: 'xywh' or 'xyxy'.

        Returns:
            Number of annotations converted.
        """
        logger.info("Converting TXT absolute: %s", txt_path)
        out_img_dir = self.output_dir / output_split / "images"
        out_lbl_dir = self.output_dir / output_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        current_img = None
        lbl_lines: List[str] = []

        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 1:
                # Image filename line
                if current_img and lbl_lines:
                    lbl_path = out_lbl_dir / f"{Path(current_img).stem}.txt"
                    lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")
                current_img = parts[0]
                lbl_lines = []

                # Copy image
                img_path = image_dir / current_img
                if img_path.exists():
                    dst_img = out_img_dir / img_path.name
                    if not dst_img.exists():
                        shutil.copy2(img_path, dst_img)
            elif len(parts) >= 5:
                # Annotation line: class x y w h
                class_name = parts[0]
                class_id = self._class_map.get(class_name, 0)

                try:
                    coords = [float(x) for x in parts[1:5]]
                except ValueError:
                    continue

                if current_img is None:
                    continue

                # Get image dimensions
                img_path = image_dir / current_img
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                if format == "xywh":
                    x, y, w, h = coords
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                else:  # xyxy
                    x1, y1, x2, y2 = coords
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    nw = (x2 - x1) / img_w
                    nh = (y2 - y1) / img_h

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                lbl_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                count += 1

        # Write last label
        if current_img and lbl_lines:
            lbl_path = out_lbl_dir / f"{Path(current_img).stem}.txt"
            lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")

        logger.info("Converted %d annotations from TXT", count)
        return count

    def create_data_yaml(self, split_ratios: Optional[Dict[str, float]] = None) -> Path:
        """Create YOLO data.yaml configuration file.

        Args:
            split_ratios: Optional split ratios (not used, just for reference).

        Returns:
            Path to created data.yaml.
        """
        yaml_content = f"""# VisionTextReader YOLO Dataset Configuration
# Auto-generated by dataset_converter.py

train: {self.output_dir}/train/images
val: {self.output_dir}/val/images
test: {self.output_dir}/test/images

nc: {len(self.class_names)}
names: {list(self.class_names.values())}
"""
        yaml_path = self.output_dir / "data.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        logger.info("Created data.yaml at %s", yaml_path)
        return yaml_path
