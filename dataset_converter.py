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

import cv2

logger = logging.getLogger("visiontextreader.dataset_converter")


class DatasetConverter:
    """Convert various annotation formats to YOLO format."""

    def __init__(self, output_dir: Path, class_names: dict[int, str] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or {0: "text"}
        self._class_map: dict[str, int] = {name: idx for idx, name in self.class_names.items()}

    def convert_coco_json(
        self,
        json_path: Path,
        image_dir: Path,
        output_split: str = "train",
    ) -> int:
        """Convert COCO JSON annotations to YOLO format. Returns annotation count."""
        logger.info("Converting COCO JSON: %s", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco.get("images", [])}
        ann_by_image: dict[int, list[dict]] = {}
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

            dst_img = out_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            lbl_lines: list[str] = []
            for ann in ann_by_image.get(img_id, []):
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                cat_id = ann.get("category_id", 0)
                cat_name = "text"
                for cat in coco.get("categories", []):
                    if cat["id"] == cat_id:
                        cat_name = cat.get("name", "text")
                        break

                class_id = self._class_map.get(cat_name, 0)
                cx = max(0.0, min(1.0, (x + w / 2) / img_w))
                cy = max(0.0, min(1.0, (y + h / 2) / img_h))
                nw = max(0.0, min(1.0, w / img_w))
                nh = max(0.0, min(1.0, h / img_h))

                lbl_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                count += 1

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
        """Copy through existing YOLO format dataset. Returns annotation count."""
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

            dst_img = out_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            if lbl_path.exists():
                dst_lbl = out_lbl_dir / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)
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
        """Convert Pascal VOC / ICDAR XML annotations to YOLO format."""
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

                filename = root.findtext("filename", "")
                size = root.find("size")
                if size is None:
                    continue
                img_w = int(size.findtext("width", "0"))
                img_h = int(size.findtext("height", "0"))
                if img_w <= 0 or img_h <= 0:
                    continue

                img_path = image_dir / filename
                if img_path.exists():
                    dst_img = out_img_dir / img_path.name
                    if not dst_img.exists():
                        shutil.copy2(img_path, dst_img)

                lbl_lines: list[str] = []
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

                    cx = max(0.0, min(1.0, ((xmin + xmax) / 2) / img_w))
                    cy = max(0.0, min(1.0, ((ymin + ymax) / 2) / img_h))
                    w = max(0.0, min(1.0, (xmax - xmin) / img_w))
                    h = max(0.0, min(1.0, (ymax - ymin) / img_h))

                    lbl_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    count += 1

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
        """Convert TXT with absolute coordinates to YOLO format.

        Line format: class_name x y w h (xywh) or class_name x1 y1 x2 y2 (xyxy).
        """
        logger.info("Converting TXT absolute: %s", txt_path)
        out_img_dir = self.output_dir / output_split / "images"
        out_lbl_dir = self.output_dir / output_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        current_img: str | None = None
        lbl_lines: list[str] = []
        img_dims: dict[str, tuple[int, int]] = {}

        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 1:
                if current_img and lbl_lines:
                    lbl_path = out_lbl_dir / f"{Path(current_img).stem}.txt"
                    lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")
                current_img = parts[0]
                lbl_lines = []

                img_path = image_dir / current_img
                if img_path.exists():
                    dst_img = out_img_dir / img_path.name
                    if not dst_img.exists():
                        shutil.copy2(img_path, dst_img)
            elif len(parts) >= 5:
                class_name = parts[0]
                class_id = self._class_map.get(class_name, 0)

                try:
                    coords = [float(x) for x in parts[1:5]]
                except ValueError:
                    continue

                if current_img is None:
                    continue

                if current_img not in img_dims:
                    img_path = image_dir / current_img
                    if not img_path.exists():
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img_dims[current_img] = (img.shape[1], img.shape[0])

                img_w, img_h = img_dims[current_img]

                if format == "xywh":
                    x, y, w, h = coords
                    cx = max(0.0, min(1.0, (x + w / 2) / img_w))
                    cy = max(0.0, min(1.0, (y + h / 2) / img_h))
                    nw = max(0.0, min(1.0, w / img_w))
                    nh = max(0.0, min(1.0, h / img_h))
                else:
                    x1, y1, x2, y2 = coords
                    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
                    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
                    nw = max(0.0, min(1.0, (x2 - x1) / img_w))
                    nh = max(0.0, min(1.0, (y2 - y1) / img_h))

                lbl_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                count += 1

        if current_img and lbl_lines:
            lbl_path = out_lbl_dir / f"{Path(current_img).stem}.txt"
            lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")

        logger.info("Converted %d annotations from TXT", count)
        return count

    def create_data_yaml(self) -> Path:
        """Create YOLO data.yaml configuration file."""
        yaml_content = f"""# VisionTextReader YOLO Dataset Configuration
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
