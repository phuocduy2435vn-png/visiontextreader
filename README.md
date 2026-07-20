# VisionTextReader

YOLO-based text detection: detect, crop text regions from images.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data data.yaml --epochs 100 --batch 16 --imgsz 640
```

| Param | Default | Description |
|-------|---------|-------------|
| `--data` | (required) | Path to data.yaml |
| `--model` | yolov8n.pt | Pre-trained model |
| `--epochs` | 100 | Training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Image size |
| `--device` | "" | `0` for GPU, `cpu` for CPU |
| `--resume` | false | Resume from checkpoint |
| `--workers` | 8 | DataLoader workers |
| `--project` | runs/detect | Output directory |
| `--name` | visiontextreader | Experiment name |

## Detect

```bash
python detect.py image.jpg --model weights/best.pt --conf 0.25
```

Options: `--save-json`, `--save-txt`, `--output <dir>`

## Crop

```bash
python crop.py image.jpg --boxes detections.json --output output/crops
```

## Dataset

```bash
# Convert raw datasets to YOLO format
python dataset_converter.py

# Split into train/val/test
python dataset_splitter.py

# Create small subset (~12K images)
python -c "from dataset_splitter import DatasetSplitter; DatasetSplitter('datasets/processed/train/images', 'datasets/processed').split_small()"

# Validate
python dataset_validator.py datasets/processed
```

Structure:
```
datasets/
├── original/          # Raw datasets (don't modify)
├── processed/         # YOLO format (train/val/test)
│   ├── train/images/ + labels/
│   ├── val/images/ + labels/
│   └── test/images/ + labels/
└── processed_small/   # Subset ~12K images
```

## Project Structure

```
VisionTextReader/
├── src/detection/         # TextDetector class
│   ├── __init__.py
│   └── detector.py
├── config.py              # Paths, constants
├── utils.py               # Image I/O, visualization
├── train.py               # YOLO training
├── detect.py              # Text detection CLI
├── crop.py                # Text region cropping
├── dataset_converter.py   # Format conversion
├── dataset_splitter.py    # Train/val/test split
├── dataset_validator.py   # Dataset validation
├── data.yaml              # YOLO dataset config
├── weights/               # Model weights
├── datasets/              # Dataset
└── requirements.txt
```

## Kaggle

1. Upload dataset to Kaggle Datasets
2. Upload project code
3. Run:

```python
!pip install -r requirements.txt
!python train.py --data data.yaml --epochs 100 --device 0
```

## API

```python
from src.detection import TextDetector, draw_boxes

detector = TextDetector("weights/best.pt")
detections = detector.detect("image.jpg")
# [{"bbox": [x1,y1,x2,y2], "confidence": 0.97, "class_id": 0, "class_name": "text"}]
```
