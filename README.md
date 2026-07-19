# VisionTextReader

Hệ thống nhận diện và đọc văn bản từ ảnh sử dụng YOLOv8.

## Cấu trúc thư mục

```
VisionTextReader/
├── src/
│   └── detection/
│       ├── __init__.py
│       └── detector.py          # TextDetector class
├── config.py                     # Cấu hình chung
├── utils.py                      # Tiện ích image I/O, BBox, visualization
├── dataset_converter.py          # Chuyển đổi dataset sang YOLO format
├── dataset_splitter.py           # Chia dataset train/val/test
├── dataset_validator.py          # Kiểm tra chất lượng dataset
├── train.py                      # Huấn luyện YOLO
├── detect.py                     # CLI detection (legacy)
├── crop.py                       # CLI crop (legacy)
├── demo_detect.py                # Demo pipeline đầy đủ
├── benchmark.py                  # Benchmark hiệu suất
├── data.yaml                     # Cấu hình YOLO dataset
├── weights/                      # Model weights
├── datasets/                     # Dataset
│   ├── raw/                      # Dataset gốc
│   └── processed/                # Dataset đã xử lý (YOLO format)
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── outputs/                      # Kết quả
│   ├── demo/                     # Kết quả demo
│   ├── crops/                    # Ảnh crop
│   └── validation/               # Báo cáo validation
└── requirements.txt
```

## Cài đặt môi trường

```bash
# Clone repository
git clone <repo-url>
cd VisionTextReader

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements.txt
```

### Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
```

## Sử dụng

### 1. Import và sử dụng trong code

```python
from src.detection import TextDetector
from src.detection.detector import crop, draw_boxes, save_crops

# Khởi tạo detector
detector = TextDetector("weights/best.pt")

# Detect text regions
image = "photo.jpg"
boxes = detector.detect(image)

# Kết quả:
# [
#     {"bbox": [x1, y1, x2, y2], "confidence": 0.97, "class_id": 0, "class_name": "text"},
#     ...
# ]

# Vẽ bounding box
annotated = draw_boxes(image, [d["bbox"] for d in boxes],
                       [d["confidence"] for d in boxes])

# Crop text regions
crops = crop(image, [d["bbox"] for d in boxes])

# Lưu crops
save_crops(crops, "outputs/crops/")
```

### 2. Train model

```bash
python train.py --data datasets/processed/data.yaml --epochs 100 --imgsz 640
```

### 3. Run detection

```bash
# Detect trên một ảnh
python demo_detect.py image.jpg --model weights/best.pt

# Detect với tham số tuỳ chỉnh
python demo_detect.py image.jpg --conf 0.3 --output outputs/demo
```

### 4. Run benchmark

```bash
# Benchmark trên thư mục test
python benchmark.py datasets/processed/test/images --model weights/best.pt

# Benchmark với tham số tuỳ chỉnh
python benchmark.py test/images --iterations 5 --warmup 3 --max-images 100
```

### 5. Validate dataset

```bash
python dataset_validator.py datasets/processed --splits train val test
```

## Demo Pipeline

```bash
python demo_detect.py image.jpg
```

Output:
```
============================================================
  VisionTextReader — Detection Pipeline Demo
============================================================
[1] Model loaded in 1234.5 ms
[2] Image loaded: photo.jpg (1920x1080)
[3] Detected 18 text regions in 45.2 ms
[4] Annotated image saved: outputs/demo/photo_annotated.jpg
[5] Cropped 18 regions → outputs/demo/crops
[6] Predictions saved: outputs/demo/photo_predictions.json

------------------------------------------------------------
  STATISTICS
------------------------------------------------------------
  Detected objects : 18
  Avg confidence   : 0.9523
  Min confidence   : 0.7234
  Max confidence   : 0.9987
  Inference time   : 45.2 ms
  Model load time  : 1234.5 ms
  Crop count       : 18
------------------------------------------------------------
  Output directory : outputs/demo
============================================================
```

## Benchmark

```bash
python benchmark.py datasets/processed/test/images --model weights/best.pt
```

Output:
```
============================================================
  BENCHMARK RESULTS
============================================================
  Total images       : 500
  Total inferences   : 1500
  Iterations         : 3
  Warmup             : 2
------------------------------------------------------------
  FPS                : 22.15
  Total time         : 67.723 s
------------------------------------------------------------
  Inference time:
    Mean             : 45.15 ms
    Median           : 42.30 ms
    Min              : 28.50 ms
    Max              : 85.20 ms
    Std              : 12.35 ms
------------------------------------------------------------
  Detections:
    Total            : 2700
    Avg per image    : 1.80
============================================================
```

## API Reference

### TextDetector

```python
class TextDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45,
                 imgsz=640, device="", class_names=None)
    
    def load_model(self) -> None
    def detect(self, image, conf_threshold=None, iou_threshold=None) -> List[Dict]
    def detect_batch(self, images, ...) -> List[List[Dict]]
    def detect_with_timing(self, image, ...) -> Tuple[List[Dict], float]
```

### crop()

```python
def crop(image, boxes, margin=0, padding=0) -> List[np.ndarray]
```

### draw_boxes()

```python
def draw_boxes(image, boxes, scores=None, class_names=None, ...) -> np.ndarray
```

### save_crops()

```python
def save_crops(crops, folder, prefix="crop", extension="png") -> List[Path]
```
