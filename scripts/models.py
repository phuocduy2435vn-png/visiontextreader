"""
models.py — Tất cả dataclass cho VisionTextReader Dataset Validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class Severity:
    """Mức độ nghiêm trọng của lỗi."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


@dataclass
class ValidationError:
    """Lỗi cụ thể trong quá trình validation."""
    split: str = ""
    image_name: str = ""
    label_name: str = ""
    error_type: str = ""
    severity: str = Severity.ERROR
    description: str = ""
    line_number: int = 0
    timestamp: str = ""


@dataclass
class ImageQualityMetrics:
    """Chỉ số chất lượng ảnh nâng cao."""
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    phash: str = ""
    ahash: str = ""
    dhash: str = ""
    md5: str = ""
    sha256: str = ""
    entropy: float = 0.0
    noise_estimate: float = 0.0
    sharpness: float = 0.0
    dynamic_range: int = 0
    over_exposed: float = 0.0
    under_exposed: float = 0.0
    saturated: float = 0.0
    histogram: List[int] = field(default_factory=list)
    is_dead: bool = False
    is_near_black: bool = False
    is_near_white: bool = False
    orientation: int = 1
    snr: float = 0.0
    motion_blur: float = 0.0
    gaussian_blur: float = 0.0
    jpeg_artifact: float = 0.0
    color_balance: List[float] = field(default_factory=list)
    hue_distribution: List[int] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class BBoxAnalysis:
    """Kết quả phân tích bounding box."""
    total_bboxes: int = 0
    areas: List[float] = field(default_factory=list)
    widths: List[float] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)
    center_x: List[float] = field(default_factory=list)
    center_y: List[float] = field(default_factory=list)
    border_touching: int = 0
    too_small: int = 0
    too_large: int = 0
    coverage: float = 0.0
    occupancy: float = 0.0
    density: float = 0.0
    iou_pairs: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Kết quả validation cho một split."""
    split_name: str = ""
    total_images: int = 0
    total_labels: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    overlapping_bboxes: int = 0
    blurry_images: int = 0
    low_contrast_images: int = 0
    quality_metrics: Dict[str, ImageQualityMetrics] = field(default_factory=dict)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    errors: List[ValidationError] = field(default_factory=list)


@dataclass
class DatasetValidationSummary:
    """Tóm tắt kết quả validation toàn bộ dataset."""
    train: Optional[ValidationResult] = None
    val: Optional[ValidationResult] = None
    test: Optional[ValidationResult] = None
    fingerprint: str = ""
    leakage_errors: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditFinding:
    """Nhận xét tự động từ AI Audit."""
    category: str = ""
    finding: str = ""
    severity: str = Severity.INFO
    details: str = ""


@dataclass
class Recommendation:
    """Gợi ý tự động từ Recommendation Engine."""
    action: str = ""
    priority: str = "Medium"
    reason: str = ""
    impact: str = ""


@dataclass
class ScorecardEntry:
    """Một dòng trong Dataset Scorecard."""
    category: str = ""
    score: float = 0.0
    max_score: float = 100.0
    details: str = ""


@dataclass
class DatasetScorecard:
    """Bảng điểm dataset."""
    entries: List[ScorecardEntry] = field(default_factory=list)
    overall_score: float = 0.0
    grade: str = ""


@dataclass
class YOLOReadiness:
    """Đánh giá mức sẵn sàng train YOLO."""
    score: float = 0.0
    level: str = ""
    reasons: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)


@dataclass
class TrainingRisk:
    """Nguy cơ khi train model."""
    risk_type: str = ""
    risk_level: str = ""
    probability: float = 0.0
    description: str = ""
    mitigation: str = ""


@dataclass
class DatasetVersion:
    """Phiên bản dataset."""
    version: str = ""
    fingerprint: str = ""
    validation_time: str = ""
    dataset_size: int = 0
    total_images: int = 0
    total_labels: int = 0
    health_score: float = 0.0
    changed_images: int = 0
    added_images: int = 0
    removed_images: int = 0
    modified_labels: int = 0


@dataclass
class ChangelogEntry:
    """Một dòng trong changelog."""
    change_type: str = ""
    count: int = 0
    description: str = ""


@dataclass
class BenchmarkResult:
    """Kết quả benchmark hiệu năng."""
    cpu_time: float = 0.0
    wall_time: float = 0.0
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    images_per_sec: float = 0.0
    labels_per_sec: float = 0.0


@dataclass
class ValidatorConfig:
    """Cấu hình validator."""
    blur_threshold: float = 50.0
    contrast_threshold: float = 15.0
    iou_threshold: float = 0.85
    leakage_hamming_threshold: int = 2
    quality_score_weight_blur: float = 0.25
    quality_score_weight_contrast: float = 0.20
    quality_score_weight_brightness: float = 0.15
    quality_score_weight_sharpness: float = 0.15
    quality_score_weight_entropy: float = 0.10
    quality_score_weight_dynamic_range: float = 0.10
    output_folder: str = "outputs/validation"
    workers: int = 4
    use_cache: bool = True
    max_fingerprint_files: int = 1000

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> ValidatorConfig:
        """Tải config từ file YAML hoặc dùng mặc định."""
        from pathlib import Path as _Path
        import logging
        _logger = logging.getLogger("dataset_validator")
        cfg = cls()
        if config_path is None:
            config_path = _Path(__file__).resolve().parent.parent / "validator_config.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for key, val in data.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, val)
                _logger.info("Đã tải config từ %s", config_path)
            except ImportError:
                _logger.warning("PyYAML không được cài — dùng config mặc định")
            except Exception as exc:
                _logger.warning("Lỗi đọc config: %s", exc)
        return cfg
