"""
image_processing.py — Image integrity checks, quality assessment worker,
quality score computation.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import ImageQualityMetrics, Severity, ValidationError
from hash_utils import calculate_file_hashes, calculate_phash, calculate_ahash, calculate_dhash

logger = logging.getLogger("dataset_validator")


def check_image_integrity(file_path: Path) -> List[ValidationError]:
    """Kiểm tra tính toàn vẹn file ảnh: rỗng, sai extension, sai mime, corrupted."""
    errors: List[ValidationError] = []
    now = datetime.now(timezone.utc).isoformat()
    if not file_path.exists():
        errors.append(ValidationError(image_name=file_path.name, error_type="MISSING_FILE",
                                       severity=Severity.CRITICAL, description="File không tồn tại", timestamp=now))
        return errors
    try:
        if file_path.stat().st_size == 0:
            errors.append(ValidationError(image_name=file_path.name, error_type="EMPTY_FILE",
                                           severity=Severity.CRITICAL, description="File rỗng", timestamp=now))
            return errors
    except (PermissionError, OSError) as exc:
        errors.append(ValidationError(image_name=file_path.name, error_type="IO_ERROR",
                                       severity=Severity.ERROR, description=f"Lỗi IO: {exc}", timestamp=now))
        return errors
    from config import IMAGE_EXTENSIONS
    valid_ext = {e.lower() for e in IMAGE_EXTENSIONS}
    if file_path.suffix.lower() not in valid_ext:
        errors.append(ValidationError(image_name=file_path.name, error_type="WRONG_EXTENSION",
                                       severity=Severity.WARNING, description=f"Extension '{file_path.suffix}' không hợp lệ", timestamp=now))
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        fmt_ok = False
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            fmt_ok = True
            try:
                with open(file_path, "rb") as f2:
                    data = f2.read()
                if len(data) >= 8 and data[-8:] != b"IEND\xaeB`\x82":
                    errors.append(ValidationError(image_name=file_path.name, error_type="CORRUPTED_PNG",
                                                   severity=Severity.WARNING, description="PNG thiếu IEND", timestamp=now))
            except (IOError, PermissionError):
                pass
        elif header[:2] == b"\xff\xd8":
            fmt_ok = True
            try:
                with open(file_path, "rb") as f2:
                    data = f2.read()
                if len(data) >= 2 and data[-2:] != b"\xff\xd9":
                    errors.append(ValidationError(image_name=file_path.name, error_type="CORRUPTED_JPEG",
                                                   severity=Severity.WARNING, description="JPEG thiếu EOI marker", timestamp=now))
            except (IOError, PermissionError):
                pass
        elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            fmt_ok = True
        elif header[:2] == b"BM":
            fmt_ok = True
        if not fmt_ok:
            errors.append(ValidationError(image_name=file_path.name, error_type="WRONG_MIME",
                                           severity=Severity.WARNING, description="Magic bytes không khớp", timestamp=now))
    except (IOError, PermissionError):
        pass
    return errors


def _compute_quality_score(blur_score: float, brightness: float, contrast: float,
                           entropy_val: float, sharpness: float, dynamic_range: int,
                           over_exposed: float, under_exposed: float, is_dead: bool) -> float:
    """Tính điểm chất lượng tổng hợp 0-100."""
    if is_dead:
        return 0.0
    blur_n = min(100.0, max(0.0, (blur_score / 500.0) * 100.0))
    contrast_n = min(100.0, max(0.0, ((contrast - 5.0) / 55.0) * 100.0))
    bright_n = 100.0 if 80 <= brightness <= 180 else max(0.0, (brightness / 80.0) * 100.0) if brightness < 80 else max(0.0, ((255.0 - brightness) / 75.0) * 100.0)
    sharp_n = min(100.0, max(0.0, (sharpness / 500.0) * 100.0))
    entropy_n = min(100.0, max(0.0, ((entropy_val - 1.0) / 6.0) * 100.0))
    dr_n = min(100.0, max(0.0, (dynamic_range / 200.0) * 100.0))
    exposure_penalty = (over_exposed + under_exposed) * 50.0
    score = blur_n * 0.25 + contrast_n * 0.20 + bright_n * 0.15 + sharp_n * 0.15 + entropy_n * 0.10 + dr_n * 0.10 - exposure_penalty
    return round(max(0.0, min(100.0, score)), 1)


def analyze_image_quality(args: Tuple[str, str]) -> Tuple[Optional[ImageQualityMetrics], List[dict]]:
    """Worker multiprocessing — phân tích chất lượng ảnh."""
    img_path_str, split_name = args
    img_path = Path(img_path_str)
    errors: List[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    try:
        image = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
        if image is None:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "CORRUPTED_IMAGE", "severity": Severity.CRITICAL, "description": "OpenCV không giải mã được", "line_number": 0, "timestamp": now})
            return None, errors
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "INVALID_DIMENSIONS", "severity": Severity.ERROR, "description": f"Kích thước: {width}x{height}", "line_number": 0, "timestamp": now})
            return None, errors
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_px = float(gray.shape[0] * gray.shape[1])

        brightness = float(np.mean(gray))

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        noise_est = float(np.median(np.abs(laplacian)))
        del laplacian

        contrast = float(np.std(gray))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_n = hist / hist.sum()
        hist_n = hist_n[hist_n > 0]
        entropy_val = float(-np.sum(hist_n * np.log2(hist_n)))
        del hist, hist_n

        signal = brightness if brightness > 0 else 1.0
        snr = signal / noise_est if noise_est > 0 else 0.0

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gx_sq = gx ** 2
        gy_sq = gy ** 2
        sharpness = float(np.mean(gx_sq + gy_sq))
        del gx, gy, gx_sq, gy_sq

        dynamic_range = int(np.ptp(gray))
        over_exp = float(np.sum(gray > 250) / total_px)
        under_exp = float(np.sum(gray < 5) / total_px)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturated = float(np.sum(hsv[:, :, 1] > 240) / total_px)
        hist_16 = [int(v) for v in cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten().tolist()]
        hue_hist = [int(v) for v in cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten().tolist()]
        del hsv

        b_m = float(np.mean(image[:, :, 0]))
        g_m = float(np.mean(image[:, :, 1]))
        r_m = float(np.mean(image[:, :, 2]))
        tc = r_m + g_m + b_m
        color_bal = [r_m / tc, g_m / tc, b_m / tc] if tc > 0 else [0.33, 0.33, 0.33]
        is_dead = blur_score < 1.0 and dynamic_range < 5

        motion_blur = float(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5).var())
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        diff = gray.astype(np.float32) - blurred.astype(np.float32)
        gaussian_blur = float(np.mean(np.abs(diff)))
        del blurred, diff

        jpeg_art = 0.0
        if width >= 8 and height >= 8:
            blocks = gray[::8, :].astype(np.float32)
            if blocks.shape[0] > 2:
                jpeg_art = float(np.mean(np.abs(np.diff(blocks, axis=0))))

        md5, sha256 = calculate_file_hashes(img_path)
        phash_s = calculate_phash(image)
        ahash_s = calculate_ahash(image)
        dhash_s = calculate_dhash(image)
        del image, gray

        q_score = _compute_quality_score(blur_score, brightness, contrast, entropy_val, sharpness, dynamic_range, over_exp, under_exp, is_dead)
        metrics = ImageQualityMetrics(blur_score=blur_score, brightness=brightness, contrast=contrast, phash=phash_s, ahash=ahash_s, dhash=dhash_s, md5=md5, sha256=sha256, entropy=entropy_val, noise_estimate=noise_est, sharpness=sharpness, dynamic_range=dynamic_range, over_exposed=over_exp, under_exposed=under_exp, saturated=saturated, histogram=hist_16, is_dead=is_dead, is_near_black=brightness < 15.0, is_near_white=brightness > 240.0, snr=snr, motion_blur=motion_blur, gaussian_blur=gaussian_blur, jpeg_artifact=jpeg_art, color_balance=color_bal, hue_distribution=hue_hist, quality_score=q_score)

        if blur_score < 50.0:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "BLURRY_IMAGE", "severity": Severity.WARNING, "description": f"Mờ (var={blur_score:.1f})", "line_number": 0, "timestamp": now})
        if contrast < 15.0:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "LOW_CONTRAST", "severity": Severity.WARNING, "description": f"Tương phản thấp (std={contrast:.1f})", "line_number": 0, "timestamp": now})
        if is_dead:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "DEAD_IMAGE", "severity": Severity.CRITICAL, "description": "Ảnh chết", "line_number": 0, "timestamp": now})
        if brightness < 15.0:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "NEAR_BLACK_IMAGE", "severity": Severity.WARNING, "description": f"Gần đen (b={brightness:.1f})", "line_number": 0, "timestamp": now})
        if brightness > 240.0:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "NEAR_WHITE_IMAGE", "severity": Severity.WARNING, "description": f"Gần trắng (b={brightness:.1f})", "line_number": 0, "timestamp": now})
        if over_exp > 0.3:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "OVER_EXPOSED", "severity": Severity.WARNING, "description": f"Over-exposed ({over_exp:.1%})", "line_number": 0, "timestamp": now})
        if under_exp > 0.3:
            errors.append({"split": split_name, "image_name": img_path.name, "error_type": "UNDER_EXPOSED", "severity": Severity.WARNING, "description": f"Under-exposed ({under_exp:.1%})", "line_number": 0, "timestamp": now})

        return metrics, errors
    except cv2.error as exc:
        errors.append({"split": split_name, "image_name": img_path.name, "error_type": "OPENCV_ERROR", "severity": Severity.ERROR, "description": f"OpenCV: {exc}", "line_number": 0, "timestamp": now})
        return None, errors
    except Exception as exc:
        errors.append({"split": split_name, "image_name": img_path.name, "error_type": "RUNTIME_ERROR", "severity": Severity.CRITICAL, "description": f"Runtime: {exc}", "line_number": 0, "timestamp": now})
        return None, errors
