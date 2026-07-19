"""
hash_utils.py — Hash và similarity functions.
"""

from __future__ import annotations

import sys

import hashlib
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Them project root vao sys.path de import config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def calculate_phash(image: np.ndarray) -> str:
    """Tính Perceptual Hash (pHash)."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        avg = np.mean(dct_low)
        bits = (dct_low > avg).flatten()
        return "".join(["1" if b else "0" for b in bits])
    except Exception:
        return ""


def calculate_ahash(image: np.ndarray) -> str:
    """Tính Average Hash (aHash)."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        avg = np.mean(resized)
        bits = (resized > avg).flatten()
        return "".join(["1" if b else "0" for b in bits])
    except Exception:
        return ""


def calculate_dhash(image: np.ndarray) -> str:
    """Tính Difference Hash (dHash)."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = []
        for row in range(8):
            for col in range(8):
                bits.append(1 if resized[row, col] > resized[row, col + 1] else 0)
        return "".join(map(str, bits))
    except Exception:
        return ""


def calculate_file_hashes(file_path: Path) -> Tuple[str, str]:
    """Tính MD5 và SHA256 hash."""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    except (IOError, PermissionError):
        return "", ""


def hamming_distance(hash1: str, hash2: str) -> int:
    """Tính khoảng cách Hamming."""
    if len(hash1) != len(hash2):
        return max(len(hash1), len(hash2))
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def histogram_similarity(hist1: List[int], hist2: List[int]) -> float:
    """Tính tương đồng histogram."""
    if not hist1 or not hist2 or len(hist1) != len(hist2):
        return 0.0
    h1 = np.array(hist1, dtype=np.float64)
    h2 = np.array(hist2, dtype=np.float64)
    if np.std(h1) == 0 or np.std(h2) == 0:
        return 0.0
    return float(np.corrcoef(h1, h2)[0, 1])


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Tính Structural Similarity Index (SSIM)."""
    try:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1 ** 2
        sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2 ** 2
        sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(np.mean(ssim_map))
    except Exception:
        return 0.0
