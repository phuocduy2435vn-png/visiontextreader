"""
audit.py — AI Audit, Recommendation Engine, Scorecard, YOLO Readiness, Training Risk.
"""

from __future__ import annotations

from pathlib import Path

import sys

from typing import List, Optional

import numpy as np

from models import (
    AuditFinding, DatasetScorecard, DatasetValidationSummary, Recommendation,
    ScorecardEntry, Severity, TrainingRisk, ValidatorConfig, YOLOReadiness,
)
from analysis import analyze_class_distribution, analyze_split_ratios
from hash_utils import hamming_distance

# Them project root vao sys.path de import config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class DatasetAudit:
    """Hệ thống Audit tự động."""

    def __init__(self, summary: DatasetValidationSummary, config: Optional[ValidatorConfig] = None):
        self.summary = summary
        self.config = config or ValidatorConfig()
        self.findings: List[AuditFinding] = []

    def run_audit(self) -> List[AuditFinding]:
        self.findings = []
        self._audit_balance()
        self._audit_quality()
        self._audit_bbox()
        self._audit_leakage()
        self._audit_resolution()
        self._audit_class_coverage()
        self._audit_readiness()
        return self.findings

    def _add(self, cat: str, finding: str, severity: str = Severity.INFO, details: str = ""):
        self.findings.append(AuditFinding(category=cat, finding=finding, severity=severity, details=details))

    def _audit_balance(self):
        train_n = self.summary.train.total_images if self.summary.train else 0
        val_n = self.summary.val.total_images if self.summary.val else 0
        test_n = self.summary.test.total_images if self.summary.test else 0
        total = train_n + val_n + test_n
        if total == 0:
            self._add("Balance", "Dataset trống", Severity.CRITICAL)
            return
        ratios = analyze_split_ratios(train_n, val_n, test_n)
        for w in ratios["warnings"]:
            self._add("Balance", w, Severity.WARNING)
        if not ratios["warnings"]:
            self._add("Balance", "Tỷ lệ split hợp lý", Severity.INFO)

    def _audit_quality(self):
        all_scores = []
        blurry_total = low_contrast_total = dead_total = 0
        for r in [self.summary.train, self.summary.val, self.summary.test]:
            if r is None:
                continue
            blurry_total += r.blurry_images
            low_contrast_total += r.low_contrast_images
            for m in r.quality_metrics.values():
                all_scores.append(m.quality_score)
                if m.is_dead:
                    dead_total += 1
        if not all_scores:
            return
        avg_score = np.mean(all_scores)
        total_imgs = sum(r.total_images for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if blurry_total > total_imgs * 0.1:
            self._add("Quality", f"Nhiều ảnh mờ ({blurry_total}/{total_imgs})", Severity.WARNING)
        elif blurry_total <= total_imgs * 0.02:
            self._add("Quality", f"Tỷ lệ ảnh mờ thấp ({blurry_total}/{total_imgs})", Severity.INFO)
        if low_contrast_total > total_imgs * 0.1:
            self._add("Quality", f"Nhiệu ảnh tương phản thấp ({low_contrast_total}/{total_imgs})", Severity.WARNING)
        if dead_total > 0:
            self._add("Quality", f"Có {dead_total} ảnh chết", Severity.ERROR)
        if avg_score >= 80:
            self._add("Quality", f"Chất lượng tốt (TB: {avg_score:.1f}/100)", Severity.INFO)
        elif avg_score >= 60:
            self._add("Quality", f"Chất lượng trung bình (TB: {avg_score:.1f}/100)", Severity.WARNING)
        else:
            self._add("Quality", f"Chất lượng kém (TB: {avg_score:.1f}/100)", Severity.ERROR)

    def _audit_bbox(self):
        overlap_total = sum(r.overlapping_bboxes for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if overlap_total > 0:
            self._add("BBox", f"{overlap_total} cặp bbox chồng lấn (IoU>{self.config.iou_threshold})", Severity.WARNING)
        else:
            self._add("BBox", "Không phát hiện bbox chồng lấn", Severity.INFO)

    def _audit_leakage(self):
        n = len(self.summary.leakage_errors)
        if n > 0:
            cross = sum(1 for e in self.summary.leakage_errors if e.error_type == "CROSS_SPLIT_LEAKAGE")
            if cross > 0:
                self._add("Leakage", f"Data Leakage nghiêm trọng ({cross} cross-split)", Severity.CRITICAL)
            dup = n - cross
            if dup > 0:
                self._add("Leakage", f"{dup} duplicate trong cùng split", Severity.WARNING)
        else:
            self._add("Leakage", "Không phát hiện data leakage", Severity.INFO)

    def _audit_resolution(self):
        low_q = 0
        total = 0
        for r in [self.summary.train, self.summary.val, self.summary.test]:
            if r is None:
                continue
            for m in r.quality_metrics.values():
                total += 1
                if m.quality_score < 50:
                    low_q += 1
        if total > 0 and low_q > total * 0.2:
            self._add("Resolution", f"{low_q} ảnh chất lượng thấp ({low_q/total:.1%})", Severity.WARNING)

    def _audit_class_coverage(self):
        combined: Dict[str, int] = {}
        for r in [self.summary.train, self.summary.val, self.summary.test]:
            if r is None:
                continue
            for cls, cnt in r.class_distribution.items():
                combined[cls] = combined.get(cls, 0) + cnt
        if not combined:
            return
        analysis = analyze_class_distribution(combined)
        if analysis["missing_classes"]:
            self._add("Class", f"Thiếu class: {', '.join(analysis['missing_classes'][:5])}", Severity.WARNING)
        if analysis["rare_classes"]:
            self._add("Class", f"Class hiếm: {', '.join(analysis['rare_classes'][:5])}", Severity.WARNING)
        if analysis["imbalance_ratio"] > 10:
            self._add("Class", f"Mất cân bằng nghiêm trọng ({analysis['imbalance_ratio']:.1f}:1)", Severity.WARNING)
        else:
            self._add("Class", "Phân bố class hợp lý", Severity.INFO)

    def _audit_readiness(self):
        critical = sum(1 for f in self.findings if f.severity == Severity.CRITICAL)
        errors = sum(1 for f in self.findings if f.severity == Severity.ERROR)
        warnings = sum(1 for f in self.findings if f.severity == Severity.WARNING)
        if critical > 0:
            self._add("Summary", "CHƯA PHÙ HỢP để train YOLO", Severity.CRITICAL)
        elif errors > 2:
            self._add("Summary", "CẦN CẢI THIỆN trước khi train", Severity.ERROR)
        elif warnings > 5:
            self._add("Summary", "Có thể train nhưng nên cải thiện", Severity.WARNING)
        else:
            self._add("Summary", "ĐỦ CHUẨN để train YOLO", Severity.INFO)


class RecommendationEngine:
    """Sinh gợi ý tự động."""

    def __init__(self, summary: DatasetValidationSummary, audit_findings: List[AuditFinding], config: Optional[ValidatorConfig] = None):
        self.summary = summary
        self.findings = audit_findings
        self.config = config or ValidatorConfig()
        self.recommendations: List[Recommendation] = []

    def generate(self) -> List[Recommendation]:
        self.recommendations = []
        self._recommend_quality()
        self._recommend_balance()
        self._recommend_leakage()
        self._recommend_bbox()
        self._recommend_duplicates()
        self._recommend_general()
        return self.recommendations

    def _add(self, action: str, priority: str, reason: str, impact: str = ""):
        self.recommendations.append(Recommendation(action=action, priority=priority, reason=reason, impact=impact))

    def _recommend_quality(self):
        blurry_total = sum(r.blurry_images for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        total_imgs = sum(r.total_images for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if blurry_total > total_imgs * 0.05:
            self._add("Loại bỏ ảnh mờ", "High", f"{blurry_total} ảnh mờ", "Cải thiện chất lượng")
        low_q = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for m in r.quality_metrics.values() if m.quality_score < 30)
        if low_q > 0:
            self._add("Xóa ảnh chất lượng thấp", "Medium", f"{low_q} ảnh score<30", "Giảm noise")

    def _recommend_balance(self):
        train_n = self.summary.train.total_images if self.summary.train else 0
        val_n = self.summary.val.total_images if self.summary.val else 0
        test_n = self.summary.test.total_images if self.summary.test else 0
        total = train_n + val_n + test_n
        if total == 0:
            return
        ratios = analyze_split_ratios(train_n, val_n, test_n)
        if ratios["warnings"]:
            self._add("Điều chỉnh Ratio", "High", "; ".join(ratios["warnings"]), "Tăng generalization")

    def _recommend_leakage(self):
        n = len(self.summary.leakage_errors)
        if n > 0:
            self._add("Xóa duplicate/leakage", "Critical", f"{n} ảnh trùng lặp", "Tránh overfitting")

    def _recommend_bbox(self):
        overlap = sum(r.overlapping_bboxes for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if overlap > 0:
            self._add("Kiểm tra bbox chồng lấn", "Medium", f"{overlap} cặp IoU>{self.config.iou_threshold}", "Annotation chính xác")

    def _recommend_duplicates(self):
        dup_ann = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for e in r.errors if e.error_type == "DUPLICATE_ANNOTATION")
        if dup_ann > 0:
            self._add("Xóa annotation trùng", "Medium", f"{dup_ann} annotation trùng", "Giảm redundancy")

    def _recommend_general(self):
        self._add("Chuẩn hóa resolution 640x640", "Low", "YOLO tối ưu ở 640x640", "Training speed")
        self._add("Convert ảnh sang RGB", "Low", "Định dạng nhất quán", "Tránh lỗi channel")


class ScorecardGenerator:
    """Sinh bảng điểm dataset."""

    def __init__(self, summary: DatasetValidationSummary, config: Optional[ValidatorConfig] = None):
        self.summary = summary
        self.config = config or ValidatorConfig()

    def generate(self) -> DatasetScorecard:
        sc = DatasetScorecard()
        sc.entries.append(self._score_structure())
        sc.entries.append(self._score_image_quality())
        sc.entries.append(self._score_annotation())
        sc.entries.append(self._score_bbox())
        sc.entries.append(self._score_class_balance())
        sc.entries.append(self._score_leakage())
        sc.overall_score = round(np.mean([e.score for e in sc.entries]), 1)
        sc.grade = _get_rating(sc.overall_score)
        return sc

    def _score_structure(self) -> ScorecardEntry:
        errors = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r is None)
        missing = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r and r.total_images == 0)
        return ScorecardEntry(category="Dataset Structure", score=max(0, 100 - (errors + missing) * 15), details=f"{errors} splits thiếu")

    def _score_image_quality(self) -> ScorecardEntry:
        scores = [m.quality_score for r in [self.summary.train, self.summary.val, self.summary.test] if r for m in r.quality_metrics.values()]
        return ScorecardEntry(category="Image Quality", score=round(np.mean(scores) if scores else 100.0, 1))

    def _score_annotation(self) -> ScorecardEntry:
        ann_errors = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for e in r.errors if e.error_type in ("INVALID_FORMAT", "INVALID_FLOAT", "NaN_VALUE", "INFINITY_VALUE", "NEGATIVE_WIDTH", "NEGATIVE_HEIGHT", "COORDS_OUTSIDE_IMAGE", "ZERO_AREA_BBOX"))
        total = sum(r.total_images for r in [self.summary.train, self.summary.val, self.summary.test] if r) or 1
        return ScorecardEntry(category="Annotation", score=round(max(0, 100 - (ann_errors / total) * 1000), 1))

    def _score_bbox(self) -> ScorecardEntry:
        overlap = sum(r.overlapping_bboxes for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        total = sum(r.total_images for r in [self.summary.train, self.summary.val, self.summary.test] if r) or 1
        return ScorecardEntry(category="Bounding Box", score=round(max(0, 100 - min(30, (overlap / total) * 100)), 1))

    def _score_class_balance(self) -> ScorecardEntry:
        combined: Dict[str, int] = {}
        for r in [self.summary.train, self.summary.val, self.summary.test]:
            if r is None:
                continue
            for cls, cnt in r.class_distribution.items():
                combined[cls] = combined.get(cls, 0) + cnt
        if not combined:
            return ScorecardEntry(category="Class Balance", score=100.0)
        analysis = analyze_class_distribution(combined)
        penalty = min(30, analysis["imbalance_ratio"] * 2) if analysis["imbalance_ratio"] < float("inf") else 30
        return ScorecardEntry(category="Class Balance", score=round(max(0, 100 - penalty), 1))

    def _score_leakage(self) -> ScorecardEntry:
        n = len(self.summary.leakage_errors)
        return ScorecardEntry(category="Leakage", score=round(max(0, 100 - min(100, n * 20)), 1))


class YOLOReadinessEvaluator:
    """Đánh giá mức sẵn sàng train YOLO."""

    def __init__(self, summary: DatasetValidationSummary, scorecard: DatasetScorecard):
        self.summary = summary
        self.scorecard = scorecard

    def evaluate(self) -> YOLOReadiness:
        yr = YOLOReadiness()
        yr.score = self.scorecard.overall_score
        yr.level = _get_rating(yr.score)
        total_imgs = sum(r.total_images for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if total_imgs == 0:
            yr.reasons.append("Dataset trống")
            yr.blocking_issues.append("Không có ảnh")
            return yr
        if self.summary.train and self.summary.train.total_images < 100:
            yr.blocking_issues.append(f"Train quá nhỏ ({self.summary.train.total_images})")
        if len(self.summary.leakage_errors) > 0:
            yr.reasons.append(f"{len(self.summary.leakage_errors)} vấn đề leakage")
        blurry = sum(r.blurry_images for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if blurry > total_imgs * 0.1:
            yr.reasons.append(f"Nhiều ảnh mờ ({blurry}/{total_imgs})")
        for entry in self.scorecard.entries:
            if entry.score < 50:
                yr.reasons.append(f"{entry.category} điểm thấp ({entry.score})")
        if not yr.blocking_issues and yr.score >= 70:
            yr.reasons.append("Đạt yêu cầu tối thiểu train YOLO")
        return yr


class TrainingRiskAnalyzer:
    """Dự đoán nguy cơ khi train."""

    def __init__(self, summary: DatasetValidationSummary):
        self.summary = summary

    def analyze(self) -> List[TrainingRisk]:
        risks: List[TrainingRisk] = []
        self._risk_overfitting(risks)
        self._risk_class_bias(risks)
        self._risk_resolution(risks)
        self._risk_background(risks)
        self._risk_leakage(risks)
        self._risk_duplicate(risks)
        self._risk_small_object(risks)
        return risks

    def _risk_overfitting(self, risks):
        train_n = self.summary.train.total_images if self.summary.train else 0
        val_n = self.summary.val.total_images if self.summary.val else 0
        if val_n > 0 and train_n / val_n > 10:
            risks.append(TrainingRisk(risk_type="Overfitting", risk_level="High", probability=0.7, description=f"Train/Val {train_n/val_n:.1f}:1", mitigation="Tăng val set"))
        dup = len(self.summary.leakage_errors)
        if dup > 0:
            risks.append(TrainingRisk(risk_type="Overfitting", risk_level="Critical", probability=0.9, description=f"{dup} duplicate/leakage", mitigation="Xóa duplicate"))

    def _risk_class_bias(self, risks):
        combined: Dict[str, int] = {}
        for r in [self.summary.train, self.summary.val, self.summary.test]:
            if r is None:
                continue
            for cls, cnt in r.class_distribution.items():
                combined[cls] = combined.get(cls, 0) + cnt
        if combined:
            analysis = analyze_class_distribution(combined)
            if analysis["imbalance_ratio"] > 10:
                risks.append(TrainingRisk(risk_type="Class Bias", risk_level="High", probability=0.8, description=f"Imbalance {analysis['imbalance_ratio']:.1f}:1", mitigation="Oversampling"))
            if analysis["missing_classes"]:
                risks.append(TrainingRisk(risk_type="Class Missing", risk_level="Medium", probability=0.5, description=f"Thiếu: {', '.join(analysis['missing_classes'][:3])}", mitigation="Thu thập thêm"))

    def _risk_resolution(self, risks):
        low_q = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for m in r.quality_metrics.values() if m.quality_score < 30)
        total = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for _ in r.quality_metrics.values())
        if total > 0 and low_q / total > 0.1:
            risks.append(TrainingRisk(risk_type="Resolution Bias", risk_level="Medium", probability=0.5, description=f"{low_q} ảnh score<30", mitigation="Chuẩn hóa 640x640"))

    def _risk_background(self, risks):
        empty = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for e in r.errors if e.error_type == "EMPTY_LABEL")
        if empty > 0:
            risks.append(TrainingRisk(risk_type="Background Bias", risk_level="Low", probability=0.3, description=f"{empty} label rỗng", mitigation="Xác nhận intentional"))

    def _risk_leakage(self, risks):
        cross = sum(1 for e in self.summary.leakage_errors if e.error_type == "CROSS_SPLIT_LEAKAGE")
        if cross > 0:
            risks.append(TrainingRisk(risk_type="Leakage Risk", risk_level="Critical", probability=0.95, description=f"{cross} cross-split", mitigation="Xóa ngay"))

    def _risk_duplicate(self, risks):
        dup = sum(1 for e in self.summary.leakage_errors if e.error_type == "DUPLICATE_CONTENT")
        if dup > 0:
            risks.append(TrainingRisk(risk_type="Duplicate Bias", risk_level="Medium", probability=0.6, description=f"{dup} duplicate", mitigation="Xóa hoặc augmentation"))

    def _risk_small_object(self, risks):
        small = sum(1 for r in [self.summary.train, self.summary.val, self.summary.test] if r for e in r.errors if "too_small" in e.error_type.lower())
        total = sum(len(r.errors) for r in [self.summary.train, self.summary.val, self.summary.test] if r)
        if total > 0 and small / total > 0.3:
            risks.append(TrainingRisk(risk_type="Small Object Bias", risk_level="Medium", probability=0.5, description=f"{small}/{total} bbox quá nhỏ", mitigation="Multi-scale training"))


def _get_rating(score: float) -> str:
    if score >= 95:
        return "Excellent"
    elif score >= 85:
        return "Good"
    elif score >= 70:
        return "Fair"
    elif score >= 50:
        return "Poor"
    return "Critical"
