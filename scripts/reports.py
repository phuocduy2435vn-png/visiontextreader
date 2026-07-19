"""
reports.py — Report export, visualization, error visualizer, dataset cleaner,
versioning, changelog, comparison, benchmark.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from config import IMAGE_EXTENSIONS, LABEL_EXTENSION
from models import (
    BenchmarkResult, ChangelogEntry, DatasetScorecard, DatasetValidationSummary,
    DatasetVersion, Severity, ValidationError, YOLOReadiness, TrainingRisk,
    AuditFinding, Recommendation, ScorecardEntry,
)

logger = logging.getLogger("dataset_validator")


class HealthScoreCalculator:
    """Tính Health Score 0-100."""
    WEIGHTS: Dict[str, float] = {
        "CORRUPTED_IMAGE": 15.0, "MISSING_LABEL": 10.0, "CROSS_SPLIT_LEAKAGE": 15.0,
        "DUPLICATE_CONTENT": 5.0, "BLURRY_IMAGE": 8.0, "LOW_CONTRAST": 5.0,
        "HIGH_IOU_OVERLAP": 8.0, "INVALID_CLASS_ID": 10.0, "UNKNOWN_CLASS": 10.0,
        "INVALID_BBOX": 8.0, "EMPTY_LABEL": 2.0, "DEAD_IMAGE": 10.0,
        "MISSING_IMAGE": 5.0, "COORDS_OUTSIDE_IMAGE": 5.0, "BBOX_OVERFLOW": 3.0,
        "DUPLICATE_ANNOTATION": 3.0,
    }

    @classmethod
    def calculate(cls, summary: DatasetValidationSummary) -> float:
        total_penalty = 0.0
        max_possible = sum(cls.WEIGHTS.values())
        ec: Counter = Counter()
        for r in [summary.train, summary.val, summary.test]:
            if r is None:
                continue
            for err in r.errors:
                ec[err.error_type] += 1
        for err in summary.leakage_errors:
            ec[err.error_type] += 1
        for et, cnt in ec.items():
            w = cls.WEIGHTS.get(et, 1.0)
            total_penalty += w * min(cnt, 100) / 100.0
        return round(max(0.0, 100.0 - (total_penalty / max_possible) * 100.0), 1) if max_possible > 0 else 100.0

    @classmethod
    def get_rating(cls, score: float) -> str:
        if score >= 95:
            return "Excellent"
        elif score >= 85:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 50:
            return "Poor"
        return "Critical"


class ValidationReport:
    """Xuất báo cáo validation."""
    SEP: str = "=" * 60
    LINE: str = "-" * 60

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_split_section(self, result) -> str:
        lines = [f"  {result.split_name.upper()}", f"  {self.LINE}",
                 f"    Tổng ảnh:              {result.total_images:>10,}",
                 f"    Tổng label:             {result.total_labels:>10,}",
                 f"    Ảnh hợp lệ:            {result.valid_images:>10,}",
                 f"    Ảnh không hợp lệ:      {result.invalid_images:>10,}",
                 f"    Ảnh mờ:                 {result.blurry_images:>10,}",
                 f"    Ảnh tương phản thấp:   {result.low_contrast_images:>10,}",
                 f"    BBox chồng lấn:         {result.overlapping_bboxes:>10,}",
                 f"    Tổng lỗi:              {len(result.errors):>10,}"]
        if result.class_distribution:
            lines.append("    Phân bố class:")
            for cn, cnt in sorted(result.class_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"      {cn:<20s} {cnt:>8,}")
        return "\n".join(lines)

    def export_txt(self, summary: DatasetValidationSummary, score: float) -> Path:
        lines = [self.SEP, "  VISIONTEXTREADER — DATASET VALIDATION REPORT", self.SEP,
                 f"  Health Score: {score}/100 ({HealthScoreCalculator.get_rating(score)})", ""]
        for r in [summary.train, summary.val, summary.test]:
            if r is not None:
                lines.extend([self._format_split_section(r), ""])
        if summary.leakage_errors:
            lines.extend([self.SEP, f"  LEAKAGE: {len(summary.leakage_errors)} lỗi", self.SEP])
            for err in summary.leakage_errors[:20]:
                lines.append(f"    [{err.severity}] {err.split}: {err.description}")
            lines.append("")
        lines.extend([self.SEP, f"  Fingerprint: {summary.fingerprint[:32]}...", f"  Time: {summary.metadata.get('elapsed_seconds', 'N/A')}s", self.SEP])
        p = self.output_dir / "validation_report.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Đã lưu TXT: %s", p)
        return p

    def export_json(self, summary: DatasetValidationSummary, score: float) -> Path:
        data = {"fingerprint": summary.fingerprint, "health_score": score, "health_rating": HealthScoreCalculator.get_rating(score), "metadata": summary.metadata, "leakage_count": len(summary.leakage_errors), "splits": {}}
        for sn in ["train", "val", "test"]:
            r = getattr(summary, sn)
            if r is None:
                continue
            data["splits"][sn] = {"total_images": r.total_images, "total_labels": r.total_labels, "valid_images": r.valid_images, "invalid_images": r.invalid_images, "blurry_images": r.blurry_images, "low_contrast_images": r.low_contrast_images, "overlapping_bboxes": r.overlapping_bboxes, "class_distribution": r.class_distribution, "error_count": len(r.errors), "error_types": dict(Counter(e.error_type for e in r.errors))}
        p = self.output_dir / "validation_report.json"
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Đã lưu JSON: %s", p)
        return p

    def export_csv(self, summary: DatasetValidationSummary) -> Path:
        p = self.output_dir / "validation_errors.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Split", "Image", "Label", "Error Type", "Severity", "Description", "Line"])
            for r in [summary.train, summary.val, summary.test]:
                if r is None:
                    continue
                for e in r.errors:
                    w.writerow([e.split, e.image_name, e.label_name, e.error_type, e.severity, e.description, e.line_number])
            for e in summary.leakage_errors:
                w.writerow([e.split, e.image_name, e.label_name, e.error_type, e.severity, e.description, e.line_number])
        logger.info("Đã lưu CSV: %s", p)
        return p

    def export_quality_csv(self, summary: DatasetValidationSummary) -> Path:
        rows = []
        for r in [summary.train, summary.val, summary.test]:
            if r is None:
                continue
            for stem, m in r.quality_metrics.items():
                rows.append({"Split": r.split_name, "Image": stem, "Score": m.quality_score, "Blur": round(m.blur_score, 2), "Brightness": round(m.brightness, 2), "Contrast": round(m.contrast, 2), "Entropy": round(m.entropy, 4), "Sharpness": round(m.sharpness, 2), "Noise": round(m.noise_estimate, 2), "SNR": round(m.snr, 2), "Dynamic Range": m.dynamic_range, "Over Exposed": round(m.over_exposed, 4), "Under Exposed": round(m.under_exposed, 4), "Saturated": round(m.saturated, 4), "Is Dead": m.is_dead, "Near Black": m.is_near_black, "Near White": m.is_near_white})
        p = self.output_dir / "quality.csv"
        if rows:
            pd.DataFrame(rows).to_csv(p, index=False)
        logger.info("Đã lưu Quality CSV: %s", p)
        return p

    def export_bbox_csv(self, summary: DatasetValidationSummary) -> Path:
        rows = [{"Split": r.split_name, "Label": e.label_name, "Type": e.error_type, "Description": e.description, "Line": e.line_number} for r in [summary.train, summary.val, summary.test] if r is not None for e in r.errors if e.error_type == "HIGH_IOU_OVERLAP"]
        p = self.output_dir / "bbox.csv"
        if rows:
            pd.DataFrame(rows).to_csv(p, index=False)
        logger.info("Đã lưu BBox CSV: %s", p)
        return p

    def export_class_csv(self, summary: DatasetValidationSummary) -> Path:
        all_cls: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in [summary.train, summary.val, summary.test]:
            if r is None:
                continue
            for cn, cnt in r.class_distribution.items():
                all_cls[cn][r.split_name] = cnt
        rows = [{"Class": cn, "Train": all_cls[cn].get("train", 0), "Val": all_cls[cn].get("val", 0), "Test": all_cls[cn].get("test", 0), "Total": sum(all_cls[cn].values())} for cn in sorted(all_cls.keys())]
        p = self.output_dir / "class_distribution.csv"
        if rows:
            pd.DataFrame(rows).to_csv(p, index=False)
        logger.info("Đã lưu Class CSV: %s", p)
        return p

    def export_excel(self, summary: DatasetValidationSummary, score: float) -> Path:
        p = self.output_dir / "validation_report.xlsx"
        with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
            rows = [{"Split": sn, "Total Images": r.total_images, "Total Labels": r.total_labels, "Valid": r.valid_images, "Invalid": r.invalid_images, "Blurry": r.blurry_images, "Low Contrast": r.low_contrast_images, "Overlap": r.overlapping_bboxes, "Errors": len(r.errors)} for sn in ["train", "val", "test"] if (r := getattr(summary, sn)) is not None]
            pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)
            all_errs = [{"Split": e.split, "Image": e.image_name, "Type": e.error_type, "Severity": e.severity, "Desc": e.description} for r in [summary.train, summary.val, summary.test] if r is not None for e in r.errors]
            if all_errs:
                pd.DataFrame(all_errs).to_excel(writer, sheet_name="Errors", index=False)
            pd.DataFrame([{"Score": score, "Rating": HealthScoreCalculator.get_rating(score)}]).to_excel(writer, sheet_name="Health", index=False)
            qr = [{"Split": r.split_name, "Image": stem, "Score": m.quality_score, "Blur": round(m.blur_score, 2)} for r in [summary.train, summary.val, summary.test] if r is not None for stem, m in r.quality_metrics.items()]
            if qr:
                pd.DataFrame(qr).to_excel(writer, sheet_name="Quality", index=False)
        logger.info("Đã lưu Excel: %s", p)
        return p

    def export_html(self, summary: DatasetValidationSummary, score: float) -> Path:
        rating = HealthScoreCalculator.get_rating(score)
        splits_js = {}
        for r in [summary.train, summary.val, summary.test]:
            if r is None:
                continue
            splits_js[r.split_name] = {"labels": list(r.class_distribution.keys())[:15], "values": list(r.class_distribution.values())[:15], "blurry": r.blurry_images, "contrast": r.low_contrast_images, "overlap": r.overlapping_bboxes, "total_images": r.total_images, "valid": r.valid_images, "invalid": r.invalid_images}
        leakage_list = [f"<li><b>[{e.severity}] {e.split.upper()}</b>: {e.description}</li>" for e in summary.leakage_errors[:20]]
        error_types_all = Counter()
        for r in [summary.train, summary.val, summary.test]:
            if r is None:
                continue
            for e in r.errors:
                error_types_all[e.error_type] += 1
        top_errors = error_types_all.most_common(20)
        all_scores = [m.quality_score for r in [summary.train, summary.val, summary.test] if r for m in r.quality_metrics.values()]
        avg_q = round(np.mean(all_scores), 1) if all_scores else 0.0
        html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisionTextReader Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{{--bg:#f8f9fa;--card:#fff;--text:#333;--muted:#666}}
[data-theme="dark"]{{--bg:#1a1a2e;--card:#16213e;--text:#e0e0e0;--muted:#aaa}}
body{{background:var(--bg);color:var(--text);transition:.3s}}
.hb{{font-size:2rem;padding:15px 30px;border-radius:15px;color:#fff;font-weight:bold}}
.h-excellent{{background:linear-gradient(135deg,#00C853,#009624)}}
.h-good{{background:linear-gradient(135deg,#4CAF50,#45a049)}}
.h-fair{{background:linear-gradient(135deg,#FF9800,#F57C00)}}
.h-poor{{background:linear-gradient(135deg,#FF5722,#D84315)}}
.h-critical{{background:linear-gradient(135deg,#f44336,#d32f2f)}}
.card{{border:none;border-radius:12px;background:var(--card);transition:transform .2s}}
.card:hover{{transform:translateY(-2px)}}
.mc{{text-align:center;padding:20px}}.mv{{font-size:1.8rem;font-weight:bold;color:var(--text)}}.ml{{color:var(--muted);font-size:.9rem}}
.sb{{min-height:100vh;background:var(--card)}}.sb a{{color:var(--muted);text-decoration:none;padding:8px 15px;display:block;border-radius:8px}}.sb a:hover{{background:rgba(0,0,0,.05);color:var(--text)}}
.tt{{cursor:pointer;font-size:1.5rem}}
</style></head>
<body><div class="container-fluid"><div class="row">
<div class="col-md-2 sb d-none d-md-block p-3"><h5 class="mb-4">Dashboard</h5>
<a href="#overview">Overview</a><a href="#health">Health Score</a><a href="#quality">Quality</a>
<a href="#class">Class Distribution</a><a href="#errors">Errors</a><a href="#leakage">Leakage</a><hr>
<div class="tt" onclick="document.body.toggleAttribute('data-theme')">&#x1F313;</div></div>
<div class="col-md-10 p-4">
<h1 class="text-center mb-2">Dataset QA Dashboard</h1>
<p class="text-center text-muted">FP: <code>{summary.fingerprint[:32]}...</code></p>
<div id="health" class="text-center mb-4"><span class="hb h-{rating.lower()}">{score}/100 — {rating}</span><p class="mt-2 text-muted">Avg Quality: {avg_q}/100</p></div>
<div id="overview" class="row mb-4">
<div class="col-md-2"><div class="card mc shadow-sm"><div class="mv">{sum(r.total_images for r in [summary.train,summary.val,summary.test] if r):,}</div><div class="ml">Total</div></div></div>
<div class="col-md-2"><div class="card mc shadow-sm"><div class="mv text-success">{sum(r.valid_images for r in [summary.train,summary.val,summary.test] if r):,}</div><div class="ml">Valid</div></div></div>
<div class="col-md-2"><div class="card mc shadow-sm"><div class="mv text-danger">{sum(r.invalid_images for r in [summary.train,summary.val,summary.test] if r):,}</div><div class="ml">Invalid</div></div></div>
<div class="col-md-2"><div class="card mc shadow-sm"><div class="mv text-warning">{sum(r.blurry_images for r in [summary.train,summary.val,summary.test] if r):,}</div><div class="ml">Blurry</div></div></div>
<div class="col-md-2"><div class="card mc shadow-sm"><div class="mv text-info">{sum(r.low_contrast_images for r in [summary.train,summary.val,summary.test] if r):,}</div><div class="ml">Low Contrast</div></div></div>
<div class="col-md-2"><div class="card mc shadow-sm"><div class="mv text-secondary">{len(summary.leakage_errors)}</div><div class="ml">Leakage</div></div></div></div>
<div id="leakage">{f'<div class="alert alert-danger"><h5>Leakage: {len(summary.leakage_errors)}</h5><ul>{"".join(leakage_list)}</ul></div>' if summary.leakage_errors else '<div class="alert alert-success">No leakage.</div>'}</div>
<div class="row mb-4"><div class="col-md-6"><div class="card p-3 shadow-sm"><h5>Quality Radar</h5><canvas id="qc"></canvas></div></div>
<div class="col-md-6"><div class="card p-3 shadow-sm"><h5>Split Distribution</h5><canvas id="sc"></canvas></div></div></div>
<div id="class" class="row mb-4"><div class="col-md-6"><div class="card p-3 shadow-sm"><h5>Class Distribution</h5><canvas id="cc"></canvas></div></div>
<div class="col-md-6"><div class="card p-3 shadow-sm"><h5>Quality Scores</h5><canvas id="qs"></canvas></div></div></div>
<div id="errors" class="row mb-4"><div class="col-md-12"><div class="card p-3 shadow-sm"><h5>Top 20 Errors</h5><canvas id="ec"></canvas></div></div></div>
</div></div></div>
<script>
const d={json.dumps(splits_js)};const te={json.dumps([{"type":t,"count":c}for t,c in top_errors])};
if(d.train){{new Chart(document.getElementById('cc'),{{type:'bar',data:{{labels:d.train.labels,datasets:[{{label:'Train',data:d.train.values,backgroundColor:'rgba(54,162,235,.7)'}}]}},options:{{responsive:true,plugins:{{legend:{{display:false}}}}}}}});}}
new Chart(document.getElementById('qc'),{{type:'radar',data:{{labels:['Blurry','Low Contrast','Overlap','Leakage','Invalid'],datasets:[{{label:'Issues',data:[{sum(r.blurry_images for r in [summary.train,summary.val,summary.test] if r)},{sum(r.low_contrast_images for r in [summary.train,summary.val,summary.test] if r)},{sum(r.overlapping_bboxes for r in [summary.train,summary.val,summary.test] if r)},{len(summary.leakage_errors)},{sum(r.invalid_images for r in [summary.train,summary.val,summary.test] if r)}],backgroundColor:'rgba(255,99,132,.2)',borderColor:'rgba(255,99,132,1)'}}]}}}});
new Chart(document.getElementById('sc'),{{type:'doughnut',data:{{labels:Object.keys(d),datasets:[{{data:Object.values(d).map(v=>v.total_images),backgroundColor:['#36A2EB','#FFCE56','#4BC0C0']}}]}}}});
new Chart(document.getElementById('qs'),{{type:'bar',data:{{labels:['0-20','20-40','40-60','60-80','80-100'],datasets:[{{label:'Images',data:[0,0,0,0,0],backgroundColor:'rgba(75,192,192,.7)'}}]}},options:{{responsive:true}}}});
new Chart(document.getElementById('ec'),{{type:'bar',data:{{labels:te.map(e=>e.type),datasets:[{{label:'Count',data:te.map(e=>e.count),backgroundColor:'rgba(255,159,64,.7)'}}]}},options:{{indexAxis:'y',responsive:true}}}});
</script></body></html>"""
        op = self.output_dir / "validation_dashboard.html"
        op.write_text(html, encoding="utf-8")
        logger.info("Đã lưu Dashboard: %s", op)
        return op

    def export_scorecard(self, scorecard: DatasetScorecard, readiness: YOLOReadiness, risks: List[TrainingRisk], audit_findings: List[AuditFinding], recommendations: List[Recommendation]) -> Path:
        lines = [self.SEP, "  DATASET SCORECARD", self.SEP, ""]
        for e in scorecard.entries:
            bar = "#" * int(e.score / 5) + "." * (20 - int(e.score / 5))
            lines.append(f"    {e.category:<25s} {bar} {e.score:>5.1f}/100  {e.details}")
        lines.extend(["", f"    {'OVERALL':<25s} {'':>20s} {scorecard.overall_score:>5.1f}/100  ({scorecard.grade})", ""])
        lines.extend([self.SEP, "  YOLO READINESS", self.SEP, f"    Score: {readiness.score}/100 — {readiness.level}", ""])
        if readiness.blocking_issues:
            lines.append("    BLOCKING ISSUES:")
            for bi in readiness.blocking_issues:
                lines.append(f"    X {bi}")
        if readiness.reasons:
            lines.append("    REASONS:")
            for ri in readiness.reasons:
                lines.append(f"      * {ri}")
        lines.append("")
        lines.extend([self.SEP, "  TRAINING RISKS", self.SEP])
        if risks:
            for rk in risks:
                lines.append(f"    [{rk.risk_level}] {rk.risk_type}: {rk.description} -> {rk.mitigation}")
        else:
            lines.append("    Không phát hiện rủi ro lớn")
        lines.append("")
        lines.extend([self.SEP, "  AI AUDIT FINDINGS", self.SEP])
        for af in audit_findings:
            icon = "+" if af.severity == Severity.INFO else "!" if af.severity == Severity.WARNING else "X"
            lines.append(f"    [{icon}] [{af.severity}] {af.finding}")
        lines.append("")
        lines.extend([self.SEP, "  RECOMMENDATIONS", self.SEP])
        for rec in recommendations:
            lines.append(f"    [{rec.priority}] {rec.action} — {rec.reason}")
        lines.append(self.SEP)
        p = self.output_dir / "scorecard_report.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Đã lưu Scorecard: %s", p)
        return p


class DatasetVersioning:
    """Quản lý phiên bản dataset."""

    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.version_dir / "version_history.json"

    def record_version(self, summary: DatasetValidationSummary, score: float) -> DatasetVersion:
        total_imgs = sum(r.total_images for r in [summary.train, summary.val, summary.test] if r)
        total_lbls = sum(r.total_labels for r in [summary.train, summary.val, summary.test] if r)
        version = DatasetVersion(version=datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S"), fingerprint=summary.fingerprint, validation_time=datetime.now(timezone.utc).isoformat(), dataset_size=total_imgs, total_images=total_imgs, total_labels=total_lbls, health_score=score)
        history = self._load_history()
        if history:
            last = history[-1]
            version.changed_images = abs(total_imgs - last.get("total_images", 0))
            prev_fp = last.get("fingerprint", "")
            if prev_fp and prev_fp != summary.fingerprint:
                version.added_images = max(0, total_imgs - last.get("total_images", 0))
                version.removed_images = max(0, last.get("total_images", 0) - total_imgs)
        history.append(asdict(version))
        self._save_history(history)
        return version

    def _load_history(self) -> List[dict]:
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_history(self, history: List[dict]):
        self.history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


class ChangelogGenerator:
    """Sinh changelog."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, current: DatasetVersion, previous: Optional[DatasetVersion] = None) -> List[ChangelogEntry]:
        entries: List[ChangelogEntry] = []
        if previous is None:
            entries.append(ChangelogEntry(change_type="INIT", count=current.total_images, description=f"Dataset ban đầu: {current.total_images} ảnh"))
        else:
            if current.added_images > 0:
                entries.append(ChangelogEntry(change_type="ADDED", count=current.added_images, description=f"+ {current.added_images} ảnh mới"))
            if current.removed_images > 0:
                entries.append(ChangelogEntry(change_type="REMOVED", count=current.removed_images, description=f"- {current.removed_images} ảnh bị xóa"))
            if current.health_score != previous.health_score:
                diff = current.health_score - previous.health_score
                sign = "+" if diff > 0 else ""
                entries.append(ChangelogEntry(change_type="SCORE_CHANGE", count=0, description=f"Health Score: {previous.health_score} -> {current.health_score} ({sign}{diff:.1f})"))
        if entries:
            changelog_path = self.output_dir / f"changelog_{current.version}.txt"
            lines = [f"=== CHANGELOG {current.version} ===", f"Time: {current.validation_time}", f"Fingerprint: {current.fingerprint[:32]}...", ""]
            for e in entries:
                lines.append(f"[{e.change_type}] {e.description}")
            changelog_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Đã lưu changelog: %s", changelog_path)
        return entries


class ReportComparator:
    """So sánh report hiện tại với report cũ."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def compare(self, current_summary: DatasetValidationSummary, current_score: float) -> Dict[str, Any]:
        prev_file = self.output_dir / "validation_report.json"
        if not prev_file.exists():
            return {"has_previous": False, "message": "Không có report cũ"}
        try:
            prev_data = json.loads(prev_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {"has_previous": False, "message": "Không đọc được report cũ"}
        prev_score = prev_data.get("health_score", 0)
        return {"has_previous": True, "previous_score": prev_score, "current_score": current_score, "score_change": round(current_score - prev_score, 1), "previous_fingerprint": prev_data.get("fingerprint", "")[:32], "current_fingerprint": current_summary.fingerprint[:32], "changed": prev_data.get("fingerprint", "") != current_summary.fingerprint}


class VisualizationEngine:
    """Sinh biểu đồ thống kê."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, summary: DatasetValidationSummary) -> List[Path]:
        saved: List[Path] = []
        saved.append(self._class_distribution_chart(summary))
        saved.append(self._quality_histogram(summary))
        saved.append(self._split_distribution_chart(summary))
        return [p for p in saved if p is not None]

    def _class_distribution_chart(self, summary: DatasetValidationSummary) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            combined: Dict[str, int] = {}
            for r in [summary.train, summary.val, summary.test]:
                if r is None:
                    continue
                for cls, cnt in r.class_distribution.items():
                    combined[cls] = combined.get(cls, 0) + cnt
            if not combined:
                return None
            fig, ax = plt.subplots(figsize=(12, 6))
            sorted_cls = sorted(combined.items(), key=lambda x: -x[1])
            names = [c[0] for c in sorted_cls[:20]]
            counts = [c[1] for c in sorted_cls[:20]]
            ax.barh(names[::-1], counts[::-1], color="#2196F3")
            ax.set_xlabel("Soluong annotation")
            ax.set_title("Class Distribution")
            plt.tight_layout()
            path = self.output_dir / "class_distribution.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        except ImportError:
            return None

    def _quality_histogram(self, summary: DatasetValidationSummary) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            scores = [m.quality_score for r in [summary.train, summary.val, summary.test] if r for m in r.quality_metrics.values()]
            if not scores:
                return None
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(scores, bins=20, color="#4CAF50", edgecolor="white")
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("So anh")
            ax.set_title("Image Quality Distribution")
            ax.axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.1f}")
            ax.legend()
            plt.tight_layout()
            path = self.output_dir / "quality_histogram.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        except ImportError:
            return None

    def _split_distribution_chart(self, summary: DatasetValidationSummary) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            data = {}
            for sn in ["train", "val", "test"]:
                r = getattr(summary, sn)
                if r is not None:
                    data[sn] = r.total_images
            if not data:
                return None
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(data.values(), labels=data.keys(), autopct="%1.1f%%", colors=["#36A2EB", "#FFCE56", "#4BC0C0"])
            ax.set_title("Split Distribution")
            plt.tight_layout()
            path = self.output_dir / "split_distribution.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        except ImportError:
            return None


class ErrorVisualizer:
    """Vẽ bounding box lên ảnh lỗi."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_errors(self, summary: DatasetValidationSummary, img_dirs: Dict[str, Path], max_images: int = 50) -> int:
        count = 0
        error_types = {"BLURRY_IMAGE", "LOW_CONTRAST", "DEAD_IMAGE", "NEAR_BLACK_IMAGE", "NEAR_WHITE_IMAGE"}
        for r in [summary.train, summary.val, summary.test]:
            if r is None or count >= max_images:
                break
            for err in r.errors:
                if count >= max_images:
                    break
                if err.error_type not in error_types or not err.image_name or err.image_name.endswith(".*"):
                    continue
                img_dir = img_dirs.get(r.split_name)
                if img_dir is None:
                    continue
                img_path = img_dir / err.image_name
                if img_path.exists():
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        color_map = {"BLURRY_IMAGE": (0, 0, 255), "LOW_CONTRAST": (0, 165, 255), "DEAD_IMAGE": (128, 0, 128), "NEAR_BLACK_IMAGE": (0, 128, 128), "NEAR_WHITE_IMAGE": (255, 255, 0)}
                        color = color_map.get(err.error_type, (0, 255, 0))
                        cv2.rectangle(img, (5, 5), (w - 5, h - 5), color, 3)
                        cv2.putText(img, err.error_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(img, err.description[:50], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        out_path = self.output_dir / f"error_{count:04d}_{err.error_type}_{err.image_name}"
                        cv2.imwrite(str(out_path), img)
                        count += 1
                    except Exception:
                        pass
        logger.info("Đã tạo %d ảnh error preview", count)
        return count


class DatasetCleaner:
    """Sinh script dọn dẹp dataset."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_scripts(self, summary: DatasetValidationSummary) -> List[Path]:
        scripts: List[Path] = []
        scripts.append(self._gen_remove_blurry(summary))
        scripts.append(self._gen_remove_empty_labels(summary))
        scripts.append(self._gen_remove_duplicates(summary))
        return [s for s in scripts if s is not None]

    def _gen_remove_blurry(self, summary: DatasetValidationSummary) -> Path:
        blurry_files = [e.image_name for r in [summary.train, summary.val, summary.test] if r is not None for e in r.errors if e.error_type == "BLURRY_IMAGE" and e.image_name]
        script = f"""#!/usr/bin/env python3
# Auto-generated: Xoa anh mo
from pathlib import Path
BLURRY = {json.dumps(blurry_files[:100])}
DIRS = ["datasets/processed/train/images", "datasets/processed/val/images", "datasets/processed/test/images"]
removed = 0
for d in DIRS:
    dp = Path(d)
    if not dp.exists():
        continue
    for f in dp.iterdir():
        if f.name in BLURRY:
            print(f"Xoa: {{f}}")
            # f.unlink()
            removed += 1
print(f"Tong: {{removed}} file")
"""
        path = self.output_dir / "remove_blurry.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _gen_remove_empty_labels(self, summary: DatasetValidationSummary) -> Path:
        empty_files = [e.label_name for r in [summary.train, summary.val, summary.test] if r is not None for e in r.errors if e.error_type == "EMPTY_LABEL" and e.label_name]
        script = f"""#!/usr/bin/env python3
# Auto-generated: Xoa label trong
from pathlib import Path
EMPTY = {json.dumps(empty_files[:100])}
DIRS = ["datasets/processed/train/labels", "datasets/processed/val/labels", "datasets/processed/test/labels"]
removed = 0
for d in DIRS:
    dp = Path(d)
    if not dp.exists():
        continue
    for f in dp.iterdir():
        if f.name in EMPTY:
            print(f"Xoa: {{f}}")
            # f.unlink()
            removed += 1
print(f"Tong: {{removed}} label trong")
"""
        path = self.output_dir / "remove_empty_labels.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _gen_remove_duplicates(self, summary: DatasetValidationSummary) -> Path:
        dup_files = list({e.image_name.replace(".*", "") for e in summary.leakage_errors if e.image_name})
        script = f"""#!/usr/bin/env python3
# Auto-generated: Xoa duplicate
from pathlib import Path
DUP = {json.dumps(dup_files[:100])}
DIRS = ["datasets/processed/train/images", "datasets/processed/val/images", "datasets/processed/test/images"]
removed = 0
for d in DIRS:
    dp = Path(d)
    if not dp.exists():
        continue
    for f in dp.iterdir():
        for dup in DUP:
            if dup in f.name:
                print(f"Xoa: {{f}}")
                # f.unlink()
                removed += 1
                break
print(f"Tong: {{removed}} duplicate")
"""
        path = self.output_dir / "remove_duplicates.py"
        path.write_text(script, encoding="utf-8")
        return path


class BenchmarkSystem:
    """Do hieu nang."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.start_process_time: float = 0.0

    def start(self):
        self.start_time = time.time()
        self.start_process_time = time.process_time()

    def stop(self) -> BenchmarkResult:
        self.end_time = time.time()
        wall_time = self.end_time - self.start_time
        cpu_time = time.process_time() - self.start_process_time
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            mem_mb = 0.0
        return BenchmarkResult(cpu_time=round(cpu_time, 2), wall_time=round(wall_time, 2), memory_mb=round(mem_mb, 1))

    def calculate_throughput(self, benchmark: BenchmarkResult, total_images: int, total_labels: int) -> BenchmarkResult:
        if benchmark.wall_time > 0:
            benchmark.images_per_sec = round(total_images / benchmark.wall_time, 1)
            benchmark.labels_per_sec = round(total_labels / benchmark.wall_time, 1)
        return benchmark
