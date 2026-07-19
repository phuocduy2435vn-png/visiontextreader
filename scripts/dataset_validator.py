"""
dataset_validator.py — Entry point cho Dataset YOLO Validator.

Import tu cac module:
    models, hash_utils, analysis, image_processing, validators,
    audit, reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Them project root vao sys.path de import config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_project_paths
from models import ValidatorConfig
from validators import DatasetValidator
from reports import (
    HealthScoreCalculator, ValidationReport, BenchmarkSystem,
    DatasetVersioning, ChangelogGenerator, ReportComparator,
    VisualizationEngine, ErrorVisualizer, DatasetCleaner,
)
from audit import (
    DatasetAudit, RecommendationEngine, ScorecardGenerator,
    YOLOReadinessEvaluator, TrainingRiskAnalyzer,
)

logger = logging.getLogger("dataset_validator")


def print_summary(summary, score: float) -> None:
    """In bang tong tat validation ra man hinh."""
    sep = "=" * 70
    rating = HealthScoreCalculator.get_rating(score)
    print(f"\n{sep}\n  VISIONTEXTREADER — DATASET VALIDATION SUMMARY\n{sep}")
    print(f"  Health Score: {score}/100 ({rating})\n")

    def _v(r, a: str) -> str:
        return f"{getattr(r, a, 0):,}" if r else "N/A"

    t, v, te = summary.train, summary.val, summary.test
    print(f"  {'':>25s} {'Train':>12s} {'Val':>12s} {'Test':>12s}")
    print(f"  {'-'*65}")
    for label, attr in [("Total Images", "total_images"), ("Total Labels", "total_labels"), ("Valid Images", "valid_images"), ("Invalid Images", "invalid_images"), ("Blurry Images", "blurry_images"), ("Low Contrast", "low_contrast_images"), ("Overlapping BBox", "overlapping_bboxes")]:
        print(f"  {label:>25s} {_v(t, attr):>12s} {_v(v, attr):>12s} {_v(te, attr):>12s}")
    print(f"  {'Total Errors':>25s} {len(t.errors) if t else 0:>12,} {len(v.errors) if v else 0:>12,} {len(te.errors) if te else 0:>12,}")
    print(f"  {'Leakage Issues':>25s} {len(summary.leakage_errors):>12,}")
    print(f"\n  {'-'*65}")
    print(f"  {'Total Images':>25s} {sum(r.total_images for r in [t,v,te] if r):>12,}")
    print(f"  {'Total Errors':>25s} {sum(len(r.errors) for r in [t,v,te] if r):>12,}")
    print(f"{sep}\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Dataset YOLO Validator — VisionTextReader")
    parser.add_argument("--split", "-s", type=str, choices=["train", "val", "test"], help="Chi kiem tra split cu the")
    parser.add_argument("--train", action="store_true", help="Kiem tra train split")
    parser.add_argument("--val", action="store_true", help="Kiem tra val split")
    parser.add_argument("--test", action="store_true", help="Kiem tra test split")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--excel", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", "-o", type=str, default="outputs/validation")
    args = parser.parse_args()

    if args.split:
        if args.split == "train":
            args.train = True
        elif args.split == "val":
            args.val = True
        elif args.split == "test":
            args.test = True

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    paths = get_project_paths()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = paths.root / output_dir

    splits_to_check: Dict[str, Tuple[Path, Path]] = {}
    if args.train or args.val or args.test:
        if args.train:
            splits_to_check["train"] = (paths.train_images, paths.train_labels)
        if args.val:
            splits_to_check["val"] = (paths.val_images, paths.val_labels)
        if args.test:
            splits_to_check["test"] = (paths.test_images, paths.test_labels)
    else:
        splits_to_check["train"] = (paths.train_images, paths.train_labels)
        splits_to_check["val"] = (paths.val_images, paths.val_labels)
        splits_to_check["test"] = (paths.test_images, paths.test_labels)

    export_all = not (args.json or args.csv or args.excel or args.html or args.dashboard)
    if export_all:
        args.json = args.csv = args.excel = args.html = args.dashboard = True

    config = ValidatorConfig.load()
    config.workers = args.workers

    benchmark = BenchmarkSystem()
    benchmark.start()
    logger.info("Bat dau kiem tra chat luong dataset...")

    cache_dir = paths.root / ".validator_cache"
    validator = DatasetValidator(root_dir=paths.processed, cache_dir=cache_dir, workers=config.workers, config=config)
    summary = validator.process_all(splits_to_check)
    score = HealthScoreCalculator.calculate(summary)

    bench_result = benchmark.stop()
    total_imgs = sum(r.total_images for r in [summary.train, summary.val, summary.test] if r)
    total_lbls = sum(r.total_labels for r in [summary.train, summary.val, summary.test] if r)
    bench_result = benchmark.calculate_throughput(bench_result, total_imgs, total_lbls)
    logger.info("Hoan tat trong %.2fs (%.1f images/s, %.1f MB RAM)", bench_result.wall_time, bench_result.images_per_sec, bench_result.memory_mb)

    print_summary(summary, score)

    # Xuat bao cao
    report = ValidationReport(output_dir)
    if args.json:
        report.export_json(summary, score)
    if args.csv:
        report.export_csv(summary)
        report.export_quality_csv(summary)
        report.export_bbox_csv(summary)
        report.export_class_csv(summary)
    if args.excel:
        report.export_excel(summary, score)
    if args.html or args.dashboard:
        report.export_html(summary, score)
    report.export_txt(summary, score)

    # Audit, Recommendations, Scorecard, Readiness, Risks
    audit = DatasetAudit(summary, config)
    audit_findings = audit.run_audit()
    engine = RecommendationEngine(summary, audit_findings, config)
    recommendations = engine.generate()
    sc_gen = ScorecardGenerator(summary, config)
    scorecard = sc_gen.generate()
    readiness_eval = YOLOReadinessEvaluator(summary, scorecard)
    readiness = readiness_eval.evaluate()
    risk_analyzer = TrainingRiskAnalyzer(summary)
    risks = risk_analyzer.analyze()
    report.export_scorecard(scorecard, readiness, risks, audit_findings, recommendations)

    # Versioning & Changelog
    versioning = DatasetVersioning(output_dir / "versions")
    version = versioning.record_version(summary, score)
    changelog_gen = ChangelogGenerator(output_dir / "versions")
    changelog_gen.generate(version)

    # Report Comparison
    comparator = ReportComparator(output_dir)
    comparison = comparator.compare(summary, score)
    if comparison.get("has_previous"):
        logger.info("So sanh: score %s -> %s (%+.1f)", comparison["previous_score"], comparison["current_score"], comparison["score_change"])

    # Visualization
    viz = VisualizationEngine(output_dir / "charts")
    viz.generate_all(summary)

    # Error Visualizer
    error_viz = ErrorVisualizer(output_dir / "error_preview")
    img_dirs_map = {sn: idir for sn, (idir, _) in splits_to_check.items()}
    error_viz.visualize_errors(summary, img_dirs_map, max_images=50)

    # Dataset Cleaner Scripts
    cleaner = DatasetCleaner(output_dir / "cleanup_scripts")
    cleaner.generate_scripts(summary)

    # Benchmark Report
    bench_path = output_dir / "benchmark.json"
    bench_path.write_text(json.dumps({"cpu_time": bench_result.cpu_time, "wall_time": bench_result.wall_time, "memory_mb": bench_result.memory_mb, "images_per_sec": bench_result.images_per_sec, "labels_per_sec": bench_result.labels_per_sec}, indent=2), encoding="utf-8")
    logger.info("Benchmark: CPU=%.2fs Wall=%.2fs RAM=%.1fMB Throughput=%.1f img/s", bench_result.cpu_time, bench_result.wall_time, bench_result.memory_mb, bench_result.images_per_sec)


if __name__ == "__main__":
    main()
