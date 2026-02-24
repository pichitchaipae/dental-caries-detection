# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Week 9 — PCA Method Comparison Runner
=======================================

Iterates over PCA Methods [0, 1, 2, 3, 5] and runs the full 500-case
evaluation pipeline for each.  All outputs are organised strictly::

    week9_pca_evaluation/
        method_0_baseline_opencv/
            cases/                      ← per-case JSONs & debug images
            per_class_metrics.csv       ← 3-class metrics for this method
            evaluation_summary.json
            confusion_matrix_coarse.png
            confusion_matrix_fine.png
            eva_summarized.log          ← human-readable run summary
        method_1_square_heuristic/
            cases/ ...
        ...
        comparison_summary.csv          ← side-by-side method comparison
        per_class_dashboard.csv         ← aggregated from all method CSVs

Usage
-----
    cd week9_pca_evaluation
    python run_evaluation.py                  # full 500 cases, all methods
    python run_evaluation.py --sample 311,33  # quick test
    python run_evaluation.py --start 1 --end 50

Author: Senior Research Engineer – Dental AI / CAD
Date:   2026-02-23
"""

import os
import sys
import json
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd  # type: ignore

# ---------------------------------------------------------------------------
# Path configuration — resolve week8 module directory so imports work
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
WEEK8_DIR = BASE_DIR / "week8-Surface Classification v4"
WEEK9_DIR = Path(__file__).resolve().parent

# Add week8 to sys.path so we can import its modules
if str(WEEK8_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK8_DIR))

# Now import evaluation engine and classifier
from evaluation_engine import (  # noqa: E402
    evaluate_all_cases,
    ALLOWED_SURFACE_CLASSES,
)
from multi_zone_classifier import (  # noqa: E402
    set_pca_method,
    get_pca_method,
    get_pca_method_name,
    perform_pca,
    PCA_METHOD_NAMES,
    VALID_PCA_METHODS,
)

# Override the output directory to point to week9
import evaluation_engine as _ee  # noqa: E402
_ee.OUTPUT_DIR = WEEK9_DIR


# =============================================================================
# Methods to evaluate  (Method 4 is placeholder — excluded)
# =============================================================================

METHODS_TO_RUN = [0, 1, 2, 3, 5]


# =============================================================================
# Helper — write per-method summarized log
# =============================================================================

def _write_eva_log(
    method_dir: Path,
    method: int,
    method_name: str,
    summary: dict,
    elapsed_sec: float,
    warnings: list,
):
    """
    Write ``eva_summarized.log`` inside *method_dir* summarising the run.
    """
    log_path = method_dir / "eva_summarized.log"
    lines = [
        "=" * 70,
        f"Method {method} Evaluation Summary  ({method_name})",
        "=" * 70,
        f"  Timestamp          : {datetime.now().isoformat()}",
        f"  Execution time     : {elapsed_sec:.1f} s",
        "",
        f"  Cases processed    : {summary.get('cases_processed', '?')}",
        f"  Cases skipped      : {summary.get('cases_skipped', '?')}",
        f"  Total GT           : {summary.get('total_gt_annotations', '?')}",
        f"  Total Predictions  : {summary.get('total_predictions', '?')}",
        "",
        f"  TP / FP / FN       : {summary.get('TP', '?')} / "
        f"{summary.get('FP', '?')} / {summary.get('FN', '?')}",
        f"  Precision          : {summary.get('precision', '?')}",
        f"  Recall             : {summary.get('recall', '?')}",
        f"  F1-Score           : {summary.get('f1_score', '?')}",
        "",
        f"  Strict Accuracy    : {summary.get('surface_classification_accuracy_strict', '?')}",
        f"  Soft Accuracy      : {summary.get('surface_classification_accuracy_soft', '?')}",
        f"  Matched (3-class)  : {summary.get('n_matched_3class', '?')}",
        "",
    ]

    # Per-class block
    per_cls = summary.get("per_class_metrics", {})
    if per_cls:
        lines.append("  Per-Class Metrics:")
        lines.append(f"  {'Class':<12s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>9s}")
        lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
        for cls in ALLOWED_SURFACE_CLASSES:
            m = per_cls.get(cls, {})
            lines.append(
                f"  {cls:<12s} {m.get('precision', 0):>8.4f} "
                f"{m.get('recall', 0):>8.4f} {m.get('f1', 0):>8.4f} "
                f"{m.get('support', 0):>9d}"
            )
        lines.append("")

    # Failed cases
    failed = summary.get("failed_cases", [])
    if failed:
        lines.append(f"  Failed Cases ({len(failed)} total):")
        for fc in failed:
            lines.append(f"    - Case {fc['case']}: {fc['error']}")
        lines.append("")

    # Warnings / errors
    if warnings:
        lines.append("  Warnings / Errors:")
        for w in warnings:
            lines.append(f"    - {w}")
    else:
        lines.append("  No critical warnings or errors.")

    lines.append("=" * 70)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return log_path


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_all_methods(
    case_list=None,
    start: int = 1,
    end: int = 500,
    reclassify: bool = True,
    verbose: bool = True,
):
    """
    Iterate through PCA methods [0, 1, 2, 3, 5], evaluate each on
    cases *start*–*end* (default 1–500, no limits), and produce:

    - Per-method: ``method_X/cases/``, ``per_class_metrics.csv``,
      ``evaluation_summary.json``, ``eva_summarized.log``
    - Aggregated: ``comparison_summary.csv``, ``per_class_dashboard.csv``
    """
    total_start = time.time()

    print("=" * 70)
    print("WEEK 9 — PCA METHOD COMPARISON EVALUATION")
    print(f"  Date       : {datetime.now().isoformat()}")
    print(f"  Methods    : {METHODS_TO_RUN}")
    print(f"  Cases      : {start}–{end}" if case_list is None
          else f"  Cases      : {case_list}")
    print(f"  Output dir : {WEEK9_DIR}")
    print(f"  Strict rule: Surface Incorrect ∈ {ALLOWED_SURFACE_CLASSES}")
    print("=" * 70)

    all_summaries = []

    for method in METHODS_TO_RUN:
        method_name = PCA_METHOD_NAMES[method]
        method_dir = WEEK9_DIR / method_name
        cases_dir = method_dir / "cases"

        # ── Task 2: Create method_X/ AND method_X/cases/ ────────────
        os.makedirs(cases_dir, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"# PCA Method {method}: {method_name}")
        print(f"#   Output   : {method_dir}")
        print(f"#   Cases dir: {cases_dir}")
        print(f"{'#' * 70}\n")

        warnings_for_log = []
        method_start = time.time()

        # ── Task 3: Run evaluation — output routed to method_dir ─────
        # evaluate_all_cases writes to OUTPUT_DIR / method_name /
        # which is WEEK9_DIR / method_name / (we set _ee.OUTPUT_DIR above).
        # It creates cases/ inside that path automatically.
        try:
            df = evaluate_all_cases(
                case_list=case_list,
                start=start,
                end=end,
                reclassify=reclassify,
                verbose=verbose,
                pca_method=method,
            )
        except Exception as exc:
            msg = f"CRITICAL: evaluate_all_cases failed for method {method}: {exc}"
            print(f"  [ERROR] {msg}")
            warnings_for_log.append(msg)
            warnings_for_log.append(traceback.format_exc())
            continue

        method_elapsed = time.time() - method_start

        # ── Read back saved summary JSON ─────────────────────────────
        summary_path = method_dir / "evaluation_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            all_summaries.append(summary)
        else:
            summary = {}
            warnings_for_log.append("evaluation_summary.json not found after run")

        # ── Task 4: Write eva_summarized.log ─────────────────────────
        log_path = _write_eva_log(
            method_dir, method, method_name,
            summary, method_elapsed, warnings_for_log,
        )
        print(f"  [Saved] eva_summarized.log -> {log_path}")

    # ══════════════════════════════════════════════════════════════════
    # Task 5: Aggregate dashboard from per-method CSVs
    # ══════════════════════════════════════════════════════════════════

    # --- comparison_summary.csv (one row per method) ---
    if all_summaries:
        comp_df = pd.DataFrame(all_summaries)
        comp_path = WEEK9_DIR / "comparison_summary.csv"
        comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
        print(f"\n  [Saved] Comparison summary -> {comp_path}")

    # --- per_class_dashboard.csv (aggregated from each method_X/per_class_metrics.csv) ---
    per_class_dfs = []
    for method in METHODS_TO_RUN:
        method_name = PCA_METHOD_NAMES[method]
        csv_path = WEEK9_DIR / method_name / "per_class_metrics.csv"
        if csv_path.exists():
            per_class_dfs.append(pd.read_csv(csv_path))
        else:
            print(f"  [WARN] Missing per_class_metrics.csv for {method_name}")

    if per_class_dfs:
        dashboard_df = pd.concat(per_class_dfs, ignore_index=True)
        dashboard_path = WEEK9_DIR / "per_class_dashboard.csv"
        dashboard_df.to_csv(dashboard_path, index=False, encoding="utf-8-sig")
        print(f"  [Saved] Per-class dashboard -> {dashboard_path}")

    # ── Final comparison table ───────────────────────────────────────
    total_elapsed = time.time() - total_start
    if all_summaries:
        print("\n" + "=" * 70)
        print("WEEK 9 — PCA METHOD COMPARISON RESULTS")
        print("=" * 70)
        print(f"  {'Method':<30s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} "
              f"{'Strict':>8s} {'Soft':>8s}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
        for s in all_summaries:
            name = s.get("pca_method_name", "?")
            print(f"  {name:<30s} "
                  f"{s['precision']:>7.4f} "
                  f"{s['recall']:>7.4f} "
                  f"{s['f1_score']:>7.4f} "
                  f"{s['surface_classification_accuracy_strict']:>8.4f} "
                  f"{s['surface_classification_accuracy_soft']:>8.4f}")
        print(f"\n  Total time: {total_elapsed:.1f} s")
        print("=" * 70)

    return all_summaries


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Week 9 — PCA Method Comparison Evaluation Runner"
    )
    parser.add_argument("--start", type=int, default=1,
                        help="First case number (default: 1)")
    parser.add_argument("--end", type=int, default=500,
                        help="Last case number (default: 500)")
    parser.add_argument("--sample", type=str, default=None,
                        help="Comma-separated case numbers, e.g. --sample 311,33,100")
    parser.add_argument("--no-reclassify", action="store_true",
                        help="Skip multi-zone reclassification")

    args = parser.parse_args()

    case_list = None
    if args.sample:
        case_list = [int(c.strip()) for c in args.sample.split(",")]

    summaries = run_all_methods(
        case_list=case_list,
        start=args.start,
        end=args.end,
        reclassify=not args.no_reclassify,
    )

    # ── Auto-Visualization (runs AFTER all evaluation is complete) ────
    dashboard_csv = WEEK9_DIR / "per_class_dashboard.csv"
    if summaries and dashboard_csv.exists():
        print("\n" + "=" * 70)
        print("--- All cases across all methods completed ---")
        print("--- Generating Comparison Dashboards... ---")
        print("=" * 70)

        viz_script = WEEK9_DIR / "plot_dashboard.py"
        for metric in ["f1", "precision", "recall"]:
            result = subprocess.run(
                [sys.executable, str(viz_script), "--no-show", "--metric", metric],
                cwd=str(WEEK9_DIR),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  [OK] {metric} chart generated")
                if result.stdout.strip():
                    print(f"        {result.stdout.strip()}")
            else:
                print(f"  [WARN] {metric} chart failed: {result.stderr.strip()}")

        print("\n" + "=" * 70)
        print("--- Pipeline Finished! ---")
        print(f"    Results   : {WEEK9_DIR}")
        print(f"    Dashboard : {dashboard_csv}")
        print(f"    Charts    : f1_score_comparison.png, precision_score_comparison.png, recall_score_comparison.png")
        print("=" * 70)
    else:
        print("\n  [SKIP] Dashboard visualization skipped (no summaries or CSV missing).")


if __name__ == "__main__":
    main()
