"""
Week 7 – Test Runner for Cases 311 & 33
========================================

Runs **both** the dental_caries_analysis (Task 2+3) and the
evaluation_engine (Task 1+4) on the two target cases, then prints
a concise summary highlighting each fix.

Usage:
    python run_test_sample.py
"""

import sys
from pathlib import Path

# Ensure week7 directory is on the path
WEEK7_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEEK7_DIR))

from dental_caries_analysis import process_single_case
from evaluation_engine import evaluate_single_case, evaluate_all_cases, soft_surface_match


BASE_DIR = WEEK7_DIR.parent
JSON_DIR   = BASE_DIR / "week2" / "500-segmentation+recognition"
ROI_DIR    = BASE_DIR / "material" / "500-roi"
OUTPUT_DIR = WEEK7_DIR / "dental_analysis_output"

SAMPLE_CASES = [311, 33]


def run_analysis_phase():
    """Run the week7 caries analysis (erosion + unassigned blobs)."""
    print("\n" + "=" * 70)
    print("PHASE 1 – Dental Caries Analysis (Task 2: Erosion, Task 3: Unassigned)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case_num in SAMPLE_CASES:
        success, msg, results = process_single_case(
            case_num, JSON_DIR, ROI_DIR, OUTPUT_DIR
        )
        status = "OK" if success else "FAIL"
        print(f"  Case {case_num}: [{status}] {msg}")

        if results:
            tooth_results = [r for r in results if r.get("tooth_id") != "UNASSIGNED"]
            unassigned    = [r for r in results if r.get("tooth_id") == "UNASSIGNED"]

            # Task 2 summary: teeth that WERE caries in week3 but now filtered
            for r in tooth_results:
                if r["caries_pixels"] == 0 and r.get("total_pixels", 0) > 0:
                    pass  # silently filtered by erosion + threshold
                elif r["has_caries"]:
                    print(f"    Tooth {r['tooth_id']}: {r['caries_pixels']} caries px "
                          f"({r['caries_percentage']:.2f}%)")

            # Task 3 summary
            if unassigned:
                for u in unassigned:
                    print(f"    [UNASSIGNED BLOB] {u.get('caries_pixels', '?')} px "
                          f"at centroid {u.get('unassigned_centroid', '?')}")
            else:
                print(f"    (No unassigned caries blobs)")


def run_evaluation_phase():
    """Run the week7 evaluation engine (PCA fix + soft matching)."""
    print("\n" + "=" * 70)
    print("PHASE 2 – Evaluation Engine (Task 1: PCA fix, Task 4: Soft Match)")
    print("=" * 70)

    df = evaluate_all_cases(
        case_list=SAMPLE_CASES,
        reclassify=True,
        verbose=True,
    )

    if df is not None and len(df) > 0:
        matched_df = df[df["match_type"].isin(["tooth_id", "centroid_proximity"])]
        if len(matched_df) > 0:
            print("\n  Per-tooth matching detail:")
            for _, row in matched_df.iterrows():
                soft = row.get("surface_match", "False")
                strict = row.get("surface_match_strict", False)
                flag = ""
                if soft in ("Partial", "True_with_warning") and not strict:
                    flag = " ← SOFT MATCH (Task 4 rescued)"
                elif soft == "True" and not strict:
                    flag = " ← COARSE MATCH"

                print(f"    Case {int(row['case'])} | Tooth {row['tooth_id']:>2s} | "
                      f"GT={row['gt_surface']:<12s} Pred={row.get('pred_surface', ''):<12s} | "
                      f"strict={strict}  soft={soft}{flag}")


def main():
    print("=" * 70)
    print("WEEK 7 – Test Runner  (sample cases: 311, 33)")
    print("  Task 1: PCA Eigenvector Swap fix (square molars)")
    print("  Task 2: Boundary Leakage fix (erosion + size threshold)")
    print("  Task 3: Missing Upstream fallback (unassigned caries blobs)")
    print("  Task 4: Soft / Partial Surface Matching")
    print("=" * 70)

    run_analysis_phase()
    run_evaluation_phase()

    print("\n✓ All done. Check week7/dental_analysis_output/ and week7/evaluation_output/")


if __name__ == "__main__":
    main()
