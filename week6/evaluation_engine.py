"""
Dental Caries Evaluation Engine
================================

End-to-end evaluation pipeline that:

1. Parses AIM-XML ground truth from ``material/500 cases with annotation``
2. Loads PCA-based predictions from ``week5/surface_classification_output``
3. Enriches week5 JSON with ``caries_coordinates`` from week3
4. Matches ground truth ↔ predictions using ROI centroid distance
5. Reclassifies caries with the Multi-Zone (M-C-D) classifier
6. Computes TP / FP / FN, Confusion Matrix, Precision, Recall, F1

Outputs (written to ``week6/evaluation_output/``):
    - evaluation_results.csv          per-lesion match table
    - confusion_matrix.png            surface confusion matrix
    - evaluation_summary.json         aggregated metrics
    - enriched_<case>_diagnosis.json  week5 JSON + caries_coordinates

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026
"""

import os
import sys
import json
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules (same directory)
from xml_ground_truth_parser import parse_case_xmls
from multi_zone_classifier import classify_multi_zone, classify_from_week_data
from snodent_tooth_map import FDI_TOOTH_NAMES

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")

# Inputs
MATERIAL_DIR   = BASE_DIR / "material" / "500 cases with annotation"
WEEK2_DIR      = BASE_DIR / "week2"  / "500-segmentation+recognition"
WEEK3_DIR      = BASE_DIR / "week3"  / "dental_analysis_output"
WEEK5_DIR      = BASE_DIR / "week5"  / "surface_classification_output"

# Output
WEEK6_DIR      = BASE_DIR / "week6"
OUTPUT_DIR     = WEEK6_DIR / "evaluation_output"

# Matching parameters
DISTANCE_THRESHOLD_PX = 60.0   # max distance (pixels) for centroid matching
IOU_THRESHOLD         = 0.10   # min IoU for polygon-based matching (backup)


# =============================================================================
# Surface Label Normalization
# =============================================================================

def normalize_surface(raw: str) -> str:
    """
    Map any surface string to one of the evaluation categories:
        Occlusal | Proximal | Other
    
    Proximal includes Mesial and Distal.
    """
    r = raw.strip().lower()
    if r in ("occlusal",):
        return "Occlusal"
    if r in ("mesial", "distal", "proximal", "proximal-mesial", "proximal-distal"):
        return "Proximal"
    if r in ("buccal", "lingual", "palatal", "cervical", "lingual/other", "other"):
        return "Other"
    # Fallback: check keywords
    if "occlusal" in r:
        return "Occlusal"
    if "mesial" in r or "distal" in r or "proximal" in r:
        return "Proximal"
    return "Other"


def normalize_surface_fine(raw: str) -> str:
    """
    Finer-grained normalization preserving Mesial / Distal / Occlusal.
    """
    r = raw.strip().lower()
    if "occlusal" in r:
        return "Occlusal"
    if "mesial" in r:
        return "Mesial"
    if "distal" in r:
        return "Distal"
    if "proximal" in r:
        return "Proximal"
    if "buccal" in r:
        return "Buccal"
    if "lingual" in r:
        return "Lingual"
    return "Other"


# =============================================================================
# Data Loaders
# =============================================================================

def load_week3_caries(case_num: int) -> Dict:
    """Load week3 caries mapping JSON → {tooth_id: {data}}."""
    p = WEEK3_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for t in data.get("teeth_caries_data", []):
        tid = t.get("tooth_id", "")
        if tid:
            lookup[tid] = t
    return lookup


def load_week2_tooth_polygons(case_num: int) -> Dict:
    """Load week2 segmentation → {tooth_id: polygon}."""
    p = WEEK2_DIR / f"case {case_num}" / f"case_{case_num}_results.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    lookup = {}
    for tooth in data.get("teeth_data", []):
        tid = tooth.get("tooth_id", "")
        if not tid:
            continue
        poly = tooth.get("polygon")
        if not poly:
            pixels = tooth.get("pixel_coordinates")
            if pixels and len(pixels) >= 3:
                arr = np.array(pixels, dtype=np.float32)
                hull = cv2.convexHull(arr)
                poly = hull.reshape(-1, 2).tolist()
        if poly:
            lookup[tid] = poly
    return lookup


def load_week5_predictions(case_num: int) -> List[Dict]:
    """Load week5 diagnosis JSON → list of tooth predictions."""
    p = WEEK5_DIR / f"case {case_num}" / f"case_{case_num}_diagnosis.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("teeth_data", [])


# =============================================================================
# Ground Truth ↔ Prediction Matching
# =============================================================================

def _point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _polygon_iou(poly_a: List[Tuple[float, float]],
                 poly_b: List[Tuple[float, float]]) -> float:
    """
    Compute Intersection-over-Union of two polygons using pixel rasterization.
    Falls back to 0.0 on error.
    """
    try:
        import cv2
        arr_a = np.array(poly_a, dtype=np.int32)
        arr_b = np.array(poly_b, dtype=np.int32)
        all_pts = np.vstack([arr_a, arr_b])
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        off = np.array([x_min, y_min])
        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_a, [arr_a - off], 1)
        cv2.fillPoly(mask_b, [arr_b - off], 1)
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def match_gt_to_predictions(
    gt_annotations: List[Dict],
    predictions: List[Dict],
    week3_lookup: Dict,
    distance_threshold: float = DISTANCE_THRESHOLD_PX,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match ground truth annotations to predictions.

    Strategy:
        1. First try tooth_id exact match
        2. Among same-tooth matches, pick the one with smallest centroid distance
        3. If no tooth_id match, try centroid proximity (< threshold)

    Returns:
        (matched_pairs, false_positives, false_negatives)
    """
    matched: List[Dict] = []
    used_gt_idx = set()
    used_pred_idx = set()

    # Build prediction centroid lookup using week3 caries centroids
    pred_centroids = {}
    for i, pred in enumerate(predictions):
        tid = pred.get("tooth_id", "")
        w3 = week3_lookup.get(tid, {})
        coords = w3.get("caries_coordinates", [])
        if coords:
            arr = np.array(coords, dtype=np.float64)
            pred_centroids[i] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
        else:
            pred_centroids[i] = None

    # ── Pass 1: Tooth-ID match ───────────────────────────────────────
    for gi, gt in enumerate(gt_annotations):
        gt_fdi = gt.get("tooth_fdi", "")
        gt_cent = gt.get("roi_centroid", (0, 0))
        if not gt_fdi:
            continue

        best_pi = None
        best_dist = float("inf")
        for pi, pred in enumerate(predictions):
            if pi in used_pred_idx:
                continue
            if pred.get("tooth_id", "") != gt_fdi:
                continue
            pc = pred_centroids.get(pi)
            if pc is not None:
                d = _point_distance(gt_cent, pc)
            else:
                d = 0  # same tooth, no coords → accept
            if d < best_dist:
                best_dist = d
                best_pi = pi

        if best_pi is not None:
            matched.append({
                "gt_idx": gi,
                "pred_idx": best_pi,
                "gt": gt,
                "pred": predictions[best_pi],
                "match_type": "tooth_id",
                "distance_px": round(best_dist, 2),
            })
            used_gt_idx.add(gi)
            used_pred_idx.add(best_pi)

    # ── Pass 2: Centroid proximity for unmatched GT ──────────────────
    for gi, gt in enumerate(gt_annotations):
        if gi in used_gt_idx:
            continue
        gt_cent = gt.get("roi_centroid", (0, 0))
        best_pi = None
        best_dist = float("inf")
        for pi, pred in enumerate(predictions):
            if pi in used_pred_idx:
                continue
            pc = pred_centroids.get(pi)
            if pc is None:
                continue
            d = _point_distance(gt_cent, pc)
            if d < distance_threshold and d < best_dist:
                best_dist = d
                best_pi = pi

        if best_pi is not None:
            matched.append({
                "gt_idx": gi,
                "pred_idx": best_pi,
                "gt": gt,
                "pred": predictions[best_pi],
                "match_type": "centroid_proximity",
                "distance_px": round(best_dist, 2),
            })
            used_gt_idx.add(gi)
            used_pred_idx.add(best_pi)

    # ── Unmatched → FP / FN ─────────────────────────────────────────
    false_positives = [{"pred_idx": pi, "pred": predictions[pi]}
                       for pi in range(len(predictions)) if pi not in used_pred_idx]
    false_negatives = [{"gt_idx": gi, "gt": gt_annotations[gi]}
                       for gi in range(len(gt_annotations)) if gi not in used_gt_idx]

    return matched, false_positives, false_negatives


# =============================================================================
# Enrichment: add caries_coordinates to week5 JSON
# =============================================================================

def enrich_week5_json(case_num: int, week3_lookup: Dict) -> Optional[Dict]:
    """
    Read the week5 diagnosis JSON, inject ``caries_coordinates`` from week3,
    and return the enriched dict.  Also saves to week6 output.
    """
    preds = load_week5_predictions(case_num)
    if not preds:
        return None

    for tooth in preds:
        tid = tooth.get("tooth_id", "")
        w3 = week3_lookup.get(tid, {})
        tooth["caries_coordinates"] = w3.get("caries_coordinates", [])
        tooth["caries_pixels"] = w3.get("caries_pixels", 0)
        tooth["caries_percentage"] = round(w3.get("caries_percentage", 0.0), 4)

    enriched = {
        "case_number": case_num,
        "enriched_at": datetime.now().isoformat(),
        "teeth_data": preds,
    }

    out_dir = OUTPUT_DIR / f"case {case_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_{case_num}_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    return enriched


# =============================================================================
# Single-Case Evaluation
# =============================================================================

def evaluate_single_case(
    case_num: int,
    reclassify: bool = True,
) -> Dict:
    """
    Evaluate one case and return detailed results.

    Parameters
    ----------
    case_num : int
    reclassify : bool
        If True, re-run the Multi-Zone classifier on each prediction to
        produce the zone_label (MO / DO / MOD / etc.).

    Returns
    -------
    dict with keys: case_number, matched, fp, fn, enriched_json, mz_results
    """
    # 1) Load ground truth
    gt_folder = MATERIAL_DIR / f"case {case_num}"
    gt_annotations = parse_case_xmls(str(gt_folder))

    # 2) Load predictions + week3 data
    predictions = load_week5_predictions(case_num)
    week3_lookup = load_week3_caries(case_num)
    tooth_polygons = load_week2_tooth_polygons(case_num)

    # 3) Enrich week5 JSON
    enriched = enrich_week5_json(case_num, week3_lookup)

    # 4) Match GT ↔ Predictions
    matched, fp, fn = match_gt_to_predictions(
        gt_annotations, predictions, week3_lookup
    )

    # 5) Multi-zone reclassification
    mz_results = []
    if reclassify:
        for pred in predictions:
            tid = pred.get("tooth_id", "")
            w3 = week3_lookup.get(tid, {})
            coords = w3.get("caries_coordinates", [])
            poly = tooth_polygons.get(tid, [])
            if coords and poly:
                mz = classify_from_week_data(tid, poly, coords)
                mz["original_surface"] = pred.get("caries_surface", "")
                mz["original_position"] = pred.get("caries_position_detail", "")
                mz_results.append(mz)

    return {
        "case_number": case_num,
        "gt_count": len(gt_annotations),
        "pred_count": len(predictions),
        "matched": matched,
        "fp": fp,
        "fn": fn,
        "enriched_json": enriched,
        "mz_results": mz_results,
    }


# =============================================================================
# Batch Evaluation (all 500 cases)
# =============================================================================

def evaluate_all_cases(
    start: int = 1,
    end: int = 500,
    reclassify: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation across all cases and aggregate metrics.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []                 # per-match rows for CSV
    all_tp = 0
    all_fp = 0
    all_fn = 0
    gt_surfaces = []          # for confusion matrix
    pred_surfaces = []
    gt_surfaces_fine = []
    pred_surfaces_fine = []
    cases_processed = 0
    cases_skipped = 0

    if verbose:
        print("=" * 70)
        print("DENTAL CARIES EVALUATION ENGINE")
        print("=" * 70)
        print(f"Cases: {start}–{end}  |  Output: {OUTPUT_DIR}")
        print("=" * 70)

    for case_num in range(start, end + 1):
        gt_folder = MATERIAL_DIR / f"case {case_num}"
        if not gt_folder.exists():
            cases_skipped += 1
            continue

        try:
            result = evaluate_single_case(case_num, reclassify=reclassify)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Case {case_num}: {e}")
            cases_skipped += 1
            continue

        cases_processed += 1

        tp = len(result["matched"])
        fp = len(result["fp"])
        fn = len(result["fn"])
        all_tp += tp
        all_fp += fp
        all_fn += fn

        # Per-match rows
        for m in result["matched"]:
            gt = m["gt"]
            pred = m["pred"]

            # Find matching MZ result (before computing surfaces)
            mz_info = {}
            tid = pred.get("tooth_id", "")
            for mz in result.get("mz_results", []):
                if mz.get("tooth_id") == tid:
                    mz_info = mz
                    break

            # ── Surface comparison ───────────────────────────────────
            # Use MZ reclassified surface when available (point-cloud
            # voted), falling back to old week5 centroid-based label.
            gt_surf = normalize_surface(gt.get("surface_name", ""))

            mz_fine = mz_info.get("predicted_surface_fine", "")
            if mz_fine and mz_fine != "Unknown":
                pred_surf = normalize_surface(mz_fine)
            else:
                pred_surf = normalize_surface(pred.get("caries_surface", ""))

            gt_surfaces.append(gt_surf)
            pred_surfaces.append(pred_surf)

            gt_fine = normalize_surface_fine(gt.get("surface_name", ""))
            if mz_fine and mz_fine != "Unknown":
                pred_fine = normalize_surface_fine(mz_fine)
            else:
                pred_fine = normalize_surface_fine(pred.get("caries_surface", ""))
            gt_surfaces_fine.append(gt_fine)
            pred_surfaces_fine.append(pred_fine)

            rows.append({
                "case": case_num,
                "tooth_id": gt.get("tooth_fdi", ""),
                "gt_surface": gt.get("surface_name", ""),
                "gt_surface_norm": gt_surf,
                "gt_severity": gt.get("severity", ""),
                "gt_roi_points": len(gt.get("roi_coordinates", [])),
                "gt_centroid_x": gt.get("roi_centroid", (0, 0))[0],
                "gt_centroid_y": gt.get("roi_centroid", (0, 0))[1],
                "pred_surface_w5": pred.get("caries_surface", ""),
                "pred_surface": mz_fine if mz_fine else pred.get("caries_surface", ""),
                "pred_surface_norm": pred_surf,
                "pred_position_detail": mz_info.get("predicted_detail", pred.get("caries_position_detail", "")),
                "mz_zone_label": mz_info.get("zone_label", ""),
                "mz_primary_surface": mz_info.get("primary_surface", ""),
                "mz_predicted_surface": mz_fine,
                "mz_predicted_detail": mz_info.get("predicted_detail", ""),
                "mz_frac_M": mz_info.get("zone_fractions", {}).get("M", 0),
                "mz_frac_C": mz_info.get("zone_fractions", {}).get("C", 0),
                "mz_frac_D": mz_info.get("zone_fractions", {}).get("D", 0),
                "all_frac_M": mz_info.get("all_zone_fractions", {}).get("M", 0),
                "all_frac_C": mz_info.get("all_zone_fractions", {}).get("C", 0),
                "all_frac_D": mz_info.get("all_zone_fractions", {}).get("D", 0),
                "match_type": m["match_type"],
                "distance_px": m["distance_px"],
                "surface_match": gt_surf == pred_surf,
            })

        # FP rows
        for fpi in result["fp"]:
            pred = fpi["pred"]
            rows.append({
                "case": case_num,
                "tooth_id": pred.get("tooth_id", ""),
                "gt_surface": "",
                "gt_surface_norm": "",
                "gt_severity": "",
                "gt_roi_points": 0,
                "gt_centroid_x": 0,
                "gt_centroid_y": 0,
                "pred_surface": pred.get("caries_surface", ""),
                "pred_surface_norm": normalize_surface(pred.get("caries_surface", "")),
                "pred_position_detail": pred.get("caries_position_detail", ""),
                "mz_zone_label": "",
                "mz_primary_surface": "",
                "mz_frac_M": 0,
                "mz_frac_C": 0,
                "mz_frac_D": 0,
                "match_type": "FP",
                "distance_px": -1,
                "surface_match": False,
            })

        # FN rows
        for fni in result["fn"]:
            gt = fni["gt"]
            rows.append({
                "case": case_num,
                "tooth_id": gt.get("tooth_fdi", ""),
                "gt_surface": gt.get("surface_name", ""),
                "gt_surface_norm": normalize_surface(gt.get("surface_name", "")),
                "gt_severity": gt.get("severity", ""),
                "gt_roi_points": len(gt.get("roi_coordinates", [])),
                "gt_centroid_x": gt.get("roi_centroid", (0, 0))[0],
                "gt_centroid_y": gt.get("roi_centroid", (0, 0))[1],
                "pred_surface": "",
                "pred_surface_norm": "",
                "pred_position_detail": "",
                "mz_zone_label": "",
                "mz_primary_surface": "",
                "mz_frac_M": 0,
                "mz_frac_C": 0,
                "mz_frac_D": 0,
                "match_type": "FN",
                "distance_px": -1,
                "surface_match": False,
            })

        if verbose and cases_processed % 50 == 0:
            print(f"  ... processed {cases_processed} cases (TP={all_tp}, FP={all_fp}, FN={all_fn})")

    # ── Aggregate metrics ────────────────────────────────────────────
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Surface classification accuracy (among matched)
    df = pd.DataFrame(rows)
    matched_df = df[df["match_type"].isin(["tooth_id", "centroid_proximity"])]
    surface_accuracy = matched_df["surface_match"].mean() if len(matched_df) > 0 else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "cases_processed": cases_processed,
        "cases_skipped": cases_skipped,
        "total_gt_annotations": all_tp + all_fn,
        "total_predictions": all_tp + all_fp,
        "TP": all_tp,
        "FP": all_fp,
        "FN": all_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "surface_classification_accuracy": round(surface_accuracy, 4),
        "distance_threshold_px": DISTANCE_THRESHOLD_PX,
    }

    # ── Save outputs ─────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Confusion matrix (coarse: Occlusal / Proximal / Other) ──────
    if gt_surfaces and pred_surfaces:
        _plot_confusion_matrix(
            gt_surfaces, pred_surfaces,
            labels=["Occlusal", "Proximal", "Other"],
            title="Surface Classification – Confusion Matrix (Coarse)",
            save_path=OUTPUT_DIR / "confusion_matrix_coarse.png",
        )

    # ── Confusion matrix (fine: Occlusal / Mesial / Distal / …) ─────
    if gt_surfaces_fine and pred_surfaces_fine:
        fine_labels = sorted(set(gt_surfaces_fine) | set(pred_surfaces_fine))
        _plot_confusion_matrix(
            gt_surfaces_fine, pred_surfaces_fine,
            labels=fine_labels,
            title="Surface Classification – Confusion Matrix (Fine)",
            save_path=OUTPUT_DIR / "confusion_matrix_fine.png",
        )

    # ── Print summary ────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Cases processed    : {cases_processed}")
        print(f"  Cases skipped      : {cases_skipped}")
        print(f"  Ground Truth (GT)  : {all_tp + all_fn} annotations")
        print(f"  Predictions (Pred) : {all_tp + all_fp}")
        print()
        print(f"  True Positives  (TP) : {all_tp}")
        print(f"  False Positives (FP) : {all_fp}")
        print(f"  False Negatives (FN) : {all_fn}")
        print()
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print()
        print(f"  Surface classification accuracy : {surface_accuracy:.4f}")
        print()
        print(f"  Outputs → {OUTPUT_DIR}")
        print("=" * 70)

    return df


# =============================================================================
# Confusion Matrix Plotting
# =============================================================================

def _plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
):
    """Plot and optionally save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5),
                                     max(5, len(labels) * 1.2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Surface")
    ax.set_ylabel("Ground Truth Surface")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Saved] {save_path}")
    plt.close(fig)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Dental Caries Evaluation Engine – week6"
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--case", type=int, default=None,
                        help="Evaluate a single case")
    parser.add_argument("--no-reclassify", action="store_true",
                        help="Skip multi-zone reclassification")
    args = parser.parse_args()

    if args.case is not None:
        result = evaluate_single_case(args.case, reclassify=not args.no_reclassify)
        print(f"\nCase {args.case}:")
        print(f"  GT annotations : {result['gt_count']}")
        print(f"  Predictions    : {result['pred_count']}")
        print(f"  Matched (TP)   : {len(result['matched'])}")
        print(f"  FP             : {len(result['fp'])}")
        print(f"  FN             : {len(result['fn'])}")
        if result["mz_results"]:
            print(f"\n  Multi-Zone Results:")
            for mz in result["mz_results"]:
                print(f"    Tooth {mz['tooth_id']}: {mz['zone_label']} "
                      f"(M={mz['zone_fractions']['M']:.0%}, "
                      f"C={mz['zone_fractions']['C']:.0%}, "
                      f"D={mz['zone_fractions']['D']:.0%})")
    else:
        evaluate_all_cases(args.start, args.end, reclassify=not args.no_reclassify)


if __name__ == "__main__":
    main()
