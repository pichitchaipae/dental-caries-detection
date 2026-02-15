"""
Dental Caries Evaluation Engine — week7 (Soft Surface Matching)
================================================================

Fixes applied vs. week6:

  **Task 4 – Label Noise & Soft Surface Matching**
    The old ``match_gt_to_predictions()`` used a strict boolean:
        ``surface_match = (GT_surface == Pred_surface)``
    When the GT XML says "Occlusal" but the Multi-Zone prediction
    clearly shows "Mesial" at 51 % (an MO label), the strict check
    gives ``surface_match=False`` even though "Occlusal" is present
    in the MO prediction at ~49 %.

    New **Soft / Partial Surface Matching**:
      - Parse the Multi-Zone zone fractions (M %, C %, D %).
      - Map each fraction to a clinical surface name.
      - If the predicted *dominant* surface matches GT → ``True``.
      - If any zone fraction for the GT surface (or an adjacent
        surface) is ≥ 30 % (0.30) → ``"Partial"`` match.
      - Otherwise → ``False``.

    This reduces false negatives caused by label noise in the GT.

Also imports the week7 multi_zone_classifier (with the PCA fix) and
the week7 dental_caries_analysis (with erosion + unassigned caries).

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026-02
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

# Local modules (week7 directory)
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
WEEK5_DIR      = BASE_DIR / "week5"  / "surface_classification_output"

# week7 analysis output (Task 2 + 3 fixed caries mapping)
WEEK7_ANALYSIS_DIR = BASE_DIR / "week7" / "dental_analysis_output"

# Fallback: use week3 if week7 analysis hasn't been run yet
WEEK3_DIR      = BASE_DIR / "week3"  / "dental_analysis_output"

# Output
WEEK7_DIR      = BASE_DIR / "week7"
OUTPUT_DIR     = WEEK7_DIR / "evaluation_output"

# Matching parameters
DISTANCE_THRESHOLD_PX = 60.0
IOU_THRESHOLD         = 0.10

# Task 4 – Soft matching threshold
SOFT_MATCH_FRACTION_THRESHOLD = 0.30   # 30 %


# =============================================================================
# Surface Label Normalization (same as week6)
# =============================================================================

def normalize_surface(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("occlusal",):
        return "Occlusal"
    if r in ("mesial", "distal", "proximal", "proximal-mesial", "proximal-distal"):
        return "Proximal"
    if r in ("buccal", "lingual", "palatal", "cervical", "lingual/other", "other"):
        return "Other"
    if "occlusal" in r:
        return "Occlusal"
    if "mesial" in r or "distal" in r or "proximal" in r:
        return "Proximal"
    return "Other"


def normalize_surface_fine(raw: str) -> str:
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

def load_week7_caries(case_num: int) -> Dict:
    """
    Load the week7-fixed caries mapping JSON.
    Falls back to week3 if week7 output doesn't exist yet.
    """
    p7 = WEEK7_ANALYSIS_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    p3 = WEEK3_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"

    path = p7 if p7.exists() else p3
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup = {}
    for t in data.get("teeth_caries_data", []):
        tid = t.get("tooth_id", "")
        if tid and tid != "UNASSIGNED":
            lookup[tid] = t

    # Also store unassigned blobs under a special key
    unassigned = data.get("unassigned_caries", [])
    if unassigned:
        lookup["__unassigned__"] = unassigned

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
    """Load week5 diagnosis JSON."""
    p = WEEK5_DIR / f"case {case_num}" / f"case_{case_num}_diagnosis.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("teeth_data", [])


# =============================================================================
# Task 4 FIX – Soft / Partial Surface Matching
# =============================================================================

# Map zone letters to clinical surface names for comparison
_ZONE_TO_SURFACE = {
    "M": "Mesial",
    "C": "Occlusal",
    "D": "Distal",
}

# Define adjacency: which surfaces are clinically adjacent
_ADJACENT_SURFACES = {
    "Occlusal": {"Mesial", "Distal"},       # MO / DO patterns
    "Mesial":   {"Occlusal"},
    "Distal":   {"Occlusal"},
    "Proximal": {"Mesial", "Distal", "Occlusal"},
}


def soft_surface_match(
    gt_surface_raw: str,
    pred_surface_fine: str,
    mz_info: Dict,
    threshold: float = SOFT_MATCH_FRACTION_THRESHOLD,
) -> str:
    """
    Perform soft / partial surface matching (Task 4).

    Parameters
    ----------
    gt_surface_raw : str
        The raw GT surface label from XML (e.g. "Occlusal").
    pred_surface_fine : str
        The fine-grained predicted surface from Multi-Zone
        (e.g. "Mesial", "Occlusal", "Distal").
    mz_info : dict
        The full Multi-Zone result dict.  Must contain
        ``all_zone_fractions`` with keys M, C, D.
    threshold : float
        Fraction (0–1) above which a zone is considered
        "present" for partial matching (default 0.30).

    Returns
    -------
    str – one of:
        "True"                  exact match
        "Partial"               GT surface present in prediction ≥ threshold
        "True_with_warning"     adjacent surface is dominant + GT present ≥ threshold
        "False"                 no match
    """
    gt_norm  = normalize_surface_fine(gt_surface_raw)
    pred_norm = normalize_surface_fine(pred_surface_fine) if pred_surface_fine else ""

    # ── Exact match (fine-grained) ───────────────────────────────────
    if gt_norm == pred_norm:
        return "True"

    # ── Coarse match (Occlusal / Proximal / Other) ───────────────────
    gt_coarse   = normalize_surface(gt_surface_raw)
    pred_coarse = normalize_surface(pred_surface_fine) if pred_surface_fine else ""
    if gt_coarse == pred_coarse:
        return "True"

    # ── If no MZ fractions available, fall back to strict ────────────
    all_fracs = mz_info.get("all_zone_fractions", {})
    if not all_fracs:
        return "False"

    # ── Build surface-fraction lookup from MZ zones ──────────────────
    # e.g. {"Mesial": 0.51, "Occlusal": 0.10, "Distal": 0.39}
    surface_fracs: Dict[str, float] = {}
    for zone_letter, frac in all_fracs.items():
        surf_name = _ZONE_TO_SURFACE.get(zone_letter, "")
        if surf_name:
            surface_fracs[surf_name] = frac

    # ── Check if GT surface has ≥ threshold fraction in prediction ───
    gt_frac = surface_fracs.get(gt_norm, 0.0)

    # Also check "Proximal" → maps to either Mesial or Distal
    if gt_norm == "Proximal":
        gt_frac = max(surface_fracs.get("Mesial", 0.0),
                      surface_fracs.get("Distal", 0.0))
    # Occlusal → C zone
    if gt_norm == "Occlusal":
        gt_frac = max(gt_frac, surface_fracs.get("Occlusal", 0.0))

    if gt_frac >= threshold:
        # The GT surface IS present in the multi-zone prediction
        # at ≥ threshold → Partial match
        return "Partial"

    # ── Check adjacent surfaces ──────────────────────────────────────
    adjacent = _ADJACENT_SURFACES.get(gt_norm, set())
    for adj_surf in adjacent:
        adj_frac = surface_fracs.get(adj_surf, 0.0)
        if adj_frac >= threshold:
            return "True_with_warning"

    return "False"


# =============================================================================
# Ground Truth ↔ Prediction Matching
# =============================================================================

def _point_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def match_gt_to_predictions(
    gt_annotations: List[Dict],
    predictions: List[Dict],
    week_lookup: Dict,
    distance_threshold: float = DISTANCE_THRESHOLD_PX,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match GT annotations to predictions (same 2-pass logic as week6).
    """
    matched: List[Dict] = []
    used_gt_idx = set()
    used_pred_idx = set()

    # Build prediction centroid lookup
    pred_centroids = {}
    for i, pred in enumerate(predictions):
        tid = pred.get("tooth_id", "")
        w = week_lookup.get(tid, {})
        coords = w.get("caries_coordinates", [])
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
        best_pi, best_dist = None, float("inf")
        for pi, pred in enumerate(predictions):
            if pi in used_pred_idx:
                continue
            if pred.get("tooth_id", "") != gt_fdi:
                continue
            pc = pred_centroids.get(pi)
            d = _point_distance(gt_cent, pc) if pc is not None else 0
            if d < best_dist:
                best_dist = d
                best_pi = pi
        if best_pi is not None:
            matched.append({
                "gt_idx": gi, "pred_idx": best_pi,
                "gt": gt, "pred": predictions[best_pi],
                "match_type": "tooth_id",
                "distance_px": round(best_dist, 2),
            })
            used_gt_idx.add(gi)
            used_pred_idx.add(best_pi)

    # ── Pass 2: Centroid proximity ───────────────────────────────────
    for gi, gt in enumerate(gt_annotations):
        if gi in used_gt_idx:
            continue
        gt_cent = gt.get("roi_centroid", (0, 0))
        best_pi, best_dist = None, float("inf")
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
                "gt_idx": gi, "pred_idx": best_pi,
                "gt": gt, "pred": predictions[best_pi],
                "match_type": "centroid_proximity",
                "distance_px": round(best_dist, 2),
            })
            used_gt_idx.add(gi)
            used_pred_idx.add(best_pi)

    # ── Unmatched ────────────────────────────────────────────────────
    false_positives = [{"pred_idx": pi, "pred": predictions[pi]}
                       for pi in range(len(predictions)) if pi not in used_pred_idx]
    false_negatives = [{"gt_idx": gi, "gt": gt_annotations[gi]}
                       for gi in range(len(gt_annotations)) if gi not in used_gt_idx]

    return matched, false_positives, false_negatives


# =============================================================================
# Enrichment
# =============================================================================

def enrich_week5_json(case_num: int, week_lookup: Dict) -> Optional[Dict]:
    preds = load_week5_predictions(case_num)
    if not preds:
        return None

    for tooth in preds:
        tid = tooth.get("tooth_id", "")
        w = week_lookup.get(tid, {})
        tooth["caries_coordinates"] = w.get("caries_coordinates", [])
        tooth["caries_pixels"] = w.get("caries_pixels", 0)
        tooth["caries_percentage"] = round(w.get("caries_percentage", 0.0), 4)

    enriched = {
        "case_number": case_num,
        "enriched_at": datetime.now().isoformat(),
        "pipeline_version": "week7",
        "teeth_data": preds,
    }

    # Include unassigned-caries info if present (Task 3)
    unassigned = week_lookup.get("__unassigned__", [])
    if unassigned:
        enriched["unassigned_caries"] = unassigned

    out_dir = OUTPUT_DIR / f"case {case_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_{case_num}_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    return enriched


# =============================================================================
# Single-Case Evaluation
# =============================================================================

def evaluate_single_case(case_num: int, reclassify: bool = True) -> Dict:
    # 1) Load ground truth
    gt_folder = MATERIAL_DIR / f"case {case_num}"
    gt_annotations = parse_case_xmls(str(gt_folder))

    # 2) Load predictions + week7 caries data (falls back to week3)
    predictions = load_week5_predictions(case_num)
    week_lookup = load_week7_caries(case_num)
    tooth_polygons = load_week2_tooth_polygons(case_num)

    # 3) Enrich
    enriched = enrich_week5_json(case_num, week_lookup)

    # 4) Match
    matched, fp, fn = match_gt_to_predictions(
        gt_annotations, predictions, week_lookup
    )

    # 5) Multi-zone reclassification (using week7 fixed PCA)
    mz_results = []
    if reclassify:
        for pred in predictions:
            tid = pred.get("tooth_id", "")
            w = week_lookup.get(tid, {})
            coords = w.get("caries_coordinates", [])
            poly = tooth_polygons.get(tid, [])
            if coords and poly:
                mz = classify_from_week_data(tid, poly, coords,
                                             save_debug=True, case_num=case_num)
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
# Batch Evaluation
# =============================================================================

def evaluate_all_cases(
    case_list: Optional[List[int]] = None,
    start: int = 1,
    end: int = 500,
    reclassify: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation.  If *case_list* is provided, iterate only those
    case numbers; otherwise iterate start..end.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_to_run = case_list if case_list else list(range(start, end + 1))

    rows = []
    all_tp = all_fp = all_fn = 0
    gt_surfaces = []
    pred_surfaces = []
    gt_surfaces_fine = []
    pred_surfaces_fine = []
    cases_processed = cases_skipped = 0

    if verbose:
        print("=" * 70)
        print("DENTAL CARIES EVALUATION ENGINE – week7")
        print("  Fixes: PCA orientation | Erosion | Unassigned caries | Soft match")
        print("=" * 70)
        print(f"Cases: {cases_to_run[:5]}{'...' if len(cases_to_run) > 5 else ''}  |  Output: {OUTPUT_DIR}")
        print("=" * 70)

    for case_num in cases_to_run:
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
        fp_count = len(result["fp"])
        fn_count = len(result["fn"])
        all_tp += tp
        all_fp += fp_count
        all_fn += fn_count

        # ── Per-match rows ───────────────────────────────────────────
        for m in result["matched"]:
            gt = m["gt"]
            pred = m["pred"]

            # Find matching MZ result
            mz_info = {}
            tid = pred.get("tooth_id", "")
            for mz in result.get("mz_results", []):
                if mz.get("tooth_id") == tid:
                    mz_info = mz
                    break

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

            # ── Task 4: Soft surface matching ────────────────────────
            strict_match = (gt_surf == pred_surf)
            soft_match_result = soft_surface_match(
                gt_surface_raw=gt.get("surface_name", ""),
                pred_surface_fine=mz_fine if mz_fine else pred.get("caries_surface", ""),
                mz_info=mz_info,
            )

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
                "pred_position_detail": mz_info.get("predicted_detail",
                                                     pred.get("caries_position_detail", "")),
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
                "rotation_angle_deg": mz_info.get("rotation_angle_deg", 0),
                "pca_clamped": mz_info.get("pca_clamped", False),
                "match_type": m["match_type"],
                "distance_px": m["distance_px"],
                # ── Task 4: both strict and soft match columns ───────
                "surface_match_strict": strict_match,
                "surface_match": soft_match_result,  # "True" / "Partial" / "True_with_warning" / "False"
            })

        # FP rows
        for fpi in result["fp"]:
            pred = fpi["pred"]
            rows.append({
                "case": case_num,
                "tooth_id": pred.get("tooth_id", ""),
                "gt_surface": "", "gt_surface_norm": "",
                "gt_severity": "", "gt_roi_points": 0,
                "gt_centroid_x": 0, "gt_centroid_y": 0,
                "pred_surface": pred.get("caries_surface", ""),
                "pred_surface_norm": normalize_surface(pred.get("caries_surface", "")),
                "pred_position_detail": pred.get("caries_position_detail", ""),
                "mz_zone_label": "", "mz_primary_surface": "",
                "mz_predicted_surface": "", "mz_predicted_detail": "",
                "mz_frac_M": 0, "mz_frac_C": 0, "mz_frac_D": 0,
                "all_frac_M": 0, "all_frac_C": 0, "all_frac_D": 0,
                "rotation_angle_deg": 0,
                "pca_clamped": False,
                "match_type": "FP", "distance_px": -1,
                "surface_match_strict": False,
                "surface_match": "False",
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
                "pred_surface": "", "pred_surface_norm": "",
                "pred_position_detail": "",
                "mz_zone_label": "", "mz_primary_surface": "",
                "mz_predicted_surface": "", "mz_predicted_detail": "",
                "mz_frac_M": 0, "mz_frac_C": 0, "mz_frac_D": 0,
                "all_frac_M": 0, "all_frac_C": 0, "all_frac_D": 0,
                "rotation_angle_deg": 0,
                "pca_clamped": False,
                "match_type": "FN", "distance_px": -1,
                "surface_match_strict": False,
                "surface_match": "False",
            })

        if verbose and cases_processed % 50 == 0:
            print(f"  ... processed {cases_processed} cases (TP={all_tp}, FP={all_fp}, FN={all_fn})")

    # ── Aggregate metrics ────────────────────────────────────────────
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    df = pd.DataFrame(rows)
    matched_df = df[df["match_type"].isin(["tooth_id", "centroid_proximity"])]

    # Strict surface accuracy (backward-compatible)
    strict_acc = matched_df["surface_match_strict"].mean() if len(matched_df) > 0 else 0

    # Soft surface accuracy (Task 4): count True + Partial + True_with_warning
    if len(matched_df) > 0:
        soft_hits = matched_df["surface_match"].isin(["True", "Partial", "True_with_warning"]).sum()
        soft_acc = soft_hits / len(matched_df)
    else:
        soft_acc = 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "week7",
        "fixes_applied": [
            "Task1_PCA_eigenvector_fix",
            "Task2_boundary_erosion",
            "Task3_unassigned_caries",
            "Task4_soft_surface_match",
        ],
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
        "surface_classification_accuracy_strict": round(strict_acc, 4),
        "surface_classification_accuracy_soft": round(float(soft_acc), 4),
        "soft_match_threshold": SOFT_MATCH_FRACTION_THRESHOLD,
        "distance_threshold_px": DISTANCE_THRESHOLD_PX,
    }

    # ── Save outputs ─────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Confusion matrices ───────────────────────────────────────────
    if gt_surfaces and pred_surfaces:
        _plot_confusion_matrix(
            gt_surfaces, pred_surfaces,
            labels=["Occlusal", "Proximal", "Other"],
            title="Surface Classification – Confusion Matrix (Coarse) – week7",
            save_path=OUTPUT_DIR / "confusion_matrix_coarse.png",
        )
    if gt_surfaces_fine and pred_surfaces_fine:
        fine_labels = sorted(set(gt_surfaces_fine) | set(pred_surfaces_fine))
        _plot_confusion_matrix(
            gt_surfaces_fine, pred_surfaces_fine,
            labels=fine_labels,
            title="Surface Classification – Confusion Matrix (Fine) – week7",
            save_path=OUTPUT_DIR / "confusion_matrix_fine.png",
        )

    # ── Print summary ────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY – week7")
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
        print(f"  Surface accuracy (strict) : {strict_acc:.4f}")
        print(f"  Surface accuracy (soft)   : {float(soft_acc):.4f}")
        print()
        print(f"  Outputs → {OUTPUT_DIR}")
        print("=" * 70)

    return df


# =============================================================================
# Confusion Matrix Plotting
# =============================================================================

def _plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix",
                           save_path=None):
    try:
        from sklearn.metrics import confusion_matrix as sk_cm
    except ImportError:
        print("  [WARN] sklearn not installed – skipping confusion matrix plot")
        return

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
        description="Dental Caries Evaluation Engine – week7 (all 4 fixes)"
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--case", type=int, default=None,
                        help="Evaluate a single case")
    parser.add_argument("--sample", type=str, default=None,
                        help="Comma-separated case numbers, e.g. --sample 311,33")
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
            print(f"\n  Multi-Zone Results (week7 – fixed PCA):")
            for mz in result["mz_results"]:
                print(f"    Tooth {mz['tooth_id']}: {mz['zone_label']} "
                      f"(M={mz['zone_fractions']['M']:.0%}, "
                      f"C={mz['zone_fractions']['C']:.0%}, "
                      f"D={mz['zone_fractions']['D']:.0%}) "
                      f"→ {mz['predicted_surface_fine']} "
                      f"[{mz['detail_explanation']}]")
    elif args.sample:
        case_list = [int(c.strip()) for c in args.sample.split(',')]
        evaluate_all_cases(case_list=case_list, reclassify=not args.no_reclassify)
    else:
        evaluate_all_cases(start=args.start, end=args.end,
                           reclassify=not args.no_reclassify)


if __name__ == "__main__":
    main()
