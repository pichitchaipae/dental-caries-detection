# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Dental Caries Evaluation Engine — week8 (Multi-PCA Method Support)
===================================================================

Inherits all week7 fixes and adds **configurable PCA method selection**
from the week5 caries_surface_classifier.py evaluation pipeline.

Fixes inherited from week7:
  Task 1 – PCA Eigenvector Swap (3-rule orientation)
  Task 2 – Boundary Leakage (morphological erosion)
  Task 3 – Unassigned Caries (missing upstream teeth)
  Task 4 – Soft / Partial Surface Matching

New in week8:
  **Multi-PCA Method Evaluation**
    Supports running evaluations with 5 different PCA methods:
      Method 1 – Square Heuristic
      Method 2 – Maximum Projected Span
      Method 3 – Split-Centroid
      Method 4 – (Placeholder)
      Method 5 – Vertical Prior + 3-Rule (default, same as week7)

    Use ``--pca-method N`` to select.  Results are saved to
    ``evaluation_output/<method_name>/`` subdirectories.

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026-02
"""

import os
import sys
import gc
import json
import math
import cv2 # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from tqdm import tqdm  # type: ignore

import matplotlib # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns # type: ignore

# Local modules (week8 directory)
from xml_ground_truth_parser import parse_case_xmls
from multi_zone_classifier import (
    classify_multi_zone,
    classify_from_week_data,
    set_pca_method,
    get_pca_method,
    get_pca_method_name,
    perform_pca,
    PCA_METHOD_NAMES,
    VALID_PCA_METHODS,
)
from snodent_tooth_map import FDI_TOOTH_NAMES


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")

# Inputs
MATERIAL_DIR   = BASE_DIR / "raw_data" / "500 cases with annotation(xml)"
WEEK2_DIR      = BASE_DIR / "week2-Tooth Detection & Segmentation"  / "500-segmentation+recognition"
WEEK5_DIR      = BASE_DIR / "week5-Surface Classification v1"  / "surface_classification_output"

# week7 analysis output (Task 2 + 3 fixed caries mapping)
WEEK7_ANALYSIS_DIR = BASE_DIR / "week7-Surface Classification v3" / "dental_analysis_output"

# Fallback: use week3 if week7 analysis hasn't been run yet
WEEK3_DIR      = BASE_DIR / "week3-Caries-to-Tooth Mapping"  / "dental_analysis_output"

# Output
WEEK8_DIR      = BASE_DIR / "week8-Surface Classification v4"
OUTPUT_DIR     = WEEK8_DIR / "evaluation_output"

# Matching parameters
DISTANCE_THRESHOLD_PX = 60.0
IOU_THRESHOLD         = 0.10

# Task 4 – Soft matching threshold
SOFT_MATCH_FRACTION_THRESHOLD = 0.30   # 30 %


# =============================================================================
# Surface Label Normalization
# =============================================================================

# ── Task 4 STRICT RULE: Surface Incorrect classes ────────────────────
# WARNING!!!  The "Surface Incorrect" classes MUST strictly be limited
# to: ['Distal', 'Mesial', 'Occlusal'].  No other classes are allowed.
ALLOWED_SURFACE_CLASSES = ["Distal", "Mesial", "Occlusal"]


def enforce_surface_rule(surface: str) -> str:
    """Enforce the strict surface classification rule.

    If *surface* is not one of ``ALLOWED_SURFACE_CLASSES``, it is
    remapped to ``'Unclassified'`` and a warning is printed.

    This ensures no unexpected classes leak into evaluation metrics.
    """
    if surface in ALLOWED_SURFACE_CLASSES:
        return surface
    if surface and surface != "Unclassified":
        print(f"  [WARN][STRICT RULE] Unexpected surface class '{surface}' "
              f"dropped → 'Unclassified'. Allowed: {ALLOWED_SURFACE_CLASSES}")
    return "Unclassified"


# NOTE: normalize_surface (coarse) and normalize_surface_fine (fine) are
#       currently IDENTICAL — both map to the same 3 classes
#       (Mesial, Occlusal, Distal).
#
#       Intended design was:
#         - Coarse: Single dominant surface only (Mesial, Occlusal, Distal)
#         - Fine:   Compound / multi-surface labels (MO, DO, MOD, M, O, D)
#                   preserving detail from multi_zone_classifier's
#                   `predicted_detail` field.
#
#       Fine matching was never differentiated, so both confusion matrices
#       currently produce the same result.

def normalize_surface(raw: str) -> str:
    """Normalize a raw surface label to {Mesial, Distal, Occlusal, Unclassified}.

    After normalization, the result is passed through ``enforce_surface_rule``
    to guarantee only allowed classes survive.
    """
    r = raw.strip().lower()
    if "occlusal" in r:
        return enforce_surface_rule("Occlusal")
    if "mesial" in r:
        return enforce_surface_rule("Mesial")
    if "distal" in r:
        return enforce_surface_rule("Distal")
    return "Unclassified"


def normalize_surface_fine(raw: str) -> str:
    """Normalize a raw surface label to {Mesial, Distal, Occlusal, Unclassified}.

    NOTE: Currently identical to normalize_surface().
    TODO: Differentiate to support compound labels (MO, DO, MOD) if needed.
    """
    r = raw.strip().lower()
    if "occlusal" in r:
        return enforce_surface_rule("Occlusal")
    if "mesial" in r:
        return enforce_surface_rule("Mesial")
    if "distal" in r:
        return enforce_surface_rule("Distal")
    return "Unclassified"


def _to_proximal_group(surface: str) -> str:
    """Map {Mesial, Distal} → 'Proximal', Occlusal unchanged.

    Used for 'Strict Surface Accuracy' which treats both Mesial and
    Distal as the same class ('Proximal').
    """
    if surface in ("Mesial", "Distal"):
        return "Proximal"
    return surface


# =============================================================================
# Data Loaders
# =============================================================================

def load_week7_caries(case_num: int) -> Dict:
    """Load the week7-fixed caries mapping JSON. Falls back to week3."""
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

    unassigned = data.get("unassigned_caries", [])
    if unassigned:
        lookup["__unassigned__"] = unassigned

    return lookup


def load_week2_tooth_polygons(case_num: int) -> Dict:
    """Load week2 segmentation -> {tooth_id: polygon}."""
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

_ZONE_TO_SURFACE = {
    "M": "Mesial",
    "C": "Occlusal",
    "D": "Distal",
}

_ADJACENT_SURFACES = {
    "Occlusal": {"Mesial", "Distal"},
    "Mesial":   {"Occlusal"},
    "Distal":   {"Occlusal"},
}


def soft_surface_match(
    gt_surface_raw: str,
    pred_surface_fine: str,
    mz_info: Dict,
    threshold: float = SOFT_MATCH_FRACTION_THRESHOLD,
) -> str:
    """
    Perform soft / partial surface matching (Task 4).

    Returns one of: "True", "Partial", "True_with_warning", "False"
    """
    gt_norm  = normalize_surface_fine(gt_surface_raw)
    pred_norm = normalize_surface_fine(pred_surface_fine) if pred_surface_fine else ""

    if gt_norm == pred_norm:
        return "True"

    # ── Proximal equivalence: Mesial ↔ Distal (both are Proximal) ─────
    _PROXIMAL = {"Mesial", "Distal"}
    if gt_norm in _PROXIMAL and pred_norm in _PROXIMAL:
        return "True"

    gt_coarse   = normalize_surface(gt_surface_raw)
    pred_coarse = normalize_surface(pred_surface_fine) if pred_surface_fine else ""
    if gt_coarse == pred_coarse:
        return "True"

    all_fracs = mz_info.get("all_zone_fractions", {})
    if not all_fracs:
        return "False"

    surface_fracs: Dict[str, float] = {}
    for zone_letter, frac in all_fracs.items():
        surf_name = _ZONE_TO_SURFACE.get(zone_letter, "")
        if surf_name:
            surface_fracs[surf_name] = frac

    gt_frac = surface_fracs.get(gt_norm, 0.0)
    if gt_norm == "Occlusal":
        gt_frac = max(gt_frac, surface_fracs.get("Occlusal", 0.0))

    if gt_frac >= threshold:
        return "Partial"

    adjacent = _ADJACENT_SURFACES.get(gt_norm, set())
    for adj_surf in adjacent:
        adj_frac = surface_fracs.get(adj_surf, 0.0)
        if adj_frac >= threshold:
            return "True_with_warning"

    return "False"


# =============================================================================
# Ground Truth <-> Prediction Matching
# =============================================================================

def _point_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def match_gt_to_predictions(
    gt_annotations: List[Dict],
    predictions: List[Dict],
    week_lookup: Dict,
    distance_threshold: float = DISTANCE_THRESHOLD_PX,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match GT annotations to predictions (2-pass: tooth-ID then centroid)."""
    matched: List[Dict] = []
    used_gt_idx = set()
    used_pred_idx = set()

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

    # Pass 1: Tooth-ID match
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

    # Pass 2: Centroid proximity
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

    false_positives = [{"pred_idx": pi, "pred": predictions[pi]}
                       for pi in range(len(predictions)) if pi not in used_pred_idx]
    false_negatives = [{"gt_idx": gi, "gt": gt_annotations[gi]}
                       for gi in range(len(gt_annotations)) if gi not in used_gt_idx]

    return matched, false_positives, false_negatives


# =============================================================================
# Enrichment
# =============================================================================

def enrich_week5_json(case_num: int, week_lookup: Dict, out_dir: Path = None) -> Optional[Dict]:
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
        "pipeline_version": "week8",
        "pca_method": get_pca_method(),
        "pca_method_name": get_pca_method_name(),
        "teeth_data": preds,
    }

    unassigned = week_lookup.get("__unassigned__", [])
    if unassigned:
        enriched["unassigned_caries"] = unassigned

    target_dir = out_dir if out_dir else OUTPUT_DIR
    case_dir = target_dir / "cases" / f"case {case_num}"
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / f"case_{case_num}_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    return enriched


# =============================================================================
# Single-Case Evaluation
# =============================================================================

def evaluate_single_case(case_num: int, reclassify: bool = True, out_dir: Path = None) -> Dict:
    gt_folder = MATERIAL_DIR / f"case {case_num}"
    gt_annotations = parse_case_xmls(str(gt_folder))

    predictions_raw = load_week5_predictions(case_num)
    week_lookup = load_week7_caries(case_num)
    tooth_polygons = load_week2_tooth_polygons(case_num)

    # Task 5 FIX: Filter phantom false positives
    predictions = [
        p for p in predictions_raw
        if week_lookup.get(p.get("tooth_id", ""), {}).get("has_caries", False)
    ]

    enriched = enrich_week5_json(case_num, week_lookup, out_dir)

    matched, fp, fn = match_gt_to_predictions(
        gt_annotations, predictions, week_lookup
    )

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
    pca_method: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run evaluation.  If *pca_method* is provided, set it before running.
    Results go to ``evaluation_output/<method_name>/`` when a PCA method is specified.
    """
    if pca_method is not None:
        set_pca_method(pca_method)

    method_name = get_pca_method_name()

    # Output subdirectory per PCA method
    eval_output = OUTPUT_DIR / method_name
    eval_output.mkdir(parents=True, exist_ok=True)

    cases_to_run = case_list if case_list else list(range(start, end + 1))

    rows = []
    all_tp = all_fp = all_fn = 0
    gt_surfaces = []
    pred_surfaces = []
    gt_surfaces_fine = []
    pred_surfaces_fine = []
    cases_processed = cases_skipped = 0
    failed_cases: List[Dict] = []   # track per-case errors for the log

    if verbose:
        print("=" * 70)
        print(f"DENTAL CARIES EVALUATION ENGINE – week8")
        print(f"  PCA Method: {get_pca_method()} ({method_name})")
        print(f"  Fixes: PCA orientation | Erosion | Unassigned caries | Soft match")
        print("=" * 70)
        print(f"Cases: {cases_to_run[:5]}{'...' if len(cases_to_run) > 5 else ''}  |  Output: {eval_output}")
        print("=" * 70)

    pbar = tqdm(
        cases_to_run,
        desc=f"Evaluating Method {get_pca_method()}",
        unit="case",
        ncols=100,
        leave=True,
    )
    for case_num in pbar:
        gt_folder = MATERIAL_DIR / f"case {case_num}"
        if not gt_folder.exists():
            cases_skipped += 1
            continue

        try:
            result = evaluate_single_case(case_num, reclassify=reclassify, out_dir=eval_output)
        except Exception as e:
            if verbose:
                pbar.set_postfix_str(f"FAIL case {case_num}")
            failed_cases.append({"case": case_num, "error": str(e)})
            cases_skipped += 1
            continue
        finally:
            # ── Strict Memory Management ──────────────────────────────
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        cases_processed += 1
        pbar.set_postfix_str(f"TP={all_tp} FP={all_fp} FN={all_fn}")
        tp = len(result["matched"])
        fp_count = len(result["fp"])
        fn_count = len(result["fn"])
        all_tp += tp
        all_fp += fp_count
        all_fn += fn_count

        for m in result["matched"]:
            gt = m["gt"]
            pred = m["pred"]

            mz_info = {}
            tid = pred.get("tooth_id", "")
            for mz in result.get("mz_results", []):
                if mz.get("tooth_id") == tid:
                    mz_info = mz
                    break

            gt_surf = normalize_surface(gt.get("surface_name", ""))
            mz_fine = mz_info.get("predicted_surface_fine", "")
            if mz_fine and mz_fine != "Unclassified":
                pred_surf = normalize_surface(mz_fine)
            else:
                pred_surf = normalize_surface(pred.get("caries_surface", ""))

            gt_surfaces.append(gt_surf)
            pred_surfaces.append(pred_surf)

            gt_fine = normalize_surface_fine(gt.get("surface_name", ""))
            if mz_fine and mz_fine != "Unclassified":
                pred_fine = normalize_surface_fine(mz_fine)
            else:
                pred_fine = normalize_surface_fine(pred.get("caries_surface", ""))
            gt_surfaces_fine.append(gt_fine)
            pred_surfaces_fine.append(pred_fine)

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
                "pca_method": get_pca_method(),
                "pca_method_name": method_name,
                "match_type": m["match_type"],
                "distance_px": m["distance_px"],
                "surface_match_strict": strict_match,
                "surface_match": soft_match_result,
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
                "pca_method": get_pca_method(),
                "pca_method_name": method_name,
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
                "pca_method": get_pca_method(),
                "pca_method_name": method_name,
                "match_type": "FN", "distance_px": -1,
                "surface_match_strict": False,
                "surface_match": "False",
            })

    # ── Aggregate metrics ────────────────────────────────────────────
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    df = pd.DataFrame(rows)
    matched_df = df[df["match_type"].isin(["tooth_id", "centroid_proximity"])]

    # Strict: Proximal-grouped ({Mesial,Distal}→Proximal vs Occlusal)
    if len(matched_df) > 0:
        strict_proximal = matched_df.apply(
            lambda r: _to_proximal_group(r["gt_surface_norm"]) == _to_proximal_group(r["pred_surface_norm"]),
            axis=1,
        )
        strict_acc = strict_proximal.mean()
    else:
        strict_acc = 0

    if len(matched_df) > 0:
        soft_hits = matched_df["surface_match"].isin(["True", "Partial", "True_with_warning"]).sum()
        soft_acc = soft_hits / len(matched_df)
    else:
        soft_acc = 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "week8",
        "pca_method": get_pca_method(),
        "pca_method_name": method_name,
        "fixes_applied": [
            "Task1_PCA_eigenvector_fix",
            "Task2_boundary_erosion",
            "Task3_unassigned_caries",
            "Task4_soft_surface_match",
            "week8_multi_pca_method_support",
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

    # ── Save outputs (CSV first, JSON after 3-class metrics) ─────────
    csv_path = eval_output / "evaluation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── 3-Class Only: filter out Unclassified ────────────────────────
    VALID_CLASSES = ["Mesial", "Occlusal", "Distal"]

    # Filter paired lists to only valid 3-class entries
    filt_gt, filt_pred = [], []
    for g, p in zip(gt_surfaces, pred_surfaces):
        if g in VALID_CLASSES and p in VALID_CLASSES:
            filt_gt.append(g)
            filt_pred.append(p)

    filt_gt_fine, filt_pred_fine = [], []
    for g, p in zip(gt_surfaces_fine, pred_surfaces_fine):
        if g in VALID_CLASSES and p in VALID_CLASSES:
            filt_gt_fine.append(g)
            filt_pred_fine.append(p)

    # Per-class precision / recall / F1  (3-class only)
    per_class_metrics = {}
    for cls in VALID_CLASSES:
        cls_tp = sum(1 for g, p in zip(filt_gt, filt_pred) if g == cls and p == cls)
        cls_fp = sum(1 for g, p in zip(filt_gt, filt_pred) if g != cls and p == cls)
        cls_fn = sum(1 for g, p in zip(filt_gt, filt_pred) if g == cls and p != cls)
        cls_prec = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        cls_rec  = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        cls_f1   = 2 * cls_prec * cls_rec / (cls_prec + cls_rec) if (cls_prec + cls_rec) > 0 else 0.0
        cls_support = sum(1 for g in filt_gt if g == cls)
        per_class_metrics[cls] = {
            "precision": round(cls_prec, 4),
            "recall": round(cls_rec, 4),
            "f1": round(cls_f1, 4),
            "support": cls_support,
            "TP": cls_tp, "FP": cls_fp, "FN": cls_fn,
        }

    # Recompute strict/soft accuracy on 3-class filtered matched rows
    matched_3cls = matched_df[
        matched_df["gt_surface_norm"].isin(VALID_CLASSES) &
        matched_df["pred_surface_norm"].isin(VALID_CLASSES)
    ] if len(matched_df) > 0 else matched_df
    # Strict 3-class: Proximal-grouped
    if len(matched_3cls) > 0:
        strict_prox_3cls = matched_3cls.apply(
            lambda r: _to_proximal_group(r["gt_surface_norm"]) == _to_proximal_group(r["pred_surface_norm"]),
            axis=1,
        )
        strict_acc_3cls = strict_prox_3cls.mean()
    else:
        strict_acc_3cls = 0
    if len(matched_3cls) > 0:
        soft_hits_3cls = matched_3cls["surface_match"].isin(["True", "Partial", "True_with_warning"]).sum()
        soft_acc_3cls = soft_hits_3cls / len(matched_3cls)
    else:
        soft_acc_3cls = 0

    n_unclassified_gt   = sum(1 for g in gt_surfaces if g not in VALID_CLASSES)
    n_unclassified_pred = sum(1 for p in pred_surfaces if p not in VALID_CLASSES)

    summary["surface_classification_accuracy_strict"] = round(strict_acc_3cls, 4)
    summary["surface_classification_accuracy_soft"]   = round(float(soft_acc_3cls), 4)
    summary["per_class_metrics"] = per_class_metrics
    summary["valid_classes"] = VALID_CLASSES
    summary["n_unclassified_gt"]   = n_unclassified_gt
    summary["n_unclassified_pred"] = n_unclassified_pred
    summary["n_matched_3class"]    = len(matched_3cls)
    summary["failed_cases"]          = failed_cases

    # ── Strict Rule Assertion ─────────────────────────────────────────
    # Verify no unexpected classes leaked through.
    _unexpected_gt   = [g for g in filt_gt   if g not in ALLOWED_SURFACE_CLASSES]
    _unexpected_pred = [p for p in filt_pred if p not in ALLOWED_SURFACE_CLASSES]
    assert len(_unexpected_gt) == 0, (
        f"Strict Rule Violated: unexpected GT classes: {set(_unexpected_gt)}"
    )
    assert len(_unexpected_pred) == 0, (
        f"Strict Rule Violated: unexpected Pred classes: {set(_unexpected_pred)}"
    )

    # ── Save summary JSON (after 3-class metrics are computed) ───────
    summary_path = eval_output / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Per-Class Metrics Dashboard CSV ──────────────────────────────
    dashboard_rows = []
    for cls in VALID_CLASSES:
        m = per_class_metrics[cls]
        dashboard_rows.append({
            "pca_method": get_pca_method(),
            "pca_method_name": method_name,
            "class": cls,
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "support":   m["support"],
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
        })
    dashboard_df = pd.DataFrame(dashboard_rows)
    dashboard_path = eval_output / "per_class_metrics.csv"
    dashboard_df.to_csv(dashboard_path, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"  [Saved] Per-class metrics dashboard -> {dashboard_path}")

    # ── Confusion matrices (3-class only) ────────────────────────────
    if filt_gt and filt_pred:
        _plot_confusion_matrix(
            filt_gt, filt_pred,
            labels=VALID_CLASSES,
            title=f"Surface Classification – Confusion Matrix (Coarse, 3-Class) – week8 PCA-{get_pca_method()}",
            save_path=eval_output / "confusion_matrix_coarse.png",
        )
    if filt_gt_fine and filt_pred_fine:
        _plot_confusion_matrix(
            filt_gt_fine, filt_pred_fine,
            labels=VALID_CLASSES,
            title=f"Surface Classification – Confusion Matrix (Fine, 3-Class) – week8 PCA-{get_pca_method()}",
            save_path=eval_output / "confusion_matrix_fine.png",
        )

    # ── Print summary ────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 70)
        print(f"EVALUATION SUMMARY – week8  (PCA Method {get_pca_method()}: {method_name})")
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
        print(f"  Surface accuracy (strict, 3-class) : {strict_acc_3cls:.4f}")
        print(f"  Surface accuracy (soft,   3-class) : {float(soft_acc_3cls):.4f}")
        print(f"  Matched pairs (3-class valid)      : {len(matched_3cls)}")
        if n_unclassified_gt or n_unclassified_pred:
            print(f"  Unclassified filtered out          : GT={n_unclassified_gt}, Pred={n_unclassified_pred}")
        print()
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │  Per-Class Metrics (3-class: Distal / Mesial / Occlusal)   │")
        print(f"  ├───────────┬───────────┬──────────┬──────────┬──────────────┤")
        print(f"  │  Class    │ Precision │  Recall  │    F1    │   Support    │")
        print(f"  ├───────────┼───────────┼──────────┼──────────┼──────────────┤")
        for cls in VALID_CLASSES:
            m = per_class_metrics[cls]
            print(f"  │  {cls:<8s}│  {m['precision']:.4f}   │  {m['recall']:.4f}  │  {m['f1']:.4f}  │  {m['support']:>5d}       │")
        # Macro averages
        n_cls_with_support = sum(1 for c in VALID_CLASSES if per_class_metrics[c]['support'] > 0)
        macro_prec = sum(per_class_metrics[c]['precision'] for c in VALID_CLASSES) / max(n_cls_with_support, 1)
        macro_rec  = sum(per_class_metrics[c]['recall']    for c in VALID_CLASSES) / max(n_cls_with_support, 1)
        macro_f1   = sum(per_class_metrics[c]['f1']        for c in VALID_CLASSES) / max(n_cls_with_support, 1)
        total_support = sum(per_class_metrics[c]['support'] for c in VALID_CLASSES)
        print(f"  ├───────────┼───────────┼──────────┼──────────┼──────────────┤")
        print(f"  │  macro    │  {macro_prec:.4f}   │  {macro_rec:.4f}  │  {macro_f1:.4f}  │  {total_support:>5d}       │")
        print(f"  └───────────┴───────────┴──────────┴──────────┴──────────────┘")
        print()
        print(f"  Outputs -> {eval_output}")
        print("=" * 70)

    return df


# =============================================================================
# Compare All PCA Methods
# =============================================================================

def compare_all_methods(
    case_list: Optional[List[int]] = None,
    start: int = 1,
    end: int = 500,
    reclassify: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation for all implemented PCA methods (1, 2, 3, 5) and
    produce a comparison summary table.
    """
    implemented_methods = [0, 1, 2, 3, 5]
    comparison_rows = []

    for method in implemented_methods:
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# Running PCA Method {method}: {PCA_METHOD_NAMES[method]}")
            print(f"{'#' * 70}\n")

        df = evaluate_all_cases(
            case_list=case_list,
            start=start,
            end=end,
            reclassify=reclassify,
            verbose=verbose,
            pca_method=method,
        )

        # Read the saved summary
        method_name = PCA_METHOD_NAMES[method]
        summary_path = OUTPUT_DIR / method_name / "evaluation_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
            comparison_rows.append(summary)

    # Save comparison table
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_path = OUTPUT_DIR / "pca_method_comparison.csv"
        comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")

        if verbose:
            print("\n" + "=" * 70)
            print("PCA METHOD COMPARISON SUMMARY")
            print("=" * 70)
            for row in comparison_rows:
                print(f"  Method {row['pca_method']} ({row['pca_method_name']}):")
                print(f"    Precision={row['precision']:.4f}  Recall={row['recall']:.4f}  F1={row['f1_score']:.4f}")
                print(f"    Surface Strict={row['surface_classification_accuracy_strict']:.4f}  "
                      f"Soft={row['surface_classification_accuracy_soft']:.4f}")
            print(f"\n  Comparison saved to: {comp_path}")
            print("=" * 70)

        return comp_df
    return pd.DataFrame()


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
        description="Dental Caries Evaluation Engine – week8 (Multi-PCA + all fixes)"
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--case", type=int, default=None,
                        help="Evaluate a single case")
    parser.add_argument("--sample", type=str, default=None,
                        help="Comma-separated case numbers, e.g. --sample 311,33")
    parser.add_argument("--pca-method", type=int, choices=[0, 1, 2, 3, 5], default=5,
                        help="PCA method to use (0/1/2/3/5, default: 5). 0=baseline")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all PCA methods and produce comparison table")
    parser.add_argument("--no-reclassify", action="store_true",
                        help="Skip multi-zone reclassification")
    args = parser.parse_args()

    set_pca_method(args.pca_method)

    if args.compare_all:
        if args.sample:
            case_list = [int(c.strip()) for c in args.sample.split(',')]
            compare_all_methods(case_list=case_list, reclassify=not args.no_reclassify)
        else:
            compare_all_methods(start=args.start, end=args.end,
                                reclassify=not args.no_reclassify)
    elif args.case is not None:
        result = evaluate_single_case(args.case, reclassify=not args.no_reclassify)
        print(f"\nCase {args.case} (PCA Method {get_pca_method()}: {get_pca_method_name()}):")
        print(f"  GT annotations : {result['gt_count']}")
        print(f"  Predictions    : {result['pred_count']}")
        print(f"  Matched (TP)   : {len(result['matched'])}")
        print(f"  FP             : {len(result['fp'])}")
        print(f"  FN             : {len(result['fn'])}")
        if result["mz_results"]:
            print(f"\n  Multi-Zone Results (week8 – PCA Method {get_pca_method()}):")
            for mz in result["mz_results"]:
                print(f"    Tooth {mz['tooth_id']}: {mz['zone_label']} "
                      f"(M={mz['zone_fractions']['M']:.0%}, "
                      f"C={mz['zone_fractions']['C']:.0%}, "
                      f"D={mz['zone_fractions']['D']:.0%}) "
                      f"-> {mz['predicted_surface_fine']} "
                      f"[{mz['detail_explanation']}]")
    elif args.sample:
        case_list = [int(c.strip()) for c in args.sample.split(',')]
        evaluate_all_cases(case_list=case_list, reclassify=not args.no_reclassify,
                           pca_method=args.pca_method)
    else:
        evaluate_all_cases(start=args.start, end=args.end,
                           reclassify=not args.no_reclassify,
                           pca_method=args.pca_method)


if __name__ == "__main__":
    main()
