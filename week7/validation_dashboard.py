"""
Dental Caries Validation Dashboard — week7
============================================

Human-in-the-Loop visualization tool for verifying model predictions
against AIM XML Ground Truth annotations.

Week 7 updates:
  - Reads caries data from ``week7/dental_analysis_output`` (Task 2+3 output)
  - Uses ``_pca_rotation_angle_fixed()`` for PCA orientation (Task 1)
  - Skips teeth with ``has_caries=False`` after erosion (Task 2)
  - Shows Multi-zone fractions + soft-match status in panel text (Task 4)
  - Saves dashboards to ``week7/dental_analysis_output/``
  - Supports ``--sample 311,33`` CLI flag

Dashboard Layout:
  ┌──────────────────────────────────────────────┐
  │              GLOBAL VIEW (Top Half)           │
  │  X-ray + GT polygons (green) + Pred scatter  │
  │  (red) + Tooth ID / TP/FP/FN labels          │
  └──────────────────────────────────────────────┘
  ┌───────┬───────┬───────┬───────┬──────────────┐
  │ Tooth │ Tooth │ Tooth │ Tooth │   ...        │
  │  PCA  │  PCA  │  PCA  │  PCA  │              │
  │  M-C-D│  M-C-D│  M-C-D│  M-C-D│              │
  └───────┴───────┴───────┴───────┘──────────────┘

Usage:
  python validation_dashboard.py --case 311
  python validation_dashboard.py --sample 311,33
  python validation_dashboard.py --start 1 --end 10

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026-02
"""

import os
import sys
import json
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

# ── Local imports (week7 directory) ──────────────────────────────────
from xml_ground_truth_parser import parse_case_xmls
from multi_zone_classifier import (
    classify_multi_zone,
    _pca_rotation_angle_fixed,  # Task 1: fixed PCA orientation
    _rotation_matrix,
    _rotate,
    _bbox,
    _is_upper_jaw,
    _get_quadrant,
)
from snodent_tooth_map import FDI_TOOTH_NAMES
from evaluation_engine import (
    normalize_surface,
    normalize_surface_fine,
    match_gt_to_predictions,
    soft_surface_match,           # Task 4: soft matching
    DISTANCE_THRESHOLD_PX,
)


# =====================================================================
# Configuration  (week7 paths)
# =====================================================================

BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")

MATERIAL_DIR = BASE_DIR / "material" / "500 cases with annotation"
WEEK2_DIR    = BASE_DIR / "week2"  / "500-segmentation+recognition"
WEEK5_DIR    = BASE_DIR / "week5"  / "surface_classification_output"

# ── week7 paths (updated from week3 / week6) ────────────────────────
WEEK7_ANALYSIS_DIR = BASE_DIR / "week7" / "dental_analysis_output"
WEEK3_DIR          = BASE_DIR / "week3" / "dental_analysis_output"   # fallback
OUTPUT_DIR         = BASE_DIR / "week7" / "dental_analysis_output"   # output

# Style constants
GT_COLOR       = "#2ecc71"   # green
PRED_COLOR     = "#e74c3c"   # red
PCA_AXIS_COLOR = "#3498db"   # blue
ZONE_M_COLOR   = "#00FFFF"   # cyan   → Mesial
ZONE_C_COLOR   = "#FFD700"   # yellow → Central
ZONE_D_COLOR   = "#FF00FF"   # magenta → Distal
TP_COLOR       = "#2ecc71"
FP_COLOR       = "#e74c3c"
FN_COLOR       = "#f39c12"
DIVIDER_COLOR  = "#FF6347"   # tomato

ROI_PAD_PX = 60   # padding around tooth ROI crop


# =====================================================================
# Data Loaders
# =====================================================================

def load_xray(case_num: int) -> Optional[np.ndarray]:
    """Load the dental X-ray image for a case."""
    p = MATERIAL_DIR / f"case {case_num}" / f"case_{case_num}.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img


def load_week2_teeth(case_num: int) -> Dict:
    """Load week2 segmentation → {tooth_id: {bbox, polygon, pixel_coordinates}}."""
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
        bbox = tooth.get("bbox", [])
        crop = tooth.get("crop_coords", [])
        lookup[tid] = {
            "polygon": poly or [],
            "bbox": bbox,
            "crop_coords": crop,
            "pixel_coordinates": tooth.get("pixel_coordinates", []),
        }
    return lookup


def load_week7_caries(case_num: int) -> Dict:
    """
    Load caries mapping from week7 analysis output (Task 2+3 fixed).
    Falls back to week3 if week7 output doesn't exist.
    """
    p7 = WEEK7_ANALYSIS_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    p3 = WEEK3_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    path = p7 if p7.exists() else p3

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        t["tooth_id"]: t
        for t in data.get("teeth_caries_data", [])
        if t.get("tooth_id") and t.get("tooth_id") != "UNASSIGNED"
    }


def load_week5_predictions(case_num: int) -> List[Dict]:
    """Load week5 diagnosis JSON → list of tooth predictions."""
    p = WEEK5_DIR / f"case {case_num}" / f"case_{case_num}_diagnosis.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("teeth_data", [])


# =====================================================================
# PCA Geometry Helpers  (Task 1: uses _pca_rotation_angle_fixed)
# =====================================================================

def compute_pca_fixed(
    pts: np.ndarray,
    tooth_id: str,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute PCA with the week7 3-rule orientation fix.

    Returns: (center, rotation_angle_rad, corrected_eigenvectors_2x2)

    Unlike the old compute_pca(), this calls _pca_rotation_angle_fixed()
    which selects the eigenvector with larger |Y| as the vertical axis
    (Rule 1), flips for occlusal/apical direction (Rule 2), and enforces
    mesial/distal direction (Rule 3).
    """
    pts = pts.astype(np.float64)
    center, rot, _was_clamped = _pca_rotation_angle_fixed(pts, tooth_id)

    # Reconstruct eigenvectors from the corrected rotation angle
    # The major axis (vertical after rotation) in original space:
    angle_from_x = math.pi / 2 - rot
    ev_major = np.array([math.cos(angle_from_x), math.sin(angle_from_x)])
    ev_minor = np.array([-ev_major[1], ev_major[0]])
    eigvecs = np.array([ev_major, ev_minor])

    return center, rot, eigvecs


def rotation_matrix_2x3(angle: float, cx: float, cy: float) -> np.ndarray:
    """2x3 affine rotation matrix around (cx, cy)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, cx - cx * c + cy * s],
        [s,  c, cy - cx * s - cy * c],
    ], dtype=np.float64)


def rotate_pts(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to Nx2 points."""
    if len(pts) == 0:
        return pts
    return cv2.transform(
        pts.reshape(-1, 1, 2).astype(np.float64), M
    ).reshape(-1, 2)


# =====================================================================
# Matching & F1 Status Computation
# =====================================================================

def compute_match_status(
    gt_annotations: List[Dict],
    predictions: List[Dict],
    week_lookup: Dict,
) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    """
    Match GT ↔ Pred and return:
      tooth_status : {tooth_id: "TP"/"FP"/"FN"}
      matched, fps, fns
    """
    matched, fps, fns = match_gt_to_predictions(
        gt_annotations, predictions, week_lookup,
        distance_threshold=DISTANCE_THRESHOLD_PX,
    )

    status = {}
    for m in matched:
        gt = m["gt"]
        pred = m["pred"]
        tid = pred.get("tooth_id", gt.get("tooth_fdi", ""))
        status[tid] = "TP"
    for fp in fps:
        tid = fp["pred"].get("tooth_id", "")
        if tid and tid not in status:
            status[tid] = "FP"
    for fn in fns:
        tid = fn["gt"].get("tooth_fdi", "")
        if tid and tid not in status:
            status[tid] = "FN"

    return status, matched, fps, fns


# =====================================================================
# Zone Classification (for visualization — uses week7 MZ classifier)
# =====================================================================

def classify_zones_for_viz(
    tooth_id: str,
    tooth_polygon: List,
    caries_coords: List,
) -> Dict:
    """Run the week7 multi-zone classifier for visualization data."""
    if not caries_coords or not tooth_polygon:
        return {
            "zone_label": "N/A",
            "zone_fractions": {"M": 0, "C": 0, "D": 0},
            "all_zone_fractions": {"M": 0, "C": 0, "D": 0},
            "rotation_angle_deg": 0,
            "n_points": 0,
            "predicted_detail": "N/A",
            "predicted_surface_fine": "Unknown",
        }
    return classify_multi_zone(
        caries_points=caries_coords,
        pca_params={
            "tooth_id": tooth_id,
            "tooth_polygon": tooth_polygon,
        },
    )


# =====================================================================
# Drawing Helpers
# =====================================================================

def draw_gt_polygons(ax, gt_annotations: List[Dict], label_added: bool = False):
    """Draw ground truth ROI polygons on an axis."""
    for gt in gt_annotations:
        coords = gt.get("roi_coordinates", [])
        if len(coords) < 3:
            continue
        poly_pts = np.array(coords, dtype=np.float64)
        polygon = MplPolygon(
            poly_pts,
            closed=True,
            fill=False,
            edgecolor=GT_COLOR,
            linewidth=1.8,
            linestyle="-",
            label="Ground Truth (XML)" if not label_added else "",
            zorder=5,
        )
        ax.add_patch(polygon)
        label_added = True
    return label_added


def draw_pred_scatter(
    ax,
    predictions: List[Dict],
    week_lookup: Dict,
    label_added: bool = False,
):
    """Draw prediction caries points as scatter on an axis."""
    for pred in predictions:
        tid = pred.get("tooth_id", "")
        w = week_lookup.get(tid, {})
        # Task 2: skip teeth where has_caries is False after erosion
        if not w.get("has_caries", False):
            continue
        coords = w.get("caries_coordinates", [])
        if not coords:
            continue
        arr = np.array(coords, dtype=np.float64)
        ax.scatter(
            arr[:, 0], arr[:, 1],
            c=PRED_COLOR,
            s=1.5,
            alpha=0.6,
            marker=".",
            label="Prediction (Model)" if not label_added else "",
            zorder=6,
        )
        label_added = True
    return label_added


def draw_tooth_annotations(
    ax,
    predictions: List[Dict],
    gt_annotations: List[Dict],
    week_lookup: Dict,
    tooth_status: Dict,
    week2_teeth: Dict,
):
    """Draw text boxes with Tooth ID + F1 status near each tooth."""
    annotated = set()

    for pred in predictions:
        tid = pred.get("tooth_id", "")
        if not tid:
            continue
        status = tooth_status.get(tid, "FP")
        color = {"TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR}.get(status, FP_COLOR)

        w = week_lookup.get(tid, {})
        coords = w.get("caries_coordinates", [])
        if coords:
            arr = np.array(coords, dtype=np.float64)
            cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
        elif tid in week2_teeth:
            bbox = week2_teeth[tid].get("bbox", [])
            if len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[1] - 10
            else:
                continue
        else:
            continue

        # Task 2: skip FP teeth whose caries were removed by erosion
        if status == "FP" and not w.get("has_caries", False):
            continue

        label_text = f"#{tid} [{status}]"
        ax.annotate(
            label_text,
            xy=(cx, cy),
            fontsize=6,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=color,
                alpha=0.85,
                edgecolor="none",
            ),
            ha="center",
            va="bottom",
            xytext=(0, -15),
            textcoords="offset points",
            zorder=10,
        )
        annotated.add(tid)

    # Also annotate FN teeth from GT
    for gt in gt_annotations:
        tid = gt.get("tooth_fdi", "")
        if tid in annotated or not tid:
            continue
        status = tooth_status.get(tid, "")
        if status != "FN":
            continue
        cent = gt.get("roi_centroid", (0, 0))
        if cent[0] == 0 and cent[1] == 0:
            continue
        ax.annotate(
            f"#{tid} [FN]",
            xy=cent,
            fontsize=6,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=FN_COLOR,
                alpha=0.85,
                edgecolor="none",
            ),
            ha="center",
            va="bottom",
            xytext=(0, -15),
            textcoords="offset points",
            zorder=10,
        )


# =====================================================================
# PCA & M-C-D Tooth Sub-plot (Bottom Row)
# =====================================================================

def draw_pca_mcd_panel(
    ax,
    xray: np.ndarray,
    tooth_id: str,
    tooth_polygon: List,
    caries_coords: List,
    gt_surface: str,
    pred_surface: str,
    mz_result: Optional[Dict] = None,
    soft_match_status: str = "",
):
    """
    Draw a single tooth PCA & M-C-D panel.

    Week 7 changes:
      - Uses compute_pca_fixed() with 3-rule orientation (Task 1)
      - Shows MZ all_zone_fractions in the title (Task 4)
      - Shows soft-match status badge when applicable
    """
    if not tooth_polygon or not caries_coords:
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5, 0.5, f"Tooth #{tooth_id}\nNo caries data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="white",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    poly_arr = np.array(tooth_polygon, dtype=np.float64)
    caries_arr = np.array(caries_coords, dtype=np.float64)

    # ── Task 1: Compute PCA with fixed 3-rule orientation ────────────
    center, rot_angle, eigvecs = compute_pca_fixed(poly_arr, tooth_id)
    M = rotation_matrix_2x3(rot_angle, center[0], center[1])

    rot_poly = rotate_pts(poly_arr, M)
    rot_caries = rotate_pts(caries_arr, M)

    bx_min, by_min = rot_poly[:, 0].min(), rot_poly[:, 1].min()
    bx_max, by_max = rot_poly[:, 0].max(), rot_poly[:, 1].max()
    bw = bx_max - bx_min
    bh = by_max - by_min

    if bw <= 0 or bh <= 0:
        ax.text(0.5, 0.5, f"#{tooth_id}\nBad bbox", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # ── Crop ROI from original image ─────────────────────────────────
    ox_min = int(max(0, poly_arr[:, 0].min() - ROI_PAD_PX))
    oy_min = int(max(0, poly_arr[:, 1].min() - ROI_PAD_PX))
    ox_max = int(min(xray.shape[1], poly_arr[:, 0].max() + ROI_PAD_PX))
    oy_max = int(min(xray.shape[0], poly_arr[:, 1].max() + ROI_PAD_PX))
    roi = xray[oy_min:oy_max, ox_min:ox_max]

    ax.imshow(roi, cmap="gray", aspect="auto")

    # Offset coordinates to ROI space
    poly_roi = poly_arr.copy()
    poly_roi[:, 0] -= ox_min
    poly_roi[:, 1] -= oy_min
    caries_roi = caries_arr.copy()
    caries_roi[:, 0] -= ox_min
    caries_roi[:, 1] -= oy_min

    # ── Draw tooth boundary ──────────────────────────────────────────
    tooth_patch = MplPolygon(
        poly_roi, closed=True, fill=False,
        edgecolor="white", linewidth=1.0, linestyle=":", alpha=0.6,
    )
    ax.add_patch(tooth_patch)

    # ── PCA Long Axis (corrected eigenvector) ────────────────────────
    center_roi = center.copy()
    center_roi[0] -= ox_min
    center_roi[1] -= oy_min

    ev1 = eigvecs[0]  # major axis (corrected by 3-rule logic)
    axis_len = max(bw, bh) * 0.7
    p1 = center_roi + ev1 * axis_len
    p2 = center_roi - ev1 * axis_len
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]],
        color=PCA_AXIS_COLOR, linewidth=2.0, linestyle="-",
        label="PCA Long Axis", zorder=7,
    )

    # ── Zone Dividers (perpendicular to major axis) ──────────────────
    ev_perp = np.array([-ev1[1], ev1[0]])
    perp_len = max(bw, bh) * 0.5

    projections = np.dot(poly_roi - center_roi, ev1)
    proj_min = projections.min()
    proj_max = projections.max()
    proj_range = proj_max - proj_min

    for frac in [1.0 / 3.0, 2.0 / 3.0]:
        proj_pos = proj_min + frac * proj_range
        div_center = center_roi + ev1 * proj_pos
        dp1 = div_center + ev_perp * perp_len
        dp2 = div_center - ev_perp * perp_len
        ax.plot(
            [dp1[0], dp2[0]], [dp1[1], dp2[1]],
            color=DIVIDER_COLOR, linewidth=1.5, linestyle="--",
            alpha=0.8, zorder=7,
        )

    # ── Classify each caries point into M/C/D zone ──────────────────
    quadrant = _get_quadrant(tooth_id)

    rel_xs = np.clip((rot_caries[:, 0] - bx_min) / bw, 0, 1)
    third = 1.0 / 3.0

    # FIX: M/D Flip — Q1/Q4 mesial = right (+X), Q2/Q3 mesial = left (−X)
    if quadrant in [1, 4]:
        d_mask = rel_xs < third
        c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        m_mask = rel_xs > 2 * third
    else:
        m_mask = rel_xs < third
        c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        d_mask = rel_xs > 2 * third

    n_total = len(caries_arr)
    m_count = int(m_mask.sum())
    c_count = int(c_mask.sum())
    d_count = int(d_mask.sum())

    m_pct = m_count / n_total * 100 if n_total > 0 else 0
    c_pct = c_count / n_total * 100 if n_total > 0 else 0
    d_pct = d_count / n_total * 100 if n_total > 0 else 0

    # ── Plot caries points colored by zone ───────────────────────────
    if m_mask.any():
        ax.scatter(
            caries_roi[m_mask, 0], caries_roi[m_mask, 1],
            c=ZONE_M_COLOR, s=4, alpha=0.8, marker=".", zorder=8,
            label=f"Mesial ({m_pct:.0f}%)",
        )
    if c_mask.any():
        ax.scatter(
            caries_roi[c_mask, 0], caries_roi[c_mask, 1],
            c=ZONE_C_COLOR, s=4, alpha=0.8, marker=".", zorder=8,
            label=f"Central ({c_pct:.0f}%)",
        )
    if d_mask.any():
        ax.scatter(
            caries_roi[d_mask, 0], caries_roi[d_mask, 1],
            c=ZONE_D_COLOR, s=4, alpha=0.8, marker=".", zorder=8,
            label=f"Distal ({d_pct:.0f}%)",
        )

    # ── Build zone label using MZ result if available ────────────────
    if mz_result and mz_result.get("predicted_detail", "N/A") != "N/A":
        zone_label = mz_result["predicted_detail"]
        mz_fracs = mz_result.get("all_zone_fractions", {})
    else:
        # Fallback: compute locally
        threshold = 0.05
        m_frac = m_count / n_total if n_total > 0 else 0
        c_frac = c_count / n_total if n_total > 0 else 0
        d_frac = d_count / n_total if n_total > 0 else 0
        has_m = m_frac >= threshold
        has_c = c_frac >= threshold
        has_d = d_frac >= threshold
        if has_m and has_c and has_d: zone_label = "MOD"
        elif has_m and has_d: zone_label = "MOD"
        elif has_m and has_c: zone_label = "MO"
        elif has_d and has_c: zone_label = "DO"
        elif has_m: zone_label = "M"
        elif has_d: zone_label = "D"
        else: zone_label = "O"
        mz_fracs = {"M": round(m_frac, 4), "C": round(c_frac, 4), "D": round(d_frac, 4)}

    # ── Title with MZ fractions (Task 4 visibility) ──────────────────
    gt_label = gt_surface if gt_surface else "N/A"
    tooth_name = FDI_TOOTH_NAMES.get(tooth_id, tooth_id)

    title_line1 = f"Tooth #{tooth_id} ({tooth_name})"
    title_line2 = (f"M: {mz_fracs.get('M', 0)*100:.0f}%  "
                   f"C: {mz_fracs.get('C', 0)*100:.0f}%  "
                   f"D: {mz_fracs.get('D', 0)*100:.0f}%  ({n_total} pts)")
    title_line3 = f"Pred: {zone_label}  |  GT: {gt_label}"

    # Task 4: append soft-match badge if applicable
    if soft_match_status and soft_match_status not in ("True", "False"):
        title_line3 += f"  [{soft_match_status}]"

    ax.set_title(
        f"{title_line1}\n{title_line2}\n{title_line3}",
        fontsize=7,
        fontweight="bold",
        color="white",
        pad=4,
    )

    # ── Zone labels on the plot ──────────────────────────────────────
    # FIX: M/D Flip — label positions match corrected zone assignment
    zone_labels_pos = [
        ("M", ZONE_M_COLOR, 0.88 if quadrant in [1, 4] else 0.12),
        ("C", ZONE_C_COLOR, 0.50),
        ("D", ZONE_D_COLOR, 0.12 if quadrant in [1, 4] else 0.88),
    ]
    for zname, zcol, xfrac in zone_labels_pos:
        ax.text(
            xfrac, 0.95, zname,
            transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=zcol,
            ha="center", va="top",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            zorder=12,
        )

    # Legend
    ax.legend(
        loc="lower right", fontsize=5, framealpha=0.7,
        facecolor="black", edgecolor="gray",
        labelcolor="white", markerscale=3,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#1a1a2e")


# =====================================================================
# Main Dashboard Generator
# =====================================================================

def generate_dashboard(case_num: int, dpi: int = 150, verbose: bool = True):
    """
    Generate a full validation dashboard for a single case.
    """
    if verbose:
        print(f"[Case {case_num}] Loading data...")

    # ── Load all data sources ────────────────────────────────────────
    xray = load_xray(case_num)
    if xray is None:
        print(f"  [SKIP] No X-ray image found for case {case_num}")
        return

    gt_folder = MATERIAL_DIR / f"case {case_num}"
    gt_annotations = parse_case_xmls(str(gt_folder))

    predictions = load_week5_predictions(case_num)
    week_lookup = load_week7_caries(case_num)              # week7 caries data
    week2_teeth = load_week2_teeth(case_num)

    # ── Compute match status ─────────────────────────────────────────
    tooth_status, matched, fps, fns = compute_match_status(
        gt_annotations, predictions, week_lookup,
    )

    tp_count = len(matched)
    fp_count = len(fps)
    fn_count = len(fns)

    # ── Pre-compute MZ results and soft-match status for each tooth ──
    mz_cache: Dict[str, Dict] = {}
    soft_match_cache: Dict[str, str] = {}

    for m in matched:
        gt = m["gt"]
        pred = m["pred"]
        tid = pred.get("tooth_id", "")
        w = week_lookup.get(tid, {})
        coords = w.get("caries_coordinates", [])
        poly = week2_teeth.get(tid, {}).get("polygon", [])

        # Run MZ classifier
        if coords and poly:
            mz = classify_zones_for_viz(tid, poly, coords)
            mz_cache[tid] = mz

            # Task 4: compute soft match
            mz_fine = mz.get("predicted_surface_fine", "")
            sm = soft_surface_match(
                gt_surface_raw=gt.get("surface_name", ""),
                pred_surface_fine=mz_fine,
                mz_info=mz,
            )
            soft_match_cache[tid] = sm

    # ── Determine which teeth to show in bottom row ──────────────────
    teeth_to_viz = []

    for pred in predictions:
        tid = pred.get("tooth_id", "")
        if not tid:
            continue
        w = week_lookup.get(tid, {})

        # Task 2: only show teeth that have caries after erosion
        if not w.get("has_caries", False):
            continue

        coords = w.get("caries_coordinates", [])
        poly = week2_teeth.get(tid, {}).get("polygon", [])

        # Find matching GT surface
        gt_surf = ""
        for m in matched:
            if m["pred"].get("tooth_id") == tid:
                gt_surf = m["gt"].get("surface_name", "")
                break

        teeth_to_viz.append({
            "tooth_id": tid,
            "polygon": poly,
            "caries_coords": coords,
            "gt_surface": gt_surf,
            "pred_surface": pred.get("caries_position_detail", ""),
            "status": tooth_status.get(tid, "FP"),
            "mz_result": mz_cache.get(tid),
            "soft_match": soft_match_cache.get(tid, ""),
        })

    # Also add FN teeth
    for fn in fns:
        gt = fn["gt"]
        tid = gt.get("tooth_fdi", "")
        if not tid or any(t["tooth_id"] == tid for t in teeth_to_viz):
            continue
        poly = week2_teeth.get(tid, {}).get("polygon", [])
        roi_coords = gt.get("roi_coordinates", [])
        teeth_to_viz.append({
            "tooth_id": tid,
            "polygon": poly,
            "caries_coords": roi_coords,
            "gt_surface": gt.get("surface_name", ""),
            "pred_surface": "MISSED",
            "status": "FN",
            "mz_result": None,
            "soft_match": "",
        })

    n_teeth = max(len(teeth_to_viz), 1)
    n_cols = min(n_teeth, 6)
    n_rows_bottom = math.ceil(n_teeth / n_cols) if n_teeth > 0 else 1

    # ── Create figure ────────────────────────────────────────────────
    fig_w = max(16, n_cols * 3.2)
    fig_h = 8 + n_rows_bottom * 4.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")

    gs = fig.add_gridspec(
        nrows=1 + n_rows_bottom,
        ncols=n_cols,
        height_ratios=[2.0] + [1.5] * n_rows_bottom,
        hspace=0.35,
        wspace=0.25,
    )

    # ══════════════════════════════════════════════════════════════════
    # TOP HALF: Global View
    # ══════════════════════════════════════════════════════════════════
    ax_global = fig.add_subplot(gs[0, :])
    ax_global.set_facecolor("#0d1117")

    ax_global.imshow(xray, cmap="gray", aspect="auto")

    gt_label_done = draw_gt_polygons(ax_global, gt_annotations)
    pred_label_done = draw_pred_scatter(ax_global, predictions, week_lookup)
    draw_tooth_annotations(
        ax_global, predictions, gt_annotations,
        week_lookup, tooth_status, week2_teeth,
    )

    # ── Global title & legend ────────────────────────────────────────
    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    ax_global.set_title(
        f"VALIDATION DASHBOARD — Case {case_num}  (week7)\n"
        f"TP={tp_count}  FP={fp_count}  FN={fn_count}  |  "
        f"Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=10,
    )

    handles = []
    if gt_label_done:
        handles.append(Line2D([0], [0], color=GT_COLOR, linewidth=2,
                              linestyle="-", label="Ground Truth (XML)"))
    if pred_label_done:
        handles.append(Line2D([0], [0], color=PRED_COLOR, linewidth=0,
                              marker=".", markersize=8,
                              label="Prediction (Model)"))
    handles.append(mpatches.Patch(color=TP_COLOR, label=f"TP ({tp_count})"))
    handles.append(mpatches.Patch(color=FP_COLOR, label=f"FP ({fp_count})"))
    handles.append(mpatches.Patch(color=FN_COLOR, label=f"FN ({fn_count})"))

    ax_global.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.8,
        facecolor="#1a1a2e",
        edgecolor="gray",
        labelcolor="white",
    )

    ax_global.set_xticks([])
    ax_global.set_yticks([])

    # ══════════════════════════════════════════════════════════════════
    # BOTTOM HALF: PCA & M-C-D Logic Per Tooth
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"  Drawing {n_teeth} tooth PCA panels...")

    for idx, tooth_info in enumerate(teeth_to_viz):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1a1a2e")

        draw_pca_mcd_panel(
            ax=ax,
            xray=xray,
            tooth_id=tooth_info["tooth_id"],
            tooth_polygon=tooth_info["polygon"],
            caries_coords=tooth_info["caries_coords"],
            gt_surface=tooth_info["gt_surface"],
            pred_surface=tooth_info["pred_surface"],
            mz_result=tooth_info.get("mz_result"),
            soft_match_status=tooth_info.get("soft_match", ""),
        )

        # Add status border
        status = tooth_info["status"]
        border_color = {
            "TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR
        }.get(status, "gray")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

    # Hide unused sub-axes
    for idx in range(n_teeth, n_rows_bottom * n_cols):
        row = 1 + idx // n_cols
        col = idx % n_cols
        if row < 1 + n_rows_bottom and col < n_cols:
            ax_empty = fig.add_subplot(gs[row, col])
            ax_empty.set_visible(False)

    # ── Footer ───────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "week7 | PCA Long Axis (Blue, 3-rule fix) | Zone Dividers (Red dashed) | "
        "Mesial=Cyan | Central=Yellow | Distal=Magenta | "
        "Threshold >= 5% (G.V. Black) | Soft match >= 30%",
        ha="center", fontsize=8, color="#888888",
        style="italic",
    )

    # ── Save ─────────────────────────────────────────────────────────
    case_dir = OUTPUT_DIR / f"case {case_num}"
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / f"validation_case_{case_num}.png"
    fig.savefig(
        str(out_path),
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)

    if verbose:
        print(f"  [SAVED] {out_path}")
        print(f"  Summary: TP={tp_count}, FP={fp_count}, FN={fn_count}, "
              f"F1={f1:.3f}, Teeth visualized={n_teeth}")


# =====================================================================
# Batch Generation
# =====================================================================

def generate_batch(
    case_list: Optional[List[int]] = None,
    start: int = 1,
    end: int = 500,
    dpi: int = 150,
    verbose: bool = True,
):
    """Generate dashboards for a list or range of cases."""
    cases = case_list if case_list else list(range(start, end + 1))

    print("=" * 70)
    print("DENTAL CARIES VALIDATION DASHBOARD — week7")
    print("=" * 70)
    print(f"Cases: {cases[:5]}{'...' if len(cases) > 5 else ''}  |  Output: {OUTPUT_DIR}")
    print("=" * 70)

    processed = 0
    skipped = 0

    for case_num in cases:
        img_path = MATERIAL_DIR / f"case {case_num}" / f"case_{case_num}.png"
        if not img_path.exists():
            skipped += 1
            continue
        try:
            generate_dashboard(case_num, dpi=dpi, verbose=verbose)
            processed += 1
        except Exception as e:
            print(f"  [ERROR] Case {case_num}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1

    print(f"\nDone. Processed={processed}, Skipped={skipped}")
    print(f"Output -> {OUTPUT_DIR}")


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Dental Caries Validation Dashboard — week7"
    )
    parser.add_argument("--case", type=int, default=None,
                        help="Generate for a single case")
    parser.add_argument("--sample", type=str, default=None,
                        help="Comma-separated case numbers, e.g. --sample 311,33")
    parser.add_argument("--start", type=int, default=1,
                        help="Start case number for batch mode")
    parser.add_argument("--end", type=int, default=500,
                        help="End case number for batch mode")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI (default 150)")
    args = parser.parse_args()

    if args.case is not None:
        generate_dashboard(args.case, dpi=args.dpi)
    elif args.sample:
        case_list = [int(c.strip()) for c in args.sample.split(",")]
        generate_batch(case_list=case_list, dpi=args.dpi)
    else:
        generate_batch(start=args.start, end=args.end, dpi=args.dpi)


if __name__ == "__main__":
    main()
