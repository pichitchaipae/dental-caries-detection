"""
Dental Caries Validation Dashboard
====================================

Human-in-the-Loop visualization tool for verifying model predictions
against AIM XML Ground Truth annotations.

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

Inputs:
  - X-ray:  material/500 cases with annotation/case N/case_N.png
  - GT XML: material/500 cases with annotation/case N/*.xml
  - Pred:   week5 JSON + week3 caries_coordinates + week2 tooth polygons

Output:
  - week6/validation_dashboard/validation_case_{N}.png

Usage:
  python validation_dashboard.py --case 1
  python validation_dashboard.py --start 1 --end 10
  python validation_dashboard.py --case 1 --dpi 200

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026
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

# ── Local imports (same directory) ───────────────────────────────────
from xml_ground_truth_parser import parse_case_xmls
from multi_zone_classifier import classify_multi_zone
from snodent_tooth_map import FDI_TOOTH_NAMES
from evaluation_engine import (
    normalize_surface,
    normalize_surface_fine,
    match_gt_to_predictions,
    DISTANCE_THRESHOLD_PX,
)


# =====================================================================
# Configuration
# =====================================================================

BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")

MATERIAL_DIR = BASE_DIR / "material" / "500 cases with annotation"
WEEK2_DIR    = BASE_DIR / "week2"  / "500-segmentation+recognition"
WEEK3_DIR    = BASE_DIR / "week3"  / "dental_analysis_output"
WEEK5_DIR    = BASE_DIR / "week5"  / "surface_classification_output"
OUTPUT_DIR   = BASE_DIR / "week6"  / "evaluation_output"

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
        # Get polygon (convex hull of pixel_coordinates)
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


def load_week3_caries(case_num: int) -> Dict:
    """Load week3 caries mapping → {tooth_id: {caries_coordinates, ...}}."""
    p = WEEK3_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        t["tooth_id"]: t
        for t in data.get("teeth_caries_data", [])
        if t.get("tooth_id")
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
# PCA Geometry Helpers
# =====================================================================

def compute_pca(pts: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute PCA on a point cloud.
    Returns: (center, rotation_angle_rad, eigenvectors_2x2)
    """
    pts = pts.astype(np.float64)
    center = np.mean(pts, axis=0)
    centered = pts - center
    mean_out, eigvecs = cv2.PCACompute(centered, mean=None)
    major = eigvecs[0]
    angle = math.atan2(major[1], major[0])
    # Rotation to make major axis vertical
    rot = math.pi / 2 - angle
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi
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
    week3_lookup: Dict,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Match GT ↔ Pred and return:
      tooth_status : {tooth_id: "TP"/"FP"/"FN"}
      matched_pairs: list of dicts
      Also returns lists for FP and FN
    """
    matched, fps, fns = match_gt_to_predictions(
        gt_annotations, predictions, week3_lookup,
        distance_threshold=DISTANCE_THRESHOLD_PX,
    )

    status = {}

    # TP: matched teeth
    for m in matched:
        gt = m["gt"]
        pred = m["pred"]
        tid = pred.get("tooth_id", gt.get("tooth_fdi", ""))
        status[tid] = "TP"

    # FP: predictions with no GT match
    for fp in fps:
        tid = fp["pred"].get("tooth_id", "")
        if tid and tid not in status:
            status[tid] = "FP"

    # FN: GT with no prediction match
    for fn in fns:
        tid = fn["gt"].get("tooth_fdi", "")
        if tid and tid not in status:
            status[tid] = "FN"

    return status, matched, fps, fns


# =====================================================================
# Zone Classification (recompute for visualization)
# =====================================================================

def classify_zones_for_viz(
    tooth_id: str,
    tooth_polygon: List,
    caries_coords: List,
) -> Dict:
    """
    Run multi-zone classifier and return zone data for visualization.
    """
    if not caries_coords or not tooth_polygon:
        return {
            "zone_label": "N/A",
            "zone_fractions": {"M": 0, "C": 0, "D": 0},
            "rotation_angle_deg": 0,
            "n_points": 0,
        }
    result = classify_multi_zone(
        caries_points=caries_coords,
        pca_params={
            "tooth_id": tooth_id,
            "tooth_polygon": tooth_polygon,
        },
    )
    return result


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
    week3_lookup: Dict,
    label_added: bool = False,
):
    """Draw prediction caries points as scatter on an axis."""
    for pred in predictions:
        tid = pred.get("tooth_id", "")
        w3 = week3_lookup.get(tid, {})
        coords = w3.get("caries_coordinates", [])
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
    week3_lookup: Dict,
    tooth_status: Dict,
    week2_teeth: Dict,
):
    """Draw text boxes with Tooth ID + F1 status near each tooth."""
    # Collect annotation positions from predictions (using caries centroids)
    annotated = set()

    for pred in predictions:
        tid = pred.get("tooth_id", "")
        if not tid:
            continue
        status = tooth_status.get(tid, "FP")
        color = {"TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR}.get(status, FP_COLOR)

        # Get position from week3 caries centroid or week2 bbox
        w3 = week3_lookup.get(tid, {})
        coords = w3.get("caries_coordinates", [])
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
):
    """
    Draw a single tooth PCA & M-C-D panel:
      - Cropped ROI
      - PCA long axis (blue)
      - Zone dividers perpendicular to PCA axis
      - Caries points colored by zone (Cyan/Yellow/Magenta)
      - Zone percentages + Final label header
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

    # ── Compute PCA ──────────────────────────────────────────────────
    center, rot_angle, eigvecs = compute_pca(poly_arr)
    M = rotation_matrix_2x3(rot_angle, center[0], center[1])

    rot_poly = rotate_pts(poly_arr, M)
    rot_caries = rotate_pts(caries_arr, M)

    # Bounding box of rotated tooth
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

    # ── Crop the ROI from original image ─────────────────────────────
    ox_min = int(max(0, poly_arr[:, 0].min() - ROI_PAD_PX))
    oy_min = int(max(0, poly_arr[:, 1].min() - ROI_PAD_PX))
    ox_max = int(min(xray.shape[1], poly_arr[:, 0].max() + ROI_PAD_PX))
    oy_max = int(min(xray.shape[0], poly_arr[:, 1].max() + ROI_PAD_PX))
    roi = xray[oy_min:oy_max, ox_min:ox_max]

    # Show ROI
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

    # ── PCA Long Axis (Eigenvector 1) ────────────────────────────────
    center_roi = center.copy()
    center_roi[0] -= ox_min
    center_roi[1] -= oy_min

    # Major eigenvector direction
    ev1 = eigvecs[0]  # major axis
    axis_len = max(bw, bh) * 0.7
    p1 = center_roi + ev1 * axis_len
    p2 = center_roi - ev1 * axis_len
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]],
        color=PCA_AXIS_COLOR, linewidth=2.0, linestyle="-",
        label="PCA Long Axis", zorder=7,
    )

    # ── Zone Dividers (perpendicular to major axis) ──────────────────
    # Divide the tooth into 3 equal sections along the major axis
    ev_perp = np.array([-ev1[1], ev1[0]])  # perpendicular
    perp_len = max(bw, bh) * 0.5

    # Project all polygon points onto the major axis to find extents
    projections = np.dot(poly_roi - center_roi, ev1)
    proj_min = projections.min()
    proj_max = projections.max()
    proj_range = proj_max - proj_min

    # Two divider lines at 1/3 and 2/3 of the range
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
    quadrant = int(tooth_id[0]) if tooth_id else 1

    # Project caries points onto major axis
    caries_proj = np.dot(caries_roi - center_roi, ev1)
    third_1 = proj_min + proj_range / 3.0
    third_2 = proj_min + 2 * proj_range / 3.0

    # Zone assignment (along major axis direction)
    # For FDI: Q1/Q4 → mesial is towards midline = one end
    # For Q2/Q3 → mesial is towards midline = other end
    # We'll use the same convention as multi_zone_classifier
    # but map by projection position: low/mid/high along axis
    # To maintain consistency, we use rotated space classification
    # Relative x in rotated space → determine M/C/D
    rel_xs = np.clip((rot_caries[:, 0] - bx_min) / bw, 0, 1)
    third = 1.0 / 3.0

    if quadrant in [1, 4]:
        m_mask = rel_xs < third
        c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        d_mask = rel_xs > 2 * third
    else:
        d_mask = rel_xs < third
        c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        m_mask = rel_xs > 2 * third

    n_total = len(caries_arr)
    n_occ = int(m_mask.sum()) + int(c_mask.sum()) + int(d_mask.sum())

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

    # ── Build zone label ─────────────────────────────────────────────
    threshold = 0.20
    m_frac = m_count / n_total if n_total > 0 else 0
    c_frac = c_count / n_total if n_total > 0 else 0
    d_frac = d_count / n_total if n_total > 0 else 0

    has_m = m_frac >= threshold
    has_c = c_frac >= threshold
    has_d = d_frac >= threshold

    if has_m and has_c and has_d:
        zone_label = "MOD"
    elif has_m and has_d:
        zone_label = "MOD"
    elif has_m and has_c:
        zone_label = "MO"
    elif has_d and has_c:
        zone_label = "DO"
    elif has_m:
        zone_label = "M"
    elif has_d:
        zone_label = "D"
    elif has_c:
        zone_label = "O"
    else:
        zone_label = "O"

    # ── Title header ─────────────────────────────────────────────────
    gt_label = gt_surface if gt_surface else "N/A"
    tooth_name = FDI_TOOTH_NAMES.get(tooth_id, tooth_id)

    title_line1 = f"Tooth #{tooth_id} ({tooth_name})"
    title_line2 = f"M: {m_pct:.0f}%  C: {c_pct:.0f}%  D: {d_pct:.0f}%  ({n_total} pts)"
    title_line3 = f"Pred: {zone_label}  |  GT: {gt_label}"

    ax.set_title(
        f"{title_line1}\n{title_line2}\n{title_line3}",
        fontsize=7,
        fontweight="bold",
        color="white",
        pad=4,
    )

    # ── Zone labels on the plot ──────────────────────────────────────
    roi_h, roi_w = roi.shape[:2]
    zone_labels_pos = [
        ("M", ZONE_M_COLOR, 0.12 if quadrant in [1, 4] else 0.88),
        ("C", ZONE_C_COLOR, 0.50),
        ("D", ZONE_D_COLOR, 0.88 if quadrant in [1, 4] else 0.12),
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

    Parameters
    ----------
    case_num : int
        Case number (1–500).
    dpi : int
        Output image resolution.
    verbose : bool
        Print progress info.
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
    week3_lookup = load_week3_caries(case_num)
    week2_teeth = load_week2_teeth(case_num)

    # ── Compute match status ─────────────────────────────────────────
    tooth_status, matched, fps, fns = compute_match_status(
        gt_annotations, predictions, week3_lookup,
    )

    tp_count = len(matched)
    fp_count = len(fps)
    fn_count = len(fns)

    # ── Determine which teeth to show in bottom row ──────────────────
    # Show all teeth that have caries (from predictions + FN from GT)
    teeth_to_viz = []

    for pred in predictions:
        tid = pred.get("tooth_id", "")
        if not tid or not pred.get("has_caries", False):
            continue
        w3 = week3_lookup.get(tid, {})
        coords = w3.get("caries_coordinates", [])
        poly = week2_teeth.get(tid, {}).get("polygon", [])

        # Find matching GT surface for this tooth
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
        })

    # Also add FN teeth (GT with no prediction)
    for fn in fns:
        gt = fn["gt"]
        tid = gt.get("tooth_fdi", "")
        if not tid or any(t["tooth_id"] == tid for t in teeth_to_viz):
            continue
        poly = week2_teeth.get(tid, {}).get("polygon", [])
        # FN teeth have GT ROI but no model prediction caries coords
        # Use GT ROI coords as approximate caries region for display
        roi_coords = gt.get("roi_coordinates", [])
        teeth_to_viz.append({
            "tooth_id": tid,
            "polygon": poly,
            "caries_coords": roi_coords,
            "gt_surface": gt.get("surface_name", ""),
            "pred_surface": "MISSED",
            "status": "FN",
        })

    n_teeth = max(len(teeth_to_viz), 1)
    n_cols = min(n_teeth, 6)
    n_rows_bottom = math.ceil(n_teeth / n_cols) if n_teeth > 0 else 1

    # ── Create figure ────────────────────────────────────────────────
    fig_w = max(16, n_cols * 3.2)
    fig_h = 8 + n_rows_bottom * 4.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")

    # GridSpec: top row = global view, bottom rows = per-tooth panels
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

    # Display X-ray
    ax_global.imshow(xray, cmap="gray", aspect="auto")

    # Overlay GT polygons (green, solid)
    gt_label_done = draw_gt_polygons(ax_global, gt_annotations)

    # Overlay prediction scatter (red)
    pred_label_done = draw_pred_scatter(
        ax_global, predictions, week3_lookup,
    )

    # Draw tooth annotation boxes
    draw_tooth_annotations(
        ax_global, predictions, gt_annotations,
        week3_lookup, tooth_status, week2_teeth,
    )

    # ── Global title & legend ────────────────────────────────────────
    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    ax_global.set_title(
        f"VALIDATION DASHBOARD — Case {case_num}\n"
        f"TP={tp_count}  FP={fp_count}  FN={fn_count}  |  "
        f"Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=10,
    )

    # Build legend handles
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
        "PCA Long Axis (Blue) · Zone Dividers (Red dashed) · "
        "Mesial=Cyan · Central=Yellow · Distal=Magenta · "
        "Threshold ≥ 20% for zone inclusion",
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
    start: int = 1,
    end: int = 500,
    dpi: int = 150,
    verbose: bool = True,
):
    """Generate dashboards for a range of cases."""
    print("=" * 70)
    print("DENTAL CARIES VALIDATION DASHBOARD GENERATOR")
    print("=" * 70)
    print(f"Cases: {start}–{end}  |  Output: {OUTPUT_DIR}")
    print("=" * 70)

    processed = 0
    skipped = 0

    for case_num in range(start, end + 1):
        img_path = MATERIAL_DIR / f"case {case_num}" / f"case_{case_num}.png"
        if not img_path.exists():
            skipped += 1
            continue
        try:
            generate_dashboard(case_num, dpi=dpi, verbose=verbose)
            processed += 1
        except Exception as e:
            print(f"  [ERROR] Case {case_num}: {e}")
            skipped += 1

    print(f"\nDone. Processed={processed}, Skipped={skipped}")
    print(f"Output → {OUTPUT_DIR}")


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Dental Caries Validation Dashboard – week6"
    )
    parser.add_argument("--case", type=int, default=None,
                        help="Generate for a single case")
    parser.add_argument("--start", type=int, default=1,
                        help="Start case number for batch mode")
    parser.add_argument("--end", type=int, default=500,
                        help="End case number for batch mode")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI (default 150)")
    args = parser.parse_args()

    if args.case is not None:
        generate_dashboard(args.case, dpi=args.dpi)
    else:
        generate_batch(args.start, args.end, dpi=args.dpi)


if __name__ == "__main__":
    main()
