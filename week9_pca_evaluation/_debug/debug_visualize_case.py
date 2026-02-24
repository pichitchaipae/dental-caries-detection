# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Week 9 — Debug Visualization Tool
====================================

Generates a comprehensive 5-panel comparison figure for a single case,
showing how **each PCA Method** affected orientation, zone classification,
and caries-to-tooth mapping.

Layers drawn per subplot:
  A) PCA Vertical Axis — cyan line from tooth centroid along the principal axis
  B) Mask Overlays     — semi-transparent colour per surface class
  C) Bounding Boxes    — white for teeth, red for caries
  D) Caries↔Tooth map  — magenta line linking caries centroid to tooth centre

Usage
-----
    cd week9_pca_evaluation
    python debug_visualize_case.py --case_id 311
    python debug_visualize_case.py --case_id 311 --tooth 25   # zoom single tooth

Output
------
    week9_pca_evaluation/debug_viz_case_311_combined.png

Author: Expert CV Engineer — Dental AI / CAD
Date:   2026-02-23
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
WEEK9_DIR  = Path(__file__).resolve().parent
WEEK2_DIR  = BASE_DIR / "week2-Tooth Detection & Segmentation" / "500-segmentation+recognition"
ROI_DIR    = BASE_DIR / "raw_data" / "500-roi"

METHODS: List[int]  = [0, 1, 2, 3, 5]

PCA_METHOD_NAMES = {
    0: "method_0_baseline_opencv",
    1: "method_1_square_heuristic",
    2: "method_2_max_span",
    3: "method_3_split_centroid",
    5: "method_5_vertical_prior",
}

# Surface → colour map  (BGR for cv2, but we convert to RGB for mpl)
SURFACE_COLOURS = {
    "Distal":       (220,  50,  50),   # Red
    "Mesial":       ( 50, 200,  50),   # Green
    "Occlusal":     ( 50, 100, 240),   # Blue
    "Proximal":     (200, 200,  50),   # Yellow
    "Other":        (180, 180, 180),   # Grey
    "Unclassified": (120, 120, 120),   # Dark grey
}

ALPHA_MASK   = 0.35    # transparency for surface overlays
ALPHA_CARIES = 0.50    # transparency for caries pixels
AXIS_LEN     = 120     # PCA axis line half-length (pixels)


# =============================================================================
# Data loading helpers
# =============================================================================

def load_base_image(case_num: int) -> Optional[np.ndarray]:
    """Load the ROI dental image, return as RGB ndarray or None."""
    img_path = ROI_DIR / f"case_{case_num}.png"
    if not img_path.exists():
        # Try jpg
        img_path = ROI_DIR / f"case_{case_num}.jpg"
    if not img_path.exists():
        print(f"  [WARN] Base image not found for case {case_num}")
        return None
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_week2_data(case_num: int) -> Optional[Dict]:
    """Load week2 detection+segmentation JSON for a case."""
    json_path = WEEK2_DIR / f"case {case_num}" / f"case_{case_num}_results.json"
    if not json_path.exists():
        print(f"  [WARN] Week2 data not found: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_diagnosis_json(case_num: int, method: int) -> Optional[Dict]:
    """Load the per-method diagnosis JSON from week9 output."""
    method_name = PCA_METHOD_NAMES[method]
    json_path = (
        WEEK9_DIR / method_name / "cases" / f"case {case_num}"
        / f"case_{case_num}_diagnosis.json"
    )
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Geometry helpers
# =============================================================================

def _centroid_of_coords(coords: List[List[int]]) -> Tuple[float, float]:
    """Return (cx, cy) mean of pixel coordinate list."""
    if not coords:
        return (0.0, 0.0)
    arr = np.array(coords, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _bbox_from_coords(coords: List[List[int]]) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) for a set of pixel coordinates."""
    if not coords:
        return (0, 0, 0, 0)
    arr = np.array(coords, dtype=np.int32)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _bbox_from_xyxy(bbox: List[float]) -> Tuple[int, int, int, int]:
    """Convert [x1, y1, x2, y2] float bbox to int tuple."""
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_pca_axis(
    canvas: np.ndarray,
    cx: float, cy: float,
    rotation_angle_deg: float,
    colour: Tuple[int, ...] = (0, 255, 255),   # cyan
    length: int = AXIS_LEN,
    thickness: int = 3,
):
    """
    Draw the PCA vertical axis on *canvas* (RGB).

    The ``rotation_angle_deg`` is the angle (in degrees) that the PCA
    vertical axis deviates from the image Y-axis, measured as a CCW rotation.
    We draw a line from (cx, cy) pointing in that direction.
    """
    angle_rad = math.radians(rotation_angle_deg)

    # Direction: the PCA axis is a rotation of the "up" direction (-Y in image coords)
    # by the given angle.  dx, dy give the offset.
    dx = length * math.sin(angle_rad)
    dy = -length * math.cos(angle_rad)     # negative because image Y is inverted

    x_end = int(cx + dx)
    y_end = int(cy + dy)
    x_start = int(cx - dx)
    y_start = int(cy - dy)

    cv2.line(canvas, (x_start, y_start), (x_end, y_end), colour, thickness, cv2.LINE_AA)
    # Draw a small diamond at the centroid
    cv2.drawMarker(canvas, (int(cx), int(cy)), colour,
                   cv2.MARKER_DIAMOND, 10, thickness, cv2.LINE_AA)


def draw_mask_overlay(
    canvas: np.ndarray,
    coords: List[List[int]],
    colour: Tuple[int, ...],
    alpha: float = ALPHA_CARIES,
):
    """Overlay semi-transparent coloured pixels on *canvas* (RGB, modified in-place)."""
    if not coords:
        return
    overlay = canvas.copy()
    for x, y in coords:
        if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
            overlay[y, x] = colour
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_tooth_mask_overlay(
    canvas: np.ndarray,
    coords: List[List[int]],
    colour: Tuple[int, ...],
    alpha: float = ALPHA_MASK,
):
    """Overlay semi-transparent tooth segmentation mask on *canvas*."""
    if not coords:
        return
    overlay = canvas.copy()
    for x, y in coords:
        if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
            overlay[y, x] = colour
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_bbox(
    canvas: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: Tuple[int, ...] = (255, 255, 255),
    thickness: int = 2,
    label: str = "",
):
    """Draw a rectangle and optional label on *canvas*."""
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    if label:
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_mapping_line(
    canvas: np.ndarray,
    cx_from: float, cy_from: float,
    cx_to: float, cy_to: float,
    colour: Tuple[int, ...] = (255, 0, 255),   # magenta
    thickness: int = 1,
):
    """Draw a thin line connecting a caries centroid to its tooth centroid."""
    cv2.line(canvas, (int(cx_from), int(cy_from)),
             (int(cx_to), int(cy_to)), colour, thickness, cv2.LINE_AA)


# =============================================================================
# Build per-method tooth lookup from week2
# =============================================================================

def _build_week2_tooth_index(week2_data: Dict) -> Dict[str, Dict]:
    """
    Build a dict   { tooth_id : { "bbox": ..., "pixel_coordinates": ..., "centroid": ... } }
    from week2 JSON.
    """
    idx = {}
    for t in week2_data.get("teeth_data", []):
        tid = str(t.get("tooth_id", ""))
        bbox_raw = t.get("bbox", [])
        px_coords = t.get("pixel_coordinates", [])
        # Flatten segments_detail pixel coordinates into one list
        for seg in t.get("segments_detail", []):
            px_coords.extend(seg.get("pixel_coordinates", []))
        cx, cy = _centroid_of_coords(px_coords) if px_coords else (0, 0)
        if bbox_raw and len(bbox_raw) == 4:
            bbox = _bbox_from_xyxy(bbox_raw)
        elif px_coords:
            bbox = _bbox_from_coords(px_coords)
        else:
            bbox = (0, 0, 0, 0)
        idx[tid] = {
            "bbox": bbox,
            "pixel_coordinates": px_coords,
            "centroid": (cx, cy),
        }
    return idx


# =============================================================================
# Main rendering function
# =============================================================================

def render_method_panel(
    base_img: np.ndarray,
    diag: Dict,
    week2_index: Dict[str, Dict],
    filter_tooth: Optional[str] = None,
) -> np.ndarray:
    """
    Render all debug layers for ONE method onto a copy of the base image.

    Parameters
    ----------
    base_img : RGB image (H, W, 3)
    diag : The diagnosis JSON dict for this method/case
    week2_index : tooth_id → { bbox, pixel_coordinates, centroid }
    filter_tooth : If given, only render this tooth_id

    Returns
    -------
    canvas : annotated RGB image
    """
    canvas = base_img.copy()

    for tooth in diag.get("teeth_data", []):
        tid = str(tooth.get("tooth_id", ""))
        if filter_tooth and tid != filter_tooth:
            continue

        has_caries = tooth.get("has_caries", False)
        surface    = tooth.get("caries_surface", "Unclassified")
        rot_deg    = tooth.get("rotation_angle", 0.0)
        caries_coords = tooth.get("caries_coordinates", [])

        w2 = week2_index.get(tid, {})
        tooth_px     = w2.get("pixel_coordinates", [])
        tooth_bbox   = w2.get("bbox", (0, 0, 0, 0))
        tooth_centre = w2.get("centroid", (0, 0))

        # ── Layer B: Tooth mask (light grey, very faint) ─────────────
        if tooth_px:
            draw_tooth_mask_overlay(canvas, tooth_px, (200, 200, 220), alpha=0.15)

        # ── Layer C-1: Tooth bounding box (white) ────────────────────
        if tooth_bbox != (0, 0, 0, 0):
            draw_bbox(canvas, *tooth_bbox, colour=(255, 255, 255),
                      thickness=1, label=f"T{tid}")

        if not has_caries:
            continue

        # ── Layer B: Caries mask overlay (coloured by surface) ───────
        surf_colour = SURFACE_COLOURS.get(surface, SURFACE_COLOURS["Unclassified"])
        if caries_coords:
            draw_mask_overlay(canvas, caries_coords, surf_colour, alpha=ALPHA_CARIES)

        # ── Layer C-2: Caries bounding box (red) ─────────────────────
        if caries_coords:
            c_bbox = _bbox_from_coords(caries_coords)
            draw_bbox(canvas, *c_bbox, colour=(255, 60, 60),
                      thickness=2, label=f"{surface}")

        # ── Layer D: Caries→Tooth mapping line (magenta) ─────────────
        if caries_coords and tooth_centre != (0, 0):
            ccx, ccy = _centroid_of_coords(caries_coords)
            draw_mapping_line(canvas, ccx, ccy,
                              tooth_centre[0], tooth_centre[1])

        # ── Layer A: PCA Axis (cyan) — MOST IMPORTANT ────────────────
        if tooth_px and caries_coords:
            # Draw from the tooth centroid
            tcx, tcy = tooth_centre
            draw_pca_axis(canvas, tcx, tcy, rot_deg,
                          colour=(0, 255, 255), length=AXIS_LEN, thickness=3)

    return canvas


# =============================================================================
# Main figure assembly
# =============================================================================

def generate_comparison_figure(
    case_num: int,
    filter_tooth: Optional[str] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Build a 1×5 panel comparing all PCA methods for *case_num*.

    Returns the saved file path, or None on failure.
    """
    print(f"\n{'=' * 60}")
    print(f"  Debug Visualization — Case {case_num}")
    print(f"{'=' * 60}")

    # Load base image
    base_img = load_base_image(case_num)
    if base_img is None:
        print("  [ERROR] Cannot load base image. Aborting.")
        return None

    # Load week2 data
    week2_data = load_week2_data(case_num)
    week2_index = _build_week2_tooth_index(week2_data) if week2_data else {}

    h, w = base_img.shape[:2]
    n_methods = len(METHODS)

    # Determine figure size: each panel shows the full image;
    # scale so each panel is ~6 inches wide
    panel_w = 6.0
    panel_h = panel_w * (h / w)
    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(panel_w * n_methods, panel_h + 1.2),
        constrained_layout=True,
    )
    if n_methods == 1:
        axes = [axes]

    fig.suptitle(
        f"Case {case_num} — PCA Method Comparison"
        + (f"  (Tooth {filter_tooth})" if filter_tooth else ""),
        fontsize=16, fontweight="bold", y=1.02,
    )

    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        method_name = PCA_METHOD_NAMES[method]

        print(f"  Loading data for Method {method} ({method_name})...")

        diag = load_diagnosis_json(case_num, method)
        if diag is None:
            print(f"    [WARN] Data missing for method {method}, case {case_num}")
            ax.text(0.5, 0.5, "Data Missing",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="red")
            ax.set_facecolor("#f0f0f0")
            ax.set_title(f"Method {method}\n{method_name}", fontsize=9, fontweight="bold")
            ax.axis("off")
            continue

        panel = render_method_panel(base_img, diag, week2_index, filter_tooth)
        ax.imshow(panel)
        ax.set_title(f"Method {method}\n{method_name}", fontsize=9, fontweight="bold")
        ax.axis("off")

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = []
    for surf, col in SURFACE_COLOURS.items():
        if surf in ("Distal", "Mesial", "Occlusal"):
            legend_handles.append(
                mpatches.Patch(color=np.array(col) / 255.0, label=surf)
            )
    legend_handles.append(
        mpatches.Patch(color=np.array([0, 255, 255]) / 255.0, label="PCA Axis")
    )
    legend_handles.append(
        mpatches.Patch(color=np.array([255, 0, 255]) / 255.0, label="Caries→Tooth")
    )
    legend_handles.append(
        mpatches.Patch(facecolor="white", edgecolor="black", label="Tooth bbox")
    )
    legend_handles.append(
        mpatches.Patch(color=np.array([255, 60, 60]) / 255.0, label="Caries bbox")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    # ── Save ─────────────────────────────────────────────────────────
    suffix = f"_tooth{filter_tooth}" if filter_tooth else ""
    out_name = f"debug_viz_case_{case_num}{suffix}_combined.png"
    out_path = WEEK9_DIR / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n  [Saved] Visualization -> {out_path}")
    print(f"{'=' * 60}")
    return out_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Week 9 — Debug Visualization: Compare PCA methods for a single case."
    )
    parser.add_argument(
        "--case_id", type=int, required=True,
        help="Case number to visualize (e.g. 311)."
    )
    parser.add_argument(
        "--tooth", type=str, default=None,
        help="Optional: filter to a single tooth_id (e.g. 25) for zoomed view."
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150)."
    )

    args = parser.parse_args()
    generate_comparison_figure(
        case_num=args.case_id,
        filter_tooth=args.tooth,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
