"""
Multi-Zone Surface Classifier — week7 (Fixed PCA Orientation)
==============================================================

Fixes applied vs. week6:
  **Task 1 – PCA Eigenvector Swap**
    The old ``_pca_rotation_angle()`` blindly chose ``eigenvectors[0]``
    as the long axis.  For square-ish molars (e.g. Tooth 48, case 311)
    this picks the *horizontal* eigenvector, producing a 90° rotation
    error.

    New 3-rule Orientation Logic:
      1. **Verticality Check** – always pick the eigenvector with the
         larger |Y| component as the vertical / long axis.
      2. **Occlusal / Apical direction** – upper teeth occlusal faces
         downward (+Y), lower teeth upward (−Y).  Flip the axis if it
         points the wrong way.
      3. **Mesial / Distal direction** – FDI quadrants 1/4 (right side)
         have mesial on the +X side; quadrants 2/3 on −X.  Choose the
         secondary eigenvector direction accordingly.

Everything else (noise removal, zone voting, dominant+extension labels)
is inherited unchanged from week6.

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026-02
"""

import cv2
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Constants (same as week6)
# =============================================================================

OCCLUSAL_ZONE_THRESHOLD = 0.20
PROXIMAL_ZONE_THRESHOLD = 0.20
DOMINANT_ZONE_MIN_FRAC  = 0.05   # 5 % – G.V. Black "Any Involvement"
MIN_CLUSTER_SIZE        = 15
DEBUG_DIR               = Path(r"C:\Users\jaopi\Desktop\SP\week7\evaluation_output")

# Rule 4 – Fallback Angle Clamp
# Malformed masks (crowns, large fillings) can produce extreme PCA tilts.
# If |angle| > MAX_TILT_DEG, clamp to 0° (straight vertical) as a safety net.
MAX_TILT_DEG = 45.0


# =============================================================================
# Geometric helpers
# =============================================================================

def _is_upper_jaw(tooth_id: str) -> bool:
    try:
        return int(tooth_id[0]) in [1, 2]
    except (ValueError, IndexError):
        return True


def _get_quadrant(tooth_id: str) -> int:
    """Return FDI quadrant number (1-4) from the first digit."""
    try:
        return int(tooth_id[0])
    except (ValueError, IndexError):
        return 1


def _centroid(pts: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


# ─────────────────────────────────────────────────────────────────────
# Task 1 FIX – PCA alignment with 3-rule Orientation Logic
# ─────────────────────────────────────────────────────────────────────

def _pca_rotation_angle_fixed(
    pts: np.ndarray,
    tooth_id: str,
) -> Tuple[np.ndarray, float]:
    """
    Compute PCA on *pts* and return ``(mean, rotation_angle_rad)``
    that rotates the tooth polygon to a canonical vertical orientation.

    **Three-rule orientation logic (Task 1 fix):**

    Rule 1 – Verticality Check (Long Axis)
        Compare ``|eigenvectors[0][1]|`` vs ``|eigenvectors[1][1]|``.
        Select the eigenvector with the **larger** absolute Y component
        as the vertical (long) axis.  This prevents the swap on
        square-ish molars where the first eigenvector happens to be
        horizontal.

    Rule 2 – Occlusal / Apical Direction
        Use the FDI ``tooth_id``:
          • Upper teeth (Q1, Q2): occlusal faces **down** → the
            vertical axis should point toward +Y.
          • Lower teeth (Q3, Q4): occlusal faces **up** → the
            vertical axis should point toward −Y.
        Flip the axis (negate) if it points the wrong way.

    Rule 3 – Mesial / Distal Direction
        Enforce the secondary (horizontal) eigenvector so that mesial
        is towards the midline:
          • Q1, Q4 (right side of patient): mesial is on the **+X**
            direction in the image.
          • Q2, Q3 (left side of patient): mesial is on the **−X**
            direction.
        Flip the secondary axis if needed.

    The rotation angle is computed from the corrected vertical axis.
    """
    pts = pts.astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # PCA via OpenCV
    _, eigvecs = cv2.PCACompute(centered, mean=None)
    ev0 = eigvecs[0]  # first eigenvector (largest variance)
    ev1 = eigvecs[1]  # second eigenvector

    # ── Rule 1: Verticality Check ────────────────────────────────────
    # The "long axis" of a tooth should be roughly vertical.
    # Pick whichever eigenvector has the larger |Y| component.
    if abs(ev0[1]) >= abs(ev1[1]):
        vertical_axis = ev0.copy()
        horizontal_axis = ev1.copy()
    else:
        # Swap: ev1 is more vertical than ev0 (square molar case)
        vertical_axis = ev1.copy()
        horizontal_axis = ev0.copy()

    # ── Rule 2: Occlusal / Apical Direction ──────────────────────────
    upper = _is_upper_jaw(tooth_id)
    if upper:
        # Upper teeth: occlusal side faces DOWN → vertical axis
        # should point toward +Y (downward in image coordinates).
        if vertical_axis[1] < 0:
            vertical_axis = -vertical_axis
    else:
        # Lower teeth: occlusal side faces UP → vertical axis
        # should point toward −Y (upward in image coordinates).
        if vertical_axis[1] > 0:
            vertical_axis = -vertical_axis

    # ── Rule 3: Mesial / Distal Direction ────────────────────────────
    quadrant = _get_quadrant(tooth_id)
    if quadrant in [1, 4]:
        # Right side of patient: mesial toward midline = +X in image
        if horizontal_axis[0] < 0:
            horizontal_axis = -horizontal_axis
    else:
        # Left side of patient: mesial toward midline = −X in image
        if horizontal_axis[0] > 0:
            horizontal_axis = -horizontal_axis

    # ── Compute rotation angle from the corrected vertical axis ──────
    # We want to rotate so that the vertical axis aligns with the Y-axis
    # (pointing downward = angle 0 from +Y).
    angle_from_x = math.atan2(vertical_axis[1], vertical_axis[0])
    rot = math.pi / 2 - angle_from_x

    # Normalise to [−π, π]
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi

    # ── Rule 4: Fallback Angle Clamp (GIGO safety net) ───────────────
    # Malformed / squashed tooth masks (e.g. crowns, large fillings)
    # can produce extreme PCA tilt angles (> 45°).  When this happens
    # the downstream zone classifier maps M/C/D to wrong anatomical
    # sides.  Clamping to 0° (straight vertical) is safer than trusting
    # a garbage eigenvector.
    was_clamped = False
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
        was_clamped = True

    return mean, rot, was_clamped


def _rotation_matrix(angle: float, cx: float, cy: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, cx - cx * c + cy * s],
        [s,  c, cy - cx * s - cy * c],
    ], dtype=np.float64)


def _rotate(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    return cv2.transform(pts.reshape(-1, 1, 2).astype(np.float64), M).reshape(-1, 2)


def _bbox(pts: np.ndarray):
    return (pts[:, 0].min(), pts[:, 1].min(),
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min())


# =============================================================================
# Noise Removal (unchanged from week6)
# =============================================================================

def _remove_small_clusters(
    caries_pts: np.ndarray,
    min_cluster: int = MIN_CLUSTER_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove isolated pixel clusters smaller than *min_cluster*."""
    if len(caries_pts) < min_cluster:
        return caries_pts, caries_pts

    pts = caries_pts.astype(np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = 2
    w = x_max - x_min + 1 + 2 * pad
    h = y_max - y_min + 1 + 2 * pad
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min - pad, y_min - pad])
    mask[shifted[:, 1], shifted[:, 0]] = 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    keep = np.zeros_like(mask, dtype=np.uint8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
            keep[labels == lbl] = 255

    kept_ys, kept_xs = np.where(keep > 0)
    if len(kept_xs) == 0:
        return caries_pts, caries_pts

    cleaned_pts = np.column_stack([
        kept_xs + x_min - pad,
        kept_ys + y_min - pad,
    ]).astype(np.float64)

    return cleaned_pts, caries_pts


# =============================================================================
# Debug Image
# =============================================================================

def _save_debug_image(
    raw_pts: np.ndarray,
    cleaned_pts: np.ndarray,
    tooth_id: str,
    case_num: int = 0,
    save_dir: Path = DEBUG_DIR,
):
    def _to_mask(pts, x_min, y_min, w, h):
        m = np.zeros((h, w), dtype=np.uint8)
        p = pts.astype(np.int32)
        xs = np.clip(p[:, 0] - x_min, 0, w - 1)
        ys = np.clip(p[:, 1] - y_min, 0, h - 1)
        m[ys, xs] = 255
        return m

    all_pts = np.vstack([raw_pts, cleaned_pts]) if len(cleaned_pts) > 0 else raw_pts
    x_min, y_min = all_pts.astype(np.int32).min(axis=0) - 2
    x_max, y_max = all_pts.astype(np.int32).max(axis=0) + 2
    w = max(x_max - x_min + 1, 1)
    h = max(y_max - y_min + 1, 1)

    mask_raw     = _to_mask(raw_pts, x_min, y_min, w, h)
    mask_cleaned = _to_mask(cleaned_pts, x_min, y_min, w, h)
    removed = len(raw_pts) - len(cleaned_pts)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(mask_raw, cmap="hot", aspect="auto")
    axes[0].set_title(f"BEFORE  ({len(raw_pts)} pts)", fontsize=10, color="white", fontweight="bold")
    axes[1].imshow(mask_cleaned, cmap="hot", aspect="auto")
    axes[1].set_title(f"AFTER  ({len(cleaned_pts)} pts, {removed} removed)", fontsize=10, color="white", fontweight="bold")
    fig.suptitle(
        f"Noise Removal Debug — Tooth #{tooth_id}  (min_cluster={MIN_CLUSTER_SIZE})",
        fontsize=11, color="white", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = save_dir / (f"case {case_num}" if case_num else "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_debug_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Public API
# =============================================================================

def classify_multi_zone(
    caries_points: List[List[float]],
    pca_params: Dict,
    dominant_threshold: float = DOMINANT_ZONE_MIN_FRAC,
    save_debug: bool = False,
    case_num: int = 0,
) -> Dict:
    """
    Classify a caries lesion into M-C-D zones using Point Cloud Voting.

    **week7 change:** Uses ``_pca_rotation_angle_fixed()`` which
    implements the 3-rule orientation logic instead of blindly using
    ``eigenvectors[0]``.
    """
    tooth_id = pca_params["tooth_id"]
    tooth_poly = np.array(pca_params["tooth_polygon"], dtype=np.float64)
    caries_pts = np.array(caries_points, dtype=np.float64)

    occ_thresh  = pca_params.get("occlusal_threshold", OCCLUSAL_ZONE_THRESHOLD)
    prox_thresh = pca_params.get("proximal_threshold", PROXIMAL_ZONE_THRESHOLD)

    n_pts_raw = len(caries_pts)
    if n_pts_raw == 0:
        return _empty_result(tooth_id)

    # ── Step 1: Noise removal ────────────────────────────────────────
    cleaned_pts, raw_pts = _remove_small_clusters(caries_pts)
    caries_pts = cleaned_pts
    n_pts = len(caries_pts)
    if n_pts == 0:
        return _empty_result(tooth_id)

    if save_debug:
        _save_debug_image(raw_pts, cleaned_pts, tooth_id, case_num)

    # ── Step 2: PCA alignment (FIXED – 4-rule orientation) ───────────
    mean, rot_angle, pca_was_clamped = _pca_rotation_angle_fixed(tooth_poly, tooth_id)
    M = _rotation_matrix(rot_angle, mean[0], mean[1])

    rot_tooth  = _rotate(tooth_poly, M)
    rot_caries = _rotate(caries_pts, M)

    bx, by, bw, bh = _bbox(rot_tooth)
    if bw <= 0 or bh <= 0:
        return _empty_result(tooth_id)

    # ── Relative coords per point ────────────────────────────────────
    rel_xs = np.clip((rot_caries[:, 0] - bx) / bw, 0, 1)
    rel_ys = np.clip((rot_caries[:, 1] - by) / bh, 0, 1)

    upper = _is_upper_jaw(tooth_id)

    # ── Zone assignment per point ────────────────────────────────────
    if upper:
        occ_mask = rel_ys >= (1.0 - occ_thresh)
    else:
        occ_mask = rel_ys <= occ_thresh

    prox_left_mask  = rel_xs <= prox_thresh
    prox_right_mask = rel_xs >= (1.0 - prox_thresh)
    other_mask = ~occ_mask & ~prox_left_mask & ~prox_right_mask

    # ── Sub-divide occlusal band into M / C / D ──────────────────────
    # FIX: M/D Flip — Mesial faces the midline.
    #   Q1/Q4 (patient-right): midline is to the RIGHT (+X) → high rel_xs = Mesial
    #   Q2/Q3 (patient-left):  midline is to the LEFT  (−X) → low  rel_xs = Mesial
    third = 1.0 / 3.0
    quadrant = _get_quadrant(tooth_id)

    if quadrant in [1, 4]:
        d_mask = occ_mask & (rel_xs < third)
        c_mask = occ_mask & (rel_xs >= third) & (rel_xs <= 2 * third)
        m_mask = occ_mask & (rel_xs > 2 * third)
    else:
        m_mask = occ_mask & (rel_xs < third)
        c_mask = occ_mask & (rel_xs >= third) & (rel_xs <= 2 * third)
        d_mask = occ_mask & (rel_xs > 2 * third)

    zone_counts = {
        "M": int(m_mask.sum()),
        "C": int(c_mask.sum()),
        "D": int(d_mask.sum()),
        "Proximal_L": int(prox_left_mask.sum()),
        "Proximal_R": int(prox_right_mask.sum()),
        "Other": int(other_mask.sum()),
    }

    # ── Voting / dominant-zone logic ─────────────────────────────────
    total_occ  = zone_counts["M"] + zone_counts["C"] + zone_counts["D"]
    total_prox = zone_counts["Proximal_L"] + zone_counts["Proximal_R"]

    if total_occ >= total_prox and total_occ >= zone_counts["Other"]:
        primary_surface = "Occlusal"
    elif total_prox > total_occ and total_prox >= zone_counts["Other"]:
        primary_surface = "Proximal"
    else:
        primary_surface = "Other"

    if total_occ > 0:
        zone_frac = {
            "M": zone_counts["M"] / total_occ,
            "C": zone_counts["C"] / total_occ,
            "D": zone_counts["D"] / total_occ,
        }
    else:
        zone_frac = {"M": 0.0, "C": 0.0, "D": 0.0}

    # Build combined zone label
    if primary_surface == "Occlusal":
        has_m = zone_frac["M"] >= dominant_threshold
        has_c = zone_frac["C"] >= dominant_threshold
        has_d = zone_frac["D"] >= dominant_threshold

        if has_m and has_c and has_d:
            zone_label = "MOD"
        elif has_m and has_c:
            zone_label = "MO"
        elif has_d and has_c:
            zone_label = "DO"
        elif has_m and has_d:
            zone_label = "MOD"
        elif has_m:
            zone_label = "M"
        elif has_d:
            zone_label = "D"
        else:
            zone_label = "O"
    elif primary_surface == "Proximal":
        # FIX: M/D Flip — Proximal_R (high rel_xs) = Mesial for Q1/Q4
        if quadrant in [1, 4]:
            zone_label = "Proximal-Mesial" if zone_counts["Proximal_R"] >= zone_counts["Proximal_L"] else "Proximal-Distal"
        else:
            zone_label = "Proximal-Mesial" if zone_counts["Proximal_L"] >= zone_counts["Proximal_R"] else "Proximal-Distal"
    else:
        zone_label = "Other"

    # ── All-Points X-Distribution Surface Classification ─────────────
    # FIX: M/D Flip — same logic as occlusal band
    if quadrant in [1, 4]:
        all_d_mask = rel_xs < third
        all_c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        all_m_mask = rel_xs > 2 * third
    else:
        all_m_mask = rel_xs < third
        all_c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        all_d_mask = rel_xs > 2 * third

    all_m_count = int(all_m_mask.sum())
    all_c_count = int(all_c_mask.sum())
    all_d_count = int(all_d_mask.sum())
    total_all   = max(all_m_count + all_c_count + all_d_count, 1)

    all_m_frac = all_m_count / total_all
    all_c_frac = all_c_count / total_all
    all_d_frac = all_d_count / total_all

    all_zone_fractions = {
        "M": round(all_m_frac, 4),
        "C": round(all_c_frac, 4),
        "D": round(all_d_frac, 4),
    }

    # ── Dominant Zone identification ─────────────────────────────────
    frac_map = {"M": all_m_frac, "C": all_c_frac, "D": all_d_frac}
    dominant_zone = max(frac_map, key=frac_map.get)

    if dominant_zone == "C":
        predicted_surface_fine = "Occlusal"
    elif dominant_zone == "M":
        predicted_surface_fine = "Mesial"
    else:
        predicted_surface_fine = "Distal"

    # ── Combined label ───────────────────────────────────────────────
    has_m_all = all_m_frac >= dominant_threshold
    has_c_all = all_c_frac >= dominant_threshold
    has_d_all = all_d_frac >= dominant_threshold

    if has_m_all and has_c_all and has_d_all:
        predicted_detail = "MOD"
    elif has_m_all and has_d_all:
        predicted_detail = "MOD"
    elif has_m_all and has_c_all:
        predicted_detail = "MO"
    elif has_d_all and has_c_all:
        predicted_detail = "DO"
    elif has_m_all:
        predicted_detail = "M"
    elif has_d_all:
        predicted_detail = "D"
    else:
        predicted_detail = "O"

    # ── Dominant + Extension annotation ──────────────────────────────
    dominant_label = {"M": "Mesial", "C": "Occlusal", "D": "Distal"}[dominant_zone]
    extensions = []
    for z, name in [("M", "Mesial"), ("C", "Occlusal"), ("D", "Distal")]:
        if z != dominant_zone and frac_map[z] >= dominant_threshold:
            extensions.append(name)
    detail_explanation = f"{dominant_label} (Dominant)"
    if extensions:
        detail_explanation += " + " + ", ".join(f"{e} (Extension)" for e in extensions)
        detail_explanation += f" → {predicted_detail}"

    # ── Aggregated metrics ───────────────────────────────────────────
    cx_orig, cy_orig = _centroid(caries_pts)
    cx_rot, cy_rot = _centroid(rot_caries)
    rel_x_cent = float(np.clip((cx_rot - bx) / bw, 0, 1))
    rel_y_cent = float(np.clip((cy_rot - by) / bh, 0, 1))

    return {
        "tooth_id": tooth_id,
        "primary_surface": primary_surface,
        "zone_label": zone_label,
        "zone_fractions": {k: round(v, 4) for k, v in zone_frac.items()},
        "n_points": n_pts,
        "zone_points": zone_counts,
        "caries_centroid": (round(cx_orig, 2), round(cy_orig, 2)),
        "caries_centroid_rot": (round(cx_rot, 2), round(cy_rot, 2)),
        "relative_position": {"rel_x": round(rel_x_cent, 4), "rel_y": round(rel_y_cent, 4)},
        "rotation_angle_deg": round(math.degrees(rot_angle), 2),
        "pca_clamped": pca_was_clamped,
        "pca_bbox": {"x": round(bx, 2), "y": round(by, 2), "w": round(bw, 2), "h": round(bh, 2)},
        "predicted_surface_fine": predicted_surface_fine,
        "predicted_detail": predicted_detail,
        "all_zone_fractions": all_zone_fractions,
        "detail_explanation": detail_explanation,
        "n_points_raw": n_pts_raw,
        "n_points_cleaned": n_pts,
        "n_points_removed": n_pts_raw - n_pts,
    }


def _empty_result(tooth_id: str) -> Dict:
    return {
        "tooth_id": tooth_id,
        "primary_surface": "Unknown",
        "zone_label": "N/A",
        "zone_fractions": {"M": 0, "C": 0, "D": 0},
        "n_points": 0,
        "zone_points": {"M": 0, "C": 0, "D": 0, "Proximal_L": 0, "Proximal_R": 0, "Other": 0},
        "caries_centroid": (0, 0),
        "caries_centroid_rot": (0, 0),
        "relative_position": {"rel_x": 0, "rel_y": 0},
        "rotation_angle_deg": 0,
        "pca_clamped": False,
        "pca_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        "predicted_surface_fine": "Unknown",
        "predicted_detail": "N/A",
        "all_zone_fractions": {"M": 0, "C": 0, "D": 0},
        "detail_explanation": "",
        "n_points_raw": 0,
        "n_points_cleaned": 0,
        "n_points_removed": 0,
    }


def classify_from_week_data(
    tooth_id: str,
    tooth_polygon: List[List[float]],
    caries_coordinates: List[List[float]],
    dominant_threshold: float = DOMINANT_ZONE_MIN_FRAC,
    save_debug: bool = False,
    case_num: int = 0,
) -> Dict:
    """Thin wrapper matching the interface used in week5."""
    return classify_multi_zone(
        caries_points=caries_coordinates,
        pca_params={"tooth_id": tooth_id, "tooth_polygon": tooth_polygon},
        dominant_threshold=dominant_threshold,
        save_debug=save_debug,
        case_num=case_num,
    )
