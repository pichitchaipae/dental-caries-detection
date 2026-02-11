"""
Multi-Zone Surface Classifier (M-C-D Logic)
=============================================

Advanced caries surface classification using a **Point Cloud Voting System**.

Instead of classifying based solely on the centroid of the caries lesion,
this module subdivides the Occlusal surface into three anatomical zones
along the PCA Long Axis:

    Mesio-occlusal (M)  |  Central (C)  |  Disto-occlusal (D)

Combined labels are generated when a lesion spans multiple zones:
    - MO  = Mesio-Occlusal (M + C)
    - DO  = Disto-Occlusal (D + C)
    - MOD = Mesio-Occlusal-Distal (all three zones)

Pipeline includes:
  1. Noise removal – connected-component filtering drops isolated
     pixel clusters < 15 px.
  2. **Dominant + Extension** labeling with a 5 % threshold (lowered
     from 20 % per G.V. Black "Any Involvement" standard).
  3. Debug image output (Before vs. After cleaning) when requested.

Also handles Proximal subdivisions (Mesial / Distal) based on quadrant
and relative X position.

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026
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
# Constants
# =============================================================================

OCCLUSAL_ZONE_THRESHOLD = 0.20   # top/bottom 20 % → occlusal zone
PROXIMAL_ZONE_THRESHOLD = 0.20   # left/right 20 % → proximal zone
DOMINANT_ZONE_MIN_FRAC  = 0.05   # ≥ 5 % of points to count as a zone (G.V. Black)

# Noise removal
MIN_CLUSTER_SIZE        = 15     # connected components < 15 px → removed
DEBUG_DIR               = Path(r"C:\Users\jaopi\Desktop\SP\week6\evaluation_output")


# =============================================================================
# Geometric helpers (same as week5 classifier, kept self-contained)
# =============================================================================

def _is_upper_jaw(tooth_id: str) -> bool:
    try:
        return int(tooth_id[0]) in [1, 2]
    except (ValueError, IndexError):
        return True


def _centroid(pts: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _pca_rotation_angle(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (mean, rotation_angle_rad) for PCA vertical alignment."""
    pts = pts.astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    _, eigvecs = cv2.PCACompute(centered, mean=None)
    major = eigvecs[0]
    angle_from_x = math.atan2(major[1], major[0])
    rot = math.pi / 2 - angle_from_x
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi
    return mean, rot


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
# Noise Removal: Connected-Component Filtering
# =============================================================================

def _remove_small_clusters(
    caries_pts: np.ndarray,
    min_cluster: int = MIN_CLUSTER_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove isolated pixel clusters smaller than *min_cluster* pixels.

    Creates a binary mask from the point cloud, runs
    ``cv2.connectedComponentsWithStats``, and drops components whose
    area is below the threshold.

    Returns
    -------
    (cleaned_pts, raw_pts) – cleaned array and the original for
                             before/after comparison.
    """
    if len(caries_pts) < min_cluster:
        # Too few points overall → keep them all (avoid deleting valid
        # small lesions entirely).
        return caries_pts, caries_pts

    pts = caries_pts.astype(np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    # Pad by 2 so border pixels aren't clipped
    pad = 2
    w = x_max - x_min + 1 + 2 * pad
    h = y_max - y_min + 1 + 2 * pad
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min - pad, y_min - pad])
    mask[shifted[:, 1], shifted[:, 0]] = 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Build a keep-mask: retain components with area ≥ min_cluster
    keep = np.zeros_like(mask, dtype=np.uint8)
    for lbl in range(1, n_labels):              # 0 = background
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
            keep[labels == lbl] = 255

    # Map back to points
    kept_ys, kept_xs = np.where(keep > 0)
    if len(kept_xs) == 0:
        # Nothing survived → fall back to raw (don't destroy all data)
        return caries_pts, caries_pts

    cleaned_pts = np.column_stack([
        kept_xs + x_min - pad,
        kept_ys + y_min - pad,
    ]).astype(np.float64)

    return cleaned_pts, caries_pts


# =============================================================================
# Debug Image: Before vs. After Cleaning
# =============================================================================

def _save_debug_image(
    raw_pts: np.ndarray,
    cleaned_pts: np.ndarray,
    tooth_id: str,
    case_num: int = 0,
    save_dir: Path = DEBUG_DIR,
):
    """
    Save a side-by-side debug image showing the caries mask
    BEFORE and AFTER connected-component noise removal.
    """
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
    axes[0].set_title(
        f"BEFORE  ({len(raw_pts)} pts)",
        fontsize=10, color="white", fontweight="bold",
    )
    axes[1].imshow(mask_cleaned, cmap="hot", aspect="auto")
    axes[1].set_title(
        f"AFTER  ({len(cleaned_pts)} pts, {removed} removed)",
        fontsize=10, color="white", fontweight="bold",
    )
    fig.suptitle(
        f"Noise Removal Debug — Tooth #{tooth_id}  "
        f"(min_cluster={MIN_CLUSTER_SIZE})",
        fontsize=11, color="white", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = save_dir / (f"case {case_num}" if case_num else "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_debug_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
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

    Pipeline:
      1. **Noise removal** – connected-component filtering removes
         isolated pixel clusters < MIN_CLUSTER_SIZE.
      2. **PCA alignment** – rotate tooth polygon to vertical.
      3. **Zone voting** – classify every *cleaned* caries point
         into Mesial / Central / Distal by relative X within the
         PCA-aligned bounding box.
      4. **Threshold = 5 %** (lowered from 20 % per G.V. Black
         "Any Involvement" standard).
      5. **Dominant + Extension labeling** – largest zone is the
         dominant surface; secondary zones passing the 5 % check
         are appended (e.g. *Occlusal (Dominant) + Mesial (Extension)
         → MO*).

    Parameters
    ----------
    caries_points : list of [x, y]
        Pixel coordinates of every caries pixel (from week3 data).
    pca_params : dict
        Must contain:
            tooth_id       : str   – FDI tooth ID
            tooth_polygon  : list  – tooth boundary polygon [[x,y], ...]
        Optional overrides:
            occlusal_threshold  : float (default 0.20)
            proximal_threshold  : float (default 0.20)
    dominant_threshold : float
        Minimum fraction of caries points in a zone for the zone to be
        included in the combined label (default 0.05).
    save_debug : bool
        If True, save a Before-vs-After noise removal debug image.
    case_num : int
        Used for naming the debug output folder.

    Returns
    -------
    dict – see code for full key list.
    """
    tooth_id = pca_params["tooth_id"]
    tooth_poly = np.array(pca_params["tooth_polygon"], dtype=np.float64)
    caries_pts = np.array(caries_points, dtype=np.float64)

    occ_thresh = pca_params.get("occlusal_threshold", OCCLUSAL_ZONE_THRESHOLD)
    prox_thresh = pca_params.get("proximal_threshold", PROXIMAL_ZONE_THRESHOLD)

    n_pts_raw = len(caries_pts)
    if n_pts_raw == 0:
        return _empty_result(tooth_id)

    # ── Step 1: Noise removal (connected-component filtering) ────────
    cleaned_pts, raw_pts = _remove_small_clusters(caries_pts)
    caries_pts = cleaned_pts          # use cleaned for all downstream
    n_pts = len(caries_pts)
    if n_pts == 0:
        return _empty_result(tooth_id)

    if save_debug:
        _save_debug_image(raw_pts, cleaned_pts, tooth_id, case_num)

    # ── PCA alignment ────────────────────────────────────────────────
    mean, rot_angle = _pca_rotation_angle(tooth_poly)
    M = _rotation_matrix(rot_angle, mean[0], mean[1])

    rot_tooth = _rotate(tooth_poly, M)
    rot_caries = _rotate(caries_pts, M)

    bx, by, bw, bh = _bbox(rot_tooth)
    if bw <= 0 or bh <= 0:
        return _empty_result(tooth_id)

    # ── Relative coords per point ────────────────────────────────────
    rel_xs = np.clip((rot_caries[:, 0] - bx) / bw, 0, 1)
    rel_ys = np.clip((rot_caries[:, 1] - by) / bh, 0, 1)

    upper = _is_upper_jaw(tooth_id)

    # ── Zone assignment per point ────────────────────────────────────
    #
    # Y-axis zones (occlusal is crown-side):
    #   Upper jaw → occlusal at bottom (rel_y ≥ 1 − occ_thresh)
    #   Lower jaw → occlusal at top    (rel_y ≤ occ_thresh)
    #
    # X-axis sub-zones within occlusal band → M / C / D
    #   Quadrant 1,4 (right side): Mesial = towards midline = LEFT  (rel_x < 1/3)
    #   Quadrant 2,3 (left side):  Mesial = towards midline = RIGHT (rel_x > 2/3)

    if upper:
        occ_mask = rel_ys >= (1.0 - occ_thresh)
    else:
        occ_mask = rel_ys <= occ_thresh

    prox_left_mask  = rel_xs <= prox_thresh
    prox_right_mask = rel_xs >= (1.0 - prox_thresh)

    # Everything that is NOT occlusal and NOT proximal → "Other" (body)
    other_mask = ~occ_mask & ~prox_left_mask & ~prox_right_mask

    # ── Sub-divide occlusal band into M / C / D  ─────────────────────
    # Divide the *width* of the tooth into three equal parts within the
    # occlusal zone.
    third = 1.0 / 3.0

    quadrant = int(tooth_id[0]) if tooth_id else 1

    # Determine which end is mesial based on FDI quadrant
    # In panoramic X-ray: Q1/Q4 right side patient → left of image
    if quadrant in [1, 4]:
        # Mesial = LEFT side of rotated bbox (rel_x < 1/3)
        m_mask = occ_mask & (rel_xs < third)
        c_mask = occ_mask & (rel_xs >= third) & (rel_xs <= 2 * third)
        d_mask = occ_mask & (rel_xs > 2 * third)
    else:
        # Mesial = RIGHT side of rotated bbox (rel_x > 2/3)
        d_mask = occ_mask & (rel_xs < third)
        c_mask = occ_mask & (rel_xs >= third) & (rel_xs <= 2 * third)
        m_mask = occ_mask & (rel_xs > 2 * third)

    zone_counts = {
        "M": int(m_mask.sum()),
        "C": int(c_mask.sum()),
        "D": int(d_mask.sum()),
        "Proximal_L": int(prox_left_mask.sum()),
        "Proximal_R": int(prox_right_mask.sum()),
        "Other": int(other_mask.sum()),
    }

    # ── Voting / dominant-zone logic ─────────────────────────────────
    total_occ = zone_counts["M"] + zone_counts["C"] + zone_counts["D"]
    total_prox = zone_counts["Proximal_L"] + zone_counts["Proximal_R"]

    # Decide primary surface by majority of points
    if total_occ >= total_prox and total_occ >= zone_counts["Other"]:
        primary_surface = "Occlusal"
    elif total_prox > total_occ and total_prox >= zone_counts["Other"]:
        primary_surface = "Proximal"
    else:
        primary_surface = "Other"

    # Zone fractions within occlusal band
    if total_occ > 0:
        zone_frac = {
            "M": zone_counts["M"] / total_occ,
            "C": zone_counts["C"] / total_occ,
            "D": zone_counts["D"] / total_occ,
        }
    else:
        zone_frac = {"M": 0.0, "C": 0.0, "D": 0.0}

    # Build combined label
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
            zone_label = "MOD"          # M+D without C is clinically MOD
        elif has_m:
            zone_label = "M"
        elif has_d:
            zone_label = "D"
        else:
            zone_label = "O"            # Only central
    elif primary_surface == "Proximal":
        # Determine Mesial vs Distal
        if quadrant in [1, 4]:
            if zone_counts["Proximal_L"] >= zone_counts["Proximal_R"]:
                zone_label = "Proximal-Mesial"
            else:
                zone_label = "Proximal-Distal"
        else:
            if zone_counts["Proximal_R"] >= zone_counts["Proximal_L"]:
                zone_label = "Proximal-Mesial"
            else:
                zone_label = "Proximal-Distal"
    else:
        zone_label = "Other"

    # ── All-Points X-Distribution Surface Classification ────────────
    #
    # Classify ALL *cleaned* caries points by X-position in PCA-aligned
    # bounding box.  Threshold lowered to 5 % (G.V. Black: "Any
    # Involvement" is clinically significant).  Uses Dominant +
    # Extension labeling.

    if quadrant in [1, 4]:
        all_m_mask = rel_xs < third               # Mesial = LEFT
        all_c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        all_d_mask = rel_xs > 2 * third           # Distal = RIGHT
    else:
        all_d_mask = rel_xs < third               # Distal = LEFT
        all_c_mask = (rel_xs >= third) & (rel_xs <= 2 * third)
        all_m_mask = rel_xs > 2 * third           # Mesial = RIGHT

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

    # ── Predicted surface from dominant zone ─────────────────────────
    if dominant_zone == "C":
        predicted_surface_fine = "Occlusal"
    elif dominant_zone == "M":
        predicted_surface_fine = "Mesial"
    else:
        predicted_surface_fine = "Distal"

    # ── Combined label: Dominant + Extensions (≥ 5 % threshold) ──────
    has_m_all = all_m_frac >= dominant_threshold    # 5 %
    has_c_all = all_c_frac >= dominant_threshold
    has_d_all = all_d_frac >= dominant_threshold

    if has_m_all and has_c_all and has_d_all:
        predicted_detail = "MOD"
    elif has_m_all and has_d_all:
        predicted_detail = "MOD"   # M+D without C is clinically MOD
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
    dominant_label = {
        "M": "Mesial", "C": "Occlusal", "D": "Distal"
    }[dominant_zone]
    extensions = []
    for z, name in [("M", "Mesial"), ("C", "Occlusal"), ("D", "Distal")]:
        if z != dominant_zone and frac_map[z] >= dominant_threshold:
            extensions.append(name)
    detail_explanation = f"{dominant_label} (Dominant)"
    if extensions:
        detail_explanation += " + " + ", ".join(
            f"{e} (Extension)" for e in extensions
        )
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
        "pca_bbox": {"x": round(bx, 2), "y": round(by, 2),
                     "w": round(bw, 2), "h": round(bh, 2)},
        # ── All-points surface classification (cleaned) ──
        "predicted_surface_fine": predicted_surface_fine,
        "predicted_detail": predicted_detail,
        "all_zone_fractions": all_zone_fractions,
        "detail_explanation": detail_explanation,
        # ── Noise removal stats ──────────────────────────
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
        "zone_points": {"M": 0, "C": 0, "D": 0,
                        "Proximal_L": 0, "Proximal_R": 0, "Other": 0},
        "caries_centroid": (0, 0),
        "caries_centroid_rot": (0, 0),
        "relative_position": {"rel_x": 0, "rel_y": 0},
        "rotation_angle_deg": 0,
        "pca_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        "predicted_surface_fine": "Unknown",
        "predicted_detail": "N/A",
        "all_zone_fractions": {"M": 0, "C": 0, "D": 0},
        "detail_explanation": "",
        "n_points_raw": 0,
        "n_points_cleaned": 0,
        "n_points_removed": 0,
    }


# =============================================================================
# Convenience: classify from week3 + week2 raw data
# =============================================================================

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
        pca_params={
            "tooth_id": tooth_id,
            "tooth_polygon": tooth_polygon,
        },
        dominant_threshold=dominant_threshold,
        save_debug=save_debug,
        case_num=case_num,
    )
