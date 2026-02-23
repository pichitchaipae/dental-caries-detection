# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Multi-Zone Surface Classifier — week8 (Multi-PCA Method Support)
=================================================================

Inherits all week7 fixes and adds **configurable PCA method selection**
from the week5 caries_surface_classifier.py evaluation pipeline.

Fixes inherited from week7:
  **Task 1 – PCA Eigenvector Swap**  (3-rule + angle clamp)

New in week8:
  **PCA Method Selection**
    Supports 5 PCA methods for evaluation:
      Method 1 – Square Heuristic (eigenvalue ratio check)
      Method 2 – Maximum Projected Span
      Method 3 – Split-Centroid (anatomical vector)
      Method 4 – (Placeholder – not implemented)
      Method 5 – Absolute Vertical Prior (largest |Y| component)
    Default is Method 5 (equivalent to week7's 3-rule logic).

    Use ``set_pca_method(N)`` or pass ``--pca-method N`` via CLI.

Everything else (noise removal, zone voting, dominant+extension labels)
is inherited unchanged from week7.

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026-02
"""

import cv2 # type: ignore
import math
import numpy as np # type: ignore
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore


# =============================================================================
# Constants (same as week7)
# =============================================================================

OCCLUSAL_ZONE_THRESHOLD = 0.20
PROXIMAL_ZONE_THRESHOLD = 0.20
DOMINANT_ZONE_MIN_FRAC  = 0.05   # 5 % – G.V. Black "Any Involvement"
MIN_CLUSTER_SIZE        = 15
DEBUG_DIR               = Path(r"C:\Users\jaopi\Desktop\SP\week8-Surface Classification v4\evaluation_output")

# Rule 4 – Fallback Angle Clamp
MAX_TILT_DEG = 45.0

# PCA Method Selection (default: 5 = week7 3-rule logic)
_PCA_METHOD = 5

PCA_METHOD_NAMES = {
    1: "method_1_square_heuristic",
    2: "method_2_max_span",
    3: "method_3_split_centroid",
    4: "method_4_placeholder",
    5: "method_5_vertical_prior",
}


def set_pca_method(method: int):
    """Set the global PCA method (1-5). Thread-unsafe."""
    global _PCA_METHOD
    if method not in (1, 2, 3, 4, 5):
        raise ValueError(f"Invalid PCA method: {method}. Must be 1-5.")
    _PCA_METHOD = method


def get_pca_method() -> int:
    """Return the currently active PCA method number."""
    return _PCA_METHOD


def get_pca_method_name() -> str:
    """Return the human-readable name of the current PCA method."""
    return PCA_METHOD_NAMES.get(_PCA_METHOD, f"method_{_PCA_METHOD}")


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
# PCA Method 1 – Square Heuristic (from week5)
# ─────────────────────────────────────────────────────────────────────

def _pca_method_1(pts: np.ndarray, tooth_id: str) -> Tuple[np.ndarray, float, bool]:
    """
    Square Heuristic: If eigenvalue ratio < 2.0 (square-ish tooth),
    pick the eigenvector with larger |Y| component as major axis.
    Otherwise use standard eigenvalue ordering.
    """
    pts = pts.astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    vec_0 = eigenvectors[:, 0]
    vec_1 = eigenvectors[:, 1]

    ratio = eigenvalues[1] / (eigenvalues[0] + 1e-6)
    is_square_like = ratio < 2.0

    if is_square_like:
        if abs(vec_0[1]) > abs(vec_1[1]):
            major_axis = vec_0
        else:
            major_axis = vec_1
    else:
        sort_indices = np.argsort(eigenvalues)[::-1]
        major_axis = eigenvectors[:, sort_indices[0]]

    rotation_angle = np.arctan2(major_axis[1], major_axis[0])

    # Apply week7 Rule 2 (occlusal direction) and Rule 4 (angle clamp)
    rot = math.pi / 2 - rotation_angle
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi

    was_clamped = False
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
        was_clamped = True

    return mean, rot, was_clamped


# ─────────────────────────────────────────────────────────────────────
# PCA Method 2 – Maximum Projected Span (from week5)
# ─────────────────────────────────────────────────────────────────────

def _pca_method_2(pts: np.ndarray, tooth_id: str) -> Tuple[np.ndarray, float, bool]:
    """
    Maximum Projected Span: Project all points onto each eigenvector,
    pick the one with the longest span as the major axis.
    """
    pts = pts.astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    vec_0 = eigenvectors[:, 0]
    vec_1 = eigenvectors[:, 1]

    proj_0 = np.dot(centered, vec_0)
    proj_1 = np.dot(centered, vec_1)

    span_0 = np.max(proj_0) - np.min(proj_0)
    span_1 = np.max(proj_1) - np.min(proj_1)

    if span_0 > span_1:
        major_axis = vec_0
    else:
        major_axis = vec_1

    rotation_angle = np.arctan2(major_axis[1], major_axis[0])

    rot = math.pi / 2 - rotation_angle
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi

    was_clamped = False
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
        was_clamped = True

    return mean, rot, was_clamped


# ─────────────────────────────────────────────────────────────────────
# PCA Method 3 – Split-Centroid (from week5)
# ─────────────────────────────────────────────────────────────────────

def _pca_method_3(pts: np.ndarray, tooth_id: str) -> Tuple[np.ndarray, float, bool]:
    """
    Split-Centroid: Split points into upper/lower halves, compute the
    anatomical vector between their centroids, pick the eigenvector
    most aligned with it.
    """
    pts = pts.astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    vec_0 = eigenvectors[:, 0]
    vec_1 = eigenvectors[:, 1]

    upper_half = pts[pts[:, 1] < mean[1]]
    lower_half = pts[pts[:, 1] >= mean[1]]

    if len(upper_half) < 2 or len(lower_half) < 2:
        major_axis = vec_0 if eigenvalues[0] > eigenvalues[1] else vec_1
    else:
        upper_centroid = np.mean(upper_half, axis=0)
        lower_centroid = np.mean(lower_half, axis=0)
        anat_vector = lower_centroid - upper_centroid
        norm = np.linalg.norm(anat_vector)
        if norm > 0:
            anat_vector = anat_vector / norm
        else:
            anat_vector = np.array([0.0, 1.0])

        dot_0 = abs(np.dot(vec_0, anat_vector))
        dot_1 = abs(np.dot(vec_1, anat_vector))

        if dot_0 > dot_1:
            major_axis = vec_0
        else:
            major_axis = vec_1

    rotation_angle = np.arctan2(major_axis[1], major_axis[0])

    rot = math.pi / 2 - rotation_angle
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi

    was_clamped = False
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
        was_clamped = True

    return mean, rot, was_clamped


# ─────────────────────────────────────────────────────────────────────
# PCA Method 5 – Vertical Prior + 3-Rule Logic (week7 default)
# ─────────────────────────────────────────────────────────────────────

def _pca_rotation_angle_fixed(
    pts: np.ndarray,
    tooth_id: str,
) -> Tuple[np.ndarray, float, bool]:
    """
    Compute PCA on *pts* and return ``(mean, rotation_angle_rad, was_clamped)``
    that rotates the tooth polygon to a canonical vertical orientation.

    **Three-rule orientation logic (Task 1 fix):**

    Rule 1 – Verticality Check (Long Axis)
        Compare ``|eigenvectors[0][1]|`` vs ``|eigenvectors[1][1]|``.
        Select the eigenvector with the **larger** absolute Y component
        as the vertical (long) axis.

    Rule 2 – Occlusal / Apical Direction
        Use the FDI ``tooth_id``:
          - Upper teeth (Q1, Q2): occlusal faces **down** → +Y.
          - Lower teeth (Q3, Q4): occlusal faces **up** → −Y.
        Flip the axis if it points the wrong way.

    Rule 3 – Mesial / Distal Direction
        Enforce the secondary (horizontal) eigenvector so that mesial
        is towards the midline.

    Rule 4 – Fallback Angle Clamp (> MAX_TILT_DEG → 0°)
    """
    pts = pts.astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # PCA via OpenCV
    _, eigvecs = cv2.PCACompute(centered, mean=None)
    ev0 = eigvecs[0]  # first eigenvector (largest variance)
    ev1 = eigvecs[1]  # second eigenvector

    # ── Rule 1: Verticality Check ────────────────────────────────────
    if abs(ev0[1]) >= abs(ev1[1]):
        vertical_axis = ev0.copy()
        horizontal_axis = ev1.copy()
    else:
        vertical_axis = ev1.copy()
        horizontal_axis = ev0.copy()

    # ── Rule 2: Occlusal / Apical Direction ──────────────────────────
    upper = _is_upper_jaw(tooth_id)
    if upper:
        if vertical_axis[1] < 0:
            vertical_axis = -vertical_axis
    else:
        if vertical_axis[1] > 0:
            vertical_axis = -vertical_axis

    # ── Rule 3: Mesial / Distal Direction ────────────────────────────
    quadrant = _get_quadrant(tooth_id)
    if quadrant in [1, 4]:
        if horizontal_axis[0] < 0:
            horizontal_axis = -horizontal_axis
    else:
        if horizontal_axis[0] > 0:
            horizontal_axis = -horizontal_axis

    # ── Compute rotation angle from the corrected vertical axis ──────
    angle_from_x = math.atan2(vertical_axis[1], vertical_axis[0])
    rot = math.pi / 2 - angle_from_x

    # Normalise to [−π, π]
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi

    # ── Rule 4: Fallback Angle Clamp ─────────────────────────────────
    was_clamped = False
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
        was_clamped = True

    return mean, rot, was_clamped


# ─────────────────────────────────────────────────────────────────────
# PCA Method Dispatcher
# ─────────────────────────────────────────────────────────────────────

def _pca_dispatch(pts: np.ndarray, tooth_id: str) -> Tuple[np.ndarray, float, bool]:
    """
    Dispatch to the selected PCA method.
    Returns (mean, rotation_angle_rad, was_clamped).
    """
    method = _PCA_METHOD
    if method == 1:
        return _pca_method_1(pts, tooth_id)
    elif method == 2:
        return _pca_method_2(pts, tooth_id)
    elif method == 3:
        return _pca_method_3(pts, tooth_id)
    elif method == 4:
        raise NotImplementedError("PCA method 4 is not implemented. Use 1, 2, 3, or 5.")
    elif method == 5:
        return _pca_rotation_angle_fixed(pts, tooth_id)
    else:
        raise ValueError(f"Invalid PCA method: {method}. Must be 1-5.")


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
# Noise Removal (unchanged from week7)
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

    out_dir = (save_dir / "cases" / f"case {case_num}") if case_num else (save_dir / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_debug_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Debug Visualization — Step 1: PCA Alignment
# =============================================================================

def _debug_step1_pca(
    tooth_poly: np.ndarray,
    caries_pts: np.ndarray,
    rot_tooth: np.ndarray,
    rot_caries: np.ndarray,
    mean: np.ndarray,
    rot_angle: float,
    pca_was_clamped: bool,
    tooth_id: str,
    case_num: int = 0,
    save_dir: Path = DEBUG_DIR,
):
    """
    Debug Panel — PCA Alignment
    ────────────────────────────
    Left   : Original tooth polygon + caries + PCA eigenvectors (arrows).
    Center : Rotation info – selected axis, angle, clamped status.
    Right  : Rotated tooth + caries after PCA alignment.
    """
    # Re-compute eigenvectors for visualization (cheap)
    pts64 = tooth_poly.astype(np.float64)
    centered = pts64 - np.mean(pts64, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    ev0 = eigenvectors[:, 0]  # largest variance
    ev1 = eigenvectors[:, 1]  # smallest variance

    cx, cy = float(mean[0]), float(mean[1])

    # Arrow scale: 40% of tooth bounding box diagonal
    diag = np.sqrt((tooth_poly[:, 0].max() - tooth_poly[:, 0].min()) ** 2 +
                   (tooth_poly[:, 1].max() - tooth_poly[:, 1].min()) ** 2)
    arrow_len = diag * 0.4

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("gray")

    # ── Left: Original + Eigenvectors ────────────────────────────────
    ax = axes[0]
    poly_closed = np.vstack([tooth_poly, tooth_poly[0:1]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="#4fc3f7", linewidth=1.5, label="Tooth polygon")
    ax.fill(tooth_poly[:, 0], tooth_poly[:, 1], color="#4fc3f7", alpha=0.08)
    if len(caries_pts) > 0:
        ax.scatter(caries_pts[:, 0], caries_pts[:, 1], c="#ff5252", s=2, alpha=0.6, label=f"Caries ({len(caries_pts)} pts)")
    # Eigenvector arrows
    ax.annotate("", xy=(cx + ev0[0] * arrow_len, cy + ev0[1] * arrow_len),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#ffeb3b", lw=2.5))
    ax.annotate("", xy=(cx + ev1[0] * arrow_len, cy + ev1[1] * arrow_len),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#69f0ae", lw=2.0))
    ax.plot(cx, cy, "wo", markersize=6, zorder=5)
    # eigenvalue labels
    ax.text(cx + ev0[0] * arrow_len * 1.12, cy + ev0[1] * arrow_len * 1.12,
            f"EV0 (λ={eigenvalues[0]:.0f})", color="#ffeb3b", fontsize=8, fontweight="bold")
    ax.text(cx + ev1[0] * arrow_len * 1.12, cy + ev1[1] * arrow_len * 1.12,
            f"EV1 (λ={eigenvalues[1]:.0f})", color="#69f0ae", fontsize=8, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
    ax.set_title("Step 1a: Original + PCA Eigenvectors", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    # ── Center: Text info panel ──────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    info_lines = [
        f"Tooth #{tooth_id}",
        f"FDI Quadrant: Q{_get_quadrant(tooth_id)}  ({'Upper' if _is_upper_jaw(tooth_id) else 'Lower'} Jaw)",
        "",
        f"PCA Method: {_PCA_METHOD} — {get_pca_method_name()}",
        "",
        f"Eigenvalue 0: {eigenvalues[0]:.1f}",
        f"Eigenvalue 1: {eigenvalues[1]:.1f}",
        f"Ratio (λ0/λ1): {eigenvalues[0] / (eigenvalues[1] + 1e-6):.2f}",
        "",
        f"Rotation Angle: {math.degrees(rot_angle):.2f}°",
        f"Angle Clamped: {'YES (→ 0°)' if pca_was_clamped else 'No'}",
        f"Max Tilt Limit: ±{MAX_TILT_DEG}°",
        "",
        f"Centroid: ({cx:.1f}, {cy:.1f})",
    ]
    for i, line in enumerate(info_lines):
        y_pos = 0.92 - i * 0.065
        color = "#ff5252" if "Clamped: YES" in line else "#e0e0e0"
        weight = "bold" if i == 0 or "PCA Method" in line else "normal"
        fontsize = 11 if i == 0 else 9
        ax.text(0.08, y_pos, line, color=color, fontsize=fontsize, fontweight=weight,
                family="monospace", transform=ax.transAxes)
    ax.set_title("Step 1b: PCA Method Info", fontsize=10, color="white", fontweight="bold")

    # ── Right: Rotated tooth + caries ────────────────────────────────
    ax = axes[2]
    rp_closed = np.vstack([rot_tooth, rot_tooth[0:1]])
    ax.plot(rp_closed[:, 0], rp_closed[:, 1], color="#4fc3f7", linewidth=1.5, label="Rotated tooth")
    ax.fill(rot_tooth[:, 0], rot_tooth[:, 1], color="#4fc3f7", alpha=0.08)
    if len(rot_caries) > 0:
        ax.scatter(rot_caries[:, 0], rot_caries[:, 1], c="#ff5252", s=2, alpha=0.6, label=f"Rotated caries")
    # Draw vertical reference line through rotated centroid
    rcx = np.mean(rot_tooth[:, 0])
    rcy_min, rcy_max = rot_tooth[:, 1].min(), rot_tooth[:, 1].max()
    ax.plot([rcx, rcx], [rcy_min - 10, rcy_max + 10], "--", color="#ffeb3b", alpha=0.5, label="Vertical ref")
    ax.legend(loc="lower left", fontsize=7, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
    ax.set_title("Step 1c: After PCA Rotation", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    fig.suptitle(
        f"DEBUG Step 1 — PCA Alignment — Tooth #{tooth_id}  |  Method {_PCA_METHOD}",
        fontsize=12, color="white", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = (save_dir / "cases" / f"case {case_num}") if case_num else (save_dir / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_step1_pca_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Debug Visualization — Step 2: Multi-Zone Assignment
# =============================================================================

def _debug_step2_zone_map(
    rot_tooth: np.ndarray,
    rot_caries: np.ndarray,
    rel_xs: np.ndarray,
    rel_ys: np.ndarray,
    m_mask: np.ndarray,
    c_mask: np.ndarray,
    d_mask: np.ndarray,
    occ_mask: np.ndarray,
    prox_left_mask: np.ndarray,
    prox_right_mask: np.ndarray,
    other_mask: np.ndarray,
    bx: float, by: float, bw: float, bh: float,
    occ_thresh: float,
    prox_thresh: float,
    tooth_id: str,
    upper: bool,
    quadrant: int,
    case_num: int = 0,
    save_dir: Path = DEBUG_DIR,
):
    """
    Debug Panel — Multi-Zone Surface Assignment
    ─────────────────────────────────────────────
    Left  : Rotated tooth with zone boundary lines + caries colored by zone.
    Right : Normalized [0,1]x[0,1] view with zone regions shaded.
    """
    ZONE_COLORS = {
        "M": "#42a5f5",       # blue
        "C": "#66bb6a",       # green
        "D": "#ef5350",       # red
        "Prox_L": "#ffa726",  # orange
        "Prox_R": "#ffee58",  # yellow
        "Other": "#78909c",   # gray
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("gray")

    # ── Left: Rotated space with zone boundaries ─────────────────────
    ax = axes[0]
    rp_closed = np.vstack([rot_tooth, rot_tooth[0:1]])
    ax.plot(rp_closed[:, 0], rp_closed[:, 1], color="#4fc3f7", linewidth=1.2, alpha=0.6)

    # Draw bounding box
    rect_xs = [bx, bx + bw, bx + bw, bx, bx]
    rect_ys = [by, by, by + bh, by + bh, by]
    ax.plot(rect_xs, rect_ys, "--", color="#b0bec5", linewidth=1.0, alpha=0.5, label="BBox")

    # Draw zone boundary lines
    third = 1.0 / 3.0
    x_third1 = bx + bw * third
    x_third2 = bx + bw * 2 * third
    ax.plot([x_third1, x_third1], [by, by + bh], ":", color="#ffffff", alpha=0.4, linewidth=0.8)
    ax.plot([x_third2, x_third2], [by, by + bh], ":", color="#ffffff", alpha=0.4, linewidth=0.8)

    # Occlusal band line
    if upper:
        occ_line_y = by + bh * (1.0 - occ_thresh)
        ax.axhline(y=occ_line_y, color="#66bb6a", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(bx + bw + 5, occ_line_y, "Occlusal band ↓", color="#66bb6a", fontsize=7, va="center")
    else:
        occ_line_y = by + bh * occ_thresh
        ax.axhline(y=occ_line_y, color="#66bb6a", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(bx + bw + 5, occ_line_y, "Occlusal band ↑", color="#66bb6a", fontsize=7, va="center")

    # Proximal band lines
    prox_l_x = bx + bw * prox_thresh
    prox_r_x = bx + bw * (1.0 - prox_thresh)
    ax.axvline(x=prox_l_x, color="#ffa726", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(x=prox_r_x, color="#ffee58", linestyle="--", alpha=0.5, linewidth=0.8)

    # Color-coded caries points
    if len(rot_caries) > 0:
        masks_colors = [
            (m_mask, ZONE_COLORS["M"], "Mesial"),
            (c_mask, ZONE_COLORS["C"], "Central/Occlusal"),
            (d_mask, ZONE_COLORS["D"], "Distal"),
            (prox_left_mask, ZONE_COLORS["Prox_L"], "Proximal L"),
            (prox_right_mask, ZONE_COLORS["Prox_R"], "Proximal R"),
            (other_mask, ZONE_COLORS["Other"], "Other"),
        ]
        for mask, color, label in masks_colors:
            pts_in = rot_caries[mask]
            if len(pts_in) > 0:
                ax.scatter(pts_in[:, 0], pts_in[:, 1], c=color, s=3, alpha=0.7, label=f"{label} ({len(pts_in)})")

    ax.legend(loc="lower left", fontsize=6, facecolor="#1a1a2e", edgecolor="gray",
              labelcolor="white", ncol=2, markerscale=2)
    ax.set_title("Step 2a: Zone Assignment (Rotated Space)", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    # ── Right: Normalized [0,1] x [0,1] view ────────────────────────
    ax = axes[1]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Shade zone regions
    # Occlusal band
    if upper:
        ax.axhspan(1.0 - occ_thresh, 1.0, color="#66bb6a", alpha=0.10, label="Occlusal zone")
    else:
        ax.axhspan(0.0, occ_thresh, color="#66bb6a", alpha=0.10, label="Occlusal zone")

    # Proximal bands
    ax.axvspan(0.0, prox_thresh, color="#ffa726", alpha=0.10, label="Proximal L")
    ax.axvspan(1.0 - prox_thresh, 1.0, color="#ffee58", alpha=0.10, label="Proximal R")

    # M/C/D third lines
    ax.axvline(x=third, color="#ffffff", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.axvline(x=2 * third, color="#ffffff", linestyle=":", alpha=0.3, linewidth=0.8)

    # Zone labels in normalized space
    if quadrant in [1, 4]:
        ax.text(third / 2,       0.5, "D", color="#ef5350", fontsize=18, ha="center", va="center", alpha=0.3, fontweight="bold")
        ax.text(0.5,             0.5, "C", color="#66bb6a", fontsize=18, ha="center", va="center", alpha=0.3, fontweight="bold")
        ax.text(1.0 - third / 2, 0.5, "M", color="#42a5f5", fontsize=18, ha="center", va="center", alpha=0.3, fontweight="bold")
    else:
        ax.text(third / 2,       0.5, "M", color="#42a5f5", fontsize=18, ha="center", va="center", alpha=0.3, fontweight="bold")
        ax.text(0.5,             0.5, "C", color="#66bb6a", fontsize=18, ha="center", va="center", alpha=0.3, fontweight="bold")
        ax.text(1.0 - third / 2, 0.5, "D", color="#ef5350", fontsize=18, ha="center", va="center", alpha=0.3, fontweight="bold")

    # Scatter normalized caries
    if len(rel_xs) > 0:
        masks_colors_norm = [
            (m_mask, ZONE_COLORS["M"], "M"),
            (c_mask, ZONE_COLORS["C"], "C"),
            (d_mask, ZONE_COLORS["D"], "D"),
            (prox_left_mask, ZONE_COLORS["Prox_L"], "PL"),
            (prox_right_mask, ZONE_COLORS["Prox_R"], "PR"),
            (other_mask, ZONE_COLORS["Other"], "Oth"),
        ]
        for mask, color, label in masks_colors_norm:
            rxs = rel_xs[mask]
            rys = rel_ys[mask]
            if len(rxs) > 0:
                ax.scatter(rxs, rys, c=color, s=3, alpha=0.7, label=f"{label} ({len(rxs)})")

    ax.legend(loc="lower left", fontsize=6, facecolor="#1a1a2e", edgecolor="gray",
              labelcolor="white", ncol=3, markerscale=2)
    ax.set_xlabel("Relative X (0=left, 1=right)", color="gray", fontsize=8)
    ax.set_ylabel("Relative Y (0=top, 1=bottom)", color="gray", fontsize=8)
    ax.set_title("Step 2b: Normalized Zone Map [0,1]²", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    fig.suptitle(
        f"DEBUG Step 2 — Multi-Zone Assignment — Tooth #{tooth_id}  |  Q{quadrant} {'Upper' if upper else 'Lower'}",
        fontsize=12, color="white", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = (save_dir / "cases" / f"case {case_num}") if case_num else (save_dir / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_step2_zones_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Debug Visualization — Step 3: Point Cloud Voting
# =============================================================================

def _debug_step3_voting(
    zone_counts: Dict,
    zone_frac: Dict,
    all_zone_fractions: Dict,
    total_occ: int,
    total_prox: int,
    _internal_surface: str,
    zone_label: str,
    dominant_threshold: float,
    tooth_id: str,
    quadrant: int,
    case_num: int = 0,
    save_dir: Path = DEBUG_DIR,
):
    """
    Debug Panel — Point Cloud Voting
    ──────────────────────────────────
    Left  : Zone point-count bar chart (all 6 zones).
    Center: Pie chart of M/C/D fractions (occlusal band only).
    Right : All-Points M/C/D distribution + dominant zone highlight.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")

    ZONE_COLORS_BAR = {
        "M": "#42a5f5", "C": "#66bb6a", "D": "#ef5350",
        "Proximal_L": "#ffa726", "Proximal_R": "#ffee58", "Other": "#78909c"
    }

    # ── Left: Zone counts bar chart ──────────────────────────────────
    ax = axes[0]
    labels_bar = list(zone_counts.keys())
    values_bar = list(zone_counts.values())
    colors_bar = [ZONE_COLORS_BAR.get(k, "#78909c") for k in labels_bar]
    bars = ax.bar(labels_bar, values_bar, color=colors_bar, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values_bar):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values_bar) * 0.02,
                    str(val), ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax.set_ylabel("Point Count", color="gray", fontsize=9)
    ax.set_title("Step 3a: Zone Point Counts", fontsize=10, color="white", fontweight="bold")

    # Annotate surface decision
    decision_text = f"Occ={total_occ}  Prox={total_prox}  Other={zone_counts.get('Other', 0)}"
    ax.text(0.5, 0.95, decision_text, transform=ax.transAxes, ha="center", fontsize=8,
            color="#b0bec5", style="italic")
    ax.text(0.5, 0.88, f"→ Internal Surface: {_internal_surface}", transform=ax.transAxes,
            ha="center", fontsize=9, color="#ffeb3b", fontweight="bold")

    # ── Center: Pie chart of occlusal M/C/D ──────────────────────────
    ax = axes[1]
    pie_vals = [zone_frac.get("M", 0), zone_frac.get("C", 0), zone_frac.get("D", 0)]
    pie_labels = ["M", "C", "D"]
    pie_colors = ["#42a5f5", "#66bb6a", "#ef5350"]

    if sum(pie_vals) > 0:
        wedges, texts, autotexts = ax.pie(
            pie_vals, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%",
            startangle=90, textprops={"color": "white", "fontsize": 9},
            wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")
    else:
        ax.text(0.5, 0.5, "No Occlusal\nPoints", transform=ax.transAxes,
                ha="center", va="center", color="#78909c", fontsize=14)

    ax.set_title("Step 3b: Occlusal M/C/D Fractions", fontsize=10, color="white", fontweight="bold")
    ax.text(0.5, -0.05, f"Zone Label (occlusal): {zone_label}", transform=ax.transAxes,
            ha="center", fontsize=9, color="#ffeb3b", fontweight="bold")

    # ── Right: All-Points M/C/D bar + threshold line ─────────────────
    ax = axes[2]
    all_labels = ["Mesial (M)", "Central (C)", "Distal (D)"]
    all_vals = [all_zone_fractions.get("M", 0), all_zone_fractions.get("C", 0), all_zone_fractions.get("D", 0)]
    all_colors = ["#42a5f5", "#66bb6a", "#ef5350"]

    # Highlight dominant
    dominant_idx = all_vals.index(max(all_vals))
    edge_colors = ["white"] * 3
    edge_widths = [0.5] * 3
    edge_colors[dominant_idx] = "#ffeb3b"
    edge_widths[dominant_idx] = 2.5

    bars2 = ax.bar(all_labels, all_vals, color=all_colors, edgecolor=edge_colors, linewidth=edge_widths)
    for bar, val in zip(bars2, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

    # Threshold line
    ax.axhline(y=dominant_threshold, color="#ff5252", linestyle="--", alpha=0.7, linewidth=1.0)
    ax.text(0.98, dominant_threshold + 0.01, f"Threshold={dominant_threshold:.0%}",
            transform=ax.get_yaxis_transform(), ha="right", color="#ff5252", fontsize=8)

    ax.set_ylabel("Fraction (All Points)", color="gray", fontsize=9)
    ax.set_ylim(0, max(max(all_vals) * 1.2, 0.1))
    ax.set_title("Step 3c: All-Points M/C/D Distribution", fontsize=10, color="white", fontweight="bold")

    # Dominant zone annotation
    dominant_names = {0: "Mesial", 1: "Occlusal", 2: "Distal"}
    ax.text(0.5, 0.95, f"★ Dominant Zone: {dominant_names[dominant_idx]}",
            transform=ax.transAxes, ha="center", fontsize=10, color="#ffeb3b", fontweight="bold")

    fig.suptitle(
        f"DEBUG Step 3 — Point Cloud Voting — Tooth #{tooth_id}  |  Q{quadrant}",
        fontsize=12, color="white", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = (save_dir / "cases" / f"case {case_num}") if case_num else (save_dir / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_step3_voting_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Debug Visualization — Step 4: Final Classification Summary
# =============================================================================

def _debug_step4_summary(
    result: Dict,
    tooth_poly: np.ndarray,
    caries_pts: np.ndarray,
    rot_tooth: np.ndarray,
    rot_caries: np.ndarray,
    rel_xs: np.ndarray,
    rel_ys: np.ndarray,
    all_m_mask: np.ndarray,
    all_c_mask: np.ndarray,
    all_d_mask: np.ndarray,
    bx: float, by: float, bw: float, bh: float,
    tooth_id: str,
    case_num: int = 0,
    save_dir: Path = DEBUG_DIR,
):
    """
    Debug Panel — Final Summary (2×2 grid)
    ────────────────────────────────────────
    Top-Left     : Original tooth + caries overlay.
    Top-Right    : Rotated + M/C/D colored caries.
    Bottom-Left  : All-points heatmap in normalized space.
    Bottom-Right : Text summary of all classification results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#1a1a2e")
    for row in axes:
        for ax in row:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="gray", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("gray")

    ZONE_COLORS = {"M": "#42a5f5", "C": "#66bb6a", "D": "#ef5350"}

    # ── Top-Left: Original view ──────────────────────────────────────
    ax = axes[0][0]
    poly_closed = np.vstack([tooth_poly, tooth_poly[0:1]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="#4fc3f7", linewidth=1.5)
    ax.fill(tooth_poly[:, 0], tooth_poly[:, 1], color="#4fc3f7", alpha=0.08)
    if len(caries_pts) > 0:
        ax.scatter(caries_pts[:, 0], caries_pts[:, 1], c="#ff5252", s=2, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title("Original View", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    # ── Top-Right: Rotated + M/C/D colored ──────────────────────────
    ax = axes[0][1]
    rp_closed = np.vstack([rot_tooth, rot_tooth[0:1]])
    ax.plot(rp_closed[:, 0], rp_closed[:, 1], color="#4fc3f7", linewidth=1.2, alpha=0.6)

    # Zone boundary lines
    third = 1.0 / 3.0
    x_t1 = bx + bw * third
    x_t2 = bx + bw * 2 * third
    ax.plot([x_t1, x_t1], [by, by + bh], ":", color="#ffffff", alpha=0.3)
    ax.plot([x_t2, x_t2], [by, by + bh], ":", color="#ffffff", alpha=0.3)

    if len(rot_caries) > 0:
        for mask, color, label in [(all_m_mask, ZONE_COLORS["M"], "M"),
                                    (all_c_mask, ZONE_COLORS["C"], "C"),
                                    (all_d_mask, ZONE_COLORS["D"], "D")]:
            pts_in = rot_caries[mask]
            if len(pts_in) > 0:
                ax.scatter(pts_in[:, 0], pts_in[:, 1], c=color, s=3, alpha=0.6, label=label)
    ax.legend(loc="lower left", fontsize=7, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
    ax.set_aspect("equal")
    ax.set_title("Rotated + M/C/D Zones", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    # ── Bottom-Left: Heatmap in normalized space ─────────────────────
    ax = axes[1][0]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if len(rel_xs) > 0:
        # color by zone
        colors_arr = np.full(len(rel_xs), "#78909c")
        colors_arr[all_m_mask] = ZONE_COLORS["M"]
        colors_arr[all_c_mask] = ZONE_COLORS["C"]
        colors_arr[all_d_mask] = ZONE_COLORS["D"]
        ax.scatter(rel_xs, rel_ys, c=colors_arr, s=4, alpha=0.6)
    # Zone labels
    quadrant = _get_quadrant(tooth_id)
    if quadrant in [1, 4]:
        zone_order = ["D", "C", "M"]
    else:
        zone_order = ["M", "C", "D"]
    for i, z in enumerate(zone_order):
        x_pos = (i + 0.5) * third
        ax.text(x_pos, 0.5, z, color=ZONE_COLORS[z], fontsize=22, ha="center", va="center",
                alpha=0.2, fontweight="bold")
    ax.axvline(x=third, color="#ffffff", linestyle=":", alpha=0.3)
    ax.axvline(x=2 * third, color="#ffffff", linestyle=":", alpha=0.3)
    ax.set_xlabel("Relative X", color="gray", fontsize=8)
    ax.set_ylabel("Relative Y", color="gray", fontsize=8)
    ax.set_title("Normalized Point Cloud", fontsize=10, color="white", fontweight="bold")
    ax.invert_yaxis()

    # ── Bottom-Right: Text summary ───────────────────────────────────
    ax = axes[1][1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    summary_lines = [
        ("CLASSIFICATION RESULT", "#ffeb3b", 13, "bold"),
        ("", "#e0e0e0", 9, "normal"),
        (f"Tooth #{result['tooth_id']}  |  Q{quadrant} {'Upper' if _is_upper_jaw(tooth_id) else 'Lower'}", "#e0e0e0", 10, "bold"),
        ("", "#e0e0e0", 9, "normal"),
        (f"Primary Surface:     {result['primary_surface']}", "#4fc3f7", 10, "bold"),
        (f"Predicted Fine:      {result['predicted_surface_fine']}", "#4fc3f7", 10, "bold"),
        (f"Predicted Detail:    {result['predicted_detail']}", "#ffeb3b", 11, "bold"),
        ("", "#e0e0e0", 9, "normal"),
        (f"Zone Label:          {result['zone_label']}", "#e0e0e0", 10, "normal"),
        (f"Detail Explanation:  {result['detail_explanation']}", "#e0e0e0", 9, "normal"),
        ("", "#e0e0e0", 9, "normal"),
        (f"PCA Method:          {result['pca_method']} ({result['pca_method_name']})", "#b0bec5", 9, "normal"),
        (f"Rotation Angle:      {result['rotation_angle_deg']}°", "#b0bec5", 9, "normal"),
        (f"PCA Clamped:         {result['pca_clamped']}", "#ff5252" if result['pca_clamped'] else "#b0bec5", 9, "normal"),
        ("", "#e0e0e0", 9, "normal"),
        (f"Points (raw/clean):  {result['n_points_raw']} / {result['n_points_cleaned']}  ({result['n_points_removed']} removed)", "#b0bec5", 9, "normal"),
        (f"All-Zone Fractions:  M={result['all_zone_fractions']['M']:.1%}  C={result['all_zone_fractions']['C']:.1%}  D={result['all_zone_fractions']['D']:.1%}", "#b0bec5", 9, "normal"),
    ]
    for i, (text, color, fontsize, weight) in enumerate(summary_lines):
        y_pos = 0.95 - i * 0.055
        ax.text(0.05, y_pos, text, color=color, fontsize=fontsize, fontweight=weight,
                family="monospace", transform=ax.transAxes)
    ax.set_title("Classification Summary", fontsize=10, color="white", fontweight="bold")

    fig.suptitle(
        f"DEBUG Step 4 — Final Summary — Tooth #{tooth_id}  |  Case {case_num}",
        fontsize=13, color="white", fontweight="bold",
    )
    plt.tight_layout()

    out_dir = (save_dir / "cases" / f"case {case_num}") if case_num else (save_dir / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_step4_summary_tooth_{tooth_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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

    **week8 change:** Uses ``_pca_dispatch()`` which routes to the
    selected PCA method (1-5).  Default is Method 5 (week7's 3-rule logic).
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

    # ── Step 2: PCA alignment (dispatched to selected method) ────────
    mean, rot_angle, pca_was_clamped = _pca_dispatch(tooth_poly, tooth_id)
    M = _rotation_matrix(rot_angle, mean[0], mean[1])

    rot_tooth  = _rotate(tooth_poly, M)
    rot_caries = _rotate(caries_pts, M)

    bx, by, bw, bh = _bbox(rot_tooth)
    if bw <= 0 or bh <= 0:
        return _empty_result(tooth_id)

    # ── DEBUG: Step 1 — PCA Alignment ────────────────────────────────
    if save_debug:
        _debug_step1_pca(
            tooth_poly, caries_pts, rot_tooth, rot_caries,
            mean, rot_angle, pca_was_clamped,
            tooth_id, case_num,
        )

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

    # ── DEBUG: Step 2 — Zone Assignment ──────────────────────────────
    if save_debug:
        _debug_step2_zone_map(
            rot_tooth, rot_caries, rel_xs, rel_ys,
            m_mask, c_mask, d_mask, occ_mask,
            prox_left_mask, prox_right_mask, other_mask,
            bx, by, bw, bh,
            occ_thresh, prox_thresh,
            tooth_id, upper, quadrant, case_num,
        )

    # ── Voting / dominant-zone logic ─────────────────────────────────
    total_occ  = zone_counts["M"] + zone_counts["C"] + zone_counts["D"]
    total_prox = zone_counts["Proximal_L"] + zone_counts["Proximal_R"]

    if total_occ >= total_prox and total_occ >= zone_counts["Other"]:
        _internal_surface = "Occlusal"
    elif total_prox > total_occ and total_prox >= zone_counts["Other"]:
        _internal_surface = "Proximal"
    else:
        _internal_surface = "Other"

    if total_occ > 0:
        zone_frac = {
            "M": zone_counts["M"] / total_occ,
            "C": zone_counts["C"] / total_occ,
            "D": zone_counts["D"] / total_occ,
        }
    else:
        zone_frac = {"M": 0.0, "C": 0.0, "D": 0.0}

    # Build combined zone label
    if _internal_surface == "Occlusal":
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
    elif _internal_surface == "Proximal":
        if quadrant in [1, 4]:
            zone_label = "M" if zone_counts["Proximal_R"] >= zone_counts["Proximal_L"] else "D"
        else:
            zone_label = "M" if zone_counts["Proximal_L"] >= zone_counts["Proximal_R"] else "D"
    else:
        zone_label = "_DEFERRED"

    # ── All-Points X-Distribution Surface Classification ─────────────
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

    # ── Resolve deferred zone_label for "Other" internal surface ─────
    if zone_label == "_DEFERRED":
        _fallback = max({"M": all_m_frac, "C": all_c_frac, "D": all_d_frac},
                        key=lambda k: {"M": all_m_frac, "C": all_c_frac, "D": all_d_frac}[k])
        zone_label = _fallback

    # ── Remap primary_surface to allowed classes ─────────────────────
    if _internal_surface == "Proximal":
        if quadrant in [1, 4]:
            primary_surface = "Mesial" if zone_counts["Proximal_R"] >= zone_counts["Proximal_L"] else "Distal"
        else:
            primary_surface = "Mesial" if zone_counts["Proximal_L"] >= zone_counts["Proximal_R"] else "Distal"
    elif _internal_surface == "Other":
        primary_surface = {"M": "Mesial", "C": "Occlusal", "D": "Distal"}.get(zone_label, "Occlusal")
    else:
        primary_surface = "Occlusal"

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

    # ── DEBUG: Step 3 — Point Cloud Voting ───────────────────────────
    if save_debug:
        _debug_step3_voting(
            zone_counts, zone_frac, all_zone_fractions,
            total_occ, total_prox, _internal_surface,
            zone_label, dominant_threshold,
            tooth_id, quadrant, case_num,
        )

    # ── Aggregated metrics ───────────────────────────────────────────
    cx_orig, cy_orig = _centroid(caries_pts)
    cx_rot, cy_rot = _centroid(rot_caries)
    rel_x_cent = float(np.clip((cx_rot - bx) / bw, 0, 1))
    rel_y_cent = float(np.clip((cy_rot - by) / bh, 0, 1))

    result = {
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
        "pca_method": _PCA_METHOD,
        "pca_method_name": get_pca_method_name(),
        "pca_bbox": {"x": round(bx, 2), "y": round(by, 2), "w": round(bw, 2), "h": round(bh, 2)},
        "predicted_surface_fine": predicted_surface_fine,
        "predicted_detail": predicted_detail,
        "all_zone_fractions": all_zone_fractions,
        "detail_explanation": detail_explanation,
        "n_points_raw": n_pts_raw,
        "n_points_cleaned": n_pts,
        "n_points_removed": n_pts_raw - n_pts,
    }

    # ── DEBUG: Step 4 — Final Summary ─────────────────────────────
    if save_debug:
        _debug_step4_summary(
            result, tooth_poly, caries_pts,
            rot_tooth, rot_caries, rel_xs, rel_ys,
            all_m_mask, all_c_mask, all_d_mask,
            bx, by, bw, bh,
            tooth_id, case_num,
        )

    return result


def _empty_result(tooth_id: str) -> Dict:
    return {
        "tooth_id": tooth_id,
        "primary_surface": "Unclassified",
        "zone_label": "N/A",
        "zone_fractions": {"M": 0, "C": 0, "D": 0},
        "n_points": 0,
        "zone_points": {"M": 0, "C": 0, "D": 0, "Proximal_L": 0, "Proximal_R": 0, "Other": 0},
        "caries_centroid": (0, 0),
        "caries_centroid_rot": (0, 0),
        "relative_position": {"rel_x": 0, "rel_y": 0},
        "rotation_angle_deg": 0,
        "pca_clamped": False,
        "pca_method": _PCA_METHOD,
        "pca_method_name": get_pca_method_name(),
        "pca_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        "predicted_surface_fine": "Unclassified",
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
