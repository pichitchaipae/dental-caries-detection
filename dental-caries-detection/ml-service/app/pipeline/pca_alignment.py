"""
Stage 2 — PCA axis alignment.

Wraps FeatureExtractor.perform_pca() from the reference pipeline,
which already handles:
  - Upper/lower jaw orientation flip
  - Left/right quadrant horizontal-axis flip
  - Clamping of rotations > MAX_TILT_DEG (45°) to 0

Output per tooth: AlignedTooth with axes metadata for result.json
and the rotation angle used by Stage 3 feature extraction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.pipeline.detection import ToothDetection

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Ported verbatim from reference pipeline feature_extraction.py
# (to avoid a cross-service import dependency)
# ------------------------------------------------------------------
MAX_TILT_DEG = 45.0


def _is_upper_jaw(fdi: int) -> bool:
    try:
        return int(str(fdi)[0]) in (1, 2)
    except (ValueError, IndexError):
        return False


def _get_quadrant(fdi: int) -> int:
    try:
        return int(str(fdi)[0])
    except (ValueError, IndexError):
        return 4


def _rotate(pts: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    p = pts - center
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return p @ R.T + center


def perform_pca(
    tooth_pts: list[list[float]],
    fdi: int,
) -> tuple[np.ndarray, float, bool]:
    """
    Compute PCA-based canonical orientation for a tooth.

    Returns:
        center (np.ndarray [x, y]):   centroid of tooth mask
        rotation_angle (float):       radians to rotate points to canonical orientation
        clamped (bool):               True if rotation exceeded MAX_TILT_DEG and was set to 0
    """
    import cv2  # imported here to avoid top-level cost in tests

    pts = np.array(tooth_pts, dtype=np.float64).reshape(-1, 2)
    center = np.mean(pts, axis=0)
    centered = pts - center

    _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
    primary = eigvecs[0].astype(np.float64)
    secondary = eigvecs[1].astype(np.float64)

    # Identify vertical vs horizontal axis
    if abs(primary[1]) >= abs(secondary[1]):
        vertical_axis = primary.copy()
        horizontal_axis = secondary.copy()
    else:
        vertical_axis = secondary.copy()
        horizontal_axis = primary.copy()

    upper = _is_upper_jaw(fdi)
    quadrant = _get_quadrant(fdi)

    # Orientation fix — prevent 180° flips
    if upper:
        if vertical_axis[1] < 0:
            vertical_axis = -vertical_axis
    else:
        if vertical_axis[1] > 0:
            vertical_axis = -vertical_axis

    if quadrant in (1, 4):
        if horizontal_axis[0] < 0:
            horizontal_axis = -horizontal_axis
    else:
        if horizontal_axis[0] > 0:
            horizontal_axis = -horizontal_axis

    angle_from_x = math.atan2(vertical_axis[1], vertical_axis[0])
    target_angle = math.pi / 2 if upper else -math.pi / 2
    rotation_angle = target_angle - angle_from_x

    # Normalise to [-π, π]
    while rotation_angle > math.pi:
        rotation_angle -= 2 * math.pi
    while rotation_angle < -math.pi:
        rotation_angle += 2 * math.pi

    clamped = False
    if abs(math.degrees(rotation_angle)) > MAX_TILT_DEG:
        rotation_angle = 0.0
        clamped = True

    return center, rotation_angle, clamped


# ------------------------------------------------------------------
# Public entry point for Stage 2
# ------------------------------------------------------------------

@dataclass
class AxesResult:
    major: tuple[float, float]      # dominant axis direction [dx, dy]
    minor: tuple[float, float]      # secondary axis direction [dx, dy]
    rotation_deg: float
    clamped: bool


def compute_axes(
    fdi: int,
    tooth_polygon: list[list[float]],
) -> AxesResult | None:
    """
    Compute PCA axes for a single tooth.

    Returns None if the polygon is degenerate (< 3 points or all collinear).
    """
    if len(tooth_polygon) < 3:
        log.warning("FDI %d: degenerate polygon (%d pts) — axes=null", fdi, len(tooth_polygon))
        return None

    try:
        import cv2
        pts = np.array(tooth_polygon, dtype=np.float32).reshape(-1, 2)
        center = np.mean(pts, axis=0)
        centered = pts - center
        _, eigvecs = cv2.PCACompute(centered, mean=None)

        center_nd, rotation_angle, clamped = perform_pca(tooth_polygon, fdi)

        rotation_deg = round(math.degrees(rotation_angle), 2)
        major = (round(float(eigvecs[0][0]), 4), round(float(eigvecs[0][1]), 4))
        minor = (round(float(eigvecs[1][0]), 4), round(float(eigvecs[1][1]), 4))

        if clamped:
            log.debug("FDI %d: rotation clamped to 0 (exceeded %.0f°)", fdi, MAX_TILT_DEG)

        return AxesResult(
            major=major,
            minor=minor,
            rotation_deg=rotation_deg,
            clamped=clamped,
        )
    except Exception as exc:
        log.warning("FDI %d: PCA failed — %s", fdi, exc)
        return None


def run_stage2(
    detections: list[ToothDetection],
) -> dict[int, AxesResult | None]:
    """
    Compute PCA axes for all detected teeth.

    Returns a dict: fdi → AxesResult (or None for degenerate cases)
    """
    axes_by_fdi: dict[int, AxesResult | None] = {}
    for det in detections:
        axes_by_fdi[det.fdi] = compute_axes(det.fdi, det.tooth_polygon)
    log.info(
        "Stage 2 PCA: computed axes for %d/%d teeth",
        sum(1 for v in axes_by_fdi.values() if v is not None),
        len(detections),
    )
    return axes_by_fdi
