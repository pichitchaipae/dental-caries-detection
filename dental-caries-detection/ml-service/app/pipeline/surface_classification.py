"""
Stage 3 — RF surface classification.

Uses the 14-feature vector from FeatureExtractor (ported from reference pipeline)
and the pre-trained RandomForestClassifier (rf_classify_ml.pkl).

Probability output:
  - Uses rf.classes_ index directly — does NOT assume class order.
  - Normalizes surface names to lowercase for the API contract.
  - Falls back to X-Thirds geometric classifier when RF returns no features.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from app.pipeline.detection import ToothDetection
from app.pipeline.postprocess import normalize_surface

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "is_upper", "x_mean", "y_mean", "x_std", "y_std",
    "x_min", "x_max", "y_min", "y_max", "x_range",
    "y_range", "x_centroid_dist", "aspect_ratio", "coverage",
]

MIN_CLUSTER_SIZE = 15
MAX_TILT_DEG = 45.0

# X-Thirds zone boundaries (from reference rf_classifier.py)
_LEFT_BOUND = 0.40
_RIGHT_BOUND = 0.60


# ---------------------------------------------------------------------------
# Feature extraction (ported from reference feature_extraction.py)
# ---------------------------------------------------------------------------

def _is_upper(fdi: int) -> bool:
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
    return p @ np.array([[c, -s], [s, c]]).T + center


def _get_bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    return mn[0], mn[1], mx[0] - mn[0], mx[1] - mn[1]


def _remove_small_clusters(caries_pts: np.ndarray) -> np.ndarray:
    """Remove connected components smaller than MIN_CLUSTER_SIZE pixels."""
    import cv2
    if len(caries_pts) < MIN_CLUSTER_SIZE:
        return caries_pts
    pts = np.array(caries_pts, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = 2
    w = int(x_max - x_min + 1 + 2 * pad)
    h = int(y_max - y_min + 1 + 2 * pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min - pad, y_min - pad])
    mask[shifted[:, 1], shifted[:, 0]] = 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_CLUSTER_SIZE:
            keep[labels == lbl] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        return caries_pts
    return np.column_stack([xs + x_min - pad, ys + y_min - pad]).astype(np.float64)


def _perform_pca(tooth_pts: np.ndarray, fdi: int) -> tuple[np.ndarray, float]:
    """Return (center, rotation_angle). Mirror of pca_alignment.perform_pca()."""
    import cv2
    center = np.mean(tooth_pts, axis=0)
    centered = tooth_pts - center
    _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
    primary = eigvecs[0].astype(np.float64)
    secondary = eigvecs[1].astype(np.float64)

    if abs(primary[1]) >= abs(secondary[1]):
        vertical_axis, horizontal_axis = primary.copy(), secondary.copy()
    else:
        vertical_axis, horizontal_axis = secondary.copy(), primary.copy()

    upper = _is_upper(fdi)
    quadrant = _get_quadrant(fdi)

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

    while rotation_angle > math.pi:
        rotation_angle -= 2 * math.pi
    while rotation_angle < -math.pi:
        rotation_angle += 2 * math.pi

    if abs(math.degrees(rotation_angle)) > MAX_TILT_DEG:
        rotation_angle = 0.0

    return center, rotation_angle


def extract_features(
    fdi: int,
    tooth_pts: list[list[float]],
    caries_pts: np.ndarray,
) -> dict | None:
    """
    Compute the 14-feature vector for the RF classifier.

    Returns None if caries region is empty after noise removal.
    Mirrors FeatureExtractor.extract_features() from reference pipeline.
    """
    caries_clean = _remove_small_clusters(np.array(caries_pts, dtype=np.float64))
    if len(caries_clean) == 0:
        return None

    tooth_np = np.array(tooth_pts, dtype=np.float64).reshape(-1, 2)
    center, angle = _perform_pca(tooth_np, fdi)
    tooth_rot = _rotate(tooth_np, center, angle)
    caries_rot = _rotate(caries_clean, center, angle)

    bbox_x, bbox_y, w, h = _get_bbox(tooth_rot)
    if w <= 0 or h <= 0:
        return None

    x_rel = np.clip((caries_rot[:, 0] - bbox_x) / w, 0.0, 1.0)
    y_rel = np.clip((caries_rot[:, 1] - bbox_y) / h, 0.0, 1.0)

    return {
        "is_upper": 1 if _is_upper(fdi) else 0,
        "x_mean": float(np.mean(x_rel)),
        "y_mean": float(np.mean(y_rel)),
        "x_std": float(np.std(x_rel)),
        "y_std": float(np.std(y_rel)),
        "x_min": float(np.min(x_rel)),
        "x_max": float(np.max(x_rel)),
        "y_min": float(np.min(y_rel)),
        "y_max": float(np.max(y_rel)),
        "x_range": float(np.max(x_rel) - np.min(x_rel)),
        "y_range": float(np.max(y_rel) - np.min(y_rel)),
        "x_centroid_dist": float(abs(np.mean(x_rel) - 0.5)),
        "aspect_ratio": float(w / h),
        "coverage": float(len(caries_clean) / (len(tooth_pts) + 1e-6)),
    }


# ---------------------------------------------------------------------------
# X-Thirds fallback classifier (mirrors reference rf_classifier.py)
# ---------------------------------------------------------------------------

def _classify_xthird(fdi: int, tooth_pts: list, caries_pts: np.ndarray) -> dict:
    """Geometric surface classifier — used when RF features cannot be computed."""
    caries_clean = _remove_small_clusters(np.array(caries_pts, dtype=np.float64))
    if len(caries_clean) == 0:
        return {
            "name": "occlusal",
            "label": "caries",
            "probability": 0.0,
            "method": "XThirds_Fallback",
            "fallback_reason": "Empty caries region after noise removal",
        }

    tooth_np = np.array(tooth_pts, dtype=np.float64).reshape(-1, 2)
    center, angle = _perform_pca(tooth_np, fdi)
    tooth_rot = _rotate(tooth_np, center, angle)
    caries_rot = _rotate(caries_clean, center, angle)
    bbox_x, _, w, _ = _get_bbox(tooth_rot)

    if w <= 0:
        surface = "occlusal"
    else:
        x_rel_mean = float(np.mean(np.clip((caries_rot[:, 0] - bbox_x) / w, 0, 1)))
        quadrant = _get_quadrant(fdi)
        if x_rel_mean < _LEFT_BOUND:
            surface = "distal" if quadrant in (1, 4) else "mesial"
        elif x_rel_mean > _RIGHT_BOUND:
            surface = "mesial" if quadrant in (1, 4) else "distal"
        else:
            surface = "occlusal"

    return {
        "name": surface,
        "label": "caries",
        "probability": 0.0,
        "method": "XThirds_Fallback",
    }


# ---------------------------------------------------------------------------
# Public entry point for Stage 3
# ---------------------------------------------------------------------------

def classify_tooth(
    det: ToothDetection,
    rf: Any,
) -> list[dict]:
    """
    Classify surface(s) for a single tooth.

    Returns:
        List of surface findings (may be empty if no caries points).
        Each finding: {"name": "occlusal", "label": "caries", "probability": 0.88}
    """
    if det.caries_points is None or len(det.caries_points) == 0:
        return []

    features = extract_features(str(det.fdi), det.tooth_polygon, det.caries_points)

    if features is None:
        log.debug("FDI %d: feature extraction returned None — X-Thirds fallback", det.fdi)
        fallback = _classify_xthird(det.fdi, det.tooth_polygon, det.caries_points)
        return [fallback]

    df = pd.DataFrame([features], columns=FEATURE_COLS)

    # Predict — use rf.classes_ index, never assume order
    probabilities = rf.predict_proba(df)[0]
    predicted_label = rf.predict(df)[0]
    class_index = list(rf.classes_).index(predicted_label)
    predicted_probability = float(probabilities[class_index])

    surface_name = normalize_surface(predicted_label)

    log.info(
        "FDI %d: RF classified surface=%s (prob=%.4f)",
        det.fdi, surface_name, predicted_probability,
    )

    return [{
        "name": surface_name,
        "label": "caries",
        "probability": round(predicted_probability, 4),
        "method": "RF",
    }]


def run_stage3(
    detections: list[ToothDetection],
    rf: Any,
) -> dict[int, list[dict]]:
    """
    Classify surfaces for all teeth with detected caries.

    Returns: fdi → list of surface findings
    """
    findings_by_fdi: dict[int, list[dict]] = {}
    for det in detections:
        findings = classify_tooth(det, rf)
        findings_by_fdi[det.fdi] = findings

    n_classified = sum(1 for v in findings_by_fdi.values() if v)
    log.info("Stage 3 RF: classified %d teeth with caries surfaces", n_classified)
    return findings_by_fdi
