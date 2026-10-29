"""Deterministic surface assignment for caries lesions.

The caries model is 1-class: it says *where* a lesion is, not which surface it
is on.  Surface is derived from geometry in the tooth's PCA-aligned frame (the
same orientation Stage 2 reports in ``teeth[].axes``):

    rotate the tooth polygon + lesion centroid by the Stage-2 angle
    x_rel = (lesion_x - tooth_x_min) / tooth_width

    FDI Q1 / Q4 (patient's right):  x_rel < 0.30 -> distal
                                    0.30 .. 0.70 -> occlusal
                                    x_rel > 0.70 -> mesial
    FDI Q2 / Q3 (patient's left):   mirrored (left -> mesial, right -> distal)

Why this direction is not flipped: it was checked against the 500-case AIM XML
annotations (1,979 labelled lesions) on a 120-case sample with the pano model
(443 lesions matched to a detected tooth).  Agreement with the labelled surface
was 79.7% overall; per-quadrant proximal direction is 77-93% for mesial and
75-86% for distal.  Occlusal recall is weaker (38-74% by quadrant) because a
centroid cannot separate a central occlusal lesion from a proximal lesion whose
centroid sits near the middle of the crown.  See
``docs-md/pipeline-v2-caries-crop.md`` for the full table.

This module uses no ROI-derived features as model input.  The heuristic only
reads the predicted lesion centroid and the predicted tooth mask, both of which
come from image pixels at inference time.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np

from app.pipeline.pca_alignment import perform_pca, MAX_TILT_DEG

log = logging.getLogger(__name__)

SURFACE_METHOD = "Geometry_Heuristic"

LEFT_BOUND = 0.30
RIGHT_BOUND = 0.70

# Quadrants where "left of the tooth" is the distal side.
_DISTAL_LEFT_QUADRANTS = frozenset({1, 4})


def _quadrant(fdi: int) -> int:
    try:
        return int(str(int(fdi))[0])
    except (ValueError, IndexError):
        return 4


def tooth_x_extent(tooth_polygon: Sequence[Sequence[float]]) -> tuple[float, float] | None:
    """Return ``(x_min, x_max)`` of the tooth mask polygon."""
    if not tooth_polygon or len(tooth_polygon) < 3:
        return None
    xs = [float(p[0]) for p in tooth_polygon]
    x_min, x_max = min(xs), max(xs)
    if x_max <= x_min:
        return None
    return x_min, x_max


def _rotate_points(
    points: np.ndarray,
    center: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate ``points`` around ``center`` (same transform as Stage 2)."""
    import math

    p = points - center
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return p @ rotation.T + center


def classify_lesion_surface(
    fdi: int,
    tooth_polygon: Sequence[Sequence[float]],
    centroid: tuple[float, float] | Sequence[float],
    left_bound: float = LEFT_BOUND,
    right_bound: float = RIGHT_BOUND,
) -> dict[str, Any]:
    """Assign one lesion to ``mesial`` / ``occlusal`` / ``distal``.

    Returns a surface assignment. The detector score is attached by
    ``findings_for_lesions`` after this lesion has been assigned to a surface.
    """
    quadrant = _quadrant(fdi)
    extent = tooth_x_extent(tooth_polygon)
    if extent is None:
        log.warning("FDI %d: degenerate tooth polygon — defaulting surface to occlusal", fdi)
        return {
            "name": "occlusal",
            "label": "caries",
            "probability": None,
            "method": SURFACE_METHOD,
            "fallback_reason": "degenerate_tooth_polygon",
        }

    try:
        tooth = np.asarray(tooth_polygon, dtype=np.float64).reshape(-1, 2)
        center, angle, clamped = perform_pca(tooth.tolist(), fdi)
        rotated_tooth = _rotate_points(tooth, center, angle)
        rotated_centroid = _rotate_points(
            np.asarray(centroid, dtype=np.float64).reshape(1, 2), center, angle,
        )[0]
        x_min = float(rotated_tooth[:, 0].min())
        x_max = float(rotated_tooth[:, 0].max())
        if x_max <= x_min:
            raise ValueError("degenerate rotated tooth width")
        x_rel = (float(rotated_centroid[0]) - x_min) / (x_max - x_min)
    except Exception as exc:  # PCA is defensive; never fail the job for this
        log.warning("FDI %d: PCA alignment failed (%s) — using pano x extent", fdi, exc)
        x_min, x_max = extent
        x_rel = (float(centroid[0]) - x_min) / (x_max - x_min)
        clamped = False

    x_rel = min(1.0, max(0.0, x_rel))

    if x_rel < left_bound:
        surface = "distal" if quadrant in _DISTAL_LEFT_QUADRANTS else "mesial"
    elif x_rel > right_bound:
        surface = "mesial" if quadrant in _DISTAL_LEFT_QUADRANTS else "distal"
    else:
        surface = "occlusal"

    finding: dict[str, Any] = {
        "name": surface,
        "label": "caries",
        "probability": None,
        "method": SURFACE_METHOD,
        "x_rel": round(x_rel, 4),
    }
    if clamped:
        finding["rotation_clamped"] = True
        finding["max_tilt_deg"] = MAX_TILT_DEG
    return finding


def findings_for_lesions(
    fdi: int,
    tooth_polygon: Sequence[Sequence[float]],
    lesions: Iterable[Any],
    left_bound: float = LEFT_BOUND,
    right_bound: float = RIGHT_BOUND,
) -> list[dict[str, Any]]:
    """Build surface findings for a tooth, one per distinct surface.

    ``lesions`` are ``CariesLesion`` objects (or anything with a ``centroid``
    and ``confidence`` attribute). When multiple lesions land on the same
    surface, the highest caries-detector confidence is copied to ``probability``.
    This is a detector score, not a calibrated surface probability.
    """
    best: dict[str, dict[str, Any]] = {}
    best_conf: dict[str, float] = {}

    for lesion in lesions:
        finding = classify_lesion_surface(
            fdi=fdi,
            tooth_polygon=tooth_polygon,
            centroid=lesion.centroid,
            left_bound=left_bound,
            right_bound=right_bound,
        )
        name = finding["name"]
        conf = float(getattr(lesion, "confidence", 0.0) or 0.0)
        if name not in best or conf > best_conf[name]:
            finding["probability"] = round(conf, 4)
            best[name] = finding
            best_conf[name] = conf

    order = {"mesial": 0, "occlusal": 1, "distal": 2}
    return sorted(best.values(), key=lambda f: order.get(f["name"], 99))
