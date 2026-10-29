"""
Postprocessing helpers — mask encoding and output normalization.

Responsibilities:
  - Encode segmentation masks as polygon lists in original image coordinates.
  - Normalize surface names to lowercase (API contract uses lowercase).
  - Build the final tooth dict structure for result.json.

All coordinates are in the ORIGINAL image coordinate space.
bbox format: [x, y, w, h]  (not xyxy)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any

# Surface name normalization — RF classes_ → API contract (lowercase)
SURFACE_NAME_MAP: dict[str, str] = {
    "Distal": "distal",
    "Mesial": "mesial",
    "Occlusal": "occlusal",
}

_KNOWN_SURFACES = frozenset(SURFACE_NAME_MAP.keys())


def normalize_surface(name: str) -> str:
    """Return lowercase surface name per API contract."""
    if name in SURFACE_NAME_MAP:
        return SURFACE_NAME_MAP[name]
    # Already lowercase or unknown — pass through lowercase
    return name.lower()


def encode_mask_polygon(
    mask_points: list[list[float]],
) -> dict[str, Any]:
    """
    Encode a list of polygon points as the API polygon format.

    Input:  [[x1,y1], [x2,y2], ...]  (original image coords)
    Output: {"encoding": "polygon", "data": [[x1,y1],[x2,y2],...]}
    """
    return {
        "encoding": "polygon",
        "data": [[float(x), float(y)] for x, y in mask_points],
    }


def mask_bits_to_polygon(
    binary_mask: np.ndarray,
    epsilon_factor: float = 0.005,
) -> list[list[float]] | None:
    """
    Convert a binary HxW uint8 mask to a simplified polygon contour.

    Returns a list of [x, y] pairs in image coordinates, or None if no contour found.
    epsilon_factor: controls Douglas-Peucker simplification (fraction of arc length).
    """
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return [[float(pt[0][0]), float(pt[0][1])] for pt in approx]


def encode_caries_lesions(lesions: list) -> list[dict[str, Any]]:
    """Encode Stage-1 caries lesion objects for result.json.

    ``lesions`` are ``detection.CariesLesion`` instances.  Bboxes are ``[x, y,
    w, h]`` and masks are polygons, both in pano coordinates — the same
    convention as the tooth entry.
    """
    out: list[dict[str, Any]] = []
    for lesion in lesions:
        x1, y1, x2, y2 = (float(v) for v in lesion.bbox_xyxy)
        out.append({
            "bbox": [
                round(x1, 2), round(y1, 2),
                round(x2 - x1, 2), round(y2 - y1, 2),
            ],
            "mask": encode_mask_polygon(lesion.polygon),
            "confidence": round(float(lesion.confidence), 4),
            "centroid": [
                round(float(lesion.centroid[0]), 2),
                round(float(lesion.centroid[1]), 2),
            ],
        })
    return out


def build_tooth_result(
    id: int,
    fdi: int,
    bbox_xywh: tuple[int, int, int, int],
    mask_polygon: list[list[float]],
    pano_confidence: float,
    axes: dict | None,
    surface_findings: list[dict],
    caries_lesions: list | None = None,
) -> dict[str, Any]:
    """
    Build a single tooth entry for the teeth array in result.json.

    surface_findings: list of {"name": "occlusal", "label": "caries",
                               "probability": <caries detector confidence>,
                               "method": "Geometry_Heuristic"}
    caries_lesions:   Stage-1 CariesLesion objects (pano coordinates)
    """
    x, y, w, h = bbox_xywh
    
    # Default axes if missing to satisfy strict Zod schema requirement
    safe_axes = axes if axes is not None else {
        "major": [0.0, 0.0],
        "minor": [0.0, 0.0],
        "rotation_deg": 0.0,
        "clamped": False,
    }

    return {
        "id": id,
        "fdi": fdi,
        "confidence": round(float(pano_confidence), 4),
        "bbox": [x, y, w, h],
        "mask": encode_mask_polygon(mask_polygon),
        "axes": safe_axes,
        "surfaces": surface_findings,
        "caries": encode_caries_lesions(caries_lesions or []),
    }
