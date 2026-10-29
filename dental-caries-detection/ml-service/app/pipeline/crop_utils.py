"""Tooth-crop helpers shared by serving and dataset preparation.

The caries model runs on a crop around one tooth.  Adjacent teeth can intrude
into that crop; leaving them visible would teach the model that a neighbour's
lesion is a negative, and at inference it could detect a neighbour's lesion and
assign it to the wrong tooth.

``suppress_other_teeth`` blacks out pixels that belong to other detected teeth,
while preserving a margin around the target mask so proximal lesions at the
contact point stay visible.  Training and serving use the same rule.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.coordinate_transform import BBoxXYXY


def _rasterize(polygon: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def suppress_other_teeth(
    crop: np.ndarray,
    crop_bbox: BBoxXYXY,
    target_polygon: list[list[float]],
    other_polygons: list[list[list[float]]],
    preserve_pad: int = 15,
) -> np.ndarray:
    """Return a copy of ``crop`` with non-target teeth blacked out.

    All polygons are in pano coordinates.  ``preserve_pad`` dilates the target
    mask so lesion pixels at the tooth boundary are not erased.
    """
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]

    offset = np.array([crop_bbox[0], crop_bbox[1]], dtype=np.float64)
    target_pts = np.asarray(target_polygon, dtype=np.float64).reshape(-1, 2) - offset
    target_mask = _rasterize(target_pts.tolist(), (h, w))
    if preserve_pad > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * preserve_pad + 1, 2 * preserve_pad + 1)
        )
        keep = cv2.dilate(target_mask, kernel)
    else:
        keep = target_mask

    other_mask = np.zeros((h, w), dtype=np.uint8)
    for polygon in other_polygons:
        if not polygon or len(polygon) < 3:
            continue
        pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2) - offset
        other_mask |= _rasterize(pts.tolist(), (h, w))

    remove = (other_mask > 0) & (keep == 0)
    if not remove.any():
        return crop
    out = crop.copy()
    out[remove] = 0
    return out


def clip_polygon_to_crop(
    polygon_pano: list[list[float]],
    crop_bbox: BBoxXYXY,
    min_area: float = 4.0,
) -> list[list[float]] | None:
    """Clip a pano polygon to the crop rectangle.

    Returns a polygon in **crop** coordinates, or ``None`` when the visible
    intersection is smaller than ``min_area`` pixels.
    """
    x1, y1, x2, y2 = crop_bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or len(polygon_pano) < 3:
        return None

    pts = np.asarray(polygon_pano, dtype=np.float64).reshape(-1, 2)
    pts = pts - np.array([x1, y1], dtype=np.float64)
    mask = _rasterize(pts.tolist(), (h, w))
    if int(mask.sum()) < min_area:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    approx = cv2.approxPolyDP(contour, 0.002 * cv2.arcLength(contour, True), True)
    if len(approx) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def normalize_polygon(
    polygon_crop: list[list[float]],
    crop_shape: tuple[int, int],
) -> list[tuple[float, float]]:
    """Normalise a crop-space polygon to YOLO ``0..1`` coordinates."""
    h, w = crop_shape
    out: list[tuple[float, float]] = []
    for x, y in polygon_crop:
        out.append((
            min(1.0, max(0.0, float(x) / w)),
            min(1.0, max(0.0, float(y) / h)),
        ))
    return out
