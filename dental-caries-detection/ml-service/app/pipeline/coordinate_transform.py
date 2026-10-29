"""Coordinate transforms between panoramic image space and tooth-crop space.

The crop is defined by ``crop_bbox`` in pano coordinates:
``(x1, y1, x2, y2)`` with ``x2 > x1`` and ``y2 > y1``.  A point ``(x, y)`` in
pano space maps to ``(x - x1, y - y1)`` in crop space.

All helpers are pure and operate on ``Nx2`` arrays or single points; they never
mutate their inputs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

BBoxXYXY = tuple[int, int, int, int]


def crop_bbox_from_xywh(
    bbox_xywh: Sequence[float],
    image_shape: tuple[int, int],
    pad: int = 20,
) -> BBoxXYXY | None:
    """Build a clamped crop bbox around a pano ``[x, y, w, h]`` bbox.

    ``image_shape`` is ``(height, width)``.  Returns ``None`` when the bbox is
    degenerate after clamping.
    """
    x, y, w, h = (float(v) for v in bbox_xywh)
    img_h, img_w = int(image_shape[0]), int(image_shape[1])

    x1 = max(0, int(np.floor(x)) - int(pad))
    y1 = max(0, int(np.floor(y)) - int(pad))
    x2 = min(img_w, int(np.ceil(x + w)) + int(pad))
    y2 = min(img_h, int(np.ceil(y + h)) + int(pad))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _as_points(points: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"expected a point or Nx2 array, got shape {arr.shape}")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected a point or Nx2 array, got shape {arr.shape}")
    return arr


def pano_to_crop(
    points: Iterable[Sequence[float]] | np.ndarray,
    crop_bbox: BBoxXYXY,
) -> np.ndarray:
    """Map pano points to crop coordinates.  Returns a new ``Nx2`` array."""
    x1, y1, _, _ = crop_bbox
    return _as_points(points) - np.array([x1, y1], dtype=np.float64)


def crop_to_pano(
    points: Iterable[Sequence[float]] | np.ndarray,
    crop_bbox: BBoxXYXY,
) -> np.ndarray:
    """Map crop points back to pano coordinates.  Returns a new ``Nx2`` array."""
    x1, y1, _, _ = crop_bbox
    return _as_points(points) + np.array([x1, y1], dtype=np.float64)


def bbox_pano_to_crop(
    bbox_xyxy: Sequence[float],
    crop_bbox: BBoxXYXY,
) -> tuple[float, float, float, float]:
    """Map a pano ``xyxy`` bbox to crop coordinates (no clipping)."""
    x1, y1 = crop_bbox[0], crop_bbox[1]
    bx1, by1, bx2, by2 = (float(v) for v in bbox_xyxy)
    return bx1 - x1, by1 - y1, bx2 - x1, by2 - y1


def bbox_crop_to_pano(
    bbox_xyxy: Sequence[float],
    crop_bbox: BBoxXYXY,
) -> tuple[float, float, float, float]:
    """Map a crop ``xyxy`` bbox back to pano coordinates."""
    x1, y1 = crop_bbox[0], crop_bbox[1]
    bx1, by1, bx2, by2 = (float(v) for v in bbox_xyxy)
    return bx1 + x1, by1 + y1, bx2 + x1, by2 + y1


def xyxy_to_xywh(bbox_xyxy: Sequence[float]) -> tuple[int, int, int, int]:
    """Convert ``xyxy`` to integer ``[x, y, w, h]`` (API output format)."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox_xyxy)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def polygon_centroid(points: Iterable[Sequence[float]] | np.ndarray) -> tuple[float, float]:
    """Area centroid of a polygon; falls back to the vertex mean if degenerate."""
    import cv2

    arr = _as_points(points)
    if len(arr) < 3:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    moments = cv2.moments(arr.astype(np.float32).reshape(-1, 1, 2), binaryImage=False)
    if abs(moments["m00"]) < 1e-9:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0
