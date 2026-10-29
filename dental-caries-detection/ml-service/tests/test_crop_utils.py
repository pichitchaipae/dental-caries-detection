"""Tests for tooth-crop helpers shared by training and serving."""
from __future__ import annotations

import numpy as np

from app.pipeline.crop_utils import (
    clip_polygon_to_crop,
    normalize_polygon,
    suppress_other_teeth,
)

CROP_BBOX = (100, 100, 300, 300)   # x1, y1, x2, y2


def _white_crop() -> np.ndarray:
    return np.full((200, 200, 3), 255, dtype=np.uint8)


def test_suppress_other_teeth_blackens_neighbour_but_keeps_target_and_margin():
    crop = _white_crop()
    target = [[130, 130], [200, 130], [200, 250], [130, 250]]
    neighbour = [[205, 130], [290, 130], [290, 250], [205, 250]]

    out = suppress_other_teeth(crop, CROP_BBOX, target, [neighbour], preserve_pad=15)

    # Inside the target tooth: untouched.
    assert out[100, 60].tolist() == [255, 255, 255]
    # Deep inside the neighbour (crop x=150): blacked out.
    assert out[100, 150].tolist() == [0, 0, 0]
    # Neighbour pixels inside the 15 px target dilation: preserved so a
    # proximal lesion at the contact point is not erased.
    assert out[100, 110].tolist() == [255, 255, 255]
    # Outside both teeth: untouched.
    assert out[10, 10].tolist() == [255, 255, 255]
    # The caller's crop is not mutated.
    assert crop[100, 150].tolist() == [255, 255, 255]


def test_suppress_other_teeth_noop_when_no_neighbours():
    crop = _white_crop()
    target = [[130, 130], [200, 130], [200, 250], [130, 250]]
    out = suppress_other_teeth(crop, CROP_BBOX, target, [], preserve_pad=15)
    assert np.array_equal(out, crop)


def test_clip_polygon_to_crop_returns_crop_coordinates():
    polygon = [[80, 80], [220, 80], [220, 220], [80, 220]]  # partly outside

    clipped = clip_polygon_to_crop(polygon, CROP_BBOX, min_area=4.0)

    assert clipped is not None
    assert len(clipped) >= 3
    xs = [p[0] for p in clipped]
    ys = [p[1] for p in clipped]
    assert min(xs) >= 0 and max(xs) <= 200
    assert min(ys) >= 0 and max(ys) <= 200


def test_clip_polygon_to_crop_returns_none_when_fully_outside():
    polygon = [[0, 0], [50, 0], [50, 50], [0, 50]]
    assert clip_polygon_to_crop(polygon, CROP_BBOX, min_area=4.0) is None


def test_clip_polygon_to_crop_returns_none_for_tiny_visible_area():
    polygon = [[99, 99], [101, 99], [101, 101], [99, 101]]  # 2x2 px visible
    assert clip_polygon_to_crop(polygon, CROP_BBOX, min_area=16.0) is None


def test_normalize_polygon_clamps_and_scales():
    polygon = [[-5, 10], [100, 10], [100, 300], [50, 300]]
    out = normalize_polygon(polygon, (200, 200))
    assert out[0] == (0.0, 0.05)
    assert out[1] == (0.5, 0.05)
    assert out[2] == (0.5, 1.0)
    assert out[3] == (0.25, 1.0)
