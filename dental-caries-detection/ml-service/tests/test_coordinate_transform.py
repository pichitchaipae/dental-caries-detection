"""Tests for pano <-> crop coordinate helpers."""
import numpy as np

from app.pipeline.coordinate_transform import (
    bbox_crop_to_pano,
    bbox_iou,
    bbox_pano_to_crop,
    crop_bbox_from_xywh,
    crop_to_pano,
    pano_to_crop,
    polygon_centroid,
    xyxy_to_xywh,
)


def test_crop_bbox_from_xywh_applies_pad_and_clamps():
    # image is 100x200 (h, w)
    assert crop_bbox_from_xywh((10, 20, 30, 40), (100, 200), pad=20) == (0, 0, 60, 80)
    assert crop_bbox_from_xywh((150, 80, 40, 20), (100, 200), pad=20) == (130, 60, 200, 100)


def test_crop_bbox_returns_none_for_degenerate_input():
    assert crop_bbox_from_xywh((0, 0, 0, 0), (100, 100), pad=0) is None


def test_pano_crop_roundtrip_preserves_points():
    points = np.array([[10.0, 20.0], [30.0, 40.0]])
    crop_bbox = (5, 5, 100, 100)
    in_crop = pano_to_crop(points, crop_bbox)
    assert np.allclose(in_crop, [[5.0, 15.0], [25.0, 35.0]])
    assert np.allclose(crop_to_pano(in_crop, crop_bbox), points)


def test_bbox_transforms_are_inverse():
    crop_bbox = (100, 200, 400, 500)
    pano = (120.0, 230.0, 180.0, 300.0)
    crop = bbox_pano_to_crop(pano, crop_bbox)
    assert crop == (20.0, 30.0, 80.0, 100.0)
    assert bbox_crop_to_pano(crop, crop_bbox) == pano


def test_polygon_centroid_square():
    cx, cy = polygon_centroid([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert abs(cx - 5.0) < 1e-6
    assert abs(cy - 5.0) < 1e-6


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9


def test_xyxy_to_xywh():
    assert xyxy_to_xywh((10.4, 20.6, 30.2, 50.9)) == (10, 21, 20, 30)
