"""Tests for the result.json encoding of caries lesions."""
from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.postprocess import build_tooth_result, encode_caries_lesions


def _lesion():
    return SimpleNamespace(
        bbox_xyxy=(110.25, 220.5, 130.75, 250.0),
        polygon=[[110.25, 220.5], [130.75, 220.5], [130.75, 250.0]],
        confidence=0.81234,
        centroid=(120.5, 235.25),
    )


def test_encode_caries_lesions_uses_xywh_and_pano_coordinates():
    encoded = encode_caries_lesions([_lesion()])
    assert encoded == [{
        "bbox": [110.25, 220.5, 20.5, 29.5],
        "mask": {
            "encoding": "polygon",
            "data": [[110.25, 220.5], [130.75, 220.5], [130.75, 250.0]],
        },
        "confidence": 0.8123,
        "centroid": [120.5, 235.25],
    }]


def test_build_tooth_result_includes_empty_caries_array():
    tooth = build_tooth_result(
        id=0,
        fdi=11,
        bbox_xywh=(100, 200, 60, 80),
        mask_polygon=[[100, 200], [160, 200], [160, 280]],
        pano_confidence=0.9,
        axes=None,
        surface_findings=[],
    )
    assert tooth["caries"] == []
    assert tooth["axes"] == {
        "major": [0.0, 0.0],
        "minor": [0.0, 0.0],
        "rotation_deg": 0.0,
        "clamped": False,
    }
