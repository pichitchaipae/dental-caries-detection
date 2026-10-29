"""Tests for the deterministic mesial/occlusal/distal heuristic."""
from types import SimpleNamespace

from app.pipeline.surface_heuristic import classify_lesion_surface, findings_for_lesions

# A tall rectangle gives PCA a stable vertical axis (a square is degenerate:
# its two eigenvalues are equal).
TOOTH = [[0, 0], [100, 0], [100, 200], [0, 200]]


def _surface(fdi, x):
    return classify_lesion_surface(fdi, TOOTH, (x, 100))["name"]


def test_q1_and_q4_left_is_distal_right_is_mesial():
    for fdi in (11, 46):
        assert _surface(fdi, 10) == "distal"
        assert _surface(fdi, 50) == "occlusal"
        assert _surface(fdi, 90) == "mesial"


def test_q2_and_q3_are_mirrored():
    for fdi in (21, 36):
        assert _surface(fdi, 10) == "mesial"
        assert _surface(fdi, 50) == "occlusal"
        assert _surface(fdi, 90) == "distal"


def test_bounds_are_configurable():
    finding = classify_lesion_surface(11, TOOTH, (35, 100), left_bound=0.5, right_bound=0.7)
    assert finding["name"] == "distal"
    assert classify_lesion_surface(11, TOOTH, (35, 100))["name"] == "occlusal"


def test_tilted_tooth_is_aligned_before_classifying():
    # 30-degree tilt: the same physical mesial point must stay mesial.
    import math

    cx, cy = 100.0, 100.0
    angle = math.radians(30)
    c, s = math.cos(angle), math.sin(angle)
    tilted = []
    for x, y in TOOTH:
        dx, dy = x - 50, y - 100
        tilted.append([cx + c * dx - s * dy, cy + s * dx + c * dy])
    # un-rotated point (90, 100) -> rotate into the tilted frame
    px, py = 90 - 50, 100 - 100
    point = (cx + c * px - s * py, cy + s * px + c * py)
    assert classify_lesion_surface(11, tilted, point)["name"] == "mesial"


def test_degenerate_polygon_falls_back_to_occlusal():
    finding = classify_lesion_surface(11, [[0, 0]], (1, 1))
    assert finding["name"] == "occlusal"
    assert finding["fallback_reason"] == "degenerate_tooth_polygon"


def test_findings_are_deduped_and_sorted():
    lesions = [
        SimpleNamespace(centroid=(10, 100), confidence=0.4),   # distal
        SimpleNamespace(centroid=(12, 100), confidence=0.9),   # distal (higher conf)
        SimpleNamespace(centroid=(50, 100), confidence=0.7),   # occlusal
    ]
    findings = findings_for_lesions(11, TOOTH, lesions)
    assert [f["name"] for f in findings] == ["occlusal", "distal"]
    assert findings[1]["probability"] == 0.9
    assert findings[0]["probability"] == 0.7
    assert findings[0]["method"] == "Geometry_Heuristic"
