"""Tests for the AIM XML annotation parser (labels only, never model input)."""
from __future__ import annotations

from pathlib import Path

from app.pipeline.xml_annotations import (
    normalize_surface,
    parse_case_dir,
    parse_xml_file,
    tooth_name_to_fdi,
)

NS = "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"


def _annotation(tooth: str, surface: str, points: list[tuple[float, float]],
                decoy: bool = False) -> str:
    coords = "".join(
        f"<TwoDimensionSpatialCoordinate><x value=\"{x}\"/><y value=\"{y}\"/>"
        f"</TwoDimensionSpatialCoordinate>"
        for x, y in points
    )
    decoy_markup = (
        "<MarkupEntity><label value=\"Other shape\"/>"
        "<twoDimensionSpatialCoordinateCollection>"
        "<TwoDimensionSpatialCoordinate><x value=\"0\"/><y value=\"0\"/>"
        "</TwoDimensionSpatialCoordinate>"
        "</twoDimensionSpatialCoordinateCollection></MarkupEntity>"
        if decoy else ""
    )
    return f"""
    <ImageAnnotation>
      <ImagingPhysicalEntity>
        <ImagingPhysicalEntityCharacteristic>
          <typeCode><displayName value="{tooth}"/></typeCode>
        </ImagingPhysicalEntityCharacteristic>
        <ImagingPhysicalEntityCharacteristic>
          <typeCode><displayName value="{surface}"/></typeCode>
        </ImagingPhysicalEntityCharacteristic>
      </ImagingPhysicalEntity>
      <markupEntityCollection>
        {decoy_markup}
        <MarkupEntity>
          <label value="Draw closed polygon ROI"/>
          <twoDimensionSpatialCoordinateCollection>{coords}
          </twoDimensionSpatialCoordinateCollection>
        </MarkupEntity>
      </markupEntityCollection>
    </ImageAnnotation>
    """


def _xml(*annotations: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<ImageAnnotationCollection xmlns="{NS}">'
        f"<imageAnnotations>{''.join(annotations)}</imageAnnotations>"
        f"</ImageAnnotationCollection>"
    )


def test_tooth_name_to_fdi_covers_all_quadrants():
    assert tooth_name_to_fdi("Permanent upper right second premolar tooth") == 15
    assert tooth_name_to_fdi("Permanent upper left first molar tooth") == 26
    assert tooth_name_to_fdi("Permanent lower left second molar tooth") == 37
    assert tooth_name_to_fdi("Permanent lower right central incisor tooth") == 41
    assert tooth_name_to_fdi("Primary upper right first molar tooth") is None
    assert tooth_name_to_fdi("Permanent upper right unknown tooth") is None


def test_normalize_surface():
    assert normalize_surface("Distal Surface") == "distal"
    assert normalize_surface("Occlusal surface") == "occlusal"
    assert normalize_surface("Mesial") == "mesial"
    assert normalize_surface("Biting Surface") is None


def test_parser_uses_only_the_roi_markup(tmp_path: Path):
    path = tmp_path / "case.xml"
    path.write_text(_xml(
        _annotation(
            "Permanent lower right second premolar tooth",
            "Distal Surface",
            [(1100.0, 1000.0), (1168.0, 1000.0), (1130.0, 1030.0)],
            decoy=True,
        ),
    ), encoding="utf-8")

    lesions, issues = parse_xml_file(path)

    assert issues == []
    assert len(lesions) == 1
    lesion = lesions[0]
    assert lesion.fdi == 45
    assert lesion.surface == "distal"
    assert len(lesion.polygon) == 3
    # The decoy 0..100 coordinate artefact must not be included.
    assert min(p[0] for p in lesion.polygon) == 1100.0
    assert lesion.centroid[0] > 1100.0


def test_parser_reports_issues_without_crashing(tmp_path: Path):
    path = tmp_path / "bad.xml"
    path.write_text(_xml(
        _annotation("Permanent upper right first molar tooth", "Distal Surface", []),
    ), encoding="utf-8")

    lesions, issues = parse_xml_file(path)

    assert lesions == []
    assert len(issues) == 1
    assert "fewer than 3 points" in issues[0].reason


def test_parse_case_dir_merges_files_and_dedupes(tmp_path: Path):
    first = tmp_path / "a.xml"
    second = tmp_path / "b.xml"
    annotation = _annotation(
        "Permanent upper left first molar tooth",
        "Occlusal surface",
        [(1900.0, 800.0), (1960.0, 800.0), (1930.0, 850.0)],
    )
    first.write_text(_xml(annotation), encoding="utf-8")
    second.write_text(_xml(annotation), encoding="utf-8")

    lesions, issues = parse_case_dir(tmp_path)

    assert issues == []
    assert len(lesions) == 1
    assert lesions[0].fdi == 26
    assert lesions[0].surface == "occlusal"
