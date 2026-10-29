"""Parse the AIM XML caries annotations used by the 500-case dataset.

Each ``ImageAnnotation`` describes one lesion:

* an ``ImagingPhysicalEntityCharacteristic`` with a SNODENT tooth name,
  e.g. ``Permanent lower right second premolar tooth``;
* an ``ImagingPhysicalEntityCharacteristic`` with the surface,
  e.g. ``Distal Surface``;
* one ``MarkupEntity`` labelled ``Draw closed polygon ROI`` whose
  ``TwoDimensionSpatialCoordinate`` points are in **panoramic pixel**
  coordinates.

The parser deliberately ignores other markup entities (the export also contains
a small 0..100 coordinate artefact) by matching the local tag names exactly and
requiring the ROI label.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.pipeline.coordinate_transform import polygon_centroid

log = logging.getLogger(__name__)

_TOOTH_TYPES = (
    ("third molar", 8),
    ("second molar", 7),
    ("first molar", 6),
    ("second premolar", 5),
    ("first premolar", 4),
    ("canine", 3),
    ("lateral incisor", 2),
    ("central incisor", 1),
)

_SURFACES = ("mesial", "distal", "occlusal", "lingual", "buccal", "palatal", "labial")


@dataclass(frozen=True)
class XmlLesion:
    fdi: int
    tooth_name: str
    surface: str
    polygon: list[list[float]]          # pano coords
    centroid: tuple[float, float]       # pano coords
    source_file: str


@dataclass(frozen=True)
class XmlIssue:
    source_file: str
    reason: str


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def tooth_name_to_fdi(name: str) -> int | None:
    """Map a SNODENT permanent-tooth name to an FDI number.

    Returns ``None`` for primary/deciduous teeth and unrecognised names.
    """
    n = " ".join(name.lower().split())
    if "permanent" not in n:
        return None
    upper = "upper" in n
    lower = "lower" in n
    right = "right" in n
    left = "left" in n
    if not (upper or lower) or not (right or left):
        return None

    position = None
    for token, value in _TOOTH_TYPES:
        if token in n:
            position = value
            break
    if position is None:
        return None

    if upper and right:
        quadrant = 1
    elif upper and left:
        quadrant = 2
    elif lower and left:
        quadrant = 3
    else:
        quadrant = 4
    return quadrant * 10 + position


def normalize_surface(value: str) -> str | None:
    text = value.lower().replace("surface", "").strip()
    return text if text in _SURFACES else None


def _display_value(element: ET.Element) -> str | None:
    for child in element.iter():
        if _local(child.tag) == "displayName":
            return child.get("value")
    return None


def _characteristic_values(annotation: ET.Element) -> tuple[str | None, str | None]:
    tooth_name = None
    surface = None
    for element in annotation.iter():
        if _local(element.tag) != "ImagingPhysicalEntityCharacteristic":
            continue
        value = _display_value(element)
        if not value:
            continue
        lowered = value.lower()
        if lowered.endswith("tooth"):
            tooth_name = value
        elif "surface" in lowered:
            surface = value
    return tooth_name, surface


def _roi_polygon(annotation: ET.Element) -> list[list[float]]:
    points: list[list[float]] = []
    for markup in annotation.iter():
        if _local(markup.tag) != "MarkupEntity":
            continue
        label = next((c for c in markup if _local(c.tag) == "label"), None)
        if label is None or label.get("value") != "Draw closed polygon ROI":
            continue
        for coordinate in markup.iter():
            if _local(coordinate.tag) != "TwoDimensionSpatialCoordinate":
                continue
            x = next((c for c in coordinate if _local(c.tag) == "x"), None)
            y = next((c for c in coordinate if _local(c.tag) == "y"), None)
            if x is None or y is None:
                continue
            try:
                points.append([float(x.get("value")), float(y.get("value"))])
            except (TypeError, ValueError):
                continue
    return points


def parse_xml_file(path: Path) -> tuple[list[XmlLesion], list[XmlIssue]]:
    lesions: list[XmlLesion] = []
    issues: list[XmlIssue] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [], [XmlIssue(path.name, f"parse_error: {exc}")]

    for annotation in root.iter():
        if _local(annotation.tag) != "ImageAnnotation":
            continue

        tooth_name, surface_value = _characteristic_values(annotation)
        if tooth_name is None or surface_value is None:
            issues.append(XmlIssue(path.name, "missing tooth name or surface"))
            continue

        fdi = tooth_name_to_fdi(tooth_name)
        if fdi is None:
            issues.append(XmlIssue(path.name, f"unmapped tooth name: {tooth_name}"))
            continue

        surface = normalize_surface(surface_value)
        if surface is None:
            issues.append(XmlIssue(path.name, f"unmapped surface: {surface_value}"))
            continue

        polygon = _roi_polygon(annotation)
        if len(polygon) < 3:
            issues.append(XmlIssue(path.name, "ROI polygon has fewer than 3 points"))
            continue

        lesions.append(XmlLesion(
            fdi=fdi,
            tooth_name=tooth_name,
            surface=surface,
            polygon=polygon,
            centroid=polygon_centroid(polygon),
            source_file=path.name,
        ))

    return lesions, issues


def parse_case_dir(case_dir: Path) -> tuple[list[XmlLesion], list[XmlIssue]]:
    lesions: list[XmlLesion] = []
    issues: list[XmlIssue] = []
    for xml_path in sorted(case_dir.glob("*.xml")):
        found, problems = parse_xml_file(xml_path)
        lesions.extend(found)
        issues.extend(problems)
    return dedupe_lesions(lesions), issues


def dedupe_lesions(lesions: Iterable[XmlLesion]) -> list[XmlLesion]:
    """Drop exact duplicate annotations of the same FDI/surface.

    A duplicate is a lesion on the same tooth and surface whose centroid is
    within 1 px of an already-seen one.
    """
    kept: list[XmlLesion] = []
    for lesion in lesions:
        duplicate = any(
            other.fdi == lesion.fdi
            and other.surface == lesion.surface
            and abs(other.centroid[0] - lesion.centroid[0]) <= 1.0
            and abs(other.centroid[1] - lesion.centroid[1]) <= 1.0
            for other in kept
        )
        if not duplicate:
            kept.append(lesion)
    return kept
