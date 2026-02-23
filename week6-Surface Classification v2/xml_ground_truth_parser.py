"""
AIM XML Ground Truth Parser
============================

Parses AIM (Annotation and Image Markup) v4.2 XML files to extract
dental caries ground truth annotations including:
  - FDI Tooth ID (from SNODENT codes)
  - Surface label (Occlusal, Mesial, Distal, Buccal, Lingual, …)
  - Caries severity / type
  - ROI polygon coordinates (TwoDimensionSpatialCoordinate)
  - Calculation metrics (Area, Mean, Std Dev)

The AIM XML namespace is:
    gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM

Each XML file in "500 cases with annotation" represents ONE caries
annotation for ONE tooth.  A case folder may contain multiple XML files
(one per annotated lesion).

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from snodent_tooth_map import (
    SNODENT_TO_FDI,
    SNODENT_SURFACE_MAP,
    DISPLAY_NAME_TO_SURFACE,
    snodent_display_to_fdi,
)


# =============================================================================
# AIM XML Namespace
# =============================================================================
AIM_NS = "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"
ISO_NS = "uri:iso.org:21090"

NS = {"aim": AIM_NS, "iso": ISO_NS}


# =============================================================================
# Core Parsing Functions
# =============================================================================

def _find_text(element, tag: str) -> str:
    """Return the 'value' attribute of a child element, or ''."""
    child = element.find(f"aim:{tag}", NS)
    if child is not None:
        return child.get("value", "")
    return ""


def _get_display_name(element) -> str:
    """Extract iso:displayName value from a typeCode element."""
    dn = element.find("iso:displayName", NS)
    if dn is not None:
        return dn.get("value", "")
    return ""


# =============================================================================
# Main Parser
# =============================================================================

def parse_aim_xml(xml_path: str) -> Optional[Dict]:
    """
    Parse a single AIM XML file and return structured ground truth data.

    Returns a dictionary with keys:
        xml_file        : str  – filename
        tooth_fdi       : str  – FDI two-digit ID (e.g. "45")
        tooth_snodent   : str  – SNODENT code for tooth
        tooth_name      : str  – Human-readable name from XML
        surface_code    : str  – SNODENT surface code
        surface_name    : str  – "Occlusal" / "Mesial" / "Distal" / …
        caries_type     : str  – e.g. "Dental caries"
        caries_subtype  : str  – e.g. "Dentine caries"
        severity        : str  – e.g. "Moderate stage coronal caries"
        area_mm         : float
        mean_intensity  : float
        std_intensity   : float
        roi_coordinates : list[tuple(float, float)]  – (x, y) pairs
        roi_centroid    : tuple(float, float)
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"[XML Parse Error] {xml_path}: {e}")
        return None

    root = tree.getroot()

    # Navigate to <ImageAnnotation>
    annotations = root.find("aim:imageAnnotations", NS)
    if annotations is None:
        return None
    ann = annotations.find("aim:ImageAnnotation", NS)
    if ann is None:
        return None

    result: Dict = {"xml_file": os.path.basename(xml_path)}

    # ------------------------------------------------------------------
    # 1. Caries type (top-level typeCode on ImageAnnotation)
    # ------------------------------------------------------------------
    ann_type = ann.find("aim:typeCode", NS)
    if ann_type is not None:
        result["caries_type"] = _get_display_name(ann_type)
    else:
        result["caries_type"] = ""

    # ------------------------------------------------------------------
    # 2. Tooth location + Surface from imagingPhysicalEntityCollection
    # ------------------------------------------------------------------
    phys_coll = ann.find("aim:imagingPhysicalEntityCollection", NS)
    tooth_fdi = ""
    tooth_snodent = ""
    tooth_name = ""
    surface_code = ""
    surface_name = ""

    if phys_coll is not None:
        entity = phys_coll.find("aim:ImagingPhysicalEntity", NS)
        if entity is not None:
            char_coll = entity.find(
                "aim:imagingPhysicalEntityCharacteristicCollection", NS
            )
            if char_coll is not None:
                chars = char_coll.findall(
                    "aim:ImagingPhysicalEntityCharacteristic", NS
                )
                for char_el in chars:
                    q_idx_el = char_el.find("aim:questionIndex", NS)
                    q_idx = q_idx_el.get("value", "") if q_idx_el is not None else ""
                    tc = char_el.find("aim:typeCode", NS)
                    if tc is None:
                        continue
                    code = tc.get("code", "")
                    display = _get_display_name(tc)

                    if q_idx == "0":
                        # Tooth anatomical position
                        tooth_snodent = code
                        tooth_name = display
                        # Try SNODENT code lookup first, then display name parse
                        tooth_fdi = SNODENT_TO_FDI.get(code, "")
                        if not tooth_fdi:
                            tooth_fdi = snodent_display_to_fdi(display)
                    elif q_idx == "1":
                        # Surface
                        surface_code = code
                        surface_name = SNODENT_SURFACE_MAP.get(code, "")
                        if not surface_name:
                            surface_name = DISPLAY_NAME_TO_SURFACE.get(display, display)

    result["tooth_fdi"] = tooth_fdi
    result["tooth_snodent"] = tooth_snodent
    result["tooth_name"] = tooth_name
    result["surface_code"] = surface_code
    result["surface_name"] = surface_name

    # ------------------------------------------------------------------
    # 3. Caries severity from imagingObservationEntityCollection
    # ------------------------------------------------------------------
    obs_coll = ann.find("aim:imagingObservationEntityCollection", NS)
    caries_subtype = ""
    severity = ""
    if obs_coll is not None:
        obs = obs_coll.find("aim:ImagingObservationEntity", NS)
        if obs is not None:
            char_coll2 = obs.find(
                "aim:imagingObservationCharacteristicCollection", NS
            )
            if char_coll2 is not None:
                obs_char = char_coll2.find(
                    "aim:ImagingObservationCharacteristic", NS
                )
                if obs_char is not None:
                    tc2 = obs_char.find("aim:typeCode", NS)
                    if tc2 is not None:
                        caries_subtype = _get_display_name(tc2)
                    quant_coll = obs_char.find(
                        "aim:characteristicQuantificationCollection", NS
                    )
                    if quant_coll is not None:
                        cq = quant_coll.find("aim:CharacteristicQuantification", NS)
                        if cq is not None:
                            sev_tc = cq.find("aim:typeCode", NS)
                            if sev_tc is not None:
                                severity = _get_display_name(sev_tc)

    result["caries_subtype"] = caries_subtype
    result["severity"] = severity

    # ------------------------------------------------------------------
    # 4. Calculation metrics (Area, Mean, Std Dev)
    # ------------------------------------------------------------------
    calc_coll = ann.find("aim:calculationEntityCollection", NS)
    area_mm = 0.0
    mean_intensity = 0.0
    std_intensity = 0.0

    if calc_coll is not None:
        for calc_entity in calc_coll.findall("aim:CalculationEntity", NS):
            desc_el = calc_entity.find("aim:description", NS)
            desc = desc_el.get("value", "") if desc_el is not None else ""
            # Extract scalar value
            result_el = calc_entity.find(
                ".//aim:calculationDataCollection/aim:CalculationData/aim:value",
                NS,
            )
            if result_el is not None:
                val = float(result_el.get("value", "0"))
                if desc == "Area":
                    area_mm = val
                elif desc == "Mean":
                    mean_intensity = val
                elif desc == "Standard Deviation":
                    std_intensity = val

    result["area_mm"] = area_mm
    result["mean_intensity"] = mean_intensity
    result["std_intensity"] = std_intensity

    # ------------------------------------------------------------------
    # 5. ROI coordinates from markupEntityCollection
    # ------------------------------------------------------------------
    markup_coll = ann.find("aim:markupEntityCollection", NS)
    roi_coords: List[Tuple[float, float]] = []

    if markup_coll is not None:
        markup = markup_coll.find("aim:MarkupEntity", NS)
        if markup is not None:
            coord_coll = markup.find(
                "aim:twoDimensionSpatialCoordinateCollection", NS
            )
            if coord_coll is not None:
                coords = coord_coll.findall(
                    "aim:TwoDimensionSpatialCoordinate", NS
                )
                indexed: List[Tuple[int, float, float]] = []
                for c in coords:
                    idx = int(_find_text(c, "coordinateIndex"))
                    x = float(_find_text(c, "x"))
                    y = float(_find_text(c, "y"))
                    indexed.append((idx, x, y))
                indexed.sort(key=lambda t: t[0])
                roi_coords = [(x, y) for _, x, y in indexed]

    result["roi_coordinates"] = roi_coords

    # Centroid of the ROI
    if roi_coords:
        arr = np.array(roi_coords, dtype=np.float64)
        result["roi_centroid"] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
    else:
        result["roi_centroid"] = (0.0, 0.0)

    return result


# =============================================================================
# Batch Parsing – Load All XML Files for a Case
# =============================================================================

def parse_case_xmls(case_folder: str) -> List[Dict]:
    """
    Parse all AIM XML files in a case folder.

    Args:
        case_folder: Path to "case N" folder under "500 cases with annotation"

    Returns:
        List of parsed annotation dicts (one per XML file / lesion)
    """
    results = []
    case_path = Path(case_folder)
    if not case_path.exists():
        return results

    for xml_file in sorted(case_path.glob("*.xml")):
        parsed = parse_aim_xml(str(xml_file))
        if parsed is not None:
            results.append(parsed)

    return results


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import json

    test_folder = r"C:\Users\jaopi\Desktop\SP\material\500 cases with annotation\case 1"
    annotations = parse_case_xmls(test_folder)

    print(f"Found {len(annotations)} annotations in case 1:\n")
    for a in annotations:
        print(f"  Tooth FDI: {a['tooth_fdi']}")
        print(f"  Tooth Name: {a['tooth_name']}")
        print(f"  Surface: {a['surface_name']}")
        print(f"  Caries Type: {a['caries_type']} → {a['caries_subtype']}")
        print(f"  Severity: {a['severity']}")
        print(f"  Area: {a['area_mm']:.2f} mm")
        print(f"  ROI Points: {len(a['roi_coordinates'])}")
        print(f"  ROI Centroid: ({a['roi_centroid'][0]:.1f}, {a['roi_centroid'][1]:.1f})")
        print()
