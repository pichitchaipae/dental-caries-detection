# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Week 9 — Find & Visualize Presentation Cases
===============================================

Automated pipeline for the Senior Project final presentation:

1. **Search & Filter** — Scan all 500 cases across 5 PCA methods and
   identify 1-3 candidates per storytelling category:
     Cat 1  Hero Cases          (Method 0 weak → Method 1/2/3 strong)
     Cat 2  Regression Cases    (Method 0 strong → Method 5 regressed)
     Cat 3  Baseline Normal     (All methods high & stable)
     Cat 4  Hard Cases          (All methods failed)
     Cat 5  Class-Specific      (Distal / Mesial / Occlusal correctly detected)

2. **Overlay Tooth Numbers** — Use week2 bounding-box PNGs (or fall back
   to ROI + week2 JSON) so the audience can identify each tooth.

3. **Auto-Visualize & Route** — Generate per-method images and organise
   them into ``presentation_export/<category>/method_<N>/``.

Usage
-----
    cd week9_pca_evaluation
    python find_presentation_cases.py
    python find_presentation_cases.py --max_per_cat 2
    python find_presentation_cases.py --dpi 200

Output
------
    week9_pca_evaluation/presentation_export/
    ├── hero_cases/
    │   ├── method_0/  (case_XXX_method_0_viz.png)
    │   ├── method_1/
    │   ├── method_2/
    │   ├── method_3/
    │   └── method_5/
    ├── regression_cases/
    │   ├── method_0/
    │   └── method_5/
    ├── baseline_normal/
    │   ├── method_0/ … method_5/
    ├── hard_cases/
    │   ├── method_0/ … method_5/
    └── class_specific/
        ├── method_0/ … method_5/

Author: Expert Data Scientist — Dental AI / CAD
Date:   2026-02-24
"""

import os
import sys
import json
import math
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
WEEK9_DIR = Path(__file__).resolve().parent
BASE_DIR  = WEEK9_DIR.parent
WEEK2_DIR = BASE_DIR / "week2-Tooth Detection & Segmentation" / "500-segmentation+recognition"
ROI_DIR   = BASE_DIR / "raw_data" / "500-roi"
GT_XML_DIR = BASE_DIR / "raw_data" / "500 cases with annotation(xml)"

EXPORT_DIR = WEEK9_DIR / "presentation_export"

VALID_PCA_METHODS: List[int] = [0, 1, 2, 3, 5]

PCA_METHOD_NAMES = {
    0: "method_0_baseline_opencv",
    1: "method_1_square_heuristic",
    2: "method_2_max_span",
    3: "method_3_split_centroid",
    5: "method_5_vertical_prior",
}

CATEGORY_DIRS = {
    "hero_cases":       "hero_cases",
    "regression_cases": "regression_cases",
    "baseline_normal":  "baseline_normal",
    "hard_cases":       "hard_cases",
    "class_specific":   "class_specific",
}

# Surface → colour map (RGB)
SURFACE_COLOURS = {
    "Distal":       (220,  50,  50),
    "Mesial":       ( 50, 200,  50),
    "Occlusal":     ( 50, 100, 240),
    "Proximal":     (200, 200,  50),
    "Other":        (180, 180, 180),
    "Unclassified": (120, 120, 120),
}

ALPHA_MASK   = 0.35
ALPHA_CARIES = 0.50
AXIS_LEN     = 120

ALLOWED_SURFACES = {"Distal", "Mesial", "Occlusal"}

ALL_CASES = list(range(1, 501))


# =============================================================================
# 1.  Compute per-case F1 from evaluation_results.csv
# =============================================================================

def load_per_case_f1(method: int) -> Dict[int, float]:
    """
    Read ``evaluation_results.csv`` for *method* and compute per-case F1.

    Per-case F1:
        TP  = rows where surface_match_strict == True
        FP  = rows where match_type != 'FN' AND surface_match_strict == False
        FN  = rows where match_type == 'FN'
        F1  = 2*TP / (2*TP + FP + FN)    (0.0 if denominator == 0)
    """
    method_name = PCA_METHOD_NAMES[method]
    csv_path = WEEK9_DIR / method_name / "evaluation_results.csv"
    if not csv_path.exists():
        print(f"  [WARN] Missing evaluation_results.csv for {method_name}")
        return {}

    df = pd.read_csv(csv_path)

    # Normalise boolean column (CSV may store as string)
    df["surface_match_strict"] = df["surface_match_strict"].astype(str).str.strip().str.lower() == "true"

    case_f1: Dict[int, float] = {}

    for case_num, grp in df.groupby("case"):
        tp = int(grp["surface_match_strict"].sum())
        fn = int((grp["match_type"] == "FN").sum())
        fp = int((~grp["surface_match_strict"]) .sum()) - fn   # matched but wrong
        fp = max(fp, 0)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        case_f1[int(case_num)] = round(f1, 4)

    return case_f1


def load_per_case_class_hits(method: int) -> Dict[int, Dict[str, int]]:
    """
    For Cat 5 (class-specific), count strict correct predictions per class
    for each case.  Returns  { case_id: { 'Distal': n, 'Mesial': n, 'Occlusal': n } }
    """
    method_name = PCA_METHOD_NAMES[method]
    csv_path = WEEK9_DIR / method_name / "evaluation_results.csv"
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    df["surface_match_strict"] = df["surface_match_strict"].astype(str).str.strip().str.lower() == "true"

    result: Dict[int, Dict[str, int]] = {}

    for case_num, grp in df.groupby("case"):
        hits: Dict[str, int] = {}
        for cls in ("Distal", "Mesial", "Occlusal"):
            mask = (grp["gt_surface_norm"] == cls) & grp["surface_match_strict"]
            hits[cls] = int(mask.sum())
        result[int(case_num)] = hits

    return result


# =============================================================================
# 2.  Strategic filtering logic (5 categories)
# =============================================================================

def find_candidates(max_per_cat: int = 3) -> Dict[str, List[int]]:
    """
    Return { category_name: [case_id, …] } with up to *max_per_cat* per cat.
    """
    print("\n" + "=" * 70)
    print("STEP 1 — Loading per-case F1 scores for all methods …")
    print("=" * 70)

    f1: Dict[int, Dict[int, float]] = {}  # { method: { case: f1 } }
    class_hits: Dict[int, Dict[int, Dict[str, int]]] = {}

    for m in VALID_PCA_METHODS:
        f1[m] = load_per_case_f1(m)
        class_hits[m] = load_per_case_class_hits(m)
        n = len(f1[m])
        print(f"  Method {m} ({PCA_METHOD_NAMES[m]}): {n} cases loaded")

    # Helper — get F1 for a case across a method, default 0
    def _f1(method: int, case: int) -> float:
        return f1.get(method, {}).get(case, 0.0)

    # ── Cat 1: Hero Cases ────────────────────────────────────────────
    # Method 0 F1 < 0.7, BUT any of Method 1/2/3 F1 >= 0.9
    hero: List[Tuple[int, float]] = []
    for c in ALL_CASES:
        f1_0 = _f1(0, c)
        if f1_0 >= 0.7:
            continue
        best_improved = max(_f1(1, c), _f1(2, c), _f1(3, c))
        if best_improved >= 0.9:
            hero.append((c, best_improved - f1_0))
    hero.sort(key=lambda x: -x[1])  # biggest improvement first
    hero_ids = [c for c, _ in hero[:max_per_cat]]

    # ── Cat 2: Regression Cases ──────────────────────────────────────
    # Method 0 F1 >= 0.9, BUT Method 5 F1 drops significantly (>= 0.3 drop)
    regression: List[Tuple[int, float]] = []
    for c in ALL_CASES:
        f1_0 = _f1(0, c)
        f1_5 = _f1(5, c)
        if f1_0 >= 0.9 and (f1_0 - f1_5) >= 0.3:
            regression.append((c, f1_0 - f1_5))
    regression.sort(key=lambda x: -x[1])  # biggest regression first
    regression_ids = [c for c, _ in regression[:max_per_cat]]

    # ── Cat 3: Baseline Normal (All Stable) ──────────────────────────
    # All methods achieve identical / high F1 (>= 0.85, spread <= 0.1)
    stable: List[Tuple[int, float]] = []
    for c in ALL_CASES:
        scores = [_f1(m, c) for m in VALID_PCA_METHODS]
        if all(s >= 0.85 for s in scores):
            spread = max(scores) - min(scores)
            if spread <= 0.1:
                stable.append((c, sum(scores) / len(scores)))
    stable.sort(key=lambda x: -x[1])  # highest average first
    stable_ids = [c for c, _ in stable[:max_per_cat]]

    # ── Cat 4: Hard Cases (All Failed) ───────────────────────────────
    # All methods F1 < 0.3
    hard: List[Tuple[int, float]] = []
    for c in ALL_CASES:
        scores = [_f1(m, c) for m in VALID_PCA_METHODS]
        if all(s < 0.3 for s in scores):
            hard.append((c, max(scores)))
    hard.sort(key=lambda x: x[1])  # worst first
    hard_ids = [c for c, _ in hard[:max_per_cat]]

    # ── Cat 5: Class-Specific ────────────────────────────────────────
    # Cases that correctly detected "Distal", "Mesial", or "Occlusal"
    # across at least one method with >= 2 correct hits per class
    class_spec: List[Tuple[int, int]] = []
    for c in ALL_CASES:
        total_class_hits = 0
        classes_detected = set()
        for m in VALID_PCA_METHODS:
            h = class_hits.get(m, {}).get(c, {})
            for cls in ("Distal", "Mesial", "Occlusal"):
                cnt = h.get(cls, 0)
                if cnt >= 1:
                    classes_detected.add(cls)
                    total_class_hits += cnt
        # Prefer cases that have all 3 classes correctly detected
        if len(classes_detected) >= 2:
            class_spec.append((c, len(classes_detected) * 100 + total_class_hits))
    class_spec.sort(key=lambda x: -x[1])
    class_spec_ids = [c for c, _ in class_spec[:max_per_cat]]

    # ── Report ───────────────────────────────────────────────────────
    selected = {
        "hero_cases":       hero_ids,
        "regression_cases": regression_ids,
        "baseline_normal":  stable_ids,
        "hard_cases":       hard_ids,
        "class_specific":   class_spec_ids,
    }

    print("\n" + "=" * 70)
    print("STEP 1 RESULTS — Selected Candidates")
    print("=" * 70)
    for cat, ids in selected.items():
        print(f"  {cat:<22s}: {ids}")
        for cid in ids:
            scores_str = "  ".join(
                f"M{m}={_f1(m, cid):.3f}" for m in VALID_PCA_METHODS
            )
            print(f"    Case {cid:>3d}  |  {scores_str}")

    # Warn if any category is empty
    for cat, ids in selected.items():
        if not ids:
            print(f"\n  [WARN] Category '{cat}' has NO candidates! "
                  "Consider relaxing thresholds.")

    return selected


# =============================================================================
# 3.  Ground-truth loader — parse AIM XML files per case
# =============================================================================

# AIM XML namespaces
_AIM_NS = "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"
_ISO_NS = "uri:iso.org:21090"
_NS = {"aim": _AIM_NS, "iso": _ISO_NS}

# Inline SNODENT display-name → FDI mapping (avoids week6 import path issues)
_DISPLAY_NAME_TO_FDI: Dict[str, str] = {
    # Upper Right (Q1)
    "Permanent upper right central incisor tooth":  "11",
    "Permanent upper right lateral incisor tooth":  "12",
    "Permanent upper right canine tooth":           "13",
    "Permanent upper right first premolar tooth":   "14",
    "Permanent upper right second premolar tooth":  "15",
    "Permanent upper right first molar tooth":      "16",
    "Permanent upper right second molar tooth":     "17",
    "Permanent upper right third molar tooth":      "18",
    # Upper Left (Q2)
    "Permanent upper left central incisor tooth":   "21",
    "Permanent upper left lateral incisor tooth":   "22",
    "Permanent upper left canine tooth":            "23",
    "Permanent upper left first premolar tooth":    "24",
    "Permanent upper left second premolar tooth":   "25",
    "Permanent upper left first molar tooth":       "26",
    "Permanent upper left second molar tooth":      "27",
    "Permanent upper left third molar tooth":       "28",
    # Lower Left (Q3)
    "Permanent lower left central incisor tooth":   "31",
    "Permanent lower left lateral incisor tooth":   "32",
    "Permanent lower left canine tooth":            "33",
    "Permanent lower left first premolar tooth":    "34",
    "Permanent lower left second premolar tooth":   "35",
    "Permanent lower left first molar tooth":       "36",
    "Permanent lower left second molar tooth":      "37",
    "Permanent lower left third molar tooth":       "38",
    # Lower Right (Q4)
    "Permanent lower right central incisor tooth":  "41",
    "Permanent lower right lateral incisor tooth":  "42",
    "Permanent lower right canine tooth":           "43",
    "Permanent lower right first premolar tooth":   "44",
    "Permanent lower right second premolar tooth":  "45",
    "Permanent lower right first molar tooth":      "46",
    "Permanent lower right second molar tooth":     "47",
    "Permanent lower right third molar tooth":      "48",
}

# Surface display-name → normalised label
_DISPLAY_NAME_TO_SURFACE: Dict[str, str] = {
    "Occlusal surface": "Occlusal",  "Occlusal Surface": "Occlusal",
    "Distal surface":   "Distal",    "Distal Surface":   "Distal",
    "Mesial surface":   "Mesial",    "Mesial Surface":   "Mesial",
    "Buccal surface":   "Buccal",    "Buccal Surface":   "Buccal",
    "Lingual surface":  "Lingual",   "Lingual Surface":  "Lingual",
    "Palatal surface":  "Palatal",   "Palatal Surface":  "Palatal",
    "Cervical surface": "Cervical",  "Cervical Surface": "Cervical",
}

# SNODENT code → surface (codes from snodent_tooth_map.py)
_SNODENT_SURFACE_MAP: Dict[str, str] = {
    "144414D": "Occlusal", "144474D": "Occlusal",
    "146014D": "Distal",   "146074D": "Distal",
    "145374D": "Mesial",   "145434D": "Mesial",
    "144854D": "Buccal",   "144914D": "Buccal",
    "145134D": "Lingual",  "145194D": "Lingual",
    "145714D": "Palatal",  "145774D": "Cervical",
}

# SNODENT code → FDI (from snodent_tooth_map.py)
_SNODENT_CODE_TO_FDI: Dict[str, str] = {
    "161006D": "11", "160842D": "12", "160288D": "13", "161286D": "14",
    "160450D": "15", "160770D": "16", "161204D": "17", "160618D": "18",
    "160194D": "21", "160132D": "22", "160506D": "23", "161340D": "24",
    "160682D": "25", "161074D": "26", "160386D": "27", "160922D": "28",
    "161136D": "31", "160556D": "32", "160068D": "33", "160326D": "34",
    "161248D": "35", "160730D": "36", "161166D": "37", "160580D": "38",
    "160964D": "41", "160350D": "42", "160894D": "43", "160230D": "44",
    "161412D": "45",                  "161102D": "47", "160488D": "48",
}


def load_ground_truth_from_xml(case_num: int) -> List[Dict]:
    """
    Parse all AIM XML files for a case and return GT annotations.

    Each XML file describes ONE caries lesion.  Returns a list of dicts:
        tooth_fdi       : str   – FDI tooth number ("16", "45", …)
        surface         : str   – normalised surface ("Distal", "Occlusal", …)
        roi_points      : list  – [(x, y), …] polygon in image-pixel space
        bbox            : tuple – (x_min, y_min, x_max, y_max) from polygon
        centroid        : tuple – (cx, cy) of the polygon
    """
    case_folder = GT_XML_DIR / f"case {case_num}"
    if not case_folder.exists():
        return []

    results: List[Dict] = []

    for xml_file in sorted(case_folder.glob("*.xml")):
        try:
            tree = ET.parse(str(xml_file))
        except ET.ParseError:
            continue

        root = tree.getroot()
        annotations = root.find("aim:imageAnnotations", _NS)
        if annotations is None:
            continue
        ann = annotations.find("aim:ImageAnnotation", _NS)
        if ann is None:
            continue

        # ── Tooth FDI + Surface ──────────────────────────────────────
        tooth_fdi = ""
        surface = ""

        phys_coll = ann.find("aim:imagingPhysicalEntityCollection", _NS)
        if phys_coll is not None:
            entity = phys_coll.find("aim:ImagingPhysicalEntity", _NS)
            if entity is not None:
                char_coll = entity.find(
                    "aim:imagingPhysicalEntityCharacteristicCollection", _NS
                )
                if char_coll is not None:
                    for char_el in char_coll.findall(
                        "aim:ImagingPhysicalEntityCharacteristic", _NS
                    ):
                        q_idx_el = char_el.find("aim:questionIndex", _NS)
                        q_idx = q_idx_el.get("value", "") if q_idx_el is not None else ""
                        tc = char_el.find("aim:typeCode", _NS)
                        if tc is None:
                            continue
                        code = tc.get("code", "")
                        dn_el = tc.find("iso:displayName", _NS)
                        display = dn_el.get("value", "") if dn_el is not None else ""

                        if q_idx == "0":
                            # Tooth: try SNODENT code first, then display name
                            tooth_fdi = _SNODENT_CODE_TO_FDI.get(code, "")
                            if not tooth_fdi:
                                tooth_fdi = _DISPLAY_NAME_TO_FDI.get(display, "")
                        elif q_idx == "1":
                            # Surface: try code first, then display name
                            surface = _SNODENT_SURFACE_MAP.get(code, "")
                            if not surface:
                                surface = _DISPLAY_NAME_TO_SURFACE.get(display, display)

        if not tooth_fdi:
            continue  # skip annotations where tooth could not be resolved

        # ── ROI polygon coordinates ──────────────────────────────────
        roi_points: List[Tuple[float, float]] = []
        markup_coll = ann.find("aim:markupEntityCollection", _NS)
        if markup_coll is not None:
            markup = markup_coll.find("aim:MarkupEntity", _NS)
            if markup is not None:
                coord_coll = markup.find(
                    "aim:twoDimensionSpatialCoordinateCollection", _NS
                )
                if coord_coll is not None:
                    indexed: List[Tuple[int, float, float]] = []
                    for c in coord_coll.findall(
                        "aim:TwoDimensionSpatialCoordinate", _NS
                    ):
                        idx_el = c.find("aim:coordinateIndex", _NS)
                        x_el = c.find("aim:x", _NS)
                        y_el = c.find("aim:y", _NS)
                        if idx_el is not None and x_el is not None and y_el is not None:
                            indexed.append((
                                int(idx_el.get("value", "0")),
                                float(x_el.get("value", "0")),
                                float(y_el.get("value", "0")),
                            ))
                    indexed.sort(key=lambda t: t[0])
                    roi_points = [(x, y) for _, x, y in indexed]

        # Compute bounding box & centroid from the polygon
        if roi_points:
            arr = np.array(roi_points, dtype=np.float64)
            x_min, y_min = arr.min(axis=0)
            x_max, y_max = arr.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
            centroid = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
        else:
            bbox = (0, 0, 0, 0)
            centroid = (0.0, 0.0)

        results.append({
            "tooth_fdi":  tooth_fdi,
            "surface":    surface,
            "roi_points": roi_points,
            "bbox":       bbox,
            "centroid":   centroid,
        })

    return results


# =============================================================================
# 4.  Image / data loading helpers
# =============================================================================

def load_base_image(case_num: int) -> Optional[np.ndarray]:
    """Load the ROI dental image (full resolution), return as RGB or None."""
    for ext in ("png", "jpg", "jpeg"):
        img_path = ROI_DIR / f"case_{case_num}.{ext}"
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def load_week2_data(case_num: int) -> Optional[Dict]:
    """Load week2 JSON for tooth bounding boxes / pixel coordinates."""
    json_path = WEEK2_DIR / f"case {case_num}" / f"case_{case_num}_results.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_diagnosis_json(case_num: int, method: int) -> Optional[Dict]:
    """Load the per-method diagnosis JSON from week9 output."""
    method_name = PCA_METHOD_NAMES[method]
    json_path = (
        WEEK9_DIR / method_name / "cases" / f"case {case_num}"
        / f"case_{case_num}_diagnosis.json"
    )
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_csv(method: int) -> pd.DataFrame:
    """
    Load ``evaluation_results.csv`` for *method*.

    This is the authoritative source for:
      - ``mz_predicted_surface``  (corrected surface class)
      - ``rotation_angle_deg``    (method-specific PCA rotation)
    The diagnosis JSON stores week5 originals which may be stale.
    """
    method_name = PCA_METHOD_NAMES[method]
    csv_path = WEEK9_DIR / method_name / "evaluation_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def build_eval_lookup(
    eval_df: pd.DataFrame, case_num: int
) -> Dict[str, Dict]:
    """
    Build ``{ tooth_id: { surface, rotation_angle_deg } }`` for *case_num*
    from the evaluation CSV.  Only includes teeth that matched (TP rows).
    """
    if eval_df.empty:
        return {}
    rows = eval_df[eval_df["case"] == case_num]
    lookup: Dict[str, Dict] = {}
    for _, r in rows.iterrows():
        if r.get("match_type") == "FN":
            continue
        tid = str(r["tooth_id"])
        # Use the multi-zone corrected surface; fall back to pred_surface_norm
        surface = str(r.get("mz_predicted_surface", "")).strip()
        if not surface or surface == "nan":
            surface = str(r.get("pred_surface_norm", "")).strip()
        lookup[tid] = {
            "surface":           surface,
            "rotation_angle_deg": float(r.get("rotation_angle_deg", 0.0)),
        }
    return lookup


# =============================================================================
# 4.  Geometry & drawing helpers  (adapted from debug_visualize_case.py)
# =============================================================================

def _centroid(coords: List[List[int]]) -> Tuple[float, float]:
    if not coords:
        return (0.0, 0.0)
    arr = np.array(coords, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _bbox_from_coords(coords: List[List[int]]) -> Tuple[int, int, int, int]:
    if not coords:
        return (0, 0, 0, 0)
    arr = np.array(coords, dtype=np.int32)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _bbox_from_xyxy(bbox: list) -> Tuple[int, int, int, int]:
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _build_week2_tooth_index(week2_data: Dict) -> Dict[str, Dict]:
    """{ tooth_id: { bbox, pixel_coordinates, centroid } }"""
    idx = {}
    for t in week2_data.get("teeth_data", []):
        tid = str(t.get("tooth_id", ""))
        bbox_raw = t.get("bbox", [])
        px_coords = t.get("pixel_coordinates", [])
        for seg in t.get("segments_detail", []):
            px_coords.extend(seg.get("pixel_coordinates", []))
        cx, cy = _centroid(px_coords) if px_coords else (0, 0)
        if bbox_raw and len(bbox_raw) == 4:
            bbox = _bbox_from_xyxy(bbox_raw)
        elif px_coords:
            bbox = _bbox_from_coords(px_coords)
        else:
            bbox = (0, 0, 0, 0)
        idx[tid] = {
            "bbox": bbox,
            "pixel_coordinates": px_coords,
            "centroid": (cx, cy),
        }
    return idx


def draw_pca_axis(canvas, cx, cy, angle_deg, colour=(0, 255, 255),
                  length=AXIS_LEN, thickness=3):
    """Draw PCA vertical axis (cyan) on RGB canvas."""
    rad = math.radians(angle_deg)
    dx = length * math.sin(rad)
    dy = -length * math.cos(rad)
    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))
    cv2.line(canvas, pt1, pt2, colour, thickness, cv2.LINE_AA)
    cv2.drawMarker(canvas, (int(cx), int(cy)), colour,
                   cv2.MARKER_DIAMOND, 10, thickness, cv2.LINE_AA)


def draw_mask_overlay(canvas, coords, colour, alpha=ALPHA_CARIES):
    """Semi-transparent pixel overlay."""
    if not coords:
        return
    overlay = canvas.copy()
    for x, y in coords:
        if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
            overlay[y, x] = colour
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_bbox(canvas, x1, y1, x2, y2, colour=(255, 255, 255),
              thickness=2, label=""):
    """Rectangle + optional label."""
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    if label:
        font_scale = 0.55
        font_thick = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thick)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1),
                      colour, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thick, cv2.LINE_AA)


def draw_mapping_line(canvas, cx_from, cy_from, cx_to, cy_to,
                      colour=(255, 0, 255), thickness=1):
    cv2.line(canvas, (int(cx_from), int(cy_from)),
             (int(cx_to), int(cy_to)), colour, thickness, cv2.LINE_AA)


def draw_gt_bbox(canvas, x1, y1, x2, y2, colour=(0, 230, 255),
                 thickness=3, label=""):
    """Ground-truth bounding box — thick yellow outline (dashed effect)."""
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    if label:
        font_scale = 0.55
        font_thick = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thick)
        # Label below the box to avoid colliding with prediction labels
        label_y = y2 + th + 8
        cv2.rectangle(canvas, (x1, y2), (x1 + tw + 6, label_y + 4),
                      colour, -1)
        cv2.putText(canvas, label, (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thick, cv2.LINE_AA)


def draw_gt_polygon(canvas, roi_points, colour=(0, 230, 255), alpha=0.20):
    """Semi-transparent yellow polygon fill for GT ROI."""
    if not roi_points or len(roi_points) < 3:
        return
    pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_tooth_numbers_from_week2(canvas: np.ndarray,
                                  week2_index: Dict[str, Dict]):
    """
    Draw prominent tooth number labels + white bounding boxes on *canvas*.
    This is used when the week2 bounding-box PNG is unavailable.
    """
    for tid, info in week2_index.items():
        bbox = info["bbox"]
        if bbox == (0, 0, 0, 0):
            continue
        x1, y1, x2, y2 = bbox
        # White bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      (255, 255, 255), 2, cv2.LINE_AA)
        # Prominent tooth number label (white bg + black text)
        label = f"T{tid}"
        font_scale = 0.7
        font_thick = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thick)
        lx = x1
        ly = max(y1 - 4, th + 10)
        cv2.rectangle(canvas, (lx, ly - th - 6), (lx + tw + 8, ly + 4),
                      (255, 255, 255), -1)
        cv2.putText(canvas, label, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thick, cv2.LINE_AA)


# =============================================================================
# 5.  Render a single-method panel for one case
# =============================================================================

def render_case_method_image(
    case_num: int,
    method: int,
    base_img: np.ndarray,
    diag: Dict,
    week2_index: Dict[str, Dict],
    eval_lookup: Dict[str, Dict],
    gt_annotations: Optional[List[Dict]] = None,
) -> np.ndarray:
    """
    Render all diagnostic layers for ONE case + ONE method.

    IMPORTANT — data sources:
      * Caries pixel coordinates   → diagnosis JSON  (global / ROI coords)
      * Tooth bboxes & centroids   → week2 JSON index (global / ROI coords)
      * Surface class (corrected)  → evaluation_results.csv via *eval_lookup*
      * PCA rotation angle (method-specific) → eval CSV via *eval_lookup*
      * Ground truth annotations   → AIM XML files via *gt_annotations*

    Layers (drawing order, back → front):
      0. GT polygon fill           — yellow, 20 % opacity
      0b.GT bounding box           — thick yellow, with "[GT] Surface" label
      1. Tooth segmentation mask   — green, 20 % opacity
      2. Tooth bounding boxes      — white, with prominent "T{id}" labels
      3. Caries mask overlay       — surface-coloured, 50 % opacity
      4. Caries bounding box       — red, with "[Pred] Surface" label
      5. Caries → Tooth mapping    — magenta line
      6. PCA vertical axis         — cyan line centred on tooth bbox centre
      7. Banner + legend
    """
    canvas = base_img.copy()

    # Determine image dimensions for coordinate-space checks
    img_h, img_w = canvas.shape[:2]

    # ── Layer 0: Ground Truth annotations (yellow) ───────────────────
    gt_colour = (0, 230, 255)   # yellow in RGB
    if gt_annotations:
        for gt in gt_annotations:
            gt_bbox = gt.get("bbox", (0, 0, 0, 0))
            gt_surface = gt.get("surface", "")
            gt_tooth = gt.get("tooth_fdi", "")
            roi_pts = gt.get("roi_points", [])

            # Scale coordinates if they fall outside the image bounds
            # (some XMLs may be annotated at a different resolution)
            if roi_pts:
                arr = np.array(roi_pts, dtype=np.float64)
                max_x, max_y = arr[:, 0].max(), arr[:, 1].max()
                sx = img_w / max_x if max_x > img_w else 1.0
                sy = img_h / max_y if max_y > img_h else 1.0
                if sx != 1.0 or sy != 1.0:
                    scale = min(sx, sy)
                    roi_pts = [(x * scale, y * scale) for x, y in roi_pts]
                    arr = np.array(roi_pts, dtype=np.float64)
                    x_min, y_min = arr.min(axis=0)
                    x_max, y_max = arr.max(axis=0)
                    gt_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

            if gt_bbox != (0, 0, 0, 0):
                # Semi-transparent polygon fill
                draw_gt_polygon(canvas, roi_pts, colour=gt_colour, alpha=0.20)
                # Thick yellow bounding box
                gt_label = f"[GT] {gt_surface}" if gt_surface else f"[GT] T{gt_tooth}"
                draw_gt_bbox(canvas, *gt_bbox, colour=gt_colour,
                             thickness=3, label=gt_label)

    # ── Layer 1 + 2: Tooth masks (green 20%) + bounding boxes ────────
    draw_tooth_numbers_from_week2(canvas, week2_index)

    for tooth in diag.get("teeth_data", []):
        tid = str(tooth.get("tooth_id", ""))
        has_caries = tooth.get("has_caries", False)
        caries_coords = tooth.get("caries_coordinates", [])

        w2 = week2_index.get(tid, {})
        tooth_px     = w2.get("pixel_coordinates", [])
        tooth_bbox   = w2.get("bbox", (0, 0, 0, 0))

        # ── Layer 1: Tooth mask (semi-transparent green, 20%) ────────
        if tooth_px:
            draw_mask_overlay(canvas, tooth_px, (50, 200, 50), alpha=0.20)

        if not has_caries or not caries_coords:
            continue

        # ── Resolve corrected surface & rotation from eval CSV ───────
        ev = eval_lookup.get(tid, {})
        surface = ev.get("surface", "")
        rot_deg = ev.get("rotation_angle_deg", 0.0)

        # Task 2 STRICT RULE: only Distal / Mesial / Occlusal
        if surface not in ALLOWED_SURFACES:
            continue

        surf_colour = SURFACE_COLOURS[surface]

        # ── Layer 3: Caries mask overlay (50 % opacity) ─────────────
        draw_mask_overlay(canvas, caries_coords, (220, 50, 50), alpha=ALPHA_CARIES)

        # ── Layer 4: Caries bounding box (red) + "[Pred]" label ─────
        c_bbox = _bbox_from_coords(caries_coords)
        draw_bbox(canvas, *c_bbox, colour=(255, 60, 60),
                  thickness=2, label=f"[Pred] {surface}")

        # ── Layer 5: Caries → Tooth mapping line (magenta) ──────────
        # Compute tooth bbox centre (guaranteed global coords)
        if tooth_bbox != (0, 0, 0, 0):
            tx1, ty1, tx2, ty2 = tooth_bbox
            tooth_cx = (tx1 + tx2) / 2.0
            tooth_cy = (ty1 + ty2) / 2.0
        else:
            tooth_cx, tooth_cy = w2.get("centroid", (0, 0))

        caries_cx, caries_cy = _centroid(caries_coords)

        if tooth_cx > 0 and tooth_cy > 0:
            draw_mapping_line(canvas, caries_cx, caries_cy,
                              tooth_cx, tooth_cy)

        # ── Layer 6: PCA Axis (cyan) — centred on tooth bbox centre ─
        #    The rotation_angle_deg is method-specific (from eval CSV).
        #    We draw the PCA axis at the tooth bbox centre which is in
        #    global (panoramic / ROI) coordinates — no offset needed.
        if tooth_cx > 0 and tooth_cy > 0:
            draw_pca_axis(canvas, tooth_cx, tooth_cy, rot_deg)

    # ── Layer 7a: Banner — method name ───────────────────────────────
    h, w = canvas.shape[:2]
    method_name = PCA_METHOD_NAMES[method]
    gt_count = len(gt_annotations) if gt_annotations else 0
    banner_text = f"Case {case_num} | Method {method}: {method_name}  (GT: {gt_count} lesions)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.7, 2
    (tw, th), _ = cv2.getTextSize(banner_text, font, fs, ft)
    cv2.rectangle(canvas, (0, 0), (tw + 20, th + 20), (40, 40, 40), -1)
    cv2.putText(canvas, banner_text, (10, th + 10), font, fs,
                (255, 255, 255), ft, cv2.LINE_AA)

    # ── Layer 7b: Legend strip at bottom ─────────────────────────────
    legend_items = [
        ("[GT]",         (0, 230, 255)),
        ("[Pred]",       (255, 60, 60)),
        ("PCA Axis",     (0, 255, 255)),
        ("Caries-Tooth", (255, 0, 255)),
        ("Tooth Mask",   (50, 200, 50)),
        ("Distal",       SURFACE_COLOURS["Distal"]),
        ("Mesial",       SURFACE_COLOURS["Mesial"]),
        ("Occlusal",     SURFACE_COLOURS["Occlusal"]),
    ]
    # Dark background bar for legend
    bar_h = 28
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    lx = 10
    ly = h - 8
    for name, col in legend_items:
        cv2.rectangle(canvas, (lx, ly - 14), (lx + 14, ly), col, -1)
        cv2.putText(canvas, name, (lx + 18, ly), font, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        lx += 18 + len(name) * 8 + 15

    return canvas


# =============================================================================
# 6.  Export pipeline
# =============================================================================

def generate_exports(
    selected: Dict[str, List[int]],
    dpi: int = 150,
):
    """
    For every (category, case_id) pair, render one image per PCA method
    and save it into the structured export folder.
    """
    print("\n" + "=" * 70)
    print("STEP 2 — Generating Presentation Visualizations …")
    print("=" * 70)

    total_images = 0
    summary_rows: List[Dict] = []

    for cat_key, case_ids in selected.items():
        if not case_ids:
            continue

        cat_dir_name = CATEGORY_DIRS[cat_key]
        print(f"\n  ── Category: {cat_key} ({len(case_ids)} cases) ──")

        for case_num in case_ids:
            print(f"    Case {case_num}:")

            # Always use full-resolution ROI as base image.
            # The week2 bounding-box PNG is half-resolution and causes
            # a coordinate mismatch with all JSON pixel coordinates.
            base_img = load_base_image(case_num)
            if base_img is None:
                print(f"      [SKIP] No ROI image found for case {case_num}")
                continue

            print(f"      [OK] ROI loaded ({base_img.shape[1]}x{base_img.shape[0]})")

            # Load week2 tooth data for bboxes / masks / tooth numbers
            week2_data = load_week2_data(case_num)
            week2_index = _build_week2_tooth_index(week2_data) if week2_data else {}

            if not week2_index:
                print(f"      [WARN] No week2 tooth data — tooth numbers will be missing")

            # Load ground-truth annotations from AIM XML (once per case)
            gt_annotations = load_ground_truth_from_xml(case_num)
            if gt_annotations:
                gt_surfs = [g["surface"] for g in gt_annotations if g["surface"]]
                print(f"      [GT]  {len(gt_annotations)} XML lesion(s): {gt_surfs}")
            else:
                print(f"      [GT]  No ground-truth XML found")

            # Generate one image per method
            for method in VALID_PCA_METHODS:
                diag = load_diagnosis_json(case_num, method)
                if diag is None:
                    print(f"      [SKIP] Method {method}: no diagnosis JSON")
                    continue

                # Load per-method eval CSV for correct surfaces & rotation
                eval_df = load_eval_csv(method)
                eval_lookup = build_eval_lookup(eval_df, case_num)

                # Render
                canvas = render_case_method_image(
                    case_num, method, base_img, diag,
                    week2_index, eval_lookup,
                    gt_annotations=gt_annotations,
                )

                # Save
                out_dir = EXPORT_DIR / cat_dir_name / f"method_{method}"
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"case_{case_num}_method_{method}_viz.png"
                out_path = out_dir / out_name

                # Convert RGB → BGR for cv2.imwrite
                cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                total_images += 1

                summary_rows.append({
                    "category":    cat_key,
                    "case_id":     case_num,
                    "method":      method,
                    "method_name": PCA_METHOD_NAMES[method],
                    "output_path": str(out_path.relative_to(WEEK9_DIR)),
                })

            print(f"      [OK] {len(VALID_PCA_METHODS)} method images saved")

    # ── Summary CSV ──────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = EXPORT_DIR / "export_manifest.csv"
        os.makedirs(EXPORT_DIR, exist_ok=True)
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n  [Saved] Export manifest → {summary_path}")

    return total_images


# =============================================================================
# 7.  Print final report
# =============================================================================

def print_final_report(selected: Dict[str, List[int]], total_images: int):
    """Pretty-print the output tree and summary."""
    print("\n" + "=" * 70)
    print("STEP 3 — Final Report")
    print("=" * 70)
    print(f"  Total images generated : {total_images}")
    print(f"  Export directory        : {EXPORT_DIR}")
    print()

    # Walk the export directory tree
    if EXPORT_DIR.exists():
        print("  Directory structure:")
        for cat_dir in sorted(EXPORT_DIR.iterdir()):
            if cat_dir.is_dir():
                print(f"    ├── {cat_dir.name}/")
                method_dirs = sorted(cat_dir.iterdir())
                for i, md in enumerate(method_dirs):
                    if md.is_dir():
                        files = sorted(md.iterdir())
                        connector = "│   ├──" if i < len(method_dirs) - 1 else "│   └──"
                        print(f"    {connector} {md.name}/  ({len(files)} files)")
                        for f in files:
                            print(f"    │   │   └── {f.name}")
            elif cat_dir.is_file():
                print(f"    ├── {cat_dir.name}")

    print()
    print("  Selected cases per category:")
    for cat, ids in selected.items():
        print(f"    {cat:<22s}: {ids if ids else '(none)'}")

    print("\n" + "=" * 70)
    print("  Pipeline complete.  Ready for presentation! 🎓")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Week 9 — Find & Visualize Strategic Presentation Cases"
    )
    parser.add_argument(
        "--max_per_cat", type=int, default=3,
        help="Maximum cases per storytelling category (default: 3).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI (default: 150).",
    )
    args = parser.parse_args()

    # Step 1: Find candidates
    selected = find_candidates(max_per_cat=args.max_per_cat)

    # Step 2: Generate visualizations
    total_images = generate_exports(selected, dpi=args.dpi)

    # Step 3: Report
    print_final_report(selected, total_images)


if __name__ == "__main__":
    main()
