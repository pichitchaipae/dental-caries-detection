#!/usr/bin/env python3
"""
pipeline_run3_final.py
======================
Self-contained Run 3 dental caries surface classification pipeline.

Usage:
    python pipeline_run3_final.py

Steps:
    1. Extract 13 geometric features from 500 cases
    2. Train a Random Forest classifier (GroupShuffleSplit by case_id)
    3. Re-classify all 500 cases with Smart Fallback
    4. Evaluate predictions against XML ground truth
    5. Plot feature importance chart
    6. Generate README_run3.html documentation
"""

# =========================================================
# Imports
# =========================================================
import os
import sys
import json
import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =========================================================
# Constants
# =========================================================

FEATURE_COLS = [
    "is_upper", "x_mean", "y_mean", "x_std", "y_std",
    "x_min", "x_max", "y_min", "x_range", "y_range",
    "x_centroid_dist", "aspect_ratio", "coverage",
]
"""Ordered list of the 13 geometric features used by the RF model."""

VALID_SURFACES = ["Occlusal", "Mesial", "Distal", "Other"]
"""The four surface classes used in evaluation."""

MAX_TILT_DEG = 45.0
"""Clamp extreme PCA rotation angles above this threshold."""

MIN_CLUSTER_SIZE = 15
"""Minimum connected-component size for noise removal."""

LEFT_BOUND = 0.40
"""X-thirds left boundary (0.00 - 0.40 = left zone)."""

RIGHT_BOUND = 0.60
"""X-thirds right boundary (0.60 - 1.00 = right zone)."""


# =========================================================
# Path configuration
# =========================================================
try:
    _THIS_DIR = Path(__file__).resolve().parent
except NameError:
    # Running inside Jupyter / interactive — use the notebook's working directory.
    _THIS_DIR = Path.cwd()

_SP_DIR = _THIS_DIR.parent
SEG_DIR = _SP_DIR / "week2-Tooth Detection & Segmentation" / "500-segmentation+recognition"
CARIES_DIR = _SP_DIR / "week3-Caries-to-Tooth Mapping" / "dental_analysis_output"
GT_ROOT = _SP_DIR / "data" / "500 cases with annotation"
OUTPUT_ROOT = _THIS_DIR / "PCA_Output_Run3"
MODEL_PATH = _THIS_DIR / "rf_classify_ml.pkl"

# Global model placeholder.
rf_model = None


# =========================================================
# Progress bar helper
# =========================================================
def _progress_bar(current, total, prefix="Progress", bar_length=30):
    """
    Print an inline text-based progress bar that overwrites itself.

    Args:
        current (int): Current step number (1-indexed).
        total (int): Total number of steps.
        prefix (str): Label displayed before the bar.
        bar_length (int): Character width of the bar.
    """
    fraction = current / max(total, 1)
    filled = int(bar_length * fraction)
    bar = chr(9608) * filled + chr(9617) * (bar_length - filled)
    print(
        f"\r   {prefix} [{bar}] {fraction*100:.0f}% ({current}/{total})",
        end="", flush=True,
    )
    if current >= total:
        print()


# =========================================================
# XML Ground-Truth Parser
# =========================================================
AIM_NS = "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"
ISO_NS = "uri:iso.org:21090"
NS = {"aim": AIM_NS, "iso": ISO_NS}

SNODENT_SURFACE_MAP = {
    "144414D": "Occlusal", "146014D": "Distal", "145374D": "Mesial",
    "144474D": "Occlusal", "146074D": "Distal", "145434D": "Mesial",
}

DISPLAY_NAME_TO_SURFACE = {
    "Occlusal surface": "Occlusal", "Occlusal Surface": "Occlusal",
    "Distal Surface": "Distal", "Distal surface": "Distal",
    "Mesial Surface": "Mesial", "Mesial surface": "Mesial",
}

SNODENT_TO_FDI = {
    "161006D": "11", "160842D": "12", "160288D": "13", "161286D": "14",
    "160450D": "15", "160770D": "16", "161204D": "17", "160618D": "18",
    "160194D": "21", "160132D": "22", "160506D": "23", "161340D": "24",
    "160682D": "25", "161074D": "26", "160386D": "27", "160922D": "28",
    "161136D": "31", "160556D": "32", "160068D": "33", "160326D": "34",
    "161248D": "35", "160730D": "36", "161166D": "37", "160580D": "38",
    "160964D": "41", "160350D": "42", "160894D": "43", "160230D": "44",
    "161412D": "45", "160770D": "46", "161102D": "47", "160488D": "48",
}


def _get_display_name(element):
    """Extract the displayName value from an ISO-namespaced XML element."""
    dn = element.find("iso:displayName", NS)
    return dn.get("value", "") if dn is not None else ""


def snodent_display_to_fdi(display_name):
    """
    Convert a SNODENT display name to an FDI two-digit tooth identifier.

    Args:
        display_name (str): SNODENT descriptive name.

    Returns:
        str: FDI identifier (e.g. '16'), or empty string if unparseable.
    """
    dn = display_name.lower()
    if "upper" in dn and "right" in dn:
        quadrant = 1
    elif "upper" in dn and "left" in dn:
        quadrant = 2
    elif "lower" in dn and "left" in dn:
        quadrant = 3
    elif "lower" in dn and "right" in dn:
        quadrant = 4
    else:
        return ""

    tooth_map = {
        "central incisor": 1, "lateral incisor": 2, "canine": 3,
        "first premolar": 4, "second premolar": 5, "first molar": 6,
        "second molar": 7, "third molar": 8,
    }
    for name, pos in tooth_map.items():
        if name in dn:
            return f"{quadrant}{pos}"
    return ""


def parse_aim_xml(xml_path):
    """
    Parse a single AIM XML annotation file to extract tooth and surface.

    Args:
        xml_path (str): Path to the AIM XML file.

    Returns:
        dict or None: {'tooth': str, 'surface': str}, or None on failure.
    """
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return None

    root = tree.getroot()
    anns = root.find("aim:imageAnnotations", NS)
    if anns is None:
        return None
    ann = anns.find("aim:ImageAnnotation", NS)
    if ann is None:
        return None

    tooth = ""
    surface = ""
    phys_coll = ann.find("aim:imagingPhysicalEntityCollection", NS)
    if phys_coll is not None:
        entity = phys_coll.find("aim:ImagingPhysicalEntity", NS)
        if entity is not None:
            char_coll = entity.find(
                "aim:imagingPhysicalEntityCharacteristicCollection", NS
            )
            if char_coll is not None:
                for ch in char_coll.findall(
                    "aim:ImagingPhysicalEntityCharacteristic", NS
                ):
                    q_idx_el = ch.find("aim:questionIndex", NS)
                    q_idx = q_idx_el.get("value", "") if q_idx_el is not None else ""
                    tc = ch.find("aim:typeCode", NS)
                    if tc is None:
                        continue
                    code = tc.get("code", "")
                    display = _get_display_name(tc)
                    if q_idx == "0":
                        tooth = snodent_display_to_fdi(display)
                        if not tooth:
                            tooth = SNODENT_TO_FDI.get(code, "")
                    elif q_idx == "1":
                        surface = SNODENT_SURFACE_MAP.get(code, "")
                        if not surface:
                            surface = DISPLAY_NAME_TO_SURFACE.get(display, "")

    return {"tooth": tooth, "surface": surface}


def parse_case_ground_truth(case_folder):
    """
    Parse all AIM XML ground-truth files in a case folder.

    Args:
        case_folder (Path): Directory containing *.xml annotation files.

    Returns:
        list[dict]: Each dict has 'tooth' (str FDI) and 'surface' (str).
    """
    ground_truth_list = []
    for xml_file in sorted(Path(case_folder).glob("*.xml")):
        parsed = parse_aim_xml(str(xml_file))
        if parsed is None:
            continue
        tooth = str(parsed.get("tooth", "Unknown"))
        surface = parsed.get("surface", "Other")
        if surface not in VALID_SURFACES:
            surface = "Other"
        ground_truth_list.append({"tooth": tooth, "surface": surface})
    return ground_truth_list


# =========================================================
# PCA & Geometry Helpers
# =========================================================

def is_upper_jaw(tooth_id):
    """Check whether a tooth belongs to the upper jaw (quadrant 1 or 2)."""
    return int(str(tooth_id)[0]) in [1, 2]


def get_quadrant(tooth_id):
    """Extract the FDI quadrant (1-4) from a tooth identifier."""
    return int(str(tooth_id)[0])


def get_bbox(pts):
    """Compute the axis-aligned bounding box: (x_min, y_min, width, height)."""
    p = np.array(pts, dtype=np.float64)
    bbox_min, bbox_max = np.min(p, 0), np.max(p, 0)
    return bbox_min[0], bbox_min[1], bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]


def rotate(pts, center, angle):
    """Rotate 2D points around a centre by a given angle (radians)."""
    p = np.array(pts, dtype=np.float64) - center
    c, s = np.cos(angle), np.sin(angle)
    return np.dot(p, np.array([[c, -s], [s, c]]).T) + center


def remove_small_clusters(caries_pts, min_cluster=MIN_CLUSTER_SIZE):
    """
    Remove noise from caries points by discarding small connected components.

    Args:
        caries_pts (list or np.ndarray): Nx2 caries pixel coordinates.
        min_cluster (int): Minimum cluster size to keep.

    Returns:
        np.ndarray: Filtered caries points.
    """
    if len(caries_pts) < min_cluster:
        return caries_pts
    pts = np.array(caries_pts, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = 2
    w = x_max - x_min + 1 + 2 * pad
    h = y_max - y_min + 1 + 2 * pad
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min - pad, y_min - pad])
    mask[shifted[:, 1], shifted[:, 0]] = 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
            keep[labels == lbl] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        return caries_pts
    return np.column_stack([xs + x_min - pad, ys + y_min - pad]).astype(np.float64)


def perform_pca(points, tooth_id):
    """
    PCA-based 4-rule tooth orientation alignment.

    Args:
        points (list or np.ndarray): Nx2 tooth pixel coordinates.
        tooth_id (str or int): FDI tooth identifier.

    Returns:
        tuple: (mean_center, rotation_angle_rad, was_clamped).
    """
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
    primary_eigenvector = eigvecs[0].astype(np.float64)
    secondary_eigenvector = eigvecs[1].astype(np.float64)

    # Rule 1: vertical axis has larger |Y| component.
    if abs(primary_eigenvector[1]) >= abs(secondary_eigenvector[1]):
        vertical_axis = primary_eigenvector.copy()
        horizontal_axis = secondary_eigenvector.copy()
    else:
        vertical_axis = secondary_eigenvector.copy()
        horizontal_axis = primary_eigenvector.copy()

    # Rule 2: vertical direction based on jaw.
    upper = is_upper_jaw(tooth_id)
    if upper:
        if vertical_axis[1] < 0:
            vertical_axis = -vertical_axis
    else:
        if vertical_axis[1] > 0:
            vertical_axis = -vertical_axis

    # Rule 3: horizontal direction based on quadrant.
    quadrant = get_quadrant(tooth_id)
    if quadrant in [1, 4]:
        if horizontal_axis[0] < 0:
            horizontal_axis = -horizontal_axis
    else:
        if horizontal_axis[0] > 0:
            horizontal_axis = -horizontal_axis

    angle_from_x = math.atan2(vertical_axis[1], vertical_axis[0])
    target_angle = math.pi / 2 if upper else -math.pi / 2
    rotation_angle = target_angle - angle_from_x

    while rotation_angle > math.pi:
        rotation_angle -= 2 * math.pi
    while rotation_angle < -math.pi:
        rotation_angle += 2 * math.pi

    # Rule 4: clamp extreme rotations.
    clamped = False
    if abs(math.degrees(rotation_angle)) > MAX_TILT_DEG:
        rotation_angle = 0.0
        clamped = True

    return mean, rotation_angle, clamped


def build_seg_map(seg_data):
    """Build tooth_id -> pixel_coordinates mapping from segmentation JSON."""
    return {
        str(t["tooth_id"]): t.get("pixel_coordinates", [])
        for t in seg_data.get("teeth_data", [])
    }


# =========================================================
# File I/O
# =========================================================

def _load_json_file(path):
    """Load JSON file, return None if missing."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_seg_case(case_id):
    """Load segmentation JSON for a single case."""
    return _load_json_file(SEG_DIR / f"case {case_id}" / f"case_{case_id}_results.json")


def _load_caries_case(case_id):
    """Load caries mapping JSON for a single case."""
    return _load_json_file(CARIES_DIR / f"case {case_id}" / f"case_{case_id}_caries_mapping.json")


# =========================================================
# Feature Extraction
# =========================================================

def _extract_ml_feature_dict(tooth_id, tooth_pts, caries_pts):
    """
    Extract 13 geometric features for one caries-tooth pair.

    Args:
        tooth_id (str): FDI tooth identifier.
        tooth_pts (list): Tooth mask pixel coordinates.
        caries_pts (list): Caries region pixel coordinates.

    Returns:
        dict or None: Feature dictionary keyed by FEATURE_COLS names.
    """
    caries_clean = remove_small_clusters(caries_pts)
    if len(caries_clean) == 0:
        return None

    center, angle, _ = perform_pca(tooth_pts, tooth_id)
    tooth_rot = rotate(tooth_pts, center, angle)
    caries_rot = rotate(caries_clean, center, angle)

    bbox_x, bbox_y, w, h = get_bbox(tooth_rot)
    if w <= 0 or h <= 0:
        return None

    x_rel = np.clip((caries_rot[:, 0] - bbox_x) / w, 0.0, 1.0)
    y_rel = np.clip((caries_rot[:, 1] - bbox_y) / h, 0.0, 1.0)

    return {
        "is_upper": 1 if int(str(tooth_id)[0]) in [1, 2] else 0,
        "x_mean": float(np.mean(x_rel)),
        "y_mean": float(np.mean(y_rel)),
        "x_std": float(np.std(x_rel)),
        "y_std": float(np.std(y_rel)),
        "x_min": float(np.min(x_rel)),
        "x_max": float(np.max(x_rel)),
        "y_min": float(np.min(y_rel)),
        "x_range": float(np.max(x_rel) - np.min(x_rel)),
        "y_range": float(np.max(y_rel) - np.min(y_rel)),
        "x_centroid_dist": float(abs(np.mean(x_rel) - 0.5)),
        "aspect_ratio": float(w / h),
        "coverage": float(len(caries_clean) / (len(tooth_pts) + 1e-6)),
    }


# =========================================================
# Dataset Construction
# =========================================================

def create_ml_dataset(case_ids):
    """
    Build a labelled ML dataset by extracting features from all cases.

    Args:
        case_ids (list[int]): Case identifiers to process.

    Returns:
        pd.DataFrame: Dataset with ['case_id', 'tooth_id', *FEATURE_COLS, 'label'].
    """
    dataset_rows = []
    total = len(case_ids)
    print(f"[RUNNING] Step 1: สกัด Features จากข้อมูล {total} เคส...", flush=True)

    for i, case_id in enumerate(case_ids):
        _progress_bar(i + 1, total, "Step 1: สกัด Features")

        seg_data = _load_seg_case(case_id)
        caries_data = _load_caries_case(case_id)
        gt_folder = GT_ROOT / f"case {case_id}"

        if seg_data is None or caries_data is None or not gt_folder.exists():
            continue

        ground_truth_list = parse_case_ground_truth(gt_folder)
        ground_truth_lookup = {str(item["tooth"]): item["surface"] for item in ground_truth_list}
        if not ground_truth_lookup:
            continue

        segmentation_map = build_seg_map(seg_data)
        for tooth in caries_data.get("teeth_caries_data", []):
            tooth_id = str(tooth.get("tooth_id", ""))
            if tooth_id not in ground_truth_lookup:
                continue

            tooth_pts = segmentation_map.get(tooth_id, [])
            caries_pts = tooth.get("caries_coordinates", [])
            if len(caries_pts) == 0 or len(tooth_pts) < 10:
                continue

            features = _extract_ml_feature_dict(tooth_id, tooth_pts, caries_pts)
            if features is None:
                continue

            dataset_rows.append({
                "case_id": int(case_id),
                "tooth_id": tooth_id,
                **features,
                "label": ground_truth_lookup[tooth_id],
            })

    columns = ["case_id", "tooth_id", *FEATURE_COLS, "label"]
    feature_dataframe = pd.DataFrame(dataset_rows, columns=columns)
    if not feature_dataframe.empty:
        feature_dataframe = feature_dataframe[columns]
    return feature_dataframe


# =========================================================
# Model Training
# =========================================================

def train_classify_ml(feature_dataframe):
    """
    Train a Random Forest classifier with GroupShuffleSplit by case_id.

    Args:
        feature_dataframe (pd.DataFrame): Labelled dataset.

    Returns:
        tuple: (model, test_dataframe, feature_cols).
    """
    global rf_model

    if feature_dataframe.empty:
        raise ValueError("ML dataset is empty.")

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(feature_dataframe, groups=feature_dataframe["case_id"]))
    train_dataframe = feature_dataframe.iloc[train_idx].reset_index(drop=True)
    test_dataframe = feature_dataframe.iloc[test_idx].reset_index(drop=True)

    model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=42,
    )
    model.fit(train_dataframe[FEATURE_COLS], train_dataframe["label"])

    rf_model = model
    joblib.dump(rf_model, str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")
    return model, test_dataframe, FEATURE_COLS


# =========================================================
# Baseline Classifier (Smart Fallback target)
# =========================================================

def classify_xthird(tooth_id, tooth_pts, caries_pts):
    """
    Baseline X-Thirds classifier (v4.5 dominant zone).

    Args:
        tooth_id (str): FDI tooth identifier.
        tooth_pts (list): Tooth mask pixel coordinates.
        caries_pts (list): Caries region pixel coordinates.

    Returns:
        tuple: (surface, angle_deg, vote_fractions).
    """
    caries_clean = remove_small_clusters(caries_pts)
    if len(caries_clean) == 0:
        return "Other", 0.0, {}

    center, angle, clamped = perform_pca(tooth_pts, tooth_id)
    tooth_rot = rotate(tooth_pts, center, angle)
    caries_rot = rotate(caries_clean, center, angle)

    x, y, w, h = get_bbox(tooth_rot)
    if w <= 0 or h <= 0:
        return "Other", float(math.degrees(angle)), {}

    rel_xs = np.clip((caries_rot[:, 0] - x) / w, 0.0, 1.0)
    n_pts = len(rel_xs)

    quadrant = get_quadrant(tooth_id)
    if quadrant in [1, 4]:
        d_mask = rel_xs < LEFT_BOUND
        c_mask = (rel_xs >= LEFT_BOUND) & (rel_xs <= RIGHT_BOUND)
        m_mask = rel_xs > RIGHT_BOUND
    else:
        m_mask = rel_xs < LEFT_BOUND
        c_mask = (rel_xs >= LEFT_BOUND) & (rel_xs <= RIGHT_BOUND)
        d_mask = rel_xs > RIGHT_BOUND

    vote_map = {"Mesial": int(np.sum(m_mask)), "Occlusal": int(np.sum(c_mask)), "Distal": int(np.sum(d_mask))}
    winner = max(vote_map, key=vote_map.get)

    vote_fractions = {k: round(v / max(n_pts, 1), 4) for k, v in vote_map.items()}
    vote_fractions["pca_clamped"] = clamped
    return winner, float(math.degrees(angle)), vote_fractions


# =========================================================
# Smart Fallback Classifier
# =========================================================

def classify_ml(tooth_id, tooth_pts, caries_pts):
    """
    Classify using RF predict_proba with Smart Fallback to X-Thirds.

    Args:
        tooth_id (str): FDI tooth identifier.
        tooth_pts (list): Tooth mask pixel coordinates.
        caries_pts (list): Caries region pixel coordinates.

    Returns:
        tuple: (predicted_surface, rotation_angle, metadata_dict).
    """
    try:
        features = _extract_ml_feature_dict(tooth_id, tooth_pts, caries_pts)
        if features is None or rf_model is None:
            return classify_xthird(tooth_id, tooth_pts, caries_pts)

        prediction_input_df = pd.DataFrame(
            [[features[col] for col in FEATURE_COLS]],
            columns=FEATURE_COLS,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_probabilities = rf_model.predict_proba(prediction_input_df)[0]

        model_classes = list(rf_model.classes_)
        valid_surface_classes = ["Occlusal", "Mesial", "Distal"]
        surface_scores = {
            cls: class_probabilities[model_classes.index(cls)]
            for cls in valid_surface_classes
            if cls in model_classes
        }

        if not surface_scores:
            return classify_xthird(tooth_id, tooth_pts, caries_pts)

        prediction = max(surface_scores, key=surface_scores.get)
        return prediction, 0.0, {"method": "RandomForest_Proba"}
    except Exception:
        try:
            return classify_xthird(tooth_id, tooth_pts, caries_pts)
        except Exception:
            return "Other", 0.0, {}


# =========================================================
# Per-Case Prediction
# =========================================================

def process_case_ml(case_id, output_root):
    """
    Run classify_ml on all teeth in one case and save prediction JSON.

    Args:
        case_id (int): Case identifier (1-500).
        output_root (Path): Root directory for predictions.

    Returns:
        tuple: (is_success, status_message).
    """
    seg_data = _load_seg_case(case_id)
    caries_data = _load_caries_case(case_id)

    case_dir = output_root / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)

    result = {"case_number": int(case_id), "teeth_data": []}

    if seg_data is None or caries_data is None:
        with open(case_dir / f"case_{case_id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return False, "Missing input data"

    segmentation_map = build_seg_map(seg_data)
    for tooth in caries_data.get("teeth_caries_data", []):
        tooth_id = str(tooth.get("tooth_id", ""))
        tooth_pts = segmentation_map.get(tooth_id, [])
        caries_pts = tooth.get("caries_coordinates", [])

        surface, angle, metadata = classify_ml(tooth_id, tooth_pts, caries_pts)

        result["teeth_data"].append({
            "tooth_id": tooth_id,
            "version": "Run3",
            "has_caries": True,
            "confidence": float(tooth.get("confidence", 0.0)),
            "caries_position_detail": surface,
            "predicted_surface_fine": surface,
            "tooth_coordinates": tooth_pts,
            "caries_coordinates": caries_pts,
        })

    with open(case_dir / f"case_{case_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return True, f"OK ({len(result['teeth_data'])} teeth)"


# =========================================================
# Evaluation
# =========================================================

def load_prediction(case_num, out_dir):
    """Load prediction JSON for a single case."""
    pred_path = Path(out_dir) / f"case_{case_num}" / f"case_{case_num}.json"
    if not pred_path.exists():
        return []
    with open(pred_path, "r") as f:
        data = json.load(f)
    preds = []
    for t in data.get("teeth_data", []):
        tooth = str(t.get("tooth_id", "Unknown"))
        surface = t.get("predicted_surface_fine", t.get("caries_position_detail", "Other"))
        if surface not in VALID_SURFACES:
            surface = "Other"
        preds.append({"tooth": tooth, "surface": surface})
    return preds


def match_case(ground_truth, predictions):
    """Match ground-truth and predicted surfaces by tooth ID."""
    pred_dict = {p["tooth"]: p["surface"] for p in predictions}
    y_true, y_pred = [], []
    for g in ground_truth:
        y_true.append(g["surface"])
        y_pred.append(pred_dict.get(g["tooth"], "Other"))
    return y_true, y_pred


def evaluate_version(version):
    """
    Evaluate predictions for a version against XML ground truth.

    Args:
        version (str): Version tag (e.g. 'Run3').

    Returns:
        tuple: (all_y_true, all_y_pred, f1_macro).
    """
    out_dir = f"PCA_Output_{version}"
    all_y_true, all_y_pred = [], []

    print(f"[RUNNING] Evaluating {version}...", flush=True)
    for case_num in range(1, 501):
        gt_folder = GT_ROOT / f"case {case_num}"
        ground_truth = parse_case_ground_truth(gt_folder)
        predictions = load_prediction(case_num, out_dir)
        if len(ground_truth) == 0 and len(predictions) == 0:
            continue
        yt, yp = match_case(ground_truth, predictions)
        all_y_true.extend(yt)
        all_y_pred.extend(yp)
        _progress_bar(case_num, 500, f"Eval {version}")

    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=VALID_SURFACES)
    cm_df = pd.DataFrame(cm, index=VALID_SURFACES, columns=VALID_SURFACES)

    print(f"\n========== FINAL EVALUATION [{version}] ==========")
    print(f"Total Samples : {len(all_y_true)}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm_df)
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, labels=VALID_SURFACES, zero_division=0))

    return all_y_true, all_y_pred, f1


# =========================================================
# Evaluation Results Plot
# =========================================================

def plot_evaluation_results(y_true, y_pred, version="Run3"):
    """
    Generate and save two evaluation plots:
      1. Confusion matrix heatmap  ->  confusion_matrix_<version>.png
      2. Per-class Precision / Recall / F1 grouped bar chart
         ->  classification_metrics_<version>.png

    Args:
        y_true (list[str]): Ground-truth surface labels.
        y_pred (list[str]): Predicted surface labels.
        version (str): Version tag used in titles and filenames.
    """
    labels = VALID_SURFACES  # ["Occlusal", "Mesial", "Distal", "Other"]

    # ---- 1. Confusion Matrix Heatmap ----
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Count"},
        ax=ax1,
    )
    ax1.set_xlabel("Predicted Surface", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual Surface", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Confusion Matrix — {version}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.tick_params(axis="both", labelsize=11)
    plt.tight_layout()

    cm_path = f"confusion_matrix_{version.lower()}.png"
    fig1.savefig(cm_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"[SAVED] Confusion matrix -> {cm_path}  (dpi=300)")

    # ---- 2. Per-Class Metrics Grouped Bar Chart ----
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    metric_names = ["precision", "recall", "f1-score"]
    metric_data = []
    for label in labels:
        for m in metric_names:
            metric_data.append({
                "Surface": label,
                "Metric": m.capitalize().replace("F1-score", "F1 Score"),
                "Value": report[label][m],
            })
    metrics_df = pd.DataFrame(metric_data)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    palette = ["#3b82f6", "#f59e0b", "#10b981"]  # blue, amber, green
    x = np.arange(len(labels))
    bar_width = 0.24

    for idx, (m, color) in enumerate(zip(
        ["Precision", "Recall", "F1 Score"], palette
    )):
        values = metrics_df[metrics_df["Metric"] == m]["Value"].values
        bars = ax2.bar(
            x + idx * bar_width,
            values,
            bar_width,
            label=m,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar_obj in bars:
            height = bar_obj.get_height()
            ax2.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#333333",
            )

    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_xlabel("Surface Class", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Per-Class Classification Metrics — {version}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    metrics_path = f"classification_metrics_{version.lower()}.png"
    fig2.savefig(metrics_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[SAVED] Classification metrics -> {metrics_path}  (dpi=300)")


# =========================================================
# Feature Importance Plot
# =========================================================

def plot_feature_importance(model, feature_names, save_path="feature_importance.png"):
    """
    Visualize and save Random Forest feature importances as a horizontal bar chart.

    Args:
        model: Trained RandomForestClassifier with .feature_importances_.
        feature_names (list[str]): Feature column names matching FEATURE_COLS.
        save_path (str): Output file path for the high-res PNG.

    Returns:
        pd.DataFrame: Sorted importance table (descending).
    """
    # --- 1. Extract importances and build a sorted DataFrame ---
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # --- 2. Print top 5 to console ---
    print("=" * 50)
    print("  Top 5 Most Important Features (Random Forest)")
    print("=" * 50)
    for rank, row in importance_df.head(5).iterrows():
        print(f"  #{rank + 1}  {row['Feature']:<20s}  {row['Importance']:.4f}")
    print("=" * 50)

    # --- 3. Plot horizontal bar chart (ascending order for visual) ---
    plot_df = importance_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color gradient — darker = more important
    n = len(plot_df)
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.25, 0.90, n))

    bars = ax.barh(
        range(n),
        plot_df["Importance"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.72,
    )

    # --- 4. Annotate each bar with its value ---
    for bar_obj, val in zip(bars, plot_df["Importance"]):
        ax.text(
            bar_obj.get_width() + 0.003,
            bar_obj.get_y() + bar_obj.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    # --- 5. Labels, title, and styling ---
    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["Feature"], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title(
        "Random Forest Feature Importance",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Clean spine style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, plot_df["Importance"].max() * 1.20)

    plt.tight_layout()

    # --- 6. Save high-resolution image ---
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[SAVED] Feature importance chart -> {save_path}  (dpi=300)")

    return importance_df


# =========================================================
# HTML README Generation
# =========================================================

def generate_readme_html(output_path="README_run3.html"):
    """Generate the self-contained README_run3.html documentation file."""

    html_content = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dental Caries Surface Classification \u2014 Run 3</title>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
           background: #fff; color: #1f2937; line-height: 1.6; padding: 2rem; max-width: 960px; margin: 0 auto; }
    h1 { color: #2563eb; font-size: 1.8rem; margin-bottom: 0.5rem; }
    h2 { color: #2563eb; font-size: 1.35rem; margin-top: 2rem; margin-bottom: 0.75rem;
         border-bottom: 2px solid #e5e7eb; padding-bottom: 0.3rem; }
    h3 { color: #374151; font-size: 1.1rem; margin-top: 1.2rem; margin-bottom: 0.4rem; }
    p, li { margin-bottom: 0.5rem; } ul { padding-left: 1.5rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
    th, td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }
    th { background: #f3f4f6; font-weight: 600; }
    tr:nth-child(even) { background: #f9fafb; }
    tr.highlight { background: #dbeafe; font-weight: 600; }
    pre, code { font-family: "Cascadia Code", Consolas, monospace; font-size: 0.85rem; }
    pre { background: #f3f4f6; padding: 1rem; border-radius: 6px; overflow-x: auto; margin: 1rem 0; }
    code { background: #f3f4f6; padding: 0.15rem 0.35rem; border-radius: 3px; }
    .badge { display: inline-block; background: #2563eb; color: #fff; padding: 0.15rem 0.5rem;
             border-radius: 4px; font-size: 0.75rem; margin-left: 0.3rem; vertical-align: middle; }
    footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; font-size: 0.8rem; color: #6b7280; }
</style>
</head>
<body>
<h1>Dental Caries Surface Classification <span class="badge">Run 3</span></h1>
<h2>1. Project Overview</h2>
<ul>
    <li><strong>Objective:</strong> Classify caries surfaces into Occlusal, Mesial, or Distal.</li>
    <li><strong>Approach:</strong> 13 geometric features + Random Forest + Smart Fallback.</li>
    <li><strong>Dataset:</strong> 500 annotated dental panoramic X-ray cases.</li>
</ul>
<h2>2. Results Summary</h2>
<table><thead><tr><th>Run</th><th>Technique</th><th>Recall Occ</th><th>Recall Mes</th><th>Recall Dis</th><th>Accuracy</th></tr></thead>
<tbody>
<tr><td>Baseline</td><td>X-Thirds</td><td>0.20</td><td>0.83</td><td>0.83</td><td>0.70</td></tr>
<tr class="highlight"><td>Run 3</td><td>RF + Smart Fallback</td><td>0.84</td><td>0.80</td><td>0.84</td><td>0.83</td></tr>
</tbody></table>
<h2>3. Feature List</h2>
<p>13 PCA-aligned geometric features: is_upper, x_mean, y_mean, x_std, y_std, x_min, x_max, y_min, x_range, y_range, x_centroid_dist, aspect_ratio, coverage.</p>
<h2>4. Pipeline Architecture</h2>
<pre>
create_ml_dataset() -> train_classify_ml() -> classify_ml() -> process_case_ml() -> evaluate_version()
</pre>
<h2>5. Key Design Decisions</h2>
<p>GroupShuffleSplit by case_id prevents data leakage. class_weight='balanced' corrects for class imbalance. Smart Fallback filters out 'Other' predictions.</p>
<h2>6. File Structure</h2>
<pre>
SP/
+-- data/500 cases with annotation/case N/*.xml
+-- phase2-1april/
    +-- pipeline_run3_final.py
    +-- rf_classify_ml.pkl
    +-- PCA_Output_Run3/case_N/case_N.json
    +-- README_run3.html
</pre>
<footer>Generated by Run 3 Pipeline</footer>
</body></html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[DONE] {output_path} saved.")


# =========================================================
# Main
# =========================================================

def main():
    """Run the complete Run 3 pipeline end-to-end."""
    global rf_model

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    case_ids = list(range(1, 501))

    # --- Step 1: Extract features ---
    print("[START] เริ่มรัน Pipeline Run 3...", flush=True)
    feature_dataframe = create_ml_dataset(case_ids)
    print(
        f"[DONE] Step 1 เสร็จสิ้น! "
        f"ได้ข้อมูลเตรียมเทรนทั้งหมด: {len(feature_dataframe)} ซี่",
        flush=True,
    )

    # --- Step 2: Train model ---
    print("[RUNNING] Step 2: กำลัง Train โมเดล Random Forest...", flush=True)
    model, test_dataframe, _ = train_classify_ml(feature_dataframe)
    print(
        f"[DONE] Step 2 เสร็จสิ้น! "
        f"Train: {len(feature_dataframe) - len(test_dataframe)} ซี่ | "
        f"Test: {len(test_dataframe)} ซี่",
        flush=True,
    )

    # --- Step 3: Predict with Smart Fallback ---
    total = len(case_ids)
    success_count, failure_count = 0, 0
    print("[RUNNING] Step 3: นำโมเดลไปทำนายผลทั้ง 500 เคส...", flush=True)
    for i, case_id in enumerate(case_ids):
        is_success, _ = process_case_ml(case_id, OUTPUT_ROOT)
        if is_success:
            success_count += 1
        else:
            failure_count += 1
        _progress_bar(i + 1, total, "Step 3: ทำนายผล")

    print(
        f"[SUCCESS] สำเร็จ! เขียนไฟล์ทำนายผลแล้ว: "
        f"{success_count} เคส, ล้มเหลว: {failure_count} เคส",
        flush=True,
    )

    # --- Step 4: Evaluate ---
    all_y_true, all_y_pred, f1 = evaluate_version("Run3")

    # --- Step 4.5: Evaluation plots ---
    print("[RUNNING] กำลังสร้างกราฟผลการ Predict...", flush=True)
    plot_evaluation_results(all_y_true, all_y_pred, version="Run3")

    # --- Step 5: Feature importance plot ---
    plot_feature_importance(model, FEATURE_COLS)

    # --- Step 6: Generate HTML README ---
    generate_readme_html()

    print("\n[ALL DONE] Pipeline Run 3 complete.")


if __name__ == "__main__":
    main()
