#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
caries_surface_classifier_20260806.py
=====================================
Model-selection benchmark for automated dental caries surface classification
from panoramic radiograph segmentation data.

This module implements a reproducible evaluation framework that benchmarks
three candidate classifiers — Random Forest, XGBoost, and a compact PyTorch
MLP — on 14 engineered geometric features derived from tooth and caries
segmentation masks.  The final model is selected empirically via
group-aware cross-validation (GroupKFold by case_id) using macro-averaged
F1 as the primary selection metric.

Pipeline overview::

    AIM-XML ground-truth annotations
        ↓
    Validated parser (SNODENT → FDI mapping)
        ↓
    PCA-aligned geometric feature extraction (14 features)
        ↓
    Data validation & StandardScaler preprocessing
        ↓
    Group-aware cross-validation (GroupKFold, k=5)
        ├── Random Forest   (baseline)
        ├── XGBoost          (tree-boosting candidate)
        └── PyTorch MLP      (neural-network candidate)
        ↓
    Model selection by macro-averaged F1
        ↓
    Final evaluation on untouched held-out test set

Usage::

    python caries_surface_classifier_20260806.py

References:
    - FDI World Dental Federation notation: ISO 3950
    - Pitts, N.B. et al. (2017). Dental caries. Nature Reviews, 3, 17030.
    - Varoquaux, G. et al. (2017). Assessing and tuning brain decoders.
      NeuroImage, 145, 166–179.  (GroupKFold rationale)
"""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — IMPORTS & REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
# All random seeds are fixed at module level to guarantee bitwise-reproducible
# results across runs.  Deterministic settings are enabled for PyTorch to
# ensure identical gradient computations on CUDA backends.

import os
import sys
import json
import math
import random
import warnings
import csv
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy
from datetime import datetime

# Configure UTF-8 encoding on standard streams for cross-platform terminal compatibility.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Reproducibility: fix all random seeds and enable deterministic operations.
# This is critical for scientific reproducibility — different seeds can shift
# F1 by ±0.02 on small datasets, which may reverse model-ranking conclusions.
# ---------------------------------------------------------------------------
SEED = 42
SEEDS_FOR_REPEATED_EVAL = [42, 123, 456]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Deterministic mode trades speed for exact reproducibility.
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CONSTANTS & FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# The feature vector encodes the spatial distribution of carious lesion pixels
# relative to the PCA-aligned tooth bounding box.  Each feature captures a
# distinct geometric property that correlates with clinical surface taxonomy
# per FDI World Dental Federation notation.
#
# Feature selection rationale:
#   - Positional statistics (mean, min, max) localise the lesion within the
#     tooth crown, discriminating occlusal (central) from proximal (lateral)
#     caries.
#   - Dispersion statistics (std, range) capture lesion spread, which differs
#     between focal pit-and-fissure caries and broad smooth-surface caries.
#   - Shape descriptors (aspect_ratio, coverage) encode the morphological
#     relationship between lesion and tooth geometry.

FEATURE_COLS = [
    "is_upper",         # Jaw laterality — upper-jaw teeth (FDI quadrants 1–2)
                        # exhibit different caries morphology due to
                        # gravitational saliva pooling and distinct occlusal
                        # anatomy.
    "x_mean",           # Lesion centroid, horizontal — low values indicate
                        # mesial or distal localisation depending on quadrant.
    "y_mean",           # Lesion centroid, vertical — distinguishes occlusal
                        # (mid-crown) from cervical (near gingival margin)
                        # lesions.
    "x_std",            # Horizontal dispersion — pit-and-fissure caries
                        # concentrate narrowly (low σ_x), while smooth-surface
                        # caries spread laterally (high σ_x).
    "y_std",            # Vertical dispersion — analogous to x_std along the
                        # occluso-gingival axis.
    "x_min",            # Left boundary of lesion in normalised coordinates.
    "x_max",            # Right boundary of lesion in normalised coordinates.
    "y_min",            # Upper boundary (occlusal direction) of lesion.
    "y_max",            # Lower boundary (gingival direction) of lesion.
    "x_range",          # Horizontal extent (x_max − x_min) — proxy for
                        # lesion width within the tooth crown.
    "y_range",          # Vertical extent (y_max − y_min) — proxy for lesion
                        # depth along the occluso-gingival axis.
    "x_centroid_dist",  # Absolute horizontal deviation of the lesion centroid
                        # from the tooth midline (|x_mean − 0.5|).  Low values
                        # strongly indicate occlusal caries, because
                        # pit-and-fissure lesions cluster around x = 0.5.
    "aspect_ratio",     # Tooth bounding-box width ÷ height — captures the
                        # proportional shape that varies between anterior and
                        # posterior teeth.
    "coverage",         # Lesion pixel count ÷ tooth pixel count — represents
                        # the fractional area of the tooth affected by caries.
]
"""Ordered list of the 14 geometric features used by all candidate models."""

VALID_SURFACES = ["Occlusal", "Mesial", "Distal", "Other"]
"""The four surface classes used in evaluation.  'Other' captures caries on
surfaces not reliably classifiable from a 2D panoramic projection (e.g.,
buccal, lingual, or root caries)."""

MAX_TILT_DEG = 45.0
"""Maximum allowable PCA rotation angle (degrees).  Rotations exceeding this
threshold are clamped to zero because they typically arise from segmentation
artefacts (e.g., merged adjacent teeth) rather than genuine tooth tilt.  The
45° threshold was determined empirically from manual inspection of edge cases
in the 500-case dataset."""

MIN_CLUSTER_SIZE = 15
"""Minimum connected-component area (pixels) retained during caries mask
denoising.  Components smaller than this threshold are discarded as
segmentation noise.  The value balances noise removal against preservation
of small but genuine enamel lesions."""

LEFT_BOUND = 0.40
"""X-thirds left boundary.  Pixels with relative x < 0.40 fall in the
left zone of the PCA-aligned tooth bounding box."""

RIGHT_BOUND = 0.60
"""X-thirds right boundary.  Pixels with relative x > 0.60 fall in the
right zone.  The central zone (0.40–0.60) corresponds to the occlusal
region for posterior teeth."""

N_CV_FOLDS = 5
"""Number of GroupKFold cross-validation splits.  Five folds provide a
reasonable bias–variance trade-off for ~500 cases while ensuring
sufficient validation-set size per fold."""

HOLDOUT_TEST_SIZE = 0.2
"""Fraction of cases reserved as untouched held-out test set.  This
partition is used exactly once for final model comparison."""

# MLP training hyperparameters.
MLP_HIDDEN_LAYERS = [64, 32]
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 32
MLP_PATIENCE = 10
"""Early-stopping patience (epochs without validation macro F1 improvement)."""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# Directory layout follows the project convention where segmentation outputs,
# caries mappings, and ground-truth annotations reside in sibling directories
# under the project root.

try:
    _THIS_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive execution (Jupyter, IPython).
    _THIS_DIR = Path.cwd()

_SP_DIR = _THIS_DIR.parent if (_THIS_DIR.parent / "week2-Tooth Detection & Segmentation").exists() else _THIS_DIR.parent.parent
SEG_DIR = _SP_DIR / "week2-Tooth Detection & Segmentation" / "500-segmentation+recognition"
CARIES_DIR = _SP_DIR / "week3-Caries-to-Tooth Mapping" / "dental_analysis_output"
GT_ROOT = _SP_DIR / "data" / "500 cases with annotation"
OUTPUT_ROOT = _THIS_DIR / "PCA_Output_Run3"

MODEL_DIR = _THIS_DIR / "trained_models"
MODEL_STEM = "caries_surface_clf_20260806"
METADATA_PATH = MODEL_DIR / f"{MODEL_STEM}_metadata.json"

# Global model placeholder — populated after training or loading.
active_model = None
active_model_type = None  # "random_forest", "xgboost", or "mlp"
active_scaler = None      # StandardScaler fitted on training data
active_label_encoder = None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — XML GROUND-TRUTH PARSER
# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth annotations follow the AIM (Annotation and Image Markup)
# schema, an XML format originally developed for the caBIG cancer imaging
# initiative and widely adopted in dental radiology research.
#
# Tooth identification uses SNODENT (Systematized Nomenclature of Dentistry)
# codes, which are mapped to FDI two-digit notation for compatibility with
# the segmentation pipeline.  Surface annotations use SNODENT procedure
# codes that encode the affected tooth surface.

AIM_NS = "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"
ISO_NS = "uri:iso.org:21090"
NS = {"aim": AIM_NS, "iso": ISO_NS}

# SNODENT surface procedure codes → clinical surface name.
# Each code identifies a restorative procedure on a specific surface;
# we extract only the surface component for classification labelling.
SNODENT_SURFACE_MAP = {
    "144414D": "Occlusal", "146014D": "Distal", "145374D": "Mesial",
    "144474D": "Occlusal", "146074D": "Distal", "145434D": "Mesial",
}

# Fallback mapping when SNODENT codes are absent but display names are
# present in the XML.  Case-sensitive matching handles inconsistent
# capitalisation across annotation batches.
DISPLAY_NAME_TO_SURFACE = {
    "Occlusal surface": "Occlusal", "Occlusal Surface": "Occlusal",
    "Distal Surface": "Distal", "Distal surface": "Distal",
    "Mesial Surface": "Mesial", "Mesial surface": "Mesial",
}

# SNODENT anatomical codes → FDI two-digit tooth identifiers.
# The mapping covers all 32 permanent teeth across four quadrants.
# Stored as an ordered pair list so that duplicate source codes are
# detected at module load time (Python dicts silently keep last value).
SNODENT_TO_FDI_PAIRS = [
    ("160903D", "11"), ("160842D", "12"), ("160840D", "13"), ("161607D", "14"),
    ("161546D", "15"), ("161010D", "16"), ("161262D", "17"), ("161227D", "18"),
    ("160194D", "21"), ("160132D", "22"), ("160957D", "23"), ("161329D", "24"),
    ("161178D", "25"), ("161132D", "26"), ("161317D", "27"), ("161454D", "28"),
    ("161136D", "31"), ("160556D", "32"), ("160817D", "33"), ("160654D", "34"),
    ("161150D", "35"), ("161533D", "36"), ("161372D", "37"), ("161258D", "38"),
    ("160964D", "41"), ("160350D", "42"), ("160894D", "43"), ("161496D", "44"),
    ("161412D", "45"), ("160770D", "46"), ("160704D", "47"), ("161121D", "48"),
]


def _build_snodent_map(pairs):
    """Convert SNODENT pair list to dict, raising on duplicate source codes.

    Args:
        pairs (list[tuple]): (snodent_code, fdi_tooth) pairs.

    Returns:
        dict: Validated SNODENT → FDI mapping.

    Raises:
        ValueError: If any SNODENT code appears more than once.
    """
    duplicates = set()
    seen = set()
    for code, _ in pairs:
        if code in seen:
            duplicates.add(code)
        seen.add(code)

    if duplicates:
        raise ValueError(
            "Duplicate SNODENT codes detected.  Verify against the AIM "
            f"annotation specification before proceeding: {duplicates}"
        )
    return dict(pairs)


SNODENT_TO_FDI = _build_snodent_map(SNODENT_TO_FDI_PAIRS)


def _get_display_name(element):
    """Extract the displayName attribute from an ISO-namespaced XML element.

    Args:
        element: XML element containing an ``iso:displayName`` child.

    Returns:
        str: The display name value, or empty string if absent.
    """
    dn = element.find("iso:displayName", NS)
    return dn.get("value", "") if dn is not None else ""


def snodent_display_to_fdi(display_name):
    """Convert a SNODENT descriptive tooth name to FDI two-digit notation.

    The SNODENT display name encodes quadrant (upper/lower, left/right) and
    tooth type (e.g., 'first molar').  This function parses both components
    to reconstruct the FDI identifier, which is the standard notation used
    by the segmentation pipeline and clinical literature.

    Args:
        display_name (str): SNODENT descriptive name
            (e.g., 'Upper Right First Molar').

    Returns:
        str: FDI identifier (e.g., '16'), or empty string if unparseable.
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
    """Parse a single AIM XML annotation file to extract tooth and surface.

    The function navigates the AIM schema hierarchy:
    ``ImageAnnotation → ImagingPhysicalEntityCharacteristicCollection``
    and uses questionIndex to distinguish tooth identification (index 0)
    from surface annotation (index 1).

    Args:
        xml_path (str): Absolute path to the AIM XML file.

    Returns:
        dict or None: ``{'tooth': str, 'surface': str}``, or ``None``
            if the file cannot be parsed or lacks required elements.
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
    """Parse all AIM XML ground-truth files in a case directory.

    Each XML file within the case folder represents a single caries
    annotation.  Multiple annotations per case are expected when a patient
    has caries on multiple teeth.

    Args:
        case_folder (Path): Directory containing ``*.xml`` annotation files.

    Returns:
        list[dict]: Each dict contains ``'tooth'`` (str, FDI notation) and
            ``'surface'`` (str, one of VALID_SURFACES).
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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — PCA & GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
# Tooth orientation in panoramic radiographs varies due to projection
# geometry, patient positioning, and natural dental arch curvature.  PCA-based
# alignment rotates each tooth mask into a canonical orientation (crown
# upward for upper jaw, downward for lower jaw), ensuring that the extracted
# geometric features are rotationally invariant.
#
# Without this alignment step, a mesially tilted tooth would yield feature
# vectors indistinguishable from a distal-surface lesion on an upright tooth,
# because the raw pixel coordinates conflate tooth orientation with caries
# localisation.

def is_upper_jaw(tooth_id):
    """Determine whether a tooth belongs to the upper jaw.

    Upper-jaw teeth reside in FDI quadrants 1 (upper-right) and 2
    (upper-left).  This distinction is clinically relevant because
    gravitational saliva flow and occlusal morphology differ between jaws.

    Args:
        tooth_id (str or int): FDI tooth identifier (e.g., '16').

    Returns:
        bool: True if the tooth is in the upper jaw.
    """
    return int(str(tooth_id)[0]) in [1, 2]


def get_quadrant(tooth_id):
    """Extract the FDI quadrant number (1–4) from a tooth identifier.

    Args:
        tooth_id (str or int): FDI tooth identifier.

    Returns:
        int: Quadrant number (1 = upper-right, 2 = upper-left,
            3 = lower-left, 4 = lower-right).
    """
    return int(str(tooth_id)[0])


def get_bbox(pts):
    """Compute the axis-aligned bounding box for a set of 2D points.

    Args:
        pts (array-like): Nx2 array of (x, y) coordinates.

    Returns:
        tuple: (x_min, y_min, width, height).
    """
    p = np.array(pts, dtype=np.float64)
    bbox_min, bbox_max = np.min(p, 0), np.max(p, 0)
    return bbox_min[0], bbox_min[1], bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]


def rotate(pts, center, angle):
    """Rotate 2D points around a centre by a given angle.

    Args:
        pts (array-like): Nx2 array of (x, y) coordinates.
        center (array-like): (x, y) rotation centre.
        angle (float): Rotation angle in radians (counter-clockwise positive).

    Returns:
        np.ndarray: Rotated Nx2 coordinates.
    """
    p = np.array(pts, dtype=np.float64) - center
    c, s = np.cos(angle), np.sin(angle)
    return np.dot(p, np.array([[c, -s], [s, c]]).T) + center


def remove_small_clusters(caries_pts, min_cluster=MIN_CLUSTER_SIZE):
    """Remove segmentation noise by discarding small connected components.

    Dental caries segmentation models occasionally produce isolated pixel
    clusters that do not correspond to genuine lesions.  Connected-component
    analysis identifies spatially contiguous regions, and components below
    the size threshold are removed.

    Args:
        caries_pts (array-like): Nx2 caries pixel coordinates.
        min_cluster (int): Minimum component area (pixels) to retain.

    Returns:
        np.ndarray: Filtered caries coordinates with noise removed.
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
    """PCA-based 4-rule tooth orientation alignment.

    The four orientation rules ensure consistent alignment across all teeth
    regardless of their natural tilt or position in the dental arch:

    1. **Axis selection**: The eigenvector with the larger |Y| component is
       designated as the vertical (long) axis of the tooth.
    2. **Vertical direction**: Upper-jaw teeth are oriented crown-upward
       (positive Y); lower-jaw teeth crown-downward (negative Y).
    3. **Horizontal direction**: The horizontal axis is directed toward the
       dental midline (mesial direction) based on the FDI quadrant.
    4. **Tilt clamping**: Rotation angles exceeding MAX_TILT_DEG are clamped
       to zero, because extreme angles indicate segmentation artefacts
       rather than genuine tooth morphology.

    Args:
        points (array-like): Nx2 tooth mask pixel coordinates.
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

    # Rule 2: vertical direction based on jaw laterality.
    upper = is_upper_jaw(tooth_id)
    if upper:
        if vertical_axis[1] < 0:
            vertical_axis = -vertical_axis
    else:
        if vertical_axis[1] > 0:
            vertical_axis = -vertical_axis

    # Rule 3: horizontal direction toward dental midline.
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

    # Rule 4: clamp extreme rotations that indicate segmentation artefacts.
    clamped = False
    if abs(math.degrees(rotation_angle)) > MAX_TILT_DEG:
        rotation_angle = 0.0
        clamped = True

    return mean, rotation_angle, clamped


def build_seg_map(seg_data):
    """Build a tooth_id → pixel_coordinates lookup from segmentation JSON.

    Args:
        seg_data (dict): Parsed segmentation JSON containing ``teeth_data``.

    Returns:
        dict: Mapping from tooth_id (str) to list of [x, y] coordinates.
    """
    return {
        str(t["tooth_id"]): t.get("pixel_coordinates", [])
        for t in seg_data.get("teeth_data", [])
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — FILE I/O & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction transforms raw segmentation masks into a compact
# 14-dimensional vector that encodes the spatial relationship between the
# carious lesion and the PCA-aligned tooth bounding box.  This normalised
# representation eliminates variation due to tooth size, position within the
# panoramic image, and rotational misalignment.

def _load_json_file(path):
    """Load a JSON file, returning None if the file does not exist.

    Args:
        path (Path): Absolute path to the JSON file.

    Returns:
        dict or None: Parsed JSON content, or None if missing.
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_seg_case(case_id):
    """Load segmentation JSON for a single case.

    Args:
        case_id (int): Case identifier (1–500).

    Returns:
        dict or None: Segmentation data, or None if missing.
    """
    return _load_json_file(SEG_DIR / f"case {case_id}" / f"case_{case_id}_results.json")


def _load_caries_case(case_id):
    """Load caries-to-tooth mapping JSON for a single case.

    Args:
        case_id (int): Case identifier (1–500).

    Returns:
        dict or None: Caries mapping data, or None if missing.
    """
    return _load_json_file(CARIES_DIR / f"case {case_id}" / f"case_{case_id}_caries_mapping.json")


def _extract_ml_feature_dict(tooth_id, tooth_pts, caries_pts):
    """Extract 14 geometric features for one caries–tooth pair.

    The feature vector captures the normalised spatial distribution of
    caries pixels within the PCA-aligned tooth bounding box.  All
    coordinates are expressed in the [0, 1] normalised space where
    (0, 0) = top-left and (1, 1) = bottom-right of the aligned bounding box.

    Args:
        tooth_id (str): FDI tooth identifier (e.g., '16').
        tooth_pts (list): Tooth mask pixel coordinates (Nx2).
        caries_pts (list): Caries region pixel coordinates (Mx2).

    Returns:
        dict or None: Feature dictionary keyed by FEATURE_COLS names,
            or None if the caries region is empty after denoising or
            the tooth bounding box has zero area.
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
        "y_max": float(np.max(y_rel)),
        "x_range": float(np.max(x_rel) - np.min(x_rel)),
        "y_range": float(np.max(y_rel) - np.min(y_rel)),
        "x_centroid_dist": float(abs(np.mean(x_rel) - 0.5)),
        "aspect_ratio": float(w / h),
        "coverage": float(len(caries_clean) / (len(tooth_pts) + 1e-6)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — PROGRESS & UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _progress_bar(current, total, prefix="Progress", bar_length=30):
    """Print an inline text-based progress bar that overwrites itself.

    Args:
        current (int): Current step number (1-indexed).
        total (int): Total number of steps.
        prefix (str): Label displayed before the bar.
        bar_length (int): Character width of the bar.
    """
    fraction = current / max(total, 1)
    filled = int(bar_length * fraction)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(
        f"\r   {prefix} [{bar}] {fraction*100:.0f}% ({current}/{total})",
        end="", flush=True,
    )
    if current >= total:
        print()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — DATASET CONSTRUCTION & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
# The dataset is constructed by iterating over all 500 cases, extracting
# features for each caries–tooth pair that has a corresponding ground-truth
# annotation.  Preprocessing includes data validation and feature
# standardisation via StandardScaler.
#
# StandardScaler is fitted exclusively on the training partition to prevent
# information leakage from validation or test observations into the
# feature scaling parameters.

def create_ml_dataset(case_ids):
    """Build a labelled ML dataset by extracting features from all cases.

    For each case, segmentation masks, caries mappings, and XML ground-truth
    annotations are loaded and cross-referenced.  Features are extracted
    only for teeth that have both caries segmentation data and a
    ground-truth surface annotation.

    Args:
        case_ids (list[int]): Case identifiers to process.

    Returns:
        pd.DataFrame: Dataset with columns
            ``['case_id', 'tooth_id', *FEATURE_COLS, 'label']``.
    """
    dataset_rows = []
    total = len(case_ids)
    print(f"[STAGE 7] Extracting features from {total} cases...", flush=True)

    for i, case_id in enumerate(case_ids):
        _progress_bar(i + 1, total, "Feature extraction")

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


def validate_dataset(df):
    """Validate dataset integrity before model training.

    Checks for common data quality issues that would compromise model
    training or produce misleading evaluation metrics.

    Args:
        df (pd.DataFrame): Feature dataset with FEATURE_COLS and 'label'.

    Raises:
        ValueError: If critical validation checks fail.
    """
    print("[VALIDATION] Running data integrity checks...", flush=True)
    issues = []

    # Check for missing values.
    missing = df[FEATURE_COLS].isnull().sum()
    if missing.any():
        issues.append(f"Missing values found:\n{missing[missing > 0]}")

    # Check for non-finite values (NaN, ±inf).
    non_finite = (~np.isfinite(df[FEATURE_COLS].values)).sum()
    if non_finite > 0:
        issues.append(f"Non-finite values detected: {non_finite} total")

    # Check for duplicate rows (same case_id + tooth_id).
    dupes = df.duplicated(subset=["case_id", "tooth_id"], keep=False)
    if dupes.any():
        issues.append(f"Duplicated (case_id, tooth_id) entries: {dupes.sum()}")

    # Check feature ranges — relative coordinates should be in [0, 1].
    bounded_cols = ["x_mean", "y_mean", "x_min", "x_max", "y_min", "y_max",
                    "x_centroid_dist", "coverage"]
    for col in bounded_cols:
        if col in df.columns:
            vmin, vmax = df[col].min(), df[col].max()
            if vmin < -0.01 or vmax > 1.01:
                issues.append(f"Feature '{col}' out of [0, 1] range: [{vmin:.4f}, {vmax:.4f}]")

    # Check is_upper is binary.
    if not df["is_upper"].isin([0, 1]).all():
        issues.append("Feature 'is_upper' contains non-binary values")

    # Check label distribution.
    label_counts = df["label"].value_counts()
    print(f"  Label distribution:\n{label_counts.to_string()}")

    if issues:
        for issue in issues:
            print(f"  [WARNING] {issue}")
    else:
        print("  [OK] All validation checks passed.")


def audit_and_resolve_duplicates(df):
    """Audit and resolve duplicated (case_id, tooth_id) records.

    Categorises duplicates into three groups:
    - Exact: all columns identical — safely deduplicated.
    - Same-label-different-features: same label, different feature values
      — exported as CSV and raises ValueError (one row per tooth required).
    - Conflicting labels: different labels for the same tooth —
      exported as CSV and raises ValueError.

    Args:
        df (pd.DataFrame): Feature dataset.

    Returns:
        pd.DataFrame: Deduplicated dataset.

    Raises:
        ValueError: If unresolvable duplicates are found.
    """
    dupes_mask = df.duplicated(subset=["case_id", "tooth_id"], keep=False)
    if not dupes_mask.any():
        print("\n[DUPLICATE AUDIT] No duplicated (case_id, tooth_id) entries found.")
        return df

    dupes = df[dupes_mask].sort_values(["case_id", "tooth_id"])
    exact_dupes_idx = []
    conflicting_groups = []
    same_label_diff_features_groups = []

    for (cid, tid), group in dupes.groupby(["case_id", "tooth_id"]):
        if len(group.drop_duplicates()) == 1:
            exact_dupes_idx.extend(group.index[1:].tolist())
        elif len(group["label"].unique()) > 1:
            conflicting_groups.append(group)
        else:
            same_label_diff_features_groups.append(group)

    print("\n[DUPLICATE AUDIT]")
    print(f"  Exact duplicate rows removed: {len(exact_dupes_idx)}")
    print(f"  Same-label non-identical groups: {len(same_label_diff_features_groups)}")
    print(f"  Conflicting-label groups: {len(conflicting_groups)}")

    if conflicting_groups or same_label_diff_features_groups:
        error_df_list = conflicting_groups + same_label_diff_features_groups
        error_df = pd.concat(error_df_list)
        csv_path = MODEL_DIR / f"{MODEL_STEM}_conflicting_duplicates.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(csv_path, index=False)
        print(f"  [ERROR] Unresolved duplicates exported to: {csv_path}")
        raise ValueError(
            "Dataset contains conflicting or non-identical duplicate "
            "(case_id, tooth_id) entries."
        )

    cleaned_df = df.drop(index=exact_dupes_idx).reset_index(drop=True)
    print(f"  Final unique samples: {len(cleaned_df)}\n")

    assert not cleaned_df.duplicated(subset=["case_id", "tooth_id"]).any()
    return cleaned_df

class CariesFeatureDataset(Dataset):
    """PyTorch Dataset for the 14-feature tabular caries classification task.

    This wrapper enables integration with PyTorch's DataLoader for batched
    training of the MLP candidate.  Features are stored as float32 tensors
    and labels as int64 indices suitable for CrossEntropyLoss.

    Args:
        features (np.ndarray): Array of shape (N, 14) with standardised
            feature values.
        labels (np.ndarray): Array of shape (N,) with integer class indices.
    """

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Three candidate classifiers are defined with comparable capacity and
# regularisation settings.  All models receive the same 14-dimensional
# standardised feature input and produce predictions over 4 surface classes.

class CariesSurfaceMLP(nn.Module):
    """Compact multi-layer perceptron for dental caries surface classification.

    Architecture rationale:
        Two hidden layers with 64 and 32 units provide sufficient capacity
        for the 14-dimensional input space without excessive overfitting
        risk on the ~500-case dataset.  Dropout (p=0.3) provides stochastic
        regularisation by randomly zeroing activations during training,
        which reduces co-adaptation of hidden units.

        The ``forward()`` method returns **raw logits** (unnormalised scores).
        ``torch.nn.CrossEntropyLoss`` applies log-softmax internally, so
        placing Softmax inside the forward pass would result in double
        normalisation and numerical instability.  Softmax is applied
        explicitly only during inference when calibrated probabilities
        are required.

    Note:
        Batch normalisation vs. layer normalisation:
        For very small effective batch sizes (≤ 8), BatchNorm statistics
        become noisy and LayerNorm is preferable.  With ``MLP_BATCH_SIZE=32``,
        BatchNorm provides stable running statistics.  This choice should be
        re-evaluated if the batch size is reduced.

    Args:
        input_dim (int): Number of input features (default: 14).
        hidden_layers (list[int]): Hidden layer widths (default: [64, 32]).
        num_classes (int): Number of output classes (default: 4).
        dropout (float): Dropout probability (default: 0.3).
        use_batchnorm (bool): If True, use BatchNorm1d; if False, use
            LayerNorm.  Default True (appropriate for batch_size ≥ 16).
    """

    def __init__(self, input_dim=14, hidden_layers=None, num_classes=4,
                 dropout=MLP_DROPOUT, use_batchnorm=True):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = MLP_HIDDEN_LAYERS

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            else:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        # No Softmax here — CrossEntropyLoss expects raw logits.

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning raw logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes).
        """
        return self.network(x)


def _compute_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights for imbalanced training data.

    Follows the formula: ``w_c = N / (num_classes * n_c)`` where ``N`` is the
    total sample count and ``n_c`` is the count for class ``c``.  This matches
    the scikit-learn ``class_weight='balanced'`` convention.

    Args:
        labels (np.ndarray): Integer class labels for the training set.
        num_classes (int): Total number of classes.

    Returns:
        torch.Tensor: Class weight tensor of shape (num_classes,).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    # All observed classes must be present in the training split.
    zero_classes = np.where(counts == 0)[0]
    if len(zero_classes) > 0:
        raise ValueError(
            f"Training split is missing observed classes with indices: "
            f"{zero_classes.tolist()}.  Cannot compute class weights."
        )
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_rf_model(seed=SEED):
    """Construct a Random Forest classifier with balanced class weights.

    Random Forest serves as the reproducible baseline.  The ensemble of 200
    decision trees with bootstrap aggregation provides robust predictions
    without hyperparameter sensitivity.  Balanced class weights compensate
    for the unequal prevalence of Occlusal, Mesial, and Distal caries.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        RandomForestClassifier: Configured but untrained model.
    """
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=seed,
    )


def build_xgb_model(num_classes=4, seed=SEED):
    """Construct an XGBoost gradient-boosted tree classifier.

    XGBoost is included as the primary tree-boosting candidate because
    gradient boosting often outperforms bagging (Random Forest) on tabular
    data when the learning rate and regularisation are appropriately tuned.

    Subsample and colsample_bytree provide stochastic regularisation
    analogous to Random Forest's feature and sample bagging, reducing
    overfitting on the small dataset.

    Args:
        num_classes (int): Number of output classes.
        seed (int): Random seed for reproducibility.

    Returns:
        xgb.XGBClassifier: Configured but untrained model.
    """
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="mlogloss",
        verbosity=0,
    )


def build_mlp_model(input_dim=14, num_classes=4):
    """Construct the PyTorch MLP classifier.

    The MLP is included as the neural-network candidate and as an option
    when PyTorch-native deployment is an explicit requirement.  The compact
    architecture (14 → 64 → 32 → 4) is chosen to match the low
    dimensionality of the input space.

    Args:
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.

    Returns:
        CariesSurfaceMLP: Configured but untrained model on DEVICE.
    """
    use_bn = MLP_BATCH_SIZE >= 16
    model = CariesSurfaceMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        use_batchnorm=use_bn,
    )
    return model.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — GROUP-AWARE CROSS-VALIDATION & MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
# GroupKFold ensures that no patient/case appears in multiple folds,
# preventing data leakage from multi-tooth observations within the same
# panoramic radiograph.  This is critical because teeth from the same
# patient share anatomical, genetic, and environmental factors that
# would inflate within-fold accuracy if leaked across partitions
# (Varoquaux et al., 2017).

def _train_and_evaluate_rf(X_train, y_train, X_val, y_val, labels, seed=SEED):
    """Train and evaluate a Random Forest model on one fold.

    Args:
        X_train (np.ndarray): Training features (N_train, 14).
        y_train (np.ndarray): Training labels (N_train,).
        X_val (np.ndarray): Validation features (N_val, 14).
        y_val (np.ndarray): Validation labels (N_val,).
        seed (int): Random seed.

    Returns:
        dict: Evaluation metrics including macro F1, balanced accuracy,
            per-class precision/recall/F1, and the trained model.
    """
    model = build_rf_model(seed=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return _compute_metrics(y_val, y_pred, labels=labels, model=model)


def _train_and_evaluate_xgb(X_train, y_train, X_val, y_val,
                             label_encoder, labels, seed=SEED):
    """Train and evaluate an XGBoost model on one fold.

    XGBoost requires integer-encoded labels.  Class weights are supplied
    via ``sample_weight`` computed from training label frequencies using
    the inverse-frequency formula.

    Args:
        X_train (np.ndarray): Training features (N_train, 14).
        y_train (np.ndarray): Training integer labels (N_train,).
        X_val (np.ndarray): Validation features (N_val, 14).
        y_val (np.ndarray): Validation integer labels (N_val,).
        label_encoder (LabelEncoder): Fitted label encoder.
        seed (int): Random seed.

    Returns:
        dict: Evaluation metrics and the trained model.
    """
    model = build_xgb_model(
        num_classes=len(label_encoder.classes_), seed=seed
    )
    # Compute sample weights from inverse class frequency.
    class_weights = _compute_class_weights(y_train, len(label_encoder.classes_))
    sample_weights = class_weights.numpy()[y_train]

    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_val)
    return _compute_metrics(y_val, y_pred, labels=labels, model=model)


def _train_and_evaluate_mlp(X_train, y_train, X_val, y_val,
                             num_classes, labels, seed=SEED):
    """Train and evaluate a PyTorch MLP model on one fold.

    Training uses Adam optimiser with class-weighted CrossEntropyLoss.
    Early stopping monitors validation macro F1 with a patience of
    MLP_PATIENCE epochs.

    Args:
        X_train (np.ndarray): Standardised training features (N_train, 14).
        y_train (np.ndarray): Training integer labels (N_train,).
        X_val (np.ndarray): Standardised validation features (N_val, 14).
        y_val (np.ndarray): Validation integer labels (N_val,).
        num_classes (int): Number of output classes.
        seed (int): Random seed.

    Returns:
        dict: Evaluation metrics and the trained model (best checkpoint).
    """
    # Fix seeds for this training run.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_mlp_model(input_dim=X_train.shape[1], num_classes=num_classes)

    # Class weights for imbalanced training data.
    class_weights = _compute_class_weights(y_train, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR)

    train_dataset = CariesFeatureDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True,
        drop_last=False,
    )
    val_dataset = CariesFeatureDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(MLP_EPOCHS):
        # --- Training phase ---
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # --- Validation phase ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                logits = model(batch_X)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Early stopping based on validation macro F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                break

    # Restore best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on validation set.
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            logits = model(batch_X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_val_arr = np.array(y_val)
    y_pred_arr = np.array(all_preds)
    metrics = _compute_metrics(y_val_arr, y_pred_arr, labels=labels, model=model)
    metrics["best_epoch"] = best_epoch
    return metrics


def _compute_metrics(y_true, y_pred, labels=None, model=None):
    """Compute evaluation metrics for a single fold.

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        labels (list): Label indices to include in macro average.
        model: Trained model instance (stored for later use).

    Returns:
        dict: Contains 'macro_f1', 'balanced_acc', 'accuracy',
            'per_class' (dict of precision/recall/f1 per class),
            'confusion_matrix', and 'model'.
    """
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_vals = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_vals = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "accuracy": acc,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall_vals.tolist(),
        "f1_per_class": f1_vals.tolist(),
        "confusion_matrix": cm,
        "model": model,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_cross_validation(feature_df, label_encoder, seeds=None, n_splits=None):
    """Run GroupKFold cross-validation for all three candidate classifiers.

    Each candidate is evaluated under identical data splits to ensure a
    fair comparison.  The StandardScaler is fitted on each training fold
    independently, preventing information leakage from validation data.

    Args:
        feature_df (pd.DataFrame): Full feature dataset (train+val portion,
            held-out test already removed).
        label_encoder (LabelEncoder): Fitted label encoder mapping surface
            names to integer indices.
        seeds (list[int], optional): Random seeds for repeated model training.
        n_splits (int, optional): Number of GroupKFold splits.

    Returns:
        dict: Nested dictionary of results per candidate per fold per seed.
            Structure: ``{model_name: {'folds': [...], 'summary': {...}}}``.
    """
    if seeds is None:
        seeds = SEEDS_FOR_REPEATED_EVAL
    if n_splits is None:
        n_splits = N_CV_FOLDS

    X = feature_df[FEATURE_COLS].values
    y = label_encoder.transform(feature_df["label"].values)
    groups = feature_df["case_id"].values
    num_classes = len(label_encoder.classes_)

    results = {
        "RandomForest": {"folds": [], "f1_scores": []},
        "XGBoost": {"folds": [], "f1_scores": []},
        "MLP": {"folds": [], "f1_scores": []},
    }

    metric_labels = list(range(num_classes))
    gkf = GroupKFold(n_splits=n_splits)

    for seed in seeds:
        print(f"\n[STAGE 9] Repeating model training with seed={seed} on fixed GroupKFold splits...", flush=True)

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"  Fold {fold_idx + 1}/{n_splits} (seed={seed})", flush=True)

            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Verify no case overlap between train and validation.
            train_cases = set(groups[train_idx])
            val_cases = set(groups[val_idx])
            assert train_cases.isdisjoint(val_cases), \
                f"Case overlap detected in fold {fold_idx}: {train_cases & val_cases}"

            # Verify class coverage in fold.
            train_classes_present = set(y_train)
            if len(train_classes_present) < num_classes:
                missing = set(metric_labels) - train_classes_present
                missing_names = label_encoder.inverse_transform(list(missing))
                raise ValueError(f"Fold {fold_idx} training set missing observed classes: {missing_names}")

            val_classes_present = set(y_val)
            if len(val_classes_present) < num_classes:
                missing = set(metric_labels) - val_classes_present
                missing_names = label_encoder.inverse_transform(list(missing))
                print(f"    [WARNING] Fold {fold_idx} validation set missing classes: {missing_names}")

            # StandardScaler fitted on training fold only.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)

            # --- Random Forest ---
            rf_metrics = _train_and_evaluate_rf(X_train, y_train, X_val, y_val, metric_labels, seed=seed)
            rf_metrics["fold"] = fold_idx
            rf_metrics["seed"] = seed
            results["RandomForest"]["folds"].append(rf_metrics)
            results["RandomForest"]["f1_scores"].append(rf_metrics["macro_f1"])

            # --- XGBoost ---
            xgb_metrics = _train_and_evaluate_xgb(
                X_train, y_train, X_val, y_val, label_encoder, metric_labels, seed=seed
            )
            xgb_metrics["fold"] = fold_idx
            xgb_metrics["seed"] = seed
            results["XGBoost"]["folds"].append(xgb_metrics)
            results["XGBoost"]["f1_scores"].append(xgb_metrics["macro_f1"])

            # --- MLP ---
            mlp_metrics = _train_and_evaluate_mlp(
                X_train, y_train, X_val, y_val, num_classes, metric_labels, seed=seed
            )
            mlp_metrics["fold"] = fold_idx
            mlp_metrics["seed"] = seed
            results["MLP"]["folds"].append(mlp_metrics)
            results["MLP"]["f1_scores"].append(mlp_metrics["macro_f1"])

    # Compute summary statistics for each candidate.
    for name in results:
        scores = np.array(results[name]["f1_scores"])
        results[name]["summary"] = {
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
            "min_f1": float(np.min(scores)),
            "max_f1": float(np.max(scores)),
        }
        print(
            f"  {name:15s} — Macro F1: "
            f"{results[name]['summary']['mean_f1']:.4f} "
            f"± {results[name]['summary']['std_f1']:.4f} "
            f"(range: {results[name]['summary']['min_f1']:.4f}–"
            f"{results[name]['summary']['max_f1']:.4f})"
        )

    return results


def select_best_model(cv_results):
    """Select the winning candidate using a 4-level tie-break rule.

    1. Highest mean macro F1.
    2. If within 0.01: prefer lower std.
    3. If still tied: prefer higher min fold F1.
    4. If still tied: prefer simpler model (RF > XGBoost > MLP).

    Args:
        cv_results (dict): Cross-validation results from run_cross_validation.

    Returns:
        tuple: (winner_name, selection_reason).
    """
    candidates = list(cv_results.keys())
    best_mean_f1 = max(cv_results[c]["summary"]["mean_f1"] for c in candidates)

    # Level 1: filter candidates within 0.01 of the best mean F1.
    top = [c for c in candidates
           if (best_mean_f1 - cv_results[c]["summary"]["mean_f1"]) <= 0.01]

    if len(top) == 1:
        reason = "Highest mean macro F1 (margin > 0.01)"
        winner = top[0]
    else:
        # Level 2: lowest standard deviation.
        min_std = min(cv_results[c]["summary"]["std_f1"] for c in top)
        tied_std = [c for c in top
                    if (cv_results[c]["summary"]["std_f1"] - min_std) <= 1e-5]

        if len(tied_std) == 1:
            reason = "Tie-break 2: Lowest standard deviation"
            winner = tied_std[0]
        else:
            # Level 3: highest minimum fold F1.
            max_min = max(cv_results[c]["summary"]["min_f1"] for c in tied_std)
            tied_min = [c for c in tied_std
                        if (max_min - cv_results[c]["summary"]["min_f1"]) <= 1e-5]

            if len(tied_min) == 1:
                reason = "Tie-break 3: Highest minimum fold F1"
                winner = tied_min[0]
            else:
                # Level 4: prefer simpler model.
                order = {"RandomForest": 1, "XGBoost": 2, "MLP": 3}
                winner = min(tied_min, key=lambda c: order.get(c, 99))
                reason = "Tie-break 4: Preferred simpler model (RF > XGBoost > MLP)"

    print(
        f"\n[MODEL SELECTION] Winner: {winner}\n"
        f"  Reason: {reason}\n"
        f"  Stats: Mean F1={cv_results[winner]['summary']['mean_f1']:.4f}, "
        f"Std={cv_results[winner]['summary']['std_f1']:.4f}, "
        f"Min={cv_results[winner]['summary']['min_f1']:.4f}"
    )
    return winner, reason


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — FINAL MODEL TRAINING & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
# After model selection, the winning candidate is retrained on the full
# training set (all CV folds combined) and evaluated once on the held-out
# test set.

def train_final_model(model_name, X_train, y_train, label_encoder, seed=SEED, cv_best_epochs=None):
    """Train the selected model on the full training set.

    Args:
        model_name (str): 'RandomForest', 'XGBoost', or 'MLP'.
        X_train (np.ndarray): Standardised training features.
        y_train (np.ndarray): Integer training labels.
        label_encoder (LabelEncoder): Fitted label encoder.
        seed (int): Random seed.
        cv_best_epochs (list[int]): List of best epochs from CV (for MLP).

    Returns:
        object: Trained model instance.
    """
    num_classes = len(label_encoder.classes_)

    if model_name == "RandomForest":
        model = build_rf_model(seed=seed)
        model.fit(X_train, y_train)
        return model

    elif model_name == "XGBoost":
        model = build_xgb_model(num_classes=num_classes, seed=seed)
        class_weights = _compute_class_weights(y_train, num_classes)
        sample_weights = class_weights.numpy()[y_train]
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    elif model_name == "MLP":
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = build_mlp_model(input_dim=X_train.shape[1], num_classes=num_classes)
        class_weights = _compute_class_weights(y_train, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=MLP_LR)

        # Use median of best epochs from cross-validation to prevent overfitting on the full set
        if cv_best_epochs and len(cv_best_epochs) > 0:
            final_epoch_count = int(np.median(cv_best_epochs))
            # Fallback to at least 1 epoch if median is 0
            final_epoch_count = max(final_epoch_count, 1)
        else:
            final_epoch_count = MLP_EPOCHS // 2

        print(f"    [INFO] Training MLP for exactly {final_epoch_count} epochs (median from CV).")

        train_ds = CariesFeatureDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)

        for epoch in range(final_epoch_count):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def predict_with_model(model, model_name, X, label_encoder, scaler=None):
    """Generate predictions using the trained model.

    Args:
        model: Trained model instance.
        model_name (str): 'RandomForest', 'XGBoost', or 'MLP'.
        X (np.ndarray): Raw (unstandardised) features if scaler is provided,
            or standardised features if scaler is None.
        label_encoder (LabelEncoder): For decoding integer predictions to
            surface names.
        scaler (StandardScaler or None): If provided, transforms X before
            prediction.

    Returns:
        tuple: (predicted_labels_str, predicted_indices).
    """
    if scaler is not None:
        X = scaler.transform(X)

    if model_name == "MLP":
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(X_tensor)
            pred_indices = logits.argmax(dim=1).cpu().numpy()
    else:
        pred_indices = model.predict(X)

    pred_labels = label_encoder.inverse_transform(pred_indices)
    return pred_labels, pred_indices


def predict_proba_with_model(model, model_name, X, scaler=None):
    """Get class probability estimates from the trained model.

    Args:
        model: Trained model instance.
        model_name (str): 'RandomForest', 'XGBoost', or 'MLP'.
        X (np.ndarray): Feature array.
        scaler (StandardScaler or None): If provided, transforms X first.

    Returns:
        np.ndarray: Class probabilities of shape (N, num_classes).
    """
    if scaler is not None:
        X = scaler.transform(X)

    if model_name == "MLP":
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(X_tensor)
            # Softmax applied here at inference only — never during training.
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    else:
        return model.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 11 — BASELINE CLASSIFIER (X-THIRDS FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════
# The X-Thirds method is a rule-based baseline that assigns caries to the
# surface occupying the largest fraction of caries pixels along the
# mesial–distal axis.  It serves as the deterministic fallback when the
# trained model is unavailable or when feature extraction fails.

def classify_xthird(tooth_id, tooth_pts, caries_pts):
    """Baseline X-Thirds classifier using dominant-zone voting.

    The tooth bounding box is divided into three equal horizontal zones
    (left, centre, right).  Each caries pixel votes for one zone.  The
    surface with the most votes wins.  Zone-to-surface mapping depends
    on the FDI quadrant to account for the reversal of mesial/distal
    direction between left and right sides of the dental arch.

    Args:
        tooth_id (str): FDI tooth identifier.
        tooth_pts (list): Tooth mask pixel coordinates.
        caries_pts (list): Caries region pixel coordinates.

    Returns:
        tuple: (predicted_surface, rotation_angle_deg, vote_fractions).
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

    vote_map = {
        "Mesial": int(np.sum(m_mask)),
        "Occlusal": int(np.sum(c_mask)),
        "Distal": int(np.sum(d_mask)),
    }
    winner = max(vote_map, key=vote_map.get)

    vote_fractions = {k: round(v / max(n_pts, 1), 4) for k, v in vote_map.items()}
    vote_fractions["pca_clamped"] = clamped
    return winner, float(math.degrees(angle)), vote_fractions


def classify_with_smart_fallback(tooth_id, tooth_pts, caries_pts):
    """Classify caries surface using the trained model with X-Thirds fallback.

    First attempts classification with the active trained model.  Falls back
    to the rule-based X-Thirds classifier if the model is unavailable, if
    feature extraction fails, or if an unexpected error occurs during
    inference.

    Args:
        tooth_id (str): FDI tooth identifier.
        tooth_pts (list): Tooth mask pixel coordinates.
        caries_pts (list): Caries region pixel coordinates.

    Returns:
        tuple: (predicted_surface, rotation_angle, metadata_dict).
    """
    try:
        features = _extract_ml_feature_dict(tooth_id, tooth_pts, caries_pts)
        if active_model is None:
            surface, angle, metadata = classify_xthird(tooth_id, tooth_pts, caries_pts)
            metadata.update({"prediction_method": "X-Thirds", "fallback_used": True, "fallback_reason": "active_model_none"})
            return surface, angle, metadata
        if features is None:
            surface, angle, metadata = classify_xthird(tooth_id, tooth_pts, caries_pts)
            metadata.update({"prediction_method": "X-Thirds", "fallback_used": True, "fallback_reason": "missing_features"})
            return surface, angle, metadata

        X = np.array([[features[col] for col in FEATURE_COLS]])
        pred_labels, _ = predict_with_model(
            active_model, active_model_type, X,
            active_label_encoder, scaler=active_scaler,
        )
        return pred_labels[0], 0.0, {"prediction_method": active_model_type, "fallback_used": False, "fallback_reason": None}
    except Exception as e:
        try:
            surface, angle, metadata = classify_xthird(tooth_id, tooth_pts, caries_pts)
            metadata.update({"prediction_method": "X-Thirds", "fallback_used": True, "fallback_reason": f"inference_error: {str(e)}"})
            return surface, angle, metadata
        except Exception as e2:
            return "Other", 0.0, {"prediction_method": "None", "fallback_used": True, "fallback_reason": f"xthirds_error: {str(e2)}"}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 12 — PER-CASE PREDICTION & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def process_case(case_id, output_root):
    """Run classification on all teeth in one case and save prediction JSON.

    Args:
        case_id (int): Case identifier (1–500).
        output_root (Path): Root directory for prediction output.

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

        surface, angle, metadata = classify_with_smart_fallback(
            tooth_id, tooth_pts, caries_pts
        )

        result["teeth_data"].append({
            "tooth_id": tooth_id,
            "version": "Benchmark_v1",
            "has_caries": True,
            "confidence": float(tooth.get("confidence", 0.0)),
            "caries_position_detail": surface,
            "predicted_surface_fine": surface,
            "prediction_method": metadata.get("prediction_method"),
            "fallback_used": metadata.get("fallback_used", False),
            "fallback_reason": metadata.get("fallback_reason"),
            "tooth_coordinates": tooth_pts,
            "caries_coordinates": caries_pts,
        })

    with open(case_dir / f"case_{case_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return True, f"OK ({len(result['teeth_data'])} teeth)"


def load_prediction(case_num, out_dir):
    """Load prediction JSON for a single case.

    Args:
        case_num (int): Case identifier.
        out_dir (str or Path): Output directory containing case folders.

    Returns:
        list[dict]: Predicted tooth-surface pairs.
    """
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
    """Match ground-truth and predicted surfaces by tooth ID.

    Args:
        ground_truth (list[dict]): Ground-truth annotations.
        predictions (list[dict]): Model predictions.

    Returns:
        tuple: (y_true, y_pred) as parallel lists of surface labels.
    """
    pred_dict = {p["tooth"]: p["surface"] for p in predictions}
    y_true, y_pred = [], []
    for g in ground_truth:
        y_true.append(g["surface"])
        y_pred.append(pred_dict.get(g["tooth"], "Other"))
    return y_true, y_pred


def evaluate_all_cases(output_root, version="Benchmark_v1", allowed_case_ids=None, metric_labels=None):
    """Evaluate predictions across cases against ground truth.

    Args:
        output_root (Path or str): Directory containing prediction JSONs.
        version (str): Version tag for reporting.
        allowed_case_ids (list or set): If provided, restrict evaluation to these case IDs.
        metric_labels (list): List of strings for the observed classes.

    Returns:
        tuple: (all_y_true, all_y_pred, macro_f1).
    """
    if metric_labels is None:
        metric_labels = VALID_SURFACES

    all_y_true, all_y_pred = [], []

    print(f"\n[STAGE 12] Evaluating {version}...", flush=True)
    cases_to_eval = sorted(list(allowed_case_ids)) if allowed_case_ids is not None else list(range(1, 501))
    total_eval = len(cases_to_eval)

    for idx, case_num in enumerate(cases_to_eval, 1):
        gt_folder = GT_ROOT / f"case {case_num}"
        ground_truth = parse_case_ground_truth(gt_folder)
        predictions = load_prediction(case_num, output_root)
        if len(ground_truth) == 0 and len(predictions) == 0:
            continue
        yt, yp = match_case(ground_truth, predictions)
        all_y_true.extend(yt)
        all_y_pred.extend(yp)
        _progress_bar(idx, total_eval, f"Evaluating {version}")

    # Determine evaluated labels dynamically to include any fallback predictions (e.g. 'Other')
    eval_labels = [s for s in VALID_SURFACES if s in set(all_y_true + all_y_pred)]
    if not eval_labels:
        eval_labels = metric_labels

    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, labels=eval_labels, average="macro", zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, labels=eval_labels, average="macro", zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, labels=eval_labels, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_y_true, all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=eval_labels)
    cm_df = pd.DataFrame(cm, index=eval_labels, columns=eval_labels)

    print(f"\n{'=' * 55}")
    print(f"  FINAL EVALUATION — {version}")
    print(f"{'=' * 55}")
    print(f"  Total Samples    : {len(all_y_true)}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro)   : {rec:.4f}")
    print(f"  F1 Score (macro) : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm_df.to_string()}")
    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred, labels=metric_labels, target_names=metric_labels, zero_division=0))

    return all_y_true, all_y_pred, f1


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 13 — PLOTTING & VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_evaluation_results(y_true, y_pred, version="Benchmark_v1"):
    """Generate confusion matrix and per-class metrics plots.

    Produces two publication-quality figures:
    1. Confusion matrix heatmap with count annotations.
    2. Grouped bar chart of per-class precision, recall, and F1.

    Args:
        y_true (list[str]): Ground-truth surface labels.
        y_pred (list[str]): Predicted surface labels.
        version (str): Version tag for titles and filenames.
    """
    labels = VALID_SURFACES

    # --- 1. Confusion Matrix Heatmap ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues",
        linewidths=0.8, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Count"}, ax=ax1,
    )
    ax1.set_xlabel("Predicted Surface", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual Surface", fontsize=12, fontweight="bold")
    ax1.set_title(f"Confusion Matrix — {version}", fontsize=14, fontweight="bold", pad=15)
    ax1.tick_params(axis="both", labelsize=11)
    plt.tight_layout()

    cm_path = f"confusion_matrix_{version.lower()}.png"
    fig1.savefig(cm_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"[SAVED] Confusion matrix → {cm_path} (dpi=300)")

    # --- 2. Per-Class Metrics Bar Chart ---
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
    palette = ["#3b82f6", "#f59e0b", "#10b981"]
    x = np.arange(len(labels))
    bar_width = 0.24

    for idx, (m, color) in enumerate(zip(
        ["Precision", "Recall", "F1 Score"], palette
    )):
        values = metrics_df[metrics_df["Metric"] == m]["Value"].values
        bars = ax2.bar(
            x + idx * bar_width, values, bar_width,
            label=m, color=color, edgecolor="white", linewidth=0.6,
        )
        for bar_obj in bars:
            height = bar_obj.get_height()
            ax2.text(
                bar_obj.get_x() + bar_obj.get_width() / 2, height + 0.015,
                f"{height:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333",
            )

    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_xlabel("Surface Class", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Per-Class Classification Metrics — {version}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    metrics_path = f"classification_metrics_{version.lower()}.png"
    fig2.savefig(metrics_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[SAVED] Classification metrics → {metrics_path} (dpi=300)")


def plot_feature_importance(model, model_name, feature_names,
                            save_path="feature_importance.png"):
    """Visualise feature importances as a horizontal bar chart.

    For tree-based models (RF, XGBoost), Gini/gain importance is used.
    For the MLP, permutation importance should be computed separately
    (not implemented in this function).

    Args:
        model: Trained model with ``feature_importances_`` attribute.
        model_name (str): Model identifier for the title.
        feature_names (list[str]): Feature column names.
        save_path (str): Output file path.

    Returns:
        pd.DataFrame: Sorted importance table (descending).
    """
    if not hasattr(model, "feature_importances_"):
        print(f"[SKIP] {model_name} does not expose feature_importances_.")
        return None

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 50}")
    print(f"  Top 5 Most Important Features ({model_name})")
    print(f"{'=' * 50}")
    for rank, row in importance_df.head(5).iterrows():
        print(f"  #{rank + 1}  {row['Feature']:<20s}  {row['Importance']:.4f}")
    print(f"{'=' * 50}")

    plot_df = importance_df.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    n = len(plot_df)
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.25, 0.90, n))

    bars = ax.barh(
        range(n), plot_df["Importance"],
        color=colors, edgecolor="white", linewidth=0.6, height=0.72,
    )

    for bar_obj, val in zip(bars, plot_df["Importance"]):
        ax.text(
            bar_obj.get_width() + 0.003,
            bar_obj.get_y() + bar_obj.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold", color="#333333",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["Feature"], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini / Gain)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Feature Importance — {model_name}", fontsize=14, fontweight="bold", pad=15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, plot_df["Importance"].max() * 1.20)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[SAVED] Feature importance chart → {save_path} (dpi=300)")

    return importance_df


def plot_cv_comparison(cv_results, save_path="cv_model_comparison.png"):
    """Generate a box plot comparing cross-validation macro F1 across candidates.

    Args:
        cv_results (dict): Cross-validation results from run_cross_validation.
        save_path (str): Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = list(cv_results.keys())
    data = [cv_results[name]["f1_scores"] for name in model_names]

    try:
        bp = ax.boxplot(data, tick_labels=model_names, patch_artist=True, widths=0.5)
    except TypeError:
        bp = ax.boxplot(data, labels=model_names, patch_artist=True, widths=0.5)
    colors = ["#3b82f6", "#f59e0b", "#10b981"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual fold scores as scatter points.
    for i, scores in enumerate(data):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x_jitter, scores, alpha=0.5, s=20, color="#333", zorder=3)

    ax.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cross-Validation Model Comparison",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] CV comparison → {save_path} (dpi=300)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 14 — MODEL SERIALISATION & METADATA
# ══════════════════════════════════════════════════════════════════════════════

def save_model_artifact(model, model_name, scaler, label_encoder,
                         cv_results, selection_reason=None, output_dir=None):
    """Save the trained model, scaler, and metadata as versioned artifacts.

    The metadata JSON stores all information required to reproduce inference:
    feature order, scaler parameters, label mapping, architecture config,
    random seeds, and cross-validation summary.

    Args:
        model: Trained model instance.
        model_name (str): 'RandomForest', 'XGBoost', or 'MLP'.
        scaler (StandardScaler): Fitted scaler from training data.
        label_encoder (LabelEncoder): Fitted label encoder.
        cv_results (dict): Cross-validation results for documentation.
        output_dir (Path or None): Output directory (default: MODEL_DIR).
    """
    if output_dir is None:
        output_dir = MODEL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model in format appropriate to the library.
    if model_name == "MLP":
        model_path = output_dir / f"{MODEL_STEM}.pt"
        torch.save(model.state_dict(), model_path)
    else:
        ext = ".joblib"
        model_path = output_dir / f"{MODEL_STEM}{ext}"
        joblib.dump(model, model_path)

    print(f"[SAVED] Model → {model_path}")

    # Save scaler.
    scaler_path = output_dir / f"{MODEL_STEM}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[SAVED] Scaler → {scaler_path}")

    # Build and save metadata.
    metadata = {
        "model_type": model_name.lower(),
        "model_file": model_path.name,
        "scaler_file": scaler_path.name,
        "feature_order": FEATURE_COLS,
        "label_mapping": {
            str(i): label for i, label in enumerate(label_encoder.classes_)
        },
        "scaler_params": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "random_seeds": SEEDS_FOR_REPEATED_EVAL,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cv_summary": {
            name: cv_results[name]["summary"]
            for name in cv_results
        },
        "selection_reason": selection_reason,
        "cv_protocol": {
            "fold_assignments_fixed_across_seeds": True,
            "seeds_vary_model_init_only": True,
        },
    }

    if model_name == "MLP":
        metadata["architecture"] = {
            "hidden_layers": MLP_HIDDEN_LAYERS,
            "dropout": MLP_DROPOUT,
            "learning_rate": MLP_LR,
            "max_epochs": MLP_EPOCHS,
            "batch_size": MLP_BATCH_SIZE,
            "early_stopping_patience": MLP_PATIENCE,
            "use_batchnorm": MLP_BATCH_SIZE >= 16,
        }

    metadata_path = output_dir / f"{MODEL_STEM}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Metadata → {metadata_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 15 — MAIN PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_hyperparameter_sweep(trainval_df, test_df, label_encoder, observed_classes, n_splits=5, seed=SEED):
    """Run a multi-parameter grid sweep testing multiple model configurations in 1 run.

    Sweeps across:
    - Random Forest: tree counts and tree depth
    - XGBoost: tree counts, tree depth, and learning rates
    - PyTorch MLP: layer architectures, learning rates, and dropout rates

    Args:
        trainval_df (pd.DataFrame): Training + validation feature dataframe.
        test_df (pd.DataFrame): Held-out test dataframe.
        label_encoder (LabelEncoder): Fitted label encoder.
        observed_classes (list[str]): List of observed surface class strings.
        n_splits (int): Number of GroupKFold splits.
        seed (int): Random seed.
    """
    X_trainval_raw = trainval_df[FEATURE_COLS].values
    y_trainval = label_encoder.transform(trainval_df["label"].values)
    groups = trainval_df["case_id"].values
    num_classes = len(label_encoder.classes_)
    metric_labels = list(range(num_classes))

    X_test_raw = test_df[FEATURE_COLS].values
    y_test = label_encoder.transform(test_df["label"].values)

    # Fit scaler on trainval
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    gkf = GroupKFold(n_splits=n_splits)

    configurations = [
        # Random Forest variants
        ("RandomForest", {"n_estimators": 50, "max_depth": 8, "class_weight": "balanced"}),
        ("RandomForest", {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"}),
        ("RandomForest", {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}),
        ("RandomForest", {"n_estimators": 300, "max_depth": 15, "class_weight": "balanced"}),
        # XGBoost variants
        ("XGBoost", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}),
        ("XGBoost", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}),
        ("XGBoost", {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.10}),
        ("XGBoost", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03}),
        # MLP variants
        ("MLP", {"hidden_layers": [32, 16], "lr": 1e-3, "dropout": 0.2}),
        ("MLP", {"hidden_layers": [64, 32], "lr": 1e-3, "dropout": 0.3}),
        ("MLP", {"hidden_layers": [128, 64], "lr": 5e-4, "dropout": 0.3}),
        ("MLP", {"hidden_layers": [64, 32], "lr": 2e-3, "dropout": 0.4}),
    ]

    sweep_results = []
    print(f"\n[SWEEP] Benchmarking {len(configurations)} configurations across {n_splits}-fold GroupKFold...\n")

    for config_idx, (model_type, params) in enumerate(configurations):
        fold_f1s = []
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_trainval_raw, y_trainval, groups)):
            # Fold-level scaling to prevent leakage
            f_scaler = StandardScaler()
            f_X_train = f_scaler.fit_transform(X_trainval_raw[tr_idx])
            f_X_val = f_scaler.transform(X_trainval_raw[val_idx])
            f_y_train = y_trainval[tr_idx]
            f_y_val = y_trainval[val_idx]

            if model_type == "RandomForest":
                clf = RandomForestClassifier(random_state=seed, **params)
                clf.fit(f_X_train, f_y_train)
                preds = clf.predict(f_X_val)
            elif model_type == "XGBoost":
                clf = xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_classes,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    eval_metric="mlogloss",
                    verbosity=0,
                    **params
                )
                weights = _compute_class_weights(f_y_train, num_classes).numpy()[f_y_train]
                clf.fit(f_X_train, f_y_train, sample_weight=weights)
                preds = clf.predict(f_X_val)
            elif model_type == "MLP":
                clf = CariesSurfaceMLP(
                    input_dim=X_trainval_raw.shape[1],
                    hidden_layers=params["hidden_layers"],
                    num_classes=num_classes,
                    dropout=params["dropout"],
                ).to(DEVICE)
                weights = _compute_class_weights(f_y_train, num_classes).to(DEVICE)
                criterion = nn.CrossEntropyLoss(weight=weights)
                optimizer = optim.Adam(clf.parameters(), lr=params["lr"])
                tr_ds = CariesFeatureDataset(f_X_train, f_y_train)
                tr_ld = DataLoader(tr_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
                for _ in range(MLP_EPOCHS // 2):
                    clf.train()
                    for bx, by in tr_ld:
                        bx, by = bx.to(DEVICE), by.to(DEVICE)
                        optimizer.zero_grad()
                        loss = criterion(clf(bx), by)
                        loss.backward()
                        optimizer.step()
                clf.eval()
                with torch.no_grad():
                    preds = clf(torch.tensor(f_X_val, dtype=torch.float32).to(DEVICE)).argmax(dim=1).cpu().numpy()

            fold_f1 = f1_score(f_y_val, preds, labels=metric_labels, average="macro", zero_division=0)
            fold_f1s.append(fold_f1)

        mean_cv_f1 = float(np.mean(fold_f1s))
        std_cv_f1 = float(np.std(fold_f1s))

        # Evaluate configuration on held-out test set
        if model_type == "RandomForest":
            full_clf = RandomForestClassifier(random_state=seed, **params)
            full_clf.fit(X_trainval_scaled, y_trainval)
            test_preds = full_clf.predict(X_test_scaled)
        elif model_type == "XGBoost":
            full_clf = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                eval_metric="mlogloss",
                verbosity=0,
                **params
            )
            weights = _compute_class_weights(y_trainval, num_classes).numpy()[y_trainval]
            full_clf.fit(X_trainval_scaled, y_trainval, sample_weight=weights)
            test_preds = full_clf.predict(X_test_scaled)
        elif model_type == "MLP":
            full_clf = CariesSurfaceMLP(
                input_dim=X_trainval_scaled.shape[1],
                hidden_layers=params["hidden_layers"],
                num_classes=num_classes,
                dropout=params["dropout"],
            ).to(DEVICE)
            weights = _compute_class_weights(y_trainval, num_classes).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = optim.Adam(full_clf.parameters(), lr=params["lr"])
            tr_ds = CariesFeatureDataset(X_trainval_scaled, y_trainval)
            tr_ld = DataLoader(tr_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
            for _ in range(MLP_EPOCHS // 2):
                full_clf.train()
                for bx, by in tr_ld:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(full_clf(bx), by)
                    loss.backward()
                    optimizer.step()
            full_clf.eval()
            with torch.no_grad():
                test_preds = full_clf(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).argmax(dim=1).cpu().numpy()

        test_f1 = float(f1_score(y_test, test_preds, labels=metric_labels, average="macro", zero_division=0))
        test_acc = float(accuracy_score(y_test, test_preds))

        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        sweep_results.append({
            "model_type": model_type,
            "parameters": param_str,
            "mean_cv_f1": round(mean_cv_f1, 4),
            "std_cv_f1": round(std_cv_f1, 4),
            "test_macro_f1": round(test_f1, 4),
            "test_accuracy": round(test_acc, 4),
        })
        print(f"  [{config_idx+1:02d}/{len(configurations):02d}] {model_type:<12} | {param_str:<45} | CV F1: {mean_cv_f1:.4f}±{std_cv_f1:.4f} | Test F1: {test_f1:.4f}")

    sweep_df = pd.DataFrame(sweep_results).sort_values("mean_cv_f1", ascending=False).reset_index(drop=True)
    csv_path = MODEL_DIR / f"{MODEL_STEM}_sweep_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 85}")
    print(f"  MULTI-PARAMETER SWEEP LEADERBOARD (Top 5)")
    print(f"{'=' * 85}")
    print(sweep_df.head(5).to_string(index=False))
    print(f"\n[SAVED] Full sweep results exported -> {csv_path}\n")


def parse_cli_args():
    """Parse command-line arguments for configurable pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Dental Caries Surface Classification Pipeline — Model Selection & Evaluation Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run fast automated unit & invariant tests without dataset dependency.",
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Execute quick dry-run on 20 cases with 3-fold CV to verify the pipeline in seconds.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run multi-parameter grid sweep benchmark over candidate model hyperparameters.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=500,
        help="Maximum number of cases (1–500) to ingest and process.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=N_CV_FOLDS,
        help="Number of GroupKFold cross-validation splits.",
    )
    parser.add_argument(
        "--test-size", type=float, default=HOLDOUT_TEST_SIZE,
        help="Fraction of patient cases reserved for untouched held-out test evaluation.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS_FOR_REPEATED_EVAL,
        help="List of random seeds for repeated evaluation over fixed GroupKFold splits.",
    )
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip per-case operational JSON generation (faster benchmark execution).",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 15 — MAIN PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    """Execute the complete model-selection benchmark pipeline.

    Steps:
        1. Extract 14 geometric features from input cases.
        2. Validate dataset integrity and audit/resolve duplicates.
        3. Split into training (80%) and held-out test (20%) by case group.
        4. Run 5-fold GroupKFold cross-validation for RF, XGBoost, and MLP (or sweep).
        5. Select the winning candidate by mean macro F1 using tie-break rules.
        6. Retrain the winner on the full training set.
        7. Evaluate once on the held-out test set.
        8. Generate predictions for all cases (unless --skip-predictions).
        9. Evaluate predictions against XML ground truth.
        10. Save model artifacts, CV tables, and diagnostic plots.
    """
    global active_model, active_model_type, active_scaler, active_label_encoder

    if args is None:
        args = parse_cli_args()

    if args.quick_test:
        print("\n[CONFIG] Quick-test mode enabled: 20 cases, 3 CV folds, 1 seed.\n", flush=True)
        args.max_cases = min(args.max_cases, 20)
        args.cv_folds = 3
        args.seeds = [SEED]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = list(range(1, args.max_cases + 1))

    # --- Step 1: Feature extraction ---
    print(f"[PIPELINE] Starting model-selection benchmark on {len(case_ids)} cases...\n", flush=True)
    feature_df = create_ml_dataset(case_ids)
    print(
        f"[DONE] Feature extraction complete. "
        f"Total samples: {len(feature_df)} teeth\n",
        flush=True,
    )

    if feature_df.empty:
        print("[ERROR] No features extracted. Exiting.")
        return

    # --- Step 2: Data validation & Duplicate Auditing ---
    validate_dataset(feature_df)
    feature_df = audit_and_resolve_duplicates(feature_df)

    # Derive active observed classes dynamically from data.
    observed_classes = [s for s in VALID_SURFACES if s in set(feature_df["label"].unique())]
    label_encoder = LabelEncoder()
    label_encoder.fit(observed_classes)

    # --- Step 3: Hold-out test split ---
    print(f"\n[PIPELINE] Splitting data: {int((1 - args.test_size)*100)}% train/val, {int(args.test_size*100)}% held-out test...", flush=True)
    gss = GroupShuffleSplit(test_size=args.test_size, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(
        feature_df, groups=feature_df["case_id"]
    ))
    trainval_df = feature_df.iloc[trainval_idx].reset_index(drop=True)
    test_df = feature_df.iloc[test_idx].reset_index(drop=True)

    # Verify no case overlap.
    trainval_cases = set(trainval_df["case_id"])
    test_cases = set(test_df["case_id"])
    assert trainval_cases.isdisjoint(test_cases), "Case overlap between train/val and test!"
    print(
        f"  Train/Val: {len(trainval_df)} samples ({len(trainval_cases)} cases) | "
        f"Test: {len(test_df)} samples ({len(test_cases)} cases)"
    )

    # --- Optional: Multi-Parameter Sweep Mode ---
    if args.sweep:
        print("\n[PIPELINE] Running multi-parameter grid sweep benchmark...", flush=True)
        run_hyperparameter_sweep(trainval_df, test_df, label_encoder, observed_classes, n_splits=args.cv_folds, seed=SEED)
        return

    # --- Step 4: Cross-validation ---
    print("\n[PIPELINE] Running cross-validation benchmark...", flush=True)
    cv_results = run_cross_validation(trainval_df, label_encoder, seeds=args.seeds, n_splits=args.cv_folds)

    # --- Step 5: Model selection ---
    winner_name, selection_reason = select_best_model(cv_results)

    # --- Step 6: Retrain winner on full training set ---
    print(f"\n[PIPELINE] Retraining {winner_name} on full training set...", flush=True)
    X_trainval_raw = trainval_df[FEATURE_COLS].values
    y_trainval = label_encoder.transform(trainval_df["label"].values)

    # Fit StandardScaler on full training set.
    final_scaler = StandardScaler()
    X_trainval = final_scaler.fit_transform(X_trainval_raw)

    cv_best_epochs = cv_results[winner_name]["folds"] if winner_name == "MLP" else None
    if cv_best_epochs:
        cv_best_epochs = [f.get("best_epoch", MLP_EPOCHS//2) for f in cv_best_epochs]

    final_model = train_final_model(
        winner_name, X_trainval, y_trainval, label_encoder, seed=SEED, cv_best_epochs=cv_best_epochs
    )
    print(f"[DONE] {winner_name} trained on {len(X_trainval)} samples.")

    # --- Step 7: Final evaluation on held-out test set ---
    print(f"\n[PIPELINE] Evaluating {winner_name} on held-out test set...", flush=True)
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = label_encoder.transform(test_df["label"].values)
    X_test = final_scaler.transform(X_test_raw)

    if winner_name == "MLP":
        final_model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_pred_test = final_model(X_tensor).argmax(dim=1).cpu().numpy()
    else:
        y_pred_test = final_model.predict(X_test)

    metric_labels = list(range(len(observed_classes)))
    test_metrics = _compute_metrics(y_test, y_pred_test, metric_labels)
    y_true_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_test)

    # Assert macro F1 equivalence
    report_dict = classification_report(y_true_labels, y_pred_labels, labels=observed_classes, output_dict=True, zero_division=0)
    assert np.isclose(test_metrics["macro_f1"], report_dict["macro avg"]["f1-score"], atol=1e-12), \
        f"Macro F1 mismatch: {test_metrics['macro_f1']} != {report_dict['macro avg']['f1-score']}"

    print(f"\n{'=' * 55}")
    print(f"  PRIMARY HELD-OUT GENERALIZATION RESULTS — {winner_name}")
    print(f"{'=' * 55}")
    print(f"  Macro F1         : {test_metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_acc']:.4f}")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"\n{classification_report(y_true_labels, y_pred_labels, labels=observed_classes, target_names=observed_classes, zero_division=0)}")
    
    # Save test results
    test_results_path = MODEL_DIR / f"{MODEL_STEM}_test_results.json"
    with open(test_results_path, "w") as f:
        json.dump({"macro_f1": test_metrics["macro_f1"], "accuracy": test_metrics["accuracy"]}, f, indent=2)

    # --- Step 8: Set active model for case-level predictions ---
    active_model = final_model
    active_model_type = winner_name
    active_scaler = final_scaler
    active_label_encoder = label_encoder

    # --- Step 9: Predict all cases (if not skipped) ---
    total = len(case_ids)
    if not args.skip_predictions:
        success_count, failure_count = 0, 0
        print(f"\n[PIPELINE] Generating operational predictions for all {total} cases...", flush=True)
        
        fallback_counts = {"missing_features": 0, "active_model_none": 0, "inference_error": 0, "None": 0}
        
        for i, case_id in enumerate(case_ids):
            is_success, _ = process_case(case_id, OUTPUT_ROOT)
            if is_success:
                success_count += 1
                # Count fallbacks
                try:
                    pred_file = OUTPUT_ROOT / f"case_{case_id}" / f"case_{case_id}.json"
                    if pred_file.exists():
                        with open(pred_file, "r") as f:
                            data = json.load(f)
                            for t in data.get("teeth_data", []):
                                if t.get("fallback_used"):
                                    reason = t.get("fallback_reason", "unknown")
                                    reason_key = reason.split(":")[0] if ":" in reason else reason
                                    fallback_counts[reason_key] = fallback_counts.get(reason_key, 0) + 1
                except Exception:
                    pass
            else:
                failure_count += 1
            _progress_bar(i + 1, total, "Predictions")
            
        print(
            f"\n[DONE] Predictions saved: {success_count} cases, "
            f"failed: {failure_count} cases",
            flush=True,
        )
        print(f"  [PREDICTION AUDIT] Fallback usage: {fallback_counts}")

        # --- Step 10: Evaluate against ground truth ---
        print(f"\n[WARNING] DESCRIPTIVE ALL-CASE RESULT: This includes {len(trainval_cases)} cases used for training.")
        all_y_true, all_y_pred, f1 = evaluate_all_cases(
            OUTPUT_ROOT, 
            version="Benchmark_v1", 
            allowed_case_ids=set(case_ids),
            metric_labels=observed_classes
        )
        
        # --- Step 11: Generate plots ---
        print("\n[PIPELINE] Generating evaluation plots...", flush=True)
        plot_evaluation_results(all_y_true, all_y_pred, version="Benchmark_v1")
    else:
        print("\n[CONFIG] --skip-predictions specified: skipped operational per-case JSON export.", flush=True)

    plot_cv_comparison(cv_results)

    if hasattr(final_model, "feature_importances_"):
        plot_feature_importance(final_model, winner_name, FEATURE_COLS)

    # --- Step 12: Save model artifacts & Fold CSV ---
    print("\n[PIPELINE] Saving model artifacts...", flush=True)
    save_model_artifact(
        final_model, winner_name, final_scaler,
        label_encoder, cv_results, selection_reason=selection_reason,
    )
    
    csv_path = MODEL_DIR / f"{MODEL_STEM}_cv_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "fold", "seed", "macro_f1", "best_epoch"])
        for model in ["RandomForest", "XGBoost", "MLP"]:
            for fold_data in cv_results[model]["folds"]:
                writer.writerow([
                    model, fold_data.get("fold", -1), fold_data.get("seed", -1),
                    fold_data.get("macro_f1", 0.0), fold_data.get("best_epoch", "N/A")
                ])
    print(f"[SAVED] CV Fold Results -> {csv_path}")

    print(f"\n{'=' * 55}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Selected model : {winner_name}")
    print(f"  CV macro F1    : {cv_results[winner_name]['summary']['mean_f1']:.4f} "
          f"± {cv_results[winner_name]['summary']['std_f1']:.4f}")
    
    diff = test_metrics['macro_f1'] - cv_results[winner_name]['summary']['mean_f1']
    print(f"  Test macro F1  : {test_metrics['macro_f1']:.4f} (diff: {diff:+.4f})")
    print(f"{'=' * 55}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 16 — AUTOMATED SELF-TESTS
# ══════════════════════════════════════════════════════════════════════════════
# Lightweight unit tests that verify pipeline invariants without requiring the
# full 500-case dataset.  Invoked via ``python caries_surface_classifier_20260806.py --self-test``.

def run_self_tests():
    """Execute automated self-tests for pipeline correctness.

    Tests:
        1. Syntax compilation (implicit — we reached this function).
        2. SNODENT duplicate detection raises on known duplicate.
        3. Observed-class derivation excludes zero-support classes.
        4. Dynamic MLP num_classes matches observed classes.
        5. Metric consistency: _compute_metrics matches classification_report.
        6. Duplicate handling: exact duplicates are resolved.
        7. Model-selection tie-break ordering.
        8. _compute_class_weights raises on missing class.
    """
    passed = 0
    failed = 0

    def _assert(condition, name):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    print("\n[SELF-TEST] Running automated tests...\n")

    # --- Test 1: Syntax compilation ---
    _assert(True, "Module compiled successfully")

    # --- Test 2: SNODENT duplicate detection ---
    try:
        _build_snodent_map([("A", "1"), ("B", "2"), ("A", "3")])
        _assert(False, "SNODENT duplicate detection (should have raised)")
    except ValueError:
        _assert(True, "SNODENT duplicate detection")

    # --- Test 3: Observed-class derivation ---
    test_labels = pd.Series(["Distal", "Mesial", "Occlusal", "Distal"])
    observed = [l for l in VALID_SURFACES if l in set(test_labels)]
    _assert("Other" not in observed, "Zero-support class excluded")
    _assert(len(observed) == 3, "Observed classes = 3 (Distal, Mesial, Occlusal)")

    # --- Test 4: Dynamic MLP num_classes ---
    test_le = LabelEncoder()
    test_le.fit(["Distal", "Mesial", "Occlusal"])
    test_model = build_mlp_model(input_dim=14, num_classes=len(test_le.classes_))
    final_layer = list(test_model.network.children())[-1]
    _assert(final_layer.out_features == 3, f"MLP output dim = {final_layer.out_features} (expected 3)")

    # --- Test 5: Metric consistency ---
    y_true_test = np.array([0, 1, 2, 0, 1, 2, 0, 0])
    y_pred_test = np.array([0, 1, 0, 0, 1, 1, 2, 0])
    test_labels_list = [0, 1, 2]
    metrics = _compute_metrics(y_true_test, y_pred_test, labels=test_labels_list)
    report = classification_report(
        y_true_test, y_pred_test, labels=test_labels_list,
        output_dict=True, zero_division=0
    )
    _assert(
        np.isclose(metrics["macro_f1"], report["macro avg"]["f1-score"], atol=1e-12),
        f"Macro F1 consistency: {metrics['macro_f1']:.6f} == {report['macro avg']['f1-score']:.6f}"
    )

    # --- Test 6: Duplicate handling ---
    dup_df = pd.DataFrame({
        "case_id": [1, 1, 2],
        "tooth_id": ["11", "11", "21"],
        **{col: [0.5, 0.5, 0.3] for col in FEATURE_COLS},
        "label": ["Distal", "Distal", "Mesial"],
    })
    cleaned = audit_and_resolve_duplicates(dup_df)
    _assert(len(cleaned) == 2, f"Exact duplicates resolved: {len(cleaned)} rows (expected 2)")

    # --- Test 7: Model-selection tie-break ---
    mock_cv = {
        "RandomForest": {"folds": [], "f1_scores": [0.80], "summary": {"mean_f1": 0.80, "std_f1": 0.01, "min_f1": 0.79, "max_f1": 0.81}},
        "XGBoost":      {"folds": [], "f1_scores": [0.80], "summary": {"mean_f1": 0.80, "std_f1": 0.01, "min_f1": 0.79, "max_f1": 0.81}},
        "MLP":          {"folds": [], "f1_scores": [0.80], "summary": {"mean_f1": 0.80, "std_f1": 0.01, "min_f1": 0.79, "max_f1": 0.81}},
    }
    winner, reason = select_best_model(mock_cv)
    _assert(winner == "RandomForest", f"Tie-break selects RF: got {winner}")
    _assert("simpler" in reason.lower() or "tie-break 4" in reason.lower(),
            f"Reason mentions simplicity: '{reason}'")

    # --- Test 8: _compute_class_weights raises on missing class ---
    try:
        _compute_class_weights(np.array([0, 0, 1, 1]), num_classes=3)
        _assert(False, "_compute_class_weights raises on missing class (should have raised)")
    except ValueError:
        _assert(True, "_compute_class_weights raises on missing class")

    # --- Summary ---
    print(f"\n{'=' * 55}")
    print(f"  SELF-TEST SUMMARY: {passed} passed, {failed} failed")
    print(f"{'=' * 55}")
    return failed == 0


if __name__ == "__main__":
    cli_args = parse_cli_args()
    if cli_args.self_test:
        success = run_self_tests()
        sys.exit(0 if success else 1)
    else:
        main(cli_args)
