"""Image-domain caries surface classification pipeline (single file).

Scope / anti-skew contract
--------------------------
* INPUT  : panoramic X-ray image only, plus tooth segments + FDI numbers produced
  by the two Tooth segmentation models:
    - ``Tooth_seg_pano_20250319.pt`` (YOLO pano segmentation, 32 FDI classes)
    - ``Tooth_seg_crop_20250424.pth`` (Detectron2 Mask R-CNN crop refiner, 1 class)
  ``caries_detect.pt`` is NEVER used here (it carries the old training-serving
  skew: XML-polygon training vs. bbox-grid serving).
* FORBIDDEN AS INPUT: ``caries_coordinates`` / ``caries_pixels`` /
  ``caries_percentage`` from ``dental_analysis_output/*_caries_mapping.json``.
  Those fields are lesion-location information. The classifier must learn the
  lesion appearance from IMAGE INTENSITY/TEXTURE inside the tooth mask, because
  a tooth mask alone carries no caries signal (hence the old 13 geometric mask
  features cannot work here).
* LABEL ONLY: XML annotations (``data/500 cases with annotation/case */*.xml``)
  or equivalently ``data/caries_per_case_summary.csv`` (``Surface_Details``,
  coordinate-free). A tooth may carry several surfaces at once -> multi-label
  target ``[Distal, Mesial, Occlusal]``. ``caries_coordinates`` are never read,
  never featurized; :func:`assert_no_leakage` raises if they ever appear in a
  model input dict.

Why image, not mask shape
-------------------------
With only (segment, FDI) as input, mask geometry describes the TOOTH, not the
LESION. The lesion signal lives in X-ray intensity/texture (radiolucency), so
a CNN looking at the masked tooth crop is the primary predictor. Contrast
enhancement (CLAHE) is a preprocessing step, not a replacement for the CNN.

Stages (all in this file)
-------------------------
1. Preprocessing: mask-based crop -> background masking -> CLAHE
   (``cv2.createCLAHE``) -> per-tooth z-score intensity norm -> resize+pad 224.
2. Orientation norm: PCA alignment on the TOOTH MASK + FDI-quadrant based
   mesial/distal canonicalization (mesial always left).
3. Model: pretrained CNN (ResNet50 / EfficientNet-B0 / DenseNet121) fine-tuned,
   sigmoid multi-label head + BCE (or focal loss). Optional late fusion of FDI
   embedding + mask geometry. RF-on-texture baseline (LBP/HOG-lite + stats).
4. Training/eval: labels from summary-CSV/XML only; GroupShuffleSplit /
   GroupKFold BY CASE (no patient leakage); pos_weight/focal imbalance
   handling; brightness/contrast + small-rotation aug (NO h-flip);
   Recall/Precision/F1/PR-AUC per surface; thresholds tuned on a nested
   tune-half and reported on a disjoint test-half (no threshold optimism),
   targeting recall>=0.80.
5. Ablation: ``off`` (no CLAHE, no PCA) vs ``clahe`` vs ``clahe_pca``.

Usage (sp_project env) -------------------------------------------------------
    D:/Users/jaopi/anaconda3/envs/sp_project/python.exe pipeline.py --help
    ... pipeline.py build-dataset [--seg-source week2json|live] [--max-cases N]
    ... pipeline.py find-epoch  (stage 1: epoch budget on tune-half, early stop)
    ... pipeline.py train --ablation clahe_pca --backbone efficientnet_b0 --loss focal --cv stratified-groupkfold
                                (stage 2: fixed budget from find-epoch -> mean±std report)
    ... pipeline.py ablate            # trains off / clahe / clahe_pca, compares
    ... pipeline.py baseline          # RF-on-texture baseline
    ... pipeline.py predict --image "data/500 cases with annotation/case 1/case_1.png" --weights outputs/best_clahe_pca.pt

Outputs: ./cache (tooth crops), ./outputs (weights, metrics, thresholds).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial as _partial
from pathlib import Path

import numpy as np

# Flush every log line: long Graphics Processing Unit runs stream progress even
# when standard output is block-buffered (same effect as python -u, for our logs).
print = _partial(print, flush=True)  # noqa: A001 (module-local rebinding only)

# Third-party deprecation noise from pinned deps (do NOT edit site-packages):
# 1. detectron2 model_zoo imports deprecated pkg_resources (UserWarning).
# 2. fvcore checkpoint calls torch.load(weights_only=False) (FutureWarning);
#    weights are our own trusted local files, so the default loader is fine.
# 3. detectron2 internals call torch.meshgrid without indexing=... (UserWarning).
# Scoped to these exact messages so our own warnings (e.g. sklearn) stay visible.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*",
                        category=UserWarning)
warnings.filterwarnings("ignore",
                        message=r"You are using `torch\.load` with `weights_only=False`.*",
                        category=FutureWarning)
warnings.filterwarnings("ignore", message=r"torch\.meshgrid.*indexing argument.*",
                        category=UserWarning)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

# ----------------------------------------------------------------------------
# Constants / paths
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
SP_ROOT = Path(r"D:\Mahidol University\SP")
PANO_WEIGHTS = SP_ROOT / "data" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_pano_20250319.pt"
CROP_WEIGHTS = SP_ROOT / "data" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_crop_20250424.pth"
IMG_ROOT = SP_ROOT / "data" / "500 cases with annotation"
LABEL_SUMMARY_CSV = SP_ROOT / "data" / "caries_per_case_summary.csv"
WEEK2_SEG_DIR = SP_ROOT / "week2-Tooth Detection & Segmentation" / "500-segmentation+recognition"

SURFACES = ["Distal", "Mesial", "Occlusal"]  # fixed class order everywhere
SURF2IDX = {s: i for i, s in enumerate(SURFACES)}

IMG_SIZE = 224
CROP_PAD = 20          # PAD around tooth mask (same as Example-checkpoint.ipynb)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
MAX_TILT_DEG = 45.0
LEFT_BOUND = 0.40      # X-thirds convention (kept for documentation/orientation)
RIGHT_BOUND = 0.60

FORBIDDEN_INPUT_KEYS = {
    "caries_coordinates", "caries_pixels", "caries_percentage",
    "caries_points", "caries_pts", "caries_mask",
}

CACHE_DIR = HERE / "cache"
CASES_CACHE = CACHE_DIR / "cases"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
OUTPUTS_DIR = HERE / "outputs"


def require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python (cv2) is required but not installed.")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Leakage guard
# ----------------------------------------------------------------------------

def assert_no_leakage(d: dict) -> None:
    """Raise if lesion-coordinate fields leak into a model-input dict."""
    bad = FORBIDDEN_INPUT_KEYS.intersection(d.keys())
    if bad:
        raise ValueError(
            f"Training-serving skew guard: forbidden caries-coordinate keys as "
            f"model input: {sorted(bad)}. Labels only, never features."
        )


# ----------------------------------------------------------------------------
# FDI helpers
# ----------------------------------------------------------------------------

def fdi_to_int(fdi) -> int | None:
    try:
        digits = re.sub(r"\D", "", str(fdi))
        if len(digits) < 2:
            return None
        return int(digits[:2])
    except (ValueError, TypeError):
        return None


def is_upper_jaw(fdi) -> bool:
    v = fdi_to_int(fdi)
    return v is not None and int(str(v)[0]) in (1, 2)


def get_quadrant(fdi) -> int:
    v = fdi_to_int(fdi)
    return int(str(v)[0]) if v is not None else 4


# ----------------------------------------------------------------------------
# Labels (coordinate-free). Priority: caries_per_case_summary.csv.
# ----------------------------------------------------------------------------

def parse_surface_details(text: str) -> dict[str, list[int]]:
    """Parse e.g. 'Distal: [15, 35, 45] | Occlusal: [46]' -> {surface: [fdi]}."""
    out: dict[str, list[int]] = {}
    for m in re.finditer(r"([A-Za-z]+)\s*:\s*\[([^\]]*)\]", text or ""):
        name = m.group(1).strip().capitalize()
        if name not in SURF2IDX:
            continue
        teeth: list[int] = []
        for tok in m.group(2).split(","):
            v = fdi_to_int(tok)
            if v is not None:
                teeth.append(v)
        out[name] = teeth
    return out


def load_label_map(label_csv: Path = LABEL_SUMMARY_CSV) -> dict[tuple[int, int], list[int]]:
    """Return {(case_id, fdi): [d, m, o] multi-hot}. No coordinates involved."""
    label_map: dict[tuple[int, int], list[int]] = {}
    with open(label_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            m_case = re.search(r"(\d+)", str(row.get("case_id", "")))
            if not m_case:
                continue
            case_id = int(m_case.group(1))
            per_surface = parse_surface_details(str(row.get("Surface_Details", "")))
            per_tooth: dict[int, set[str]] = defaultdict(set)
            for surf, teeth in per_surface.items():
                for t in teeth:
                    per_tooth[t].add(surf)
            for fdi, surfs in per_tooth.items():
                vec = [0, 0, 0]
                for s in surfs:
                    vec[SURF2IDX[s]] = 1
                label_map[(case_id, fdi)] = vec
    return label_map


# --- XML fallback (reads tooth+surface codes only, ignores polygon coords) ---

_AIM_NS = "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"
_ISO_NS = "uri:iso.org:21090"
_NS = {"aim": _AIM_NS, "iso": _ISO_NS}
_SNODENT_SURFACE = {
    "144414D": "Occlusal", "146014D": "Distal", "145374D": "Mesial",
    "144474D": "Occlusal", "146074D": "Distal", "145434D": "Mesial",
}
_DISPLAY_SURFACE = {
    "occlusal surface": "Occlusal", "distal surface": "Distal",
    "mesial surface": "Mesial",
}
_TOOTH_WORDS = {
    "central incisor": 1, "lateral incisor": 2, "canine": 3,
    "first premolar": 4, "second premolar": 5, "first molar": 6,
    "second molar": 7, "third molar": 8,
}


def _display_to_fdi(display: str) -> int | None:
    dn = display.lower()
    quad = 1 if ("upper" in dn and "right" in dn) else (
        2 if ("upper" in dn and "left" in dn) else (
            3 if ("lower" in dn and "left" in dn) else (
                4 if ("lower" in dn and "right" in dn) else 0)))
    if not quad:
        return None
    for name, pos in _TOOTH_WORDS.items():
        if name in dn:
            return int(f"{quad}{pos}")
    return None


def parse_aim_xml_label(xml_path: Path) -> tuple[int | None, str | None]:
    """Extract (FDI, Surface) label from one AIM XML. Polygon coords ignored."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return None, None
    anns = root.find("aim:imageAnnotations", _NS)
    if anns is None:
        return None, None
    ann = anns.find("aim:ImageAnnotation", _NS)
    if ann is None:
        return None, None
    tooth: int | None = None
    surface: str | None = None
    coll = ann.find("aim:imagingPhysicalEntityCollection", _NS)
    if coll is not None:
        ent = coll.find("aim:ImagingPhysicalEntity", _NS)
        chars = ent.find("aim:imagingPhysicalEntityCharacteristicCollection", _NS) if ent is not None else None
        if chars is not None:
            for ch in chars.findall("aim:ImagingPhysicalEntityCharacteristic", _NS):
                q = ch.find("aim:questionIndex", _NS)
                qv = q.get("value", "") if q is not None else ""
                tc = ch.find("aim:typeCode", _NS)
                if tc is None:
                    continue
                code, dn_el = tc.get("code", ""), tc.find("iso:displayName", _NS)
                disp = dn_el.get("value", "") if dn_el is not None else ""
                if qv == "0":
                    tooth = _display_to_fdi(disp)
                elif qv == "1":
                    surface = _SNODENT_SURFACE.get(code) or _DISPLAY_SURFACE.get(disp.lower(), None)
    return tooth, surface


def load_label_map_from_xml(img_root: Path = IMG_ROOT) -> dict[tuple[int, int], list[int]]:
    """Union multi-label map from per-case XMLs (label codes only)."""
    per_tooth: dict[tuple[int, int], set[str]] = defaultdict(set)
    for case_dir in sorted(img_root.glob("case *")):
        m = re.search(r"(\d+)", case_dir.name)
        if not m:
            continue
        case_id = int(m.group(1))
        for xml in case_dir.glob("*.xml"):
            tooth, surf = parse_aim_xml_label(xml)
            if tooth is not None and surf in SURF2IDX:
                per_tooth[(case_id, tooth)].add(surf)
    return {k: [1 if s in v else 0 for s in SURFACES] for k, v in per_tooth.items()}


# ----------------------------------------------------------------------------
# PCA orientation (tooth MASK only — same 4-rule convention as the RF work)
# ----------------------------------------------------------------------------

def pca_angle(mask_pts: np.ndarray, fdi) -> tuple[np.ndarray, float]:
    """Return (centroid_xy, rotation_rad) canonicalizing the tooth long axis."""
    require_cv2()
    pts = np.asarray(mask_pts, dtype=np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    centered = pts - center
    _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
    # Long (anatomical crown-root) axis = primary eigenvector, ALWAYS.
    # (An earlier revision picked whichever axis was more vertical; past ~45 deg
    #  tilt that silently switches to the SHORT axis, creating a pose
    #  discontinuity between near-identical teeth. Fixed 2026-10-13.)
    vert = eigvecs[0].astype(np.float64).copy()
    upper = is_upper_jaw(fdi)
    if upper:
        if vert[1] < 0:
            vert = -vert
    else:
        if vert[1] > 0:
            vert = -vert
    ang = math.atan2(vert[1], vert[0])
    target = math.pi / 2 if upper else -math.pi / 2
    rot = target - ang
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
    return center, float(rot)


def canonical_flip_needed(fdi) -> bool:
    """True -> mirror horizontally so MESIAL is always on the LEFT.

    X-thirds convention (RF work): quadrants 1&4 have Distal on the left and
    Mesial on the right; quadrants 2&3 are the opposite. Flipping Q1/Q4 once
    puts Mesial on the left for every tooth, so the CNN sees one convention.
    """
    return get_quadrant(fdi) in (1, 4)


# ----------------------------------------------------------------------------
# Preprocessing (image domain; NO caries coordinates anywhere)
# ----------------------------------------------------------------------------

def apply_clahe(gray: np.ndarray, clip: float = CLAHE_CLIP,
                tile: tuple[int, int] = CLAHE_TILE) -> np.ndarray:
    require_cv2()
    g = np.ascontiguousarray(gray.astype(np.uint8))
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile)).apply(g)


def zscore_foreground(gray_f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fg = mask > 0
    out = np.zeros_like(gray_f, dtype=np.float32)
    if fg.sum() < 8:
        return out
    vals = gray_f[fg]
    mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
    out[fg] = (vals - mu) / sd
    return out


def rotate_crop(img: np.ndarray, center_xy: tuple[float, float], angle_deg: float,
                  interp: int | None = None) -> np.ndarray:
    """Rotate image CONTENT by +angle_deg in atan2 image-angle terms.

    NOTE (bugfix 2026-10-13): cv2.getRotationMatrix2D(+a) + warpAffine moves
    content by -a in atan2 (x-right/y-down) angle terms (dot-test verified: dot
    at (70,20) about (50,50) with +30 lands at (52,14), i.e. R(-30)). The PCA
    angle from pca_angle() assumes +angle moves content forward, so we negate
    here to match. Without this, canonicalization DOUBLED tilt error instead
    of removing it (e.g. tooth-44 stayed diagonal in model inputs).
    """
    require_cv2()
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(map(float, center_xy)), -float(angle_deg), 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=interp if interp is not None else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def resize_pad(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Scale longest side to `size`, center-pad with 0. Keeps aspect ratio."""
    require_cv2()
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_LINEAR if img.dtype != bool else cv2.INTER_NEAREST
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    top, bottom = (size - nh) // 2, size - nh - (size - nh) // 2
    left, right = (size - nw) // 2, size - nw - (size - nw) // 2
    return cv2.copyMakeBorder(small, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=0)


@dataclass
class PreConfig:
    use_clahe: bool = True
    use_pca: bool = True
    img_size: int = 224
    input_norm: str = "zscore"   # zscore | imagenet (torchvision canonical)
    aug_plus: bool = False       # extended augmentation (translate/scale/noise)


def preprocess_tooth(gray_crop: np.ndarray, mask_crop: np.ndarray, fdi,
                     cfg: PreConfig = PreConfig()) -> np.ndarray:
    """Full per-tooth preprocessing -> float32 (HxW) model-ready gray image.

    Order: background masking -> CLAHE? -> foreground z-score -> PCA rotate? ->
    FDI mesial-left flip? -> resize+pad 224. Background stays 0 at the end.
    """
    require_cv2()
    gray = np.asarray(gray_crop)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    mask = (np.asarray(mask_crop) > 0).astype(np.uint8)
    if mask.shape != gray.shape[:2]:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    # CLAHE FIRST on the unmasked crop: background zeros would otherwise
    # dominate tile histograms and compress foreground contrast (bugfix).
    work = apply_clahe(gray) if cfg.use_clahe else gray
    if cfg.input_norm == "imagenet":
        norm = (work.astype(np.float32) / 255.0) * mask.astype(np.float32)
    else:
        norm = zscore_foreground(work.astype(np.float32), mask)
    if cfg.use_pca:
        ys, xs = np.where(mask > 0)
        if len(xs) >= 10:
            mpts = np.column_stack([xs, ys]).astype(np.float64)
            center, rot = pca_angle(mpts, fdi)
            norm = rotate_crop(norm, (float(center[0]), float(center[1])),
                               math.degrees(rot))
            mask = rotate_crop(mask.astype(np.float32), (float(center[0]), float(center[1])),
                               math.degrees(rot), cv2.INTER_NEAREST) > 0.5
            norm = norm * mask.astype(np.float32)
            if canonical_flip_needed(fdi):
                norm = np.ascontiguousarray(np.fliplr(norm))
    out = resize_pad(norm, cfg.img_size).astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Tooth detector (live): YOLO pano + Detectron2 crop. Lazy imports.
# ----------------------------------------------------------------------------

@dataclass
class ToothObs:
    fdi: int
    pano_conf: float
    bbox_xyxy: tuple[int, int, int, int]
    crop_coords: tuple[int, int, int, int]
    crop_bgr: np.ndarray = field(repr=False)
    mask_crop: np.ndarray = field(repr=False)  # bool HxW in crop coords


def _bbox_iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return float(inter / u) if u > 0 else 0.0


class ToothDetector:
    """Live segmenter. Heavy deps imported lazily; CPU fallback supported."""

    def __init__(self, pano_weights: Path = PANO_WEIGHTS,
                 crop_weights: Path = CROP_WEIGHTS,
                 device: str = "cuda", pano_conf: float = 0.25,
                 crop_thresh: float = 0.5):
        self.pano_weights = Path(pano_weights)
        self.crop_weights = Path(crop_weights)
        self.device = device
        self.pano_conf = pano_conf
        self.crop_thresh = crop_thresh
        self._pano = None
        self._predictor = None

    def _load(self) -> None:
        if self._pano is None:
            from ultralytics import YOLO

            if not self.pano_weights.is_file():
                raise FileNotFoundError(f"Pano weights not found: {self.pano_weights}")
            self._pano = YOLO(str(self.pano_weights))
        if self._predictor is None:
            if not self.crop_weights.is_file():
                raise FileNotFoundError(f"Crop weights not found: {self.crop_weights}")
            import torch
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.WEIGHTS = str(self.crop_weights)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(self.crop_thresh)
            dev = self.device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
            cfg.MODEL.DEVICE = dev
            self._predictor = DefaultPredictor(cfg)

    def detect(self, image_path: str | Path) -> tuple[np.ndarray, list[ToothObs]]:
        """Run pano YOLO (+ crop refiner) -> (pano_bgr, observations)."""
        require_cv2()
        self._load()
        import torch

        pano = cv2.imread(str(image_path))
        if pano is None:
            raise ValueError(f"Cannot read image: {image_path}")
        H, W = pano.shape[:2]
        res = self._pano.predict(str(image_path), conf=float(self.pano_conf), verbose=False)
        if not res or res[0].masks is None:
            return pano, []
        r = res[0]
        names = getattr(self._pano, "names", {})
        cand: list[dict] = []
        n = len(r.boxes)
        polys = r.masks.xy if r.masks is not None else []
        for i in range(n):
            raw_name = str(names.get(int(r.boxes.cls[i].item()), ""))
            fdi = fdi_to_int(raw_name)
            if fdi is None:
                continue
            box = r.boxes.xyxy[i].cpu().numpy().tolist()
            poly = np.asarray(polys[i], dtype=np.float64) if i < len(polys) else np.zeros((0, 2))
            cand.append({"fdi": fdi, "conf": float(r.boxes.conf[i].item()),
                         "box": box, "poly": poly})
        kept: list[dict] = []
        for c in sorted(cand, key=lambda d: -d["conf"]):
            if all(_bbox_iou(c["box"], k["box"]) <= 0.5 for k in kept):
                kept.append(c)
        obs: list[ToothObs] = []
        for c in kept:
            full_mask = np.zeros((H, W), dtype=np.uint8)
            if len(c["poly"]) >= 3:
                cv2.fillPoly(full_mask, [c["poly"].astype(np.int32).reshape(-1, 1, 2)], 1)
            else:
                x1, y1, x2, y2 = map(int, c["box"])
                full_mask[max(0, y1):y2, max(0, x1):x2] = 1
            ys, xs = np.where(full_mask > 0)
            if len(xs) == 0:
                continue
            x1 = max(0, int(xs.min()) - CROP_PAD)
            y1 = max(0, int(ys.min()) - CROP_PAD)
            x2 = min(W, int(xs.max()) + CROP_PAD)
            y2 = min(H, int(ys.max()) + CROP_PAD)
            crop = pano[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mask_crop = full_mask[y1:y2, x1:x2].astype(bool)
            try:
                with torch.no_grad():
                    out = self._predictor(crop)
                inst = out["instances"].to("cpu")
                if len(inst) > 0:
                    scores = inst.scores.numpy()
                    best = int(np.argmax(scores))
                    m = inst.pred_masks[best].numpy().astype(bool)
                    if m.shape == mask_crop.shape and m.sum() > 50:
                        mask_crop = m
            except Exception:
                pass  # keep pano mask
            obs.append(ToothObs(fdi=c["fdi"], pano_conf=c["conf"],
                                bbox_xyxy=bbox_xywh_to_xyxy(c["box"]),
                                crop_coords=(x1, y1, x2, y2),
                                crop_bgr=crop, mask_crop=mask_crop))
        return pano, obs


def bbox_xywh_to_xyxy(box) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return (x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))


# ----------------------------------------------------------------------------
# Dataset builder -> cache/cases + cache/manifest.json
# ----------------------------------------------------------------------------

def _week2_mask_and_crop(pano_gray: np.ndarray, pix: np.ndarray,
                         pad: int = CROP_PAD):
    H, W = pano_gray.shape[:2]
    pix = np.asarray(pix, dtype=np.int64).reshape(-1, 2)
    keep = (pix[:, 0] >= 0) & (pix[:, 0] < W) & (pix[:, 1] >= 0) & (pix[:, 1] < H)
    pix = pix[keep]
    if len(pix) < 10:
        return None
    x1 = max(0, int(pix[:, 0].min()) - pad)
    y1 = max(0, int(pix[:, 1].min()) - pad)
    x2 = min(W, int(pix[:, 0].max()) + pad)
    y2 = min(H, int(pix[:, 1].max()) + pad)
    crop = pano_gray[y1:y2, x1:x2]
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    ox, oy = pix[:, 0] - x1, pix[:, 1] - y1
    ok = (ox >= 0) & (ox < mask.shape[1]) & (oy >= 0) & (oy < mask.shape[0])
    mask[oy[ok], ox[ok]] = True
    return crop, mask, (x1, y1, x2, y2)


def _save_case_teeth(case_id: int, teeth: list[dict]) -> int:
    d = CASES_CACHE / f"case_{case_id}"
    d.mkdir(parents=True, exist_ok=True)
    for t in teeth:
        cv2.imwrite(str(d / f"tooth_{t['fdi']}_{t['idx']}.png"), t["crop"])
        cv2.imwrite(str(d / f"tooth_{t['fdi']}_{t['idx']}_mask.png"),
                    (t["mask"].astype(np.uint8)) * 255)
    return len(teeth)


def build_dataset(seg_source: str = "week2json", max_cases: int | None = None,
                  label_source: str = "summary",
                  img_root: Path = IMG_ROOT) -> Path:
    """Build the coordinate-free tooth-crop cache. Returns manifest path."""
    require_cv2()
    set_seed(42)
    label_map = (load_label_map() if label_source == "summary"
                 else load_label_map_from_xml(img_root))
    print(f"[labels] {label_source}: {len(label_map)} (case, fdi) positive entries")
    case_dirs = sorted(img_root.glob("case *"),
                       key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))  # type: ignore[union-attr]
    if max_cases:
        case_dirs = case_dirs[:max_cases]
    detector: ToothDetector | None = None
    if seg_source == "live":
        import torch

        detector = ToothDetector(device="cuda" if torch.cuda.is_available() else "cpu")
    manifest: list[dict] = []
    n_pos = n_neg = 0
    for ci, cdir in enumerate(case_dirs, 1):
        m = re.search(r"(\d+)", cdir.name)
        if not m:
            continue
        case_id = int(m.group(1))
        imgs = sorted(cdir.glob("case_*.png")) or sorted(cdir.glob("*.png"))
        if not imgs:
            print(f"[skip] case {case_id}: no png")
            continue
        pano = cv2.imread(str(imgs[0]))
        if pano is None:
            print(f"[skip] case {case_id}: unreadable png")
            continue
        gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        teeth: list[dict] = []
        if seg_source == "week2json":
            # NOTE (provenance, verified by experiment 2026-10-13):
            # week2 JSONs were built from the same two weights, but their crop
            # windows come from boundingRect of the LOW-RES YOLO mask matrix
            # (masks.data, ~160px, NEAREST-upscaled -> ~19px quantization), while
            # live detection derives bounds from full-resolution mask polygons.
            # Measured: bounds differ ~20px mean; mask IoU(week2, live) ~= 0.87.
            # => NEVER mix sources between train and serve. Prefer live build.
            seg_file = None
            for cand in WEEK2_SEG_DIR.glob(f"case {case_id}/case_{case_id}_results.json"):
                seg_file = cand
                break
            if seg_file is None:
                print(f"[skip] case {case_id}: no week2 seg json")
                continue
            seg = json.loads(seg_file.read_text(encoding="utf-8"))
            for j, t in enumerate(seg.get("teeth_data", [])):
                fdi = fdi_to_int(t.get("tooth_id", ""))
                pix = t.get("pixel_coordinates", [])
                if fdi is None or len(pix) < 10:
                    continue
                r = _week2_mask_and_crop(gray, np.asarray(pix))
                if r is None:
                    continue
                crop, mask, coords = r
                teeth.append({"fdi": fdi, "idx": j, "crop": crop, "mask": mask,
                              "bbox": list(coords),
                              "conf": float(t.get("confidence", 0.0))})
        else:
            assert detector is not None
            _, obs = detector.detect(imgs[0])
            for j, o in enumerate(obs):
                g = cv2.cvtColor(o.crop_bgr, cv2.COLOR_BGR2GRAY)
                teeth.append({"fdi": o.fdi, "idx": j, "crop": g,
                              "mask": np.asarray(o.mask_crop, dtype=bool),
                              "bbox": list(o.crop_coords), "conf": o.pano_conf})
        _save_case_teeth(case_id, teeth)
        for t in teeth:
            vec = label_map.get((case_id, t["fdi"]), [0, 0, 0])
            rec = {"case_id": case_id, "fdi": int(t["fdi"]),
                   "label": [int(v) for v in vec],
                   "crop_path": f"cases/case_{case_id}/tooth_{t['fdi']}_{t['idx']}.png",
                   "mask_path": f"cases/case_{case_id}/tooth_{t['fdi']}_{t['idx']}_mask.png",
                   "bbox": [int(v) for v in t["bbox"]],
                   "conf": float(t.get("conf", 0.0))}
            assert_no_leakage(rec)
            manifest.append(rec)
            if any(vec):
                n_pos += 1
            else:
                n_neg += 1
        print(f"[{ci}/{len(case_dirs)}] case {case_id}: {len(teeth)} teeth "
              f"(cum pos={n_pos} neg={n_neg})")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(
        {"meta": {"seg_source": seg_source, "label_source": label_source,
                  "n_pos_teeth": n_pos, "n_neg_teeth": n_neg,
                  "surfaces": SURFACES, "img_size": IMG_SIZE,
                  "note": "inputs are masked tooth crops only; "
                          "caries_coordinates never stored/used"},
         "samples": manifest}, indent=1), encoding="utf-8")
    print(f"[done] {len(manifest)} teeth (pos={n_pos} neg={n_neg}) -> {MANIFEST_PATH}")
    return MANIFEST_PATH


def load_manifest() -> tuple[dict, list[dict]]:
    meta_samples = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return meta_samples["meta"], meta_samples["samples"]


# ----------------------------------------------------------------------------
# Torch dataset (preprocessing applied on the fly -> ablation-safe)
# ----------------------------------------------------------------------------

def _augment_small(img: np.ndarray, rng: np.random.RandomState,
                   plus: bool = False) -> np.ndarray:
    """Brightness/contrast jitter + small rotation (+translate/scale/noise if plus).
    NO horizontal flip (would swap Mesial/Distal labels)."""
    out = img.astype(np.float32)
    out = out * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.08, 0.08))
    ang = float(rng.uniform(-10.0, 10.0))
    tx, ty, sc = 0.0, 0.0, 1.0
    if plus:
        h0, w0 = out.shape[:2]
        tx = float(rng.uniform(-0.05, 0.05)) * w0
        ty = float(rng.uniform(-0.05, 0.05)) * h0
        sc = float(rng.uniform(0.9, 1.1))
    if abs(ang) > 0.5 or tx or ty or sc != 1.0:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if plus:
        out = out + rng.normal(0.0, 0.02, out.shape).astype(np.float32)
    return out


try:
    import torch  # noqa: F401 (needed for the Dataset base at import time)
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pipeline still imports where torch is unneeded
    _TorchDataset = object  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class ToothCropSet(_TorchDataset):
    """Per-tooth crops, module level so DataLoader workers can pickle it.

    Holds only paths/labels (light); images are read per item, so workers stay
    light and never touch the Graphics Processing Unit (tensors move in the
    main process). Stochastic augmentation uses a per-instance generator.
    """

    def __init__(self, samples, cfg, train, seed):
        self.samples = samples
        self.cfg = cfg
        self.train = train
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torch

        s = self.samples[i]
        gray = cv2.imread(str(CACHE_DIR / s["crop_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(CACHE_DIR / s["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            gray = np.zeros((self.cfg.img_size, self.cfg.img_size), np.uint8)
            mask = np.zeros((self.cfg.img_size, self.cfg.img_size), np.uint8)
        mask = (mask > 127)
        img = preprocess_tooth(gray, mask, s["fdi"], self.cfg)
        if self.train:
            img = _augment_small(img, self.rng, self.cfg.aug_plus)
        x = np.stack([img, img, img], axis=0).astype(np.float32)  # 3ch for ImageNet backbones
        if self.cfg.input_norm == "imagenet":
            # Torchvision canonical normalization (matches pretraining).
            mu = np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
            sd = np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]
            x = (x - mu) / sd
        y = np.asarray(s["label"], dtype=np.float32)
        meta = np.asarray([1.0 if is_upper_jaw(s["fdi"]) else 0.0,
                           float(get_quadrant(s["fdi"])) / 4.0], dtype=np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(meta), int(s["fdi"]), int(s["case_id"]))


def torch_dataset(samples: list[dict], cfg: PreConfig, train: bool, seed: int):
    """Thin factory (stable call sites; picklable module-level dataset)."""
    return ToothCropSet(samples, cfg, train, seed)


# ----------------------------------------------------------------------------
# CNN (transfer learning, multi-label head, optional late fusion)
# ----------------------------------------------------------------------------

def build_cnn(backbone: str = "resnet50", use_metadata: bool = False,
              dropout: float = 0.3):
    import torch
    import torch.nn as nn

    bb = backbone.lower()
    if bb == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif bb == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        feat_dim = 1280
        net.classifier = nn.Identity()
    elif bb == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        net = densenet121(weights=DenseNet121_Weights.DEFAULT)
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    class ToothCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.use_metadata = use_metadata
            in_dim = feat_dim + (16 if use_metadata else 0)
            if use_metadata:
                self.meta_mlp = nn.Sequential(nn.Linear(2 + 32, 16), nn.ReLU())
                self.fdi_emb = nn.Embedding(60, 32)  # FDI 11..48 -> index
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 3))

        def forward(self, x, meta=None, fdi=None):
            f = self.backbone(x)
            if f.dim() > 2:
                f = f.flatten(1)
            if self.use_metadata:
                assert meta is not None and fdi is not None
                e = self.fdi_emb(torch.clamp(fdi, 0, 59))
                m = torch.cat([e, meta], dim=1)
                f = torch.cat([f, self.meta_mlp(m)], dim=1)
            return self.head(f)

    return ToothCNN()


class SigmoidFocalLoss:
    """BCE focal loss for multi-label (gamma=2). Callable module factory."""

    def __init__(self, gamma: float = 2.0, pos_weight=None):
        import torch.nn as nn

        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def __call__(self, logits, targets):
        import torch

        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ----------------------------------------------------------------------------
# Metrics + threshold tuning (recall >= 0.80 target)
# ----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds) -> dict:
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score)

    if np.ndim(thresholds) == 0:
        thresholds = [float(thresholds)] * y_true.shape[1]
    out: dict = {}
    for j, name in enumerate(SURFACES):
        yt, yp = y_true[:, j], y_prob[:, j]
        thr = float(thresholds[j])
        pred = (yp >= thr).astype(int)
        out[name] = {
            "threshold": thr,
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "precision": float(precision_score(yt, pred, zero_division=0)),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(yt, yp)
                            if yt.sum() > 0 else 0.0),
            "n_pos": int(yt.sum()),
        }
    any_true, any_prob = (y_true.sum(1) > 0).astype(int), y_prob.max(1)
    any_pred = (y_prob >= np.asarray(list(thresholds))).any(1).astype(int)
    from sklearn.metrics import f1_score as _f1

    out["HAS_CARIES(any)"] = {
        "recall": float(__import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(
            any_true, any_pred, zero_division=0)),
        "precision": float(__import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(
            any_true, any_pred, zero_division=0)),
        "f1": float(_f1(any_true, any_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(any_true, any_prob)
                        if any_true.sum() > 0 else 0.0),
        "n_pos": int(any_true.sum()),
    }
    return out


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    target_recall: float = 0.80) -> list[float]:
    from sklearn.metrics import f1_score

    thrs: list[float] = []
    for j in range(y_true.shape[1]):
        best, best_f1, fallback, fb_f1 = 0.5, -1.0, 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            pred = (y_prob[:, j] >= t).astype(int)
            rec = float((pred[y_true[:, j] == 1]).sum() / max(1, (y_true[:, j] == 1).sum()))
            f1 = float(f1_score(y_true[:, j], pred, zero_division=0))
            if f1 > fb_f1:
                fallback, fb_f1 = float(t), f1
            if rec >= target_recall and f1 > best_f1:
                best, best_f1 = float(t), f1
        thrs.append(best if best_f1 >= 0 else fallback)
    return thrs


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(history: list[dict], best_epoch: int, out_path: Path) -> None:
    """Learning curves: train vs tune loss (+best-epoch mark) and Average Precision."""
    try:
        plt = _mpl()
    except ImportError:
        print("[plots] matplotlib missing -> curves skipped")
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax[0].plot(ep, [h["tune_loss"] for h in history], label="tune loss")
    ax[0].axvline(best_epoch, color="k", ls="--", label=f"best ep {best_epoch}")
    ax[0].set_xlabel("epoch")
    ax[0].legend(fontsize=9)
    ax[0].set_title("loss (widening gap = overfitting)")
    ax[1].plot(ep, [h["tune_macro_ap"] for h in history], label="macro Average Precision")
    for s in SURFACES:
        ax[1].plot(ep, [h["tune_ap"][s] for h in history], ls=":",
                   label=f"Average Precision {s[:4]}")
    ax[1].axvline(best_epoch, color="k", ls="--")
    ax[1].set_xlabel("epoch")
    ax[1].legend(fontsize=8)
    ax[1].set_title("tune Average Precision per surface (threshold-free)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plots] {out_path}")


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    """Precision-Recall curves per surface (preferred over ROC under imbalance;
    Saito & Rehmsmeier 2015)."""
    try:
        plt = _mpl()
    except ImportError:
        print("[plots] matplotlib missing -> PR skipped")
        return
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for j, s in enumerate(SURFACES):
        if y_true[:, j].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true[:, j], y_prob[:, j])
        ap = average_precision_score(y_true[:, j], y_prob[:, j])
        ax.plot(rec, prec, label=f"{s} (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("PR curves (test-half)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plots] {out_path}")


def bootstrap_case_ci(y_true: np.ndarray, y_prob: np.ndarray, thresholds,
                      groups, n_boot: int = 1000, seed: int = 123) -> dict:
    """Case-level bootstrap 95% CI for reported metrics at FIXED thresholds.

    Resamples cases (not teeth) with replacement, so the CI reflects
    patient-level sampling noise. Matters because a test-half holds ~50 cases
    and the Occlusal minority has few positives: a wide CI is the honest
    signal that the point estimate is unstable.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.RandomState(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    thr = np.asarray(list(thresholds), dtype=float)
    acc = {s: {"recall": [], "precision": [], "f1": [], "pr_auc": []}
           for s in SURFACES}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == c)[0] for c in pick])
        yt, yp = y_true[idx], y_prob[idx]
        pred = (yp >= thr).astype(int)
        for j, s in enumerate(SURFACES):
            tp = int(((pred[:, j] == 1) & (yt[:, j] == 1)).sum())
            fp = int(((pred[:, j] == 1) & (yt[:, j] == 0)).sum())
            fn = int(((pred[:, j] == 0) & (yt[:, j] == 1)).sum())
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (2 * prec * rec / (prec + rec)
                  if (prec + rec) and not (np.isnan(prec) or np.isnan(rec))
                  else float("nan"))
            try:
                auc = (float(average_precision_score(yt[:, j], yp[:, j]))
                       if yt[:, j].sum() > 0 and (yt[:, j] == 0).sum() > 0
                       else float("nan"))
            except ValueError:
                auc = float("nan")
            acc[s]["recall"].append(rec)
            acc[s]["precision"].append(prec)
            acc[s]["f1"].append(f1)
            acc[s]["pr_auc"].append(auc)
    return {s: {m: {"mean": float(np.nanmean(v)),
                    "lo": float(np.nanpercentile(v, 2.5)),
                    "hi": float(np.nanpercentile(v, 97.5))}
                for m, v in mm.items()}
            for s, mm in acc.items()}


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    backbone: str = "resnet50"
    ablation: str = "clahe_pca"  # off | clahe | clahe_pca
    loss: str = "bce"            # bce | focal
    use_metadata: bool = False
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    cv: str = "single"           # single | groupkfold | stratified-groupkfold
    num_workers: int = 4           # DataLoader preprocessing workers (0 = off)
    patience: int | None = None  # None = fixed budget (reporting);
                                 # int = early stopping on tune macro Average Precision (epoch search)
    recall_target: float = 0.80  # tune-set Recall target for thresholds (1.4: 0.85 = margin)
    freeze_backbone: bool = False  # True = linear probing: train head only
    partial_blocks: int = 0      # >0 = unfreeze last-N backbone blocks + head
    backbone_lr: float | None = None  # None = same as lr; else separate LR group
    excluded_n: int = 0          # provenance: cases dropped via --exclude


def ablation_to_pre(name: str, img_size: int = 224,
                    input_norm: str = "zscore", aug_plus: bool = False) -> PreConfig:
    base = {"img_size": img_size, "input_norm": input_norm, "aug_plus": aug_plus}
    if name == "off":
        return PreConfig(use_clahe=False, use_pca=False, **base)
    if name == "clahe":
        return PreConfig(use_clahe=True, use_pca=False, **base)
    if name == "pca":
        return PreConfig(use_clahe=False, use_pca=True, **base)
    if name == "clahe_pca":
        return PreConfig(use_clahe=True, use_pca=True, **base)
    raise ValueError(f"Unknown ablation: {name} (want off/clahe/pca/clahe_pca)")


def _split_indices(cases: np.ndarray, seed: int, cv: str, strata=None,
                   repeats: int = 3):
    from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                         StratifiedGroupKFold)

    cases = np.asarray(cases)
    if cv == "single":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        return [next(gss.split(np.zeros(len(cases)), groups=cases))]
    if cv == "stratified-groupkfold":
        # Stratify on has_caries (binary): balances pos/neg teeth per fold while
        # keeping cases intact. Per-surface balance is approximate (multi-label)
        # -> verify with the per-fold pos counts printed by run_train.
        if strata is None:
            raise ValueError("stratified-groupkfold needs strata (e.g. has_caries)")
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(sgkf.split(np.zeros(len(cases)), np.asarray(strata),
                               groups=cases))
    if cv == "repeated-stratified-groupkfold":
        # Repeated stratified grouped folds (1.2): several seeds back to back.
        # Eases seed lottery; pair folds across configs for paired comparison.
        if strata is None:
            raise ValueError("repeated-stratified-groupkfold needs strata")
        out = []
        for _r in range(max(1, repeats)):
            _sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                         random_state=seed + 1000 * _r)
            out.extend(list(_sgkf.split(np.zeros(len(cases)),
                                        np.asarray(strata), groups=cases)))
        return out
    folds = []
    for tr, va in GroupKFold(n_splits=5).split(np.zeros(len(cases)), groups=cases):
        folds.append((tr, va))
    return folds


def parse_exclude(spec: str) -> set[int]:
    """Parse '1-50,200' or '@/path.json' into {case ids}.

    The JSON form accepts a bare list or a dict with keys 'cases' /
    'peeked_cases' / 'test_half_diagnosed' / 'tune_half_selection'
    (see outputs/peeked_cases.json). Used to keep diagnostically-peeked cases
    out of later training/evaluation (test-hygiene note, plan section 4).
    """
    s = (spec or "").strip()
    if not s:
        return set()
    if s.startswith("@"):
        obj = json.loads(Path(s[1:]).read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return {int(v) for v in obj}
        vals: list = list(obj.get("cases", []) or obj.get("peeked_cases", []))
        for k in ("test_half_diagnosed", "tune_half_selection"):
            vals = vals + list(obj.get(k, []) or [])
        return {int(v) for v in vals}
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        m = re.match(r"(\d+)\s*-\s*(\d+)$", tok)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tok.isdigit():
            out.add(int(tok))
        else:
            raise SystemExit(f"[exclude] bad token: {tok} (want '1-50,200' or '@/file.json')")
    return out


def apply_exclude(samples: list[dict], spec: str):
    ids = parse_exclude(spec)
    if not ids:
        return samples, 0
    kept = [s for s in samples if s["case_id"] not in ids]
    print(f"[exclude] dropped {len(samples) - len(kept)} teeth from {len(ids)} cases")
    return kept, len(ids)


def _tune_test_split(samples: list[dict], seed: int):
    """Nested split of a validation set BY CASE: tune thresholds on one half,
    report threshold-dependent metrics on the other.

    Tuning thresholds and reporting recall/precision/F1 on the SAME set makes
    recall look better than it is (threshold-selection optimism). PR-AUC is
    threshold-free and unaffected.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([s["case_id"] for s in samples])
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.5,
                                  random_state=seed).split(
                                      np.zeros(len(samples)), groups=groups))
    return [samples[i] for i in a], [samples[i] for i in b]


def apply_partial_freeze(model, backbone: str, n: int) -> list[str]:
    """Experiment B: freeze all, then unfreeze head (+metadata) and the last-N
    backbone blocks. n=0 == full linear probing (same as --freeze-backbone)."""
    import torch.nn as nn

    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if getattr(model, "use_metadata", False):
        for p in model.meta_mlp.parameters():
            p.requires_grad = True
        for p in model.fdi_emb.parameters():
            p.requires_grad = True
    bb = model.backbone
    if backbone == "resnet50":
        blocks = [bb.layer1, bb.layer2, bb.layer3, bb.layer4]
    elif backbone == "efficientnet_b0":
        blocks = [c for c in bb.features.children()
                  if isinstance(c, nn.Sequential) or "Conv" in type(c).__name__]
    elif backbone == "densenet121":
        blocks = [bb.features.denseblock1, bb.features.denseblock2,
                  bb.features.denseblock3, bb.features.denseblock4]
    else:
        blocks = []
    opened = blocks[max(0, len(blocks) - n):] if n > 0 else []
    for blk in opened:
        for p in blk.parameters():
            p.requires_grad = True
    return [f"{backbone}:{type(b).__name__}" for b in opened]


def macro_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    """Threshold-free macro Average Precision for epoch selection (1.1).

    Selecting checkpoints on tuned-F1 re-tunes thresholds every epoch on ~50
    tune cases (noisy selection: swinging F1, jumping thresholds). Average
    Precision needs no thresholds; thresholds are tuned once, after selection.
    """
    from sklearn.metrics import average_precision_score

    out = {}
    for j, sname in enumerate(SURFACES):
        yt = y_true[:, j]
        out[sname] = (float(average_precision_score(yt, y_prob[:, j]))
                      if yt.sum() > 0 and (yt == 0).sum() > 0 else float("nan"))
    vals = [v for v in out.values() if v == v]
    return (float(np.mean(vals)) if vals else float("nan"), out)


def _set_train_mode(model, backbone_frozen: bool):
    """train() with BatchNorm frozen when the backbone is frozen/partial.

    Updating BatchNorm running stats while backbone weights are frozen mixes
    ImageNet statistics with tooth-batch statistics and differs between train
    and eval ( Detectron2-style fine-tuning freezes them by default).
    Gradients still flow to unfrozen late blocks; eval() only fixes the norm.
    """
    model.train()
    if backbone_frozen:
        try:
            model.backbone.eval()
        except AttributeError:
            pass


def _cfg_fingerprint(cfg: TrainConfig, pre: PreConfig) -> dict:
    """A4: everything that defines the meaning of a saved fold state.

    Two runs may share a checkpoint only when this fingerprint is identical;
    resuming with a different configuration raises instead of silently mixing.
    """
    return {"backbone": cfg.backbone, "ablation": cfg.ablation,
            "loss": cfg.loss, "use_metadata": bool(cfg.use_metadata),
            "epochs": int(cfg.epochs), "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr), "seed": int(cfg.seed), "cv": cfg.cv,
            "freeze_backbone": bool(cfg.freeze_backbone),
            "partial_blocks": int(cfg.partial_blocks),
            "backbone_lr": cfg.backbone_lr,
            "weight_decay": float(cfg.weight_decay),
            "recall_target": float(cfg.recall_target),
            "img_size": int(pre.img_size), "input_norm": pre.input_norm,
            "aug_plus": bool(pre.aug_plus)}


def _save_fold_state(last_path, hist_path, ep, model, opt, sched, best_ap,
                     best_state, best_epoch, since_best, history, patience,
                     fingerprint, ds_rng=None) -> None:
    """A4: durable per-epoch state so a killed run resumes mid-fold.

    Rewrites {prefix}_history.json and saves {prefix}_last.pt via an atomic
    replace, so a kill during the write never leaves a torn checkpoint.
    """
    import random as _random

    import torch

    hist_path.write_text(json.dumps(
        {"history": history, "best_epoch": best_epoch, "patience": patience,
         "in_progress": True}, indent=1), encoding="utf-8")
    rng = {"torch": torch.get_rng_state(),
           "cuda": (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None),
           "numpy": np.random.get_state(), "python": _random.getstate()}
    tmp = last_path.with_name(last_path.name + ".tmp")
    torch.save({"epoch": ep, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "best_ap": best_ap, "best_state": best_state,
                "best_epoch": best_epoch, "since_best": since_best,
                "history": history, "fingerprint": fingerprint,
                "ds_rng": ds_rng, "rng": rng}, tmp)
    tmp.replace(last_path)


def _resume_completed_fold(wpath, model, cfg, pre, tune_dl, test_dl, test_s,
                           test_groups, out_prefix, bb_frozen) -> dict | None:
    """A4: rebuild a finished fold's result dict without retraining.

    Re-derives the tune-set thresholds from the checkpoint and refuses to
    reuse the artifact when they drift by more than floating-point noise --
    this catches a changed --recall-target or a checkpoint from another run.
    """
    import torch

    try:
        ckpt = torch.load(str(wpath), map_location="cpu")
    except Exception as exc:
        print(f"[resume] {wpath.name}: unreadable ({exc}); retraining fold")
        return None
    for key, wanted in (("backbone", cfg.backbone), ("ablation", cfg.ablation),
                        ("use_metadata", bool(cfg.use_metadata)),
                        ("img_size", int(pre.img_size)),
                        ("input_norm", pre.input_norm),
                        ("aug_plus", bool(pre.aug_plus))):
        if key in ckpt and ckpt[key] != wanted:
            raise SystemExit(
                f"[resume] {wpath.name}: {key}={ckpt[key]!r} does not match "
                f"this run ({wanted!r}); archive the artifacts or retrain")
    model.load_state_dict(ckpt["state_dict"])
    dev = next(model.parameters()).device
    tune_m = evaluate(model, tune_dl, dev, None, bb_frozen, return_logits=True)
    final_thr = tune_thresholds(tune_m["y_true"], tune_m["y_prob"],
                                cfg.recall_target)
    saved_thr = [float(v) for v in ckpt["thresholds"]]
    if len(saved_thr) != len(final_thr):
        raise SystemExit(f"[resume] {wpath.name}: threshold vector mismatch")
    tpath = OUTPUTS_DIR / f"{out_prefix}_thresholds.json"
    if tpath.is_file():
        tj = json.loads(tpath.read_text(encoding="utf-8"))
        if float(tj.get("recall_target", cfg.recall_target)) != float(
                cfg.recall_target):
            raise SystemExit(
                f"[resume] {tpath.name}: recall_target="
                f"{tj.get('recall_target')} != {cfg.recall_target}; archive "
                "the artifacts or retrain")
    drift = max(abs(a - b) for a, b in zip(final_thr, saved_thr))
    if drift > 0.026:
        raise SystemExit(
            f"[resume] {wpath.name}: stored thresholds {saved_thr} != "
            f"recomputed {[float(v) for v in final_thr]} with "
            f"--recall-target {cfg.recall_target}; artifact belongs to a "
            "different configuration (archive it or retrain)")
    if drift > 1e-9:
        print(f"[resume] note {out_prefix}: threshold drift {drift:.2e} "
              "(<= half the 0.05 search grid); using recomputed values")
    test_m = evaluate(model, test_dl, dev, None, bb_frozen, return_logits=True)
    mets = compute_metrics(test_m["y_true"], test_m["y_prob"], final_thr)
    ci = bootstrap_case_ci(test_m["y_true"], test_m["y_prob"], final_thr,
                           test_groups)
    test_macro = float(np.mean([mets[s]["f1"] for s in SURFACES]))
    mpath = OUTPUTS_DIR / f"{out_prefix}_metrics.json"
    mj = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
    stored_f1 = mj.get("test", {}).get(SURFACES[0], {}).get("f1")
    if stored_f1 is not None and abs(float(stored_f1)
                                     - mets[SURFACES[0]]["f1"]) > 5e-4:
        print(f"[resume] WARNING {out_prefix}: recomputed {SURFACES[0]} F1 "
              f"{mets[SURFACES[0]]['f1']:.6f} != metrics.json "
              f"{float(stored_f1):.6f} (non-determinism?)")
    print(f"[resume] {out_prefix}: completed fold reused "
          f"(best_epoch={mj.get('best_epoch')}, "
          f"thresholds={[round(v, 3) for v in final_thr]})")
    return {"weights": str(wpath), "macro_f1": test_macro, "metrics": mets,
            "metrics_ci": ci, "thresholds": final_thr,
            "best_epoch": int(mj.get("best_epoch", 0)),
            "history": OUTPUTS_DIR / f"{out_prefix}_history.json",
            "test_y_true": test_m["y_true"], "test_y_prob": test_m["y_prob"],
            "test_keys": [(int(s["case_id"]), int(s["fdi"])) for s in test_s],
            "test_logits": test_m["logits"], "tune_logits": tune_m["logits"],
            "tune_y_true": tune_m["y_true"], "tune_y_prob": tune_m["y_prob"],
            "tune_best_macro_ap": float(mj.get("tune_best_macro_ap",
                                               float("nan")))}


def train_fold(tr_samples, va_samples, cfg: TrainConfig, pre: PreConfig,
               out_prefix: str, resume: bool = False) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Nested protocol FIRST: checkpoint selection sees the tune-half only;
    # threshold-dependent metrics are reported on the disjoint test-half.
    tune_s, test_s = _tune_test_split(va_samples, cfg.seed + 7919)
    test_groups = np.array([s["case_id"] for s in test_s])
    train_dl = DataLoader(torch_dataset(tr_samples, pre, train=True, seed=cfg.seed),
                          batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers)
    tune_dl = DataLoader(torch_dataset(tune_s, pre, train=False, seed=cfg.seed),
                         batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers)
    test_dl = DataLoader(torch_dataset(test_s, pre, train=False, seed=cfg.seed),
                         batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers)
    model = build_cnn(cfg.backbone, cfg.use_metadata).to(device)
    if cfg.freeze_backbone:
        apply_partial_freeze(model, cfg.backbone, 0)
        print("[transfer] Experiment A: backbone fully frozen (linear probing)")
    elif cfg.partial_blocks > 0:
        opened = apply_partial_freeze(model, cfg.backbone, cfg.partial_blocks)
        print(f"[transfer] Experiment B: unfrozen late blocks: {opened}")
    else:
        print("[transfer] Experiment C: full fine-tuning")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[transfer] trainable params={n_tr}")
    # --- A4 resume support -------------------------------------------------
    bb_frozen = bool(cfg.freeze_backbone or cfg.partial_blocks > 0)
    fingerprint = _cfg_fingerprint(cfg, pre)
    wpath = OUTPUTS_DIR / f"{out_prefix}.pt"
    last_path = OUTPUTS_DIR / f"{out_prefix}_last.pt"
    hist_path = OUTPUTS_DIR / f"{out_prefix}_history.json"
    done_marks = (wpath.is_file()
                  and (OUTPUTS_DIR / f"{out_prefix}_metrics.json").is_file())
    if resume and done_marks:
        got = _resume_completed_fold(wpath, model, cfg, pre, tune_dl, test_dl,
                                     test_s, test_groups, out_prefix, bb_frozen)
        if got is not None:
            return got
    ytr = np.stack([s["label"] for s in tr_samples]).astype(np.float32)
    neg, pos = (len(ytr) - ytr.sum(0)), ytr.sum(0)
    pos_w = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0),
                         dtype=torch.float32).to(device)
    criterion = (SigmoidFocalLoss(gamma=2.0, pos_weight=pos_w) if cfg.loss == "focal"
                 else nn.BCEWithLogitsLoss(pos_weight=pos_w))
    bb_lr = cfg.backbone_lr if cfg.backbone_lr else cfg.lr

    def _wd_split(named, want_bb):
        # No weight decay on 1-dim params (biases, norms) — standard practice.
        decay, nodecay = [], []
        for n, p in named:
            if not p.requires_grad:
                continue
            if want_bb is not None and n.startswith("backbone.") != want_bb:
                continue
            (decay if p.dim() > 1 else nodecay).append(p)
        return decay, nodecay

    bb_d, bb_nd = _wd_split(model.named_parameters(), True)
    h_d, h_nd = _wd_split(model.named_parameters(), False)
    opt_groups = []
    if bb_d:
        opt_groups.append({"params": bb_d, "lr": bb_lr})
    if bb_nd:
        opt_groups.append({"params": bb_nd, "lr": bb_lr, "weight_decay": 0.0})
    if h_d:
        opt_groups.append({"params": h_d, "lr": cfg.lr})
    if h_nd:
        opt_groups.append({"params": h_nd, "lr": cfg.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(opt_groups, weight_decay=cfg.weight_decay)
    # ReduceLROnPlateau on tune-loss: independent of the total-epoch budget, so
    # best-epochs stay comparable across runs (unlike cosine tied to T_max).
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3)
    best_ap, best_state, best_epoch, since_best = -1.0, None, 1, 0
    history: list[dict] = []
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    bb_frozen = bool(cfg.freeze_backbone or cfg.partial_blocks > 0)
    start_ep = 1
    if resume and not done_marks and last_path.is_file():
        st = torch.load(str(last_path), map_location="cpu")
        if st.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"[resume] {last_path.name}: saved configuration does not "
                "match this run; archive or delete the state before resuming")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        best_ap, best_state = float(st["best_ap"]), st["best_state"]
        best_epoch, since_best = int(st["best_epoch"]), int(st["since_best"])
        history = list(st["history"])
        rng = st.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            import random as _random
            _random.setstate(rng["python"])
        _ds = getattr(getattr(train_dl, "dataset", None), "rng", None)
        if st.get("ds_rng") is not None and _ds is not None:
            _ds.set_state(st["ds_rng"])
            print(f"[resume] {out_prefix}: augmentation RNG state restored")
        start_ep = int(st["epoch"]) + 1
        print(f"[resume] {out_prefix}: partial fold continues at epoch "
              f"{start_ep} (best_ap={best_ap:.4f}, best_epoch={best_epoch})")
    for ep in range(start_ep, cfg.epochs + 1):
        _set_train_mode(model, bb_frozen)
        tot, nb = 0.0, 0
        for xb, yb, mb, fb, _cb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            mb = mb.to(device)
            fb_t = fb.to(device)
            opt.zero_grad()
            logits = (model(xb, mb, fb_t) if cfg.use_metadata else model(xb))
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        tune_e = evaluate(model, tune_dl, device, criterion, bb_frozen)
        tune_loss = float(tune_e["loss"]) if tune_e["loss"] is not None else float("nan")
        sched.step(tune_loss)
        ap_macro, ap_per = macro_pr_auc(tune_e["y_true"], tune_e["y_prob"])
        lr_now = float(opt.param_groups[0]["lr"])
        history.append({"epoch": ep, "train_loss": tot / max(1, nb),
                        "tune_loss": tune_loss, "tune_macro_ap": ap_macro,
                        "tune_ap": ap_per, "lr": lr_now})
        print(f"[ep {ep}/{cfg.epochs}] train={tot / max(1, nb):.4f} "
              f"tune_loss={tune_loss:.4f} tune_macroAP={ap_macro:.4f} "
              f"AP={[round(ap_per[s], 3) for s in SURFACES]} lr={lr_now:.2e}")
        if ap_macro == ap_macro and ap_macro > best_ap:
            best_ap = ap_macro
            # .clone() matters: on CPU, .cpu() returns the same tensor, so
            # without it the "best" snapshot would keep aliasing live weights.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = ep
            since_best = 0
        else:
            since_best += 1
        _ds = getattr(getattr(train_dl, "dataset", None), "rng", None)
        _save_fold_state(last_path, hist_path, ep, model, opt, sched, best_ap,
                         best_state, best_epoch, since_best, history,
                         cfg.patience, fingerprint,
                         _ds.get_state() if _ds is not None else None)
        if cfg.patience is not None and since_best >= cfg.patience:
            print(f"[early-stop] no tune_macroAP gain for {since_best} epochs "
                  f"-> stop at ep {ep} (best ep {best_epoch})")
            break
    assert best_state is not None
    wpath = OUTPUTS_DIR / f"{out_prefix}.pt"
    model.load_state_dict({k: v.to(next(model.parameters()).device)
                           for k, v in best_state.items()})
    dev = next(model.parameters()).device
    tune_m = evaluate(model, tune_dl, dev, None, bb_frozen, return_logits=True)
    final_thr = tune_thresholds(tune_m["y_true"], tune_m["y_prob"],
                                cfg.recall_target)
    test_m = evaluate(model, test_dl, dev, None, bb_frozen, return_logits=True)
    mets = compute_metrics(test_m["y_true"], test_m["y_prob"], final_thr)
    ci = bootstrap_case_ci(test_m["y_true"], test_m["y_prob"], final_thr,
                           test_groups)
    test_macro = float(np.mean([mets[s]["f1"] for s in SURFACES]))
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()},
                "backbone": cfg.backbone,
                "use_metadata": cfg.use_metadata, "ablation": cfg.ablation,
                "thresholds": final_thr, "surfaces": SURFACES,
                "img_size": pre.img_size, "input_norm": pre.input_norm,
                "aug_plus": pre.aug_plus}, wpath)
    (OUTPUTS_DIR / f"{out_prefix}_history.json").write_text(
        json.dumps({"history": history, "best_epoch": best_epoch,
                    "patience": cfg.patience,
                    "n_tune_cases": len({s["case_id"] for s in tune_s}),
                    "n_test_cases": len({s["case_id"] for s in test_s})},
                   indent=1), encoding="utf-8")
    plot_training_curves(history, best_epoch, OUTPUTS_DIR / f"{out_prefix}_curves.png")
    plot_pr_curves(test_m["y_true"], test_m["y_prob"],
                   OUTPUTS_DIR / f"{out_prefix}_pr.png")
    (OUTPUTS_DIR / f"{out_prefix}_metrics.json").write_text(
        json.dumps({"test": mets, "test_ci95": ci, "test_macro_f1": test_macro,
                    "tune_best_macro_ap": best_ap, "best_epoch": best_epoch,
                    "thresholds": final_thr,
                    "n_tune_cases": len({s["case_id"] for s in tune_s}),
                    "n_test_cases": len({s["case_id"] for s in test_s}),
                    "excluded_cases": cfg.excluded_n,
                    "note": "checkpoint selected on tune-half; thresholds tuned on "
                            "tune-half; metrics + bootstrap CI reported on disjoint "
                            "test-half"},
                   indent=1), encoding="utf-8")
    (OUTPUTS_DIR / f"{out_prefix}_thresholds.json").write_text(
        json.dumps({"thresholds": [float(v) for v in final_thr],
                    "recall_target": cfg.recall_target,
                    "out_prefix": out_prefix}, indent=1), encoding="utf-8")
    print(f"[saved] {wpath} test_macroF1={test_macro:.4f} " +
          " ".join(f"{s[:4]}:R={mets[s]['recall']:.2f}"
                   f"[{ci[s]['recall']['lo']:.2f}-{ci[s]['recall']['hi']:.2f}]"
                   for s in SURFACES))
    return {"weights": str(wpath), "macro_f1": test_macro, "metrics": mets,
            "metrics_ci": ci, "thresholds": final_thr,
            "best_epoch": best_epoch,
            "history": OUTPUTS_DIR / f"{out_prefix}_history.json",
            "test_y_true": test_m["y_true"], "test_y_prob": test_m["y_prob"],
            "test_keys": [(int(s["case_id"]), int(s["fdi"])) for s in test_s],
            "test_logits": test_m["logits"], "tune_logits": tune_m["logits"],
            "tune_y_true": tune_m["y_true"], "tune_y_prob": tune_m["y_prob"],
            "tune_best_macro_ap": best_ap}


def evaluate(model, loader, device, criterion=None, backbone_frozen=False,
             return_logits=False) -> dict:
    import torch

    was_training = model.training
    model.eval()
    ys, ps, tot, nb, lgs = [], [], 0.0, 0, []
    with torch.no_grad():
        for xb, yb, mb, fb, _cb in loader:
            xb = xb.to(device)
            ybd = yb.to(device)
            mb = mb.to(device)
            fb_t = fb.to(device)
            use_meta = getattr(model, "use_metadata", False)
            logits = model(xb, mb, fb_t) if use_meta else model(xb)
            if criterion is not None:
                tot += float(criterion(logits, ybd).item())
                nb += 1
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.numpy())
            if return_logits:
                lgs.append(logits.cpu().numpy())
    if was_training:
        _set_train_mode(model, backbone_frozen)
    return {"y_true": np.concatenate(ys) if ys else np.zeros((0, 3)),
            "y_prob": np.concatenate(ps) if ps else np.zeros((0, 3)),
            "loss": (tot / nb) if nb else None,
            "logits": np.concatenate(lgs) if lgs else None}


def run_train(args) -> None:
    meta, samples = load_manifest()
    n_all = len(samples)
    n_all_cases = len({int(s["case_id"]) for s in samples})
    samples, n_exc = apply_exclude(samples, args.exclude)
    print(f"[manifest] {n_all} teeth in {n_all_cases} cases; "
          f"{n_exc} cases excluded -> training on {len(samples)} teeth in "
          f"{len({int(s['case_id']) for s in samples})} cases | "
          f"{meta.get('meta', meta)}")
    pre = ablation_to_pre(args.ablation, img_size=args.img_size,
                          input_norm=args.input_norm, aug_plus=args.aug_plus)
    epochs = args.epochs
    if epochs is None:
        fp = OUTPUTS_DIR / f"find_epoch_{args.ablation}.json"
        if not fp.is_file():
            raise SystemExit(
                f"[train] --epochs not given and {fp} missing. "
                f"Run find-epoch first (stage 1: epoch budget on tune-half).")
        epochs = int(json.loads(fp.read_text(encoding="utf-8"))["best_epoch"])
        print(f"[train] epoch budget from {fp.name}: {epochs}")
    cfgs = TrainConfig(backbone=args.backbone, ablation=args.ablation,
                       loss=args.loss, use_metadata=args.use_metadata,
                       epochs=epochs, batch_size=args.batch_size,
                       lr=args.lr, seed=args.seed, cv=args.cv,
                       num_workers=args.num_workers,
                       patience=args.patience,
                       freeze_backbone=args.freeze_backbone,
                       partial_blocks=args.partial_blocks,
                       backbone_lr=args.backbone_lr, excluded_n=n_exc,
                       recall_target=args.recall_target)
    cases = np.array([s["case_id"] for s in samples])
    strata = ([1 if any(s["label"]) else 0 for s in samples]
              if args.cv in ("stratified-groupkfold",
                             "repeated-stratified-groupkfold") else None)
    splits = _split_indices(cases, args.seed, args.cv, strata=strata,
                            repeats=args.repeats)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _fold_of_case = {}
    for _fi2, (_tr2, _va2) in enumerate(splits):
        # `cases` is per TOOTH, so dedupe: one fold id per case per repeat.
        for _ci in sorted(set(cases[_va2].tolist())):
            _fold_of_case.setdefault(int(_ci), []).append(_fi2)
    _part = {}
    for _cc, _fs in _fold_of_case.items():
        _part.setdefault(tuple(sorted(set(_fs))), []).append(_cc)
    assert sum(len(v) for v in _part.values()) == len(_fold_of_case)
    print("[folds] partition sizes: %s" % {k: len(v) for k, v in
                                           sorted(_part.items())})
    with open(OUTPUTS_DIR / ("folds_%s_%s_seed%d_exc%d.csv"
                             % (args.ablation, args.cv, args.seed, n_exc)),
              "w", encoding="utf-8") as _ff:
        _ff.write("case_id,folds\n")
        for _cc in sorted(_fold_of_case):
            _ff.write("%d,%s\n" % (_cc, ";".join(map(str, _fold_of_case[_cc]))))
    print("[folds] %d cases mapped across %d folds" % (len(_fold_of_case),
                                                       len(splits)))
    results = []
    for fi, (tr, va) in enumerate(splits):
        tr_s = [samples[i] for i in tr]
        va_s = [samples[i] for i in va]
        # patient isolation sanity check
        assert not (set(cases[tr]) & set(cases[va])), "patient leakage across split!"
        tr_pos = sum(1 for s in tr_s if any(s["label"]))
        va_pos = sum(1 for s in va_s if any(s["label"]))
        print(f"[split {fi}] train cases={len(set(cases[tr]))} (pos teeth={tr_pos}) "
              f"val cases={len(set(cases[va]))} (pos teeth={va_pos})")
        # Seed is part of the checkpoint name so a multi-seed replication never
        # overwrites the checkpoints of the delivered model.
        r = train_fold(tr_s, va_s, cfgs, pre,
                       out_prefix="best_%s_s%d_f%d" % (args.ablation,
                                                       args.seed, fi),
                       resume=getattr(args, "resume", False))
        results.append(r)
    print(json.dumps({f"fold{i}": {"test_macro_f1": r["macro_f1"],
                                    "thresholds": r["thresholds"]}
                      for i, r in enumerate(results)}, indent=1))
    if len(results) > 1:
        macros = np.array([r["macro_f1"] for r in results])
        print(f"[cv] test_macroF1 mean={macros.mean():.4f} std={macros.std():.4f} "
              f"folds={len(results)}")
        for s in SURFACES:
            rs = np.array([r["metrics"][s]["recall"] for r in results])
            fs = np.array([r["metrics"][s]["f1"] for r in results])
            print(f"  {s:<8} recall {rs.mean():.3f}±{rs.std():.3f}   "
                  f"f1 {fs.mean():.3f}±{fs.std():.3f}")
    if getattr(args, "save_oof", False):
        _write_oof(args, results)


def verify_nested_thresholds(results, recall_target):
    """Runtime unit check (table item 1/4): every fold's applied thresholds must
    equal tune_thresholds() recomputed on that fold's INNER (tune) arrays.
    Raises AssertionError otherwise — thresholds must never come from test data.
    """
    for fi, r in enumerate(results):
        expect = tune_thresholds(r["tune_y_true"], r["tune_y_prob"], recall_target)
        got = [float(v) for v in r["thresholds"]]
        assert len(expect) == len(got) == 3, (fi, expect, got)
        assert all(abs(a - b) < 1e-9 for a, b in zip(expect, got)), (fi, expect, got)
    return True



def _write_oof(args, results):
    """Pooled out-of-fold report (1.3): every tooth scored once while held out.

    Decisions reuse each fold's OWN tune thresholds (nested); the pooled
    Recall is therefore not threshold-optimistic. Bootstrap resamples cases.
    """
    verify_nested_thresholds(results, args.recall_target)
    import csv as _csv
    import json as _json

    import numpy as _np
    import csv as _csv
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score)

    PY, PP, PD, PG, PK, PF = [], [], [], [], [], []
    for fi, r in enumerate(results):
        thr = _np.asarray(r["thresholds"], dtype=float)
        PY.append(r["test_y_true"])
        PP.append(r["test_y_prob"])
        PD.append((r["test_y_prob"] >= thr).astype(int))
        PG.extend([k[0] for k in r["test_keys"]])
        PK.extend(r["test_keys"])
        PF.extend([fi] * len(r["test_keys"]))
    Y = _np.concatenate(PY)
    Pp = _np.concatenate(PP)
    D = _np.concatenate(PD)
    G = _np.array(PG)
    pooled, pooled_ci = {}, {}
    # Per-fold Average Precision (table item 4): probabilities from different
    # folds have different calibration, so never pool them into one curve.
    # Decisions (already thresholded per fold) MAY be pooled; curves may not.
    _fold_aps, _off = [], 0
    for _fr in results:
        _n = len(_fr["test_keys"])
        _yy, _pp = Y[_off:_off + _n], Pp[_off:_off + _n]
        _off += _n
        _row = {}
        for _j, _s in enumerate(SURFACES):
            _row[_s] = (float(average_precision_score(_yy[:, _j], _pp[:, _j]))
                        if _yy[:, _j].sum() > 0 else float("nan"))
        _fold_aps.append(_row)
    rng = _np.random.RandomState(args.seed + 31337)
    uniq = _np.unique(G)
    for j, sname in enumerate(SURFACES):
        pooled[sname] = {
            "recall": float(recall_score(Y[:, j], D[:, j], zero_division=0)),
            "precision": float(precision_score(Y[:, j], D[:, j], zero_division=0)),
            "f1": float(f1_score(Y[:, j], D[:, j], zero_division=0)),
            "pr_auc_fold_mean": float(_np.nanmean([_r[sname] for _r in _fold_aps])),
            "pr_auc_fold_std": float(_np.nanstd([_r[sname] for _r in _fold_aps])),
            "pr_auc_fold_values": [
                None if _r[sname] != _r[sname] else float(_r[sname])
                for _r in _fold_aps],
            "n_pos": int(Y[:, j].sum())}
        for mname, fn in (("recall", recall_score), ("precision", precision_score),
                          ("f1", f1_score)):
            vals = []
            for _ in range(1000):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                idx = _np.concatenate(
                    [_np.where(G == c)[0] for c in pick])
                vals.append(float(fn(Y[idx, j], D[idx, j], zero_division=0)))
            pooled_ci.setdefault(sname, {})[mname] = {
                "mean": float(_np.mean(vals)),
                "lo": float(_np.percentile(vals, 2.5)),
                "hi": float(_np.percentile(vals, 97.5))}
    _exc_n = len(parse_exclude(getattr(args, "exclude", "") or ""))
    tag = "oof_%s_%s_seed%d_exc%d" % (args.ablation, args.cv, args.seed, _exc_n)
    with open(OUTPUTS_DIR / (tag + ".csv"), "w", newline="",
              encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["case_id", "tooth_fdi", "fold", "true_Distal",
                    "true_Mesial", "true_Occlusal", "prob_Distal",
                    "prob_Mesial", "prob_Occlusal", "pred_Distal",
                    "pred_Mesial", "pred_Occlusal"])
        for k in range(len(PK)):
            w.writerow([PK[k][0], PK[k][1], PF[k],
                        *[int(v) for v in Y[k].tolist()],
                        *[round(float(v), 4) for v in Pp[k]],
                        *D[k].tolist()])
    (OUTPUTS_DIR / (tag + ".json")).write_text(
        _json.dumps({"pooled": pooled, "pooled_ci95": pooled_ci,
                                  "n_teeth": len(PK),
                                  "n_cases": len(set(PG))}, indent=1),
        encoding="utf-8")
    print("[oof] %d teeth pooled across %d folds -> %s.*" % (len(PK),
                                                             len(results), tag))


# ----------------------------------------------------------------------------
# Evaluation stage: E1 out-of-fold report, E1b paired arm decision,
# E3 ensemble calibration on the peeked set, E4 inter-rater ceiling,
# E5 external-test size planning.
#
# Design notes (why these exact estimators):
#   * Average Precision is averaged over folds, never computed on pooled
#     probabilities: each fold has its own calibration, so one pooled curve
#     would be a curve no single delivered model ever produced.
#   * Decisions ARE pooled, because each fold's decision used that fold's own
#     nested tune-set threshold; the pooled Recall is therefore not
#     threshold-optimistic.
#   * Pooled Recall/Precision are count ratios, so a Wilson interval is exact
#     for the binomial part, while a patient-level cluster bootstrap adds the
#     clustering component (several teeth belong to one patient).
#   * Comparing two training methods on one seed only measures test-set
#     variance. Seed replicates plus the Nadeau & Bengio (2003) correction are
#     required before a method may be called better.
# ----------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def load_oof_csv(path) -> dict:
    """Read the pooled out-of-fold CSV written by _write_oof()."""
    import csv as _csv

    rows = list(_csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("empty OOF file: %s" % path)
    return {"case": np.array([int(r["case_id"]) for r in rows]),
            "fdi": np.array([int(r["tooth_fdi"]) for r in rows]),
            "fold": np.array([int(float(r["fold"])) for r in rows]),
            "y": np.stack([[int(float(r["true_" + s])) for s in SURFACES]
                           for r in rows]).astype(int),
            "p": np.stack([[float(r["prob_" + s]) for s in SURFACES]
                           for r in rows]),
            "d": np.stack([[int(float(r["pred_" + s])) for s in SURFACES]
                           for r in rows]).astype(int),
            "path": str(path)}


def oof_per_fold(oof: dict) -> dict:
    """Prevalence / Average Precision / Lift / AUROC per fold and surface."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {s: {"ap": [], "prevalence": [], "lift": [], "auroc": [],
               "n": [], "n_pos": []} for s in SURFACES}
    for f in sorted(set(oof["fold"].tolist())):
        m = oof["fold"] == f
        for j, s in enumerate(SURFACES):
            y, p = oof["y"][m, j], oof["p"][m, j]
            prev = float(y.mean()) if len(y) else float("nan")
            degen = int(y.sum()) == 0 or int(y.sum()) == len(y)
            ap = float("nan") if degen else float(average_precision_score(y, p))
            auc = float("nan") if degen else float(roc_auc_score(y, p))
            out[s]["ap"].append(ap)
            out[s]["prevalence"].append(prev)
            out[s]["lift"].append(float(ap / prev) if (prev > 0 and ap == ap)
                                  else float("nan"))
            out[s]["auroc"].append(auc)
            out[s]["n"].append(int(len(y)))
            out[s]["n_pos"].append(int(y.sum()))
    return out


def _plot_oof_reliability(rel: dict, brier: dict, op: Path) -> None:
    """Reliability diagram data is JSON; this renders it for the report."""
    try:
        plt = _mpl()
    except ImportError:
        print("[oof-report] matplotlib missing -> reliability plot skipped")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, s in zip(axes, list(SURFACES) + ["HAS_CARIES(any)"]):
        rr = rel[s]
        xs = [v for v in rr["bin_mean_prob"] if v is not None]
        ys = [yy for yy, v in zip(rr["bin_frac_pos"], rr["bin_mean_prob"])
              if v is not None]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, "o-", ms=4)
        ax.set_title("%s (Brier %.3f)" % (s, brier[s]["fold_mean"]),
                     fontsize=9)
        ax.set_xlabel("predicted probability", fontsize=8)
        ax.set_ylabel("observed frequency", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    outp = op.with_name(op.stem + "_reliability.png")
    fig.savefig(outp, dpi=120)
    plt.close(fig)
    print("[oof-report] reliability -> %s" % outp)


def run_oof_report(args) -> None:
    """E1 (plan 2.1): every number the results table needs, from one OOF file."""
    from sklearn.metrics import f1_score

    oof = load_oof_csv(args.oof)
    pf = oof_per_fold(oof)
    n_folds = len(set(oof["fold"].tolist()))
    rng = np.random.RandomState(args.seed + 5150)
    cases = np.unique(oof["case"])
    out = {"oof_file": oof["path"], "n_teeth": int(len(oof["case"])),
           "n_cases": int(len(cases)), "n_folds": n_folds,
           "recall_target": float(args.recall_target),
           "n_boot": int(args.n_boot), "per_surface": {}}
    print("[oof-report] %s teeth=%d cases=%d folds=%d"
          % (Path(oof["path"]).name, len(oof["case"]), len(cases), n_folds))
    for j, s in enumerate(SURFACES):
        y, d = oof["y"][:, j], oof["d"][:, j]
        tp = int(((y == 1) & (d == 1)).sum())
        fp = int(((y == 0) & (d == 1)).sum())
        fn = int(((y == 1) & (d == 0)).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        boot = np.zeros((args.n_boot, 2))
        for b in range(args.n_boot):
            pick = rng.choice(cases, size=len(cases), replace=True)
            idx = np.concatenate([np.where(oof["case"] == c)[0] for c in pick])
            yb, db = y[idx], d[idx]
            tpb = int(((yb == 1) & (db == 1)).sum())
            fnb = int(((yb == 1) & (db == 0)).sum())
            fpb = int(((yb == 0) & (db == 1)).sum())
            boot[b] = (tpb / max(tpb + fnb, 1), tpb / max(tpb + fpb, 1))
        ap_vals = np.array(pf[s]["ap"], dtype=float)
        out["per_surface"][s] = {
            "prevalence_mean": float(np.nanmean(pf[s]["prevalence"])),
            "ap_fold_mean": float(np.nanmean(ap_vals)),
            "ap_fold_std": float(np.nanstd(ap_vals)),
            "ap_fold_values": [None if v != v else round(float(v), 4)
                               for v in ap_vals],
            "lift_mean": float(np.nanmean(pf[s]["lift"])),
            "auroc_fold_mean": float(np.nanmean(pf[s]["auroc"])),
            "auroc_fold_std": float(np.nanstd(pf[s]["auroc"])),
            "n_pos": int(y.sum()),
            "recall_pooled": float(recall),
            "precision_pooled": float(precision),
            "f1_pooled": float(f1_score(y, d, zero_division=0)),
            "tp": tp, "fp": fp, "fn": fn,
            "recall_wilson": _wilson_ci(tp, tp + fn),
            "precision_wilson": _wilson_ci(tp, tp + fp),
            "recall_boot_ci": [float(np.percentile(boot[:, 0], 2.5)),
                               float(np.percentile(boot[:, 0], 97.5))],
            "precision_boot_ci": [float(np.percentile(boot[:, 1], 2.5)),
                                  float(np.percentile(boot[:, 1], 97.5))],
            "meets_recall_target": bool(recall >= args.recall_target)}
        r = out["per_surface"][s]
        print("  %-8s prev=%.3f AP=%.3f+-%.3f lift=%.2f AUROC=%.3f | "
              "pooled recall=%.3f [wilson %.3f-%.3f boot %.3f-%.3f] "
              "precision=%.3f %s"
              % (s, r["prevalence_mean"], r["ap_fold_mean"], r["ap_fold_std"],
                 r["lift_mean"], r["auroc_fold_mean"], r["recall_pooled"],
                 r["recall_wilson"][0], r["recall_wilson"][1],
                 r["recall_boot_ci"][0], r["recall_boot_ci"][1],
                 r["precision_pooled"],
                 "OK" if r["meets_recall_target"] else "BELOW-TARGET"))
    # ---- B4: HAS_CARIES(any), Brier scores, reliability bins --------------
    # Probabilities are never pooled across folds (each fold has its own
    # calibration): per-fold scores are averaged and reliability bins are
    # count-weighted, exactly like the per-fold Average Precision above.
    from sklearn.metrics import average_precision_score, roc_auc_score

    any_y = (oof["y"].sum(1) > 0).astype(int)
    any_d = oof["d"].any(1).astype(int)
    any_p = oof["p"].max(1)
    fold_ids = np.asarray(oof["fold"])
    fold_list = sorted(set(fold_ids.tolist()))
    brier, rel = {}, {}
    for j, s in enumerate(list(SURFACES) + ["HAS_CARIES(any)"]):
        y_t = any_y if s == "HAS_CARIES(any)" else oof["y"][:, j]
        p_t = any_p if s == "HAS_CARIES(any)" else oof["p"][:, j]
        bs = []
        edges = np.linspace(0.0, 1.0, 11)
        n_bin = np.zeros(10)
        sum_p = np.zeros(10)
        sum_y = np.zeros(10)
        for f in fold_list:
            m = fold_ids == f
            bs.append(float(np.mean((p_t[m] - y_t[m]) ** 2)))
            b = np.clip(np.digitize(p_t[m], edges[1:-1]), 0, 9)
            for bi in range(10):
                sel = b == bi
                n_bin[bi] += int(sel.sum())
                sum_p[bi] += float(p_t[m][sel].sum())
                sum_y[bi] += float(y_t[m][sel].sum())
        brier[s] = {"fold_mean": float(np.mean(bs)),
                    "fold_std": float(np.std(bs)),
                    "fold_values": [round(v, 4) for v in bs]}
        rel[s] = {"bin_edges": [round(float(v), 2) for v in edges],
                  "bin_count": [int(v) for v in n_bin],
                  "bin_mean_prob": [None if c == 0 else round(float(sp) / c, 4)
                                    for sp, c in zip(sum_p, n_bin)],
                  "bin_frac_pos": [None if c == 0 else round(float(sy) / c, 4)
                                   for sy, c in zip(sum_y, n_bin)]}
    out["brier"], out["reliability"] = brier, rel
    tp_a = int(((any_y == 1) & (any_d == 1)).sum())
    fp_a = int(((any_y == 0) & (any_d == 1)).sum())
    fn_a = int(((any_y == 1) & (any_d == 0)).sum())
    boot_a = np.zeros((args.n_boot, 2))
    for b in range(args.n_boot):
        pick = rng.choice(cases, size=len(cases), replace=True)
        idx = np.concatenate([np.where(oof["case"] == c)[0] for c in pick])
        yb, db = any_y[idx], any_d[idx]
        tpb = int(((yb == 1) & (db == 1)).sum())
        fnb = int(((yb == 1) & (db == 0)).sum())
        fpb = int(((yb == 0) & (db == 1)).sum())
        boot_a[b] = (tpb / max(tpb + fnb, 1), tpb / max(tpb + fpb, 1))
    ap_a, auroc_a, prev_a = [], [], []
    for f in fold_list:
        m = fold_ids == f
        yy, pp = any_y[m], any_p[m]
        prev_a.append(float(yy.mean()))
        if yy.sum() > 0 and (yy == 0).sum() > 0:
            ap_a.append(float(average_precision_score(yy, pp)))
            auroc_a.append(float(roc_auc_score(yy, pp)))
    prev_m = float(np.mean(prev_a))
    out["any_surface"] = {
        "prevalence_mean": prev_m,
        "ap_fold_mean": float(np.nanmean(ap_a)),
        "ap_fold_std": float(np.nanstd(ap_a)),
        "ap_fold_values": [round(v, 4) for v in ap_a],
        "lift_mean": (float(np.nanmean(ap_a) / prev_m) if prev_m > 0
                      else float("nan")),
        "auroc_fold_mean": float(np.nanmean(auroc_a)),
        "auroc_fold_std": float(np.nanstd(auroc_a)),
        "n_pos": int(any_y.sum()),
        "recall_pooled": tp_a / max(tp_a + fn_a, 1),
        "precision_pooled": tp_a / max(tp_a + fp_a, 1),
        "f1_pooled": float(f1_score(any_y, any_d, zero_division=0)),
        "tp": tp_a, "fp": fp_a, "fn": fn_a,
        "recall_wilson": _wilson_ci(tp_a, tp_a + fn_a),
        "precision_wilson": _wilson_ci(tp_a, tp_a + fp_a),
        "recall_boot_ci": [float(np.percentile(boot_a[:, 0], 2.5)),
                           float(np.percentile(boot_a[:, 0], 97.5))],
        "precision_boot_ci": [float(np.percentile(boot_a[:, 1], 2.5)),
                              float(np.percentile(boot_a[:, 1], 97.5))],
        "meets_recall_target": bool(tp_a / max(tp_a + fn_a, 1)
                                    >= args.recall_target)}
    ra = out["any_surface"]
    print("  %-8s prev=%.3f AP=%.3f+-%.3f lift=%.2f AUROC=%.3f | "
          "pooled recall=%.3f [wilson %.3f-%.3f boot %.3f-%.3f] "
          "precision=%.3f %s"
          % ("ANY", ra["prevalence_mean"], ra["ap_fold_mean"],
             ra["ap_fold_std"], ra["lift_mean"], ra["auroc_fold_mean"],
             ra["recall_pooled"], ra["recall_wilson"][0],
             ra["recall_wilson"][1], ra["recall_boot_ci"][0],
             ra["recall_boot_ci"][1], ra["precision_pooled"],
             "OK" if ra["meets_recall_target"] else "BELOW-TARGET"))
    for s in list(SURFACES) + ["HAS_CARIES(any)"]:
        print("  brier %-16s %.4f+-%.4f" % (s, brier[s]["fold_mean"],
                                            brier[s]["fold_std"]))
    if not all(v["meets_recall_target"] for v in out["per_surface"].values()):
        out["note"] = ("Threshold recall always regresses on unseen data: the "
                       "value picked as the largest threshold still meeting the "
                       "target on one sample gives less than the target on the "
                       "next. Either raise the tuning target (e.g. 0.85) or "
                       "report Recall with the interval above instead of the "
                       "point estimate.")
    stem = Path(oof["path"]).stem
    op = Path(args.out) if getattr(args, "out", None) else (
        OUTPUTS_DIR / ("oof_report_%s.json" % stem))
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["| surface | prevalence | AP (mean+-SD) | lift | AUROC | recall pooled"
          " | precision pooled | Brier (fold mean+-SD) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in list(SURFACES) + ["HAS_CARIES(any)"]:
        r = (out["any_surface"] if s == "HAS_CARIES(any)"
             else out["per_surface"][s])
        md.append("| %s | %.3f | %.3f+-%.3f | %.2f | %.3f | %.3f (%.3f-%.3f) | "
                  "%.3f (%.3f-%.3f) | %.3f+-%.3f |"
                  % (s, r["prevalence_mean"], r["ap_fold_mean"],
                     r["ap_fold_std"], r["lift_mean"], r["auroc_fold_mean"],
                     r["recall_pooled"], r["recall_wilson"][0],
                     r["recall_wilson"][1], r["precision_pooled"],
                     r["precision_wilson"][0], r["precision_wilson"][1],
                     brier[s]["fold_mean"], brier[s]["fold_std"]))
    (op.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    _plot_oof_reliability(rel, brier, op)
    print("[oof-report] -> %s/.md (+ reliability png)" % op)


def _nb_corrected_t(deltas, n_train: int, n_test: int) -> tuple[float, float, float]:
    """Nadeau & Bengio (2003) corrected resampled t interval for the mean of
    cross-validated differences. The standard error is inflated by
    (1/k + n_test/n_train) because every fold's training set overlaps.
    """
    d = np.asarray([float(x) for x in deltas if x == x], dtype=float)
    k = len(d)
    if k == 0:
        return (float("nan"), float("nan"), float("nan"))
    if k < 2:
        return (float(d.mean()), float("nan"), float("nan"))
    se = float(np.sqrt((1.0 / k + n_test / max(n_train, 1)) * d.var(ddof=1)))
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, k - 1))
    except Exception:  # pragma: no cover
        tcrit = 1.959963985
    m = float(d.mean())
    return (m, m - tcrit * se, m + tcrit * se)


def run_compare_oof(args) -> None:
    """E1b: decide between two arms from their OOF files (plan 2.2 + fixes 1, 3).

    Each --pair is 'A.csv,B.csv' from the SAME split and seed. Repeating the
    option with more seeds turns seed into a replicate of the training
    procedure, which plain test-set bootstrap cannot see.

    Verdicts per surface:
      distinguishable : patient-level cluster bootstrap interval of delta-AP
                        excludes zero
      superior        : corrected-t lower bound > 0 (training variance in)
      non_inferior    : corrected-t lower bound > -margin (fix 1). A wide
                        interval that crosses zero is NOT evidence of equality,
                        so the margin is declared before the run.
    """
    pairs = []
    for spec in args.pair:
        a, _, b = spec.rpartition(",")
        if not a.strip() or not b.strip():
            raise SystemExit("--pair expects 'A.csv,B.csv', got %r" % spec)
        pairs.append((Path(a.strip()), Path(b.strip())))
    out = {"margin": float(args.margin), "n_boot": int(args.n_boot),
           "pairs": [[str(a), str(b)] for a, b in pairs], "per_surface": {}}
    from sklearn.metrics import average_precision_score as _ap

    deltas = {s: [] for s in SURFACES}
    boots = {s: [] for s in SURFACES}
    n_rows = []
    for pa, pb in pairs:
        A, B = load_oof_csv(pa), load_oof_csv(pb)
        keyA = list(zip(A["case"].tolist(), A["fdi"].tolist(),
                        A["fold"].tolist()))
        keyB = list(zip(B["case"].tolist(), B["fdi"].tolist(),
                        B["fold"].tolist()))
        if keyA != keyB:
            raise SystemExit("arms are not the same teeth on the same folds: "
                             "%s vs %s" % (pa, pb))
        folds = sorted(set(A["fold"].tolist()))
        fold_of = A["fold"]
        n_rows.append(len(keyA))
        cases = np.unique(A["case"])
        rng = np.random.RandomState(args.seed + 977)
        for j, s in enumerate(SURFACES):
            def _dap_folds(idx, j=j):
                """Per-fold Average Precision difference, degenerate folds
                dropped (a fold without a positive has no curve)."""
                vals = []
                for f in folds:
                    m = fold_of[idx] == f
                    y = A["y"][idx][m, j]
                    if int(y.sum()) == 0:
                        continue
                    vals.append(float(_ap(y, A["p"][idx][m, j]))
                                - float(_ap(y, B["p"][idx][m, j])))
                return vals

            deltas[s].extend(_dap_folds(np.arange(len(keyA))))
            bs = np.full(args.n_boot, np.nan)
            for b in range(args.n_boot):
                pick = rng.choice(cases, size=len(cases), replace=True)
                idx = np.concatenate([np.where(A["case"] == c)[0]
                                      for c in pick])
                v = _dap_folds(idx)
                if v:
                    bs[b] = float(np.mean(v))
            boots[s].append(bs)
    n_folds_seen = max(1, len(set(load_oof_csv(pairs[0][0])["fold"].tolist())))
    n_test = int(round((np.mean(n_rows) if n_rows else 0) / n_folds_seen))
    n_train = n_test * max(1, n_folds_seen - 1)
    for s in SURFACES:
        arr = np.stack(boots[s], axis=0) if boots[s] else np.zeros((0, 0))
        mean_boot = np.nanmean(arr, axis=0) if arr.size else np.zeros(0)
        mean_boot = mean_boot[~np.isnan(mean_boot)]
        lo, hi = ((float(np.percentile(mean_boot, 2.5)),
                   float(np.percentile(mean_boot, 97.5)))
                  if mean_boot.size else (float("nan"), float("nan")))
        m, nb_lo, nb_hi = _nb_corrected_t(deltas[s], n_train, n_test)
        out["per_surface"][s] = {
            "delta_ap_foldwise": [None if v != v else round(v, 4)
                                  for v in deltas[s]],
            "delta_ap_mean": m, "delta_ap_boot_ci": [lo, hi],
            "delta_ap_corrected_t_ci": [nb_lo, nb_hi],
            "k_estimates": len(deltas[s]),
            "n_train_per_fold": int(n_train), "n_test_per_fold": int(n_test),
            "distinguishable": bool(lo > 0 or hi < 0),
            "superior": bool(nb_lo > 0),
            "non_inferior": bool(nb_lo > -float(args.margin))}
        r = out["per_surface"][s]
        print("  %-8s dAP=%+.3f boot[%+.3f,%+.3f] corrected-t[%+.3f,%+.3f] "
              "k=%d  %s%s"
              % (s, r["delta_ap_mean"], lo, hi, nb_lo, nb_hi,
                 r["k_estimates"],
                 "SUPERIOR" if r["superior"] else
                 ("NON-INFERIOR" if r["non_inferior"] else "WORSE"),
                 " (distinguishable)" if r["distinguishable"] else ""))
    _out = Path(args.out) if getattr(args, "out", None) else (
        OUTPUTS_DIR / ("compare_oof_%s.json"
                       % "_".join(Path(a).stem for a, _ in pairs[:1])))
    _out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[compare-oof] margin=%.3f pairs=%d -> %s"
          % (args.margin, len(pairs), _out))


def _fit_temperature(z, y, bounds=(-3.0, 3.0)) -> float:
    """Temperature scaling: minimize mean log-loss over log T (Guo et al. 2017)."""
    from scipy.optimize import minimize_scalar

    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(log_t):
        zz = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, zz) - y * zz))

    return float(np.exp(minimize_scalar(nll, bounds=bounds,
                                        method="bounded").x))


def _reliability_bins(y, p, n_bins=10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        hi = edges[i + 1]
        m = (p >= edges[i]) & ((p <= hi) if i == n_bins - 1 else (p < hi))
        n = int(m.sum())
        rows.append({"bin": i, "lo": float(edges[i]), "hi": float(hi), "n": n,
                     "mean_prob": float(p[m].mean()) if n else None,
                     "frac_pos": float(y[m].mean()) if n else None})
    return rows


def _score_checkpoint(weights, samples, seed: int, num_workers: int = 4):
    """Forward one checkpoint over explicit samples; returns logits (n, 3)."""
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(weights), map_location=device)
    pre = ablation_to_pre(str(ckpt.get("ablation", "pca")),
                          img_size=int(ckpt.get("img_size", IMG_SIZE)),
                          input_norm=str(ckpt.get("input_norm", "zscore")))
    model = build_cnn(str(ckpt.get("backbone", "efficientnet_b0")),
                      bool(ckpt.get("use_metadata", False))).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
    model.eval()
    dl = DataLoader(torch_dataset(samples, pre, train=False, seed=seed),
                    batch_size=64, shuffle=False, num_workers=num_workers)
    chunks = []
    with torch.no_grad():
        for xb, _yb, mb, fb, _cb in dl:
            xb = xb.to(device)
            use_meta = getattr(model, "use_metadata", False)
            lg = (model(xb, mb.to(device), fb.to(device)) if use_meta
                  else model(xb))
            chunks.append(lg.cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def run_calibrate(args) -> None:
    """E3 (fix 2): fit the ENSEMBLE's temperature and thresholds on the peeked
    set, not on out-of-fold single-model scores.

    The 100 peeked cases were excluded from every fold's training, so all five
    fold models are out-of-sample on them; averaging their probabilities
    changes the score distribution, which would move a single-model threshold
    away from its recall target. This command re-derives both the temperature
    and the threshold on that set. The peeked set has already been inspected,
    so the numbers printed here are DIAGNOSTIC ONLY and must never be quoted
    as performance; performance comes from the external test alone.
    """
    # PROTOCOL_LOCK.md L59-62: temperature + threshold are fitted on the 100
    # peeked cases, which every fold excluded from training. apply_exclude()
    # REMOVES those cases (in-sample calibration) -- keep them instead.
    _ids = parse_exclude(args.calib)
    samples = [s for s in load_manifest()[1] if int(s["case_id"]) in _ids]
    n_exc = len(_ids)
    print("[calibrate] calibration set: %d teeth in %d cases (held out from "
          "every fold)" % (len(samples), len({int(s["case_id"]) for s in samples})))
    samples = samples[:args.limit] if args.limit else samples
    if not samples:
        raise SystemExit("calibration set is empty (check --calib)")
    cases = sorted({int(s["case_id"]) for s in samples})
    Y = np.stack([s["label"] for s in samples]).astype(int)
    logs = []
    for fi in range(args.n_folds):
        w = OUTPUTS_DIR / ("%s%d.pt" % (args.prefix, fi))
        if not w.exists():
            raise SystemExit("missing fold checkpoint: %s" % w)
        logs.append(_score_checkpoint(w, samples, args.seed, args.num_workers))
        print("[calibrate] fold %d scored (%s)" % (fi, w.name))
    L = np.stack(logs, axis=0)                      # (n_folds, n, 3)
    p_ens = 1.0 / (1.0 + np.exp(-L)).mean(axis=0)   # averaged probabilities
    eps = 1e-6
    z_ens = np.log(np.clip(p_ens, eps, 1 - eps)
                   / (1.0 - np.clip(p_ens, eps, 1 - eps)))
    out = {"calibration_set": {"n_cases": len(cases), "n_teeth": len(samples),
                               "source": args.calib, "excluded_from_training": n_exc,
                               "warning": ("peeked set: diagnostic only, never "
                                           "report as performance")},
           "n_folds": args.n_folds, "prefix": args.prefix,
           "recall_target": float(args.recall_target),
           "temperature": {}, "threshold": {}, "diagnostics": {}}
    from sklearn.metrics import precision_recall_curve
    for j, s in enumerate(SURFACES):
        y = Y[:, j]
        t = _fit_temperature(z_ens[:, j], y)
        p_cal = 1.0 / (1.0 + np.exp(-z_ens[:, j] / t))
        prec, rec, thr = precision_recall_curve(y, p_cal)
        ok = np.where(rec[:-1] >= args.recall_target)[0]
        if len(ok):
            thr_final = float(thr[ok[-1]])
        else:
            thr_final = float(thr[int(np.argmax(rec[:-1]))])
        pred = (p_cal >= thr_final).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        out["temperature"][s] = t
        out["threshold"][s] = thr_final
        out["diagnostics"][s] = {
            "brier_raw": float(np.mean((p_ens[:, j] - y) ** 2)),
            "brier_calibrated": float(np.mean((p_cal - y) ** 2)),
            "recall_at_threshold": tp / max(tp + fn, 1),
            "precision_at_threshold": tp / max(tp + fp, 1),
            "n_pos": int(y.sum()),
            "recall_wilson": _wilson_ci(tp, tp + fn),
            "reliability_calibrated": _reliability_bins(y, p_cal)}
        d = out["diagnostics"][s]
        print("  %-8s T=%.3f thr=%.3f  calib-recall=%.3f precision=%.3f "
              "Brier %.4f -> %.4f"
              % (s, t, thr_final, d["recall_at_threshold"],
                 d["precision_at_threshold"], d["brier_raw"],
                 d["brier_calibrated"]))
    try:
        plt = _mpl()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, s in zip(axes, SURFACES):
            rows = out["diagnostics"][s]["reliability_calibrated"]
            xs = [r["mean_prob"] for r in rows if r["n"]]
            ys_ = [r["frac_pos"] for r in rows if r["n"]]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot(xs, ys_, "o-")
            ax.set_title(s + " reliability")
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("observed frequency")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / ("calibration_reliability_%s.png"
                                   % args.ablation), dpi=120)
    except Exception as exc:  # pragma: no cover
        print("[calibrate] reliability plot skipped: %s" % exc)
    op = Path(args.out) if args.out else (OUTPUTS_DIR /
                                         ("calibration_%s.json" % args.ablation))
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[calibrate] -> %s (peeked set: diagnostic only)" % op)


def run_kappa(args) -> None:
    """E4 (fix 7): inter-rater agreement as the ceiling the model can reach.

    Expects a CSV with columns: case_id,tooth_fdi,surface,rater_a,rater_b
    (surface in Distal/Mesial/Occlusal, ratings 0/1). Cohen's kappa is reported
    per surface with a patient-level bootstrap interval. A low kappa on a
    surface means that surface's Average Precision is limited by the reference
    standard, not by the classifier.
    """
    import csv as _csv

    rows = list(_csv.DictReader(open(args.pairs, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("empty rating file: %s" % args.pairs)
    per = {s: {"case": [], "a": [], "b": []} for s in SURFACES}
    for r in rows:
        s = str(r["surface"]).strip()
        s = s[0].upper() + s[1:].lower()
        if s not in per:
            continue
        per[s]["case"].append(int(r["case_id"]))
        per[s]["a"].append(int(float(r["rater_a"])))
        per[s]["b"].append(int(float(r["rater_b"])))

    def _kappa(a, b):
        a, b = np.asarray(a), np.asarray(b)
        n = len(a)
        if n == 0:
            return (float("nan"), float("nan"))
        po = float((a == b).mean())
        pa1, pb1 = float(a.mean()), float(b.mean())
        pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
        if abs(1 - pe) < 1e-12:
            return (float("nan"), po)
        return (float((po - pe) / (1 - pe)), po)

    rng = np.random.RandomState(args.seed + 4242)
    out = {"file": args.pairs, "n_boot": int(args.n_boot), "per_surface": {}}
    all_a, all_b, all_c = [], [], []
    for s in SURFACES:
        a, b, c = per[s]["a"], per[s]["b"], per[s]["case"]
        all_a += a
        all_b += b
        all_c += c
        k, po = _kappa(a, b)
        cases = np.unique(c) if c else np.array([])
        bs = []
        for _ in range(args.n_boot):
            if len(cases) == 0:
                break
            pick = rng.choice(cases, size=len(cases), replace=True)
            idx = np.concatenate([np.where(np.asarray(c) == x)[0] for x in pick])
            bs.append(_kappa(np.asarray(a)[idx], np.asarray(b)[idx])[0])
        bs = np.asarray([v for v in bs if v == v], dtype=float)
        out["per_surface"][s] = {
            "n_ratings": len(a), "n_cases": int(len(cases)),
            "kappa": k, "percent_agreement": po,
            "kappa_boot_ci": ([float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))]
                              if bs.size else [None, None]),
            "n_pos_rater_a": int(np.sum(a))}
        print("  %-8s n=%4d kappa=%.3f [%.3f,%.3f] agreement=%.3f"
              % (s, len(a), k, out["per_surface"][s]["kappa_boot_ci"][0]
                 if bs.size else float("nan"),
                 out["per_surface"][s]["kappa_boot_ci"][1] if bs.size
                 else float("nan"), po))
    k_all, po_all = _kappa(all_a, all_b)
    out["overall"] = {"n_ratings": len(all_a), "kappa": k_all,
                      "percent_agreement": po_all}
    print("  %-8s n=%4d kappa=%.3f agreement=%.3f"
          % ("OVERALL", len(all_a), k_all, po_all))
    op = Path(args.out) if args.out else (OUTPUTS_DIR / "interrater_kappa.json")
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[kappa] -> %s" % op)


def run_sample_size(args) -> None:
    """E5 prep (fix 7): external-test size from the prevalence actually measured.

    Closed form for a proportion (Wald on the logit-free scale is adequate at
    these sizes), inflated by a design effect because the teeth per case are
    correlated, then divided by the measured positives per case.
    """
    prev = {}
    if args.oof_json:
        src = json.loads(Path(args.oof_json).read_text(encoding="utf-8"))
        block = src.get("per_surface", src.get("pooled", {}))
        for s in SURFACES:
            if s in block and "prevalence_mean" in block[s]:
                prev[s] = float(block[s]["prevalence_mean"])
            elif s in block and "prevalence" in block[s]:
                prev[s] = float(block[s]["prevalence"])
    for spec in (args.prevalence or []):
        name, _, val = spec.partition("=")
        prev[name.strip().capitalize()] = float(val)
    if len(prev) != 3:
        raise SystemExit("need prevalence for all surfaces "
                         "(use --oof-json or three --prevalence S=v)")
    z = {0.90: 1.644853627, 0.95: 1.959963985, 0.99: 2.575829304}[args.confidence]
    out = {"sensitivity": args.sensitivity, "half_width": args.half_width,
           "design_effect": args.design_effect, "confidence": args.confidence,
           "teeth_per_case": args.teeth_per_case, "per_surface": {}}
    print("[sample-size] target sensitivity %.2f  half-width %.2f  "
          "design effect %.2f  confidence %.2f"
          % (args.sensitivity, args.half_width, args.design_effect,
             args.confidence))
    for s in SURFACES:
        p, w = args.sensitivity, args.half_width
        n_pos = z * z * p * (1 - p) / (w * w)
        n_pos_de = n_pos * args.design_effect
        per_case = prev[s] * args.teeth_per_case
        n_cases = n_pos_de / max(per_case, 1e-9)
        out["per_surface"][s] = {
            "prevalence": prev[s], "positives_needed": int(np.ceil(n_pos)),
            "positives_needed_with_design_effect": int(np.ceil(n_pos_de)),
            "positives_per_case": per_case,
            "cases_needed": int(np.ceil(n_cases))}
        r = out["per_surface"][s]
        print("  %-8s prevalence=%.3f -> positives>=%d (x%.2f design) -> "
              "cases >= %d"
              % (s, r["prevalence"], r["positives_needed_with_design_effect"],
                 args.design_effect, r["cases_needed"]))
    out["cases_needed_worst_surface"] = max(
        v["cases_needed"] for v in out["per_surface"].values())
    print("[sample-size] plan for at least %d external cases"
          % out["cases_needed_worst_surface"])
    op = Path(args.out) if args.out else (OUTPUTS_DIR / "external_sample_size.json")
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[sample-size] -> %s" % op)


def run_find_epoch(args) -> None:
    """Stage 1 of 2 (Varma & Simon 2006): pick the epoch budget on tune ONLY.

    Single case-grouped split -> nested tune/test. The test-half is held out
    and NEVER used for any decision (its metrics file exists for provenance
    but must not drive choices). Early stopping on tune macro Average Precision
    with patience. Thresholds tuned once, after selection, at --recall-target.
    Fixed config (efficientnet_b0/focal/clahe_pca): no multiple comparisons.
    """
    _meta, samples = load_manifest()
    samples, n_exc = apply_exclude(samples, args.exclude)
    pre = ablation_to_pre(args.ablation, img_size=args.img_size,
                          input_norm=args.input_norm, aug_plus=args.aug_plus)
    cfg = TrainConfig(backbone=args.backbone, ablation=args.ablation,
                      loss=args.loss, use_metadata=args.use_metadata,
                      epochs=args.max_epochs, batch_size=args.batch_size,
                      lr=args.lr, seed=args.seed, patience=args.patience,
                      num_workers=args.num_workers,
                      freeze_backbone=args.freeze_backbone,
                      partial_blocks=args.partial_blocks,
                      backbone_lr=args.backbone_lr, excluded_n=n_exc,
                      recall_target=args.recall_target)
    cases = np.array([s["case_id"] for s in samples])
    tr, va = _split_indices(cases, args.seed, "single")[0]
    tr_s, va_s = [samples[i] for i in tr], [samples[i] for i in va]
    print(f"[find-epoch] train cases={len(set(cases[tr]))} "
          f"tuneval cases={len(set(cases[va]))} (test-half held out, untouched)")
    tag = args.tag
    r = train_fold(tr_s, va_s, cfg, pre, out_prefix=f"epsearch_{args.ablation}{tag}")
    out = {"best_epoch": r["best_epoch"], "ablation": args.ablation,
           "backbone": args.backbone, "loss": args.loss,
           "use_metadata": args.use_metadata, "seed": args.seed,
           "lr": args.lr, "freeze_backbone": args.freeze_backbone,
           "partial_blocks": args.partial_blocks, "backbone_lr": args.backbone_lr,
           "excluded_cases": n_exc,
           "weights": r["weights"],
           "note": "epoch budget selected on tune-half only; test-half metrics "
                   "in epsearch files are provenance-only, not for decisions"}
    (OUTPUTS_DIR / f"find_epoch_{args.ablation}{tag}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"[find-epoch] BEST_EPOCH={r['best_epoch']} -> next: "
          f"train --epochs {r['best_epoch']} --cv stratified-groupkfold "
          f"(same --ablation/--backbone/--loss)")


def run_ablate(args) -> None:
    meta, samples = load_manifest()
    samples, n_exc = apply_exclude(samples, args.exclude)
    cases = np.array([s["case_id"] for s in samples])
    tr, va = _split_indices(cases, args.seed, "single")[0]
    tr_s, va_s = [samples[i] for i in tr], [samples[i] for i in va]
    table = {}
    for ab in ("off", "clahe", "pca", "clahe_pca"):
        pre = ablation_to_pre(ab, img_size=args.img_size,
                              input_norm=args.input_norm, aug_plus=args.aug_plus)
        cfg = TrainConfig(backbone=args.backbone, ablation=ab, loss=args.loss,
                          use_metadata=args.use_metadata, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                          num_workers=args.num_workers,
                          partial_blocks=args.partial_blocks,
                          backbone_lr=args.backbone_lr,
                          freeze_backbone=args.freeze_backbone, excluded_n=n_exc,
                          recall_target=args.recall_target)
        print(f"\n===== ablation: {ab} =====")
        r = train_fold(tr_s, va_s, cfg, pre, out_prefix=f"abl_{ab}")
        table[ab] = {"macro_f1": r["macro_f1"],
                     "per_surface": {s: {k: r["metrics"][s][k]
                                         for k in ("recall", "precision", "f1", "pr_auc")}
                                     for s in SURFACES}}
    print("\n===== ABLATION SUMMARY (same split/seed) =====")
    print(f"{'mode':<10} {'macroF1':>8} | " + " | ".join(f"{s[:4]:>14}" for s in SURFACES))
    for ab in ("off", "clahe", "pca", "clahe_pca"):
        row = table[ab]
        detail = " | ".join(f"R={row['per_surface'][s]['recall']:.2f}/F1={row['per_surface'][s]['f1']:.2f}"
                             for s in SURFACES)
        print(f"{ab:<10} {row['macro_f1']:>8.4f} | {detail}")
    (OUTPUTS_DIR / "ablation.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Texture baseline (handcrafted LBP/HOG-lite + intensity stats + RF)
# ----------------------------------------------------------------------------

def texture_features(gray224: np.ndarray, mask224: np.ndarray | None = None) -> np.ndarray:
    """Handcrafted descriptor: intensity stats + Sobel stats + LBP-lite + HOG-lite."""
    require_cv2()
    g = gray224.astype(np.float32)
    fg = (mask224 > 0) if mask224 is not None else np.ones_like(g, bool)
    if fg.sum() < 8:
        fg = np.ones_like(g, bool)
    v = g[fg]
    qs = np.quantile(v, [0.1, 0.5, 0.9]) if len(v) else np.zeros(3)
    feats = [float(v.mean()) if len(v) else 0.0, float(v.std()) if len(v) else 0.0,
             *[float(q) for q in qs]]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mv = mag[fg]
    feats += [float(mv.mean()) if len(mv) else 0.0, float(mv.std()) if len(mv) else 0.0]
    hist, _ = np.histogram((ang[fg] % 180) if fg.sum() else ang.ravel(),
                           bins=9, range=(0, 180))
    feats += (hist / max(1.0, hist.sum())).tolist()
    c = (g[1:-1, 1:-1] > g[:-2, :-2]).astype(np.uint8) * 1 + \
        (g[1:-1, 1:-1] > g[:-2, 1:-1]).astype(np.uint8) * 2 + \
        (g[1:-1, 1:-1] > g[:-2, 2:]).astype(np.uint8) * 4 + \
        (g[1:-1, 1:-1] > g[1:-1, 2:]).astype(np.uint8) * 8 + \
        (g[1:-1, 1:-1] > g[2:, 2:]).astype(np.uint8) * 16 + \
        (g[1:-1, 1:-1] > g[2:, 1:-1]).astype(np.uint8) * 32 + \
        (g[1:-1, 1:-1] > g[2:, :-2]).astype(np.uint8) * 64 + \
        (g[1:-1, 1:-1] > g[1:-1, :-2]).astype(np.uint8) * 128
    lh, _ = np.histogram(c.ravel(), bins=10, range=(0, 256))
    feats += (lh / max(1.0, lh.sum())).tolist()
    return np.asarray(feats, dtype=np.float32)


def run_baseline(args) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit

    _meta, samples = load_manifest()
    samples, n_exc = apply_exclude(samples, args.exclude)
    if args.features == "fdi":
        # Tooth-identity shortcut probe (1.7): Federation Dentaire Internationale (FDI)
        # one-hot + is_upper + quadrant + bbox geometry. NO pixels, NO coordinates.
        # If this reaches the Convolutional Neural Network (CNN), tooth type alone
        # predicts surfaces and the image model may ride the same shortcut.
        print("[baseline] FDI-only tabular features (no pixels) ...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.pipeline import Pipeline as _Pipe
        from sklearn.preprocessing import StandardScaler

        FDI_ALL = [q * 10 + p for q in (1, 2, 3, 4) for p in range(1, 9)]
        X, Y, G = [], [], []
        for s in samples:
            fdi = int(s["fdi"])
            x1, y1, x2, y2 = (float(v) for v in s["bbox"])
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            row = [1.0 if is_upper_jaw(fdi) else 0.0,
                   float(get_quadrant(fdi)) / 4.0, w / h, float(np.log(w * h))]
            row += [1.0 if fdi == u else 0.0 for u in FDI_ALL]
            X.append(row)
            Y.append(s["label"])
            G.append(s["case_id"])
        X, Y, G = map(np.asarray, (X, Y, G))
        clf = MultiOutputClassifier(_Pipe([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000))]))
    else:
        # Baseline texture pipeline is fixed to zscore norm (never ran otherwise);
        # size follows --img-size so it can match the Convolutional Neural Network (CNN).
        pre = ablation_to_pre(args.ablation, img_size=args.img_size)
        print(f"[baseline] extracting texture features ({args.ablation}) ...")
        from sklearn.ensemble import RandomForestClassifier

        X, Y, G = [], [], []
        for s in samples:
            gray = cv2.imread(str(CACHE_DIR / s["crop_path"]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(CACHE_DIR / s["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            img = preprocess_tooth(gray, mask > 127, s["fdi"], pre)
            X.append(texture_features((img * 50 + 128).clip(0, 255).astype(np.uint8)))
            Y.append(s["label"])
            G.append(s["case_id"])
        X, Y, G = map(np.asarray, (X, Y, G))
        clf = RandomForestClassifier(n_estimators=300, random_state=args.seed,
                                     class_weight="balanced_subsample", n_jobs=-1)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=args.seed).split(X, groups=G))
    clf.fit(X[tr], Y[tr])

    def _proba(idx):
        plist = clf.predict_proba(X[idx])
        return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(idx))
                                for p in plist])

    # Nested protocol: tune thresholds on tune-half, report on disjoint test-half.
    va_cases = G[va]
    tune_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=args.seed + 7919).split(
            np.zeros(len(va)), groups=va_cases))
    tune_g, test_g = va[tune_idx], va[test_idx]
    thrs = tune_thresholds(Y[tune_g], _proba(tune_g), args.recall_target)
    mets = compute_metrics(Y[test_g], _proba(test_g), thrs)
    ci = bootstrap_case_ci(Y[test_g], _proba(test_g), thrs, G[test_g])
    print(json.dumps({"thresholds": thrs, "metrics": mets, "ci95": ci,
                      "n_tune_cases": len(set(G[tune_g].tolist())),
                      "n_test_cases": len(set(G[test_g].tolist())),
                      "note": "thresholds tuned on tune-half, metrics + bootstrap CI "
                              "on disjoint test-half (no threshold-selection optimism)"},
                     indent=1))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / f"baseline_{args.features}_{args.ablation}.json").write_text(
        json.dumps({"thresholds": thrs, "metrics": mets, "ci95": ci,
                    "excluded_cases": n_exc,
                    "n_tune_cases": len(set(G[tune_g].tolist())),
                    "n_test_cases": len(set(G[test_g].tolist()))}, indent=1),
        encoding="utf-8")
    import joblib

    joblib.dump(clf, OUTPUTS_DIR / f"baseline_{args.features}_{args.ablation}.pkl")


# ----------------------------------------------------------------------------
# Inference: pano image -> per-tooth prediction (single model)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Paired model comparison on identical test teeth (1.2)
# ----------------------------------------------------------------------------

def run_compare(args) -> None:
    """Paired A-against-B comparison with bootstrap Confidence Interval of the difference.

    Both checkpoints run on the SAME nested test-half (same seed); metrics are
    recomputed on case-resampled teeth and the per-resample A-minus-B difference
    is summarized. A difference whose 95% interval excludes zero is
    distinguishable on this data; otherwise the two models tie here.
    Each checkpoint uses its OWN preprocessing (no retraining, no tuning here).
    """
    import torch
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader

    _, samples = load_manifest()
    cases = np.array([s["case_id"] for s in samples])
    tr, va = _split_indices(cases, args.seed, "single")[0]
    _, test_s = _tune_test_split([samples[i] for i in va], args.seed + 7919)
    groups = np.array([s["case_id"] for s in test_s])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs = {}
    for tag, w in (("a", args.weights_a), ("b", args.weights_b)):
        ckpt = torch.load(str(w), map_location=device)
        pre = ablation_to_pre(ckpt.get("ablation", "clahe_pca"),
                              img_size=int(ckpt.get("img_size", 224)),
                              input_norm=str(ckpt.get("input_norm", "zscore")))
        model = build_cnn(ckpt.get("backbone", "efficientnet_b0"),
                          ckpt.get("use_metadata", False)).to(device)
        model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
        dl = DataLoader(torch_dataset(test_s, pre, train=False, seed=args.seed),
                        batch_size=64, shuffle=False,
                        num_workers=args.num_workers)
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, yb, mb, fb, cb in dl:
                xb = xb.to(device)
                use_meta = getattr(model, "use_metadata", False)
                lg = (model(xb, mb.to(device), fb.to(device)) if use_meta
                      else model(xb))
                ps.append(torch.sigmoid(lg).cpu().numpy())
        probs[tag] = np.concatenate(ps)
    Y = np.stack([s["label"] for s in test_s]).astype(int)
    out = {"n_teeth": len(test_s), "n_cases": len(set(groups.tolist())),
           "n_boot": args.n_boot, "per_surface": {}}
    rng = np.random.RandomState(args.seed + 271)
    uniq = np.unique(groups)
    for j, sname in enumerate(SURFACES):
        yt = Y[:, j]
        ap = {t: (float(average_precision_score(yt, probs[t][:, j]))
                   if yt.sum() > 0 else float("nan")) for t in ("a", "b")}
        diffs = []
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == c)[0] for c in pick])
            yb = yt[idx]
            if yb.sum() == 0:
                continue
            diffs.append(float(average_precision_score(yb, probs["a"][idx, j]))
                         - float(average_precision_score(yb, probs["b"][idx, j])))
        d = np.asarray(diffs, dtype=float)
        out["per_surface"][sname] = {
            "ap_a": ap["a"], "ap_b": ap["b"],
            "delta_a_minus_b": {"mean": float(np.mean(d)),
                                "lo": float(np.percentile(d, 2.5)),
                                "hi": float(np.percentile(d, 97.5))},
            "n_pos": int(yt.sum())}
        dd = out["per_surface"][sname]["delta_a_minus_b"]
        flag = "DISTINGUISHABLE" if dd["lo"] > 0 or dd["hi"] < 0 else "tie"
        print("  %-8s AP a=%.3f b=%.3f delta=%+.3f [%+.3f,%+.3f] %s"
              % (sname, ap["a"], ap["b"], dd["mean"], dd["lo"], dd["hi"], flag))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / ("compare_" + Path(args.weights_a).stem
                              + "_vs_" + Path(args.weights_b).stem + ".json")
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[compare] %s" % out_path)



def load_trained(backbone: str, weights: Path, use_metadata: bool, device: str):
    import torch

    ckpt = torch.load(str(weights), map_location=device)
    model = build_cnn(ckpt.get("backbone", backbone),
                      ckpt.get("use_metadata", use_metadata)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_image(image_path: str | Path, weights: str | Path,
                  ablation: str = "clahe_pca", backbone: str = "resnet50",
                  use_metadata: bool = False, device: str | None = None) -> dict:
    import torch

    require_cv2()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = ToothDetector(device=device)
    pano, obs = detector.detect(image_path)
    gray_full = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    model, ckpt = load_trained(backbone, Path(weights), use_metadata, device)
    thrs = ckpt.get("thresholds", [0.5, 0.5, 0.5])
    pre = ablation_to_pre(ckpt.get("ablation", ablation),
                          img_size=int(ckpt.get("img_size", IMG_SIZE)),
                          input_norm=str(ckpt.get("input_norm", "zscore")))
    teeth_out = []
    with torch.no_grad():
        for o in obs:
            g = gray_full[o.crop_coords[1]:o.crop_coords[3],
                          o.crop_coords[0]:o.crop_coords[2]]
            if g.size == 0:
                continue
            img = preprocess_tooth(g, np.asarray(o.mask_crop, bool), o.fdi, pre)
            x = torch.from_numpy(np.stack([img] * 3)[None]).float().to(device)
            if ckpt.get("use_metadata", use_metadata):
                meta = torch.tensor([[1.0 if is_upper_jaw(o.fdi) else 0.0,
                                      get_quadrant(o.fdi) / 4.0]]).float().to(device)
                fdi_t = torch.tensor([o.fdi % 60]).to(device)
                prob = torch.sigmoid(model(x, meta, fdi_t)).cpu().numpy()[0]
            else:
                prob = torch.sigmoid(model(x)).cpu().numpy()[0]
            flags = [(prob[j] >= thrs[j]) for j in range(3)]
            teeth_out.append({
                "tooth_id": str(o.fdi),
                "has_caries": bool(any(flags)),
                "confidence": float(prob.max()),
                "predicted_surfaces": [SURFACES[j] for j in range(3) if flags[j]],
                "probabilities": {SURFACES[j]: float(prob[j]) for j in range(3)},
                "pano_confidence": float(o.pano_conf),
                "bbox": [int(v) for v in o.bbox_xyxy],
            })
    return {"image": str(image_path), "ablation": ckpt.get("ablation", ablation),
            "thresholds": [float(t) for t in thrs], "teeth": teeth_out}


# ----------------------------------------------------------------------------
# Data QA: 4-panel figure per tooth (plan section 2.2). Read-only, no decisions.
# ----------------------------------------------------------------------------

def _qa_single(case_id: int, fdi: int, rec: dict, pre: PreConfig,
               outdir: Path) -> dict:
    """Panels: (1) pano+FDI, (2) zoom+detector bbox, (3) mask overlay,
    (4) final model input. Returns stats dict (also merged into qa_stats.json).
    Never touches caries coordinates; never relabels."""
    require_cv2()
    pano = cv2.imread(str(IMG_ROOT / f"case {case_id}" / f"case_{case_id}.png"))
    if pano is None:
        raise ValueError(f"Cannot read pano for case {case_id}")
    H, W = pano.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in rec["bbox"])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    crop = pano[y1:y2, x1:x2]
    gcrop = gray[y1:y2, x1:x2]
    mask = (cv2.imread(str(CACHE_DIR / rec["mask_path"]), cv2.IMREAD_GRAYSCALE) > 127)
    if mask.shape != gcrop.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (gcrop.shape[1], gcrop.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0

    mh, mw = mask.shape
    mask_px = int(mask.sum())
    area = max(1, (x2 - x1) * (y2 - y1))
    stats = {
        "case_id": case_id, "fdi": fdi, "quadrant": get_quadrant(fdi),
        "label": [SURFACES[j] for j in range(3) if rec["label"][j]],
        "crop_xyxy": [x1, y1, x2, y2], "crop_wh": [x2 - x1, y2 - y1],
        "mask_pixels": mask_px, "mask_per_crop_area": round(mask_px / area, 4),
        "border_touch": {"left": bool(mask[:, 0].any()),
                         "right": bool(mask[:, -1].any()),
                         "top": bool(mask[0, :].any()),
                         "bottom": bool(mask[-1, :].any())},
        "crop_at_pano_edge": bool(x1 == 0 or y1 == 0 or x2 >= W or y2 >= H),
        "conf": rec.get("conf", 0.0),
    }

    def _thumb(img, h=420):
        s = h / max(1, img.shape[0])
        return cv2.resize(img, (max(1, int(img.shape[1] * s)), h),
                          interpolation=cv2.INTER_AREA)

    p1 = _thumb(pano)
    s = p1.shape[0] / H
    cv2.rectangle(p1, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
                  (0, 255, 0), 2)
    cv2.putText(p1, f"FDI {fdi}", (int(x1 * s), max(20, int(y1 * s) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    p2 = crop.copy()
    cv2.rectangle(p2, (CROP_PAD, CROP_PAD),
                  (max(CROP_PAD + 1, p2.shape[1] - CROP_PAD),
                   max(CROP_PAD + 1, p2.shape[0] - CROP_PAD)),
                  (0, 255, 255), 2)
    p2 = _thumb(p2)
    ov = crop.copy().astype(np.float32)
    mm = cv2.resize(mask.astype(np.float32), (crop.shape[1], crop.shape[0]),
                    interpolation=cv2.INTER_NEAREST) > 0.5
    ov[mm] = ov[mm] * 0.5 + np.array([0, 255, 0], np.float32) * 0.5
    cnts, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
    p3 = _thumb(ov.astype(np.uint8))
    inp = preprocess_tooth(gcrop, mask, fdi, pre)
    d = inp - inp.min()
    p4 = _thumb(((d / (d.max() + 1e-6)) * 255).astype(np.uint8))
    if p4.ndim == 2:
        p4 = cv2.cvtColor(p4, cv2.COLOR_GRAY2BGR)

    panels = []
    for img, name in ((p1, "1 pano+FDI"), (p2, "2 zoom+bbox"),
                      (p3, "3 mask overlay"), (p4, "4 model input")):
        head = np.zeros((34, img.shape[1], 3), np.uint8)
        cv2.putText(head, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        panels.append(np.concatenate([head, bgr], axis=0))
    canvas = np.concatenate(panels, axis=1)
    out = outdir / f"qa_case{case_id}_fdi{fdi}.png"
    cv2.imwrite(str(out), canvas)
    print(f"[qa] {out} mask/bbox={stats['mask_per_crop_area']} "
          f"border={stats['border_touch']} pano_edge={stats['crop_at_pano_edge']}")
    return stats


def run_qa(args) -> None:
    """4-panel QA figures + mask stats for listed CASE:FDI teeth (plan 2.2)."""
    require_cv2()
    outdir = HERE / "outputs" / "qa"
    outdir.mkdir(parents=True, exist_ok=True)
    pre = ablation_to_pre(args.pre, img_size=args.img_size)
    _, samples = load_manifest()
    by_key = {(s["case_id"], s["fdi"]): s for s in samples}
    stats_path = outdir / "qa_stats.json"
    all_stats = (json.loads(stats_path.read_text(encoding="utf-8"))
                 if stats_path.is_file() else {})
    for tok in args.tooth:
        m = re.match(r"\s*(\d+)\s*:\s*(\d+)\s*$", tok)
        if not m:
            raise SystemExit(f"[qa] bad --tooth (want CASE:FDI): {tok}")
        case_id, fdi = int(m.group(1)), int(m.group(2))
        rec = by_key.get((case_id, fdi))
        if rec is None:
            print(f"[qa] WARNING: (case {case_id}, fdi {fdi}) not in manifest; skip")
            continue
        all_stats[f"{case_id}:{fdi}"] = _qa_single(case_id, fdi, rec, pre, outdir)
    stats_path.write_text(json.dumps(all_stats, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Orientation unit tests (plan section 2.4). No model, no labels-as-input.
# ----------------------------------------------------------------------------

def run_orientation_tests(args) -> int:
    """T1 FDI map; T2 flip-invariance on real masks; T3 convention proof images;
    T4 edge cases (steep-tilt clamp, narrow/tiny masks, overlapping molar)."""
    require_cv2()
    outdir = HERE / "outputs" / "orientation"
    outdir.mkdir(parents=True, exist_ok=True)
    fails: list[str] = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            fails.append(name)

    print("[T1] FDI mapping over all 32 teeth")
    bad = [fdi for q in (1, 2, 3, 4) for pos in range(1, 9)
           for fdi in [q * 10 + pos]
           if not (get_quadrant(fdi) == q and is_upper_jaw(fdi) == (q in (1, 2))
                   and canonical_flip_needed(fdi) == (q in (1, 4)))]
    check("32 FDIs map correctly", not bad, f"bad={bad}")

    print("[T2] flip-invariance: canonical(mirror(x)) == canonical(x)")
    _, samples = load_manifest()
    per_q: dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for s in samples:
        q = get_quadrant(s["fdi"])
        if len(per_q[q]) < 2 and not any(s["label"]):
            per_q[q].append(s)
        if all(len(v) >= 2 for v in per_q.values()):
            break
    pre = ablation_to_pre("clahe_pca")
    diffs = []
    for q, ss in per_q.items():
        for s in ss:
            gray = cv2.imread(str(CACHE_DIR / s["crop_path"]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(CACHE_DIR / s["mask_path"]), cv2.IMREAD_GRAYSCALE) > 127
            if gray is None or mask.sum() < 2000:
                continue
            a = preprocess_tooth(gray, mask, s["fdi"], pre)
            b = preprocess_tooth(np.ascontiguousarray(np.fliplr(gray)),
                                 np.ascontiguousarray(np.fliplr(mask)),
                                 s["fdi"], pre)
            d = float(np.abs(a - b).mean())
            diffs.append(d)
            check(f"Q{q} fdi={s['fdi']} invariance", d < 0.40, f"meandiff={d:.3f}")
    check("invariance measured", bool(diffs), f"n={len(diffs)}")

    print("[T3] convention proof images (left half reads Mesial side)")
    shown = 0
    for q in (1, 2, 3, 4):
        cand = next((s for s in samples
                     if get_quadrant(s["fdi"]) == q and not any(s["label"])), None)
        if cand is None:
            continue
        gray = cv2.imread(str(CACHE_DIR / cand["crop_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(CACHE_DIR / cand["mask_path"]), cv2.IMREAD_GRAYSCALE) > 127
        if gray is None or mask.sum() < 2000:
            continue
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(bgr, cnts, -1, (0, 255, 0), 2)
        after = preprocess_tooth(gray, mask, cand["fdi"], pre)
        d = after - after.min()
        au = ((d / (d.max() + 1e-6)) * 255).astype(np.uint8)
        au = cv2.cvtColor(cv2.resize(au, (224, 224)), cv2.COLOR_GRAY2BGR)
        cv2.putText(au, "M", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)
        cv2.putText(au, "D", (182, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 3)
        left = cv2.resize(bgr, (224, 224))
        cv2.imwrite(str(outdir / f"proof_Q{q}_fdi{cand['fdi']}.png"),
                    np.concatenate([left, au], axis=1))
        shown += 1
    print(f"  proof images: {shown} -> {outdir}")

    print("[T4] edge cases (no crash + sane angles)")
    rect = np.zeros((200, 200), np.uint8)
    cv2.ellipse(rect, (100, 100), (70, 30), 65, 0, 360, 255, -1)  # long axis 25 deg off vertical
    ys, xs = np.where(rect > 0)
    _, ang = pca_angle(np.column_stack([xs, ys]).astype(float), 14)
    check("tilted mask corrected (~25 deg)", abs(abs(math.degrees(ang)) - 25.0) < 4.0,
          f"{math.degrees(ang):.1f} deg")
    steep = np.zeros((200, 200), np.uint8)
    cv2.ellipse(steep, (100, 100), (70, 30), 30, 0, 360, 255, -1)  # 60 deg off vertical
    ys2, xs2 = np.where(steep > 0)
    _, ang2 = pca_angle(np.column_stack([xs2, ys2]).astype(float), 14)
    check("extreme tilt clamped to 0", ang2 == 0.0, f"{math.degrees(ang2):.1f} deg")
    strip = np.zeros((160, 40), np.uint8)
    strip[10:150, 8:32] = 255
    out = preprocess_tooth(np.full((160, 40), 128, np.uint8), strip > 0, 25,
                           ablation_to_pre("clahe_pca"))
    check("narrow mask survives", out.shape == (224, 224) and bool(np.isfinite(out).all()))
    tiny = np.zeros((50, 50), np.uint8)
    tiny[24:26, 24:26] = 255
    out2 = preprocess_tooth(np.full((50, 50), 128, np.uint8), tiny > 0, 36,
                            ablation_to_pre("clahe_pca"))
    check("tiny mask survives", out2.shape == (224, 224) and bool(np.isfinite(out2).all()))

    print("[T5] rotation direction (regression: warpAffine sign)")
    dot = np.zeros((100, 100), np.float32)
    dot[20, 70] = 255.0
    turned = rotate_crop(dot, (50.0, 50.0), 30.0)
    ys5, xs5 = np.where(turned > 1)
    # R(+30deg) of (70,20) about (50,50) == (82.3, 34.0)
    ok5 = (len(xs5) > 0 and abs(float(xs5.mean()) - 82.3) < 3.0
           and abs(float(ys5.mean()) - 34.0) < 3.0)
    check("rotate_crop moves content by +angle", ok5,
          f"landed=({float(xs5.mean()):.1f},{float(ys5.mean()):.1f})" if len(xs5) else "empty")

    print(f"[orientation] {'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
    return 1 if fails else 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Image-domain caries surface pipeline (tooth segments + FDI in, "
                    "caries coords never in).")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="detect teeth, cache masked crops + labels")
    b.add_argument("--seg-source", choices=["week2json", "live"], default="week2json")
    b.add_argument("--label-source", choices=["summary", "xml"], default="summary")
    b.add_argument("--max-cases", type=int, default=None)

    t = sub.add_parser("train", help="stage 2: report with fixed epoch budget")
    t.add_argument("--ablation", choices=["off", "clahe", "pca", "clahe_pca"], default="clahe_pca")
    t.add_argument("--backbone", choices=["resnet50", "efficientnet_b0", "densenet121"],
                   default="resnet50")
    t.add_argument("--loss", choices=["bce", "focal"], default="bce")
    t.add_argument("--use-metadata", action="store_true")
    t.add_argument("--epochs", type=int, default=None,
                   help="fixed budget; omit to reuse find-epoch best_epoch")
    t.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience (default None = fixed budget)")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cv", choices=["single", "groupkfold", "stratified-groupkfold",
                               "repeated-stratified-groupkfold"],
                   default="single")
    t.add_argument("--freeze-backbone", action="store_true",
                   help="linear probing: train classifier head only")
    t.add_argument("--partial-blocks", type=int, default=0,
                   help="Experiment B: unfreeze last-N backbone blocks + head")
    t.add_argument("--backbone-lr", type=float, default=None,
                   help="separate LR for backbone (default: same as --lr)")
    t.add_argument("--exclude", type=str, default="",
                   help="'1-50,200' or '@/path.json' (peeked_cases) to drop cases")
    t.add_argument("--img-size", type=int, default=224)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--input-norm", choices=["zscore", "imagenet"], default="zscore")
    t.add_argument("--aug-plus", action="store_true")
    t.add_argument("--recall-target", type=float, default=0.80,
                   help="tune-set Recall target for thresholds (1.4: 0.85 = margin)")
    t.add_argument("--repeats", type=int, default=3,
                   help="repeats for repeated-stratified-groupkfold")
    t.add_argument("--save-oof", action="store_true",
                   help="save pooled out-of-fold predictions + metrics (1.3)")
    t.add_argument("--resume", action="store_true",
                   help="A4: reuse finished folds (checkpoint + metrics "
                        "present) and continue a partially trained fold from "
                        "*_last.pt; refuses artifact/config mismatches")

    fe = sub.add_parser("find-epoch",
                        help="stage 1: pick epoch budget on tune-half only")
    fe.add_argument("--ablation", choices=["off", "clahe", "pca", "clahe_pca"], default="clahe_pca")
    fe.add_argument("--backbone", choices=["resnet50", "efficientnet_b0", "densenet121"],
                    default="efficientnet_b0")
    fe.add_argument("--loss", choices=["bce", "focal"], default="focal")
    fe.add_argument("--use-metadata", action="store_true")
    fe.add_argument("--max-epochs", type=int, default=50)
    fe.add_argument("--patience", type=int, default=8)
    fe.add_argument("--batch-size", type=int, default=32)
    fe.add_argument("--lr", type=float, default=3e-4)
    fe.add_argument("--seed", type=int, default=42)
    fe.add_argument("--freeze-backbone", action="store_true",
                    help="linear probing: train classifier head only")
    fe.add_argument("--tag", type=str, default="",
                    help="suffix for outputs, e.g. _frz (keeps runs separate)")
    fe.add_argument("--partial-blocks", type=int, default=0,
                    help="Experiment B: unfreeze last-N backbone blocks + head")
    fe.add_argument("--backbone-lr", type=float, default=None,
                    help="separate LR for backbone (default: same as --lr)")
    fe.add_argument("--exclude", type=str, default="",
                    help="'1-50,200' or '@/path.json' (peeked_cases) to drop cases")
    fe.add_argument("--img-size", type=int, default=224)
    fe.add_argument("--num-workers", type=int, default=4)
    fe.add_argument("--input-norm", choices=["zscore", "imagenet"], default="zscore")
    fe.add_argument("--aug-plus", action="store_true")
    fe.add_argument("--recall-target", type=float, default=0.80,
                    help="tune-set Recall target for thresholds (1.4: 0.85 = margin)")

    a = sub.add_parser("ablate", help="same split/seed across off/clahe/clahe_pca")
    a.add_argument("--backbone", choices=["resnet50", "efficientnet_b0", "densenet121"],
                   default="resnet50")
    a.add_argument("--loss", choices=["bce", "focal"], default="bce")
    a.add_argument("--use-metadata", action="store_true")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--partial-blocks", type=int, default=0)
    a.add_argument("--backbone-lr", type=float, default=None)
    a.add_argument("--freeze-backbone", action="store_true")
    a.add_argument("--exclude", type=str, default="")
    a.add_argument("--img-size", type=int, default=224)
    a.add_argument("--num-workers", type=int, default=4)
    a.add_argument("--input-norm", choices=["zscore", "imagenet"], default="zscore")
    a.add_argument("--aug-plus", action="store_true")
    a.add_argument("--recall-target", type=float, default=0.80,
                   help="tune-set Recall target for thresholds (1.4: 0.85 = margin)")

    bl = sub.add_parser("baseline", help="RF on handcrafted texture features")
    bl.add_argument("--ablation", choices=["off", "clahe", "pca", "clahe_pca"], default="clahe_pca")
    bl.add_argument("--seed", type=int, default=42)
    bl.add_argument("--exclude", type=str, default="")
    bl.add_argument("--img-size", type=int, default=224)
    bl.add_argument("--features", choices=["texture", "fdi"], default="texture",
                    help="texture = handcrafted pixels; fdi = tooth-identity shortcut probe (1.7)")
    bl.add_argument("--recall-target", type=float, default=0.80)

    q = sub.add_parser("qa", help="4-panel data-QA figure per tooth (plan 2.2)")
    q.add_argument("--tooth", action="append", default=[],
                   help="CASE:FDI, repeatable, e.g. --tooth 476:25 --tooth 239:46")
    q.add_argument("--pre", choices=["off", "clahe", "pca", "clahe_pca"], default="clahe_pca")
    q.add_argument("--img-size", type=int, default=224)

    ot = sub.add_parser("test-orientation", help="orientation unit tests (plan 2.4)")

    co = sub.add_parser("compare",
                        help="paired A-vs-B comparison, bootstrap CI of difference (1.2)")
    co.add_argument("--weights-a", required=True)
    co.add_argument("--weights-b", required=True)
    co.add_argument("--seed", type=int, default=42)
    co.add_argument("--n-boot", type=int, default=1000)
    co.add_argument("--num-workers", type=int, default=4)


    oor = sub.add_parser("oof-report",
                         help="E1: per-fold AP/prevalence/lift/AUROC + pooled decisions")
    oor.add_argument("--oof", required=True,
                     help="pooled OOF csv written by train --save-oof")
    oor.add_argument("--recall-target", type=float, default=0.80)
    oor.add_argument("--n-boot", type=int, default=1000)
    oor.add_argument("--seed", type=int, default=42)
    oor.add_argument("--out", default=None)

    cop = sub.add_parser("compare-oof",
                         help="E1b: paired delta-AP with corrected t + non-inferiority margin")
    cop.add_argument("--pair", action="append", required=True,
                     help="'A.csv,B.csv' from the SAME split and seed; repeat per seed")
    cop.add_argument("--margin", type=float, default=0.05,
                     help="non-inferiority margin, locked in the protocol first")
    cop.add_argument("--n-boot", type=int, default=500)
    cop.add_argument("--seed", type=int, default=42)
    cop.add_argument("--out", default=None)

    cb = sub.add_parser("calibrate",
                        help="E3: ensemble temperature + thresholds on the peeked set")
    cb.add_argument("--prefix", default="best_pca_s42_f",
                    help="fold checkpoint prefix: best_pca_s42_f -> best_pca_s42_f0.pt")
    cb.add_argument("--n-folds", type=int, default=5)
    cb.add_argument("--calib", required=True,
                    help="'@/path.json' peeked cases used as the ensemble calibration set")
    cb.add_argument("--ablation", choices=["off", "clahe", "pca", "clahe_pca"],
                    default="pca")
    cb.add_argument("--recall-target", type=float, default=0.85,
                    help="tune above the 0.80 bar so the delivered threshold still meets it")
    cb.add_argument("--limit", type=int, default=None,
                    help="use only the first N calibration teeth (smoke test)")
    cb.add_argument("--seed", type=int, default=42)
    cb.add_argument("--num-workers", type=int, default=4)
    cb.add_argument("--out", default=None)

    kp = sub.add_parser("kappa",
                        help="E4: inter-rater Cohen kappa per surface (label ceiling)")
    kp.add_argument("--pairs", required=True,
                    help="csv: case_id,tooth_fdi,surface,rater_a,rater_b")
    kp.add_argument("--n-boot", type=int, default=1000)
    kp.add_argument("--seed", type=int, default=42)
    kp.add_argument("--out", default=None)

    ss = sub.add_parser("sample-size",
                        help="E5: external-test size implied by measured prevalence")
    ss.add_argument("--oof-json", default=None,
                    help="oof_report json to read prevalence from")
    ss.add_argument("--prevalence", action="append", default=[],
                    help="Surface=value, repeatable (e.g. Occlusal=0.026)")
    ss.add_argument("--sensitivity", type=float, default=0.80)
    ss.add_argument("--half-width", type=float, default=0.10)
    ss.add_argument("--design-effect", type=float, default=1.3)
    ss.add_argument("--teeth-per-case", type=float, default=30.0)
    ss.add_argument("--confidence", type=float, default=0.95,
                    choices=[0.90, 0.95, 0.99])
    ss.add_argument("--out", default=None)

    pr = sub.add_parser("predict", help="pano image -> per-tooth surfaces")
    pr.add_argument("--image", required=True)
    pr.add_argument("--weights", required=True)
    pr.add_argument("--ablation", choices=["off", "clahe", "pca", "clahe_pca"], default="clahe_pca")
    pr.add_argument("--backbone", default="resnet50")
    pr.add_argument("--use-metadata", action="store_true")
    pr.add_argument("--out", default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if cv2 is None and args.cmd in ("build-dataset", "baseline", "predict"):
        raise ImportError("cv2 required for this command.")
    if args.cmd == "build-dataset":
        build_dataset(seg_source=args.seg_source, max_cases=args.max_cases,
                      label_source=args.label_source)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "find-epoch":
        run_find_epoch(args)
    elif args.cmd == "ablate":
        run_ablate(args)
    elif args.cmd == "baseline":
        run_baseline(args)
    elif args.cmd == "compare":
        run_compare(args)
    elif args.cmd == "oof-report":
        run_oof_report(args)
    elif args.cmd == "compare-oof":
        run_compare_oof(args)
    elif args.cmd == "calibrate":
        run_calibrate(args)
    elif args.cmd == "kappa":
        run_kappa(args)
    elif args.cmd == "sample-size":
        run_sample_size(args)
    elif args.cmd == "qa":
        run_qa(args)
    elif args.cmd == "test-orientation":
        raise SystemExit(run_orientation_tests(args))
    elif args.cmd == "predict":
        out = predict_image(args.image, args.weights, ablation=args.ablation,
                            backbone=args.backbone, use_metadata=args.use_metadata)
        s = json.dumps(out, indent=1, ensure_ascii=False)
        print(s)
        if args.out:
            Path(args.out).write_text(s, encoding="utf-8")
    else:  # pragma: no cover
        raise SystemExit(2)


if __name__ == "__main__":
    main()
