"""Stage 3b (reference only) -- CNN surface scorer / shadow runner (ml-service side).

STATUS: NOT USED BY SERVING.  The orchestrator no longer imports this module;
production surfaces come from ``app/pipeline/surface_heuristic.py``.  The CNN
remains a documented future option (Decision Log: keep ``cnn_pca42`` as an
option, not primary).  The tests under ``tests/test_surface_cnn_parity.py`` and
``tests/bench_surface_cnn.py`` still exercise it for reference.

WHY THIS FILE IS A VENDORED COPY
--------------------------------
Training and serving must apply the *identical* per-tooth transform. The original
implementation lives in ``caries-detection-model/pipeline.py`` (functions below
carry their source line numbers). It cannot be imported here because that project
is not inside the ml-service image (and its module imports detectron2/sklearn),
so the functions are copied verbatim and guarded by
``tests/test_surface_cnn_parity.py``:

  * static check : function source text must equal the original (byte compare)
  * numeric check: preprocessed tensor + probabilities must equal a reference
                   produced by the ORIGINAL pipeline code (see
                   ``caries-detection-model/deploy/make_parity_reference.py``)

MODE (environment only -- this file never changes ml-service config defaults)
----------------------------------------------------------------------------
  SURFACE_CNN_MODE      off | shadow | cnn      (default: off)
  SURFACE_CNN_DIR       folder with best_pca_s42_f*.pt + calibration (default /weights/cnn_pca42)
  SURFACE_CNN_CALIB     path to calibration json (default <dir>/calibration_pca.json)
  SURFACE_CNN_DEVICE    cpu | cuda               (default ML_DEVICE or cpu)
  SURFACE_CNN_THREADS   torch threads            (default 4)
  SURFACE_CNN_MAX_TEETH skip the stage when more teeth than this (0 = no limit)

``shadow`` computes CNN probabilities and returns them as a SEPARATE result block;
it never edits or removes the Random-Forest findings.  Any failure is caught and
reported inside that block, so the RF result cannot be affected.

Deliberate deviations from the training code (documented, not silent)
--------------------------------------------------------------------
1. ``torchvision`` backbone weights are constructed with ``weights=None``.
   The checkpoint's ``state_dict`` overwrites every weight, and the container may
   be offline (``build_cnn`` in pipeline.py downloads ImageNet weights).
2. Crops are built from the pano mask polygon + ``CROP_PAD`` (same rule as
   ``pipeline.ToothDetector.detect``); the Detectron2 crop refiner is NOT applied.
   Training used the refiner -- this gap is measured by the step-4 end-to-end run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SURFACES = ["Distal", "Mesial", "Occlusal"]
CROP_PAD = 20                  # pipeline.py L119
CLAHE_CLIP = 2.0               # pipeline.py L120
CLAHE_TILE = (8, 8)            # pipeline.py L121
MAX_TILT_DEG = 45.0            # pipeline.py L122

_IMAGENET_MU = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_SD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

MODE = os.environ.get("SURFACE_CNN_MODE", "off").strip().lower()
DEFAULT_DIR = os.environ.get("SURFACE_CNN_DIR", "/weights/cnn_pca42")


# ---------------------------------------------------------------------------
# Vendored preprocessing -- byte-for-byte logic of pipeline.py (see module doc)
# ---------------------------------------------------------------------------

def _fdi_to_int(fdi) -> int | None:                      # pipeline.py L173
    import re

    try:
        digits = re.sub(r"\D", "", str(fdi))
        if len(digits) < 2:
            return None
        return int(digits[:2])
    except (ValueError, TypeError):
        return None


def _is_upper_jaw(fdi) -> bool:                          # pipeline.py L183
    v = _fdi_to_int(fdi)
    return v is not None and int(str(v)[0]) in (1, 2)


def _get_quadrant(fdi) -> int:                           # pipeline.py L188
    v = _fdi_to_int(fdi)
    return int(str(v)[0]) if v is not None else 4


def _pca_angle(mask_pts: np.ndarray, fdi) -> tuple[np.ndarray, float]:
    """pipeline.py L322 -- long axis is ALWAYS the primary eigenvector."""
    import cv2
    import math

    pts = np.asarray(mask_pts, dtype=np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    centered = pts - center
    _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
    vert = eigvecs[0].astype(np.float64).copy()
    if _is_upper_jaw(fdi):
        if vert[1] < 0:
            vert = -vert
    else:
        if vert[1] > 0:
            vert = -vert
    ang = math.atan2(vert[1], vert[0])
    target = math.pi / 2 if _is_upper_jaw(fdi) else -math.pi / 2
    rot = target - ang
    while rot > math.pi:
        rot -= 2 * math.pi
    while rot < -math.pi:
        rot += 2 * math.pi
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
    return center, float(rot)


def _canonical_flip_needed(fdi) -> bool:                 # pipeline.py L353
    """True -> mirror horizontally so MESIAL is always on the LEFT."""
    return _get_quadrant(fdi) in (1, 4)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:        # pipeline.py L367
    import cv2

    g = np.ascontiguousarray(gray.astype(np.uint8))
    return cv2.createCLAHE(clipLimit=float(CLAHE_CLIP),
                           tileGridSize=tuple(CLAHE_TILE)).apply(g)


def _zscore_foreground(gray_f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """pipeline.py L374 -- mean/std over foreground only, background stays 0."""
    fg = mask > 0
    out = np.zeros_like(gray_f, dtype=np.float32)
    if fg.sum() < 8:
        return out
    vals = gray_f[fg]
    mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
    out[fg] = (vals - mu) / sd
    return out


def _rotate_crop(img: np.ndarray, center_xy: tuple[float, float],
                 angle_deg: float, interp: int | None = None) -> np.ndarray:
    """pipeline.py L385 -- negates the angle on purpose (bugfix 2026-10-13)."""
    import cv2

    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(tuple(map(float, center_xy)), -float(angle_deg), 1.0)
    return cv2.warpAffine(img, m, (w, h),
                          flags=interp if interp is not None else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:   # pipeline.py L403
    import cv2

    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_LINEAR if img.dtype != bool else cv2.INTER_NEAREST
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    top, bottom = (size - nh) // 2, size - nh - (size - nh) // 2
    left, right = (size - nw) // 2, size - nw - (size - nw) // 2
    return cv2.copyMakeBorder(small, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=0)


def _preprocess_tooth(gray_crop: np.ndarray, mask_crop: np.ndarray, fdi,
                      use_clahe: bool, use_pca: bool, img_size: int,
                      input_norm: str = "zscore") -> np.ndarray:
    """pipeline.py L426 preprocess_tooth (same order, same constants).

    NOTE: when ``input_norm == "imagenet"`` the original returns a masked
    ``[0, 1]`` image (pipeline.py L445-446) and the *caller* standardises with
    ImageNet mean/std -- reproduced identically in ``CnnBundle.preprocess_one``.
    """
    import cv2
    import math

    gray = np.asarray(gray_crop)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    mask = (np.asarray(mask_crop) > 0).astype(np.uint8)
    if mask.shape != gray.shape[:2]:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    # CLAHE FIRST on the unmasked crop (bugfix: background zeros would dominate
    # tile histograms and compress foreground contrast).
    work = _apply_clahe(gray) if use_clahe else gray
    if input_norm == "imagenet":                        # pipeline.py L445-446
        norm = (work.astype(np.float32) / 255.0) * mask.astype(np.float32)
    else:
        norm = _zscore_foreground(work.astype(np.float32), mask)
    if use_pca:
        ys, xs = np.where(mask > 0)
        if len(xs) >= 10:
            mpts = np.column_stack([xs, ys]).astype(np.float64)
            center, rot = _pca_angle(mpts, fdi)
            norm = _rotate_crop(norm, (float(center[0]), float(center[1])),
                                math.degrees(rot))
            mask = _rotate_crop(mask.astype(np.float32),
                                (float(center[0]), float(center[1])),
                                math.degrees(rot), cv2.INTER_NEAREST) > 0.5
            norm = norm * mask.astype(np.float32)
            if _canonical_flip_needed(fdi):
                norm = np.ascontiguousarray(np.fliplr(norm))
    return _resize_pad(norm, img_size).astype(np.float32)


def _ablation_to_pre(ablation: str, img_size: int) -> tuple[bool, bool]:
    """pipeline.ablation_to_pre() -> (use_clahe, use_pca)."""
    a = str(ablation).lower()
    return ("clahe" in a, "pca" in a and a != "clahe")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_backbone(backbone: str):
    """pipeline.build_cnn() with weights=None (offline containers, see docstring)."""
    import torch.nn as nn
    from torchvision.models import (densenet121, efficientnet_b0, resnet50)

    bb = str(backbone).lower()
    if bb == "resnet50":
        net = resnet50(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif bb == "efficientnet_b0":
        net = efficientnet_b0(weights=None)
        feat_dim = 1280
        net.classifier = nn.Identity()
    elif bb == "densenet121":
        net = densenet121(weights=None)
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"unsupported backbone in checkpoint: {backbone}")

    use_metadata = False

    class ToothCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = net
            self.use_metadata = use_metadata
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 3))

        def forward(self, x, meta=None, fdi=None):
            f = self.backbone(x)
            if f.dim() > 2:
                f = f.flatten(1)
            return self.head(f)

    return ToothCNN()


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CnnBundle:
    """Loads the 5 fold checkpoints + calibration; identical maths to
    ``caries-detection-model/deploy_effnet_service.py`` (sigmoid average over
    folds, then temperature scaling on logit(mean))."""

    def __init__(self, checkpoints: list[Path], device: str = "cpu",
                 calibration_path: str | None = None, threads: int = 4):
        import torch

        try:
            torch.set_num_threads(int(threads))
        except Exception:  # pragma: no cover
            pass

        self.torch = torch
        self.device = device
        self.errors: list[str] = []
        self.models: list[Any] = []
        self.checkpoints: list[dict] = []
        self.calibration: dict | None = None
        self.calibration_path = calibration_path
        self.cfg: dict | None = None

        if calibration_path and Path(calibration_path).is_file():
            try:
                raw = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
                if "temperature" in raw and "threshold" in raw:
                    self.calibration = raw
                else:
                    self.errors.append(f"calibration missing keys: {calibration_path}")
            except Exception as exc:
                self.errors.append(f"calibration unreadable: {exc}")
        elif calibration_path:
            self.errors.append(f"calibration not found: {calibration_path}")

        ref: dict | None = None
        for cp in checkpoints:
            if not cp.is_file():
                self.errors.append(f"missing checkpoint: {cp}")
                continue
            try:
                ck = torch.load(str(cp), map_location="cpu", weights_only=False)
            except Exception as exc:
                self.errors.append(f"cannot load {cp.name}: {exc}")
                continue
            missing = ({"state_dict", "backbone", "use_metadata", "ablation",
                        "img_size", "input_norm", "surfaces"} - set(ck))
            if missing:
                self.errors.append(f"{cp.name}: missing keys {sorted(missing)}")
                continue
            if bool(ck["use_metadata"]):
                self.errors.append(f"{cp.name}: use_metadata=True not supported here")
                continue
            if list(ck["surfaces"]) != SURFACES:
                self.errors.append(
                    f"{cp.name}: surface order {ck['surfaces']} != {SURFACES}")
                continue
            spec = {
                "ablation": str(ck["ablation"]),
                "backbone": str(ck["backbone"]),
                "img_size": int(ck["img_size"]),
                "input_norm": str(ck["input_norm"]),
            }
            if "seed" in ck:
                # pipeline.py does not store the seed today; if a future
                # checkpoint records it, require every fold to agree on it.
                spec["seed"] = int(ck["seed"])
            if ref is None:
                ref = spec
            elif any(k not in ref or spec[k] != ref[k] for k in spec):
                self.errors.append(
                    f"{cp.name}: config differs from the first checkpoint "
                    f"({ref} vs {spec})")
                continue
            try:
                model = _build_backbone(spec["backbone"]).to(device)
                model.load_state_dict({k: v.to(device)
                                       for k, v in ck["state_dict"].items()})
                model.eval()
            except Exception as exc:
                self.errors.append(f"{cp.name}: model load failed: {exc}")
                continue
            self.models.append(model)
            self.checkpoints.append({"file": cp.name, "sha256": _sha256_file(cp),
                                     "fold": _fold_from_name(cp.name)})
        self.cfg = ref

    @property
    def ready(self) -> bool:
        return bool(self.models) and not self.errors

    def preprocess_one(self, gray: np.ndarray, mask: np.ndarray, fdi: int) -> np.ndarray:
        use_clahe, use_pca = _ablation_to_pre(self.cfg["ablation"], self.cfg["img_size"])
        img = _preprocess_tooth(gray, mask, fdi, use_clahe, use_pca,
                                self.cfg["img_size"], self.cfg["input_norm"])
        x = np.stack([img, img, img], axis=0).astype(np.float32)
        if self.cfg["input_norm"] == "imagenet":
            x = (x - _IMAGENET_MU) / _IMAGENET_SD
        return x

    def predict(self, items: list[tuple[np.ndarray, np.ndarray, int]]) -> list[dict]:
        torch = self.torch
        xs = np.stack([self.preprocess_one(g, m, f) for g, m, f in items])
        with torch.no_grad():
            xb = torch.from_numpy(xs).to(self.device)
            probs = None
            for model in self.models:
                p = torch.sigmoid(model(xb)).cpu().numpy()
                probs = p if probs is None else probs + p
            probs = probs / float(len(self.models))
            if self.calibration is not None:
                temp = np.asarray([float(self.calibration["temperature"][s])
                                   for s in SURFACES], dtype=np.float32)
                eps = 1e-6
                pc = np.clip(probs, eps, 1 - eps)
                z = np.log(pc / (1 - pc))
                probs = 1.0 / (1.0 + np.exp(-z / temp))
        out = []
        for (_g, _m, fdi), p in zip(items, probs):
            rec: dict[str, Any] = {"fdi": int(fdi),
                                   "probabilities": {s: float(p[j])
                                                     for j, s in enumerate(SURFACES)}}
            if self.calibration is not None:
                rec["thresholds"] = {s: float(self.calibration["threshold"][s])
                                     for s in SURFACES}
                rec["positive"] = {s: bool(p[j] >= self.calibration["threshold"][s])
                                   for j, s in enumerate(SURFACES)}
            out.append(rec)
        return out

    def info(self) -> dict:
        warn = ["STAGING ONLY: internal cross-validation data, no external validation",
                "shadow mode: Random-Forest findings are unchanged"]
        if self.calibration is None:
            warn.append("probabilities only: no deployment threshold loaded")
        return {"ready": self.ready, "device": self.device,
                "n_folds": len(self.models), "errors": self.errors,
                "checkpoints": self.checkpoints, "config": self.cfg,
                "surfaces": SURFACES, "calibration": self.calibration_path,
                "warnings": warn}


def _fold_from_name(name: str) -> int | None:
    """``best_pca_s42_f3.pt`` -> 3 (mirrors deploy_effnet_service._fold_from_name)."""
    import re

    m = re.search(r"_f(\d+)\.pt$", name)
    return int(m.group(1)) if m else None


_BUNDLE: CnnBundle | None = None
_BUNDLE_TRIED = False


def _env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def get_bundle() -> CnnBundle | None:
    """Load once per process; never raises."""
    global _BUNDLE, _BUNDLE_TRIED

    if _BUNDLE_TRIED:
        return _BUNDLE
    _BUNDLE_TRIED = True
    if MODE not in ("shadow", "cnn"):
        return None
    try:
        directory = Path(DEFAULT_DIR)
        calib = _env_flag("SURFACE_CNN_CALIB") or str(directory / "calibration_pca.json")
        device = _env_flag("SURFACE_CNN_DEVICE") or os.environ.get("ML_DEVICE", "cpu")
        threads = int(_env_flag("SURFACE_CNN_THREADS", "4") or 4)
        cps = sorted(directory.glob("best_pca_s42_f[0-9].pt"))
        bundle = CnnBundle(cps, device=device, calibration_path=calib, threads=threads)
        if not cps:
            # No loose fallback on purpose: a wrong seed/ablation would silently
            # serve different weights.
            bundle.errors.append(
                f"no best_pca_s42_f[0-9].pt in {directory} "
                f"(found: {sorted(p.name for p in directory.glob('*.pt'))})")
        log.info("surface CNN bundle: %s", json.dumps(bundle.info())[:2000])
        if not bundle.ready:
            log.warning("surface CNN not ready: %s", "; ".join(bundle.errors))
        _BUNDLE = bundle
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("surface CNN load failed: %r", exc)
        _BUNDLE = None
    return _BUNDLE


# ---------------------------------------------------------------------------
# Crop construction from Stage-1 detections (training rule: mask pixels + PAD)
# ---------------------------------------------------------------------------

def _polygon_crop(image_bgr: np.ndarray, polygon: list[list[float]],
                  pad: int = CROP_PAD) -> tuple[np.ndarray, np.ndarray] | None:
    """Rasterise the polygon, take its bounding box +/- pad -> (gray_crop, mask_crop)."""
    import cv2

    h, w = image_bgr.shape[:2]
    if not polygon or len(polygon) < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad)
    y2 = min(h, int(ys.max()) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray, mask[y1:y2, x1:x2].astype(bool)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def build_items(image_bgr: np.ndarray, detections: list) -> tuple[list, list[str]]:
    """Detections -> model items, de-duplicating FDI exactly like training
    (``pipeline.ToothDetector.detect`` keeps the highest-confidence box when
    IoU > 0.5)."""
    notes: list[str] = []
    kept: list[tuple[float, Any]] = []
    for det in sorted(detections, key=lambda d: -float(getattr(d, "pano_confidence", 0.0))):
        poly = list(getattr(det, "tooth_polygon", []) or [])
        if len(poly) < 3:
            continue
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        box = (min(xs), min(ys), max(xs), max(ys))
        if any(_iou(box, kb) > 0.5 for _c, kb in kept):
            notes.append(f"FDI {getattr(det, 'fdi', '?')}: dropped duplicate box (IoU>0.5)")
            continue
        kept.append((float(getattr(det, "pano_confidence", 0.0)), box))

    order: list[tuple[Any, Any]] = []
    for det in sorted(detections, key=lambda d: -float(getattr(d, "pano_confidence", 0.0))):
        poly = list(getattr(det, "tooth_polygon", []) or [])
        if len(poly) < 3:
            continue
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        box = (min(xs), min(ys), max(xs), max(ys))
        matches = [b for b in kept if b[1] == box]
        if not matches:
            continue
        order.append((det, box))
        kept.remove(matches[0])

    items = []
    for det, box in order:
        out = _polygon_crop(image_bgr, list(getattr(det, "tooth_polygon", []) or []))
        if out is None:
            notes.append(f"FDI {getattr(det, 'fdi', '?')}: empty crop")
            continue
        gray, m = out
        items.append((gray, m, int(det.fdi)))
    return items, notes


# ---------------------------------------------------------------------------
# Public entry point (called from the orchestrator; never raises)
# ---------------------------------------------------------------------------

def run_shadow(image_bgr: np.ndarray, detections: list) -> dict | None:
    """Compute CNN probabilities for the given teeth.

    Returns ``None`` when the mode is off.  Every exception is captured into the
    returned block so the RF path is untouched.
    """
    if MODE not in ("shadow", "cnn"):
        return None
    started = time.perf_counter()
    try:
        bundle = get_bundle()
        if bundle is None or not bundle.ready:
            return {"mode": MODE, "ready": False,
                    "errors": [] if bundle is None else bundle.errors,
                    "stage_ms": round((time.perf_counter() - started) * 1000, 1)}
        items, notes = build_items(image_bgr, detections)
        limit = int(_env_flag("SURFACE_CNN_MAX_TEETH", "0") or 0)
        skipped = 0
        if limit and len(items) > limit:
            skipped = len(items) - limit
            items = items[:limit]
            notes.append(f"MAX_TEETH={limit}: skipped {skipped} teeth")
        if not items:
            return {"mode": MODE, "ready": True, "teeth": [], "notes": notes,
                    "info": bundle.info(),
                    "stage_ms": round((time.perf_counter() - started) * 1000, 1)}
        t0 = time.perf_counter()
        preds = bundle.predict(items)
        predict_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "mode": MODE,
            "ready": True,
            "n_teeth": len(preds),
            "teeth": preds,
            "notes": notes,
            "info": bundle.info(),
            "timings_ms": {"predict": predict_ms,
                           "predict_per_tooth": round(predict_ms / max(1, len(preds)), 1),
                           "stage": round((time.perf_counter() - started) * 1000, 1)},
        }
    except Exception as exc:  # pragma: no cover - defensive: shadow must not break jobs
        log.warning("surface CNN shadow failed: %r", exc, exc_info=True)
        return {"mode": MODE, "ready": False, "error": repr(exc),
                "stage_ms": round((time.perf_counter() - started) * 1000, 1)}


# ---------------------------------------------------------------------------
# Primary promotion (SURFACE_CNN_MODE=cnn -- CNN replaces RF as the finding)
# ---------------------------------------------------------------------------

def primary_findings(
    rf_findings: dict[int, list[dict]],
    shadow_block: dict | None,
    detections: list,
) -> tuple[dict[int, list[dict]], str]:
    """Promote calibrated CNN results to the primary surface findings.

    Only meaningful when ``MODE == "cnn"``.  Returns ``(findings_by_fdi, source)``
    where ``source`` is:

      ``"cnn"``  CNN findings are primary (bundle ran, thresholds loaded)
      ``"rf"``   RF findings unchanged (mode off/shadow, or the CNN could not
                 run / has no deployment thresholds) -- the RF is deliberately
                 kept as the fallback so a CNN failure cannot blank a job.

    Contract notes (deliberate, not accidents):
      * only teeth that already carry a caries detection are scored, matching
        the lesion-driven contract of ``run_stage3``;
      * a tooth gets one finding per surface whose probability is at or above
        its deployment threshold (0..3 findings) -- exactly the operating point
        defined by ``calibration_pca.json``;
      * a tooth whose surfaces are all below threshold gets no finding (the
        lesion detection stays, the surface call is "below threshold").
    """
    if MODE != "cnn":
        return rf_findings, "rf"
    if not shadow_block or not shadow_block.get("ready"):
        log.warning("CNN primary requested but bundle not ready -- RF kept")
        return rf_findings, "rf"

    teeth = shadow_block.get("teeth") or []
    if not teeth or "positive" not in (teeth[0] or {}):
        log.warning("CNN primary requested but no predictions / no deployment "
                    "thresholds loaded -- RF kept")
        return rf_findings, "rf"

    caries_fdi: set[int] = set()
    for det in detections:
        pts = getattr(det, "caries_points", None)
        if pts is not None and len(pts) > 0:
            caries_fdi.add(int(det.fdi))

    surfaces_by_fdi: dict[int, dict[str, dict]] = {}
    for rec in teeth:
        fdi_raw = rec.get("fdi")
        if fdi_raw is None:
            continue
        fdi = int(fdi_raw)
        if fdi not in caries_fdi:
            continue
        positive = rec.get("positive") or {}
        probs = rec.get("probabilities") or {}
        for surface in SURFACES:
            if not positive.get(surface):
                continue
            name = surface.lower()
            probability = round(float(probs.get(surface, 0.0)), 4)
            previous = surfaces_by_fdi.setdefault(fdi, {}).get(name)
            if previous is None or probability > float(previous["probability"]):
                # same de-duplication rule as run_stage3 (keep max prob per surface)
                surfaces_by_fdi[fdi][name] = {
                    "name": name,
                    "label": "caries",
                    "probability": probability,
                    "method": "CNN",
                }

    findings: dict[int, list[dict]] = {
        fdi: list(surfaces.values())
        for fdi, surfaces in surfaces_by_fdi.items()
        if surfaces
    }

    log.info("Stage 3b CNN primary: %d teeth flagged (scored=%d, caries detections=%d)",
             len(findings), len(teeth), len(caries_fdi))
    return findings, "cnn"
