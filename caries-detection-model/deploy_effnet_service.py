"""Staging HTTP service for the Stage-2 per-tooth caries surface classifier.

Design rules
------------
* It imports the training module's own preprocessing and model factory
  (``pipeline.preprocess_tooth``, ``pipeline.build_cnn``,
  ``pipeline.ablation_to_pre``) so the serving path cannot drift from training.
* It never writes to ``outputs/`` and never modifies training artifacts.
* It returns probabilities. Threshold decisions are returned ONLY when a
  calibration JSON produced by ``pipeline.py calibrate`` is supplied; otherwise
  ``threshold``/``positive`` are null. This is deliberate: the protocol forbids
  quoting uncalibrated deployment thresholds.
* Multiple fold checkpoints are averaged in *probability* space (mean of
  sigmoid outputs), matching the ensemble definition in ``run_calibrate``.
* ``fold`` may be supplied per request to force one fold model; this is used by
  the OOF smoke test so no sample is scored by a model that trained on it.

Run (staging, CPU to avoid competing with the training queue)::

    $env:EFFNET_CHECKPOINTS="outputs/best_pca_s42_f0.pt;outputs/best_pca_s42_f1.pt"
    $env:EFFNET_DEVICE="cpu"
    python -m uvicorn deploy_effnet_service:app --host 127.0.0.1 --port 8011
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402  (import-safe: training main runs under __main__)

log = logging.getLogger("effnet-service")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

SURFACES = list(P.SURFACES)
_IMAGENET_MU = np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
_IMAGENET_SD = np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fold_from_name(name: str) -> int | None:
    m = re.search(r"_f(\d+)\.pt$", name)
    return int(m.group(1)) if m else None


def discover_checkpoints() -> list[Path]:
    spec = os.environ.get("EFFNET_CHECKPOINTS", "").strip()
    if spec:
        return [Path(p.strip()) for p in spec.split(";") if p.strip()]
    # Default: the locked reference arm, completed folds only. ``_last.pt``
    # does not match the pattern.
    return sorted(P.OUTPUTS_DIR.glob("best_pca_s42_f[0-9].pt"))


def load_calibration(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"calibration file not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    for key in ("temperature", "threshold"):
        if key not in obj:
            raise ValueError(f"calibration missing '{key}'")
        missing = [s for s in SURFACES if s not in obj[key]]
        if missing:
            raise ValueError(f"calibration missing surfaces in {key}: {missing}")
    return obj


class SurfaceEnsemble:
    """Loads 1..N fold checkpoints and scores preprocessed tooth crops."""

    def __init__(self, checkpoints: list[Path], device: str = "cpu",
                 calibration_path: str | None = None):
        import torch

        threads = int(os.environ.get("EFFNET_TORCH_THREADS", "4"))
        try:
            torch.set_num_threads(threads)
        except Exception:  # pragma: no cover - defensive only
            pass

        self.torch = torch
        self.device = device
        self.errors: list[str] = []
        self.models: list[Any] = []
        self.checkpoints: list[dict] = []
        self.calibration = load_calibration(calibration_path)
        self.calibration_path = calibration_path
        self.cfg: dict | None = None
        self.pre = None

        if not checkpoints:
            self.errors.append("no checkpoints found")
            return

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
            if list(ck["surfaces"]) != SURFACES:
                self.errors.append(
                    f"{cp.name}: surface order {ck['surfaces']} != {SURFACES}")
                continue
            spec = {
                "ablation": str(ck["ablation"]),
                "backbone": str(ck["backbone"]),
                "use_metadata": bool(ck["use_metadata"]),
                "img_size": int(ck["img_size"]),
                "input_norm": str(ck["input_norm"]),
                "aug_plus": bool(ck.get("aug_plus", False)),
                "thresholds": [float(v) for v in ck.get("thresholds", [])],
            }
            if ref is None:
                ref = spec
            else:
                keys = ("ablation", "backbone", "use_metadata", "img_size",
                        "input_norm", "aug_plus")
                if any(spec[k] != ref[k] for k in keys):
                    self.errors.append(
                        f"{cp.name}: preprocessing/model config differs from "
                        f"the first checkpoint")
                    continue
            try:
                model = P.build_cnn(spec["backbone"], spec["use_metadata"]).to(device)
                model.load_state_dict(
                    {k: v.to(device) for k, v in ck["state_dict"].items()})
                model.eval()
            except Exception as exc:
                self.errors.append(f"{cp.name}: model load failed: {exc}")
                continue
            self.models.append(model)
            self.checkpoints.append({
                "file": cp.name,
                "path": str(cp),
                "sha256": _sha256(cp),
                "fold": _fold_from_name(cp.name),
                "thresholds": spec["thresholds"],
            })

        self.cfg = ref
        if ref is not None:
            self.pre = P.ablation_to_pre(
                ref["ablation"], img_size=ref["img_size"],
                input_norm=ref["input_norm"], aug_plus=ref["aug_plus"])

    @property
    def ready(self) -> bool:
        return bool(self.models) and not self.errors

    def preprocess_one(self, gray: np.ndarray, mask: np.ndarray,
                       fdi: int) -> np.ndarray:
        """Exactly the training transform (``ToothCropSet.__getitem__``)."""
        img = P.preprocess_tooth(gray, mask, fdi, self.pre)
        x = np.stack([img, img, img], axis=0).astype(np.float32)
        if self.pre.input_norm == "imagenet":
            x = (x - _IMAGENET_MU) / _IMAGENET_SD
        return x

    def _models_for(self, fold: int | None):
        if fold is None:
            return self.models, self.checkpoints
        picked = [(m, c) for m, c in zip(self.models, self.checkpoints)
                  if c["fold"] == fold]
        if not picked:
            raise ValueError(f"fold {fold} is not loaded; available="
                             f"{[c['fold'] for c in self.checkpoints]}")
        return [p[0] for p in picked], [p[1] for p in picked]

    def predict_batch(self, items: list[tuple[np.ndarray, np.ndarray, int]],
                      fold: int | None = None) -> list[dict]:
        if not self.ready:
            raise RuntimeError("; ".join(self.errors) or "model not loaded")
        models, _specs = self._models_for(fold)
        torch = self.torch
        xs = np.stack([self.preprocess_one(g, m, f) for g, m, f in items])
        with torch.no_grad():
            xb = torch.from_numpy(xs).to(self.device)
            probs = None
            for model in models:
                p = torch.sigmoid(model(xb)).cpu().numpy()
                probs = p if probs is None else probs + p
            probs = probs / float(len(models))
            if self.calibration is not None:
                temp = np.asarray(
                    [float(self.calibration["temperature"][s]) for s in SURFACES],
                    dtype=np.float32)
                eps = 1e-6
                pc = np.clip(probs, eps, 1 - eps)
                z = np.log(pc / (1 - pc))
                probs = 1.0 / (1.0 + np.exp(-z / temp))
        out = []
        for (_g, _m, fdi), p in zip(items, probs):
            surfaces = {}
            for j, s in enumerate(SURFACES):
                rec = {"probability": float(p[j])}
                if self.calibration is None:
                    rec["threshold"] = None
                    rec["positive"] = None
                else:
                    thr = float(self.calibration["threshold"][s])
                    rec["threshold"] = thr
                    rec["positive"] = bool(p[j] >= thr)
                surfaces[s] = rec
            out.append({"fdi": int(fdi), "surfaces": surfaces,
                        "probabilities": [float(v) for v in p]})
        return out

    def info(self) -> dict:
        cfg = None
        if self.cfg is not None:
            cfg = {k: self.cfg[k] for k in
                   ("ablation", "backbone", "use_metadata", "img_size",
                    "input_norm", "aug_plus")}
        warnings = ["STAGING ONLY: internal CV data, no external validation"]
        if self.calibration is None:
            warnings.append(
                "probabilities only: no deployment threshold until "
                "`pipeline.py calibrate` is run on all five folds")
        return {
            "ready": self.ready,
            "device": self.device,
            "n_folds": len(self.models),
            "checkpoints": self.checkpoints,
            "config": cfg,
            "surfaces": SURFACES,
            "input_contract": {
                "crops": "grayscale tooth crop PNG (same crop as training)",
                "masks": "binary tooth-mask PNG, same size as the crop",
                "fdis": "comma-separated FDI numbers, one per crop",
                "fold": "optional integer: force one fold model",
            },
            "calibration": {
                "loaded": self.calibration is not None,
                "path": self.calibration_path,
                "decisions_enabled": self.calibration is not None,
            },
            "errors": self.errors,
            "warnings": warnings,
        }


_SERVICE: SurfaceEnsemble | None = None


def get_service() -> SurfaceEnsemble:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = SurfaceEnsemble(
            discover_checkpoints(),
            device=os.environ.get("EFFNET_DEVICE", "cpu"),
            calibration_path=os.environ.get("EFFNET_CALIBRATION") or None,
        )
        if _SERVICE.ready:
            log.info("surface ensemble ready: %d fold(s), device=%s",
                     len(_SERVICE.models), _SERVICE.device)
        else:
            log.error("surface ensemble NOT ready: %s", _SERVICE.errors)
    return _SERVICE


app = FastAPI(
    title="Dental Caries Surface Classifier - Stage 2 (staging)",
    version="staging-1",
    description="EfficientNet-B0 per-tooth multi-label surface classifier. "
                "STAGING ONLY - not clinically validated.",
)


@app.get("/health")
def health() -> dict:
    return get_service().info()


@app.get("/model/info")
def model_info() -> dict:
    return get_service().info()


@app.post("/v1/surface/predict")
async def predict(
    crops: list[UploadFile] = File(...),
    masks: list[UploadFile] = File(...),
    fdis: str = Form(...),
    fold: int | None = Form(None),
) -> dict:
    service = get_service()
    if not service.ready:
        raise HTTPException(status_code=503, detail={
            "code": "MODEL_NOT_READY", "errors": service.errors})
    try:
        fdi_list = [int(v) for v in fdis.split(",") if v.strip()]
    except ValueError:
        raise HTTPException(status_code=422,
                            detail="fdis must be comma-separated integers")
    if not (len(crops) == len(masks) == len(fdi_list)):
        raise HTTPException(
            status_code=422,
            detail=(f"length mismatch: crops={len(crops)} masks={len(masks)} "
                    f"fdis={len(fdi_list)}"))
    limit = int(os.environ.get("EFFNET_MAX_BATCH", "64"))
    if len(crops) > limit:
        raise HTTPException(status_code=413,
                            detail=f"at most {limit} items per request")

    import cv2

    items: list[tuple[np.ndarray, np.ndarray, int]] = []
    for i, (cf, mf, fdi) in enumerate(zip(crops, masks, fdi_list)):
        cb, mb = await cf.read(), await mf.read()
        gray = cv2.imdecode(np.frombuffer(cb, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(mb, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise HTTPException(status_code=422,
                                detail=f"item {i}: unreadable image")
        if mask.shape != gray.shape:
            raise HTTPException(
                status_code=422,
                detail=f"item {i}: crop {gray.shape} and mask {mask.shape} differ")
        items.append((gray, mask > 127, fdi))

    t0 = time.perf_counter()
    try:
        preds = service.predict_batch(items, fold=fold)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "predictions": preds,
        "latency_ms": round(latency_ms, 2),
        "model": {
            "n_folds_used": 1 if fold is not None else len(service.models),
            "fold": fold,
            "checkpoints": [c["file"] for c in service.checkpoints],
            "input_norm": service.cfg["input_norm"] if service.cfg else None,
            "calibration": ("loaded" if service.calibration is not None
                            else "not_calibrated"),
            "staging_only": True,
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8011")))
