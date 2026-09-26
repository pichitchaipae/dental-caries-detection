"""
Artifact registry — verify checksums and load models inside child process.

Public surface:
  verify_artifacts(weights_dir)  -> ArtifactStatus   (fast, called on /health + /infer)
  load_models_in_child(weights_dir, device) -> dict  (slow, called only in child process)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.versions import MODEL_VERSIONS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact status
# ---------------------------------------------------------------------------

_STATUS_VERIFIED = "verified"
_STATUS_MISSING = "missing"
_STATUS_CHECKSUM_FAIL = "checksum_fail"
_STATUS_DISABLED = "disabled"
_STATUS_LOAD_FAILED = "load_failed"


@dataclass(frozen=True)
class ArtifactStatus:
    pano_detector: str
    crop_segmenter: str
    caries_detector: str
    surface_classifier: str

    def is_ready(self) -> bool:
        """Core pipeline is ready when all 3 required models are verified."""
        core = [self.pano_detector, self.caries_detector, self.surface_classifier]
        return all(s == _STATUS_VERIFIED for s in core)

    def as_dict(self) -> dict[str, str]:
        return {
            "pano_detector": self.pano_detector,
            "crop_segmenter": self.crop_segmenter,
            "caries_detector": self.caries_detector,
            "surface_classifier": self.surface_classifier,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_artifact(weights_dir: str, key: str) -> str:
    """Return status string for a single artifact."""
    spec = MODEL_VERSIONS[key]
    path = Path(weights_dir) / spec["filename"]
    if not path.is_file():
        log.warning("artifact missing: %s", path)
        return _STATUS_MISSING
    digest = _sha256(path)
    if digest.lower() != spec["sha256"].lower():
        log.error(
            "checksum mismatch for %s: expected %s, got %s",
            spec["filename"], spec["sha256"], digest,
        )
        return _STATUS_CHECKSUM_FAIL
    return _STATUS_VERIFIED


def verify_artifacts(weights_dir: str, enable_crop_segmenter: bool = True) -> ArtifactStatus:
    """
    Verify file presence and SHA-256 for all model weights.
    Fast enough to call on every /health and /infer request.
    """
    crop_status = (
        _check_artifact(weights_dir, "crop_segmenter")
        if enable_crop_segmenter
        else _STATUS_DISABLED
    )
    return ArtifactStatus(
        pano_detector=_check_artifact(weights_dir, "pano_detector"),
        crop_segmenter=crop_status,
        caries_detector=_check_artifact(weights_dir, "caries_detector"),
        surface_classifier=_check_artifact(weights_dir, "surface_classifier"),
    )


# ---------------------------------------------------------------------------
# Model loading — called ONLY inside child inference process
# ---------------------------------------------------------------------------

def load_models_in_child(
    weights_dir: str,
    device: str,
    enable_crop_segmenter: bool = True,
) -> dict[str, Any]:
    """
    Load all model weights.  Raises on any failure so the child process exits
    non-zero and the monitor thread marks the job as 'fail'.

    Returns a dict with keys: pano, caries, rf, crop (may be None).
    """
    w = Path(weights_dir)

    # -- 1. YOLO pano detector --
    log.info("loading pano_detector …")
    from ultralytics import YOLO  # type: ignore[import]
    pano = YOLO(str(w / MODEL_VERSIONS["pano_detector"]["filename"]))
    log.info("pano_detector loaded  (task=%s, classes=%d)", pano.task, len(pano.names))

    # -- 2. Caries YOLO detector --
    log.info("loading caries_detector …")
    caries = YOLO(str(w / MODEL_VERSIONS["caries_detector"]["filename"]))
    log.info("caries_detector loaded")

    # -- 3. Detectron2 crop segmenter (optional) --
    crop: Any = None
    if enable_crop_segmenter:
        log.info("loading crop_segmenter …")
        crop = _load_detectron2_crop(w)
        log.info("crop_segmenter loaded")
    else:
        log.info("crop_segmenter disabled — using pano mask as fallback")

    # -- 4. RF surface classifier --
    log.info("loading surface_classifier …")
    import joblib  # type: ignore[import]
    spec = MODEL_VERSIONS["surface_classifier"]
    rf = joblib.load(str(w / spec["filename"]))
    # Verify feature contract
    if rf.n_features_in_ != spec["n_features"]:
        raise ValueError(
            f"RF feature count mismatch: expected {spec['n_features']}, "
            f"got {rf.n_features_in_}"
        )
    log.info(
        "surface_classifier loaded (estimators=%d, features=%d, classes=%s)",
        rf.n_estimators, rf.n_features_in_, list(rf.classes_),
    )

    return {"pano": pano, "caries": caries, "crop": crop, "rf": rf}


def _load_detectron2_crop(weights_dir: Path) -> Any:
    """Build Detectron2 config and load Mask R-CNN R50-FPN checkpoint."""
    try:
        from detectron2.config import get_cfg  # type: ignore[import]
        from detectron2 import model_zoo  # type: ignore[import]
        from detectron2.modeling import build_model  # type: ignore[import]
        from detectron2.checkpoint import DetectionCheckpointer  # type: ignore[import]
        import torch
    except ImportError as exc:
        raise ImportError(
            "detectron2 is not installed. Add it to requirements.txt "
            "or set ENABLE_CROP_SEGMENTER=false."
        ) from exc

    spec = MODEL_VERSIONS["crop_segmenter"]
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = spec["num_classes"]  # 1 — tooth
    cfg.MODEL.WEIGHTS = ""  # weights loaded via checkpointer below
    cfg.MODEL.DEVICE = "cpu"  # always CPU for inference process

    model = build_model(cfg)
    model.eval()

    ckpt_path = str(weights_dir / spec["filename"])
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(ckpt_path)
    log.info("detectron2 crop_segmenter loaded from %s", ckpt_path)
    return {"model": model, "cfg": cfg}
