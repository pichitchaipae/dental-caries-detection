"""
Artifact registry — verify checksums and load models inside child process.

Public surface:
  verify_artifacts(weights_dir)  -> ArtifactStatus   (fast, called on /health + /infer)
  load_models_in_child(weights_dir, device) -> dict  (slow, called only in child process)

The caries detector is optional.  It is loaded only when a verified entry exists
in ``models.versions.MODEL_VERSIONS``; otherwise the pipeline still runs tooth
segmentation and reports ``caries_detector="not_configured"``.
"""

from __future__ import annotations

import hashlib
import logging
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
_STATUS_NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True)
class ArtifactStatus:
    pano_detector: str
    crop_segmenter: str
    caries_detector: str

    def is_ready(self) -> bool:
        """Core tooth pipeline is ready when pano is verified and crop-seg is
        either verified or deliberately disabled (pano masks are the fallback).

        ``caries_detector`` is intentionally not part of readiness: it is
        optional until the crop-trained caries model is published, and /health
        reports its state explicitly as ``not_configured``.
        """
        if self.pano_detector != _STATUS_VERIFIED:
            return False
        return self.crop_segmenter in (_STATUS_VERIFIED, _STATUS_DISABLED)

    def as_dict(self) -> dict[str, str]:
        return {
            "pano_detector": self.pano_detector,
            "crop_segmenter": self.crop_segmenter,
            "caries_detector": self.caries_detector,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_artifact(weights_dir: str, key: str) -> str:
    """Return status string for a single artifact."""
    spec = MODEL_VERSIONS.get(key)
    if spec is None:
        return _STATUS_NOT_CONFIGURED
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
    Verify file presence and SHA-256 for all configured model weights.
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
    Load all configured model weights.  Raises on a required-model failure so
    the child process exits non-zero and the monitor thread marks the job
    'fail'.

    Returns a dict with keys: pano, crop (may be None), caries (may be None).
    """
    w = Path(weights_dir)

    # -- 1. YOLO pano detector (required) --
    log.info("loading pano_detector …")
    from ultralytics import YOLO  # type: ignore[import]
    pano = YOLO(str(w / MODEL_VERSIONS["pano_detector"]["filename"]))
    log.info("pano_detector loaded  (task=%s, classes=%d)", pano.task, len(pano.names))

    # -- 2. Detectron2 crop segmenter (optional, env-gated) --
    crop: Any = None
    if enable_crop_segmenter:
        log.info("loading crop_segmenter …")
        crop = _load_detectron2_crop(w, device)
        log.info("crop_segmenter loaded")
    else:
        log.info("crop_segmenter disabled — using pano mask as fallback")

    # -- 3. Crop-trained caries YOLO-seg (optional until trained) --
    caries: Any = None
    spec = MODEL_VERSIONS.get("caries_detector")
    if spec is None:
        log.warning(
            "caries_detector not configured — Stage 1 will run tooth "
            "segmentation only. Add a verified 'caries_detector' entry to "
            "models/versions.py after training."
        )
    else:
        # Never serve an artifact that /health would report as failed: a
        # checksum mismatch means the pinned model is not the file on disk.
        status = _check_artifact(str(w), "caries_detector")
        path = w / spec["filename"]
        if status != _STATUS_VERIFIED:
            log.warning(
                "caries_detector is %s — not loading it (%s). "
                "Run without caries until the pinned file is restored.",
                status, path,
            )
        else:
            log.info("loading caries_detector …")
            caries = YOLO(str(path))
            log.info("caries_detector loaded  (task=%s, classes=%s)", caries.task, caries.names)

    return {"pano": pano, "crop": crop, "caries": caries}


def _load_detectron2_crop(weights_dir: Path, device: str = "cpu") -> Any:
    """Build Detectron2 config and load Mask R-CNN R50-FPN checkpoint."""
    try:
        from detectron2.config import get_cfg  # type: ignore[import]
        from detectron2 import model_zoo  # type: ignore[import]
        from detectron2.data import transforms as T  # type: ignore[import]
        from detectron2.modeling import build_model  # type: ignore[import]
        from detectron2.checkpoint import DetectionCheckpointer  # type: ignore[import]
        import torch
    except ImportError as exc:
        raise ImportError(
            "detectron2 is not installed. Install the pre-built wheel "
            "(see Dockerfile / requirements-detectron2.txt) or set "
            "ENABLE_CROP_SEGMENTER=false."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("ML_DEVICE=%s but CUDA is unavailable — falling back to cpu", device)
        device = "cpu"

    spec = MODEL_VERSIONS["crop_segmenter"]
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = spec["num_classes"]  # 1 — tooth
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(spec.get("score_thresh_test", 0.5))
    # These two are the DefaultPredictor preprocessing parameters.  The model
    # itself never resizes its input.
    cfg.INPUT.MIN_SIZE_TEST = int(spec.get("min_size_test", 800))
    cfg.INPUT.MAX_SIZE_TEST = int(spec.get("max_size_test", 1333))
    cfg.MODEL.WEIGHTS = ""  # weights loaded via checkpointer below
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    model = build_model(cfg)
    model.eval()

    ckpt_path = str(weights_dir / spec["filename"])
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(ckpt_path)

    # Same augmenter as detectron2.engine.DefaultPredictor.  Exposed in the
    # bundle so Stage 1 can apply identical preprocessing per crop.
    resize = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )

    log.info(
        "detectron2 crop_segmenter loaded from %s "
        "(device=%s, score_thresh=%.2f, min_size_test=%d, max_size_test=%d)",
        ckpt_path, device, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST,
    )
    return {"model": model, "cfg": cfg, "resize": resize}
