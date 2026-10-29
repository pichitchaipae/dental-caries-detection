"""
Stage 1 — Tooth detection, crop refinement, and caries detection on crops.

Flow (matches the approved pipeline):

    pano image
      -> Pano-Seg YOLO (Tooth_seg_pano_20250319.pt)       # bbox + mask + FDI
      -> dedupe by FDI + bbox IoU > 0.5
      -> crop each tooth with PAD 20
      -> Crop-Seg Detectron2 (Tooth_seg_crop_20250424.pth) # refined tooth mask
      -> caries YOLO-seg (1 class, runs on the crop image)
      -> transform caries bbox/mask/centroid crop -> pano
      -> ToothDetection list

There is deliberately no caries-to-tooth ownership heuristic: a lesion detected
inside a tooth crop belongs to that tooth by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.coordinate_transform import (
    bbox_crop_to_pano,
    bbox_iou,
    crop_bbox_from_xywh,
    crop_to_pano,
    polygon_centroid,
    xyxy_to_xywh,
)
from app.pipeline.crop_utils import suppress_other_teeth
from app.pipeline.numbering import class_id_to_fdi

log = logging.getLogger(__name__)

# Crop padding in pano pixels — same value as the reference notebook and the
# surface-CNN training rule.
CROP_PAD = 20
DEDUPE_IOU = 0.50
CARIES_CONF_DEFAULT = 0.25
CARIES_IOU_DEFAULT = 0.50


# ---------------------------------------------------------------------------
# Data contract between Stage 1 and downstream stages
# ---------------------------------------------------------------------------

@dataclass
class CariesLesion:
    """One predicted caries instance in panoramic coordinates."""
    bbox_xyxy: tuple[float, float, float, float]
    polygon: list[list[float]]          # pano coords
    confidence: float
    centroid: tuple[float, float]       # pano coords


@dataclass
class ToothDetection:
    fdi: int
    bbox_xywh: tuple[int, int, int, int]   # [x, y, w, h] in pano coords
    tooth_polygon: list[list[float]]        # refined mask polygon (pano coords)
    pano_confidence: float
    caries_points: np.ndarray | None = None  # Nx2 polygon points (pano coords)
    caries_lesions: list[CariesLesion] = field(default_factory=list)
    crop_bbox_xyxy: tuple[int, int, int, int] | None = None
    mask_source: str = "pano"               # "pano" | "crop_seg"


# ---------------------------------------------------------------------------
# Stage 1a — Panoramic YOLO
# ---------------------------------------------------------------------------

def run_pano_detection(
    image_path: str,
    pano_model: Any,
    detection_threshold: float = 0.25,
) -> list[ToothDetection]:
    """Run the panoramic segmentation YOLO model and return per-tooth instances."""
    results = pano_model.predict(
        image_path,
        conf=detection_threshold,
        verbose=False,
    )
    if not results or results[0].masks is None:
        log.warning("pano YOLO: no detections in %s", image_path)
        return []

    detections: list[ToothDetection] = []
    r = results[0]
    img_h, img_w = r.orig_shape

    for i, box in enumerate(r.boxes):
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        try:
            fdi = class_id_to_fdi(class_id, pano_model.names)
        except (KeyError, ValueError) as exc:
            log.warning("pano YOLO: skipping class_id=%d — %s", class_id, exc)
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)

        if r.masks is not None and i < len(r.masks.xy) and len(r.masks.xy[i]) >= 3:
            polygon = [[float(p[0]), float(p[1])] for p in r.masks.xy[i]]
        else:
            polygon = [
                [float(x1), float(y1)], [float(x2), float(y1)],
                [float(x2), float(y2)], [float(x1), float(y2)],
            ]

        detections.append(ToothDetection(
            fdi=fdi,
            bbox_xywh=bbox_xywh,
            tooth_polygon=polygon,
            pano_confidence=conf,
        ))

    log.info("pano YOLO: %d teeth detected in %s", len(detections), Path(image_path).name)
    return detections


def dedupe_detections(
    detections: list[ToothDetection],
    iou_thresh: float = DEDUPE_IOU,
) -> list[ToothDetection]:
    """Drop duplicate detections of the same FDI with bbox IoU > ``iou_thresh``.

    The reference notebook dedupes by IoU.  We additionally guarantee one
    detection per FDI (the output contract is keyed by FDI): if two boxes of the
    same FDI do not overlap enough to be called duplicates, the lower-confidence
    one is dropped with a warning.
    """
    if len(detections) <= 1:
        return detections

    by_fdi: dict[int, list[int]] = {}
    for idx, det in enumerate(detections):
        by_fdi.setdefault(det.fdi, []).append(idx)

    keep: set[int] = set()
    for fdi, indices in by_fdi.items():
        ranked = sorted(indices, key=lambda i: detections[i].pano_confidence, reverse=True)
        kept_for_fdi: list[int] = []
        for idx in ranked:
            box = _bbox_xyxy(detections[idx].bbox_xywh)
            duplicate = any(
                bbox_iou(box, _bbox_xyxy(detections[k].bbox_xywh)) > iou_thresh
                for k in kept_for_fdi
            )
            if duplicate:
                log.info(
                    "dedupe: FDI %d drop duplicate (conf %.3f, IoU > %.2f)",
                    fdi, detections[idx].pano_confidence, iou_thresh,
                )
                continue
            kept_for_fdi.append(idx)

        if len(kept_for_fdi) > 1:
            log.warning(
                "dedupe: FDI %d still has %d non-overlapping detections — "
                "keeping the highest-confidence one",
                fdi, len(kept_for_fdi),
            )
            kept_for_fdi = kept_for_fdi[:1]

        keep.update(kept_for_fdi)

    return [det for i, det in enumerate(detections) if i in keep]


def _bbox_xyxy(bbox_xywh: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return float(x), float(y), float(x + w), float(y + h)


# ---------------------------------------------------------------------------
# Stage 1b — Crop-Seg (Detectron2) on each tooth crop
# ---------------------------------------------------------------------------

def assign_crop_bboxes(
    detections: list[ToothDetection],
    image_shape: tuple[int, int],
    pad: int = CROP_PAD,
) -> None:
    """Populate ``crop_bbox_xyxy`` from each pano bbox (clamped, with pad)."""
    for det in detections:
        det.crop_bbox_xyxy = crop_bbox_from_xywh(det.bbox_xywh, image_shape, pad=pad)


def refine_with_crop_segmenter(
    image: np.ndarray,
    detections: list[ToothDetection],
    crop_model_bundle: dict | None,
    pad: int = CROP_PAD,
) -> list[ToothDetection]:
    """Refine each tooth mask with the Detectron2 crop segmenter.

    Falls back to the pano mask when the model is disabled, produces no
    instance, or fails.  ``crop_bbox_xyxy`` is always populated so the caries
    stage can run on the same crop.
    """
    img_h, img_w = image.shape[:2]
    assign_crop_bboxes(detections, (img_h, img_w), pad=pad)

    if crop_model_bundle is None:
        log.debug("crop_segmenter disabled — using pano masks")
        return detections

    try:
        import torch
        from app.pipeline.postprocess import mask_bits_to_polygon
    except ImportError:
        log.warning("torch not importable in child — skipping crop refinement")
        return detections

    model = crop_model_bundle["model"]
    resize = crop_model_bundle.get("resize")
    if resize is None:
        log.warning(
            "crop_seg bundle has no ResizeShortestEdge transform; Detectron2 "
            "expects MIN_SIZE_TEST preprocessing and will likely return no instances"
        )

    refined = 0
    for det in detections:
        crop_bbox = det.crop_bbox_xyxy
        if crop_bbox is None:
            continue
        x1, y1, x2, y2 = crop_bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            # Match detectron2.engine.DefaultPredictor exactly: BGR -> RGB,
            # then ResizeShortestEdge(MIN_SIZE_TEST, MAX_SIZE_TEST).  Detectron2
            # does not resize internally; without this every crop scores ~0.
            model_input = crop[:, :, ::-1]
            if resize is not None:
                model_input = resize.get_transform(model_input).apply_image(model_input)
            model_input = np.ascontiguousarray(model_input)

            with torch.no_grad():
                outputs = model([{
                    "image": torch.as_tensor(
                        np.ascontiguousarray(
                            model_input.transpose(2, 0, 1), dtype=np.float32
                        )
                    ),
                    "height": y2 - y1,
                    "width": x2 - x1,
                }])
            instances = outputs[0]["instances"]
            if len(instances) == 0:
                log.debug("crop_seg: no instance for FDI %d — keeping pano mask", det.fdi)
                continue

            scores = instances.scores.cpu().numpy()
            best = int(np.argmax(scores))
            mask_crop = instances.pred_masks[best].cpu().numpy().astype(np.uint8)

            # pred_masks are in the resized model-input space.  Map back to the
            # original crop before converting to pano coordinates.
            if mask_crop.shape != crop.shape[:2]:
                mask_crop = cv2.resize(
                    mask_crop,
                    (crop.shape[1], crop.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            polygon_crop = mask_bits_to_polygon(mask_crop)
            if polygon_crop is None or len(polygon_crop) < 3:
                continue

            polygon_pano = crop_to_pano(polygon_crop, crop_bbox)
            det.tooth_polygon = polygon_pano.tolist()
            det.mask_source = "crop_seg"
            refined += 1

            xs, ys = polygon_pano[:, 0], polygon_pano[:, 1]
            det.bbox_xywh = xyxy_to_xywh((xs.min(), ys.min(), xs.max(), ys.max()))
        except Exception as exc:
            log.warning("crop_segmenter failed for FDI=%d: %s", det.fdi, exc)
            # Keep the pano mask — no re-raise.

    if detections:
        if refined == 0:
            log.warning(
                "crop_seg refined 0/%d tooth masks — check preprocessing and "
                "SCORE_THRESH_TEST; every tooth is using its pano mask",
                len(detections),
            )
        else:
            log.info("crop_seg refined %d/%d tooth masks", refined, len(detections))

    return detections


# ---------------------------------------------------------------------------
# Stage 1c — Caries YOLO-seg on each tooth crop
# ---------------------------------------------------------------------------

def _validate_caries_model(caries_model: Any) -> None:
    names = getattr(caries_model, "names", None)
    if not names:
        return
    labels = list(names.values()) if isinstance(names, dict) else list(names)
    if len(labels) != 1:
        raise ValueError(
            "caries_detector must be a 1-class model ('caries'); "
            f"got {len(labels)} classes: {labels}. The 3-class "
            "Occlusal/Proximal/Lingual taxonomy was rejected."
        )
    if str(labels[0]).lower() != "caries":
        raise ValueError(
            "caries_detector must emit the class 'caries'; "
            f"got {labels!r}. Re-train with data.yaml names: ['caries']."
        )


def _polygon_from_bbox(bbox_xyxy: tuple[float, float, float, float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox_xyxy
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def run_caries_on_crops(
    image: np.ndarray,
    detections: list[ToothDetection],
    caries_model: Any | None,
    caries_conf: float = CARIES_CONF_DEFAULT,
    caries_iou: float = CARIES_IOU_DEFAULT,
    suppress_neighbours: bool = True,
) -> list[ToothDetection]:
    """Run the crop-trained caries model on every tooth crop.

    Predictions are transformed back to pano coordinates and attached to the
    tooth they were detected on.  No global caries pass, no ownership scoring.
    """
    if caries_model is None:
        log.warning("caries_detector not configured — no caries findings will be produced")
        return detections

    _validate_caries_model(caries_model)

    total = 0
    for det in detections:
        crop_bbox = det.crop_bbox_xyxy
        if crop_bbox is None:
            continue
        x1, y1, x2, y2 = crop_bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if suppress_neighbours:
            other_polygons = [
                other.tooth_polygon for other in detections if other is not det
            ]
            crop = suppress_other_teeth(
                crop, crop_bbox, det.tooth_polygon, other_polygons,
            )

        results = caries_model.predict(
            source=crop,
            conf=caries_conf,
            iou=caries_iou,
            verbose=False,
        )
        if not results:
            continue
        r = results[0]

        lesions: list[CariesLesion] = []
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            bbox_crop = tuple(float(v) for v in box.xyxy[0].tolist())

            if r.masks is not None and i < len(r.masks.xy) and len(r.masks.xy[i]) >= 3:
                polygon_crop = np.asarray(r.masks.xy[i], dtype=np.float64)
            else:
                polygon_crop = np.asarray(_polygon_from_bbox(bbox_crop), dtype=np.float64)

            polygon_pano = crop_to_pano(polygon_crop, crop_bbox)
            bbox_pano = bbox_crop_to_pano(bbox_crop, crop_bbox)
            centroid = polygon_centroid(polygon_pano)

            lesions.append(CariesLesion(
                bbox_xyxy=bbox_pano,
                polygon=polygon_pano.tolist(),
                confidence=conf,
                centroid=centroid,
            ))

        if lesions:
            det.caries_lesions = lesions
            det.caries_points = np.vstack([
                np.asarray(les.polygon, dtype=np.float64) for les in lesions
            ])
            total += len(lesions)

    n_teeth = sum(1 for d in detections if d.caries_lesions)
    log.info("caries YOLO: %d lesion(s) on %d/%d teeth", total, n_teeth, len(detections))
    return detections


# ---------------------------------------------------------------------------
# Public entry point for Stage 1
# ---------------------------------------------------------------------------

def run_stage1(
    image_path: str,
    models: dict,
    detection_threshold: float = 0.25,
    caries_conf: float = CARIES_CONF_DEFAULT,
    caries_iou: float = CARIES_IOU_DEFAULT,
    crop_pad: int = CROP_PAD,
    dedupe_iou: float = DEDUPE_IOU,
    suppress_neighbours: bool = True,
) -> tuple[list[ToothDetection], np.ndarray]:
    """
    Run all Stage 1 sub-stages.

    Returns:
        detections: List of ToothDetection (with caries_lesions if found)
        image:      Loaded BGR image array (reused by the orchestrator)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 1a — pano detection + dedupe
    detections = run_pano_detection(image_path, models["pano"], detection_threshold)
    if not detections:
        return detections, image
    detections = dedupe_detections(detections, iou_thresh=dedupe_iou)

    # 1b — crop + Crop-Seg refinement (also sets crop_bbox_xyxy)
    detections = refine_with_crop_segmenter(
        image, detections, models.get("crop"), pad=crop_pad,
    )

    # 1c — caries on the tooth crop, then back to pano coordinates
    detections = run_caries_on_crops(
        image, detections, models.get("caries"),
        caries_conf=caries_conf,
        caries_iou=caries_iou,
        suppress_neighbours=suppress_neighbours,
    )

    return detections, image
