"""
Stage 1 — Tooth detection and segmentation.

Sub-stages:
  1a. Panoramic YOLO (Tooth_seg_pano_20250319.pt)
      → 32 FDI tooth classes, bbox + mask polygon per tooth
  1b. Caries YOLO (caries_detect.pt)
      → caries lesion bboxes mapped to owning tooth via mask overlap
      → caries_points: Nx2 point cloud per tooth
  1c. Detectron2 crop segmenter (Tooth_seg_crop_20250424.pth)  [optional]
      → refined tooth mask per crop; falls back to pano mask if disabled

Output per tooth: ToothDetection dataclass (see pipeline/orchestrator.py).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.numbering import class_id_to_fdi

log = logging.getLogger(__name__)

# Caries detection tuning (mirrors caries_detection.py in reference pipeline)
_CARIES_CONF_DEFAULT = 0.005
_CARIES_IOU = 0.30
_MIN_OWNERSHIP_SCORE = 0.25
_POINT_CLOUD_DENSITY = 5  # pts per pixel along bbox edge


# ---------------------------------------------------------------------------
# Data contract between Stage 1 and downstream stages
# ---------------------------------------------------------------------------

@dataclass
class ToothDetection:
    fdi: int
    bbox_xywh: tuple[int, int, int, int]   # [x, y, w, h] in original image coords
    tooth_polygon: list[list[float]]        # mask polygon (original coords)
    caries_points: np.ndarray | None        # Nx2 or None (no caries detected)
    pano_confidence: float


# ---------------------------------------------------------------------------
# Stage 1a — Panoramic YOLO
# ---------------------------------------------------------------------------

def run_pano_detection(
    image_path: str,
    pano_model: Any,
    detection_threshold: float = 0.25,
) -> list[ToothDetection]:
    """
    Run the panoramic segmentation YOLO model and return per-tooth instances.
    """
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

        # bbox xyxy → xywh
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)

        # Mask polygon
        if r.masks is not None and i < len(r.masks.xy):
            polygon = [[float(p[0]), float(p[1])] for p in r.masks.xy[i]]
        else:
            # Fallback: bbox corners as degenerate polygon
            polygon = [
                [float(x1), float(y1)], [float(x2), float(y1)],
                [float(x2), float(y2)], [float(x1), float(y2)],
            ]

        detections.append(ToothDetection(
            fdi=fdi,
            bbox_xywh=bbox_xywh,
            tooth_polygon=polygon,
            caries_points=None,
            pano_confidence=conf,
        ))

    log.info("pano YOLO: %d teeth detected in %s", len(detections), Path(image_path).name)
    return detections


# ---------------------------------------------------------------------------
# Stage 1b — Caries YOLO (maps lesions → owning tooth)
# ---------------------------------------------------------------------------

def _bbox_to_point_cloud(bbox: list[float], density: int) -> np.ndarray:
    """Sample a uniform grid of points inside a bbox."""
    x1, y1, x2, y2 = bbox
    xs = np.linspace(x1, x2, max(2, int((x2 - x1) / density)))
    ys = np.linspace(y1, y2, max(2, int((y2 - y1) / density)))
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _polygon_to_mask(
    points: list[list[float]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a polygon to a binary HxW uint8 mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _mask_overlap_score(
    tooth_polygon: list[list[float]],
    caries_bbox: list[float],
    image_shape: tuple[int, int],
) -> float:
    """Fraction of caries bbox area covered by the tooth mask."""
    cx1, cy1, cx2, cy2 = [int(v) for v in caries_bbox]
    tooth_mask = _polygon_to_mask(tooth_polygon, image_shape)
    roi = tooth_mask[max(0, cy1):cy2, max(0, cx1):cx2]
    if roi.size == 0:
        return 0.0
    return float(roi.sum()) / roi.size


def _centroid_inside(
    tooth_polygon: list[list[float]],
    cx: float,
    cy: float,
    image_shape: tuple[int, int],
) -> bool:
    """True if the point (cx, cy) falls inside the tooth mask polygon."""
    if len(tooth_polygon) < 3:
        return False
    tooth_mask = _polygon_to_mask(tooth_polygon, image_shape)
    iy, ix = int(cy), int(cx)
    h, w = image_shape
    if 0 <= iy < h and 0 <= ix < w:
        return bool(tooth_mask[iy, ix])
    return False


def run_caries_detection(
    image_path: str,
    caries_model: Any,
    detections: list[ToothDetection],
    image_shape: tuple[int, int],
    caries_conf: float = _CARIES_CONF_DEFAULT,
    min_ownership_score: float = _MIN_OWNERSHIP_SCORE,
) -> list[ToothDetection]:
    """
    Detect caries lesions and assign point clouds to owning teeth.
    Mutates detections in-place (sets caries_points).
    Returns the same list.
    """
    results = caries_model.predict(
        image_path,
        conf=caries_conf,
        iou=_CARIES_IOU,
        verbose=False,
    )
    if not results or not results[0].boxes:
        log.info("caries YOLO: no lesions detected")
        return detections

    r = results[0]
    caries_map: dict[int, list[np.ndarray]] = {}  # index into detections → list of clouds

    for box in r.boxes:
        caries_bbox = box.xyxy[0].tolist()
        cx1, cy1, cx2, cy2 = caries_bbox
        ccx, ccy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0

        best_idx = -1
        best_score = -1.0

        for i, det in enumerate(detections):
            ci = _centroid_inside(det.tooth_polygon, ccx, ccy, image_shape)
            ov = _mask_overlap_score(det.tooth_polygon, caries_bbox, image_shape)
            score = (2.0 if ci else 0.0) + ov

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score < min_ownership_score:
            log.debug(
                "caries YOLO: lesion bbox=%s rejected (best_score=%.4f < %.4f)",
                caries_bbox, best_score, min_ownership_score,
            )
            continue

        pts = _bbox_to_point_cloud(caries_bbox, _POINT_CLOUD_DENSITY)
        caries_map.setdefault(best_idx, []).append(pts)

    for idx, clouds in caries_map.items():
        detections[idx].caries_points = np.vstack(clouds)

    n_with_caries = sum(1 for d in detections if d.caries_points is not None)
    log.info("caries YOLO: %d/%d teeth have caries", n_with_caries, len(detections))
    return detections


# ---------------------------------------------------------------------------
# Stage 1c — Detectron2 crop segmenter (optional)
# ---------------------------------------------------------------------------

def refine_with_crop_segmenter(
    image: np.ndarray,
    detections: list[ToothDetection],
    crop_model_bundle: dict | None,
) -> list[ToothDetection]:
    """
    Refine tooth_polygon for each detection using the Detectron2 crop segmenter.
    Falls back to the existing pano mask if the model is disabled or fails.
    """
    if crop_model_bundle is None:
        log.debug("crop_segmenter disabled — using pano masks")
        return detections

    try:
        import torch
        from detectron2.engine import DefaultPredictor  # type: ignore[import]
        from app.pipeline.postprocess import mask_bits_to_polygon
    except ImportError:
        log.warning("detectron2 not importable in child — skipping crop refinement")
        return detections

    model = crop_model_bundle["model"]
    cfg = crop_model_bundle["cfg"]

    img_h, img_w = image.shape[:2]

    for det in detections:
        x, y, w, h = det.bbox_xywh
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            with torch.no_grad():
                outputs = model([{
                    "image": torch.as_tensor(
                        crop.transpose(2, 0, 1).astype("float32")
                    ),
                    "height": y2 - y1,
                    "width": x2 - x1,
                }])
            instances = outputs[0]["instances"]
            if len(instances) == 0:
                continue

            # Take highest-scoring instance mask
            scores = instances.scores.cpu().numpy()
            best = int(np.argmax(scores))
            mask_crop = instances.pred_masks[best].cpu().numpy().astype(np.uint8)

            polygon = mask_bits_to_polygon(mask_crop)
            if polygon is None or len(polygon) < 3:
                continue

            # Shift polygon coords back to original image space
            det.tooth_polygon = [
                [pt[0] + x1, pt[1] + y1] for pt in polygon
            ]
        except Exception as exc:
            log.warning("crop_segmenter failed for FDI=%d: %s", det.fdi, exc)
            # Keep pano mask — no re-raise

    return detections


# ---------------------------------------------------------------------------
# Public entry point for Stage 1
# ---------------------------------------------------------------------------

def run_stage1(
    image_path: str,
    models: dict,
    detection_threshold: float = 0.25,
    caries_conf: float = _CARIES_CONF_DEFAULT,
) -> tuple[list[ToothDetection], np.ndarray]:
    """
    Run all Stage 1 sub-stages.

    Returns:
        detections: List of ToothDetection (with caries_points if found)
        image:      Loaded BGR image array (reused by Stage 1c and Canvas)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_h, img_w = image.shape[:2]

    # 1a — pano detection
    detections = run_pano_detection(image_path, models["pano"], detection_threshold)
    if not detections:
        return detections, image

    # 1b — caries detection
    detections = run_caries_detection(
        image_path, models["caries"], detections,
        image_shape=(img_h, img_w), caries_conf=caries_conf,
    )

    # 1c — optional crop refinement
    detections = refine_with_crop_segmenter(image, detections, models.get("crop"))

    return detections, image
