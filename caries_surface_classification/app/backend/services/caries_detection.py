"""
Caries Detection Service
========================
Detects WHERE caries lesions are on the panoramic radiograph using
``caries_detect.pt`` (YOLOv8s).  This service answers only one question:

    **Which tooth has a caries lesion, and where exactly is it?**

Surface classification (Occlusal / Mesial / Distal) is handled separately
by the RF classifier (``rf_classify_ml.pkl``) in ``rf_classifier.py``.
The class labels produced by ``caries_detect.pt`` (Occlusal / Proximal /
Lingual) are intentionally **ignored** here — the RF model is the canonical
surface classifier.

Pipeline role:
    Pano YOLO (tooth detection)
        ↓
    CariesDetectionService  ← you are here
        - detects caries bounding boxes on the full panoramic image
        - maps each caries bbox to the tooth that contains its centroid
        - returns only the set of tooth detection_ids that have caries
        - also returns the caries pixel cloud (from bbox) for feature
          extraction in the RF pipeline
        ↓
    FeatureExtractor + RFClassifierService (surface: Occlusal/Mesial/Distal)

Confidence note:
    The model was trained on a 500-case panoramic dataset. Genuine caries
    detections often appear at confidence 0.03–0.07 because the lesions are
    small relative to the full-image input size.  ``CARIES_CONF = 0.03`` is
    the empirically determined threshold that balances recall vs. false
    positives on this dataset.
"""

import os
import logging
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CARIES_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "caries_detect.pt"
)

# Default threshold for the live pipeline. The evaluator can override this per run
# to measure the precision/recall trade-off against the annotated cases.
CARIES_CONF = 0.005
CARIES_IOU  = 0.30   # NMS IOU threshold
MIN_OWNERSHIP_SCORE = 0.25


class CariesDetectionService:
    """
    Detects caries lesion locations in a panoramic radiograph and maps
    each detected lesion to the tooth whose segmentation mask actually owns it.

    This version intentionally avoids centroid-only or bbox-IoU matching because
    adjacent teeth in panoramic radiographs often overlap in bounding boxes.
    """

    def __init__(self, device: str | int | None = None):
        self.model = None
        self.device = device

    # ------------------------------------------------------------------
    def _load_model(self):
        if self.model is None:
            if not os.path.exists(CARIES_MODEL_PATH):
                raise FileNotFoundError(
                    f"Missing caries detection model: {CARIES_MODEL_PATH}"
                )
            self.model = YOLO(CARIES_MODEL_PATH)

    def detect_caries_raw(self, image_path: str, confidence: float) -> list[dict]:
        self._load_model()
        predict_kwargs = {
            "conf": confidence,
            "iou": CARIES_IOU,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        results = self.model.predict(image_path, **predict_kwargs)
        if not results:
            return []
        return [
            {
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
                "confidence": float(box.conf[0]),
            }
            for box in results[0].boxes
        ]

    @staticmethod
    def _polygon_to_mask(points: List[List[float]], image_shape: tuple[int, int]) -> np.ndarray:
        """Rasterize a segmented tooth polygon to a binary mask."""
        if not points:
            return np.zeros(image_shape[:2], dtype=np.uint8)

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            return np.zeros(image_shape[:2], dtype=np.uint8)

        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts_int = np.clip(np.round(pts).astype(np.int32), [0, 0], [w - 1, h - 1])
        cv2.fillPoly(mask, [pts_int], 1)
        return mask

    @staticmethod
    def _mask_overlap_score(
        tooth_pts: List[List[float]],
        caries_bbox: list,
        image_shape: tuple[int, int],
        tooth_mask: np.ndarray | None = None,
    ) -> float:
        """Fraction of the lesion bbox that overlaps with the tooth mask."""
        if not tooth_pts:
            return 0.0

        tooth_mask = tooth_mask if tooth_mask is not None else CariesDetectionService._polygon_to_mask(tooth_pts, image_shape)
        if not np.any(tooth_mask):
            return 0.0

        x1, y1, x2, y2 = [max(0, int(v)) for v in caries_bbox]
        x2 = min(x2, image_shape[1] - 1)
        y2 = min(y2, image_shape[0] - 1)

        lesion_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.rectangle(lesion_mask, (x1, y1), (x2, y2), 1, thickness=-1)

        overlap = np.logical_and(tooth_mask > 0, lesion_mask > 0).sum()
        lesion_area = lesion_mask.sum()
        return float(overlap / lesion_area) if lesion_area > 0 else 0.0

    @staticmethod
    def _centroid_inside_mask(
        tooth_pts: List[List[float]],
        cx: float,
        cy: float,
        image_shape: tuple[int, int],
        tooth_mask: np.ndarray | None = None,
    ) -> bool:
        if not tooth_pts:
            return False
        mask = tooth_mask if tooth_mask is not None else CariesDetectionService._polygon_to_mask(tooth_pts, image_shape)
        x = int(np.clip(round(cx), 0, image_shape[1] - 1))
        y = int(np.clip(round(cy), 0, image_shape[0] - 1))
        return bool(mask[y, x] > 0)

    @staticmethod
    def _iou(a, b) -> float:
        """Intersection-over-Union of two [x1,y1,x2,y2] bounding boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = float(area_a + area_b - inter)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_to_point_cloud(bbox, density: int = 5) -> List[List[float]]:
        """Convert a caries bounding box to a dense 2D point cloud for feature extraction."""
        x1, y1, x2, y2 = bbox
        pts = []
        for i in range(density):
            for j in range(density):
                fx = i / max(density - 1, 1)
                fy = j / max(density - 1, 1)
                pts.append([x1 + fx * (x2 - x1), y1 + fy * (y2 - y1)])
        return pts

    # ------------------------------------------------------------------
    def detect_caries(
        self,
        image_path: str,
        tooth_detections: list,
        confidence: float | None = None,
        min_ownership_score: float = MIN_OWNERSHIP_SCORE,
        raw_detections: list[dict] | None = None,
    ) -> Dict[str, List[List[float]]]:
        try:
            self._load_model()
        except FileNotFoundError as e:
            logger.warning(f"CariesDetectionService: {e} — returning empty map.")
            raise

        prediction_confidence = CARIES_CONF if confidence is None else confidence
        if raw_detections is None:
            raw_detections = self.detect_caries_raw(image_path, prediction_confidence)
        raw_detections = [
            detection for detection in raw_detections
            if detection["confidence"] >= prediction_confidence
        ]

        caries_map: Dict[str, List[List[float]]] = {}

        if not raw_detections:
            logger.info("CariesDetectionService: no caries lesions detected.")
            return caries_map

        img = cv2.imread(image_path)
        image_shape = img.shape[:2] if img is not None else (2048, 2048)
        tooth_masks = [
            self._polygon_to_mask(det.tooth_pts, image_shape) if det.tooth_pts else None
            for det in tooth_detections
        ]

        raw_count = len(raw_detections)
        logger.info(
            f"CariesDetectionService: {raw_count} raw caries detection(s) "
            f"(conf ≥ {prediction_confidence})."
        )

        for detection in raw_detections:
            cx1, cy1, cx2, cy2 = [int(v) for v in detection["bbox"]]
            caries_conf = float(detection["confidence"])
            caries_bbox = [cx1, cy1, cx2, cy2]

            ccx = (cx1 + cx2) / 2.0
            ccy = (cy1 + cy2) / 2.0

            best_idx = None
            best_score = -1.0
            candidates = []

            for i, det in enumerate(tooth_detections):
                tooth_bbox = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                tooth_mask = tooth_masks[i]

                if det.tooth_pts:
                    centroid_inside = self._centroid_inside_mask(
                        det.tooth_pts, ccx, ccy, image_shape, tooth_mask=tooth_mask
                    )
                    overlap_score = self._mask_overlap_score(
                        det.tooth_pts, caries_bbox, image_shape, tooth_mask=tooth_mask
                    )
                    score = 2.0 if centroid_inside else 0.0
                    score += overlap_score
                else:
                    centroid_inside = False
                    overlap_score = self._iou(caries_bbox, tooth_bbox)
                    score = overlap_score

                candidates.append({
                    "i": i,
                    "fdi": det.fdi,
                    "bbox": tooth_bbox,
                    "centroid_inside": centroid_inside,
                    "mask_overlap": overlap_score,
                    "score": score,
                })

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None or best_score < min_ownership_score:
                logger.debug(
                    "CariesDetectionService: caries bbox rejected — "
                    "best ownership score %.4f is below %.4f.",
                    best_score,
                    min_ownership_score,
                )
                continue

            det = tooth_detections[best_idx]
            det_id = det.detection_id
            pts = self._bbox_to_point_cloud(caries_bbox, density=5)

            if det_id not in caries_map:
                caries_map[det_id] = []
            caries_map[det_id].extend(pts)

            logger.info(
                "CariesDetectionService: lesion bbox=%s centroid=(%.2f, %.2f) "
                "candidates=%s selected_fdi=%s selected_id=%s selected_score=%.4f",
                caries_bbox,
                ccx,
                ccy,
                [
                    {
                        "fdi": c["fdi"],
                        "bbox": c["bbox"],
                        "centroid_inside": c["centroid_inside"],
                        "mask_overlap": round(c["mask_overlap"], 4),
                        "score": round(c["score"], 4),
                    }
                    for c in candidates
                ],
                det.fdi,
                det_id,
                best_score,
            )

        logger.info(
            f"CariesDetectionService: {len(caries_map)} tooth/teeth "
            f"identified as having caries."
        )
        return caries_map
