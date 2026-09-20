import os
import cv2
import numpy as np
from ultralytics import YOLO

from schemas.inference import ToothDetectionResult, BoundingBox

PANO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Tooth_seg_pano_20250319.pt")


class ToothDetectionService:
    def __init__(self):
        self.pano_model = None

    def load_model(self):
        if self.pano_model is None:
            if not os.path.exists(PANO_MODEL_PATH):
                raise FileNotFoundError(f"Missing model: {PANO_MODEL_PATH}")
            self.pano_model = YOLO(PANO_MODEL_PATH)

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        return interArea / unionArea if unionArea > 0 else 0

    @staticmethod
    def compute_mask_overlap_score(points_a, points_b):
        """Rough overlap of two tooth masks from segmentation polygons.
        This is more reliable than bbox IoU when adjacent teeth are close.
        """
        if not points_a or not points_b:
            return 0.0

        pts_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
        pts_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
        if pts_a.shape[0] < 3 or pts_b.shape[0] < 3:
            return 0.0

        min_x = max(int(np.min(pts_a[:, 0])), int(np.min(pts_b[:, 0])))
        min_y = max(int(np.min(pts_a[:, 1])), int(np.min(pts_b[:, 1])))
        max_x = min(int(np.max(pts_a[:, 0])), int(np.max(pts_b[:, 0])))
        max_y = min(int(np.max(pts_a[:, 1])), int(np.max(pts_b[:, 1])))
        if max_x <= min_x or max_y <= min_y:
            return 0.0

        # Use polygon mask overlap inside the narrow shared bounding box.
        h = max_y - min_y + 1
        w = max_x - min_x + 1
        if h <= 0 or w <= 0:
            return 0.0

        a_shift = pts_a - np.array([min_x, min_y], dtype=np.float32)
        b_shift = pts_b - np.array([min_x, min_y], dtype=np.float32)

        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_a, [np.round(a_shift).astype(np.int32)], 1)
        cv2.fillPoly(mask_b, [np.round(b_shift).astype(np.int32)], 1)

        inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
        union = np.logical_or(mask_a > 0, mask_b > 0).sum()
        return float(inter / union) if union > 0 else 0.0

    def detect_teeth(self, image_path: str) -> list[ToothDetectionResult]:
        self.load_model()
        results = self.pano_model(image_path)

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names
        masks = result.masks

        temp_detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            class_name = names[cls_idx] if cls_idx in names else "unknown"

            # Parse FDI
            fdi = "46"
            original_class = class_name
            if "_" in class_name and class_name.split("_")[0].isdigit():
                parts = class_name.split("_", 1)
                fdi = parts[0]
                original_class = parts[1]
            elif class_name.isdigit():
                fdi = class_name

            tooth_pts = []
            if masks is not None and masks.xy is not None and len(masks.xy) > i:
                tooth_pts = masks.xy[i].tolist()

            detection_id = f"det_{i}"
            temp_detections.append(ToothDetectionResult(
                detection_id=detection_id,
                fdi=fdi,
                bbox=BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
                confidence=conf,
                original_class=original_class,
                tooth_pts=tooth_pts
            ))

        # Filter duplicates by same FDI + strong mask/bbox overlap.
        kept = []
        for m in temp_detections:
            duplicate = False
            for idx, k in enumerate(kept):
                if m.fdi != k.fdi:
                    continue

                box_iou = self.compute_iou(
                    [m.bbox.x1, m.bbox.y1, m.bbox.x2, m.bbox.y2],
                    [k.bbox.x1, k.bbox.y1, k.bbox.x2, k.bbox.y2],
                )
                mask_overlap = self.compute_mask_overlap_score(m.tooth_pts, k.tooth_pts)

                if box_iou > 0.5 or mask_overlap > 0.35:
                    duplicate = True
                    if m.confidence > k.confidence:
                        kept[idx] = m
                    break

            if not duplicate:
                kept.append(m)

        return kept
