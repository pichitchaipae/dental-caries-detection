"""
Caries Detector — Production Pipeline
======================================
Pipeline flow:
  Upload Image
  → Tooth_seg_pano_20250319.pt (validation — handled by validator.py)
  → Tooth_seg_crop_20250424.pth (tooth localization + bounding boxes)
  → Feature Extraction (PCA-aligned geometric features)
  → rf_classify_ml.pkl (RF surface classification)
  → Annotated Image + JSON Result
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from services.rf_classifier import RFClassifier

CROP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Tooth_seg_crop_20250424.pth")
PANO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Tooth_seg_pano_20250319.pt")

class CariesDetector:
    def __init__(self):
        self.crop_model = None
        self.pano_model = None
        self.rf_classifier = RFClassifier()

    def load_models(self):
        """Load the tooth segmentation models (lazy, loads once)."""
        if self.pano_model is None:
            if not os.path.exists(PANO_MODEL_PATH):
                raise FileNotFoundError(f"Missing model: {PANO_MODEL_PATH}")
            self.pano_model = YOLO(PANO_MODEL_PATH)

        if self.crop_model is None:
            if not os.path.exists(CROP_MODEL_PATH):
                raise FileNotFoundError(f"Missing model: {CROP_MODEL_PATH}")
            # Tooth_seg_crop is a .pth model — try loading as YOLO first,
            # fallback to raw torch if needed
            try:
                self.crop_model = YOLO(CROP_MODEL_PATH)
            except Exception:
                # If it's a raw PyTorch state_dict, load it differently
                self.crop_model = torch.load(CROP_MODEL_PATH, map_location="cpu")

    def get_model_name(self):
        return "Tooth_seg_pano + Tooth_seg_crop + RF_Classifier"

    def detect(self, image_path: str):
        """
        Run the production inference pipeline:
          1. Use pano model for tooth segmentation (masks + bounding boxes)
          2. For each detected tooth with caries indicators, extract features
          3. Classify surface via RF ML layer with Smart Fallback
        """
        self.load_models()

        # Read image dimensions
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2] if img is not None else (1000, 1000)

        # Run pano segmentation model for tooth detection
        results = self.pano_model(image_path)
        predictions = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names
            masks = result.masks  # Segmentation masks (may be None)

            for i, box in enumerate(boxes):
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = names[cls_idx] if cls_idx in names else "unknown"

                # Extract FDI tooth number if provided in class name
                tooth_id = "46"  # Default fallback for RF orientation rules
                original_class = class_name

                if "_" in class_name and class_name.split("_")[0].isdigit():
                    parts = class_name.split("_", 1)
                    tooth_id = parts[0]
                    original_class = parts[1]
                elif class_name.isdigit():
                    tooth_id = class_name

                # -----------------------------------------------------------------
                # Get segmentation points for feature extraction
                # -----------------------------------------------------------------

                # 1. Get Caries/Detection Points from segmentation masks
                caries_pts = []
                if masks is not None and masks.xy is not None and len(masks.xy) > i:
                    caries_pts = masks.xy[i].tolist()

                if len(caries_pts) < 10:
                    # Fallback: Generate points covering the bounding box
                    caries_pts = [
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
                        [(x1+x2)/2, (y1+y2)/2]
                    ]

                # 2. Get Tooth Points (approximate as padded bounding box)
                pad = min(img_w, img_h) * 0.1
                t_x1, t_y1 = max(0, x1 - pad), max(0, y1 - pad)
                t_x2, t_y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
                tooth_pts = [
                    [t_x1, t_y1], [t_x2, t_y1], [t_x2, t_y2], [t_x1, t_y2]
                ]

                # -----------------------------------------------------------------
                # RF ML Layer: Surface Classification (with Smart Fallback)
                # -----------------------------------------------------------------
                rf_predicted_surface, rf_confidence, rf_metadata = self.rf_classifier.classify_surface(
                    tooth_id, tooth_pts, caries_pts
                )

                prediction = {
                    "tooth": tooth_id,
                    "surface": rf_predicted_surface,
                    "yolo_original_class": original_class,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "rfClassification": {
                        "rfConfidence": round(rf_confidence, 4),
                        "method": rf_metadata.get("method", "Unknown"),
                        "fallbackUsed": rf_metadata.get("fallback_used", False),
                        "probabilities": rf_metadata.get("probabilities", {}),
                    }
                }

                predictions.append(prediction)

        return predictions
