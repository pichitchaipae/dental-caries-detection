import os
import cv2
from ultralytics import YOLO

VALIDATOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Tooth_seg_pano_20250319.pt")

class PanoramicValidator:
    """
    Panoramic dental radiograph validator using Tooth_seg_pano model.
    
    Validation logic:
      - The pano model detects individual teeth (not a single panoramic ROI).
      - A valid panoramic radiograph should have MULTIPLE tooth detections
        whose combined bounding box covers a significant portion of the image.
      - We aggregate all tooth detections into a combined ROI for coverage.
    """

    def __init__(self):
        self.model = None

    def load_model(self):
        if not os.path.exists(VALIDATOR_MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {VALIDATOR_MODEL_PATH}")
        if self.model is None:
            self.model = YOLO(VALIDATOR_MODEL_PATH)

    def validate(self, image_path: str):
        self.load_model()
        
        # Run inference
        results = self.model(image_path)
        
        # Initialize validation metrics
        num_detections = 0
        max_confidence = 0.0
        mean_confidence = 0.0
        coverage_ratio = 0.0
        
        # Process results — aggregate all tooth detections into a combined ROI
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            # Collect all confidences
            confidences = [float(box.conf[0]) for box in boxes]
            max_confidence = max(confidences)
            mean_confidence = sum(confidences) / len(confidences)
            
            # Compute combined bounding box of ALL detections (the dental ROI)
            all_x1, all_y1, all_x2, all_y2 = [], [], [], []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_x1.append(x1)
                all_y1.append(y1)
                all_x2.append(x2)
                all_y2.append(y2)
            
            combined_x1 = min(all_x1)
            combined_y1 = min(all_y1)
            combined_x2 = max(all_x2)
            combined_y2 = max(all_y2)
            
            # Calculate coverage ratio using the combined ROI
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                img_area = h * w
                combined_area = (combined_x2 - combined_x1) * (combined_y2 - combined_y1)
                coverage_ratio = combined_area / img_area

        # ---------------------------------------------------------
        # Validation Rules
        # ---------------------------------------------------------
        
        # Rule 1: Must detect at least some teeth
        if num_detections == 0:
            return {
                "success": False,
                "reason": "Invalid panoramic dental radiograph: No dental structures detected.",
                "validation": {
                    "confidence": max_confidence,
                    "meanConfidence": mean_confidence,
                    "coverageRatio": coverage_ratio,
                    "numDetections": num_detections,
                }
            }
        
        # Rule 2: Best detection must have reasonable confidence
        if max_confidence < 0.30:
            return {
                "success": False,
                "reason": "Invalid panoramic dental radiograph: Detection confidence too low.",
                "validation": {
                    "confidence": max_confidence,
                    "meanConfidence": mean_confidence,
                    "coverageRatio": coverage_ratio,
                    "numDetections": num_detections,
                }
            }
        
        # Rule 3: Combined dental ROI must cover enough of the image
        # For a panoramic X-ray with tooth segmentation, the combined ROI
        # of all teeth should cover at least 10% of the image
        if coverage_ratio < 0.10:
            return {
                "success": False,
                "reason": "Invalid panoramic dental radiograph: Dental ROI coverage below threshold.",
                "validation": {
                    "confidence": max_confidence,
                    "meanConfidence": mean_confidence,
                    "coverageRatio": coverage_ratio,
                    "numDetections": num_detections,
                }
            }
            
        # Accept
        return {
            "success": True,
            "validation": {
                "confidence": max_confidence,
                "meanConfidence": mean_confidence,
                "coverageRatio": coverage_ratio,
                "numDetections": num_detections,
            }
        }
