import os
import cv2
import numpy as np

from schemas.inference import ToothDetectionResult, PipelineError

CROP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Tooth_seg_crop_20250424.pth")

class ToothRecognitionService:
    def __init__(self):
        self.predictor = None

    def load_model(self):
        if not os.path.exists(CROP_MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {CROP_MODEL_PATH}")
            
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
        except ImportError:
            raise ImportError(
                "detectron2 is not installed. Caries segmentation requires detectron2."
            )
            
        if self.predictor is None:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.WEIGHTS = CROP_MODEL_PATH
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.predictor = DefaultPredictor(cfg)

    def segment_caries(self, image_path: str, detections: list[ToothDetectionResult]) -> list[ToothDetectionResult]:
        try:
            self.load_model()
        except ImportError as e:
            # We raise a specialized exception so the pipeline orchestrator can handle it
            raise RuntimeError(f"MODEL_DEPENDENCY_MISSING: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"MODEL_ARTIFACT_MISSING: {str(e)}")
            
        img = cv2.imread(image_path)
        PAD = 20
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
            
            # Crop image around tooth with padding
            cx1 = max(x1 - PAD, 0)
            cy1 = max(y1 - PAD, 0)
            cx2 = min(x2 + PAD, img.shape[1])
            cy2 = min(y2 + PAD, img.shape[0])
            
            cropped_img = img[cy1:cy2, cx1:cx2]
            
            detectron_output = self.predictor(cropped_img)
            
            # Extract masks from detectron_output
            instances = detectron_output["instances"]
            if len(instances) > 0:
                # Find the mask with the highest score
                best_idx = instances.scores.argmax().item()
                mask = instances.pred_masks[best_idx].cpu().numpy()
                
                # Convert boolean mask to polygons
                # mask is HxW boolean array
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Take the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Shift contour points back to global image coordinates
                    shifted_contour = []
                    for pt in largest_contour:
                        px, py = pt[0]
                        shifted_contour.append([float(px + cx1), float(py + cy1)])
                        
                    det.caries_pts = shifted_contour
                else:
                    det.caries_pts = []
            else:
                det.caries_pts = []
                
        return detections
