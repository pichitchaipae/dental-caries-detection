"""
Two-Stage Dental Caries Detection System with Instance Segmentation (v3.0)
================================================================================

Pipeline:
    Stage 1: YOLO Caries Detection (yolov8s, 3-class: Occlusal/Proximal/Lingual)
    Stage 2: Hybrid Tooth Localization with Instance Segmentation
        Stage 2a: YOLO panoramic detection (bounding boxes + coarse masks)
        Stage 2b: Crop individual tooth regions with padding
        Stage 2c: Detectron2 fine segmentation (real tooth contours)
        Stage 2d: Map local coordinates back to global panoramic space

Output Compatibility:
    - Exports tooth segmentation masks as pixel_coordinates [[x,y], ...]
    - Compatible with Week 2/3 visualization tools
    - Readable class names (Occlusal, Proximal, Lingual)

Caries Classes:
    0: Occlusal  - Caries on the chewing surface
    1: Proximal  - Caries on the side surfaces
    2: Lingual   - Caries on the tongue-facing surface

Author: Lead CV Engineer
Date: 2025-01-27
Version: 3.0 (Instance Segmentation Integration)
"""

import os
import sys
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

import cv2
import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Install with: pip install ultralytics")
    sys.exit(1)

# Detectron2 for fine-grained instance segmentation
DETECTRON2_AVAILABLE = False
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    print("⚠ Detectron2 not available. Fine segmentation will use YOLO masks as fallback.")


# =============================================================================
# CONFIGURATION
# =============================================================================

class LocalizationMethod(Enum):
    """Enum for tooth localization methods."""
    DYNAMIC_MODEL = "dynamic"  # Use YOLO tooth detection model
    STATIC_MAP = "static"      # Use coordinate mapping fallback


# Caries class mapping
CARIES_CLASS_NAMES: Dict[int, str] = {
    0: 'Occlusal',
    1: 'Proximal',
    2: 'Lingual'
}

# Visualization colors (BGR format for OpenCV)
CARIES_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),     # Green for Occlusal
    1: (0, 165, 255),   # Orange for Proximal
    2: (255, 0, 0)      # Blue for Lingual
}

TOOTH_BOX_COLOR = (255, 255, 0)  # Cyan for tooth boxes
CARIES_BOX_COLOR = (0, 0, 255)   # Red for caries boxes

# Default paths
DEFAULT_TOOTH_MAP_PATH = Path(__file__).parent / 'tooth_map_config.json'
DEFAULT_TOOTH_MODEL_PATH = Path(__file__).parent.parent / 'material' / 'Tooth Segmentation + Recognition model' / 'weights' / 'Tooth_seg_pano_20250319.pt'
# Detectron2 crop segmentation model path
DEFAULT_CROP_MODEL_PATH = Path(__file__).parent.parent / 'material' / 'Tooth Segmentation + Recognition model' / 'weights' / 'Tooth_seg_crop_20250424.pth'

# Crop padding for fine segmentation
CROP_PAD = 20  # Pixels to pad around tooth bounding box


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ToothDetection:
    """Data class for a single tooth detection."""
    tooth_id: str                    # FDI notation (e.g., "11", "46")
    tooth_name: str                  # Full name (e.g., "Upper Right Central Incisor")
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coordinates
    center: Tuple[float, float]      # (cx, cy) normalized coordinates
    mask: Optional[np.ndarray] = None
    # NEW: Real segmentation contour in global panoramic coordinates [[x,y], ...]
    pixel_coordinates: Optional[List[List[int]]] = None


@dataclass
class CariesDetection:
    """Data class to store a single caries detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    center: Tuple[float, float]       # (cx, cy) normalized coordinates
    center_pixel: Tuple[int, int]     # (cx, cy) pixel coordinates
    tooth_id: Optional[str] = None
    tooth_name: Optional[str] = None
    localization_method: str = "unknown"
    iou_with_tooth: float = 0.0


@dataclass
class InferenceResult:
    """Data class to store complete inference results."""
    image_path: str
    image_size: Tuple[int, int]  # (width, height)
    caries_detections: List[CariesDetection]
    tooth_detections: List[ToothDetection]
    processing_time_ms: float
    localization_method: str
    annotated_image: Optional[np.ndarray] = None


# =============================================================================
# FDI TOOTH NAMING
# =============================================================================

FDI_TOOTH_NAMES: Dict[str, str] = {
    # Quadrant 1 - Upper Right
    "11": "Upper Right Central Incisor",
    "12": "Upper Right Lateral Incisor",
    "13": "Upper Right Canine",
    "14": "Upper Right First Premolar",
    "15": "Upper Right Second Premolar",
    "16": "Upper Right First Molar",
    "17": "Upper Right Second Molar",
    "18": "Upper Right Third Molar",
    # Quadrant 2 - Upper Left
    "21": "Upper Left Central Incisor",
    "22": "Upper Left Lateral Incisor",
    "23": "Upper Left Canine",
    "24": "Upper Left First Premolar",
    "25": "Upper Left Second Premolar",
    "26": "Upper Left First Molar",
    "27": "Upper Left Second Molar",
    "28": "Upper Left Third Molar",
    # Quadrant 3 - Lower Left
    "31": "Lower Left Central Incisor",
    "32": "Lower Left Lateral Incisor",
    "33": "Lower Left Canine",
    "34": "Lower Left First Premolar",
    "35": "Lower Left Second Premolar",
    "36": "Lower Left First Molar",
    "37": "Lower Left Second Molar",
    "38": "Lower Left Third Molar",
    # Quadrant 4 - Lower Right
    "41": "Lower Right Central Incisor",
    "42": "Lower Right Lateral Incisor",
    "43": "Lower Right Canine",
    "44": "Lower Right First Premolar",
    "45": "Lower Right Second Premolar",
    "46": "Lower Right First Molar",
    "47": "Lower Right Second Molar",
    "48": "Lower Right Third Molar",
}


def get_tooth_name(tooth_id: str) -> str:
    """Get the full tooth name from FDI number."""
    return FDI_TOOTH_NAMES.get(str(tooth_id), f"Tooth {tooth_id}")


# =============================================================================
# GEOMETRIC UTILITIES
# =============================================================================

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box_a: First box (x1, y1, x2, y2)
        box_b: Second box (x1, y1, x2, y2)
    
    Returns:
        IoU value between 0 and 1
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)
    
    return inter_area / union_area if union_area > 0 else 0


def point_in_box(point: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        box: (x1, y1, x2, y2) bounding box
    
    Returns:
        True if point is inside box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_euclidean_distance(
    point1: Tuple[float, float], 
    point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get the center point of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# =============================================================================
# STAGE 2A: DYNAMIC TOOTH LOCALIZATION (PRIMARY)
# =============================================================================

def load_tooth_model(model_path: str) -> Optional[YOLO]:
    """
    Load the YOLO tooth detection model.
    
    Args:
        model_path: Path to the tooth detection model weights
    
    Returns:
        Loaded YOLO model or None if loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"Warning: Tooth model not found at {model_path}")
        return None
    
    try:
        print(f"Loading tooth detection model from: {model_path}")
        model = YOLO(str(model_path))
        print(f"Tooth model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading tooth model: {e}")
        return None


# =============================================================================
# STAGE 2B/2C: DETECTRON2 FINE SEGMENTATION
# =============================================================================

def load_detectron2_predictor(model_path: str, score_thresh: float = 0.5) -> Optional[Any]:
    """
    Load Detectron2 predictor for fine-grained tooth segmentation.
    
    Args:
        model_path: Path to the crop segmentation model weights (.pth)
        score_thresh: Confidence threshold for predictions
    
    Returns:
        DefaultPredictor instance or None if loading fails
    """
    if not DETECTRON2_AVAILABLE:
        print("⚠ Detectron2 not available, skipping fine segmentation model")
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"⚠ Detectron2 crop model not found at {model_path}")
        return None
    
    try:
        print(f"Loading Detectron2 crop segmentation model from: {model_path}")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class: tooth
        cfg.MODEL.WEIGHTS = str(model_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = "cuda"
        
        predictor = DefaultPredictor(cfg)
        print("✓ Detectron2 predictor loaded successfully")
        return predictor
    except Exception as e:
        print(f"⚠ Error loading Detectron2 predictor: {e}")
        return None


def crop_and_segment_tooth(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    predictor: Any,
    pad: int = CROP_PAD
) -> Optional[List[List[int]]]:
    """
    Crop a tooth region and run fine segmentation with Detectron2.
    
    Args:
        image: Full panoramic image (BGR)
        bbox: Tooth bounding box (x1, y1, x2, y2)
        predictor: Detectron2 DefaultPredictor
        pad: Padding around bounding box
    
    Returns:
        List of [x, y] points representing the tooth contour in GLOBAL coordinates,
        or None if segmentation fails
    """
    if predictor is None:
        return None
    
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Apply padding with bounds checking
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(w, x2 + pad)
    crop_y2 = min(h, y2 + pad)
    
    # Crop the tooth region
    cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if cropped_img.size == 0:
        return None
    
    try:
        # Run Detectron2 inference on the crop
        outputs = predictor(cropped_img)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) == 0:
            return None
        
        # Get the best mask (highest score)
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        best_idx = scores.argmax()
        best_mask = masks[best_idx]
        
        # Convert mask to contour
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Map local contour coordinates to global panoramic space
        global_contour = []
        for point in largest_contour:
            local_x, local_y = point[0]
            global_x = int(local_x + crop_x1)
            global_y = int(local_y + crop_y1)
            global_contour.append([global_x, global_y])
        
        return global_contour
        
    except Exception as e:
        # Silently fail for individual teeth - will use fallback
        return None


def bbox_to_polygon(bbox: Tuple[int, int, int, int]) -> List[List[int]]:
    """
    Convert bounding box to polygon coordinates (fallback when segmentation fails).
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box
    
    Returns:
        List of [x, y] corner points
    """
    x1, y1, x2, y2 = bbox
    return [
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2],  # Bottom-left
    ]


def detect_teeth(
    model: YOLO,
    image: np.ndarray,
    confidence_threshold: float = 0.5,
    detectron_predictor: Optional[Any] = None
) -> List[ToothDetection]:
    """
    Run YOLO tooth detection on an image with optional Detectron2 fine segmentation.
    
    Args:
        model: Loaded YOLO tooth model
        image: Input image (BGR format)
        confidence_threshold: Minimum confidence for detections
        detectron_predictor: Optional Detectron2 predictor for fine segmentation
    
    Returns:
        List of ToothDetection objects with pixel_coordinates (real contours)
    """
    height, width = image.shape[:2]
    
    # Run inference
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )
    
    detections = []
    
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes
            
            # Filter duplicates by IoU (same as week2 logic)
            temp_detections = []
            
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                tooth_id = result.names[class_id]  # FDI notation from model
                
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Normalized center
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                
                temp_detections.append({
                    'tooth_id': str(tooth_id),
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
            
            # Filter duplicates by IoU
            kept = []
            for det in temp_detections:
                duplicate = False
                for k in kept:
                    iou = compute_iou(det['bbox'], k['bbox'])
                    if iou > 0.5:
                        if det['confidence'] > k['confidence']:
                            kept.remove(k)
                            kept.append(det)
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(det)
            
            # Convert to ToothDetection objects with fine segmentation
            segmented_count = 0
            for det in kept:
                # Try Detectron2 fine segmentation
                pixel_coords = None
                if detectron_predictor is not None:
                    pixel_coords = crop_and_segment_tooth(
                        image=image,
                        bbox=det['bbox'],
                        predictor=detectron_predictor
                    )
                    if pixel_coords:
                        segmented_count += 1
                
                # Fallback to bbox polygon if segmentation failed
                if pixel_coords is None:
                    pixel_coords = bbox_to_polygon(det['bbox'])
                
                detections.append(ToothDetection(
                    tooth_id=det['tooth_id'],
                    tooth_name=get_tooth_name(det['tooth_id']),
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    center=det['center'],
                    pixel_coordinates=pixel_coords
                ))
            
            if detectron_predictor is not None:
                print(f"  → Fine segmentation: {segmented_count}/{len(detections)} teeth")
    
    return detections


def find_tooth_for_caries_dynamic(
    caries_center_pixel: Tuple[int, int],
    caries_bbox: Tuple[int, int, int, int],
    tooth_detections: List[ToothDetection]
) -> Tuple[Optional[str], Optional[str], float, str]:
    """
    Find the tooth containing a caries detection using dynamic model results.
    
    Logic:
        1. First check if caries center is INSIDE any tooth bounding box
        2. If multiple matches, select the one with highest IoU
        3. If no direct hit, find the nearest tooth by center distance
    
    Args:
        caries_center_pixel: (x, y) pixel coordinates of caries center
        caries_bbox: Caries bounding box
        tooth_detections: List of detected teeth
    
    Returns:
        Tuple of (tooth_id, tooth_name, iou_or_distance, match_type)
    """
    if not tooth_detections:
        return None, None, 0.0, "no_teeth"
    
    # Strategy 1: Check if caries center is inside a tooth box
    containing_teeth = []
    for tooth in tooth_detections:
        if point_in_box(caries_center_pixel, tooth.bbox):
            iou = compute_iou(caries_bbox, tooth.bbox)
            containing_teeth.append((tooth, iou))
    
    if containing_teeth:
        # Sort by IoU and select best match
        containing_teeth.sort(key=lambda x: x[1], reverse=True)
        best_tooth, best_iou = containing_teeth[0]
        return best_tooth.tooth_id, best_tooth.tooth_name, best_iou, "box_containment"
    
    # Strategy 2: Find tooth with highest IoU overlap
    best_iou = 0.0
    best_tooth = None
    for tooth in tooth_detections:
        iou = compute_iou(caries_bbox, tooth.bbox)
        if iou > best_iou:
            best_iou = iou
            best_tooth = tooth
    
    if best_tooth and best_iou > 0.1:  # Minimum IoU threshold
        return best_tooth.tooth_id, best_tooth.tooth_name, best_iou, "iou_overlap"
    
    # Strategy 3: Fallback to nearest tooth by center distance
    min_distance = float('inf')
    nearest_tooth = None
    
    for tooth in tooth_detections:
        tooth_center = get_box_center(tooth.bbox)
        distance = calculate_euclidean_distance(caries_center_pixel, tooth_center)
        if distance < min_distance:
            min_distance = distance
            nearest_tooth = tooth
    
    if nearest_tooth:
        return nearest_tooth.tooth_id, nearest_tooth.tooth_name, min_distance, "nearest_center"
    
    return None, None, 0.0, "no_match"


# =============================================================================
# STAGE 2B: STATIC COORDINATE MAP (FALLBACK)
# =============================================================================

def load_tooth_map(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load tooth coordinate mapping from JSON configuration file.
    
    Args:
        config_path: Path to the tooth_map_config.json file
    
    Returns:
        Dictionary mapping FDI tooth numbers to their coordinates
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Tooth map config not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        tooth_coords = config.get('tooth_coordinates', {})
        tooth_map = {
            tooth_id: data 
            for tooth_id, data in tooth_coords.items() 
            if not tooth_id.startswith('_')
        }
        
        print(f"Loaded static tooth map with {len(tooth_map)} positions")
        return tooth_map
    except Exception as e:
        print(f"Error loading tooth map: {e}")
        return {}


def find_tooth_for_caries_static(
    caries_center_normalized: Tuple[float, float],
    tooth_map: Dict[str, Dict[str, Any]],
    max_distance: float = 0.15
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Find the nearest tooth using static coordinate mapping (fallback).
    
    Args:
        caries_center_normalized: Normalized (x, y) coordinates
        tooth_map: Static tooth coordinate mapping
        max_distance: Maximum allowed distance for valid match
    
    Returns:
        Tuple of (tooth_id, tooth_name, distance)
    """
    if not tooth_map:
        return None, None, float('inf')
    
    min_distance = float('inf')
    nearest_tooth_id = None
    nearest_tooth_name = None
    
    for tooth_id, tooth_data in tooth_map.items():
        tooth_center = (
            tooth_data.get('center_x', 0),
            tooth_data.get('center_y', 0)
        )
        
        distance = calculate_euclidean_distance(caries_center_normalized, tooth_center)
        
        if distance < min_distance:
            min_distance = distance
            nearest_tooth_id = tooth_id
            nearest_tooth_name = tooth_data.get('tooth_name', get_tooth_name(tooth_id))
    
    return nearest_tooth_id, nearest_tooth_name, min_distance


# =============================================================================
# STAGE 1: CARIES DETECTION
# =============================================================================

def load_caries_model(model_path: str) -> YOLO:
    """
    Load the YOLO caries detection model.
    
    Args:
        model_path: Path to the caries detection model weights
    
    Returns:
        Loaded YOLO model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Caries model not found: {model_path}")
    
    print(f"Loading caries detection model from: {model_path}")
    model = YOLO(str(model_path))
    print(f"Caries model loaded successfully")
    
    return model


def detect_caries(
    model: YOLO,
    image: np.ndarray,
    confidence_threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Run YOLO caries detection on an image.
    
    Args:
        model: Loaded YOLO caries model
        image: Input image (BGR format)
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of raw detection dictionaries
    """
    height, width = image.shape[:2]
    
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )
    
    detections = []
    
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Calculate centers
                center_x_pixel = (x1 + x2) // 2
                center_y_pixel = (y1 + y2) // 2
                center_x_norm = (x1 + x2) / 2 / width
                center_y_norm = (y1 + y2) / 2 / height
                
                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center_pixel': (center_x_pixel, center_y_pixel),
                    'center_normalized': (center_x_norm, center_y_norm)
                })
    
    return detections


# =============================================================================
# MAIN INFERENCE PIPELINE
# =============================================================================

def run_inference(
    image_path: str,
    caries_model: YOLO,
    tooth_model: Optional[YOLO] = None,
    tooth_map: Optional[Dict[str, Dict[str, Any]]] = None,
    detectron_predictor: Optional[Any] = None,
    caries_confidence: float = 0.25,
    tooth_confidence: float = 0.5,
    visualize: bool = True
) -> InferenceResult:
    """
    Run the complete inference pipeline with optional instance segmentation.
    
    Stage 1: YOLO caries detection
    Stage 2: Hybrid Tooth Localization
        Stage 2a: YOLO panoramic detection (bounding boxes)
        Stage 2b: Crop tooth regions
        Stage 2c: Detectron2 fine segmentation (if available)
        Stage 2d: Map coordinates to global space
    
    Args:
        image_path: Path to input image
        caries_model: Loaded YOLO caries model
        tooth_model: Optional YOLO tooth model (for dynamic localization)
        tooth_map: Optional static tooth coordinate map (fallback)
        detectron_predictor: Optional Detectron2 predictor for fine segmentation
        caries_confidence: Confidence threshold for caries detection
        tooth_confidence: Confidence threshold for tooth detection
        visualize: Whether to create annotated visualization
    
    Returns:
        InferenceResult object
    """
    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    height, width = image.shape[:2]
    
    print(f"\n{'='*70}")
    print(f"PROCESSING: {image_path.name}")
    print(f"Image size: {width}x{height}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # =========================================================================
    # STAGE 1: Caries Detection
    # =========================================================================
    print("\n[STAGE 1] Detecting caries lesions...")
    raw_caries = detect_caries(
        model=caries_model,
        image=image,
        confidence_threshold=caries_confidence
    )
    print(f"  → Found {len(raw_caries)} caries detection(s)")
    
    # =========================================================================
    # STAGE 2: Tooth Localization with Instance Segmentation
    # =========================================================================
    tooth_detections = []
    localization_method = "none"
    
    # Try dynamic model first
    if tooth_model is not None:
        seg_mode = "with Detectron2 fine segmentation" if detectron_predictor else "bounding boxes only"
        print(f"\n[STAGE 2] Running tooth detection ({seg_mode})...")
        tooth_detections = detect_teeth(
            model=tooth_model,
            image=image,
            confidence_threshold=tooth_confidence,
            detectron_predictor=detectron_predictor
        )
        print(f"  → Found {len(tooth_detections)} teeth")
        
        if tooth_detections:
            localization_method = "dynamic_model"
        else:
            print("  ⚠ No teeth detected, falling back to static map...")
    
    # Fallback to static map if needed
    if not tooth_detections and tooth_map:
        print("\n[STAGE 2] Using static coordinate map (FALLBACK)...")
        localization_method = "static_map"
    
    if not tooth_detections and not tooth_map:
        print("\n  ⚠ WARNING: No tooth localization method available!")
        localization_method = "none"
    
    # =========================================================================
    # Match Caries to Teeth
    # =========================================================================
    print("\n[MATCHING] Assigning caries to teeth...")
    caries_detections = []
    
    for i, caries in enumerate(raw_caries):
        class_id = caries['class_id']
        class_name = CARIES_CLASS_NAMES.get(class_id, f'Class_{class_id}')
        
        tooth_id = None
        tooth_name = None
        iou_or_distance = 0.0
        match_method = "none"
        
        if localization_method == "dynamic_model" and tooth_detections:
            # Use dynamic model matching
            tooth_id, tooth_name, iou_or_distance, match_method = find_tooth_for_caries_dynamic(
                caries_center_pixel=caries['center_pixel'],
                caries_bbox=caries['bbox'],
                tooth_detections=tooth_detections
            )
        elif localization_method == "static_map" and tooth_map:
            # Use static map matching
            tooth_id, tooth_name, iou_or_distance = find_tooth_for_caries_static(
                caries_center_normalized=caries['center_normalized'],
                tooth_map=tooth_map
            )
            match_method = "static_map"
        
        detection = CariesDetection(
            class_id=class_id,
            class_name=class_name,
            confidence=caries['confidence'],
            bbox=caries['bbox'],
            center=caries['center_normalized'],
            center_pixel=caries['center_pixel'],
            tooth_id=tooth_id,
            tooth_name=tooth_name,
            localization_method=match_method,
            iou_with_tooth=iou_or_distance if match_method in ["box_containment", "iou_overlap"] else 0.0
        )
        caries_detections.append(detection)
        
        # Print detection
        print(f"\n  Detection {i+1}:")
        print(f"    └─ Caries Type:  {class_name} ({caries['confidence']:.1%})")
        print(f"    └─ Tooth ID:     {tooth_id or 'Unknown'}")
        print(f"    └─ Match Method: {match_method}")
        if tooth_name:
            print(f"    └─ Tooth Name:   {tooth_name}")
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000
    
    # Create visualization
    annotated_image = None
    if visualize:
        annotated_image = visualize_detections(
            image=image.copy(),
            caries_detections=caries_detections,
            tooth_detections=tooth_detections,
            show_teeth=True
        )
    
    result = InferenceResult(
        image_path=str(image_path),
        image_size=(width, height),
        caries_detections=caries_detections,
        tooth_detections=tooth_detections,
        processing_time_ms=processing_time,
        localization_method=localization_method,
        annotated_image=annotated_image
    )
    
    print(f"\n{'='*70}")
    print(f"Processing completed in {processing_time:.2f} ms")
    print(f"Localization method: {localization_method.upper()}")
    print(f"{'='*70}")
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_detections(
    image: np.ndarray,
    caries_detections: List[CariesDetection],
    tooth_detections: List[ToothDetection],
    show_teeth: bool = True,
    font_scale: float = 0.6,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection results on the image.
    
    Args:
        image: Input image (will be modified)
        caries_detections: List of CariesDetection objects
        tooth_detections: List of ToothDetection objects
        show_teeth: Whether to draw tooth bounding boxes
        font_scale: Font scale for labels
        thickness: Line thickness
    
    Returns:
        Annotated image
    """
    # Draw tooth boxes first (background)
    if show_teeth and tooth_detections:
        for tooth in tooth_detections:
            x1, y1, x2, y2 = tooth.bbox
            # Draw semi-transparent tooth box
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), TOOTH_BOX_COLOR, 1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            
            # Draw tooth ID label (smaller, at bottom)
            label = f"T{tooth.tooth_id}"
            cv2.putText(
                image, label, (x1 + 2, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TOOTH_BOX_COLOR, 1
            )
    
    # Draw caries boxes (foreground)
    for det in caries_detections:
        color = CARIES_COLORS.get(det.class_id, CARIES_BOX_COLOR)
        x1, y1, x2, y2 = det.bbox
        
        # Draw caries bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if det.tooth_id:
            label = f"Tooth {det.tooth_id}: {det.class_name} ({det.confidence:.0%})"
        else:
            label = f"{det.class_name} ({det.confidence:.0%})"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        label_y = max(y1 - 10, label_size[1] + 10)
        
        cv2.rectangle(
            image,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 5, label_y + 5),
            color, -1
        )
        
        # Draw label text
        cv2.putText(
            image, label, (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
        
        # Draw center marker
        cv2.circle(image, det.center_pixel, 4, color, -1)
    
    return image


def save_results(
    result: InferenceResult,
    output_dir: str,
    save_image: bool = True,
    save_json: bool = True
) -> None:
    """Save inference results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(result.image_path).stem
    
    # Save annotated image
    if save_image and result.annotated_image is not None:
        image_output = output_path / f"{image_name}_annotated.jpg"
        cv2.imwrite(str(image_output), result.annotated_image)
        print(f"Saved: {image_output}")
    
    # Save JSON results
    if save_json:
        json_output = output_path / f"{image_name}_results.json"
        
        results_dict = {
            'image_path': result.image_path,
            'image_size': {'width': result.image_size[0], 'height': result.image_size[1]},
            'processing_time_ms': result.processing_time_ms,
            'localization_method': result.localization_method,
            'num_caries': len(result.caries_detections),
            'num_teeth_detected': len(result.tooth_detections),
            'caries_detections': [],
            'tooth_detections': []
        }
        
        for det in result.caries_detections:
            results_dict['caries_detections'].append({
                'class_id': det.class_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': {'x1': det.bbox[0], 'y1': det.bbox[1], 'x2': det.bbox[2], 'y2': det.bbox[3]},
                'center': {'x': det.center[0], 'y': det.center[1]},
                'tooth_id': det.tooth_id,
                'tooth_name': det.tooth_name,
                'localization_method': det.localization_method,
                'iou_with_tooth': det.iou_with_tooth
            })
        
        for tooth in result.tooth_detections:
            tooth_data = {
                'tooth_id': tooth.tooth_id,
                'tooth_name': tooth.tooth_name,
                'confidence': tooth.confidence,
                'bbox': {'x1': tooth.bbox[0], 'y1': tooth.bbox[1], 'x2': tooth.bbox[2], 'y2': tooth.bbox[3]},
                # NEW: Real tooth contour in global coordinates
                'pixel_coordinates': tooth.pixel_coordinates if tooth.pixel_coordinates else []
            }
            results_dict['tooth_detections'].append(tooth_data)
        
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Saved: {json_output}")


def print_summary(result: InferenceResult) -> None:
    """Print formatted detection summary."""
    print(f"\n{'='*70}")
    print("DETECTION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nLocalization Method: {result.localization_method.upper()}")
    print(f"Teeth Detected: {len(result.tooth_detections)}")
    print(f"Caries Detected: {len(result.caries_detections)}")
    
    if not result.caries_detections:
        print("\n✓ No caries detected in this image.")
        return
    
    print(f"\n{'─'*70}")
    print("CARIES FINDINGS:")
    print(f"{'─'*70}")
    
    for i, det in enumerate(result.caries_detections, 1):
        tooth_info = f"Tooth {det.tooth_id}" if det.tooth_id else "Unknown Location"
        print(f"\n  [{i}] {tooth_info} : {det.class_name} Caries detected")
        print(f"      Confidence: {det.confidence:.1%}")
        if det.tooth_name:
            print(f"      Full Name:  {det.tooth_name}")
        print(f"      Method:     {det.localization_method}")
    
    print(f"\n{'='*70}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the dental caries detection system with instance segmentation."""
    parser = argparse.ArgumentParser(
        description='Dental Caries Detection System (v3.0 - Instance Segmentation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with both models (with Detectron2 fine segmentation)
  python inference.py -i ./test.jpg -m ./caries_model.pt -t ./tooth_model.pt
  
  # Skip fine segmentation (faster, bounding boxes only)
  python inference.py -i ./test.jpg -m ./caries_model.pt -t ./tooth_model.pt --no_fine_seg
  
  # Batch processing
  python inference.py --image_dir ./images -m ./caries_model.pt -t ./tooth_model.pt

Pipeline (v3.0):
  Stage 1:   YOLO Caries Detection (3-class)
  Stage 2a:  YOLO Panoramic Tooth Detection
  Stage 2b:  Crop tooth regions with padding
  Stage 2c:  Detectron2 fine segmentation (real contours)
  Stage 2d:  Map local coordinates to global space

Caries Classes:
  0: Occlusal  - Caries on chewing surface
  1: Proximal  - Caries on side surfaces  
  2: Lingual   - Caries on tongue-facing surface
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--image_path', '-i',
        type=str,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing images for batch processing'
    )
    
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        required=True,
        help='Path to YOLO caries detection model (.pt)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--tooth_model', '-t',
        type=str,
        default=str(DEFAULT_TOOTH_MODEL_PATH),
        help=f'Path to YOLO tooth detection model (default: {DEFAULT_TOOTH_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--tooth_map',
        type=str,
        default=str(DEFAULT_TOOTH_MAP_PATH),
        help='Path to static tooth map JSON (fallback)'
    )
    
    parser.add_argument(
        '--use_static_only',
        action='store_true',
        help='Use only static coordinate map (skip dynamic model)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./inference_output',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--caries_conf', '-c',
        type=float,
        default=0.05,
        help='Confidence threshold for caries detection (default: 0.05)'
    )
    
    parser.add_argument(
        '--tooth_conf',
        type=float,
        default=0.5,
        help='Confidence threshold for tooth detection (default: 0.5)'
    )
    
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Disable visualization'
    )
    
    parser.add_argument(
        '--no_fine_seg',
        action='store_true',
        help='Skip Detectron2 fine segmentation (faster, uses bounding boxes only)'
    )
    
    parser.add_argument(
        '--crop_model',
        type=str,
        default=str(DEFAULT_CROP_MODEL_PATH),
        help=f'Path to Detectron2 crop segmentation model (default: {DEFAULT_CROP_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results in window'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        parser.error("Either --image_path or --image_dir must be provided")
    
    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n" + "="*70)
    print("DENTAL CARIES DETECTION SYSTEM v3.0 (Instance Segmentation)")
    print("="*70)
    
    # Load caries model (required)
    try:
        caries_model = load_caries_model(args.model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load tooth model (optional, for dynamic localization)
    tooth_model = None
    if not args.use_static_only:
        tooth_model = load_tooth_model(args.tooth_model)
        if tooth_model is None:
            print("⚠ Tooth model not loaded, will use static map fallback")
    else:
        print("ℹ Static-only mode: Skipping dynamic tooth model")
    
    # Load Detectron2 predictor for fine segmentation (optional)
    detectron_predictor = None
    if not args.no_fine_seg and DETECTRON2_AVAILABLE:
        detectron_predictor = load_detectron2_predictor(args.crop_model)
        if detectron_predictor is None:
            print("⚠ Detectron2 predictor not loaded, will use bbox fallback")
    elif args.no_fine_seg:
        print("ℹ Fine segmentation disabled, using bounding boxes only")
    
    # Load static tooth map (fallback)
    tooth_map = load_tooth_map(args.tooth_map)
    
    # Report configuration
    print(f"\n{'─'*70}")
    print("CONFIGURATION:")
    print(f"  Caries Model:     {args.model_path}")
    print(f"  Tooth Model:      {'Loaded' if tooth_model else 'Not available'}")
    print(f"  Fine Segmentation: {'Detectron2 Loaded' if detectron_predictor else 'Disabled (bbox fallback)'}")
    print(f"  Static Map:       {'Loaded' if tooth_map else 'Not available'}")
    print(f"  Strategy:         {'Dynamic Model (Primary)' if tooth_model else 'Static Map (Fallback)'}")
    print(f"{'─'*70}")
    
    # Collect images
    image_paths = []
    if args.image_path:
        image_paths.append(args.image_path)
    if args.image_dir:
        image_dir = Path(args.image_dir)
        if image_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_paths.extend(image_dir.glob(ext))
                image_paths.extend(image_dir.glob(ext.upper()))
    
    if not image_paths:
        print("Error: No images found")
        sys.exit(1)
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    # Process images
    all_results = []
    
    for image_path in image_paths:
        try:
            result = run_inference(
                image_path=str(image_path),
                caries_model=caries_model,
                tooth_model=tooth_model,
                tooth_map=tooth_map,
                detectron_predictor=detectron_predictor,
                caries_confidence=args.caries_conf,
                tooth_confidence=args.tooth_conf,
                visualize=not args.no_visualize
            )
            
            print_summary(result)
            
            save_results(
                result=result,
                output_dir=args.output_dir,
                save_image=not args.no_visualize
            )
            
            if args.show and result.annotated_image is not None:
                cv2.imshow('Detection Results', result.annotated_image)
                cv2.waitKey(0)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if args.show:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Results saved to: {args.output_dir}")
    print(f"Total images processed: {len(all_results)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
