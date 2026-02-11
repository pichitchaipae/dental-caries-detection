"""
Dental X-ray Dataset Preparation Script
========================================
Converts XML annotations (AIM format) to YOLO segmentation format
and organizes dataset into train/test splits.

Author: Senior Computer Vision Engineer
Project: Phase 1 - Dental Analysis

Input Structure:
    material/
    ├── 500 cases with annotation/
    │   ├── case 1/
    │   │   ├── case_1.png
    │   │   └── *.xml (tooth annotations)
    │   └── ...
    └── 500-roi/
        └── case_*.png (grayscale images)

Output Structure:
    week1/dataset/
    ├── images/
    │   ├── train/  (80%)
    │   └── val/    (20%)
    └── labels/
        ├── train/
        └── val/
"""

import os
import re
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FDI TOOTH MAPPING
# =============================================================================

# Mapping from tooth name patterns to FDI numbers
TOOTH_NAME_TO_FDI: Dict[str, int] = {
    # Upper Right Quadrant (Q1)
    "upper right third molar": 18,
    "upper right second molar": 17,
    "upper right first molar": 16,
    "upper right second premolar": 15,
    "upper right first premolar": 14,
    "upper right canine": 13,
    "upper right lateral incisor": 12,
    "upper right central incisor": 11,
    
    # Upper Left Quadrant (Q2)
    "upper left central incisor": 21,
    "upper left lateral incisor": 22,
    "upper left canine": 23,
    "upper left first premolar": 24,
    "upper left second premolar": 25,
    "upper left first molar": 26,
    "upper left second molar": 27,
    "upper left third molar": 28,
    
    # Lower Left Quadrant (Q3)
    "lower left central incisor": 31,
    "lower left lateral incisor": 32,
    "lower left canine": 33,
    "lower left first premolar": 34,
    "lower left second premolar": 35,
    "lower left first molar": 36,
    "lower left second molar": 37,
    "lower left third molar": 38,
    
    # Lower Right Quadrant (Q4)
    "lower right central incisor": 41,
    "lower right lateral incisor": 42,
    "lower right canine": 43,
    "lower right first premolar": 44,
    "lower right second premolar": 45,
    "lower right first molar": 46,
    "lower right second molar": 47,
    "lower right third molar": 48,
}

# FDI to YOLO class index mapping
# COMPATIBLE WITH PRE-TRAINED MODEL: Tooth_seg_pano_20250319.pt
# Sequential order within each quadrant: 11-18, 21-28, 31-38, 41-48
FDI_TO_CLASS_ID: Dict[int, int] = {
    # Upper Right (Q1): 11-18 -> classes 0-7
    11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7,
    # Upper Left (Q2): 21-28 -> classes 8-15
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    # Lower Left (Q3): 31-38 -> classes 16-23
    31: 16, 32: 17, 33: 18, 34: 19, 35: 20, 36: 21, 37: 22, 38: 23,
    # Lower Right (Q4): 41-48 -> classes 24-31
    41: 24, 42: 25, 43: 26, 44: 27, 45: 28, 46: 29, 47: 30, 48: 31,
}


@dataclass
class ToothAnnotation:
    """Represents a single tooth annotation with polygon coordinates."""
    fdi_number: int
    class_id: int
    polygon: List[Tuple[float, float]]  # Normalized coordinates
    tooth_name: str


def extract_tooth_name(xml_content: str) -> Optional[str]:
    """Extract tooth name from XML using regex pattern matching."""
    # Pattern to match tooth descriptions like "Permanent lower right second premolar tooth"
    pattern = r'displayName.*?value="Permanent\s+((?:upper|lower)\s+(?:right|left)\s+(?:third|second|first)?\s*(?:molar|premolar|canine|lateral\s+incisor|central\s+incisor))\s+tooth"'
    
    match = re.search(pattern, xml_content, re.IGNORECASE)
    if match:
        return match.group(1).lower().strip()
    return None


def extract_polygon_coordinates(xml_content: str) -> List[Tuple[float, float]]:
    """Extract polygon coordinates from XML."""
    coordinates = []
    
    # Pattern to match TwoDimensionSpatialCoordinate blocks
    coord_pattern = r'<TwoDimensionSpatialCoordinate>\s*<coordinateIndex[^>]*>\s*<x value="([^"]+)"/>\s*<y value="([^"]+)"/>\s*</TwoDimensionSpatialCoordinate>'
    
    matches = re.findall(coord_pattern, xml_content, re.DOTALL)
    
    for x_str, y_str in matches:
        try:
            x = float(x_str)
            y = float(y_str)
            coordinates.append((x, y))
        except ValueError:
            continue
    
    return coordinates


def normalize_polygon(polygon: List[Tuple[float, float]], 
                      img_width: int, img_height: int) -> List[Tuple[float, float]]:
    """Normalize polygon coordinates to 0-1 range."""
    normalized = []
    for x, y in polygon:
        norm_x = x / img_width
        norm_y = y / img_height
        # Clamp to valid range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        normalized.append((norm_x, norm_y))
    return normalized


def parse_xml_annotation(xml_path: Path, img_width: int, img_height: int) -> Optional[ToothAnnotation]:
    """Parse a single XML annotation file and return ToothAnnotation."""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract tooth name
        tooth_name = extract_tooth_name(content)
        if not tooth_name:
            logger.debug(f"No tooth name found in {xml_path.name}")
            return None
        
        # Get FDI number
        fdi_number = TOOTH_NAME_TO_FDI.get(tooth_name)
        if fdi_number is None:
            logger.warning(f"Unknown tooth: {tooth_name} in {xml_path.name}")
            return None
        
        # Get class ID
        class_id = FDI_TO_CLASS_ID.get(fdi_number)
        if class_id is None:
            logger.warning(f"No class mapping for FDI {fdi_number}")
            return None
        
        # Extract polygon
        polygon = extract_polygon_coordinates(content)
        if len(polygon) < 3:
            logger.debug(f"Insufficient polygon points ({len(polygon)}) in {xml_path.name}")
            return None
        
        # Normalize polygon
        normalized_polygon = normalize_polygon(polygon, img_width, img_height)
        
        return ToothAnnotation(
            fdi_number=fdi_number,
            class_id=class_id,
            polygon=normalized_polygon,
            tooth_name=tooth_name
        )
        
    except Exception as e:
        logger.error(f"Error parsing {xml_path}: {e}")
        return None


def polygon_to_yolo_format(class_id: int, polygon: List[Tuple[float, float]]) -> str:
    """Convert polygon to YOLO segmentation format string."""
    coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_id} {coords_str}"


def process_case(case_dir: Path, img_width: int = 3036, img_height: int = 1536) -> Tuple[Optional[Path], List[str]]:
    """
    Process a single case directory.
    
    Returns:
        Tuple of (image_path, list of YOLO annotation lines)
    """
    # Find PNG image
    png_files = list(case_dir.glob("*.png"))
    if not png_files:
        logger.warning(f"No PNG found in {case_dir}")
        return None, []
    
    image_path = png_files[0]
    
    # Get actual image dimensions
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        logger.warning(f"Could not read image {image_path}: {e}, using default dimensions")
    
    # Find and parse all XML annotations
    xml_files = list(case_dir.glob("*.xml"))
    annotations = []
    
    for xml_file in xml_files:
        annotation = parse_xml_annotation(xml_file, img_width, img_height)
        if annotation:
            yolo_line = polygon_to_yolo_format(annotation.class_id, annotation.polygon)
            annotations.append(yolo_line)
            logger.debug(f"  Found: {annotation.tooth_name} (FDI {annotation.fdi_number}, class {annotation.class_id})")
    
    return image_path, annotations


def create_directory_structure(base_path: Path):
    """Create the required YOLO directory structure."""
    dirs = [
        base_path / "images" / "train",
        base_path / "images" / "val",
        base_path / "labels" / "train",
        base_path / "labels" / "val",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {dir_path}")


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Main function to prepare the YOLO dataset.
    
    Args:
        source_dir: Path to "500 cases with annotation" folder
        output_dir: Path to output dataset folder
        train_ratio: Ratio of training data (default 0.8 for 80/20 split)
        seed: Random seed for reproducibility
    """
    logger.info("=" * 60)
    logger.info("DENTAL DATASET PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Train/Val Split: {train_ratio*100:.0f}/{(1-train_ratio)*100:.0f}")
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    create_directory_structure(output_dir)
    
    # Find all case directories
    case_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith("case")])
    logger.info(f"Found {len(case_dirs)} case directories")
    
    # Process all cases and collect valid ones
    valid_cases = []
    skipped_cases = 0
    total_annotations = 0
    
    for i, case_dir in enumerate(case_dirs):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing case {i+1}/{len(case_dirs)}...")
        
        image_path, annotations = process_case(case_dir)
        
        if image_path and annotations:
            valid_cases.append({
                'case_name': case_dir.name,
                'image_path': image_path,
                'annotations': annotations
            })
            total_annotations += len(annotations)
        else:
            skipped_cases += 1
    
    logger.info(f"Valid cases: {len(valid_cases)}")
    logger.info(f"Skipped cases: {skipped_cases}")
    logger.info(f"Total annotations: {total_annotations}")
    
    # Shuffle and split
    random.shuffle(valid_cases)
    split_idx = int(len(valid_cases) * train_ratio)
    
    train_cases = valid_cases[:split_idx]
    val_cases = valid_cases[split_idx:]
    
    logger.info(f"Training set: {len(train_cases)} cases")
    logger.info(f"Validation set: {len(val_cases)} cases")
    
    # Copy files and create labels
    def save_split(cases: List[dict], split_name: str):
        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name
        
        for case in cases:
            # Standardize filename
            case_num = re.search(r'\d+', case['case_name'])
            if case_num:
                base_name = f"case_{case_num.group()}"
            else:
                base_name = case['case_name'].replace(" ", "_")
            
            # Copy image
            dst_image = images_dir / f"{base_name}.png"
            shutil.copy2(case['image_path'], dst_image)
            
            # Write label file
            label_file = labels_dir / f"{base_name}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(case['annotations']))
        
        logger.info(f"Saved {len(cases)} {split_name} samples")
    
    save_split(train_cases, "train")
    save_split(val_cases, "val")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training images: {output_dir / 'images' / 'train'}")
    logger.info(f"Validation images: {output_dir / 'images' / 'val'}")
    logger.info(f"Training labels: {output_dir / 'labels' / 'train'}")
    logger.info(f"Validation labels: {output_dir / 'labels' / 'val'}")
    
    return {
        'total_cases': len(case_dirs),
        'valid_cases': len(valid_cases),
        'train_cases': len(train_cases),
        'val_cases': len(val_cases),
        'total_annotations': total_annotations,
    }


def verify_dataset(dataset_dir: Path):
    """Verify the created dataset structure and print statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 60)
    
    for split in ['train', 'val']:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        
        images = list(images_dir.glob("*.png"))
        labels = list(labels_dir.glob("*.txt"))
        
        logger.info(f"\n{split.upper()} Split:")
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Labels: {len(labels)}")
        
        # Check matching
        image_names = {img.stem for img in images}
        label_names = {lbl.stem for lbl in labels}
        
        missing_labels = image_names - label_names
        missing_images = label_names - image_names
        
        if missing_labels:
            logger.warning(f"  Images without labels: {len(missing_labels)}")
        if missing_images:
            logger.warning(f"  Labels without images: {len(missing_images)}")
        
        # Count annotations
        total_annotations = 0
        class_counts = {}
        for label_file in labels:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        total_annotations += 1
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        logger.info(f"  Total annotations: {total_annotations}")
        logger.info(f"  Unique classes: {len(class_counts)}")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(__file__).parent.parent  # SP folder
    SOURCE_DIR = BASE_DIR / "material" / "500 cases with annotation"
    OUTPUT_DIR = Path(__file__).parent / "dataset_v2"  # week1/dataset_v2 (pre-trained compatible)
    
    # Prepare dataset
    stats = prepare_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,  # 80/20 split
        seed=42
    )
    
    # Verify dataset
    verify_dataset(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Review the created dataset in week1/dataset/")
    print("2. Run: python train.py")
    print("=" * 60)
