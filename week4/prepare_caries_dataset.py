"""
Caries Dataset Preparation Script
==================================
Extracts caries annotations from XML files and converts to YOLO format.
Creates a simplified 3-class dataset for caries surface detection.

Class Mapping:
    0: Occlusal  - Caries on the chewing surface
    1: Proximal  - Caries on side surfaces (Mesial/Distal)
    2: Lingual   - Caries on tongue-facing surface (Lingual/Buccal/Labial/Palatal)

Input:
    material/500 cases with annotation/case X/*.xml

Output:
    week4/dataset_3class/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

Author: Lead CV Engineer
Date: 2026-01-27
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
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SURFACE TYPE MAPPING
# =============================================================================

# Map surface keywords to class IDs
SURFACE_TO_CLASS: Dict[str, int] = {
    # Class 0: Occlusal
    'occlusal': 0,
    'occlusal surface': 0,
    
    # Class 1: Proximal (Mesial and Distal)
    'proximal': 1,
    'mesial': 1,
    'mesial surface': 1,
    'distal': 1,
    'distal surface': 1,
    'interproximal': 1,
    
    # Class 2: Lingual (and related surfaces)
    'lingual': 2,
    'lingual surface': 2,
    'buccal': 2,
    'buccal surface': 2,
    'labial': 2,
    'labial surface': 2,
    'palatal': 2,
    'palatal surface': 2,
}

CLASS_NAMES = {
    0: 'Occlusal',
    1: 'Proximal',
    2: 'Lingual'
}


@dataclass
class CariesAnnotation:
    """Represents a single caries annotation."""
    class_id: int
    class_name: str
    surface_type: str
    polygon: List[Tuple[float, float]]  # Normalized coordinates
    tooth_id: Optional[str] = None


def extract_surface_type(xml_content: str) -> Optional[str]:
    """Extract caries surface type from XML."""
    # Pattern to match surface type in displayName
    patterns = [
        r'displayName.*?value="((?:Occlusal|Mesial|Distal|Lingual|Buccal|Labial|Palatal|Proximal)\s*[sS]urface?)"',
        r'displayName.*?value="(Occlusal surface)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, xml_content, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip()
    
    return None


def get_class_from_surface(surface_type: str) -> Optional[int]:
    """Map surface type string to class ID."""
    surface_lower = surface_type.lower().strip()
    
    # Direct match
    if surface_lower in SURFACE_TO_CLASS:
        return SURFACE_TO_CLASS[surface_lower]
    
    # Partial match
    for key, class_id in SURFACE_TO_CLASS.items():
        if key in surface_lower or surface_lower in key:
            return class_id
    
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


def extract_tooth_id(xml_content: str) -> Optional[str]:
    """Extract tooth FDI number from XML if available."""
    # Pattern for tooth number
    pattern = r'displayName.*?value="Permanent\s+((?:upper|lower)\s+(?:right|left)\s+(?:third|second|first)?\s*(?:molar|premolar|canine|lateral\s+incisor|central\s+incisor))'
    
    match = re.search(pattern, xml_content, re.IGNORECASE)
    if match:
        return match.group(1).lower().strip()
    return None


def is_caries_annotation(xml_content: str) -> bool:
    """Check if XML contains caries annotation (not just tooth segmentation)."""
    # Look for dental caries code
    if 'Dental caries' in xml_content or '118065D' in xml_content:
        return True
    return False


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


def polygon_to_bbox(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Convert polygon to bounding box (x_center, y_center, width, height)."""
    if not polygon:
        return (0, 0, 0, 0)
    
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return (x_center, y_center, width, height)


def parse_xml_for_caries(xml_path: Path, img_width: int, img_height: int) -> Optional[CariesAnnotation]:
    """Parse XML file and extract caries annotation."""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if this is a caries annotation
        if not is_caries_annotation(content):
            return None
        
        # Extract surface type
        surface_type = extract_surface_type(content)
        if not surface_type:
            logger.debug(f"No surface type found in {xml_path.name}")
            return None
        
        # Get class ID
        class_id = get_class_from_surface(surface_type)
        if class_id is None:
            logger.warning(f"Unknown surface type: {surface_type} in {xml_path.name}")
            return None
        
        # Extract polygon
        polygon = extract_polygon_coordinates(content)
        if len(polygon) < 3:
            logger.debug(f"Insufficient polygon points ({len(polygon)}) in {xml_path.name}")
            return None
        
        # Normalize polygon
        normalized_polygon = normalize_polygon(polygon, img_width, img_height)
        
        # Extract tooth ID if available
        tooth_id = extract_tooth_id(content)
        
        return CariesAnnotation(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            surface_type=surface_type,
            polygon=normalized_polygon,
            tooth_id=tooth_id
        )
        
    except Exception as e:
        logger.error(f"Error parsing {xml_path}: {e}")
        return None


def format_yolo_segmentation(class_id: int, polygon: List[Tuple[float, float]]) -> str:
    """Format annotation as YOLO segmentation line."""
    coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in polygon)
    return f'{class_id} {coords}'


def format_yolo_detection(class_id: int, polygon: List[Tuple[float, float]]) -> str:
    """Format annotation as YOLO detection line (bbox)."""
    x_center, y_center, width, height = polygon_to_bbox(polygon)
    return f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'


def process_case(case_dir: Path, output_format: str = 'detection') -> Tuple[Optional[Path], List[CariesAnnotation]]:
    """Process a single case directory and return annotations."""
    # Find image file
    case_num = case_dir.name.replace('case ', '')
    image_files = list(case_dir.glob(f'case_{case_num}.png')) + list(case_dir.glob('*.png'))
    
    if not image_files:
        logger.warning(f"No image found in {case_dir}")
        return None, []
    
    image_path = image_files[0]
    
    # Get image dimensions
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return None, []
    
    # Find and parse XML files
    xml_files = list(case_dir.glob('*.xml'))
    annotations = []
    
    for xml_path in xml_files:
        annotation = parse_xml_for_caries(xml_path, img_width, img_height)
        if annotation:
            annotations.append(annotation)
    
    return image_path, annotations


def create_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    output_format: str = 'detection',  # 'detection' or 'segmentation'
    seed: int = 42
):
    """
    Create the 3-class caries dataset.
    
    Args:
        input_dir: Path to '500 cases with annotation' folder
        output_dir: Path to output dataset folder
        train_ratio: Ratio of training data (default 0.8)
        output_format: 'detection' for bbox, 'segmentation' for polygon
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    dirs = [
        output_path / 'images' / 'train',
        output_path / 'images' / 'val',
        output_path / 'labels' / 'train',
        output_path / 'labels' / 'val',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find all case directories
    case_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('case')])
    
    logger.info(f"Found {len(case_dirs)} case directories")
    
    # Process all cases
    all_data = []
    stats = {0: 0, 1: 0, 2: 0}
    cases_with_caries = 0
    
    for case_dir in case_dirs:
        image_path, annotations = process_case(case_dir, output_format)
        
        if image_path and annotations:
            all_data.append((image_path, annotations))
            cases_with_caries += 1
            for ann in annotations:
                stats[ann.class_id] += 1
    
    logger.info(f"\nCases with caries annotations: {cases_with_caries}/{len(case_dirs)}")
    logger.info(f"Total annotations found:")
    for class_id, count in stats.items():
        logger.info(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {count}")
    
    # Shuffle and split
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    logger.info(f"\nTrain set: {len(train_data)} images")
    logger.info(f"Val set:   {len(val_data)} images")
    
    # Process train set
    logger.info("\nProcessing train set...")
    for image_path, annotations in train_data:
        case_name = image_path.stem
        
        # Copy image
        dst_image = output_path / 'images' / 'train' / image_path.name
        shutil.copy2(image_path, dst_image)
        
        # Create label file
        label_path = output_path / 'labels' / 'train' / f'{case_name}.txt'
        with open(label_path, 'w') as f:
            for ann in annotations:
                if output_format == 'segmentation':
                    line = format_yolo_segmentation(ann.class_id, ann.polygon)
                else:
                    line = format_yolo_detection(ann.class_id, ann.polygon)
                f.write(line + '\n')
    
    # Process val set
    logger.info("Processing val set...")
    for image_path, annotations in val_data:
        case_name = image_path.stem
        
        # Copy image
        dst_image = output_path / 'images' / 'val' / image_path.name
        shutil.copy2(image_path, dst_image)
        
        # Create label file
        label_path = output_path / 'labels' / 'val' / f'{case_name}.txt'
        with open(label_path, 'w') as f:
            for ann in annotations:
                if output_format == 'segmentation':
                    line = format_yolo_segmentation(ann.class_id, ann.polygon)
                else:
                    line = format_yolo_detection(ann.class_id, ann.polygon)
                f.write(line + '\n')
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': {
            0: 'Occlusal',
            1: 'Proximal',
            2: 'Lingual'
        }
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"# Dental Caries Detection Dataset - 3 Classes\n")
        f.write(f"# Generated: 2026-01-27\n")
        f.write(f"# Format: {'segmentation' if output_format == 'segmentation' else 'detection'}\n\n")
        f.write(f"path: {output_path.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: 3\n\n")
        f.write(f"names:\n")
        f.write(f"  0: Occlusal\n")
        f.write(f"  1: Proximal\n")
        f.write(f"  2: Lingual\n")
    
    logger.info(f"\nCreated data.yaml at: {yaml_path}")
    
    # Create summary statistics file
    summary = {
        'total_cases': len(case_dirs),
        'cases_with_caries': cases_with_caries,
        'train_images': len(train_data),
        'val_images': len(val_data),
        'class_distribution': {
            'Occlusal': stats[0],
            'Proximal': stats[1],
            'Lingual': stats[2]
        },
        'output_format': output_format,
        'train_ratio': train_ratio
    }
    
    summary_path = output_path / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Created summary at: {summary_path}")
    
    print(f"\n{'='*60}")
    print("DATASET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Train images:     {len(train_data)}")
    print(f"Val images:       {len(val_data)}")
    print(f"\nClass Distribution:")
    print(f"  0 (Occlusal):  {stats[0]}")
    print(f"  1 (Proximal):  {stats[1]}")
    print(f"  2 (Lingual):   {stats[2]}")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create 3-class caries detection dataset from XML annotations'
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='../material/500 cases with annotation',
        help='Path to "500 cases with annotation" folder'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./dataset_3class',
        help='Output dataset directory'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['detection', 'segmentation'],
        default='detection',
        help='Output format (detection for bbox, segmentation for polygon)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Train/val split ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    create_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        output_format=args.format,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
