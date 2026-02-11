"""
Prepare Dataset with Class Order Matching Pre-trained Model
Converts XML annotations to YOLO segmentation format
Compatible with: Tooth_seg_pano_20250319.pt
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
SOURCE_DIR = Path(r"C:\Users\jaopi\Desktop\SP\material\500 cases with annotation")
OUTPUT_DIR = Path(r"C:\Users\jaopi\Desktop\SP\week1\dataset_v2")
TRAIN_RATIO = 0.8

# FDI to Class Index mapping - MATCHES PRE-TRAINED MODEL ORDER
# Sequential order within each quadrant: 11-18, 21-28, 31-38, 41-48
FDI_TO_CLASS = {
    # Upper Right (11-18)
    "11": 0, "12": 1, "13": 2, "14": 3, "15": 4, "16": 5, "17": 6, "18": 7,
    # Upper Left (21-28)
    "21": 8, "22": 9, "23": 10, "24": 11, "25": 12, "26": 13, "27": 14, "28": 15,
    # Lower Left (31-38)
    "31": 16, "32": 17, "33": 18, "34": 19, "35": 20, "36": 21, "37": 22, "38": 23,
    # Lower Right (41-48)
    "41": 24, "42": 25, "43": 26, "44": 27, "45": 28, "46": 29, "47": 30, "48": 31,
}


def extract_tooth_name(annotation_text: str) -> str:
    """Extract FDI tooth number from annotation text."""
    if not annotation_text:
        return None
    parts = annotation_text.strip().split()
    for part in parts:
        clean = ''.join(c for c in part if c.isdigit())
        if clean in FDI_TO_CLASS:
            return clean
    return None


def extract_polygon_coordinates(geometry_elem) -> list:
    """Extract polygon coordinates from XML geometry element."""
    coords = []
    for coord in geometry_elem.findall('.//Coordinate'):
        x = float(coord.get('X', 0))
        y = float(coord.get('Y', 0))
        coords.append((x, y))
    return coords


def normalize_polygon(coords: list, img_width: int, img_height: int) -> list:
    """Normalize polygon coordinates to 0-1 range."""
    normalized = []
    for x, y in coords:
        nx = min(max(x / img_width, 0), 1)
        ny = min(max(y / img_height, 0), 1)
        normalized.extend([nx, ny])
    return normalized


def process_xml_file(xml_path: Path) -> tuple:
    """Process XML annotation file and return (image_path, annotations)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Error parsing {xml_path}: {e}")
        return None, []
    
    # Get image dimensions and path
    img_elem = root.find('.//Image')
    if img_elem is None:
        return None, []
    
    img_width = int(img_elem.get('Width', 0))
    img_height = int(img_elem.get('Height', 0))
    img_path = img_elem.get('FilePath', '')
    
    if not img_path or img_width == 0 or img_height == 0:
        return None, []
    
    annotations = []
    
    # Process each annotation
    for annotation in root.findall('.//Annotation'):
        text = annotation.get('Text', '')
        tooth_name = extract_tooth_name(text)
        
        if tooth_name is None:
            continue
        
        class_id = FDI_TO_CLASS.get(tooth_name)
        if class_id is None:
            continue
        
        # Get polygon coordinates
        geometry = annotation.find('.//Geometry')
        if geometry is None:
            continue
        
        coords = extract_polygon_coordinates(geometry)
        if len(coords) < 3:
            continue
        
        # Normalize coordinates
        normalized = normalize_polygon(coords, img_width, img_height)
        
        annotations.append({
            'class_id': class_id,
            'tooth_name': tooth_name,
            'polygon': normalized
        })
    
    return img_path, annotations


def create_yolo_label(annotations: list) -> str:
    """Create YOLO format label string."""
    lines = []
    for ann in annotations:
        coords_str = ' '.join(f"{c:.6f}" for c in ann['polygon'])
        lines.append(f"{ann['class_id']} {coords_str}")
    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("DATASET PREPARATION (Pre-trained Model Compatible)")
    print("=" * 60)
    
    # Create output directories
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Find all case folders
    case_folders = sorted([
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and d.name.startswith('case')
    ], key=lambda x: int(x.name.split()[-1]))
    
    print(f"Found {len(case_folders)} case folders")
    
    # Process each case
    processed = []
    total_annotations = 0
    
    for case_folder in case_folders:
        xml_files = list(case_folder.glob('*.xml'))
        if not xml_files:
            continue
        
        xml_path = xml_files[0]
        img_path, annotations = process_xml_file(xml_path)
        
        if not img_path or not annotations:
            continue
        
        # Find actual image file
        img_name = Path(img_path).name
        possible_paths = [
            case_folder / img_name,
            case_folder / img_name.replace('.png', '.jpg'),
            case_folder / img_name.replace('.jpg', '.png'),
        ]
        
        actual_img_path = None
        for p in possible_paths:
            if p.exists():
                actual_img_path = p
                break
        
        if actual_img_path is None:
            for img_file in case_folder.glob('*.png'):
                actual_img_path = img_file
                break
            if actual_img_path is None:
                for img_file in case_folder.glob('*.jpg'):
                    actual_img_path = img_file
                    break
        
        if actual_img_path is None:
            continue
        
        processed.append({
            'case': case_folder.name,
            'image': actual_img_path,
            'annotations': annotations
        })
        total_annotations += len(annotations)
    
    print(f"Successfully processed {len(processed)} cases")
    print(f"Total annotations: {total_annotations}")
    
    # Split into train/val
    random.seed(42)
    random.shuffle(processed)
    
    split_idx = int(len(processed) * TRAIN_RATIO)
    train_cases = processed[:split_idx]
    val_cases = processed[split_idx:]
    
    print(f"\nSplit: {len(train_cases)} train, {len(val_cases)} val")
    
    # Copy files
    for split, cases in [('train', train_cases), ('val', val_cases)]:
        for item in cases:
            # Generate filename
            case_num = item['case'].split()[-1]
            img_ext = item['image'].suffix
            new_name = f"case_{case_num}"
            
            # Copy image
            dst_img = OUTPUT_DIR / 'images' / split / f"{new_name}{img_ext}"
            shutil.copy2(item['image'], dst_img)
            
            # Create label
            label_content = create_yolo_label(item['annotations'])
            dst_label = OUTPUT_DIR / 'labels' / split / f"{new_name}.txt"
            dst_label.write_text(label_content)
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Train: {len(train_cases)} images")
    print(f"Val: {len(val_cases)} images")
    print(f"Class mapping: Sequential FDI order (matches pre-trained model)")
    print("=" * 60)


if __name__ == "__main__":
    main()
