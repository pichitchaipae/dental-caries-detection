"""
Dataset Converter for Dental Caries Detection
==============================================
This script converts complex multi-class dental caries labels (e.g., "46_Occlusal")
to simplified 3-class format for YOLO training.

Class Mapping:
    0: Occlusal  - Caries on the chewing surface
    1: Proximal  - Caries on the side surfaces (Mesial/Distal)
    2: Lingual   - Caries on the tongue-facing surface (includes Buccal/Labial)

Author: Senior CV Engineer
Date: 2026-01-27
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CLASS MAPPING CONFIGURATION
# =============================================================================

# Surface type keywords to simplified class ID mapping
SURFACE_MAPPING: Dict[str, int] = {
    # Class 0: Occlusal
    'occlusal': 0,
    'occ': 0,
    
    # Class 1: Proximal (Mesial and Distal surfaces)
    'proximal': 1,
    'mesial': 1,
    'distal': 1,
    'interproximal': 1,
    
    # Class 2: Lingual (and related surfaces)
    'lingual': 2,
    'buccal': 2,
    'labial': 2,
    'palatal': 2,
}

# Simplified class names for reference
CLASS_NAMES: Dict[int, str] = {
    0: 'Occlusal',
    1: 'Proximal',
    2: 'Lingual'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_surface_type(class_name: str) -> Optional[int]:
    """
    Extract the surface type from a complex class name and return the simplified class ID.
    
    Args:
        class_name: Original class name (e.g., "46_Occlusal", "11_Proximal", "tooth_23_mesial")
    
    Returns:
        Simplified class ID (0, 1, or 2) or None if no match found
    
    Examples:
        >>> extract_surface_type("46_Occlusal")
        0
        >>> extract_surface_type("11_Proximal")
        1
        >>> extract_surface_type("tooth_23_lingual_caries")
        2
    """
    # Convert to lowercase for case-insensitive matching
    class_name_lower = class_name.lower()
    
    # Check each surface keyword
    for surface_keyword, class_id in SURFACE_MAPPING.items():
        if surface_keyword in class_name_lower:
            return class_id
    
    # No match found
    return None


def parse_yolo_label_line(line: str) -> Tuple[int, List[float]]:
    """
    Parse a single line from a YOLO format label file.
    
    Args:
        line: A line from YOLO label file (class_id x_center y_center width height)
    
    Returns:
        Tuple of (class_id, [x_center, y_center, width, height])
    """
    parts = line.strip().split()
    class_id = int(parts[0])
    bbox = [float(p) for p in parts[1:5]]
    
    # Handle segmentation masks (additional coordinates after bbox)
    extra_coords = [float(p) for p in parts[5:]] if len(parts) > 5 else []
    
    return class_id, bbox, extra_coords


def format_yolo_label_line(class_id: int, bbox: List[float], extra_coords: List[float] = None) -> str:
    """
    Format a YOLO label line from class ID and bounding box.
    
    Args:
        class_id: The class ID
        bbox: List of [x_center, y_center, width, height]
        extra_coords: Optional additional coordinates (for segmentation masks)
    
    Returns:
        Formatted YOLO label string
    """
    bbox_str = ' '.join(f'{coord:.6f}' for coord in bbox)
    
    if extra_coords:
        extra_str = ' '.join(f'{coord:.6f}' for coord in extra_coords)
        return f'{class_id} {bbox_str} {extra_str}'
    
    return f'{class_id} {bbox_str}'


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def load_class_mapping(mapping_file: str) -> Dict[int, str]:
    """
    Load class ID to class name mapping from a file.
    
    Supports:
        - YAML format (data.yaml with 'names' key)
        - JSON format
        - TXT format (one class name per line)
    
    Args:
        mapping_file: Path to the mapping file
    
    Returns:
        Dictionary mapping class ID to class name
    """
    file_path = Path(mapping_file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.yaml', '.yml']:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            names = data.get('names', {})
            # Handle both list and dict formats
            if isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
            return names
    
    elif suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, list):
                return {i: name for i, name in enumerate(data)}
            return {int(k): v for k, v in data.items()}
    
    elif suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            return {i: name for i, name in enumerate(lines)}
    
    else:
        raise ValueError(f"Unsupported mapping file format: {suffix}")


def convert_yolo_labels(
    input_dir: str,
    output_dir: str,
    class_mapping: Optional[Dict[int, str]] = None,
    class_mapping_file: Optional[str] = None
) -> Dict[str, int]:
    """
    Convert YOLO label files from complex multi-class to simplified 3-class format.
    
    Args:
        input_dir: Directory containing original YOLO label files (.txt)
        output_dir: Directory to save converted label files
        class_mapping: Optional dictionary mapping old class IDs to class names
        class_mapping_file: Optional path to class mapping file
    
    Returns:
        Statistics dictionary with conversion counts
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load class mapping if provided
    old_class_names = {}
    if class_mapping:
        old_class_names = class_mapping
    elif class_mapping_file:
        old_class_names = load_class_mapping(class_mapping_file)
    
    # Statistics
    stats = {
        'total_files': 0,
        'converted_files': 0,
        'total_annotations': 0,
        'converted_annotations': 0,
        'skipped_annotations': 0,
        'class_distribution': {0: 0, 1: 0, 2: 0}
    }
    
    # Process each label file
    label_files = list(input_path.glob('*.txt'))
    stats['total_files'] = len(label_files)
    
    print(f"\n{'='*60}")
    print(f"Converting YOLO Labels")
    print(f"{'='*60}")
    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Total files:      {len(label_files)}")
    print(f"{'='*60}\n")
    
    for label_file in label_files:
        converted_lines = []
        
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            stats['total_annotations'] += 1
            
            try:
                old_class_id, bbox, extra_coords = parse_yolo_label_line(line)
                
                # Get the old class name
                if old_class_names and old_class_id in old_class_names:
                    old_class_name = old_class_names[old_class_id]
                else:
                    # If no mapping provided, try to use the class ID directly
                    # This handles cases where the "class name" might be embedded in filename
                    old_class_name = str(old_class_id)
                
                # Extract surface type and get new class ID
                new_class_id = extract_surface_type(old_class_name)
                
                if new_class_id is not None:
                    new_line = format_yolo_label_line(new_class_id, bbox, extra_coords)
                    converted_lines.append(new_line)
                    stats['converted_annotations'] += 1
                    stats['class_distribution'][new_class_id] += 1
                else:
                    # If no surface type found, skip this annotation
                    print(f"  Warning: Could not map class '{old_class_name}' (ID: {old_class_id})")
                    stats['skipped_annotations'] += 1
                    
            except (ValueError, IndexError) as e:
                print(f"  Error parsing line in {label_file.name}: {line}")
                stats['skipped_annotations'] += 1
                continue
        
        # Save converted labels
        if converted_lines:
            output_file = output_path / label_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(converted_lines))
            stats['converted_files'] += 1
    
    return stats


def convert_json_labels(
    input_dir: str,
    output_dir: str,
    output_format: str = 'yolo'
) -> Dict[str, int]:
    """
    Convert JSON annotation files to simplified 3-class YOLO format.
    
    Handles various JSON annotation formats (COCO, LabelMe, custom).
    
    Args:
        input_dir: Directory containing JSON annotation files
        output_dir: Directory to save converted label files
        output_format: Output format ('yolo' or 'json')
    
    Returns:
        Statistics dictionary with conversion counts
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'converted_files': 0,
        'total_annotations': 0,
        'converted_annotations': 0,
        'skipped_annotations': 0,
        'class_distribution': {0: 0, 1: 0, 2: 0}
    }
    
    json_files = list(input_path.glob('*.json'))
    stats['total_files'] = len(json_files)
    
    print(f"\n{'='*60}")
    print(f"Converting JSON Labels")
    print(f"{'='*60}")
    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Total files:      {len(json_files)}")
    print(f"{'='*60}\n")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_annotations = []
        image_width = data.get('imageWidth', 1)
        image_height = data.get('imageHeight', 1)
        
        # Handle different JSON formats
        annotations = data.get('shapes', data.get('annotations', []))
        
        for ann in annotations:
            stats['total_annotations'] += 1
            
            # Get class name (handle different key names)
            class_name = ann.get('label', ann.get('category', ann.get('class_name', '')))
            
            # Extract surface type
            new_class_id = extract_surface_type(class_name)
            
            if new_class_id is None:
                print(f"  Warning: Could not map class '{class_name}'")
                stats['skipped_annotations'] += 1
                continue
            
            # Get bounding box (handle different formats)
            if 'points' in ann:
                # LabelMe format: [[x1, y1], [x2, y2]]
                points = ann['points']
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                else:
                    continue
            elif 'bbox' in ann:
                # COCO format: [x, y, width, height]
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
            else:
                continue
            
            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = abs(x2 - x1) / image_width
            height = abs(y2 - y1) / image_height
            
            converted_annotations.append({
                'class_id': new_class_id,
                'bbox': [x_center, y_center, width, height]
            })
            stats['converted_annotations'] += 1
            stats['class_distribution'][new_class_id] += 1
        
        # Save converted labels
        if converted_annotations:
            if output_format == 'yolo':
                output_file = output_path / f"{json_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for ann in converted_annotations:
                        line = format_yolo_label_line(ann['class_id'], ann['bbox'])
                        f.write(line + '\n')
            else:
                output_file = output_path / json_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(converted_annotations, f, indent=2)
            
            stats['converted_files'] += 1
    
    return stats


def print_statistics(stats: Dict[str, int]) -> None:
    """Print conversion statistics in a formatted way."""
    print(f"\n{'='*60}")
    print("CONVERSION STATISTICS")
    print(f"{'='*60}")
    print(f"Total files processed:     {stats['total_files']}")
    print(f"Files with annotations:    {stats['converted_files']}")
    print(f"Total annotations:         {stats['total_annotations']}")
    print(f"Converted annotations:     {stats['converted_annotations']}")
    print(f"Skipped annotations:       {stats['skipped_annotations']}")
    print(f"\nClass Distribution:")
    for class_id, count in stats['class_distribution'].items():
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        percentage = (count / stats['converted_annotations'] * 100) if stats['converted_annotations'] > 0 else 0
        print(f"  {class_id}: {class_name:10} - {count:5} ({percentage:.1f}%)")
    print(f"{'='*60}\n")


def create_data_yaml(output_dir: str, train_path: str = None, val_path: str = None) -> None:
    """
    Create a data.yaml file for YOLO training with the simplified class structure.
    
    Args:
        output_dir: Directory to save the data.yaml file
        train_path: Path to training images (optional)
        val_path: Path to validation images (optional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': train_path or 'images/train',
        'val': val_path or 'images/val',
        'nc': 3,
        'names': {
            0: 'Occlusal',
            1: 'Proximal',
            2: 'Lingual'
        }
    }
    
    yaml_path = output_path / 'data.yaml'
    
    try:
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to manual YAML writing
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {data_yaml['path']}\n")
            f.write(f"train: {data_yaml['train']}\n")
            f.write(f"val: {data_yaml['val']}\n")
            f.write(f"nc: {data_yaml['nc']}\n")
            f.write("names:\n")
            for k, v in data_yaml['names'].items():
                f.write(f"  {k}: {v}\n")
    
    print(f"Created data.yaml at: {yaml_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the dataset converter."""
    parser = argparse.ArgumentParser(
        description='Convert complex dental caries labels to simplified 3-class format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert YOLO labels with class mapping file
  python dataset_converter.py --input_dir ./labels --output_dir ./labels_converted --mapping_file data.yaml
  
  # Convert JSON annotations
  python dataset_converter.py --input_dir ./annotations --output_dir ./labels_converted --format json
  
  # Create data.yaml file for training
  python dataset_converter.py --input_dir ./labels --output_dir ./dataset --create_yaml

Class Mapping:
  0: Occlusal  - Caries on chewing surface
  1: Proximal  - Caries on side surfaces (Mesial/Distal)
  2: Lingual   - Caries on tongue-facing surface
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing original label files'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Directory to save converted label files'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['yolo', 'json'],
        default='yolo',
        help='Input label format (default: yolo)'
    )
    
    parser.add_argument(
        '--mapping_file', '-m',
        type=str,
        default=None,
        help='Path to class mapping file (YAML, JSON, or TXT)'
    )
    
    parser.add_argument(
        '--create_yaml',
        action='store_true',
        help='Create data.yaml file for YOLO training'
    )
    
    parser.add_argument(
        '--train_path',
        type=str,
        default=None,
        help='Path to training images (for data.yaml)'
    )
    
    parser.add_argument(
        '--val_path',
        type=str,
        default=None,
        help='Path to validation images (for data.yaml)'
    )
    
    args = parser.parse_args()
    
    # Run conversion based on format
    if args.format == 'yolo':
        stats = convert_yolo_labels(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            class_mapping_file=args.mapping_file
        )
    else:
        stats = convert_json_labels(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
    
    # Print statistics
    print_statistics(stats)
    
    # Create data.yaml if requested
    if args.create_yaml:
        create_data_yaml(
            output_dir=args.output_dir,
            train_path=args.train_path,
            val_path=args.val_path
        )
    
    print("Conversion complete!")


if __name__ == '__main__':
    main()
