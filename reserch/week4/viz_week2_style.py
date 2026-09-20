"""
Week 2 Style Visualization Tool
================================
Visualizes dental X-ray analysis results from Week 2 JSON format.

Week 2 JSON Structure:
{
    "case_number": str,
    "num_teeth_detected": int,
    "teeth_data": [
        {
            "tooth_id": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "crop_coords": [x1, y1, x2, y2],
            "num_segments": int,
            "total_pixels": int,
            "pixel_coordinates": [[x,y], ...]
        }
    ]
}

Usage:
    python viz_week2_style.py --case_path "path/to/case_1_results.json"
    python viz_week2_style.py --case_path "path/to/case_1_results.json" --show_mask
    python viz_week2_style.py --case_dir "path/to/case 1" --case_num 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    import cv2
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install matplotlib opencv-python numpy")
    sys.exit(1)


# Color scheme for visualization
COLORS = {
    'tooth_bbox': '#00FF00',      # Green for tooth bounding boxes
    'tooth_mask': '#00FF0050',    # Semi-transparent green for tooth mask
    'caries_bbox': '#FF0000',     # Red for caries
    'caries_mask': '#FF000080',   # Semi-transparent red for caries
    'text_bg': '#000000CC',       # Dark background for text
    'text_fg': '#FFFFFF',         # White text
}

# FDI Tooth naming
TOOTH_NAMES = {
    '11': 'Upper Right Central Incisor', '12': 'Upper Right Lateral Incisor',
    '13': 'Upper Right Canine', '14': 'Upper Right First Premolar',
    '15': 'Upper Right Second Premolar', '16': 'Upper Right First Molar',
    '17': 'Upper Right Second Molar', '18': 'Upper Right Third Molar',
    '21': 'Upper Left Central Incisor', '22': 'Upper Left Lateral Incisor',
    '23': 'Upper Left Canine', '24': 'Upper Left First Premolar',
    '25': 'Upper Left Second Premolar', '26': 'Upper Left First Molar',
    '27': 'Upper Left Second Molar', '28': 'Upper Left Third Molar',
    '31': 'Lower Left Central Incisor', '32': 'Lower Left Lateral Incisor',
    '33': 'Lower Left Canine', '34': 'Lower Left First Premolar',
    '35': 'Lower Left Second Premolar', '36': 'Lower Left First Molar',
    '37': 'Lower Left Second Molar', '38': 'Lower Left Third Molar',
    '41': 'Lower Right Central Incisor', '42': 'Lower Right Lateral Incisor',
    '43': 'Lower Right Canine', '44': 'Lower Right First Premolar',
    '45': 'Lower Right Second Premolar', '46': 'Lower Right First Molar',
    '47': 'Lower Right Second Molar', '48': 'Lower Right Third Molar',
}


def load_json(filepath: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}


def find_image_for_case(json_path: str, case_num: Optional[int] = None) -> Optional[str]:
    """Find the corresponding image file for a case JSON."""
    json_path = Path(json_path)
    case_dir = json_path.parent
    
    # Extract case number from JSON filename if not provided
    if case_num is None:
        import re
        match = re.search(r'case_(\d+)', json_path.stem)
        if match:
            case_num = int(match.group(1))
    
    # Search patterns for image file
    patterns = [
        f"case_{case_num}.png",
        f"case_{case_num}.jpg",
        f"case_{case_num}_*.png",
        f"case_{case_num}_*.jpg",
        "*.png",
        "*.jpg"
    ]
    
    for pattern in patterns:
        matches = list(case_dir.glob(pattern))
        if matches:
            # Filter out annotated/overlay images
            for m in matches:
                if 'annotated' not in m.name.lower() and 'overlay' not in m.name.lower() and 'bounding' not in m.name.lower():
                    return str(m)
            # If all are annotated, return first match
            return str(matches[0])
    
    # Check parent's material folder structure
    material_path = case_dir.parent.parent / "material" / "500 cases with annotation" / f"case {case_num}"
    if material_path.exists():
        for pattern in ["*.png", "*.jpg"]:
            matches = list(material_path.glob(pattern))
            if matches:
                return str(matches[0])
    
    return None


def create_mask_from_pixels(pixels: List[List[int]], img_shape: Tuple[int, int]) -> np.ndarray:
    """Create a binary mask from pixel coordinates."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for x, y in pixels:
        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            mask[y, x] = 255
    return mask


def visualize_week2_json(
    json_path: str,
    image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show_mask: bool = False,
    show_labels: bool = True,
    show_confidence: bool = True,
    figsize: Tuple[int, int] = (20, 12),
    show: bool = True
):
    """
    Visualize Week 2 JSON results on the dental X-ray image.
    
    Args:
        json_path: Path to the Week 2 results JSON file
        image_path: Optional path to the image (auto-detected if not provided)
        output_path: Optional path to save the visualization
        show_mask: Whether to show tooth segmentation masks
        show_labels: Whether to show tooth ID labels
        show_confidence: Whether to show confidence scores
        figsize: Figure size in inches
        show: Whether to display the plot
    """
    
    # Load JSON data
    data = load_json(json_path)
    if not data:
        print(f"Failed to load JSON from {json_path}")
        return
    
    case_number = data.get('case_number', 'Unknown')
    teeth_data = data.get('teeth_data', [])
    
    print(f"\n{'='*60}")
    print(f"WEEK 2 VISUALIZATION - Case {case_number}")
    print(f"{'='*60}")
    print(f"Teeth detected: {len(teeth_data)}")
    
    # Find image if not provided
    if image_path is None:
        image_path = find_image_for_case(json_path)
    
    if image_path is None or not os.path.exists(image_path):
        print(f"⚠ Could not find image for case {case_number}")
        print("  Specify image path with --image_path argument")
        return
    
    print(f"Image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(f"Case {case_number} - {len(teeth_data)} Teeth Detected", fontsize=16, fontweight='bold')
    
    # Create mask overlay if requested
    if show_mask:
        mask_overlay = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    
    # Process each tooth
    for i, tooth in enumerate(teeth_data):
        tooth_id = tooth.get('tooth_id', f'T{i}')
        confidence = tooth.get('confidence', 0.0)
        bbox = tooth.get('bbox', [])
        pixels = tooth.get('pixel_coordinates', [])
        
        # Get tooth name
        tooth_name = TOOTH_NAMES.get(str(tooth_id), f'Tooth {tooth_id}')
        
        # Draw bounding box
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=COLORS['tooth_bbox'],
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            if show_labels:
                label = f"#{tooth_id}"
                if show_confidence:
                    label += f" ({confidence*100:.1f}%)"
                
                ax.text(
                    x1, y1 - 5,
                    label,
                    fontsize=8,
                    color=COLORS['text_fg'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['text_bg'], alpha=0.7),
                    verticalalignment='bottom'
                )
        
        # Draw mask if requested
        if show_mask and pixels:
            for x, y in pixels:
                if 0 <= y < img_height and 0 <= x < img_width:
                    mask_overlay[y, x] = [0, 255, 0, 80]  # Green with alpha
    
    # Apply mask overlay
    if show_mask:
        ax.imshow(mask_overlay)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=COLORS['tooth_bbox'], linewidth=2, label='Tooth Bounding Box'),
    ]
    if show_mask:
        legend_elements.append(patches.Patch(facecolor='#00FF0050', edgecolor='none', label='Tooth Segmentation'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add info text
    info_text = f"Image Size: {img_width}x{img_height}\nTeeth: {len(teeth_data)}"
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to: {output_path}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print tooth summary
    print(f"\n{'─'*60}")
    print("TEETH SUMMARY:")
    print(f"{'─'*60}")
    for tooth in sorted(teeth_data, key=lambda x: float(x.get('confidence', 0)), reverse=True)[:10]:
        tid = tooth.get('tooth_id', '?')
        conf = tooth.get('confidence', 0) * 100
        name = TOOTH_NAMES.get(str(tid), 'Unknown')
        pixels = len(tooth.get('pixel_coordinates', []))
        print(f"  Tooth {tid:>2}: {name:<35} | Conf: {conf:5.1f}% | Pixels: {pixels:,}")
    
    if len(teeth_data) > 10:
        print(f"  ... and {len(teeth_data) - 10} more teeth")


def visualize_week4_json(
    json_path: str,
    image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (20, 12),
    show: bool = True
):
    """
    Visualize Week 4 JSON results (alternative format support).
    """
    data = load_json(json_path)
    if not data:
        return
    
    # Check if this is Week 4 format
    if 'caries_detections' in data or 'teeth_caries_data' in data:
        print("Detected Week 4 format - using appropriate visualization")
        # Handle Week 4 format similarly
        case_number = data.get('case_number', 'Unknown')
        teeth = data.get('tooth_detections', []) or data.get('teeth_caries_data', [])
        caries = data.get('caries_detections', [])
        
        if image_path is None:
            image_path = find_image_for_case(json_path, case_number)
        
        if not image_path or not os.path.exists(image_path):
            print(f"Could not find image for case {case_number}")
            return
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img_rgb)
        ax.set_title(f"Case {case_number} - Week 4 Format", fontsize=16)
        
        # Draw tooth boxes
        for tooth in teeth:
            bbox = tooth.get('bbox', tooth.get('tooth_bbox', {}))
            if isinstance(bbox, dict):
                x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
                x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
            else:
                continue
            
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            
            if show_labels:
                tid = tooth.get('tooth_id', '?')
                ax.text(x1, y1-5, f"#{tid}", fontsize=8, color='white',
                       bbox=dict(facecolor='green', alpha=0.7))
        
        # Draw caries boxes
        for c in caries:
            bbox = c.get('bbox', {})
            if isinstance(bbox, dict):
                x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
                x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
                )
                ax.add_patch(rect)
                
                if show_labels:
                    ctype = c.get('class_name', 'Caries')
                    ax.text(x1, y2+15, ctype, fontsize=8, color='white',
                           bbox=dict(facecolor='red', alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Week 2 dental analysis JSON results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python viz_week2_style.py --case_path "week2/500-segmentation+recognition/case 1/case_1_results.json"
  python viz_week2_style.py --case_path "path/to/results.json" --image_path "path/to/image.png"
  python viz_week2_style.py --case_path "path/to/results.json" --show_mask --output "viz.png"
        """
    )
    
    parser.add_argument(
        '--case_path', '-c',
        type=str,
        required=True,
        help='Path to the Week 2 JSON results file (case_X_results.json)'
    )
    
    parser.add_argument(
        '--image_path', '-i',
        type=str,
        default=None,
        help='Path to the X-ray image (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save the visualization image'
    )
    
    parser.add_argument(
        '--show_mask', '-m',
        action='store_true',
        help='Show tooth segmentation masks (pixel-level)'
    )
    
    parser.add_argument(
        '--no_labels',
        action='store_true',
        help='Hide tooth ID labels'
    )
    
    parser.add_argument(
        '--no_confidence',
        action='store_true',
        help='Hide confidence scores'
    )
    
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display the plot (useful for batch processing)'
    )
    
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[20, 12],
        help='Figure size in inches (width height)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.case_path):
        print(f"Error: JSON file not found: {args.case_path}")
        sys.exit(1)
    
    # Detect format and visualize
    data = load_json(args.case_path)
    
    if 'teeth_data' in data:
        # Week 2 format
        visualize_week2_json(
            json_path=args.case_path,
            image_path=args.image_path,
            output_path=args.output,
            show_mask=args.show_mask,
            show_labels=not args.no_labels,
            show_confidence=not args.no_confidence,
            figsize=tuple(args.figsize),
            show=not args.no_show
        )
    elif 'caries_detections' in data or 'teeth_caries_data' in data:
        # Week 4 format
        visualize_week4_json(
            json_path=args.case_path,
            image_path=args.image_path,
            output_path=args.output,
            show_labels=not args.no_labels,
            figsize=tuple(args.figsize),
            show=not args.no_show
        )
    else:
        print(f"Unknown JSON format. Expected 'teeth_data' (Week 2) or 'caries_detections' (Week 4)")
        sys.exit(1)


if __name__ == "__main__":
    main()
