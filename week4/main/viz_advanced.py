"""
Advanced Dental Visualization Script (v1.0)
=============================================
Visualizes Week 4 Inference Pipeline output with:
- Real tooth segmentation contours (from Detectron2)
- Caries lesion detection overlays
- Clean medical report styling

Author: Data Visualization Expert
Date: 2025
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2


# =============================================================================
# COLOR PALETTE (Medical Imaging Style)
# =============================================================================

# Tooth segmentation colors
TOOTH_FILL_COLOR = '#00CED1'      # Dark Cyan
TOOTH_EDGE_COLOR = '#008B8B'      # Dark Cyan (darker)
TOOTH_ALPHA = 0.3

# Caries colors by type
CARIES_COLORS = {
    'Occlusal': '#FF4444',        # Red
    'Proximal': '#FF6B00',        # Orange-Red
    'Lingual': '#FF1493',         # Deep Pink
    'occlusal': '#FF4444',
    'proximal': '#FF6B00',
    'lingual': '#FF1493',
    'Unknown': '#FF0000'          # Pure Red
}
CARIES_ALPHA = 0.6
CARIES_EDGE_COLOR = '#8B0000'     # Dark Red

# Label styling
LABEL_FONTSIZE_TOOTH = 7
LABEL_FONTSIZE_CARIES = 8
LABEL_BG_COLOR = 'white'
LABEL_BG_ALPHA = 0.85


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_json(json_path: str) -> Dict:
    """Load JSON file with error handling."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_image(img_path: str, grayscale: bool = True) -> np.ndarray:
    """Load image, optionally converting to grayscale."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    if grayscale:
        # Convert to grayscale but keep 3 channels for overlay
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def calculate_centroid(coords: List[List[int]]) -> Tuple[float, float]:
    """Calculate the centroid of a polygon."""
    if not coords:
        return (0, 0)
    
    coords_array = np.array(coords)
    cx = np.mean(coords_array[:, 0])
    cy = np.mean(coords_array[:, 1])
    return (cx, cy)


def bbox_to_coords(bbox: Dict) -> List[List[int]]:
    """Convert bounding box dict to polygon coordinates."""
    x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
    x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def get_bbox_center(bbox: Dict) -> Tuple[float, float]:
    """Get center point of a bounding box."""
    x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
    x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def draw_tooth_segmentations(
    ax: plt.Axes,
    teeth_data: List[Dict],
    show_labels: bool = True
) -> int:
    """
    Draw tooth segmentation polygons on the axes.
    
    Args:
        ax: Matplotlib axes
        teeth_data: List of tooth records from teeth_caries_data
        show_labels: Whether to show tooth ID labels
    
    Returns:
        Number of teeth drawn
    """
    drawn_count = 0
    
    for tooth in teeth_data:
        tooth_id = tooth.get('tooth_id', '?')
        pixel_coords = tooth.get('pixel_coordinates', [])
        tooth_bbox = tooth.get('tooth_bbox', {})
        
        # Determine coordinates to use
        if pixel_coords and len(pixel_coords) > 2:
            # Use real segmentation contour
            coords = np.array(pixel_coords)
            is_segmentation = True
        elif tooth_bbox:
            # Fallback to bounding box
            coords = np.array(bbox_to_coords(tooth_bbox))
            is_segmentation = False
        else:
            continue  # Skip if no coordinates available
        
        # Create and add polygon
        polygon = Polygon(
            coords,
            closed=True,
            facecolor=TOOTH_FILL_COLOR,
            edgecolor=TOOTH_EDGE_COLOR,
            alpha=TOOTH_ALPHA,
            linewidth=1.0 if is_segmentation else 1.5,
            linestyle='-' if is_segmentation else '--'
        )
        ax.add_patch(polygon)
        drawn_count += 1
        
        # Add tooth ID label at centroid
        if show_labels:
            if is_segmentation:
                cx, cy = calculate_centroid(pixel_coords)
            else:
                cx, cy = get_bbox_center(tooth_bbox)
            
            ax.text(
                cx, cy, str(tooth_id),
                fontsize=LABEL_FONTSIZE_TOOTH,
                fontweight='bold',
                color='#006666',
                ha='center', va='center',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor=TOOTH_EDGE_COLOR,
                    alpha=0.7
                )
            )
    
    return drawn_count


def draw_caries_detections(
    ax: plt.Axes,
    caries_data: List[Dict],
    show_labels: bool = True
) -> int:
    """
    Draw caries detection overlays on the axes.
    
    Args:
        ax: Matplotlib axes
        caries_data: List of caries detection records
        show_labels: Whether to show caries labels
    
    Returns:
        Number of caries drawn
    """
    drawn_count = 0
    
    for caries in caries_data:
        class_name = caries.get('class_name', 'Unknown')
        confidence = caries.get('confidence', 0.0)
        bbox = caries.get('bbox', {})
        
        if not bbox:
            continue
        
        # Get color based on caries type
        color = CARIES_COLORS.get(class_name, CARIES_COLORS['Unknown'])
        
        # Get bbox coordinates
        x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
        x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
        width = x2 - x1
        height = y2 - y1
        
        # Draw filled rectangle for caries
        rect = FancyBboxPatch(
            (x1, y1), width, height,
            boxstyle="round,pad=0.02,rounding_size=3",
            facecolor=color,
            edgecolor=CARIES_EDGE_COLOR,
            alpha=CARIES_ALPHA,
            linewidth=2
        )
        ax.add_patch(rect)
        drawn_count += 1
        
        # Add label with class name and confidence
        if show_labels:
            # Position label above the caries box
            label_x = (x1 + x2) / 2
            label_y = y1 - 5  # Above the box
            
            # Format confidence as percentage
            conf_pct = f"{confidence * 100:.1f}%" if confidence < 1 else f"{confidence:.1f}%"
            label_text = f"{class_name}\n{conf_pct}"
            
            ax.text(
                label_x, label_y, label_text,
                fontsize=LABEL_FONTSIZE_CARIES,
                fontweight='bold',
                color='#8B0000',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=LABEL_BG_COLOR,
                    edgecolor=color,
                    alpha=LABEL_BG_ALPHA,
                    linewidth=1.5
                )
            )
    
    return drawn_count


def create_legend(ax: plt.Axes, num_teeth: int, num_caries: int):
    """Create a custom legend for the visualization."""
    legend_elements = [
        mpatches.Patch(
            facecolor=TOOTH_FILL_COLOR, 
            edgecolor=TOOTH_EDGE_COLOR,
            alpha=TOOTH_ALPHA, 
            label=f'Tooth Segmentation ({num_teeth})'
        ),
        mpatches.Patch(
            facecolor='#FF4444', 
            edgecolor=CARIES_EDGE_COLOR,
            alpha=CARIES_ALPHA, 
            label=f'Caries Detection ({num_caries})'
        ),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=9,
        framealpha=0.9,
        edgecolor='gray'
    )


def visualize_dental_analysis(
    json_path: str,
    img_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 150
) -> None:
    """
    Main visualization function.
    
    Args:
        json_path: Path to standardized JSON result file
        img_path: Path to original OPG image
        output_path: Optional path to save the result image
        show_plot: Whether to display the plot
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved image
    """
    print(f"\n{'='*60}")
    print("ADVANCED DENTAL VISUALIZATION")
    print(f"{'='*60}")
    
    # Load data
    print(f"\n[1] Loading JSON: {json_path}")
    data = load_json(json_path)
    
    print(f"[2] Loading Image: {img_path}")
    image = load_image(img_path, grayscale=True)
    
    # Extract data
    case_number = data.get('case_number', 'Unknown')
    teeth_data = data.get('teeth_caries_data', [])
    
    # Get caries from original_week4_data (more detailed)
    original_data = data.get('original_week4_data', {})
    caries_data = original_data.get('caries_detections', [])
    
    # Get summary info
    summary = data.get('summary', {})
    model_info = data.get('model_info', {})
    
    print(f"\n[3] Data Summary:")
    print(f"    - Case Number: {case_number}")
    print(f"    - Teeth Detected: {len(teeth_data)}")
    print(f"    - Caries Detected: {len(caries_data)}")
    print(f"    - Segmentation Model: {model_info.get('crop_seg_model', 'N/A')}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Display image
    ax.imshow(image, cmap='gray', aspect='equal')
    
    # Draw tooth segmentations (background layer)
    print(f"\n[4] Drawing tooth segmentations...")
    num_teeth_drawn = draw_tooth_segmentations(ax, teeth_data, show_labels=True)
    print(f"    - Drew {num_teeth_drawn} tooth polygons")
    
    # Draw caries detections (foreground layer)
    print(f"[5] Drawing caries detections...")
    num_caries_drawn = draw_caries_detections(ax, caries_data, show_labels=True)
    print(f"    - Drew {num_caries_drawn} caries overlays")
    
    # Create legend
    create_legend(ax, num_teeth_drawn, num_caries_drawn)
    
    # Style the plot (clean medical report look)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Title
    title_text = f"Dental Analysis - Case {case_number}"
    subtitle_text = f"Teeth: {num_teeth_drawn} | Caries: {num_caries_drawn}"
    ax.set_title(f"{title_text}\n{subtitle_text}", fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        print(f"\n[6] Saving visualization: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"    - Saved successfully!")
    
    # Show plot
    if show_plot:
        print(f"\n[7] Displaying plot...")
        plt.show()
    else:
        plt.close()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Advanced Dental Visualization - Week 4 Pipeline Output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python viz_advanced.py --json_path results/case_1.json --img_path images/case_1.png
  
  # Save to file
  python viz_advanced.py --json_path results/case_1.json --img_path images/case_1.png --output viz_case_1.png
  
  # No display, just save
  python viz_advanced.py --json_path results/case_1.json --img_path images/case_1.png --output viz_case_1.png --no_show

Features:
  - Real tooth segmentation contours (from Detectron2)
  - Caries lesion detection overlays with confidence scores
  - Clean medical report styling
  - Automatic fallback to bounding boxes if segmentation unavailable
        """
    )
    
    parser.add_argument(
        '--json_path', '-j',
        type=str,
        required=True,
        help='Path to the standardized JSON result file (from standardize_week4.py)'
    )
    
    parser.add_argument(
        '--img_path', '-i',
        type=str,
        required=True,
        help='Path to the original OPG image'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='(Optional) Path to save the visualization image'
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
        default=[16, 10],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 16 10)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved image (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Run visualization
    visualize_dental_analysis(
        json_path=args.json_path,
        img_path=args.img_path,
        output_path=args.output,
        show_plot=not args.no_show,
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
