"""
Visualization Utilities for Caries Surface Classification
==========================================================

This module provides visualization functions for:
1. Displaying PCA alignment results
2. Showing classification zones
3. Creating debug visualizations
4. Batch visualization generation

Author: Computer Vision Engineer
Date: 2026
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

from caries_surface_classifier import (
    compute_centroid,
    perform_pca,
    create_rotation_matrix,
    rotate_points,
    get_bounding_box,
    is_upper_jaw,
    OCCLUSAL_ZONE_THRESHOLD,
    PROXIMAL_ZONE_THRESHOLD
)


# =============================================================================
# Color Schemes
# =============================================================================

SURFACE_COLORS = {
    0: '#E74C3C',   # Occlusal - Red
    1: '#3498DB',   # Proximal - Blue
    2: '#2ECC71',   # Lingual/Other - Green
    -1: '#95A5A6'   # Invalid - Gray
}

SURFACE_LABELS = {
    0: 'Occlusal',
    1: 'Proximal',
    2: 'Lingual/Other'
}


# =============================================================================
# Single Tooth Visualization
# =============================================================================

def visualize_pca_alignment(
    tooth_poly: List[List[float]],
    caries_poly: List[List[float]],
    tooth_id: str,
    title: str = "",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a visualization showing PCA alignment process.
    
    Left panel: Original tooth and caries positions
    Right panel: PCA-aligned positions with major/minor axes shown
    
    Args:
        tooth_poly: Tooth polygon coordinates
        caries_poly: Caries coordinates
        tooth_id: FDI tooth identifier
        title: Optional title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    tooth_points = np.array(tooth_poly, dtype=np.float64)
    caries_points = np.array(caries_poly, dtype=np.float64)
    
    # Compute PCA
    centroid = compute_centroid(tooth_points)
    mean, eigenvectors, rotation_angle = perform_pca(tooth_points)
    
    # Create rotation matrix and rotate
    rot_matrix = create_rotation_matrix(rotation_angle, centroid)
    rotated_tooth = rotate_points(tooth_points, rot_matrix)
    rotated_caries = rotate_points(caries_points, rot_matrix)
    
    # =========================================================================
    # Left: Original with PCA axes
    # =========================================================================
    ax1 = axes[0]
    ax1.set_title('Original with PCA Axes')
    
    # Plot tooth
    ax1.fill(tooth_points[:, 0], tooth_points[:, 1], 
             alpha=0.3, color='blue', label='Tooth')
    ax1.plot(np.append(tooth_points[:, 0], tooth_points[0, 0]),
             np.append(tooth_points[:, 1], tooth_points[0, 1]),
             'b-', linewidth=2)
    
    # Plot caries
    ax1.scatter(caries_points[:, 0], caries_points[:, 1], 
               c='red', s=15, alpha=0.7, label='Caries')
    
    # Plot centroid
    ax1.plot(centroid[0], centroid[1], 'ko', markersize=10, label='Centroid')
    
    # Plot PCA axes
    scale = np.max([np.ptp(tooth_points[:, 0]), np.ptp(tooth_points[:, 1])]) * 0.5
    
    # Major axis (first eigenvector)
    major = eigenvectors[0] * scale
    ax1.annotate('', xy=(centroid[0] + major[0], centroid[1] + major[1]),
                 xytext=(centroid[0], centroid[1]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.annotate('', xy=(centroid[0] - major[0], centroid[1] - major[1]),
                 xytext=(centroid[0], centroid[1]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Minor axis (second eigenvector)
    minor = eigenvectors[1] * scale * 0.5
    ax1.annotate('', xy=(centroid[0] + minor[0], centroid[1] + minor[1]),
                 xytext=(centroid[0], centroid[1]),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend(loc='best')
    ax1.axis('equal')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Add rotation info
    ax1.text(0.02, 0.02, f'Rotation: {math.degrees(rotation_angle):.1f}°',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # =========================================================================
    # Right: Rotated/Aligned
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title('PCA-Aligned (Major Axis Vertical)')
    
    # Plot rotated tooth
    ax2.fill(rotated_tooth[:, 0], rotated_tooth[:, 1], 
             alpha=0.3, color='blue', label='Tooth (aligned)')
    ax2.plot(np.append(rotated_tooth[:, 0], rotated_tooth[0, 0]),
             np.append(rotated_tooth[:, 1], rotated_tooth[0, 1]),
             'b-', linewidth=2)
    
    # Plot rotated caries
    ax2.scatter(rotated_caries[:, 0], rotated_caries[:, 1], 
               c='red', s=15, alpha=0.7, label='Caries (aligned)')
    
    # Bounding box
    bbox = get_bounding_box(rotated_tooth)
    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                     fill=False, edgecolor='black', linestyle='--', linewidth=2)
    ax2.add_patch(rect)
    
    # Draw reference axes
    rot_centroid = compute_centroid(rotated_tooth)
    ax2.axhline(y=rot_centroid[1], color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=rot_centroid[0], color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.legend(loc='best')
    ax2.axis('equal')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    
    # Add tooth ID and jaw type
    jaw = "Upper" if is_upper_jaw(tooth_id) else "Lower"
    ax2.text(0.02, 0.02, f'Tooth {tooth_id} ({jaw} Jaw)',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_classification_zones(
    tooth_poly: List[List[float]],
    caries_poly: List[List[float]],
    tooth_id: str,
    classification: int,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Create a visualization showing classification zones on aligned tooth.
    
    Shows:
    - Aligned tooth polygon
    - Occlusal zone (red)
    - Proximal zones (blue)
    - Central/Lingual zone (green)
    - Caries position with classification result
    
    Args:
        tooth_poly: Tooth polygon coordinates
        caries_poly: Caries coordinates
        tooth_id: FDI tooth identifier
        classification: Classification result (0, 1, or 2)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    tooth_points = np.array(tooth_poly, dtype=np.float64)
    caries_points = np.array(caries_poly, dtype=np.float64)
    
    # PCA alignment
    centroid = compute_centroid(tooth_points)
    _, _, rotation_angle = perform_pca(tooth_points)
    rot_matrix = create_rotation_matrix(rotation_angle, centroid)
    
    rotated_tooth = rotate_points(tooth_points, rot_matrix)
    rotated_caries = rotate_points(caries_points, rot_matrix)
    
    # Bounding box
    bbox_x, bbox_y, bbox_w, bbox_h = get_bounding_box(rotated_tooth)
    
    # Determine jaw type
    upper_jaw = is_upper_jaw(tooth_id)
    
    # Calculate zone boundaries
    left_bound = bbox_x + bbox_w * PROXIMAL_ZONE_THRESHOLD
    right_bound = bbox_x + bbox_w * (1 - PROXIMAL_ZONE_THRESHOLD)
    
    if upper_jaw:
        occlusal_bound = bbox_y + bbox_h * (1 - OCCLUSAL_ZONE_THRESHOLD)
        occlusal_y_start = occlusal_bound
        occlusal_y_end = bbox_y + bbox_h
    else:
        occlusal_bound = bbox_y + bbox_h * OCCLUSAL_ZONE_THRESHOLD
        occlusal_y_start = bbox_y
        occlusal_y_end = occlusal_bound
    
    # Draw zones
    # Occlusal zone (red)
    ax.fill([bbox_x, bbox_x + bbox_w, bbox_x + bbox_w, bbox_x],
            [occlusal_y_start, occlusal_y_start, occlusal_y_end, occlusal_y_end],
            alpha=0.2, color='red', label='Occlusal Zone')
    
    # Left proximal (blue)
    ax.fill([bbox_x, left_bound, left_bound, bbox_x],
            [bbox_y, bbox_y, bbox_y + bbox_h, bbox_y + bbox_h],
            alpha=0.2, color='blue', label='Proximal Zone')
    
    # Right proximal (blue)
    ax.fill([right_bound, bbox_x + bbox_w, bbox_x + bbox_w, right_bound],
            [bbox_y, bbox_y, bbox_y + bbox_h, bbox_y + bbox_h],
            alpha=0.2, color='blue')
    
    # Plot tooth outline
    ax.fill(rotated_tooth[:, 0], rotated_tooth[:, 1], 
            alpha=0.4, color='lightgray', edgecolor='black', linewidth=2)
    
    # Plot caries with classification color
    color = SURFACE_COLORS.get(classification, '#95A5A6')
    ax.scatter(rotated_caries[:, 0], rotated_caries[:, 1], 
              c=color, s=30, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Mark caries centroid
    caries_cent = compute_centroid(rotated_caries)
    ax.plot(caries_cent[0], caries_cent[1], '*', color=color, 
            markersize=25, markeredgecolor='black', markeredgewidth=2)
    
    # Bounding box outline
    rect = Rectangle((bbox_x, bbox_y), bbox_w, bbox_h,
                     fill=False, edgecolor='black', linestyle='--', linewidth=2)
    ax.add_patch(rect)
    
    # Zone boundary lines
    ax.axvline(x=left_bound, color='blue', linestyle=':', linewidth=1.5)
    ax.axvline(x=right_bound, color='blue', linestyle=':', linewidth=1.5)
    ax.axhline(y=occlusal_bound, color='red', linestyle=':', linewidth=1.5)
    
    # Labels
    surface_name = SURFACE_LABELS.get(classification, 'Unknown')
    jaw_type = "Upper" if upper_jaw else "Lower"
    
    ax.set_title(f'Tooth {tooth_id} ({jaw_type} Jaw)\n'
                 f'Classification: {surface_name} (Class {classification})',
                 fontsize=12, fontweight='bold')
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Add relative position info
    rel_x = (caries_cent[0] - bbox_x) / bbox_w if bbox_w > 0 else 0
    rel_y = (caries_cent[1] - bbox_y) / bbox_h if bbox_h > 0 else 0
    
    info_text = f"Relative Position:\nX: {rel_x:.2f}, Y: {rel_y:.2f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    return fig


def visualize_all_teeth_in_case(
    teeth_data: List[Dict],
    case_num: int,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a grid visualization of all classified teeth in a case.
    
    Args:
        teeth_data: List of dictionaries with tooth_poly, caries_poly, 
                   tooth_id, and classification
        case_num: Case number
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_teeth = len(teeth_data)
    if n_teeth == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, 'No teeth with caries in this case',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Calculate grid size
    cols = min(4, n_teeth)
    rows = (n_teeth + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, tooth_info in enumerate(teeth_data):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        tooth_poly = tooth_info['tooth_poly']
        caries_poly = tooth_info['caries_poly']
        tooth_id = tooth_info['tooth_id']
        classification = tooth_info.get('classification', -1)
        
        # Process data
        tooth_points = np.array(tooth_poly, dtype=np.float64)
        caries_points = np.array(caries_poly, dtype=np.float64)
        
        centroid = compute_centroid(tooth_points)
        _, _, rotation_angle = perform_pca(tooth_points)
        rot_matrix = create_rotation_matrix(rotation_angle, centroid)
        
        rotated_tooth = rotate_points(tooth_points, rot_matrix)
        rotated_caries = rotate_points(caries_points, rot_matrix)
        
        # Plot
        color = SURFACE_COLORS.get(classification, '#95A5A6')
        surface_name = SURFACE_LABELS.get(classification, 'Unknown')
        
        ax.fill(rotated_tooth[:, 0], rotated_tooth[:, 1], 
                alpha=0.3, color='lightgray', edgecolor='black')
        ax.scatter(rotated_caries[:, 0], rotated_caries[:, 1], 
                  c=color, s=10, alpha=0.8)
        
        jaw = "U" if is_upper_jaw(tooth_id) else "L"
        ax.set_title(f'Tooth {tooth_id} ({jaw})\n{surface_name}', fontsize=10)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_teeth, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    fig.suptitle(f'Case {case_num} - Caries Surface Classification', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Save Utility
# =============================================================================

def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 150):
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)


# =============================================================================
# Demo
# =============================================================================

def demo_visualization():
    """
    Demonstrate visualization functions with sample data.
    """
    print("Creating demo visualizations...")
    
    # Sample tooth polygon (molar-like shape)
    tooth_poly = [
        [100, 40],
        [140, 35],
        [160, 60],
        [165, 120],
        [155, 180],
        [130, 200],
        [110, 195],
        [90, 170],
        [85, 110],
        [90, 60],
    ]
    
    # Sample caries at different positions
    test_cases = [
        ('16', [[125, 185], [130, 190], [135, 188], [128, 192]], 0),  # Upper molar, occlusal
        ('16', [[92, 120], [95, 125], [90, 130]], 1),                  # Upper molar, proximal left
        ('46', [[125, 50], [130, 55], [128, 48]], 0),                  # Lower molar, occlusal
    ]
    
    output_dir = Path(r"C:\Users\jaopi\Desktop\SP\week5\demo_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (tooth_id, caries, expected_class) in enumerate(test_cases):
        # PCA alignment visualization
        fig1 = visualize_pca_alignment(tooth_poly, caries, tooth_id,
                                        title=f'Demo {i+1}: Tooth {tooth_id}')
        save_figure(fig1, output_dir / f'demo_{i+1}_pca_alignment.png')
        
        # Classification zones visualization
        fig2 = visualize_classification_zones(tooth_poly, caries, tooth_id, expected_class)
        save_figure(fig2, output_dir / f'demo_{i+1}_classification.png')
    
    print(f"Demo visualizations saved to: {output_dir}")


if __name__ == "__main__":
    demo_visualization()
