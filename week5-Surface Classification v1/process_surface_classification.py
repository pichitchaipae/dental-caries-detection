"""
Process Caries Surface Classification for All Cases
====================================================

This script processes dental caries data from week2 and week3 outputs,
applying the PCA-based surface classification algorithm to determine
which tooth surface each caries lesion is located on.

Input Data:
- week2/500-segmentation+recognition: Tooth segmentation with polygon boundaries
- week3/dental_analysis_output: Caries mapping with pixel coordinates

Output:
- week5/surface_classification_output: Classification results per case
- CSV summary of all classifications

Author: Computer Vision Engineer
Date: 2026
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection

# Import local classifier module
from caries_surface_classifier import (
    classify_caries_surface,
    classify_caries_surface_detailed,
    get_surface_name,
    polygon_from_points,
    SURFACE_NAMES
)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")
WEEK2_DIR = BASE_DIR / "week2" / "500-segmentation+recognition"
WEEK3_DIR = BASE_DIR / "week3" / "dental_analysis_output"
WEEK5_OUTPUT_DIR = BASE_DIR / "week5" / "surface_classification_output"
MATERIAL_DIR = BASE_DIR / "material" / "500 cases with annotation"


# =============================================================================
# Surface Name Mapping for JSON Output
# =============================================================================

SURFACE_NAME_MAP = {
    0: "Occlusal",
    1: "Proximal", 
    2: "Other"
}


# =============================================================================
# Caries Position Detail Logic (Mesial/Distal)
# =============================================================================

def get_caries_position_detail(tooth_id: str, classification: int, rel_x: float) -> str:
    """
    Determine the detailed caries position based on ISO 3950 quadrants and rel_x.
    
    For Proximal surfaces, determines if cavity is Mesial (towards midline) or 
    Distal (away from midline) based on:
    
    - Quadrant 1 (11-18) & Quadrant 4 (41-48): Right side of patient
        - rel_x < 0.5 (Left side of bounding box) → Mesial (towards midline)
        - rel_x > 0.5 (Right side of bounding box) → Distal
        
    - Quadrant 2 (21-28) & Quadrant 3 (31-38): Left side of patient  
        - rel_x > 0.5 (Right side of bounding box) → Mesial (towards midline)
        - rel_x < 0.5 (Left side of bounding box) → Distal
    
    Args:
        tooth_id: FDI tooth notation (e.g., "11", "46")
        classification: Surface classification (0=Occlusal, 1=Proximal, 2=Other)
        rel_x: Relative X position of caries centroid in bounding box (0.0-1.0)
        
    Returns:
        Position detail string: "Mesial", "Distal", or "Center"
    """
    # For non-Proximal surfaces, return "Center"
    if classification != 1:
        return "Center"
    
    # Parse quadrant from tooth_id (first digit)
    try:
        quadrant = int(tooth_id[0])
    except (ValueError, IndexError):
        return "Center"  # Default if parsing fails
    
    # Quadrant 1 (11-18) & Quadrant 4 (41-48): Right side of patient's mouth
    # In panoramic X-ray view: these appear on LEFT side of image
    # Mesial = towards midline = LEFT side of tooth = rel_x < 0.5
    if quadrant in [1, 4]:
        if rel_x < 0.5:
            return "Mesial"
        else:
            return "Distal"
    
    # Quadrant 2 (21-28) & Quadrant 3 (31-38): Left side of patient's mouth
    # In panoramic X-ray view: these appear on RIGHT side of image
    # Mesial = towards midline = RIGHT side of tooth = rel_x > 0.5
    elif quadrant in [2, 3]:
        if rel_x > 0.5:
            return "Mesial"
        else:
            return "Distal"
    
    return "Center"  # Default fallback


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_tooth_segmentation_data(case_num: int) -> dict:
    """
    Load tooth segmentation data from week2 output.
    
    The week2 JSON contains tooth polygons and pixel coordinates.
    Since the files are large (>50MB), we need to handle them carefully.
    
    Args:
        case_num: Case number to load
        
    Returns:
        Dictionary with teeth_data containing polygons
    """
    case_folder = WEEK2_DIR / f"case {case_num}"
    json_path = case_folder / f"case_{case_num}_results.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading week2 data for case {case_num}: {e}")
        return None


def load_caries_mapping_data(case_num: int) -> dict:
    """
    Load caries mapping data from week3 output.
    
    Contains caries pixel coordinates mapped to each tooth.
    
    Args:
        case_num: Case number to load
        
    Returns:
        Dictionary with teeth_caries_data
    """
    case_folder = WEEK3_DIR / f"case {case_num}"
    json_path = case_folder / f"case_{case_num}_caries_mapping.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading week3 data for case {case_num}: {e}")
        return None


def extract_tooth_polygon(tooth_data: dict) -> list:
    """
    Extract the boundary polygon from tooth data.
    
    The week2 data stores pixel coordinates in 'pixel_coordinates' field.
    We compute the convex hull to get the boundary polygon.
    
    Args:
        tooth_data: Dictionary containing tooth information
        
    Returns:
        List of [x, y] coordinates forming the polygon boundary
    """
    # Try 'polygon' field first (if pre-computed boundary points exist)
    if 'polygon' in tooth_data and tooth_data['polygon']:
        return tooth_data['polygon']
    
    # Compute polygon from pixel_coordinates using convex hull
    if 'pixel_coordinates' in tooth_data and tooth_data['pixel_coordinates']:
        pixels = np.array(tooth_data['pixel_coordinates'], dtype=np.float32)
        if len(pixels) >= 3:
            hull = cv2.convexHull(pixels)
            return hull.reshape(-1, 2).tolist()
    
    return None


# =============================================================================
# Visualization Functions
# =============================================================================

def create_classification_visualization(
    case_num: int,
    tooth_id: str,
    tooth_poly: list,
    caries_poly: list,
    classification_result: dict,
    output_path: Path
):
    """
    Create a visualization showing the PCA alignment and classification zones.
    
    This generates a figure with:
    - Left: Original tooth and caries positions
    - Right: Rotated/aligned tooth with classification zones marked
    
    Args:
        case_num: Case number
        tooth_id: FDI tooth ID
        tooth_poly: Original tooth polygon
        caries_poly: Original caries polygon/points
        classification_result: Detailed classification result dictionary
        output_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Color mapping for surfaces
    surface_colors = {
        0: '#FF6B6B',   # Occlusal - Red
        1: '#4ECDC4',   # Proximal - Teal
        2: '#95E1D3',   # Lingual - Light green
        -1: '#CCCCCC'   # Invalid - Gray
    }
    
    classification = classification_result.get('classification', -1)
    surface_name = classification_result.get('surface_name', 'Unknown')
    color = surface_colors.get(classification, '#CCCCCC')
    
    # =========================================================================
    # Left Plot: Original positions
    # =========================================================================
    ax1 = axes[0]
    ax1.set_title(f'Case {case_num} - Tooth {tooth_id} (Original)')
    
    # Plot tooth polygon
    if tooth_poly:
        tooth_array = np.array(tooth_poly)
        ax1.fill(tooth_array[:, 0], tooth_array[:, 1], 
                 alpha=0.3, color='blue', label='Tooth')
        ax1.plot(np.append(tooth_array[:, 0], tooth_array[0, 0]),
                 np.append(tooth_array[:, 1], tooth_array[0, 1]),
                 'b-', linewidth=2)
    
    # Plot caries
    if caries_poly:
        caries_array = np.array(caries_poly)
        ax1.scatter(caries_array[:, 0], caries_array[:, 1], 
                   c='red', s=10, alpha=0.6, label='Caries')
        
        # Draw convex hull of caries
        if len(caries_array) >= 3:
            hull = cv2.convexHull(caries_array.astype(np.float32))
            hull_pts = hull.reshape(-1, 2)
            ax1.fill(hull_pts[:, 0], hull_pts[:, 1], 
                     alpha=0.3, color='red')
    
    # Mark centroids
    if 'tooth_centroid' in classification_result:
        cx, cy = classification_result['tooth_centroid']
        ax1.plot(cx, cy, 'b*', markersize=15, label='Tooth Centroid')
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend(loc='upper right')
    ax1.axis('equal')
    ax1.invert_yaxis()  # Image coordinates (Y increases downward)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Right Plot: Rotated positions with zones
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title(f'PCA-Aligned (Rotation: {classification_result.get("rotation_angle_deg", 0):.1f}°)\n'
                  f'Classification: {surface_name}')
    
    # Get rotated polygons
    rotated_tooth = classification_result.get('rotated_tooth_polygon', [])
    rotated_caries = classification_result.get('rotated_caries_polygon', [])
    bbox = classification_result.get('bounding_box', {})
    
    if rotated_tooth:
        tooth_array = np.array(rotated_tooth)
        ax2.fill(tooth_array[:, 0], tooth_array[:, 1], 
                 alpha=0.3, color='blue', label='Tooth (rotated)')
        ax2.plot(np.append(tooth_array[:, 0], tooth_array[0, 0]),
                 np.append(tooth_array[:, 1], tooth_array[0, 1]),
                 'b-', linewidth=2)
    
    # Draw bounding box
    if bbox:
        rect = Rectangle(
            (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
            fill=False, edgecolor='black', linestyle='--', linewidth=2
        )
        ax2.add_patch(rect)
        
        # Draw zone boundaries
        jaw_type = classification_result.get('jaw_type', 'upper')
        zone_checks = classification_result.get('zone_checks', {})
        occ_thresh = zone_checks.get('occlusal_threshold', 0.2)
        prox_thresh = zone_checks.get('proximal_threshold', 0.2)
        
        # Proximal zones (left/right)
        left_x = bbox['x'] + bbox['width'] * prox_thresh
        right_x = bbox['x'] + bbox['width'] * (1 - prox_thresh)
        
        ax2.axvline(x=left_x, color='teal', linestyle=':', alpha=0.7)
        ax2.axvline(x=right_x, color='teal', linestyle=':', alpha=0.7)
        
        # Occlusal zone (top or bottom depending on jaw)
        if jaw_type == 'upper':
            # Occlusal at bottom
            occ_y = bbox['y'] + bbox['height'] * (1 - occ_thresh)
            ax2.axhline(y=occ_y, color='red', linestyle=':', alpha=0.7, label='Occlusal Zone')
            ax2.fill_between([bbox['x'], bbox['x'] + bbox['width']], 
                             occ_y, bbox['y'] + bbox['height'],
                             alpha=0.15, color='red')
        else:
            # Occlusal at top
            occ_y = bbox['y'] + bbox['height'] * occ_thresh
            ax2.axhline(y=occ_y, color='red', linestyle=':', alpha=0.7, label='Occlusal Zone')
            ax2.fill_between([bbox['x'], bbox['x'] + bbox['width']], 
                             bbox['y'], occ_y,
                             alpha=0.15, color='red')
        
        # Shade proximal zones
        ax2.fill_between([bbox['x'], left_x], 
                         bbox['y'], bbox['y'] + bbox['height'],
                         alpha=0.1, color='teal')
        ax2.fill_between([right_x, bbox['x'] + bbox['width']], 
                         bbox['y'], bbox['y'] + bbox['height'],
                         alpha=0.1, color='teal')
    
    # Plot rotated caries
    if rotated_caries:
        caries_array = np.array(rotated_caries)
        ax2.scatter(caries_array[:, 0], caries_array[:, 1], 
                   c=color, s=20, alpha=0.8, label='Caries', edgecolors='black')
        
        # Mark caries centroid
        caries_cent = classification_result.get('caries_centroid_rotated')
        if caries_cent:
            ax2.plot(caries_cent[0], caries_cent[1], 
                    '*', color=color, markersize=20, 
                    markeredgecolor='black', markeredgewidth=1,
                    label=f'Caries Center ({surface_name})')
    
    # Add relative position info
    rel_pos = classification_result.get('relative_position', {})
    if rel_pos:
        info_text = f"Rel. Position:\nX: {rel_pos.get('rel_x', 0):.2f}\nY: {rel_pos.get('rel_y', 0):.2f}"
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.legend(loc='upper right')
    ax2.axis('equal')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_case(case_num: int, save_viz: bool = True) -> list:
    """
    Process surface classification for all teeth with caries in a single case.
    
    Args:
        case_num: Case number to process
        save_viz: Whether to save visualization images
        
    Returns:
        List of classification results for all carious teeth in the case
    """
    results = []
    
    # Load data
    tooth_data = load_tooth_segmentation_data(case_num)
    caries_data = load_caries_mapping_data(case_num)
    
    if tooth_data is None or caries_data is None:
        return results
    
    # Create output directory for visualizations and JSON
    case_output_dir = WEEK5_OUTPUT_DIR / f"case {case_num}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build lookup for tooth polygons and confidence by tooth_id
    tooth_polygon_lookup = {}
    tooth_confidence_lookup = {}
    for tooth in tooth_data.get('teeth_data', []):
        tid = tooth.get('tooth_id')
        if tid:
            poly = extract_tooth_polygon(tooth)
            if poly:
                tooth_polygon_lookup[tid] = poly
            # Store confidence from week2 data
            tooth_confidence_lookup[tid] = tooth.get('confidence', 0.0)
    
    # JSON output data structure for frontend
    json_teeth_data = []
    
    # Process each tooth with caries
    for caries_entry in caries_data.get('teeth_caries_data', []):
        if not caries_entry.get('has_caries', False):
            continue
        
        tooth_id = caries_entry.get('tooth_id')
        caries_coords = caries_entry.get('caries_coordinates', [])
        
        if not tooth_id or not caries_coords:
            continue
        
        # Get tooth polygon
        tooth_poly = tooth_polygon_lookup.get(tooth_id)
        if not tooth_poly:
            # Try to find from pixel_coordinates in caries data
            continue
        
        # Classify
        try:
            detailed_result = classify_caries_surface_detailed(
                tooth_id, tooth_poly, caries_coords
            )
            
            classification = detailed_result['classification']
            rel_x = detailed_result.get('relative_position', {}).get('rel_x', 0.5)
            rel_y = detailed_result.get('relative_position', {}).get('rel_y', 0.5)
            rotation_angle = detailed_result.get('rotation_angle_deg', 0)
            
            # Get surface name for JSON (clean mapping)
            caries_surface = SURFACE_NAME_MAP.get(classification, "Other")
            
            # Determine position detail (Mesial/Distal/Center)
            caries_position_detail = get_caries_position_detail(
                tooth_id, classification, rel_x
            )
            
            # Get confidence from week2 data
            confidence = tooth_confidence_lookup.get(tooth_id, 0.0)
            
            # Build result entry for CSV
            result_entry = {
                'case_number': case_num,
                'tooth_id': tooth_id,
                'classification': classification,
                'surface_name': detailed_result['surface_name'],
                'jaw_type': detailed_result.get('jaw_type', ''),
                'rotation_angle_deg': rotation_angle,
                'rel_x': rel_x,
                'rel_y': rel_y,
                'caries_pixels': caries_entry.get('caries_pixels', 0),
                'caries_percentage': caries_entry.get('caries_percentage', 0),
                'caries_position_detail': caries_position_detail
            }
            results.append(result_entry)
            
            # Build JSON entry for frontend
            json_tooth_entry = {
                'tooth_id': tooth_id,
                'has_caries': True,
                'confidence': round(confidence, 4),
                'caries_surface': caries_surface,
                'caries_position_detail': caries_position_detail,
                'rotation_angle': round(rotation_angle, 2)
            }
            json_teeth_data.append(json_tooth_entry)
            
            # Save visualization
            if save_viz:
                viz_path = case_output_dir / f"tooth_{tooth_id}_classification.png"
                create_classification_visualization(
                    case_num, tooth_id, tooth_poly, caries_coords,
                    detailed_result, viz_path
                )
        
        except Exception as e:
            print(f"Error classifying case {case_num}, tooth {tooth_id}: {e}")
    
    # Save JSON file for frontend (Dental Chart/Odontogram)
    if json_teeth_data:
        json_output = {
            'case_number': case_num,
            'teeth_data': json_teeth_data
        }
        json_path = case_output_dir / f"case_{case_num}_diagnosis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    return results


def process_all_cases(
    start_case: int = 1,
    end_case: int = 500,
    save_viz: bool = False
) -> pd.DataFrame:
    """
    Process surface classification for all cases (batch processing 1-500).
    
    For each case:
    1. Loads tooth polygon data from week2
    2. Loads caries mapping data from week3
    3. Classifies caries surfaces using PCA+OBB
    4. Generates JSON diagnosis file for frontend (Dental Chart/Odontogram)
    5. Appends results to master CSV
    
    Handles missing cases gracefully by logging and skipping.
    
    Args:
        start_case: First case number (default: 1)
        end_case: Last case number (default: 500)
        save_viz: Whether to save visualization images
        
    Returns:
        DataFrame with all classification results
    """
    all_results = []
    skipped_cases = []
    processed_cases = 0
    json_files_created = 0
    total_teeth_classified = 0
    
    total_cases = end_case - start_case + 1
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING: Dental Caries Surface Classification")
    print(f"{'='*60}")
    print(f"Processing cases {start_case} to {end_case} ({total_cases} total)")
    print(f"Output directory: {WEEK5_OUTPUT_DIR}")
    print(f"Save visualizations: {save_viz}")
    print(f"{'='*60}\n")
    
    # Create progress bar with detailed description
    pbar = tqdm(
        range(start_case, end_case + 1), 
        desc="Processing cases",
        unit="case",
        ncols=100
    )
    
    for case_num in pbar:
        # Update progress bar description
        pbar.set_description(f"Processing Case {case_num}/{end_case}")
        
        try:
            # Check if required data exists before processing
            week2_path = WEEK2_DIR / f"case {case_num}" / f"case_{case_num}_results.json"
            week3_path = WEEK3_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
            
            if not week2_path.exists():
                skipped_cases.append({'case': case_num, 'reason': 'Week2 data missing'})
                continue
                
            if not week3_path.exists():
                skipped_cases.append({'case': case_num, 'reason': 'Week3 data missing'})
                continue
            
            # Process the case
            case_results = process_single_case(case_num, save_viz=save_viz)
            
            if case_results:
                all_results.extend(case_results)
                processed_cases += 1
                total_teeth_classified += len(case_results)
                
                # Check if JSON was created
                json_path = WEEK5_OUTPUT_DIR / f"case {case_num}" / f"case_{case_num}_diagnosis.json"
                if json_path.exists():
                    json_files_created += 1
            else:
                # Case exists but no caries found
                skipped_cases.append({'case': case_num, 'reason': 'No caries detected'})
                
        except Exception as e:
            skipped_cases.append({'case': case_num, 'reason': f'Error: {str(e)}'})
            continue
    
    pbar.close()
    
    # Create DataFrame with all results
    df = pd.DataFrame(all_results)
    
    # Save summary CSV
    WEEK5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = WEEK5_OUTPUT_DIR / "surface_classification_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save skipped cases log
    if skipped_cases:
        skipped_df = pd.DataFrame(skipped_cases)
        skipped_path = WEEK5_OUTPUT_DIR / "skipped_cases_log.csv"
        skipped_df.to_csv(skipped_path, index=False)
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Total cases attempted: {total_cases}")
    print(f"   Cases processed successfully: {processed_cases}")
    print(f"   Cases skipped: {len(skipped_cases)}")
    print(f"   JSON diagnosis files created: {json_files_created}")
    print(f"   Total teeth with caries classified: {total_teeth_classified}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   CSV Results: {csv_path}")
    if skipped_cases:
        print(f"   Skipped Cases Log: {WEEK5_OUTPUT_DIR / 'skipped_cases_log.csv'}")
    
    # Print classification statistics
    if not df.empty:
        print(f"\n📈 CLASSIFICATION STATISTICS:")
        print(f"   Total caries lesions: {len(df)}")
        
        print(f"\n   Surface Distribution:")
        surface_counts = df['surface_name'].value_counts()
        for surface, count in surface_counts.items():
            percentage = count / len(df) * 100
            print(f"     • {surface}: {count} ({percentage:.1f}%)")
        
        print(f"\n   Position Detail Distribution:")
        if 'caries_position_detail' in df.columns:
            position_counts = df['caries_position_detail'].value_counts()
            for position, count in position_counts.items():
                percentage = count / len(df) * 100
                print(f"     • {position}: {count} ({percentage:.1f}%)")
        
        print(f"\n   Jaw Type Distribution:")
        jaw_counts = df['jaw_type'].value_counts()
        for jaw, count in jaw_counts.items():
            print(f"     • {jaw}: {count}")
    
    # Print skipped cases summary
    if skipped_cases:
        print(f"\n⚠️  SKIPPED CASES ({len(skipped_cases)}):")
        reason_counts = {}
        for item in skipped_cases:
            reason = item['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            print(f"     • {reason}: {count} cases")
    
    print(f"\n{'='*60}")
    
    return df


def process_sample_case(case_num: int = 1):
    """
    Process a single sample case with detailed output and visualization.
    
    Args:
        case_num: Case number to process (default: 1)
    """
    print(f"\n{'='*60}")
    print(f"Processing Sample Case {case_num}")
    print(f"{'='*60}\n")
    
    results = process_single_case(case_num, save_viz=True)
    
    if not results:
        print(f"No caries found in case {case_num} or data not available.")
        return
    
    print(f"Found {len(results)} teeth with caries:\n")
    
    for r in results:
        surface_name = SURFACE_NAME_MAP.get(r['classification'], 'Other')
        position_detail = r.get('caries_position_detail', 'Center')
        
        print(f"Tooth {r['tooth_id']} ({r['jaw_type']} jaw):")
        print(f"  Surface: {surface_name} (Class {r['classification']})")
        print(f"  Position Detail: {position_detail}")
        print(f"  Rotation: {r['rotation_angle_deg']:.2f}°")
        print(f"  Position: X={r['rel_x']:.3f}, Y={r['rel_y']:.3f}")
        print(f"  Caries: {r['caries_pixels']} pixels ({r['caries_percentage']:.2f}%)")
        print()
    
    output_dir = WEEK5_OUTPUT_DIR / f"case {case_num}"
    print(f"Visualizations saved to: {output_dir}")
    print(f"JSON diagnosis saved to: {output_dir / f'case_{case_num}_diagnosis.json'}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dental Caries Surface Classification using PCA+OBB"
    )
    parser.add_argument(
        '--mode', type=str, default='sample',
        choices=['sample', 'all', 'range'],
        help='Processing mode: sample (case 1), all (1-500), or range'
    )
    parser.add_argument(
        '--case', type=int, default=1,
        help='Case number for sample mode'
    )
    parser.add_argument(
        '--start', type=int, default=1,
        help='Start case for range mode'
    )
    parser.add_argument(
        '--end', type=int, default=500,
        help='End case for range mode'
    )
    parser.add_argument(
        '--viz', action='store_true',
        help='Save visualization images for all cases'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DENTAL CARIES SURFACE CLASSIFIER")
    print("PCA-based Orientation Normalization + Zone Classification")
    print("=" * 60)
    print(f"Output directory: {WEEK5_OUTPUT_DIR}")
    
    if args.mode == 'sample':
        process_sample_case(args.case)
    elif args.mode == 'all':
        process_all_cases(1, 500, save_viz=args.viz)
    elif args.mode == 'range':
        process_all_cases(args.start, args.end, save_viz=args.viz)


if __name__ == "__main__":
    main()
