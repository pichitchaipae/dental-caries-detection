"""
Dental Caries Analysis Pipeline
================================
Combined script that processes 500 dental cases to:
1. Map tooth positions to caries regions (JSON output)
2. Generate visual alignment debug images (PNG output)

Data Sources:
- JSON: week2/500-segmentation+recognition/case X/case_X_results.json
  Contains pixel_coordinates (list of [x, y]) for each tooth
- ROI Image: material/500-roi/case_X.png
  Binary mask where non-zero = caries, 0 = normal

Output:
- week3/caries_mapping_output/case X/case_X_caries_mapping.json
- week3/alignment_debug_output/case X/case_X_alignment_detailed.png
- week3/caries_mapping_output/caries_mapping_results.csv (summary)
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


# =============================================================================
# ROI & JSON Loading Functions
# =============================================================================

def load_roi_image(roi_path):
    """
    Load the ROI binary mask image.
    
    Args:
        roi_path: Path to the ROI image file
        
    Returns:
        numpy array: Grayscale image where non-zero = caries, 0 = normal
    """
    roi_img = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi_img is None:
        raise FileNotFoundError(f"Could not load ROI image: {roi_path}")
    return roi_img


def load_tooth_json(json_path):
    """
    Load tooth segmentation JSON data.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        dict: Parsed JSON data containing teeth_data with pixel_coordinates
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# =============================================================================
# Caries Mapping Functions
# =============================================================================

def calculate_caries_overlap(roi_img, pixel_coordinates):
    """
    Calculate the overlap between tooth pixels and caries regions.
    
    Args:
        roi_img: Binary ROI image (non-zero = caries, 0 = normal)
        pixel_coordinates: List of [x, y] coordinates for the tooth
        
    Returns:
        tuple: (caries_pixel_count, total_pixels, percentage, caries_coordinates)
    """
    if not pixel_coordinates:
        return 0, 0, 0.0, []
    
    height, width = roi_img.shape
    caries_count = 0
    caries_coords = []
    valid_pixels = 0
    
    for coord in pixel_coordinates:
        x, y = coord[0], coord[1]
        
        # Check bounds
        if 0 <= x < width and 0 <= y < height:
            valid_pixels += 1
            # Check if this pixel is in a caries region (any non-zero value)
            if roi_img[y, x] > 0:
                caries_count += 1
                caries_coords.append([x, y])
    
    # Calculate percentage
    percentage = (caries_count / valid_pixels * 100) if valid_pixels > 0 else 0.0
    
    return caries_count, valid_pixels, percentage, caries_coords


def generate_caries_mapping(roi_img, tooth_data, case_num):
    """
    Generate caries mapping results for all teeth in a case.
    
    Args:
        roi_img: Binary ROI image
        tooth_data: Dictionary with teeth_data
        case_num: Case number
        
    Returns:
        list: Caries mapping results for each tooth
    """
    results = []
    
    for tooth in tooth_data.get('teeth_data', []):
        tooth_id = tooth.get('tooth_id', 'unknown')
        pixel_coords = tooth.get('pixel_coordinates', [])
        confidence = tooth.get('confidence', 0.0)
        
        # Calculate caries overlap
        caries_count, total_pixels, percentage, caries_coords = calculate_caries_overlap(
            roi_img, pixel_coords
        )
        
        results.append({
            'case_number': case_num,
            'tooth_id': tooth_id,
            'confidence': confidence,
            'total_pixels': total_pixels,
            'caries_pixels': caries_count,
            'caries_percentage': round(percentage, 4),
            'has_caries': caries_count > 0,
            'caries_coordinates': caries_coords
        })
    
    return results


# =============================================================================
# Visual Alignment Functions
# =============================================================================

def create_alignment_visualization(roi_img, tooth_data, alpha=0.5):
    """
    Create a visualization showing tooth coordinates overlaid on ROI image.
    
    Args:
        roi_img: Binary ROI image (grayscale, non-zero = caries)
        tooth_data: Dictionary containing teeth_data with pixel_coordinates
        alpha: Transparency level for blending (0.0 to 1.0)
        
    Returns:
        tuple: (visualization image, stats dict)
    """
    height, width = roi_img.shape
    
    # Normalize ROI to 0-255 range (handles both 1 and 255 as caries values)
    roi_normalized = np.where(roi_img > 0, 255, 0).astype(np.uint8)
    
    # Create base visualization (ROI in grayscale converted to BGR)
    roi_bgr = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)
    
    # Create tooth overlay mask (all teeth in green)
    tooth_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Colors for different teeth
    colors = [
        (0, 255, 0),    # Green
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (0, 165, 255),  # Orange
        (255, 0, 255),  # Magenta
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Light Blue
        (0, 128, 255),  # Dark Orange
    ]
    
    tooth_count = 0
    total_pixels_drawn = 0
    out_of_bounds = 0
    
    for tooth in tooth_data.get('teeth_data', []):
        pixel_coords = tooth.get('pixel_coordinates', [])
        color = colors[tooth_count % len(colors)]
        tooth_count += 1
        
        for coord in pixel_coords:
            x, y = coord[0], coord[1]
            if 0 <= x < width and 0 <= y < height:
                tooth_overlay[y, x] = color
                total_pixels_drawn += 1
            else:
                out_of_bounds += 1
    
    # Alpha blending
    tooth_mask = np.any(tooth_overlay > 0, axis=2)
    result = roi_bgr.copy()
    blended = cv2.addWeighted(roi_bgr, 1 - alpha, tooth_overlay, alpha, 0)
    result[tooth_mask] = blended[tooth_mask]
    strong_overlay = cv2.addWeighted(result, 0.7, tooth_overlay, 0.3, 0)
    result[tooth_mask] = strong_overlay[tooth_mask]
    
    return result, {
        'teeth_count': tooth_count,
        'pixels_drawn': total_pixels_drawn,
        'out_of_bounds': out_of_bounds,
        'roi_shape': (height, width)
    }


def create_detailed_alignment_image(roi_img, tooth_data):
    """
    Create a detailed 3-panel visualization:
    - Panel 1: Original ROI (Red = caries)
    - Panel 2: Tooth Overlay (Green = teeth)
    - Panel 3: Blended alignment check
    
    Args:
        roi_img: Binary ROI image
        tooth_data: Dictionary with teeth data
        
    Returns:
        tuple: (combined image, stats, overlap_stats)
    """
    height, width = roi_img.shape
    
    # Normalize ROI
    roi_normalized = np.where(roi_img > 0, 255, 0).astype(np.uint8)
    
    # Panel 1: ROI with caries in red
    roi_colored = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)
    caries_mask = roi_img > 0
    roi_colored[caries_mask] = [0, 0, 255]  # Red for caries
    
    # Panel 2: Tooth overlay only
    tooth_only = np.zeros((height, width, 3), dtype=np.uint8)
    for tooth in tooth_data.get('teeth_data', []):
        pixel_coords = tooth.get('pixel_coordinates', [])
        for coord in pixel_coords:
            x, y = coord[0], coord[1]
            if 0 <= x < width and 0 <= y < height:
                tooth_only[y, x] = [0, 255, 0]  # Green
    
    # Panel 3: Blended
    blended, stats = create_alignment_visualization(roi_img, tooth_data, alpha=0.6)
    
    # Calculate overlap statistics
    tooth_mask = np.any(tooth_only > 0, axis=2)
    overlap_pixels = np.sum(tooth_mask & caries_mask)
    caries_pixels = np.sum(caries_mask)
    tooth_pixels = np.sum(tooth_mask)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    cv2.putText(roi_colored, "ROI (Red=Caries)", (20, 60), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(tooth_only, "Teeth (Green)", (20, 60), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(blended, "ALIGNMENT CHECK", (20, 60), font, font_scale, (0, 255, 255), thickness)
    
    # Add statistics to blended panel
    stats_text = [
        f"Teeth: {stats['teeth_count']}",
        f"Tooth pixels: {tooth_pixels:,}",
        f"Caries pixels: {caries_pixels:,}",
        f"Overlap: {overlap_pixels:,}",
    ]
    
    y_offset = 120
    for text in stats_text:
        cv2.putText(blended, text, (20, y_offset), font, 1.0, (255, 255, 0), 2)
        y_offset += 40
    
    # Combine panels (scaled down)
    scale = 0.5
    roi_small = cv2.resize(roi_colored, None, fx=scale, fy=scale)
    tooth_small = cv2.resize(tooth_only, None, fx=scale, fy=scale)
    blended_small = cv2.resize(blended, None, fx=scale, fy=scale)
    
    combined = np.hstack([roi_small, tooth_small, blended_small])
    
    return combined, stats, {
        'overlap_pixels': int(overlap_pixels),
        'caries_pixels': int(caries_pixels),
        'tooth_pixels': int(tooth_pixels)
    }


# =============================================================================
# Main Processing Function
# =============================================================================

def process_single_case(case_num, json_dir, roi_dir, output_dir):
    """
    Process a single case - generate both caries mapping JSON and alignment image
    in the same case folder.
    
    Args:
        case_num: Case number
        json_dir: Directory with tooth JSON files
        roi_dir: Directory with ROI images
        output_dir: Output directory (will contain case 1, case 2, ... case 500)
        
    Returns:
        tuple: (success, message, caries_results)
    """
    # Build file paths
    json_path = Path(json_dir) / f"case {case_num}" / f"case_{case_num}_results.json"
    roi_path = Path(roi_dir) / f"case_{case_num}.png"
    
    # Check files exist
    if not json_path.exists():
        return False, "JSON not found", None
    if not roi_path.exists():
        return False, "ROI not found", None
    
    # Load data
    roi_img = load_roi_image(roi_path)
    tooth_data = load_tooth_json(json_path)
    
    # Create single case output folder (contains both JSON and PNG)
    case_output_dir = Path(output_dir) / f"case {case_num}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Generate Caries Mapping JSON ---
    caries_results = generate_caries_mapping(roi_img, tooth_data, case_num)
    
    caries_json_path = case_output_dir / f"case_{case_num}_caries_mapping.json"
    with open(caries_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'case_number': case_num,
            'teeth_caries_data': caries_results
        }, f, indent=2)
    
    # --- Generate Alignment Debug Image ---
    combined_img, stats, overlap_stats = create_detailed_alignment_image(roi_img, tooth_data)
    alignment_img_path = case_output_dir / f"case_{case_num}_alignment_detailed.png"
    cv2.imwrite(str(alignment_img_path), combined_img)
    
    # Calculate summary stats
    teeth_with_caries = sum(1 for r in caries_results if r['has_caries'])
    
    return True, f"{len(caries_results)} teeth, {teeth_with_caries} with caries", caries_results


def process_all_cases(json_dir, roi_dir, output_dir, num_cases=500):
    """
    Process all 500 cases.
    
    Args:
        json_dir: Directory with tooth JSON files
        roi_dir: Directory with ROI images
        output_dir: Single output directory (contains case 1, case 2, ..., case 500)
        num_cases: Number of cases to process
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_caries_results = []
    summary_stats = {
        'total_cases': 0,
        'processed_cases': 0,
        'failed_cases': 0,
        'total_teeth': 0,
        'teeth_with_caries': 0
    }
    
    print("=" * 70)
    print("Dental Caries Analysis Pipeline")
    print("=" * 70)
    print(f"JSON Source: {json_dir}")
    print(f"ROI Source: {roi_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print("\nOutput structure:")
    print("  dental_analysis_output/")
    print("    case 1/")
    print("      case_1_caries_mapping.json")
    print("      case_1_alignment_detailed.png")
    print("    case 2/")
    print("      ...")
    print("    case 500/")
    print("=" * 70)
    
    pbar = tqdm(range(1, num_cases + 1), desc="Processing cases", unit="case")
    
    for case_num in pbar:
        summary_stats['total_cases'] += 1
        
        try:
            success, msg, caries_results = process_single_case(
                case_num, json_dir, roi_dir, output_dir
            )
            
            if success and caries_results:
                summary_stats['processed_cases'] += 1
                summary_stats['total_teeth'] += len(caries_results)
                summary_stats['teeth_with_caries'] += sum(1 for r in caries_results if r['has_caries'])
                
                # Add to all results (without caries_coordinates for CSV)
                for r in caries_results:
                    csv_result = {k: v for k, v in r.items() if k != 'caries_coordinates'}
                    all_caries_results.append(csv_result)
                
                pbar.set_postfix_str(f"case {case_num}: {msg}")
            else:
                summary_stats['failed_cases'] += 1
                pbar.set_postfix_str(f"case {case_num}: FAIL - {msg}")
                
        except Exception as e:
            summary_stats['failed_cases'] += 1
            pbar.set_postfix_str(f"case {case_num}: ERROR - {str(e)[:30]}")
    
    # Save summary CSV
    if all_caries_results:
        df = pd.DataFrame(all_caries_results)
        csv_path = Path(output_dir) / "caries_mapping_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults CSV saved to: {csv_path}")
        
        # Generate summary by case
        summary_df = df.groupby('case_number').agg({
            'tooth_id': 'count',
            'caries_pixels': 'sum',
            'has_caries': 'sum',
            'caries_percentage': 'mean'
        }).rename(columns={
            'tooth_id': 'total_teeth',
            'has_caries': 'teeth_with_caries',
            'caries_percentage': 'avg_caries_percentage'
        })
        
        summary_csv_path = Path(output_dir) / "caries_summary_by_case.csv"
        summary_df.to_csv(summary_csv_path)
        print(f"Summary CSV saved to: {summary_csv_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Total cases: {summary_stats['total_cases']}")
    print(f"Processed: {summary_stats['processed_cases']}")
    print(f"Failed: {summary_stats['failed_cases']}")
    print(f"Total teeth analyzed: {summary_stats['total_teeth']}")
    print(f"Teeth with caries: {summary_stats['teeth_with_caries']}")
    if summary_stats['total_teeth'] > 0:
        pct = summary_stats['teeth_with_caries'] / summary_stats['total_teeth'] * 100
        print(f"Caries prevalence: {pct:.2f}%")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 70)
    
    return all_caries_results, summary_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dental Caries Analysis Pipeline - Generate caries mapping and alignment visuals"
    )
    parser.add_argument(
        "--cases", type=int, default=500,
        help="Number of cases to process (default: 500)"
    )
    parser.add_argument(
        "--sample", type=int, nargs='+',
        help="Process only specific case numbers (e.g., --sample 1 5 10)"
    )
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path(__file__).parent.parent  # SP folder
    
    json_dir = base_dir / "week2" / "500-segmentation+recognition"
    roi_dir = base_dir / "material" / "500-roi"
    output_dir = base_dir / "week3" / "dental_analysis_output"
    
    if args.sample:
        # Process specific cases
        print(f"Processing specific cases: {args.sample}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for case_num in args.sample:
            success, msg, _ = process_single_case(
                case_num, json_dir, roi_dir, output_dir
            )
            print(f"Case {case_num}: {msg}")
    else:
        # Process all cases
        process_all_cases(
            json_dir, roi_dir, output_dir,
            num_cases=args.cases
        )


if __name__ == "__main__":
    main()
