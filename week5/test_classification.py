"""
Test Caries Surface Classification with Sample Case
=====================================================

This script tests the PCA-based surface classification algorithm
using actual data from a sample case (case 1 by default).

It demonstrates:
1. Loading tooth polygon data from week2
2. Loading caries coordinates from week3
3. Running the classification algorithm
4. Generating visualizations

Author: Computer Vision Engineer  
Date: 2026
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Add week5 to path
sys.path.insert(0, str(Path(__file__).parent))

from caries_surface_classifier import (
    classify_caries_surface,
    classify_caries_surface_detailed,
    get_surface_name,
    compute_centroid,
    perform_pca,
    create_rotation_matrix,
    rotate_points,
    get_bounding_box,
    is_upper_jaw
)

from visualization_utils import (
    visualize_pca_alignment,
    visualize_classification_zones,
    save_figure
)

# Paths
BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")
WEEK2_DIR = BASE_DIR / "week2" / "500-segmentation+recognition"
WEEK3_DIR = BASE_DIR / "week3" / "dental_analysis_output"
WEEK5_OUTPUT = BASE_DIR / "week5" / "test_output"


def load_week2_tooth_data(case_num: int) -> dict:
    """
    Load tooth segmentation data from week2 output.
    
    The week2 JSON contains tooth data in 'teeth_data' list.
    Each tooth has 'tooth_id' and 'pixel_coordinates' (list of [x,y] points).
    We compute the convex hull to get the boundary polygon.
    
    Args:
        case_num: Case number to load
        
    Returns:
        Dictionary mapping tooth_id to polygon coordinates
    """
    case_folder = WEEK2_DIR / f"case {case_num}"
    json_path = case_folder / f"case_{case_num}_results.json"
    
    if not json_path.exists():
        print(f"Week2 file not found: {json_path}")
        return {}
    
    try:
        print(f"Loading week2 data (this may take a moment for large files)...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Build lookup: tooth_id -> polygon (computed from pixel_coordinates)
        tooth_polygons = {}
        for tooth in data.get('teeth_data', []):
            tid = tooth.get('tooth_id')
            pixel_coords = tooth.get('pixel_coordinates', [])
            
            if tid and pixel_coords and len(pixel_coords) >= 3:
                # Compute convex hull to get boundary polygon
                coords = np.array(pixel_coords, dtype=np.float32)
                hull = cv2.convexHull(coords)
                polygon = hull.reshape(-1, 2).tolist()
                tooth_polygons[tid] = polygon
        
        print(f"Loaded {len(tooth_polygons)} tooth polygons from week2")
        return tooth_polygons
        
    except Exception as e:
        print(f"Error loading week2 data for case {case_num}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_week3_caries_data(case_num: int) -> dict:
    """Load caries mapping data from week3."""
    case_folder = WEEK3_DIR / f"case {case_num}"
    json_path = case_folder / f"case_{case_num}_caries_mapping.json"
    
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_tooth_polygon_from_pixel_coords(pixel_coordinates: list) -> list:
    """
    Create tooth boundary polygon from pixel coordinates using convex hull.
    
    When we have all pixel coordinates of a tooth (from week2),
    we compute the convex hull to get the boundary polygon.
    
    Args:
        pixel_coordinates: List of [x, y] pixel coordinates
        
    Returns:
        List of polygon vertices (convex hull)
    """
    if not pixel_coordinates or len(pixel_coordinates) < 3:
        return None
    
    coords = np.array(pixel_coordinates, dtype=np.float32)
    
    # Compute convex hull
    hull = cv2.convexHull(coords)
    polygon = hull.reshape(-1, 2).tolist()
    
    return polygon


def create_tooth_polygon_from_caries(caries_coords: list, padding: int = 50) -> list:
    """
    Create an approximate tooth polygon from caries coordinates.
    
    FALLBACK METHOD: When we don't have the actual tooth polygon, 
    we can estimate it by expanding around the caries region.
    This is NOT accurate for real classification - use actual tooth polygon!
    
    Args:
        caries_coords: List of [x, y] caries pixel coordinates
        padding: Expansion amount in pixels
        
    Returns:
        List of polygon vertices
    """
    if not caries_coords or len(caries_coords) < 3:
        return None
    
    coords = np.array(caries_coords, dtype=np.float32)
    
    # Get bounding box of caries
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # Expand to create approximate tooth region
    # Teeth are generally taller than wide
    width = x_max - x_min
    height = y_max - y_min
    
    # Create expanded bounding box (tooth-like shape)
    tooth_width = max(width * 2, 100)
    tooth_height = max(height * 4, 200)  # Teeth are elongated
    
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    # Create octagonal shape (simplified tooth)
    hw = tooth_width / 2
    hh = tooth_height / 2
    corner = min(hw, hh) * 0.3
    
    polygon = [
        [cx - hw + corner, cy - hh],
        [cx + hw - corner, cy - hh],
        [cx + hw, cy - hh + corner],
        [cx + hw, cy + hh - corner],
        [cx + hw - corner, cy + hh],
        [cx - hw + corner, cy + hh],
        [cx - hw, cy + hh - corner],
        [cx - hw, cy - hh + corner],
    ]
    
    return polygon
    
    # Create octagonal shape (simplified tooth)
    hw = tooth_width / 2
    hh = tooth_height / 2
    corner = min(hw, hh) * 0.3
    
    polygon = [
        [cx - hw + corner, cy - hh],
        [cx + hw - corner, cy - hh],
        [cx + hw, cy - hh + corner],
        [cx + hw, cy + hh - corner],
        [cx + hw - corner, cy + hh],
        [cx - hw + corner, cy + hh],
        [cx - hw, cy + hh - corner],
        [cx - hw, cy - hh + corner],
    ]
    
    return polygon


def test_with_synthetic_data():
    """
    Test the classifier with synthetic tooth and caries data.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Synthetic Data")
    print("=" * 60)
    
    # Create a synthetic tooth polygon (elongated shape like a molar)
    tooth_polygon = [
        [100, 30],   # Top (root area)
        [130, 25],
        [160, 40],
        [170, 80],
        [175, 140],
        [165, 200],  # Bottom (crown/occlusal area)
        [130, 210],
        [100, 205],
        [85, 160],
        [80, 100],
        [85, 50],
    ]
    
    test_cases = [
        {
            'name': 'Upper Molar - Occlusal (bottom)',
            'tooth_id': '16',
            'caries': [[120, 195], [130, 200], [125, 205], [140, 198]],
            'expected': 0  # Occlusal
        },
        {
            'name': 'Upper Molar - Proximal Left',
            'tooth_id': '16',
            'caries': [[85, 120], [90, 130], [88, 140]],
            'expected': 1  # Proximal
        },
        {
            'name': 'Upper Molar - Proximal Right',
            'tooth_id': '16',
            'caries': [[165, 120], [170, 130], [168, 140]],
            'expected': 1  # Proximal
        },
        {
            'name': 'Upper Molar - Lingual/Center',
            'tooth_id': '16',
            'caries': [[125, 120], [130, 125], [128, 130]],
            'expected': 2  # Lingual/Other
        },
        {
            'name': 'Lower Molar - Occlusal (top)',
            'tooth_id': '46',
            'caries': [[120, 35], [130, 40], [125, 32]],
            'expected': 0  # Occlusal
        },
        {
            'name': 'Lower Molar - Proximal',
            'tooth_id': '46',
            'caries': [[85, 100], [90, 110], [88, 105]],
            'expected': 1  # Proximal
        },
    ]
    
    # Create output directory
    output_dir = WEEK5_OUTPUT / "synthetic_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning tests...")
    results = []
    
    for i, test in enumerate(test_cases):
        result = classify_caries_surface_detailed(
            test['tooth_id'],
            tooth_polygon,
            test['caries']
        )
        
        classification = result['classification']
        passed = classification == test['expected']
        status = "✓ PASS" if passed else "✗ FAIL"
        
        results.append({
            'test': test['name'],
            'expected': test['expected'],
            'got': classification,
            'passed': passed
        })
        
        print(f"\n{status} - {test['name']}")
        print(f"   Expected: {test['expected']} ({get_surface_name(test['expected'])})")
        print(f"   Got:      {classification} ({result['surface_name']})")
        print(f"   Jaw: {result['jaw_type']}, Rot: {result['rotation_angle_deg']:.1f}°")
        print(f"   Rel Position: X={result['relative_position']['rel_x']:.3f}, "
              f"Y={result['relative_position']['rel_y']:.3f}")
        
        # Save visualization
        fig = visualize_classification_zones(
            tooth_polygon, test['caries'], test['tooth_id'], classification
        )
        save_figure(fig, output_dir / f"test_{i+1}_{test['name'].replace(' ', '_').replace('/', '-')}.png")
    
    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print("\n" + "-" * 60)
    print(f"SUMMARY: {passed_count}/{total_count} tests passed")
    print(f"Visualizations saved to: {output_dir}")
    
    return results


def test_with_actual_case(case_num: int = 1, use_week2_polygons: bool = True):
    """
    Test the classifier with actual case data from week2 and week3.
    
    Args:
        case_num: Case number to process
        use_week2_polygons: If True, load actual tooth polygons from week2.
                           If False, use approximate polygons from caries region.
    """
    print("\n" + "=" * 60)
    print(f"TEST 2: Actual Case Data (Case {case_num})")
    print("=" * 60)
    
    # Load week3 caries data
    caries_data = load_week3_caries_data(case_num)
    
    if caries_data is None:
        print("Could not load caries data!")
        return
    
    # Load week2 tooth polygons if requested
    tooth_polygons = {}
    if use_week2_polygons:
        tooth_polygons = load_week2_tooth_data(case_num)
    
    # Create output directory
    output_dir = WEEK5_OUTPUT / f"case_{case_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find teeth with caries
    teeth_with_caries = [
        tooth for tooth in caries_data.get('teeth_caries_data', [])
        if tooth.get('has_caries', False) and tooth.get('caries_coordinates')
    ]
    
    if not teeth_with_caries:
        print(f"No teeth with caries found in case {case_num}")
        return
    
    print(f"\nFound {len(teeth_with_caries)} teeth with caries")
    
    results = []
    
    for tooth in teeth_with_caries:
        tooth_id = tooth.get('tooth_id', 'unknown')
        caries_coords = tooth.get('caries_coordinates', [])
        caries_pixels = tooth.get('caries_pixels', 0)
        caries_pct = tooth.get('caries_percentage', 0)
        
        print(f"\n--- Tooth {tooth_id} ---")
        print(f"    Caries: {caries_pixels} pixels ({caries_pct:.2f}%)")
        
        if len(caries_coords) < 3:
            print(f"    Too few caries coordinates ({len(caries_coords)}), skipping")
            continue
        
        # Get tooth polygon - prefer week2 actual polygon
        tooth_polygon = None
        polygon_source = "none"
        
        if tooth_id in tooth_polygons:
            tooth_polygon = tooth_polygons[tooth_id]
            polygon_source = "week2"
            print(f"    Using actual tooth polygon from week2 ({len(tooth_polygon)} points)")
        else:
            # Fallback: create approximate polygon from caries
            tooth_polygon = create_tooth_polygon_from_caries(caries_coords)
            polygon_source = "approximated"
            print(f"    WARNING: Using approximated tooth polygon (no week2 data for {tooth_id})")
        
        if tooth_polygon is None:
            print("    Could not create tooth polygon, skipping")
            continue
        
        # Classify
        try:
            result = classify_caries_surface_detailed(
                tooth_id,
                tooth_polygon,
                caries_coords
            )
            
            classification = result['classification']
            surface_name = result['surface_name']
            jaw_type = result['jaw_type']
            rel_pos = result['relative_position']
            
            print(f"    Classification: {classification} - {surface_name}")
            print(f"    Jaw: {jaw_type}, Rotation: {result['rotation_angle_deg']:.1f}°")
            print(f"    Relative Position: X={rel_pos['rel_x']:.3f}, Y={rel_pos['rel_y']:.3f}")
            
            results.append({
                'tooth_id': tooth_id,
                'classification': classification,
                'surface_name': surface_name,
                'jaw_type': jaw_type,
                'caries_pixels': caries_pixels,
                'polygon_source': polygon_source
            })
            
            # Save visualization
            fig = visualize_classification_zones(
                tooth_polygon, caries_coords, tooth_id, classification
            )
            save_figure(fig, output_dir / f"tooth_{tooth_id}_classification.png")
            
            # Also save PCA alignment visualization
            fig2 = visualize_pca_alignment(
                tooth_polygon, caries_coords, tooth_id,
                title=f'Case {case_num} - Tooth {tooth_id} (polygon: {polygon_source})'
            )
            save_figure(fig2, output_dir / f"tooth_{tooth_id}_pca_alignment.png")
            
        except Exception as e:
            print(f"    Error during classification: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "-" * 60)
    print("CLASSIFICATION SUMMARY:")
    
    if results:
        from collections import Counter
        surface_counts = Counter(r['surface_name'] for r in results)
        for surface, count in surface_counts.items():
            print(f"  {surface}: {count}")
        
        # Show polygon source breakdown
        source_counts = Counter(r['polygon_source'] for r in results)
        print("\nPolygon sources:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    
    return results


def main():
    """Run all tests."""
    print("=" * 60)
    print("DENTAL CARIES SURFACE CLASSIFICATION - TEST SUITE")
    print("=" * 60)
    
    # Ensure output directory exists
    WEEK5_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Synthetic data
    synthetic_results = test_with_synthetic_data()
    
    # Test 2: Actual case data
    actual_results = test_with_actual_case(case_num=1)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
