"""
Week 4 Output Standardization Script
=====================================
Converts Week 4 inference output to match Week 2/3 schema and folder structure.

Schema Mapping:
---------------
Week 3 Schema (teeth_caries_data):
{
    "case_number": int,
    "teeth_caries_data": [
        {
            "case_number": int,
            "tooth_id": str,
            "confidence": float,
            "total_pixels": int,
            "caries_pixels": int,
            "caries_percentage": float,
            "has_caries": bool,
            "caries_coordinates": [[x,y], ...]
        }
    ]
}

Week 4 Schema (caries_detections + tooth_detections):
{
    "image_path": str,
    "caries_detections": [
        {
            "class_id": int,
            "class_name": str,  # "Occlusal", "Proximal", "Lingual"
            "confidence": float,
            "bbox": {"x1":, "y1":, "x2":, "y2":},
            "tooth_id": str,
            "tooth_name": str
        }
    ],
    "tooth_detections": [
        {
            "tooth_id": str,
            "confidence": float,
            "bbox": {"x1":, "y1":, "x2":, "y2":}
        }
    ]
}

Converted Output Schema (matching Week 3):
{
    "case_number": int,
    "source": "week4_inference",
    "teeth_caries_data": [...],      # Week 3 format
    "original_week4_data": {...}      # Preserved for reference
}
"""

import json
import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


# Surface type mapping from Week 4 class names to standardized format
SURFACE_TYPE_MAP = {
    "Occlusal": "occlusal",
    "Proximal": "proximal",  # Covers mesial/distal
    "Lingual": "lingual"
}


def extract_case_number(filename: str) -> Optional[int]:
    """Extract case number from filename like 'case_6_results.json' -> 6"""
    match = re.search(r'case_(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_json(filepath: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Error loading {filepath}: {e}")
        return {}


def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save JSON file with pretty formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def estimate_caries_pixels(bbox: Dict, confidence: float) -> int:
    """
    Estimate caries pixel count from bounding box.
    Since Week 4 uses detection (not segmentation), we estimate based on bbox area.
    """
    width = bbox['x2'] - bbox['x1']
    height = bbox['y2'] - bbox['y1']
    area = width * height
    # Assume caries fills roughly 60-80% of bbox (adjusted by confidence)
    fill_factor = 0.6 + (confidence * 0.2)
    return int(area * fill_factor)


def generate_bbox_coordinates(bbox: Dict) -> List[List[int]]:
    """
    Generate representative coordinates for caries region.
    Returns corner and center points of the bounding box.
    """
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Return key points: corners + center
    return [
        [x1, y1], [x2, y1],  # Top corners
        [x1, y2], [x2, y2],  # Bottom corners
        [cx, cy]              # Center
    ]


# Class ID to readable name mapping (ensures we never output "Class_0")
CARIES_CLASS_ID_TO_NAME = {
    0: "Occlusal",
    1: "Proximal",
    2: "Lingual"
}


def get_readable_class_name(class_id: int, class_name: str) -> str:
    """
    Ensure we always return a readable class name, never 'Class_X'.
    
    Priority:
    1. Use explicit mapping from class_id
    2. Use provided class_name if it's valid
    3. Default to 'Unknown' (never 'Class_X')
    """
    # First try the explicit mapping
    if class_id in CARIES_CLASS_ID_TO_NAME:
        return CARIES_CLASS_ID_TO_NAME[class_id]
    
    # Check if provided class_name is already readable (not 'Class_X' format)
    if class_name and not class_name.startswith('Class_'):
        return class_name
    
    # Last resort - return descriptive unknown
    return "Unknown"


def convert_week4_to_week3_schema(week4_data: Dict, case_number: int) -> Dict:
    """
    Convert Week 4 JSON format to Week 3 schema.
    
    Key transformations:
    1. Group caries detections by tooth_id
    2. Map class_name to surface type (NEVER 'Class_0')
    3. Calculate caries_percentage estimate
    4. Generate teeth_caries_data array for ALL teeth (with/without caries)
    5. Include pixel_coordinates (real tooth contours from Detectron2)
    """
    
    teeth_caries_data = []
    
    # Get all detected teeth
    tooth_detections = week4_data.get('tooth_detections', [])
    caries_detections = week4_data.get('caries_detections', [])
    
    # Group caries by tooth_id
    caries_by_tooth = defaultdict(list)
    for caries in caries_detections:
        tooth_id = caries.get('tooth_id', 'unknown')
        caries_by_tooth[tooth_id].append(caries)
    
    # Process each detected tooth
    for tooth in tooth_detections:
        tooth_id = tooth.get('tooth_id', 'unknown')
        tooth_confidence = tooth.get('confidence', 0.0)
        tooth_bbox = tooth.get('bbox', {})
        
        # Get pixel_coordinates (real segmentation contour from Detectron2, or bbox fallback)
        pixel_coordinates = tooth.get('pixel_coordinates', [])
        
        # Calculate tooth area (total pixels estimate)
        if tooth_bbox:
            tooth_width = tooth_bbox.get('x2', 0) - tooth_bbox.get('x1', 0)
            tooth_height = tooth_bbox.get('y2', 0) - tooth_bbox.get('y1', 0)
            total_pixels = int(tooth_width * tooth_height * 0.7)  # ~70% fill for tooth shape
        else:
            total_pixels = 0
        
        # Check if this tooth has caries
        tooth_caries = caries_by_tooth.get(tooth_id, [])
        has_caries = len(tooth_caries) > 0
        
        # Calculate caries metrics
        caries_pixels = 0
        caries_coordinates = []
        caries_surfaces = []
        caries_details = []
        
        for caries in tooth_caries:
            caries_bbox = caries.get('bbox', {})
            caries_conf = caries.get('confidence', 0.0)
            
            # Get readable class name (NEVER 'Class_0')
            class_id = caries.get('class_id', -1)
            raw_class_name = caries.get('class_name', '')
            readable_class_name = get_readable_class_name(class_id, raw_class_name)
            surface_type = SURFACE_TYPE_MAP.get(readable_class_name, 'unknown')
            
            # Accumulate caries pixels
            pixels = estimate_caries_pixels(caries_bbox, caries_conf)
            caries_pixels += pixels
            
            # Generate coordinates
            coords = generate_bbox_coordinates(caries_bbox)
            caries_coordinates.extend(coords)
            
            # Track surface types
            if surface_type not in caries_surfaces:
                caries_surfaces.append(surface_type)
            
            # Store detailed caries info with READABLE class name
            caries_details.append({
                "surface": surface_type,
                "class_name": readable_class_name,  # Always readable (Occlusal/Proximal/Lingual)
                "confidence": round(caries_conf, 6),
                "bbox": caries_bbox,
                "estimated_pixels": pixels
            })
        
        # Calculate caries percentage
        caries_percentage = round((caries_pixels / total_pixels * 100), 4) if total_pixels > 0 else 0.0
        
        # Build Week 3 compatible record with instance segmentation support
        tooth_record = {
            "case_number": case_number,
            "tooth_id": str(tooth_id),
            "confidence": round(tooth_confidence, 6),
            "total_pixels": total_pixels,
            "caries_pixels": caries_pixels,
            "caries_percentage": caries_percentage,
            "has_caries": has_caries,
            "caries_coordinates": caries_coordinates,
            # NEW: Real tooth segmentation contour from Detectron2
            "pixel_coordinates": pixel_coordinates,
            # Week 4 specific extensions (additional detail)
            "caries_surfaces": caries_surfaces if has_caries else [],
            "caries_details": caries_details if has_caries else [],
            "tooth_name": tooth.get('tooth_name', ''),
            "tooth_bbox": tooth_bbox
        }
        
        teeth_caries_data.append(tooth_record)
    
    # Sort by confidence (highest first) to match Week 3 ordering
    teeth_caries_data.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Count teeth with real segmentation (not just bbox)
    num_with_segmentation = sum(
        1 for t in teeth_caries_data 
        if t.get('pixel_coordinates') and len(t['pixel_coordinates']) > 4
    )
    
    # Build final output
    converted_data = {
        "case_number": case_number,
        "source": "week4_inference_v3",  # Updated version
        "model_info": {
            "caries_model": "yolov8s_3class",
            "tooth_model": "Tooth_seg_pano_20250319",
            "crop_seg_model": "Tooth_seg_crop_20250424",  # Detectron2 crop model
            "classes": ["Occlusal", "Proximal", "Lingual"],
            "segmentation": "detectron2_instance"
        },
        "summary": {
            "num_teeth_detected": len(tooth_detections),
            "num_teeth_with_caries": sum(1 for t in teeth_caries_data if t['has_caries']),
            "num_teeth_with_segmentation": num_with_segmentation,
            "total_caries_detections": len(caries_detections),
            "localization_method": week4_data.get('localization_method', 'unknown'),
            "processing_time_ms": week4_data.get('processing_time_ms', 0)
        },
        "teeth_caries_data": teeth_caries_data,
        # Preserve original Week 4 data for reference
        "original_week4_data": {
            "image_path": week4_data.get('image_path', ''),
            "image_size": week4_data.get('image_size', {}),
            "caries_detections": caries_detections
        }
    }
    
    return converted_data


def process_all_files(source_dir: str, output_dir: str) -> Dict:
    """
    Process all Week 4 files and convert to Week 3 schema.
    Organize into case folders matching Week 2 structure.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_cases": 0,
        "cases_with_caries": 0,
        "total_teeth": 0,
        "total_caries": 0,
        "files_processed": 0,
        "errors": []
    }
    
    # Find all JSON files
    json_files = list(source_path.glob("*_results.json"))
    image_files = {f.stem.replace('_results', ''): f for f in source_path.glob("*_annotated.jpg")}
    
    print(f"\nFound {len(json_files)} JSON files and {len(image_files)} image files")
    print("-" * 60)
    
    for json_file in sorted(json_files, key=lambda x: extract_case_number(x.name) or 0):
        case_number = extract_case_number(json_file.name)
        if case_number is None:
            stats["errors"].append(f"Could not extract case number from {json_file.name}")
            continue
        
        # Create case folder (matching Week 2 format: "case 6")
        case_folder = output_path / f"case {case_number}"
        case_folder.mkdir(exist_ok=True)
        
        # Load and convert JSON
        week4_data = load_json(str(json_file))
        if not week4_data:
            continue
        
        converted_data = convert_week4_to_week3_schema(week4_data, case_number)
        
        # Save converted JSON with Week 3 naming convention
        output_json = case_folder / f"case_{case_number}_caries_mapping.json"
        save_json(converted_data, str(output_json))
        
        # Copy annotated image
        image_key = f"case_{case_number}_annotated"
        if image_key in image_files:
            src_image = image_files[image_key]
            # Keep original naming for annotated image
            dst_image = case_folder / f"case_{case_number}_annotated.jpg"
            shutil.copy2(str(src_image), str(dst_image))
        
        # Update stats
        stats["total_cases"] += 1
        stats["files_processed"] += 2  # JSON + image
        stats["total_teeth"] += converted_data["summary"]["num_teeth_detected"]
        stats["total_caries"] += converted_data["summary"]["total_caries_detections"]
        if converted_data["summary"]["num_teeth_with_caries"] > 0:
            stats["cases_with_caries"] += 1
        
        print(f"  ✓ Case {case_number}: {converted_data['summary']['num_teeth_detected']} teeth, "
              f"{converted_data['summary']['total_caries_detections']} caries")
    
    return stats


def validate_conversion(week3_template_path: str, converted_path: str):
    """
    Validate that converted JSON matches Week 3 schema structure.
    Print comparison for visual verification.
    """
    print("\n" + "=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)
    
    # Load templates
    week3_data = load_json(week3_template_path)
    converted_data = load_json(converted_path)
    
    if not week3_data or not converted_data:
        print("  ⚠ Could not load files for validation")
        return
    
    # Compare top-level keys
    week3_keys = set(week3_data.keys())
    converted_keys = set(converted_data.keys())
    
    print("\n[TOP-LEVEL KEYS]")
    print(f"  Week 3 Template: {sorted(week3_keys)}")
    print(f"  Converted Week 4: {sorted(converted_keys)}")
    
    common_keys = week3_keys & converted_keys
    print(f"  Common Keys: {sorted(common_keys)}")
    
    # Compare teeth_caries_data structure
    if "teeth_caries_data" in week3_data and "teeth_caries_data" in converted_data:
        week3_tooth = week3_data["teeth_caries_data"][0] if week3_data["teeth_caries_data"] else {}
        converted_tooth = converted_data["teeth_caries_data"][0] if converted_data["teeth_caries_data"] else {}
        
        week3_tooth_keys = set(week3_tooth.keys())
        converted_tooth_keys = set(converted_tooth.keys())
        
        print("\n[TEETH_CARIES_DATA RECORD KEYS]")
        print(f"  Week 3 Template: {sorted(week3_tooth_keys)}")
        print(f"  Converted Week 4: {sorted(converted_tooth_keys)}")
        
        # Check required Week 3 fields
        required_fields = {'case_number', 'tooth_id', 'confidence', 'total_pixels', 
                          'caries_pixels', 'caries_percentage', 'has_caries', 'caries_coordinates'}
        
        missing = required_fields - converted_tooth_keys
        if missing:
            print(f"  ⚠ Missing required fields: {missing}")
        else:
            print(f"  ✓ All required fields present!")
        
        # Check for new v3.0 fields
        v3_fields = {'pixel_coordinates'}
        v3_present = v3_fields & converted_tooth_keys
        if v3_present:
            print(f"  ✓ Instance segmentation fields present: {v3_present}")
        
        # Show sample comparison
        print("\n[SAMPLE TOOTH RECORD COMPARISON]")
        print("\n  --- Week 3 Template (first tooth with caries) ---")
        for tooth in week3_data["teeth_caries_data"]:
            if tooth.get("has_caries"):
                sample_week3 = {k: v for k, v in tooth.items() if k != 'caries_coordinates'}
                sample_week3['caries_coordinates'] = f"[{len(tooth.get('caries_coordinates', []))} points]"
                print(f"  {json.dumps(sample_week3, indent=4)[:500]}...")
                break
        
        print("\n  --- Converted Week 4 (first tooth with caries) ---")
        for tooth in converted_data["teeth_caries_data"]:
            if tooth.get("has_caries"):
                sample_converted = {k: v for k, v in tooth.items() 
                                   if k not in ['caries_coordinates', 'caries_details', 'tooth_bbox', 'pixel_coordinates']}
                sample_converted['caries_coordinates'] = f"[{len(tooth.get('caries_coordinates', []))} points]"
                sample_converted['caries_details'] = f"[{len(tooth.get('caries_details', []))} entries]"
                # Show pixel_coordinates summary
                pixel_coords = tooth.get('pixel_coordinates', [])
                if pixel_coords and len(pixel_coords) > 4:
                    sample_converted['pixel_coordinates'] = f"[{len(pixel_coords)} contour points - Detectron2]"
                else:
                    sample_converted['pixel_coordinates'] = f"[{len(pixel_coords)} points - bbox fallback]"
                print(f"  {json.dumps(sample_converted, indent=4)}")
                break


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standardize Week 4 output to Week 3 schema')
    parser.add_argument('--source', '-s', type=str, 
                       default=r"C:\Users\jaopi\Desktop\SP\week4\inference_output_batch",
                       help='Source directory with Week 4 inference results')
    parser.add_argument('--output', '-o', type=str,
                       default=r"C:\Users\jaopi\Desktop\SP\week4\final_structured_output",
                       help='Output directory for standardized results')
    parser.add_argument('--template', '-t', type=str,
                       default=r"C:\Users\jaopi\Desktop\SP\week3\dental_analysis_output\case 6\case_6_caries_mapping.json",
                       help='Week 3 JSON template for validation')
    
    args = parser.parse_args()
    
    # Define paths
    source_dir = args.source
    output_dir = args.output
    week3_template = args.template
    
    print("=" * 60)
    print("WEEK 4 OUTPUT STANDARDIZATION")
    print("=" * 60)
    print(f"\nSource:      {source_dir}")
    print(f"Output:      {output_dir}")
    print(f"Template:    {week3_template}")
    
    # Step 1: Process and convert all files
    print("\n" + "=" * 60)
    print("STEP 1: CONVERTING FILES")
    print("=" * 60)
    
    stats = process_all_files(source_dir, output_dir)
    
    # Print summary
    print("\n" + "-" * 60)
    print("CONVERSION SUMMARY")
    print("-" * 60)
    print(f"  Total cases processed:    {stats['total_cases']}")
    print(f"  Cases with caries:        {stats['cases_with_caries']}")
    print(f"  Total teeth detected:     {stats['total_teeth']}")
    print(f"  Total caries detected:    {stats['total_caries']}")
    print(f"  Files processed:          {stats['files_processed']}")
    
    if stats["errors"]:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"    - {err}")
    
    # Step 2: Validate conversion
    # Find first converted case for validation
    output_path = Path(output_dir)
    converted_files = list(output_path.glob("*/case_*_caries_mapping.json"))
    if converted_files:
        validate_conversion(week3_template, str(converted_files[0]))
    
    # Step 3: Show folder structure comparison
    print("\n" + "=" * 60)
    print("FOLDER STRUCTURE COMPARISON")
    print("=" * 60)
    
    print("\n  Week 2/3 Structure:")
    print("    500-segmentation+recognition/")
    print("      case 6/")
    print("        case_6_bounding_boxes.png")
    print("        case_6_mask_overlay.png")
    print("        case_6_results.json")
    print("    dental_analysis_output/")
    print("      case 6/")
    print("        case_6_caries_mapping.json")
    
    print("\n  Week 4 Final Structure:")
    print("    final_structured_output/")
    sample_folders = sorted(output_path.iterdir())[:3]
    for folder in sample_folders:
        if folder.is_dir():
            print(f"      {folder.name}/")
            for f in sorted(folder.iterdir()):
                print(f"        {f.name}")
    print("      ...")
    
    print("\n" + "=" * 60)
    print("STANDARDIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
