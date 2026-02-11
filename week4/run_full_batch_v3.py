"""
Full Batch Processing Script - V3.0 Instance Segmentation Pipeline
===================================================================
Processes ALL 500 cases through the complete pipeline:
    Step 1: Inference (V3.0 with Detectron2)
    Step 2: Standardization (Legacy format with pixel_coordinates)
    Step 3: Visualization (Advanced overlay)

Author: Senior MLOps Engineer
Date: 2025-01-27
"""

import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Python executable - use the current environment's Python
PYTHON_EXE = sys.executable

# Base paths
BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")
WEEK4_DIR = BASE_DIR / "week4"

# Input sources (priority order)
INPUT_SOURCES = [
    BASE_DIR / "material" / "500 cases with annotation",
    BASE_DIR / "week2" / "500-segmentation+recognition",
]

# Output destination
OUTPUT_DIR = WEEK4_DIR / "test_v3_standardized"

# Model paths
CARIES_MODEL = WEEK4_DIR / "runs" / "caries_train" / "caries_3class" / "weights" / "best.pt"
TOOTH_MODEL = BASE_DIR / "material" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_pano_20250319.pt"
CROP_MODEL = BASE_DIR / "material" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_crop_20250424.pth"

# Scripts
INFERENCE_SCRIPT = WEEK4_DIR / "inference.py"
STANDARDIZE_SCRIPT = WEEK4_DIR / "standardize_week4.py"
VIZ_SCRIPT = WEEK4_DIR / "viz_advanced.py"

# Inference parameters
CARIES_CONF = 0.01  # 1% threshold to catch weak detections
TOOTH_CONF = 0.5

# Temporary output for raw inference
RAW_OUTPUT_DIR = WEEK4_DIR / "inference_output_v3_batch"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_input_source() -> Path:
    """Find the first available input source directory."""
    for source in INPUT_SOURCES:
        if source.exists():
            print(f"✓ Found input source: {source}")
            return source
    raise FileNotFoundError("No input source directory found!")


def discover_cases(input_dir: Path) -> List[Tuple[int, Path, Path]]:
    """
    Discover all case folders and their images.
    
    Returns:
        List of (case_number, case_folder, image_path) tuples
    """
    cases = []
    
    for folder in input_dir.iterdir():
        if not folder.is_dir():
            continue
        
        # Check if folder name matches "case X" pattern
        folder_name = folder.name.lower()
        if not folder_name.startswith("case "):
            continue
        
        try:
            case_num = int(folder_name.replace("case ", ""))
        except ValueError:
            continue
        
        # Find image file (case_X.png or case_X.jpg)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            potential_path = folder / f"case_{case_num}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        # Also check without underscore
        if image_path is None:
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                potential_path = folder / f"case{case_num}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
        
        if image_path:
            cases.append((case_num, folder, image_path))
    
    # Sort by case number
    cases.sort(key=lambda x: x[0])
    return cases


def run_command(cmd: List[str], description: str, timeout: int = 300) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command list
        description: Description for logging
        timeout: Timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WEEK4_DIR)
        )
        
        if result.returncode != 0:
            print(f"\n  ⚠ {description} failed:")
            print(f"    {result.stderr[:500] if result.stderr else 'No error message'}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n  ⚠ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"\n  ⚠ {description} error: {e}")
        return False


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step1_inference(image_path: Path, case_num: int) -> bool:
    """
    Step 1: Run V3.0 inference with Detectron2 fine segmentation.
    """
    cmd = [
        PYTHON_EXE, str(INFERENCE_SCRIPT),
        "-i", str(image_path),
        "-m", str(CARIES_MODEL),
        "-t", str(TOOTH_MODEL),
        "--crop_model", str(CROP_MODEL),
        "-c", str(CARIES_CONF),
        "--tooth_conf", str(TOOTH_CONF),
        "-o", str(RAW_OUTPUT_DIR),
        "--no_visualize"  # Skip built-in viz, we'll use viz_advanced.py
    ]
    
    return run_command(cmd, f"Inference case {case_num}", timeout=120)


def step2_standardize(case_num: int) -> bool:
    """
    Step 2: Convert raw inference output to standardized format.
    
    Directly converts the JSON without calling the full standardize script.
    """
    import sys
    sys.path.insert(0, str(WEEK4_DIR))
    
    # Import standardization function
    try:
        from standardize_week4 import convert_week4_to_week3_schema, save_json
    except ImportError:
        # Fallback: run the script
        raw_json = RAW_OUTPUT_DIR / f"case_{case_num}_results.json"
        if not raw_json.exists():
            return False
        
        case_output_dir = OUTPUT_DIR / f"case {case_num}"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            PYTHON_EXE, str(STANDARDIZE_SCRIPT),
            "-s", str(RAW_OUTPUT_DIR),
            "-o", str(OUTPUT_DIR)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(WEEK4_DIR))
        case_json = case_output_dir / f"case_{case_num}_caries_mapping.json"
        return case_json.exists()
    
    # Find the raw JSON file
    raw_json = RAW_OUTPUT_DIR / f"case_{case_num}_results.json"
    
    if not raw_json.exists():
        return False
    
    # Create case output folder
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load raw data
        with open(raw_json, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert to standardized format
        converted_data = convert_week4_to_week3_schema(raw_data, case_num)
        
        # Save converted JSON
        case_json = case_output_dir / f"case_{case_num}_caries_mapping.json"
        with open(case_json, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        return False


def step3_visualize(image_path: Path, case_num: int) -> bool:
    """
    Step 3: Generate advanced visualization overlay.
    """
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    json_path = case_output_dir / f"case_{case_num}_caries_mapping.json"
    output_path = case_output_dir / f"case_{case_num}_overlay.png"
    
    if not json_path.exists():
        print(f"\n  ⚠ Standardized JSON not found: {json_path}")
        return False
    
    cmd = [
        PYTHON_EXE, str(VIZ_SCRIPT),
        "-j", str(json_path),
        "-i", str(image_path),
        "-o", str(output_path),
        "--no_show",
        "--dpi", "150"
    ]
    
    return run_command(cmd, f"Visualization case {case_num}", timeout=60)


def copy_original_image(image_path: Path, case_num: int) -> bool:
    """
    Copy original image to output folder.
    """
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    ext = image_path.suffix
    output_path = case_output_dir / f"case_{case_num}{ext}"
    
    try:
        shutil.copy2(image_path, output_path)
        return True
    except Exception as e:
        print(f"\n  ⚠ Copy error: {e}")
        return False


# =============================================================================
# MAIN BATCH PROCESSOR
# =============================================================================

def process_single_case(
    case_num: int,
    case_folder: Path,
    image_path: Path,
    stats: Dict,
    skip_existing: bool = True
) -> bool:
    """
    Process a single case through all pipeline steps.
    
    Returns:
        True if all steps successful, False otherwise
    """
    # Check if already processed (skip existing)
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    case_json = case_output_dir / f"case_{case_num}_caries_mapping.json"
    case_overlay = case_output_dir / f"case_{case_num}_overlay.png"
    
    if skip_existing and case_json.exists() and case_overlay.exists():
        stats['skipped'] += 1
        return True
    
    # Step 0: Copy original image
    if not copy_original_image(image_path, case_num):
        stats['copy_failed'] += 1
        return False
    
    # Step 1: Inference
    if not step1_inference(image_path, case_num):
        stats['inference_failed'] += 1
        return False
    stats['inference_ok'] += 1
    
    # Step 2: Standardization
    if not step2_standardize(case_num):
        stats['standardize_failed'] += 1
        return False
    stats['standardize_ok'] += 1
    
    # Step 3: Visualization
    if not step3_visualize(image_path, case_num):
        stats['viz_failed'] += 1
        return False
    stats['viz_ok'] += 1
    
    return True


def run_batch_processing():
    """
    Main batch processing function.
    """
    print("\n" + "=" * 70)
    print("V3.0 FULL BATCH PROCESSING - 500 CASES")
    print("Instance Segmentation + Readable Labels Pipeline")
    print("=" * 70)
    
    start_time = time.time()
    
    # Verify paths
    print("\n[1] VERIFYING CONFIGURATION")
    print("-" * 50)
    
    # Check models
    models_ok = True
    for name, path in [
        ("Caries Model", CARIES_MODEL),
        ("Tooth Model", TOOTH_MODEL),
        ("Crop Model (Detectron2)", CROP_MODEL),
    ]:
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ✗ {name}: NOT FOUND - {path}")
            models_ok = False
    
    if not models_ok:
        print("\n❌ Missing models. Aborting.")
        return
    
    # Check scripts
    for name, path in [
        ("Inference Script", INFERENCE_SCRIPT),
        ("Standardize Script", STANDARDIZE_SCRIPT),
        ("Visualization Script", VIZ_SCRIPT),
    ]:
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ✗ {name}: NOT FOUND - {path}")
            return
    
    # Find input source
    print("\n[2] DISCOVERING CASES")
    print("-" * 50)
    
    input_dir = find_input_source()
    cases = discover_cases(input_dir)
    
    print(f"  → Found {len(cases)} cases")
    print(f"  → Case range: {cases[0][0]} to {cases[-1][0]}")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\n[3] PIPELINE CONFIGURATION")
    print("-" * 50)
    print(f"  Caries Confidence: {CARIES_CONF} ({CARIES_CONF*100:.0f}%)")
    print(f"  Tooth Confidence:  {TOOTH_CONF}")
    print(f"  Output Directory:  {OUTPUT_DIR}")
    
    # Initialize stats
    stats = {
        'total': len(cases),
        'processed': 0,
        'successful': 0,
        'skipped': 0,
        'failed': 0,
        'inference_ok': 0,
        'inference_failed': 0,
        'standardize_ok': 0,
        'standardize_failed': 0,
        'viz_ok': 0,
        'viz_failed': 0,
        'copy_failed': 0,
    }
    
    # Process cases
    print("\n[4] PROCESSING CASES")
    print("-" * 50)
    
    failed_cases = []
    
    if TQDM_AVAILABLE:
        iterator = tqdm(cases, desc="Processing", unit="case", ncols=80)
    else:
        iterator = cases
    
    for case_num, case_folder, image_path in iterator:
        if not TQDM_AVAILABLE and stats['processed'] % 50 == 0:
            print(f"  Processing case {case_num}... ({stats['processed']}/{stats['total']})")
        
        success = process_single_case(case_num, case_folder, image_path, stats, skip_existing=True)
        
        stats['processed'] += 1
        if success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
            failed_cases.append(case_num)
        
        # Update progress bar description
        if TQDM_AVAILABLE:
            iterator.set_postfix({
                'OK': stats['successful'],
                'Skip': stats['skipped'],
                'Fail': stats['failed']
            })
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    
    print(f"\n[SUMMARY]")
    print(f"  Total Cases:      {stats['total']}")
    print(f"  Successful:       {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    print(f"  Skipped:          {stats['skipped']} (already processed)")
    print(f"  Failed:           {stats['failed']}")
    print(f"  Processing Time:  {elapsed_min:.1f} minutes ({elapsed_time/max(1, stats['total']-stats['skipped']):.1f}s/case)")
    
    print(f"\n[STEP BREAKDOWN]")
    print(f"  Inference:        {stats['inference_ok']} ok / {stats['inference_failed']} failed")
    print(f"  Standardization:  {stats['standardize_ok']} ok / {stats['standardize_failed']} failed")
    print(f"  Visualization:    {stats['viz_ok']} ok / {stats['viz_failed']} failed")
    
    if failed_cases:
        print(f"\n[FAILED CASES]")
        print(f"  {failed_cases[:20]}{'...' if len(failed_cases) > 20 else ''}")
    
    print(f"\n[OUTPUT]")
    print(f"  {OUTPUT_DIR}")
    
    # Save processing log
    log_path = OUTPUT_DIR / "batch_processing_log.json"
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'caries_conf': CARIES_CONF,
            'tooth_conf': TOOTH_CONF,
            'input_source': str(input_dir),
            'output_dir': str(OUTPUT_DIR),
        },
        'stats': stats,
        'failed_cases': failed_cases,
        'elapsed_seconds': elapsed_time,
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"  Log saved: {log_path.name}")
    print("\n" + "=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V3.0 Full Batch Processing - 500 Cases")
    parser.add_argument("--no-skip", action="store_true", help="Reprocess all cases (don't skip existing)")
    parser.add_argument("--case", type=int, help="Process only specific case number")
    args = parser.parse_args()
    
    if args.case:
        # Process single case
        print(f"Processing single case: {args.case}")
        input_dir = find_input_source()
        cases = discover_cases(input_dir)
        case_item = next((c for c in cases if c[0] == args.case), None)
        if case_item:
            stats = {'total': 1, 'processed': 0, 'successful': 0, 'skipped': 0, 'failed': 0,
                     'inference_ok': 0, 'inference_failed': 0, 'standardize_ok': 0,
                     'standardize_failed': 0, 'viz_ok': 0, 'viz_failed': 0, 'copy_failed': 0}
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            success = process_single_case(case_item[0], case_item[1], case_item[2], stats, skip_existing=not args.no_skip)
            print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        else:
            print(f"Case {args.case} not found")
    else:
        run_batch_processing()
