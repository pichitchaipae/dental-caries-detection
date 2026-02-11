#!/usr/bin/env python3
"""
================================================================================
V3.0 ADVANCED BATCH PROCESSING - 500 CASES
================================================================================
Pipeline: Inference (Detectron2) -> Standardization -> Advanced Visualization

Output Structure:
    final_advanced_output/
    ├── case 1/
    │   ├── case_1.png                    (Original Image)
    │   ├── case_1_caries_mapping.json    (Standardized JSON)
    │   └── case_1_advanced_viz.jpg       (Color-Coded Segmentation)
    ...
    └── case 500/

Author: MLOps Pipeline
Version: 3.0 Advanced
"""

import os
import sys
import json
import shutil
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# =============================================================================
# CONFIGURATION - HIGH ACCURACY
# =============================================================================

# Base paths - Fixed to main folder structure
MAIN_DIR = Path(__file__).parent.resolve()  # week4/main folder
WEEK4_DIR = MAIN_DIR.parent  # week4 folder
SP_ROOT = WEEK4_DIR.parent  # SP folder

# Input source
INPUT_DIR = SP_ROOT / "material" / "500 cases with annotation"

# Output destination
OUTPUT_DIR = WEEK4_DIR / "final_advanced_output"
RAW_OUTPUT_DIR = OUTPUT_DIR / "_raw_inference"

# Model paths
CARIES_MODEL = WEEK4_DIR / "runs" / "caries_train" / "caries_3class" / "weights" / "best.pt"
TOOTH_MODEL = SP_ROOT / "material" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_pano_20250319.pt"
CROP_MODEL = SP_ROOT / "material" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_crop_20250424.pth"

# Scripts - Main scripts in main folder, standardize in week4
INFERENCE_SCRIPT = MAIN_DIR / "inference.py"
STANDARDIZE_SCRIPT = WEEK4_DIR / "standardize_week4.py"
VIZ_SCRIPT = MAIN_DIR / "viz_advanced.py"

# Python executable (use current environment)
PYTHON_EXE = Path(sys.executable)

# Pipeline parameters - HIGH ACCURACY
CARIES_CONF = 0.01  # 1% confidence threshold to maximize recall
TOOTH_CONF = 0.35   # Lower threshold for better tooth coverage

# Progress tracking
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_image_in_folder(folder: Path) -> Optional[Path]:
    """Find the main image file in a case folder."""
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    for ext in extensions:
        candidates = list(folder.glob(f"*{ext}"))
        # Filter out annotation/mask images
        for img in candidates:
            name_lower = img.stem.lower()
            if not any(x in name_lower for x in ['mask', 'annotation', 'label', 'seg', 'overlay']):
                return img
    
    # Fallback: return first image found
    for ext in extensions:
        candidates = list(folder.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    
    return None


def discover_cases(input_dir: Path) -> List[Tuple[int, Path, Path]]:
    """
    Discover all case folders and their images.
    
    Returns:
        List of (case_number, case_folder, image_path)
    """
    cases = []
    
    for folder in input_dir.iterdir():
        if not folder.is_dir():
            continue
        
        # Extract case number
        match = re.match(r'case\s*(\d+)', folder.name, re.IGNORECASE)
        if not match:
            continue
        
        case_num = int(match.group(1))
        image_path = find_image_in_folder(folder)
        
        if image_path:
            cases.append((case_num, folder, image_path))
    
    # Sort by case number
    cases.sort(key=lambda x: x[0])
    return cases


def run_command(cmd: List, description: str, timeout: int = 180, debug: bool = False) -> bool:
    """
    Run a subprocess command with timeout and error handling.
    """
    try:
        # Convert all command parts to strings
        cmd_str = [str(c) for c in cmd]
        
        if debug:
            print(f"\n  [DEBUG] Command: {' '.join(cmd_str[:4])}...")
        
        result = subprocess.run(
            cmd_str,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WEEK4_DIR),
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            if debug:
                print(f"  [DEBUG] STDERR: {result.stderr[:500] if result.stderr else 'None'}")
            return False
        return True
        
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step1_inference(image_path: Path, case_num: int, debug: bool = False) -> bool:
    """
    Step 1: Run V3.0 inference with Detectron2 segmentation.
    """
    # Verify image exists before running
    if not image_path.exists():
        if debug:
            print(f"  [ERROR] Image not found: {image_path}")
        return False
    
    cmd = [
        str(PYTHON_EXE), 
        str(INFERENCE_SCRIPT),
        "-i", str(image_path),
        "-m", str(CARIES_MODEL),
        "--tooth_model", str(TOOTH_MODEL),
        "--crop_model", str(CROP_MODEL),
        "-c", str(CARIES_CONF),
        "--tooth_conf", str(TOOTH_CONF),
        "-o", str(RAW_OUTPUT_DIR),
        "--no_visualize"
    ]
    
    return run_command(cmd, f"Inference case {case_num}", timeout=180, debug=debug)


def step2_standardize(case_num: int) -> bool:
    """
    Step 2: Convert raw inference output to standardized format.
    Performs inline conversion without calling external script.
    """
    # Add week4 to path for imports
    if str(WEEK4_DIR) not in sys.path:
        sys.path.insert(0, str(WEEK4_DIR))
    
    try:
        from standardize_week4 import convert_week4_to_week3_schema
    except ImportError:
        # Fallback to subprocess if import fails
        raw_json = RAW_OUTPUT_DIR / f"case_{case_num}_results.json"
        if not raw_json.exists():
            return False
        
        cmd = [
            PYTHON_EXE, str(STANDARDIZE_SCRIPT),
            "-s", str(RAW_OUTPUT_DIR),
            "-o", str(OUTPUT_DIR)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(WEEK4_DIR))
        case_json = OUTPUT_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
        return case_json.exists()
    
    # Find raw JSON
    raw_json = RAW_OUTPUT_DIR / f"case_{case_num}_results.json"
    if not raw_json.exists():
        return False
    
    # Create output folder
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
    Output: case_X_advanced_viz.jpg
    """
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    json_path = case_output_dir / f"case_{case_num}_caries_mapping.json"
    output_path = case_output_dir / f"case_{case_num}_advanced_viz.jpg"
    
    if not json_path.exists():
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
    Copy original image to output folder as case_X.png
    """
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = case_output_dir / f"case_{case_num}.png"
    
    try:
        shutil.copy2(image_path, output_path)
        return True
    except Exception:
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
    Process a single case through all 3 pipeline steps.
    
    Returns:
        True if all steps successful, False otherwise
    """
    # Check if already processed
    case_output_dir = OUTPUT_DIR / f"case {case_num}"
    case_json = case_output_dir / f"case_{case_num}_caries_mapping.json"
    case_viz = case_output_dir / f"case_{case_num}_advanced_viz.jpg"
    
    if skip_existing and case_json.exists() and case_viz.exists():
        stats['skipped'] += 1
        return True
    
    # Step 0: Copy original image
    if not copy_original_image(image_path, case_num):
        stats['copy_failed'] += 1
        return False
    
    # Step 1: Inference (Deep Scan V3.0)
    if not step1_inference(image_path, case_num, debug=False):
        stats['inference_failed'] += 1
        return False
    stats['inference_ok'] += 1
    
    # Step 2: Standardization
    if not step2_standardize(case_num):
        stats['standardize_failed'] += 1
        return False
    stats['standardize_ok'] += 1
    
    # Step 3: Advanced Visualization
    if not step3_visualize(image_path, case_num):
        stats['viz_failed'] += 1
        return False
    stats['viz_ok'] += 1
    
    return True


def run_batch_processing(skip_existing: bool = True):
    """
    Main batch processing function for all 500 cases.
    """
    print("\n" + "=" * 70)
    print("V3.0 ADVANCED BATCH PROCESSING - 500 CASES")
    print("Detectron2 Instance Segmentation + Color-Coded Visualization")
    print("=" * 70)
    
    start_time = time.time()
    
    # Verify configuration
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
            print(f"  [OK] {name}: {path.name}")
        else:
            print(f"  [X]  {name}: NOT FOUND - {path}")
            models_ok = False
    
    if not models_ok:
        print("\n[ERROR] Missing models. Aborting.")
        return
    
    # Check scripts
    for name, path in [
        ("Inference Script", INFERENCE_SCRIPT),
        ("Standardize Script", STANDARDIZE_SCRIPT),
        ("Visualization Script", VIZ_SCRIPT),
    ]:
        if path.exists():
            print(f"  [OK] {name}: {path.name}")
        else:
            print(f"  [X]  {name}: NOT FOUND - {path}")
            return
    
    # Discover cases
    print("\n[2] DISCOVERING CASES")
    print("-" * 50)
    
    if not INPUT_DIR.exists():
        print(f"  [X] Input directory not found: {INPUT_DIR}")
        return
    
    cases = discover_cases(INPUT_DIR)
    
    if not cases:
        print("  [X] No cases found!")
        return
    
    print(f"  [OK] Input source: {INPUT_DIR}")
    print(f"  -> Found {len(cases)} cases")
    print(f"  -> Case range: {cases[0][0]} to {cases[-1][0]}")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\n[3] PIPELINE CONFIGURATION")
    print("-" * 50)
    print(f"  Caries Confidence:  {CARIES_CONF} ({CARIES_CONF*100:.0f}%)")
    print(f"  Tooth Confidence:   {TOOTH_CONF}")
    print(f"  Skip Existing:      {skip_existing}")
    print(f"  Output Directory:   {OUTPUT_DIR}")
    
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
        
        try:
            success = process_single_case(case_num, case_folder, image_path, stats, skip_existing)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Batch processing stopped by user.")
            break
        except Exception as e:
            success = False
            stats['failed'] += 1
            failed_cases.append(case_num)
        
        stats['processed'] += 1
        if success:
            stats['successful'] += 1
        else:
            if case_num not in failed_cases:
                stats['failed'] += 1
                failed_cases.append(case_num)
        
        # Update progress bar
        if TQDM_AVAILABLE:
            iterator.set_postfix({
                'OK': stats['successful'],
                'Skip': stats['skipped'],
                'Fail': stats['failed']
            })
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    cases_processed = stats['total'] - stats['skipped']
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    
    print(f"\n[SUMMARY]")
    print(f"  Total Cases:       {stats['total']}")
    print(f"  Successful:        {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    print(f"  Skipped:           {stats['skipped']} (already processed)")
    print(f"  Failed:            {stats['failed']}")
    print(f"  Processing Time:   {elapsed_min:.1f} minutes")
    if cases_processed > 0:
        print(f"  Avg Time/Case:     {elapsed_time/cases_processed:.1f}s")
    
    print(f"\n[STEP BREAKDOWN]")
    print(f"  Inference:         {stats['inference_ok']} ok / {stats['inference_failed']} failed")
    print(f"  Standardization:   {stats['standardize_ok']} ok / {stats['standardize_failed']} failed")
    print(f"  Visualization:     {stats['viz_ok']} ok / {stats['viz_failed']} failed")
    
    if failed_cases:
        print(f"\n[FAILED CASES]")
        if len(failed_cases) <= 20:
            print(f"  {failed_cases}")
        else:
            print(f"  {failed_cases[:20]}... (+{len(failed_cases)-20} more)")
    
    print(f"\n[OUTPUT LOCATION]")
    print(f"  {OUTPUT_DIR}")
    
    # Save processing log
    log_path = OUTPUT_DIR / "batch_processing_log.json"
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'version': 'V3.0 Advanced',
        'config': {
            'caries_conf': CARIES_CONF,
            'tooth_conf': TOOTH_CONF,
            'input_source': str(INPUT_DIR),
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
    
    parser = argparse.ArgumentParser(
        description="V3.0 Advanced Batch Processing - 500 Cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
  final_advanced_output/
  +-- case N/
      +-- case_N.png                 (Original Image)
      +-- case_N_caries_mapping.json (Standardized JSON)
      +-- case_N_advanced_viz.jpg    (Color-Coded Visualization)
        """
    )
    parser.add_argument(
        "--no-skip", 
        action="store_true", 
        help="Reprocess all cases (don't skip existing)"
    )
    parser.add_argument(
        "--case", 
        type=int, 
        help="Process only a specific case number"
    )
    args = parser.parse_args()
    
    if args.case:
        # Process single case
        print(f"\n[SINGLE CASE MODE] Processing case {args.case}")
        cases = discover_cases(INPUT_DIR)
        case_item = next((c for c in cases if c[0] == args.case), None)
        
        if case_item:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            stats = {
                'total': 1, 'processed': 0, 'successful': 0, 'skipped': 0, 
                'failed': 0, 'inference_ok': 0, 'inference_failed': 0,
                'standardize_ok': 0, 'standardize_failed': 0,
                'viz_ok': 0, 'viz_failed': 0, 'copy_failed': 0
            }
            
            success = process_single_case(
                case_item[0], case_item[1], case_item[2], 
                stats, skip_existing=not args.no_skip
            )
            
            print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
            if success:
                print(f"Output: {OUTPUT_DIR / f'case {args.case}'}")
        else:
            print(f"[ERROR] Case {args.case} not found in {INPUT_DIR}")
    else:
        # Full batch processing
        run_batch_processing(skip_existing=not args.no_skip)
