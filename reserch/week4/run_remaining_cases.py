#!/usr/bin/env python3
"""
Run inference on remaining cases that haven't been processed yet.
Checks what's already done and only processes missing cases.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
MATERIAL_DIR = Path(r"C:\Users\jaopi\Desktop\SP\material\500 cases with annotation")
WEEK4_DIR = Path(r"C:\Users\jaopi\Desktop\SP\week4")
OUTPUT_DIR = WEEK4_DIR / "inference_full_500"
TEMP_IMAGES_DIR = OUTPUT_DIR / "_temp_images"

MODEL_PATH = WEEK4_DIR / "runs/caries_train/caries_3class/weights/best.pt"
TOOTH_MODEL = Path(r"C:\Users\jaopi\Desktop\SP\material\Tooth Segmentation + Recognition model\weights\Tooth_seg_pano_20250319.pt")

def get_processed_cases():
    """Get list of case numbers already processed (have JSON output)"""
    processed = set()
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("case_*_results.json"):
            # Extract case number from filename like "case_123_results.json"
            name = f.stem  # "case_123_results"
            parts = name.split('_')
            if len(parts) >= 2:
                try:
                    case_num = int(parts[1])
                    processed.add(case_num)
                except ValueError:
                    pass
    return processed

def get_all_cases():
    """Get all available case numbers from material directory"""
    all_cases = {}
    for case_dir in MATERIAL_DIR.iterdir():
        if case_dir.is_dir() and case_dir.name.startswith("case "):
            try:
                case_num = int(case_dir.name.replace("case ", ""))
                # Find image file
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    images = list(case_dir.glob(f"*{ext}"))
                    if images:
                        all_cases[case_num] = images[0]
                        break
            except ValueError:
                pass
    return all_cases

def main():
    print("=" * 60)
    print("REMAINING CASES INFERENCE")
    print("=" * 60)
    
    # Get processed and all cases
    processed = get_processed_cases()
    all_cases = get_all_cases()
    
    print(f"\nTotal cases available: {len(all_cases)}")
    print(f"Already processed: {len(processed)}")
    
    # Find missing cases
    missing = set(all_cases.keys()) - processed
    print(f"Remaining to process: {len(missing)}")
    
    if not missing:
        print("\n✓ All cases already processed!")
        return
    
    # Show first few missing
    missing_sorted = sorted(missing)
    print(f"\nMissing cases: {missing_sorted[:20]}...")
    
    # Create temp directory for remaining images
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_IMAGES_DIR.mkdir(exist_ok=True)
    
    # Copy only missing images to temp directory
    print(f"\nCopying {len(missing)} remaining images...")
    for case_num in missing_sorted:
        src_path = all_cases[case_num]
        # Name with case number prefix for proper sorting
        dst_name = f"case_{case_num}{src_path.suffix}"
        dst_path = TEMP_IMAGES_DIR / dst_name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  Copied case {case_num}")
    
    print(f"\n{'=' * 60}")
    print("Running inference on remaining cases...")
    print("=" * 60)
    
    # Build inference command
    cmd = [
        sys.executable,
        str(WEEK4_DIR / "inference.py"),
        "--image_dir", str(TEMP_IMAGES_DIR),
        "--model_path", str(MODEL_PATH),
        "--tooth_model", str(TOOTH_MODEL),
        "--output_dir", str(OUTPUT_DIR),
        "--caries_conf", "0.05"
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\n" + "-" * 60)
    
    # Run inference
    result = subprocess.run(cmd, cwd=str(WEEK4_DIR))
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ Remaining inference complete!")
        print("=" * 60)
        
        # Count results
        final_processed = get_processed_cases()
        print(f"\nTotal cases now processed: {len(final_processed)}")
    else:
        print(f"\n✗ Inference failed with return code {result.returncode}")

if __name__ == "__main__":
    main()
