"""
Full 500 Cases Inference Script - High Accuracy
=================================================
Runs inference on all 500 cases from the material folder.

This script:
1. Collects all PNG images from the 500 case folders
2. Runs the Week 4 caries detection model on each image (HIGH ACCURACY mode)
3. Saves results to inference_full_500 directory
4. Optionally runs standardization afterwards
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List

# Path configuration
MAIN_DIR = Path(__file__).parent  # week4/main folder
WEEK4_DIR = MAIN_DIR.parent  # week4 folder
BASE_DIR = WEEK4_DIR.parent  # SP folder


def collect_all_images(source_dir: str) -> List[str]:
    """
    Collect all PNG images from the 500 cases folder structure.
    Structure: case 1/case_1.png, case 2/case_2.png, etc.
    """
    source_path = Path(source_dir)
    images = []
    
    for case_folder in sorted(source_path.iterdir()):
        if case_folder.is_dir() and case_folder.name.lower().startswith('case'):
            # Find PNG images in the case folder
            for img_file in case_folder.glob("*.png"):
                # Skip any processed images (overlay, bounding_boxes, etc.)
                if 'overlay' not in img_file.name.lower() and 'bounding' not in img_file.name.lower():
                    images.append(str(img_file))
                    break  # Take first matching image per case
    
    return images


def create_temp_image_list(images: List[str], output_dir: str) -> str:
    """Create a temporary directory with symlinks/copies to all images for batch processing."""
    temp_dir = Path(output_dir) / "_temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in images:
        img_file = Path(img_path)
        # Copy image to temp directory with unique name
        dest = temp_dir / img_file.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
    
    return str(temp_dir)


def run_inference_batch(
    image_dir: str,
    output_dir: str,
    model_path: str,
    tooth_model_path: str,
    caries_conf: float = 0.01  # High accuracy default
):
    """
    Run inference.py on a directory of images.
    """
    inference_script = MAIN_DIR / "inference.py"
    
    cmd = [
        sys.executable,
        str(inference_script),
        "--image_dir", image_dir,
        "--model_path", model_path,
        "--tooth_model", tooth_model_path,
        "--output_dir", output_dir,
        "--caries_conf", str(caries_conf)
    ]
    
    print(f"\nRunning inference command:")
    print(f"  {' '.join(cmd)}")
    print("\n" + "="*60)
    
    result = subprocess.run(cmd, cwd=str(WEEK4_DIR))
    return result.returncode


def main():
    # Configuration - Using relative paths from MAIN_DIR
    SOURCE_DIR = BASE_DIR / "material" / "500 cases with annotation"
    OUTPUT_DIR = WEEK4_DIR / "inference_full_500"
    MODEL_PATH = WEEK4_DIR / "runs" / "caries_train" / "caries_3class" / "weights" / "best.pt"
    TOOTH_MODEL_PATH = BASE_DIR / "material" / "Tooth Segmentation + Recognition model" / "weights" / "Tooth_seg_pano_20250319.pt"
    
    print("="*60)
    print("FULL 500 CASES INFERENCE")
    print("="*60)
    print(f"\nSource:      {SOURCE_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Caries Model: {MODEL_PATH}")
    print(f"Tooth Model:  {TOOTH_MODEL_PATH}")
    
    # Check paths
    if not SOURCE_DIR.exists():
        print(f"\n❌ Source directory not found: {SOURCE_DIR}")
        return 1
    
    if not MODEL_PATH.exists():
        print(f"\n❌ Caries model not found: {MODEL_PATH}")
        return 1
    
    if not TOOTH_MODEL_PATH.exists():
        print(f"\n❌ Tooth model not found: {TOOTH_MODEL_PATH}")
        return 1
    
    # Step 1: Collect all images
    print("\n" + "="*60)
    print("STEP 1: COLLECTING IMAGES")
    print("="*60)
    
    images = collect_all_images(str(SOURCE_DIR))
    print(f"\nFound {len(images)} images across 500 case folders")
    
    if len(images) == 0:
        print("❌ No images found!")
        return 1
    
    # Show sample
    print("\nSample images:")
    for img in images[:5]:
        print(f"  {img}")
    print(f"  ... and {len(images) - 5} more")
    
    # Step 2: Create temp directory with all images
    print("\n" + "="*60)
    print("STEP 2: PREPARING BATCH")
    print("="*60)
    
    temp_dir = create_temp_image_list(images, str(OUTPUT_DIR))
    print(f"Created temp directory: {temp_dir}")
    print(f"Copied {len(images)} images")
    
    # Step 3: Run inference
    print("\n" + "="*60)
    print("STEP 3: RUNNING INFERENCE")
    print("="*60)
    
    return_code = run_inference_batch(
        image_dir=temp_dir,
        output_dir=str(OUTPUT_DIR),
        model_path=str(MODEL_PATH),
        tooth_model_path=str(TOOTH_MODEL_PATH),
        caries_conf=0.01  # High accuracy - low confidence threshold
    )
    
    # Step 4: Cleanup temp directory
    print("\n" + "="*60)
    print("STEP 4: CLEANUP")
    print("="*60)
    
    temp_path = Path(temp_dir)
    if temp_path.exists():
        shutil.rmtree(temp_path)
        print(f"Removed temp directory: {temp_dir}")
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    
    output_files = list(OUTPUT_DIR.glob("*_results.json"))
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"JSON files created: {len(output_files)}")
    
    if return_code == 0:
        print("\n✓ Inference completed successfully!")
        print("\nNext step: Run standardize_week4.py to organize into case folders:")
        print(f"  python standardize_week4.py --source {OUTPUT_DIR}")
    else:
        print(f"\n⚠ Inference completed with return code: {return_code}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
