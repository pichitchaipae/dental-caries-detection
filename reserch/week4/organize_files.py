"""
File Organization Script
========================
Reorganizes flat inference output into case-folder structure matching week2 format.

Reference Structure (week2/500-segmentation+recognition):
    case 1/
        case_1_bounding_boxes.png
        case_1_mask_overlay.png
        case_1_results.json
    case 2/
        ...

Source Structure (week4/inference_output_batch):
    case_1_annotated.jpg
    case_1_results.json
    case_2_annotated.jpg
    ...

Target Structure (week4/inference_organized):
    case 1/
        case_1_annotated.jpg
        case_1_results.json
    case 2/
        ...
"""

import os
import shutil
import re
from pathlib import Path


def analyze_reference_structure(reference_dir: str) -> dict:
    """Analyze the reference directory to understand the naming pattern."""
    reference_path = Path(reference_dir)
    
    analysis = {
        "folder_pattern": None,
        "file_types": set(),
        "sample_folders": [],
        "naming_convention": None
    }
    
    # Get sample folders
    folders = [f for f in reference_path.iterdir() if f.is_dir()]
    analysis["sample_folders"] = [f.name for f in folders[:5]]
    
    # Analyze folder naming pattern
    folder_pattern = re.compile(r'case\s+(\d+)', re.IGNORECASE)
    
    # Analyze files in first folder
    if folders:
        sample_folder = folders[0]
        files = list(sample_folder.iterdir())
        for f in files:
            analysis["file_types"].add(f.suffix)
        
        # Check naming convention
        file_names = [f.name for f in files]
        analysis["naming_convention"] = file_names
    
    return analysis


def extract_case_id(filename: str) -> str:
    """Extract case ID from filename like 'case_123_annotated.jpg' -> '123'"""
    match = re.search(r'case_(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def organize_files(source_dir: str, target_dir: str, dry_run: bool = False):
    """
    Organize flat files into case-folder structure.
    
    Args:
        source_dir: Directory with flat file structure
        target_dir: Directory to create organized structure
        dry_run: If True, only print what would be done
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Get all files from source
    all_files = [f for f in source_path.iterdir() if f.is_file()]
    
    # Group files by case ID
    case_files = {}
    for file in all_files:
        case_id = extract_case_id(file.name)
        if case_id:
            if case_id not in case_files:
                case_files[case_id] = []
            case_files[case_id].append(file)
    
    print("=" * 60)
    print("FILE ORGANIZATION SCRIPT")
    print("=" * 60)
    print(f"\nSource: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Mode:   {'DRY RUN (no changes)' if dry_run else 'EXECUTE'}")
    print(f"\nFound {len(all_files)} files across {len(case_files)} cases")
    print("-" * 60)
    
    # Create target directory
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
    
    # Process each case
    organized_count = 0
    for case_id in sorted(case_files.keys(), key=lambda x: int(x)):
        # Create folder name matching reference pattern: "case 123" (with space)
        folder_name = f"case {case_id}"
        case_folder = target_path / folder_name
        
        if not dry_run:
            case_folder.mkdir(exist_ok=True)
        
        # Move/copy files into case folder
        for file in case_files[case_id]:
            dest_file = case_folder / file.name
            
            if dry_run:
                print(f"  [DRY] {file.name} -> {folder_name}/")
            else:
                shutil.copy2(file, dest_file)
                organized_count += 1
    
    print("-" * 60)
    
    if dry_run:
        print(f"\n✓ DRY RUN complete. {len(all_files)} files would be organized.")
        print(f"  Run with dry_run=False to execute.")
    else:
        print(f"\n✓ Organization complete!")
        print(f"  Files organized: {organized_count}")
        print(f"  Case folders created: {len(case_files)}")
        print(f"  Output directory: {target_dir}")
    
    # Show sample of organized structure
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT STRUCTURE")
    print("=" * 60)
    
    sample_cases = sorted(case_files.keys(), key=lambda x: int(x))[:3]
    for case_id in sample_cases:
        folder_name = f"case {case_id}"
        print(f"\n  {folder_name}/")
        for file in case_files[case_id]:
            print(f"    └─ {file.name}")
    
    if len(case_files) > 3:
        print(f"\n  ... and {len(case_files) - 3} more case folders")
    
    return len(case_files), organized_count


def verify_organization(target_dir: str, reference_dir: str):
    """Verify the organized structure matches reference pattern."""
    target_path = Path(target_dir)
    reference_path = Path(reference_dir)
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Compare folder structure
    ref_folders = sorted([f.name for f in reference_path.iterdir() if f.is_dir()])
    target_folders = sorted([f.name for f in target_path.iterdir() if f.is_dir()])
    
    print(f"\nReference folders (sample): {ref_folders[:5]}")
    print(f"Target folders (sample):    {target_folders[:5]}")
    
    # Check naming pattern match
    ref_pattern = ref_folders[0] if ref_folders else None
    target_pattern = target_folders[0] if target_folders else None
    
    if ref_pattern and target_pattern:
        ref_has_space = " " in ref_pattern
        target_has_space = " " in target_pattern
        
        if ref_has_space == target_has_space:
            print(f"\n✓ Folder naming pattern matches!")
        else:
            print(f"\n⚠ Folder naming pattern differs:")
            print(f"  Reference: '{ref_pattern}'")
            print(f"  Target:    '{target_pattern}'")
    
    # Check file structure in first folder
    if target_folders:
        sample_target = target_path / target_folders[0]
        target_files = [f.name for f in sample_target.iterdir()]
        print(f"\nSample folder contents ({target_folders[0]}):")
        for f in target_files:
            print(f"  └─ {f}")


def main():
    # Define paths
    reference_dir = r"C:\Users\jaopi\Desktop\SP\week2\500-segmentation+recognition"
    source_dir = r"C:\Users\jaopi\Desktop\SP\week4\inference_output_batch"
    target_dir = r"C:\Users\jaopi\Desktop\SP\week4\inference_organized"
    
    # Step 1: Analyze reference structure
    print("\n" + "=" * 60)
    print("STEP 1: ANALYZING REFERENCE STRUCTURE")
    print("=" * 60)
    
    analysis = analyze_reference_structure(reference_dir)
    print(f"\nReference: {reference_dir}")
    print(f"Sample folders: {analysis['sample_folders']}")
    print(f"File types: {analysis['file_types']}")
    print(f"Sample files: {analysis['naming_convention']}")
    
    # Step 2: Organize files (execute mode)
    print("\n" + "=" * 60)
    print("STEP 2: ORGANIZING FILES")
    print("=" * 60)
    
    num_cases, num_files = organize_files(source_dir, target_dir, dry_run=False)
    
    # Step 3: Verify organization
    if num_cases > 0:
        verify_organization(target_dir, reference_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
