"""
Dental Caries Analysis Pipeline — week7 (Boundary Leakage + Unassigned Caries)
================================================================================

Fixes applied vs. week3:

  **Task 2 – Boundary Leakage (False Positives from Overlapping Masks)**
    The old ``calculate_caries_overlap()`` counted raw pixel overlap.
    When a large caries lesion on an adjacent tooth (e.g. Tooth 47)
    spills into the bounding-box/mask of the neighbouring tooth
    (e.g. Tooth 48), even a tiny overlap (82 pixels) was flagged as
    ``has_caries=True``.

    New safeguards:
      1. **Morphological Erosion** – apply ``cv2.erode()`` with a 5×5
         kernel on the individual tooth mask *before* overlap checking.
         This strips edge pixels that are likely boundary bleed.
      2. **Size / Percentage Thresholding** –
         a tooth is only marked ``has_caries=True`` when
         ``caries_pixels > 100`` **OR** ``caries_percentage > 1.0 %``.
         Otherwise the hit is discarded.

  **Task 3 – Cascading Errors (Missing Upstream Tooth Data)**
    When the week2 segmentation model fails to detect a tooth (e.g.
    Molars 17, 18, 47 in case 33), valid GT caries in the ROI mask
    are silently ignored.

    New **Fallback Logic**: after mapping all detected teeth, any
    remaining ROI caries blob of > 150 pixels that was *not* assigned
    to a tooth is logged as:
        ``{"unassigned_caries": true, "pixels": N, "centroid": [x,y]}``
    in the output JSON.

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026-02
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
# Constants for Task 2 thresholds
# =============================================================================

# Morphological erosion kernel size (Task 2, step 1)
EROSION_KERNEL_SIZE = 5

# Minimum absolute pixels to declare caries (Task 2, step 2)
MIN_CARIES_PIXELS = 100

# Minimum percentage of tooth area to declare caries (Task 2, step 2)
MIN_CARIES_PERCENTAGE = 1.0  # percent

# Minimum blob size for unassigned-caries fallback (Task 3)
MIN_UNASSIGNED_BLOB_PIXELS = 150


# =============================================================================
# ROI & JSON Loading Functions
# =============================================================================

def load_roi_image(roi_path):
    roi_img = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi_img is None:
        raise FileNotFoundError(f"Could not load ROI image: {roi_path}")
    return roi_img


def load_tooth_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# =============================================================================
# Task 2 FIX – Caries Overlap with Erosion + Size/Percentage Guard
# =============================================================================

def _build_tooth_mask(pixel_coordinates, height, width):
    """
    Create a binary mask for a single tooth from its pixel coordinates.
    Returns the mask as uint8 (255 = tooth, 0 = background).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for coord in pixel_coordinates:
        x, y = int(coord[0]), int(coord[1])
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 255
    return mask


def calculate_caries_overlap(roi_img, pixel_coordinates):
    """
    Calculate caries-tooth overlap **with boundary-leakage protection**.

    Task 2 fixes applied here:
      1. Build a binary tooth mask and erode it (5×5 kernel) to remove
         edge pixels before checking overlap.
      2. After counting caries pixels, apply a minimum threshold:
         the tooth is ``has_caries`` only when caries_pixels > 100
         OR caries_percentage > 1.0 %.

    Returns:
        (caries_count, valid_pixels, percentage, caries_coords)
    """
    if not pixel_coordinates:
        return 0, 0, 0.0, []

    height, width = roi_img.shape

    # ── Step 1: Build tooth mask & apply morphological erosion ───────
    tooth_mask = _build_tooth_mask(pixel_coordinates, height, width)

    kernel = np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE), np.uint8)
    eroded_mask = cv2.erode(tooth_mask, kernel, iterations=1)

    # ── Step 2: Count overlap with eroded tooth mask ─────────────────
    # Only pixels that survive erosion AND are in the ROI count.
    overlap = cv2.bitwise_and(eroded_mask, roi_img)

    caries_coords = []
    caries_ys, caries_xs = np.where(overlap > 0)
    caries_count = len(caries_xs)
    for x, y in zip(caries_xs.tolist(), caries_ys.tolist()):
        caries_coords.append([x, y])

    # Total valid (eroded) pixels
    valid_pixels = int(np.count_nonzero(eroded_mask))

    percentage = (caries_count / valid_pixels * 100) if valid_pixels > 0 else 0.0

    # ── Step 3: Size / Percentage Thresholding (Task 2, step 2) ──────
    # Discard if below BOTH thresholds  →  treat as boundary noise
    if caries_count <= MIN_CARIES_PIXELS and percentage <= MIN_CARIES_PERCENTAGE:
        # Below threshold – discard as boundary leakage
        return 0, valid_pixels, 0.0, []

    return caries_count, valid_pixels, percentage, caries_coords


# =============================================================================
# Task 3 FIX – Detect Unassigned Caries Blobs (Missing Upstream Teeth)
# =============================================================================

def find_unassigned_caries(roi_img, tooth_data):
    """
    After all teeth have been mapped, check if there are remaining
    caries blobs in the ROI mask that were NOT assigned to any tooth.

    A blob is considered "unassigned" if:
      - Its area is > MIN_UNASSIGNED_BLOB_PIXELS (150 px)
      - None of its pixels fall inside any detected tooth mask

    Returns a list of dicts:
        [{"unassigned_caries": True, "pixels": N, "centroid": [x, y]}, ...]
    """
    height, width = roi_img.shape

    # ── Build combined mask of ALL detected teeth ────────────────────
    all_teeth_mask = np.zeros((height, width), dtype=np.uint8)
    for tooth in tooth_data.get('teeth_data', []):
        pixel_coords = tooth.get('pixel_coordinates', [])
        for coord in pixel_coords:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < width and 0 <= y < height:
                all_teeth_mask[y, x] = 255

    # ── ROI caries pixels NOT covered by any tooth ───────────────────
    roi_binary = np.where(roi_img > 0, 255, 0).astype(np.uint8)
    unassigned_mask = cv2.bitwise_and(roi_binary, cv2.bitwise_not(all_teeth_mask))

    # ── Connected-component analysis on unassigned mask ──────────────
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        unassigned_mask, connectivity=8
    )

    unassigned_blobs = []
    for lbl in range(1, n_labels):  # skip background (0)
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > MIN_UNASSIGNED_BLOB_PIXELS:
            cx, cy = centroids[lbl]
            unassigned_blobs.append({
                "unassigned_caries": True,
                "pixels": int(area),
                "centroid": [round(float(cx), 2), round(float(cy), 2)],
            })

    return unassigned_blobs


# =============================================================================
# Caries Mapping (updated with Task 2 + Task 3)
# =============================================================================

def generate_caries_mapping(roi_img, tooth_data, case_num):
    """
    Map each detected tooth to the ROI caries mask.
    Uses the Task 2-fixed ``calculate_caries_overlap()``.
    After all teeth, runs Task 3 ``find_unassigned_caries()``.
    """
    results = []

    for tooth in tooth_data.get('teeth_data', []):
        tooth_id = tooth.get('tooth_id', 'unknown')
        pixel_coords = tooth.get('pixel_coordinates', [])
        confidence = tooth.get('confidence', 0.0)

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
            'caries_coordinates': caries_coords,
        })

    # ── Task 3: Detect unassigned caries ─────────────────────────────
    unassigned = find_unassigned_caries(roi_img, tooth_data)
    if unassigned:
        for blob in unassigned:
            results.append({
                'case_number': case_num,
                'tooth_id': 'UNASSIGNED',
                'confidence': 0.0,
                'total_pixels': 0,
                'caries_pixels': blob['pixels'],
                'caries_percentage': 0.0,
                'has_caries': True,
                'caries_coordinates': [],
                'unassigned_caries': True,
                'unassigned_centroid': blob['centroid'],
            })

    return results


# =============================================================================
# Visual Alignment Functions (unchanged from week3)
# =============================================================================

def create_alignment_visualization(roi_img, tooth_data, alpha=0.5):
    height, width = roi_img.shape
    roi_normalized = np.where(roi_img > 0, 255, 0).astype(np.uint8)
    roi_bgr = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)
    tooth_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [
        (0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 165, 255),
        (255, 0, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255),
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
        'roi_shape': (height, width),
    }


def create_detailed_alignment_image(roi_img, tooth_data):
    height, width = roi_img.shape
    roi_normalized = np.where(roi_img > 0, 255, 0).astype(np.uint8)
    roi_colored = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)
    caries_mask = roi_img > 0
    roi_colored[caries_mask] = [0, 0, 255]
    tooth_only = np.zeros((height, width, 3), dtype=np.uint8)
    for tooth in tooth_data.get('teeth_data', []):
        pixel_coords = tooth.get('pixel_coordinates', [])
        for coord in pixel_coords:
            x, y = coord[0], coord[1]
            if 0 <= x < width and 0 <= y < height:
                tooth_only[y, x] = [0, 255, 0]
    blended, stats = create_alignment_visualization(roi_img, tooth_data, alpha=0.6)
    tooth_mask = np.any(tooth_only > 0, axis=2)
    overlap_pixels = np.sum(tooth_mask & caries_mask)
    caries_pixels = np.sum(caries_mask)
    tooth_pixels = np.sum(tooth_mask)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    cv2.putText(roi_colored, "ROI (Red=Caries)", (20, 60), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(tooth_only, "Teeth (Green)", (20, 60), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(blended, "ALIGNMENT CHECK", (20, 60), font, font_scale, (0, 255, 255), thickness)
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
    scale = 0.5
    roi_small = cv2.resize(roi_colored, None, fx=scale, fy=scale)
    tooth_small = cv2.resize(tooth_only, None, fx=scale, fy=scale)
    blended_small = cv2.resize(blended, None, fx=scale, fy=scale)
    combined = np.hstack([roi_small, tooth_small, blended_small])
    return combined, stats, {
        'overlap_pixels': int(overlap_pixels),
        'caries_pixels': int(caries_pixels),
        'tooth_pixels': int(tooth_pixels),
    }


# =============================================================================
# Main Processing Function
# =============================================================================

def process_single_case(case_num, json_dir, roi_dir, output_dir):
    json_path = Path(json_dir) / f"case {case_num}" / f"case_{case_num}_results.json"
    roi_path = Path(roi_dir) / f"case_{case_num}.png"

    if not json_path.exists():
        return False, "JSON not found", None
    if not roi_path.exists():
        return False, "ROI not found", None

    roi_img = load_roi_image(roi_path)
    tooth_data = load_tooth_json(json_path)

    case_output_dir = Path(output_dir) / f"case {case_num}"
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Caries mapping (with Task 2 erosion + Task 3 unassigned detection)
    caries_results = generate_caries_mapping(roi_img, tooth_data, case_num)

    # Separate unassigned blobs for JSON output
    tooth_results = [r for r in caries_results if r.get('tooth_id') != 'UNASSIGNED']
    unassigned_results = [r for r in caries_results if r.get('tooth_id') == 'UNASSIGNED']

    # Save JSON
    caries_json_path = case_output_dir / f"case_{case_num}_caries_mapping.json"
    output_data = {
        'case_number': case_num,
        'pipeline_version': 'week7',
        'fixes_applied': [
            'Task2_boundary_erosion',
            'Task2_size_threshold',
            'Task3_unassigned_caries_detection',
        ],
        'erosion_kernel': EROSION_KERNEL_SIZE,
        'min_caries_pixels': MIN_CARIES_PIXELS,
        'min_caries_percentage': MIN_CARIES_PERCENTAGE,
        'min_unassigned_blob': MIN_UNASSIGNED_BLOB_PIXELS,
        'teeth_caries_data': tooth_results,
    }
    if unassigned_results:
        output_data['unassigned_caries'] = unassigned_results
        print(f"  [WARN] Case {case_num}: {len(unassigned_results)} unassigned caries blob(s) detected")

    with open(caries_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Alignment debug image
    combined_img, stats, overlap_stats = create_detailed_alignment_image(roi_img, tooth_data)
    alignment_img_path = case_output_dir / f"case_{case_num}_alignment_detailed.png"
    cv2.imwrite(str(alignment_img_path), combined_img)

    teeth_with_caries = sum(1 for r in tooth_results if r['has_caries'])
    msg = f"{len(tooth_results)} teeth, {teeth_with_caries} with caries"
    if unassigned_results:
        msg += f", {len(unassigned_results)} unassigned blob(s)"
    return True, msg, caries_results


def process_all_cases(json_dir, roi_dir, output_dir, num_cases=500):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_caries_results = []
    summary_stats = {
        'total_cases': 0, 'processed_cases': 0, 'failed_cases': 0,
        'total_teeth': 0, 'teeth_with_caries': 0, 'unassigned_blobs': 0,
    }

    print("=" * 70)
    print("Dental Caries Analysis Pipeline – week7 (with Erosion + Unassigned)")
    print("=" * 70)
    print(f"JSON Source: {json_dir}")
    print(f"ROI Source:  {roi_dir}")
    print(f"Output:      {output_dir}")
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
                tooth_results = [r for r in caries_results if r.get('tooth_id') != 'UNASSIGNED']
                unassigned = [r for r in caries_results if r.get('tooth_id') == 'UNASSIGNED']
                summary_stats['total_teeth'] += len(tooth_results)
                summary_stats['teeth_with_caries'] += sum(1 for r in tooth_results if r['has_caries'])
                summary_stats['unassigned_blobs'] += len(unassigned)
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

    if all_caries_results:
        df = pd.DataFrame(all_caries_results)
        csv_path = Path(output_dir) / "caries_mapping_results.csv"
        df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    for k, v in summary_stats.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    return all_caries_results, summary_stats


def main():
    parser = argparse.ArgumentParser(
        description="Dental Caries Analysis Pipeline – week7 (Erosion + Unassigned Caries)"
    )
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--sample", type=str, default=None,
                        help="Comma-separated case numbers, e.g. --sample 311,33")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    json_dir = base_dir / "week2" / "500-segmentation+recognition"
    roi_dir  = base_dir / "material" / "500-roi"
    output_dir = base_dir / "week7" / "dental_analysis_output"

    if args.sample:
        case_nums = [int(c.strip()) for c in args.sample.split(',')]
        print(f"Processing specific cases: {case_nums}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for case_num in case_nums:
            success, msg, results = process_single_case(case_num, json_dir, roi_dir, output_dir)
            status = "OK" if success else "FAIL"
            print(f"  Case {case_num}: [{status}] {msg}")
    else:
        process_all_cases(json_dir, roi_dir, output_dir, num_cases=args.cases)


if __name__ == "__main__":
    main()
