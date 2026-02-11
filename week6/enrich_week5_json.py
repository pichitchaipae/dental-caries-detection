"""
Enrich Week5 Diagnosis JSON with caries_coordinates
=====================================================

This script reads each ``case_N_diagnosis.json`` in
``week5/surface_classification_output``, injects the missing
``caries_coordinates`` (pixel-level [x, y] lists) from the
week3 ``dental_analysis_output`` data, and **also** re-runs the
Multi-Zone (M-C-D) classifier to add ``zone_label`` and zone fractions.

Enriched files are saved to:
    week6/evaluation_output/case N/case_N_diagnosis.json

Additionally, a summary CSV is written:
    week6/evaluation_output/enrichment_summary.csv

Usage:
    python enrich_week5_json.py                   # all 500 cases
    python enrich_week5_json.py --case 1          # single case
    python enrich_week5_json.py --start 1 --end 10

Author: Senior Research Engineer – Dental AI / CAD
Date: 2026
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from multi_zone_classifier import classify_from_week_data

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"C:\Users\jaopi\Desktop\SP")

WEEK2_DIR  = BASE_DIR / "week2"  / "500-segmentation+recognition"
WEEK3_DIR  = BASE_DIR / "week3"  / "dental_analysis_output"
WEEK5_DIR  = BASE_DIR / "week5"  / "surface_classification_output"
OUTPUT_DIR = BASE_DIR / "week6"  / "evaluation_output"


# =============================================================================
# Helpers
# =============================================================================

def load_week3_caries(case_num: int) -> Dict:
    p = WEEK3_DIR / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["tooth_id"]: t for t in data.get("teeth_caries_data", []) if t.get("tooth_id")}


def load_week2_polygons(case_num: int) -> Dict:
    p = WEEK2_DIR / f"case {case_num}" / f"case_{case_num}_results.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for tooth in data.get("teeth_data", []):
        tid = tooth.get("tooth_id", "")
        if not tid:
            continue
        poly = tooth.get("polygon")
        if not poly:
            pixels = tooth.get("pixel_coordinates")
            if pixels and len(pixels) >= 3:
                arr = np.array(pixels, dtype=np.float32)
                hull = cv2.convexHull(arr)
                poly = hull.reshape(-1, 2).tolist()
        if poly:
            out[tid] = poly
    return out


def load_week5_json(case_num: int) -> Optional[Dict]:
    p = WEEK5_DIR / f"case {case_num}" / f"case_{case_num}_diagnosis.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Core Enrichment
# =============================================================================

def enrich_single_case(case_num: int) -> Optional[Dict]:
    """
    Enrich a single case's week5 JSON:
      1) Add ``caries_coordinates``, ``caries_pixels``, ``caries_percentage``
      2) Add ``zone_label``, ``zone_fractions`` from Multi-Zone classifier
      3) Save to week6 output folder

    Returns the enriched dict or None.
    """
    w5 = load_week5_json(case_num)
    if w5 is None:
        return None

    w3_lookup = load_week3_caries(case_num)
    w2_polys  = load_week2_polygons(case_num)

    teeth_data = w5.get("teeth_data", [])

    for tooth in teeth_data:
        tid = tooth.get("tooth_id", "")
        w3  = w3_lookup.get(tid, {})

        # ── Inject caries_coordinates ────────────────────────────────
        coords = w3.get("caries_coordinates", [])
        tooth["caries_coordinates"] = coords
        tooth["caries_pixels"]      = w3.get("caries_pixels", len(coords))
        tooth["caries_percentage"]  = round(w3.get("caries_percentage", 0.0), 4)

        # ── Multi-zone classification ────────────────────────────────
        poly = w2_polys.get(tid, [])
        if coords and poly:
            mz = classify_from_week_data(tid, poly, coords)
            tooth["zone_label"]     = mz.get("zone_label", "")
            tooth["zone_fractions"] = mz.get("zone_fractions", {})
            tooth["mz_primary_surface"] = mz.get("primary_surface", "")
            # New: all-points surface classification
            tooth["predicted_surface_fine"] = mz.get("predicted_surface_fine", "")
            tooth["predicted_detail"]       = mz.get("predicted_detail", "")
            tooth["all_zone_fractions"]     = mz.get("all_zone_fractions", {})
        else:
            tooth["zone_label"]     = ""
            tooth["zone_fractions"] = {"M": 0, "C": 0, "D": 0}
            tooth["mz_primary_surface"] = ""
            tooth["predicted_surface_fine"] = ""
            tooth["predicted_detail"]       = ""
            tooth["all_zone_fractions"]     = {"M": 0, "C": 0, "D": 0}

    enriched = {
        "case_number": case_num,
        "enriched_at": datetime.now().isoformat(),
        "source": "week5 + week3 caries_coordinates + multi_zone_classifier",
        "teeth_data": teeth_data,
    }

    # Save
    out_dir = OUTPUT_DIR / f"case {case_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_{case_num}_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    return enriched


# =============================================================================
# Batch Processing
# =============================================================================

def enrich_all_cases(start: int = 1, end: int = 500, verbose: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    enriched_count = 0
    skipped_count  = 0

    if verbose:
        print("=" * 60)
        print("ENRICH WEEK5 JSON → week6 (with caries_coordinates + MZ)")
        print("=" * 60)

    for case_num in range(start, end + 1):
        result = enrich_single_case(case_num)
        if result is None:
            skipped_count += 1
            continue

        enriched_count += 1
        for tooth in result.get("teeth_data", []):
            summary_rows.append({
                "case": case_num,
                "tooth_id": tooth.get("tooth_id", ""),
                "caries_surface": tooth.get("caries_surface", ""),
                "caries_position_detail": tooth.get("caries_position_detail", ""),
                "zone_label": tooth.get("zone_label", ""),
                "zone_M": tooth.get("zone_fractions", {}).get("M", 0),
                "zone_C": tooth.get("zone_fractions", {}).get("C", 0),
                "zone_D": tooth.get("zone_fractions", {}).get("D", 0),
                "caries_pixels": tooth.get("caries_pixels", 0),
                "caries_percentage": tooth.get("caries_percentage", 0),
                "has_coordinates": len(tooth.get("caries_coordinates", [])) > 0,
            })

        if verbose and enriched_count % 50 == 0:
            print(f"  ... enriched {enriched_count} cases")

    # Save summary CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = OUTPUT_DIR / "enrichment_summary.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\n  Summary CSV → {csv_path}")

    if verbose:
        print(f"\n  Enriched : {enriched_count}")
        print(f"  Skipped  : {skipped_count}")
        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Enrich week5 JSON with caries_coordinates and MZ zones"
    )
    parser.add_argument("--case", type=int, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    args = parser.parse_args()

    if args.case is not None:
        result = enrich_single_case(args.case)
        if result:
            print(f"✓ Case {args.case} enriched → {OUTPUT_DIR / f'case {args.case}'}")
            for t in result["teeth_data"]:
                n = len(t.get("caries_coordinates", []))
                zl = t.get("zone_label", "N/A")
                print(f"  Tooth {t['tooth_id']}: {n} coords, zone={zl}")
        else:
            print(f"✗ Case {args.case}: no week5 data found")
    else:
        enrich_all_cases(args.start, args.end)


if __name__ == "__main__":
    main()
