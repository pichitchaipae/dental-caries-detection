"""Quick test of refined multi-zone classifier."""
import json, cv2, numpy as np
from pathlib import Path
from multi_zone_classifier import classify_from_week_data

BASE = Path(r"C:\Users\jaopi\Desktop\SP")

for case_num in [1, 5, 10]:
    w2p = BASE / "week2" / "500-segmentation+recognition" / f"case {case_num}" / f"case_{case_num}_results.json"
    with open(w2p) as f:
        w2 = json.load(f)
    polys = {}
    for t in w2.get("teeth_data", []):
        tid = t.get("tooth_id", "")
        poly = t.get("polygon")
        if not poly:
            pix = t.get("pixel_coordinates")
            if pix and len(pix) >= 3:
                hull = cv2.convexHull(np.array(pix, dtype=np.float32))
                poly = hull.reshape(-1, 2).tolist()
        if poly:
            polys[tid] = poly

    w3p = BASE / "week3" / "dental_analysis_output" / f"case {case_num}" / f"case_{case_num}_caries_mapping.json"
    with open(w3p) as f:
        w3 = json.load(f)
    w3_lookup = {t["tooth_id"]: t for t in w3.get("teeth_caries_data", []) if t.get("tooth_id")}

    print(f"=== Case {case_num} ===")
    for tid, w3d in w3_lookup.items():
        coords = w3d.get("caries_coordinates", [])
        poly = polys.get(tid, [])
        if not coords or not poly:
            continue
        mz = classify_from_week_data(tid, poly, coords, save_debug=(case_num == 1), case_num=case_num)
        surf = mz["predicted_surface_fine"]
        detail = mz["predicted_detail"]
        af = mz["all_zone_fractions"]
        raw = mz["n_points_raw"]
        clean = mz["n_points_cleaned"]
        removed = mz["n_points_removed"]
        expl = mz.get("detail_explanation", "")
        print(f"  Tooth {tid}: {surf:>9s} | {detail:>4s} | "
              f"M={af['M']:.0%} C={af['C']:.0%} D={af['D']:.0%} | "
              f"raw={raw} clean={clean} removed={removed}")
        if expl:
            print(f"          -> {expl}")
    print()
