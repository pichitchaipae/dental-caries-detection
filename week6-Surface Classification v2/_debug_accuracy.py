"""Quick diagnostic: show what eval compares vs what MZ produces."""
import sys
sys.path.insert(0, ".")
from evaluation_engine import evaluate_single_case, normalize_surface, normalize_surface_fine

total_match = 0
total_tp = 0

for case_num in [1, 2, 3, 4, 5]:
    r = evaluate_single_case(case_num, reclassify=True)
    print(f"{'='*80}")
    print(f"CASE {case_num}: GT={r['gt_count']} Pred={r['pred_count']} "
          f"TP={len(r['matched'])} FP={len(r['fp'])} FN={len(r['fn'])}")
    print(f"{'='*80}")

    for m in r["matched"]:
        gt = m["gt"]
        pred = m["pred"]
        tid = pred.get("tooth_id", "")
        gt_raw = gt.get("surface_name", "")

        # Find MZ
        mz = {}
        for x in r.get("mz_results", []):
            if x.get("tooth_id") == tid:
                mz = x
                break

        gt_n = normalize_surface(gt_raw)
        mz_fine = mz.get("predicted_surface_fine", "")
        mz_detail = mz.get("predicted_detail", "")
        all_frac = mz.get("all_zone_fractions", {})
        
        if mz_fine and mz_fine != "Unknown":
            pred_n = normalize_surface(mz_fine)
        else:
            pred_n = normalize_surface(pred.get("caries_surface", ""))

        pred_w5 = pred.get("caries_surface", "")
        pred_w5_n = normalize_surface(pred_w5)

        match_new = "OK" if gt_n == pred_n else "MISS"
        match_old = "OK" if gt_n == pred_w5_n else "MISS"
        total_tp += 1
        if gt_n == pred_n:
            total_match += 1

        print(f"  T#{tid:3s} | GT={gt_raw:12s}({gt_n:10s}) | "
              f"OLD={pred_w5:10s}({pred_w5_n:10s}) {match_old:4s} | "
              f"NEW={mz_fine:10s}({pred_n:10s}) {match_new:4s} | "
              f"detail={mz_detail:4s} allM={all_frac.get('M',0):.2f} "
              f"C={all_frac.get('C',0):.2f} D={all_frac.get('D',0):.2f}")
    print()

print(f"{'='*80}")
print(f"Surface accuracy (NEW): {total_match}/{total_tp} = {total_match/total_tp:.4f}")
print(f"{'='*80}")
