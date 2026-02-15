"""
select_hero_shots_v2.py  –  Week 7 Hero-Shot Selector (5 "Win" Categories)

Picks the most impactful validation dashboards that showcase each Week 7
improvement.

Reads:
  ● week7/evaluation_output/evaluation_results.csv
  ● week7/dental_analysis_output/case {N}/validation_case_{N}.png

Writes:
  ● week7/hero_shots/
        1_Fallback_Angle_Clamp_Win/
        2_Rescued_TP_Soft_Match_Win/
        3_Squarish_Molar_PCA_Fix_Win/
        4_Missing_Upstream_Data_Logging/
        5_Overall_Dashboard_Summary/
        hero_shots_manifest.json

Usage:
    python select_hero_shots_v2.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\jaopi\Desktop\SP")
EVAL_CSV      = BASE_DIR / "week7" / "evaluation_output" / "evaluation_results.csv"
DASHBOARD_DIR = BASE_DIR / "week7" / "dental_analysis_output"
OUTPUT_DIR    = BASE_DIR / "week7" / "hero_shots"

CATEGORIES = [
    "1_Fallback_Angle_Clamp_Win",
    "2_Rescued_TP_Soft_Match_Win",
    "3_Squarish_Molar_PCA_Fix_Win",
    "4_Missing_Upstream_Data_Logging",
    "5_Overall_Dashboard_Summary",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _dashboard(case_num: int) -> Path:
    return DASHBOARD_DIR / f"case {case_num}" / f"validation_case_{case_num}.png"


def _copy(case_num: int, category: str, suffix: str) -> bool:
    src = _dashboard(case_num)
    if not src.exists():
        return False
    dest_dir = OUTPUT_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"case_{case_num}_{suffix}.png"
    shutil.copy2(src, dest)
    return True


def _label(row: pd.Series) -> str:
    for col in ("mz_predicted_detail", "pred_surface_norm", "gt_surface_norm"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            return str(val).strip()
    return "Unknown"


def _tp(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["match_type"].isin(["tooth_id", "centroid_proximity"])].copy()


# ── Category Selectors ───────────────────────────────────────────────────────

def cat1_angle_clamp(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Category 1: Fallback Angle Clamp Win.
    
    Teeth where Rule 4 fired (pca_clamped == True) AND the prediction was
    still correct (surface_match in True/True_with_warning).
    Shows that clamping wild PCA angles to 0° preserved correct classification.
    """
    tp = _tp(df)
    pool = tp[
        (tp["pca_clamped"] == True)
        & (tp["surface_match"].isin(["True", "True_with_warning"]))
    ].copy()

    # Prefer variety: one per case, prefer Case 33 first (user-mentioned)
    pool = pool.sort_values("case")
    pool = pool.drop_duplicates(subset="case", keep="first")

    # Put Case 33 on top if present
    priority = pool[pool["case"] == 33]
    rest = pool[pool["case"] != 33]
    pool = pd.concat([priority, rest], ignore_index=True).head(n)

    results = []
    for _, row in pool.iterrows():
        c = int(row["case"])
        t = int(row["tooth_id"])
        detail = _label(row)
        match = str(row["surface_match"])
        suffix = f"tooth{t}_{detail}_clamped"
        if match == "True_with_warning":
            suffix += "_soft"
        if _copy(c, CATEGORIES[0], suffix):
            results.append({
                "case": c, "tooth_id": t, "detail": detail,
                "surface_match": match, "pca_clamped": True,
            })
    return results


def cat2_soft_match(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Category 2: Rescued TP (Soft Match Win).
    
    Teeth where strict surface matching would fail (surface_match_strict=False)
    but the soft matcher rescued them (surface_match = True_with_warning or Partial).
    Demonstrates the value of flexible surface matching.
    """
    tp = _tp(df)
    pool = tp[
        (tp["surface_match_strict"] == False)
        & (tp["surface_match"].isin(["True_with_warning", "Partial"]))
    ].copy()

    # Prefer multi-surface predictions (MOD > MO/DO > single)
    pool["_complexity"] = pool["mz_predicted_detail"].apply(
        lambda x: 3 if "MOD" in str(x)
        else (2 if ("MO" in str(x) or "DO" in str(x)) else 1)
    )

    # Favor Case 311 (user-mentioned)
    pool["_priority"] = pool["case"].apply(lambda c: 0 if c == 311 else 1)
    pool = pool.sort_values(["_priority", "_complexity"], ascending=[True, False])
    pool = pool.drop_duplicates(subset="case", keep="first").head(n)

    results = []
    for _, row in pool.iterrows():
        c = int(row["case"])
        t = int(row["tooth_id"])
        detail = _label(row)
        gt = str(row.get("gt_surface_norm", ""))
        match = str(row["surface_match"])
        suffix = f"tooth{t}_{detail}_gt-{gt}_{match}"
        if _copy(c, CATEGORIES[1], suffix):
            results.append({
                "case": c, "tooth_id": t, "detail": detail,
                "gt_surface": gt, "surface_match": match,
            })
    return results


def cat3_squarish_molar(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Category 3: Squarish Molar (PCA Fix Win).
    
    Wisdom teeth (18, 28, 38, 48) that are TP with correct surface.
    Shows that the 3-rule PCA orientation logic handles square molars properly
    where simple eigenvector analysis would fail.
    """
    tp = _tp(df)
    wisdom = {18, 28, 38, 48}
    pool = tp[
        (tp["tooth_id"].isin(wisdom))
        & (tp["surface_match"].isin(["True", "True_with_warning"]))
    ].copy()

    # Diversify: try to pick one of each tooth if possible
    pool = pool.sort_values(["tooth_id", "case"])
    pool = pool.drop_duplicates(subset="tooth_id", keep="first")
    # If we have fewer than n unique teeth, add more cases
    if len(pool) < n:
        extra = tp[
            (tp["tooth_id"].isin(wisdom))
            & (tp["surface_match"].isin(["True", "True_with_warning"]))
        ].sort_values("case")
        extra = extra[~extra.index.isin(pool.index)]
        extra = extra.drop_duplicates(subset="case", keep="first")
        pool = pd.concat([pool, extra], ignore_index=True).head(n)
    else:
        pool = pool.head(n)

    results = []
    for _, row in pool.iterrows():
        c = int(row["case"])
        t = int(row["tooth_id"])
        detail = _label(row)
        angle = row.get("rotation_angle_deg", 0.0)
        angle_str = f"{float(angle):.1f}" if pd.notna(angle) else "0.0"
        suffix = f"tooth{t}_{detail}_angle{angle_str}"
        if _copy(c, CATEGORIES[2], suffix):
            results.append({
                "case": c, "tooth_id": t, "detail": detail,
                "rotation_angle_deg": float(angle) if pd.notna(angle) else 0.0,
            })
    return results


def cat4_missing_upstream(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Category 4: Missing Upstream (Data Logging Win).
    
    FN rows — ground-truth caries that the pipeline missed.
    These could be missed by the upstream YOLO detector, or lost during the
    image-processing pipeline.  Logging these helps identify where to focus
    next improvements.
    """
    fn = df[df["match_type"] == "FN"].copy()

    # Prefer cases where multiple FNs coexist (shows a systematic gap)
    fn_counts = fn.groupby("case").size().reset_index(name="fn_count")
    fn = fn.merge(fn_counts, on="case")
    fn = fn.sort_values(["fn_count", "case"], ascending=[False, True])
    fn = fn.drop_duplicates(subset="case", keep="first").head(n)

    results = []
    for _, row in fn.iterrows():
        c = int(row["case"])
        t = int(row["tooth_id"]) if pd.notna(row["tooth_id"]) else 0
        gt = str(row.get("gt_surface_norm", "Unknown"))
        n_fn = int(row["fn_count"])
        suffix = f"FN_tooth{t}_gt-{gt}_{n_fn}missed"
        if _copy(c, CATEGORIES[3], suffix):
            results.append({
                "case": c, "tooth_id": t, "gt_surface": gt,
                "fn_in_case": n_fn,
            })
    return results


def cat5_overall_dashboard(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Category 5: Overall Dashboard Summary.
    
    Cases with many detected caries teeth (>=5) and high soft surface
    accuracy.  Shows the pipeline working well end-to-end.
    """
    tp = _tp(df)
    stats = tp.groupby("case").agg(
        n_teeth=("tooth_id", "count"),
        n_strict=("surface_match_strict", "sum"),
        n_soft=("surface_match", lambda x: x.isin(["True", "True_with_warning"]).sum()),
    ).reset_index()
    stats["acc_soft"] = stats["n_soft"] / stats["n_teeth"]
    stats["acc_strict"] = stats["n_strict"] / stats["n_teeth"]

    # Require >= 5 teeth, sort by soft accuracy then strict, then n_teeth
    good = stats[stats["n_teeth"] >= 5].sort_values(
        ["acc_soft", "acc_strict", "n_teeth"], ascending=[False, False, False]
    ).head(n)

    results = []
    for _, row in good.iterrows():
        c = int(row["case"])
        nt = int(row["n_teeth"])
        acc_s = row["acc_soft"]
        acc_st = row["acc_strict"]
        suffix = f"{nt}teeth_soft{acc_s:.0%}_strict{acc_st:.0%}"
        if _copy(c, CATEGORIES[4], suffix):
            results.append({
                "case": c, "n_teeth": nt,
                "soft_accuracy": round(float(acc_s), 3),
                "strict_accuracy": round(float(acc_st), 3),
            })
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Hero-Shot Selector v2  –  Week 7 (5 Win Categories)")
    print("=" * 65)

    # Load
    print("\n[1] Loading evaluation CSV ...")
    df = pd.read_csv(EVAL_CSV)
    print(f"    {len(df)} rows loaded")

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for cat in CATEGORIES:
        (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    # Run selectors
    all_results = {}

    print("\n── Cat 1: Fallback Angle Clamp Win ──")
    r1 = cat1_angle_clamp(df, n=5)
    all_results[CATEGORIES[0]] = r1
    print(f"   Selected {len(r1)} cases")
    for r in r1:
        print(f"     Case {r['case']:>3}  tooth {r['tooth_id']}  {r['detail']}  [{r['surface_match']}]")

    print("\n── Cat 2: Rescued TP (Soft Match Win) ──")
    r2 = cat2_soft_match(df, n=5)
    all_results[CATEGORIES[1]] = r2
    print(f"   Selected {len(r2)} cases")
    for r in r2:
        print(f"     Case {r['case']:>3}  tooth {r['tooth_id']}  pred={r['detail']}  gt={r['gt_surface']}  [{r['surface_match']}]")

    print("\n── Cat 3: Squarish Molar (PCA Fix Win) ──")
    r3 = cat3_squarish_molar(df, n=5)
    all_results[CATEGORIES[2]] = r3
    print(f"   Selected {len(r3)} cases")
    for r in r3:
        print(f"     Case {r['case']:>3}  tooth {r['tooth_id']}  {r['detail']}  angle={r['rotation_angle_deg']:.1f}°")

    print("\n── Cat 4: Missing Upstream (Data Logging) ──")
    r4 = cat4_missing_upstream(df, n=5)
    all_results[CATEGORIES[3]] = r4
    print(f"   Selected {len(r4)} cases")
    for r in r4:
        print(f"     Case {r['case']:>3}  tooth {r['tooth_id']}  gt={r['gt_surface']}  ({r['fn_in_case']} FN in case)")

    print("\n── Cat 5: Overall Dashboard Summary ──")
    r5 = cat5_overall_dashboard(df, n=5)
    all_results[CATEGORIES[4]] = r5
    print(f"   Selected {len(r5)} cases")
    for r in r5:
        print(f"     Case {r['case']:>3}  {r['n_teeth']} teeth  soft={r['soft_accuracy']:.0%}  strict={r['strict_accuracy']:.0%}")

    # Manifest
    total = sum(len(v) for v in all_results.values())
    manifest = {**all_results, "total_images": total}
    manifest_path = OUTPUT_DIR / "hero_shots_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "=" * 65)
    print(f"  TOTAL: {total} hero-shot images copied")
    print(f"  Output → {OUTPUT_DIR}")
    print(f"  Manifest → {manifest_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
