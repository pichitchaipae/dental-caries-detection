"""
select_hero_shots.py  –  Cherry-pick the most impactful validation dashboards
                         for presentation / report.

Reads:
  ● evaluation_output/evaluation_results.csv   (main evaluation CSV)
  ● week4/inference_full_500/case_*_results.json  (caries confidence + IoU)
  ● evaluation_output/case {N}/validation_case_{N}.png  (dashboard images)

Writes:
  ● week6/presentation_hero_shots/
        1_The_Perfect_Match/
        2_The_Complex_Win/
        3_The_AI_Eye_Potential/
        4_The_Honest_Mistake/

Usage:
    python select_hero_shots.py
"""

import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\jaopi\Desktop\SP")
EVAL_CSV     = BASE_DIR / "week6" / "evaluation_output" / "evaluation_results.csv"
WEEK4_DIR    = BASE_DIR / "week4" / "inference_full_500"
DASHBOARD_DIR = BASE_DIR / "week6" / "evaluation_output"
OUTPUT_DIR   = BASE_DIR / "week6" / "presentation_hero_shots"

CATEGORIES = {
    "1_The_Perfect_Match":   OUTPUT_DIR / "1_The_Perfect_Match",
    "2_The_Complex_Win":     OUTPUT_DIR / "2_The_Complex_Win",
    "3_The_AI_Eye_Potential": OUTPUT_DIR / "3_The_AI_Eye_Potential",
    "4_The_Honest_Mistake":  OUTPUT_DIR / "4_The_Honest_Mistake",
}

random.seed(42)  # reproducible random picks


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_week4_confidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load caries-detection confidence + IoU from week4 JSON files.

    Returns:
        tooth_df : per-(case, tooth_id) best detection
        case_df  : per-case max confidence (fallback for FP rows)
    """
    rows = []
    for case_num in range(1, 501):
        json_path = WEEK4_DIR / f"case_{case_num}_results.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        for det in data.get("caries_detections", []):
            rows.append({
                "case":              case_num,
                "tooth_id":          int(det["tooth_id"]) if det.get("tooth_id") else None,
                "caries_confidence": det.get("confidence", 0.0),
                "iou_with_tooth":    det.get("iou_with_tooth", 0.0),
                "class_name_w4":     det.get("class_name", ""),
            })
    all_df = pd.DataFrame(rows)
    if all_df.empty:
        empty = pd.DataFrame(columns=["case", "tooth_id", "caries_confidence", "iou_with_tooth"])
        return empty, pd.DataFrame(columns=["case", "case_max_confidence"])

    # Per-tooth best detection
    tooth_df = (
        all_df.sort_values("caries_confidence", ascending=False)
              .drop_duplicates(subset=["case", "tooth_id"], keep="first")
    )
    # Per-case max confidence (fallback)
    case_df = (
        all_df.groupby("case")["caries_confidence"]
              .max().reset_index()
              .rename(columns={"caries_confidence": "case_max_confidence"})
    )
    return tooth_df, case_df


def _best_label(row: pd.Series) -> str:
    """Pick the most informative surface label from a row."""
    for col in ["mz_predicted_detail", "pred_surface_norm", "pred_surface", "gt_surface_norm"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip() and str(val) != "nan":
            return str(val).strip()
    return "Unknown"


def dashboard_path(case_num: int) -> Path:
    return DASHBOARD_DIR / f"case {case_num}" / f"validation_case_{case_num}.png"


def copy_image(case_num: int, dest_dir: Path, suffix: str) -> bool:
    """Copy dashboard image to dest_dir with a descriptive filename.
    Returns True if successful."""
    src = dashboard_path(case_num)
    if not src.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"case_{case_num}_{suffix}.png"
    shutil.copy2(src, dest)
    return True


# ── Selection Logic ──────────────────────────────────────────────────────────
def select_perfect_match(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Category 1: TP + GT==Pred surface + highest IoU (lowest distance_px)."""
    tp = df[df["match_type"].isin(["tooth_id", "centroid_proximity"])].copy()
    perfect = tp[tp["surface_match"] == True].copy()

    # Rank by IoU desc then distance_px asc
    has_iou = "iou_with_tooth" in perfect.columns and perfect["iou_with_tooth"].notna().any()
    if has_iou:
        perfect = perfect.sort_values(
            ["iou_with_tooth", "distance_px"], ascending=[False, True]
        )
    else:
        perfect = perfect.sort_values("distance_px", ascending=True)

    # De-duplicate to one entry per case (pick best row per case)
    perfect = perfect.drop_duplicates(subset="case", keep="first")
    selected = perfect.head(top_n)

    results = []
    for _, row in selected.iterrows():
        c = int(row["case"])
        iou_val = row.get("iou_with_tooth")
        dist_val = row.get("distance_px", 999)
        metric_str = f"iou{iou_val:.2f}" if pd.notna(iou_val) else f"dist{dist_val:.1f}"
        label = _best_label(row)
        suffix = f"{label}_{metric_str}"
        if copy_image(c, CATEGORIES["1_The_Perfect_Match"], suffix):
            results.append({"case": c, "label": label, "metric": metric_str})
    return results


def select_complex_win(df: pd.DataFrame) -> list[dict]:
    """Category 2: TP + multi-surface (MO / DO / MOD)."""
    tp = df[df["match_type"].isin(["tooth_id", "centroid_proximity"])].copy()
    multi = tp[
        tp["mz_predicted_detail"].str.contains("MO|DO|MOD", na=False, regex=True)
    ].copy()

    # De-duplicate per case (keep row with most complex label: MOD > MO/DO)
    multi["_complexity"] = multi["mz_predicted_detail"].map(
        lambda x: 3 if "MOD" in str(x) else (2 if "MO" in str(x) or "DO" in str(x) else 1)
    )
    multi = multi.sort_values("_complexity", ascending=False)
    multi = multi.drop_duplicates(subset="case", keep="first")

    results = []
    for _, row in multi.iterrows():
        c = int(row["case"])
        label = row["mz_predicted_detail"]
        conf_val = row.get("caries_confidence")
        conf = f"conf{conf_val:.2f}" if pd.notna(conf_val) else ""
        suffix = f"{label}_{conf}" if conf else label
        if copy_image(c, CATEGORIES["2_The_Complex_Win"], suffix):
            results.append({"case": c, "label": label})
    return results


def select_ai_eye(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Category 3: FP + high caries confidence (>0.85) → "Second Opinion".
    Uses per-tooth confidence when available, falls back to case-level max."""
    fp = df[df["match_type"] == "FP"].copy()

    # Effective confidence: per-tooth first, then case-level fallback
    fp["eff_confidence"] = fp["caries_confidence"]
    if "case_max_confidence" in fp.columns:
        fp["eff_confidence"] = fp["eff_confidence"].fillna(fp["case_max_confidence"])

    high = fp[fp["eff_confidence"] > 0.85].sort_values(
        "eff_confidence", ascending=False
    )
    high = high.drop_duplicates(subset="case", keep="first").head(top_n)

    results = []
    for _, row in high.iterrows():
        c = int(row["case"])
        conf_val = row["eff_confidence"]
        label = _best_label(row)
        suffix = f"{label}_conf{conf_val:.2f}"
        if copy_image(c, CATEGORIES["3_The_AI_Eye_Potential"], suffix):
            results.append({"case": c, "confidence": round(float(conf_val), 3), "label": label})
    return results


def select_honest_mistake(df: pd.DataFrame, n_fp: int = 10, n_fn: int = 10) -> list[dict]:
    """Category 4: Low-confidence FPs + FNs → error analysis."""
    # FP with confidence < 0.6
    fp_low = df[(df["match_type"] == "FP")].copy()
    if "caries_confidence" in fp_low.columns:
        fp_low = fp_low[fp_low["caries_confidence"] < 0.6]
    fp_low = fp_low.drop_duplicates(subset="case", keep="first")
    fp_cases = fp_low["case"].tolist()
    fp_pick = random.sample(fp_cases, min(n_fp, len(fp_cases)))

    # FN
    fn = df[df["match_type"] == "FN"].drop_duplicates(subset="case", keep="first")
    fn_cases = fn["case"].tolist()
    fn_pick = random.sample(fn_cases, min(n_fn, len(fn_cases)))

    results = []
    for c in fp_pick:
        row = df[(df["case"] == c) & (df["match_type"] == "FP")].iloc[0]
        label = _best_label(row)
        conf_val = row.get("caries_confidence")
        if pd.isna(conf_val):
            conf_val = row.get("case_max_confidence", 0.0)
        conf_val = float(conf_val) if pd.notna(conf_val) else 0.0
        suffix = f"FP_{label}_conf{conf_val:.2f}"
        if copy_image(int(c), CATEGORIES["4_The_Honest_Mistake"], suffix):
            results.append({"case": int(c), "type": "FP", "label": label,
                            "confidence": round(conf_val, 3)})

    for c in fn_pick:
        row = df[(df["case"] == c) & (df["match_type"] == "FN")].iloc[0]
        gt_label = row.get("gt_surface_norm", row.get("gt_surface", "FN"))
        suffix = f"FN_gt-{gt_label}"
        if copy_image(int(c), CATEGORIES["4_The_Honest_Mistake"], suffix):
            results.append({"case": int(c), "type": "FN", "gt_label": str(gt_label)})

    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Hero-Shot Selector  –  Cherry Picking for Presentation")
    print("=" * 65)

    # ── 1) Load evaluation CSV ───────────────────────────────────────────
    print("\n[1/4] Loading evaluation CSV ...")
    df = pd.read_csv(EVAL_CSV)
    print(f"      {len(df)} rows, columns: {list(df.columns)[:8]} ...")

    # ── 2) Load week4 confidence + IoU ───────────────────────────────────
    print("[2/4] Loading week4 caries confidence & IoU ...")
    w4_tooth, w4_case = load_week4_confidence()
    print(f"      {len(w4_tooth)} tooth-level detections, {len(w4_case)} cases")

    # ── 3) Merge on (case, tooth_id) + case-level fallback ───────────────
    df["case"]     = df["case"].astype(int)
    df["tooth_id"] = pd.to_numeric(df["tooth_id"], errors="coerce")
    if not w4_tooth.empty:
        w4_tooth["case"]     = w4_tooth["case"].astype(int)
        w4_tooth["tooth_id"] = pd.to_numeric(w4_tooth["tooth_id"], errors="coerce")
        # Per-tooth join
        df = df.merge(
            w4_tooth[["case", "tooth_id", "caries_confidence", "iou_with_tooth"]],
            on=["case", "tooth_id"], how="left"
        )
        # Case-level max confidence (fallback for FP rows without tooth match)
        df = df.merge(w4_case, on="case", how="left")
    else:
        df["caries_confidence"]   = 0.0
        df["iou_with_tooth"]      = 0.0
        df["case_max_confidence"] = 0.0

    # ── 4) Clean output folder ───────────────────────────────────────────
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for cat_dir in CATEGORIES.values():
        cat_dir.mkdir(parents=True, exist_ok=True)

    # ── 5) Run selectors ─────────────────────────────────────────────────
    print("\n── Category 1: The Perfect Match ──")
    r1 = select_perfect_match(df, top_n=20)
    print(f"   Copied {len(r1)} images")
    for r in r1[:5]:
        print(f"     case {r['case']:>3}  {r['label']:<6}  {r['metric']}")

    print("\n── Category 2: The Complex Win ──")
    r2 = select_complex_win(df)
    print(f"   Copied {len(r2)} images")
    detail_counts = {}
    for r in r2:
        detail_counts[r["label"]] = detail_counts.get(r["label"], 0) + 1
    for lbl, cnt in sorted(detail_counts.items()):
        print(f"     {lbl}: {cnt} cases")

    print("\n── Category 3: The AI Eye (Potential Second Opinions) ──")
    r3 = select_ai_eye(df, top_n=20)
    print(f"   Copied {len(r3)} images")
    for r in r3[:5]:
        print(f"     case {r['case']:>3}  conf={r['confidence']:.3f}  {r['label']}")

    print("\n── Category 4: The Honest Mistake ──")
    r4 = select_honest_mistake(df, n_fp=10, n_fn=10)
    n_fp = sum(1 for r in r4 if r.get("type") == "FP")
    n_fn = sum(1 for r in r4 if r.get("type") == "FN")
    print(f"   Copied {len(r4)} images  (FP={n_fp}, FN={n_fn})")
    for r in r4[:5]:
        kind = r.get("type", "?")
        label = r.get("label", r.get("gt_label", ""))
        print(f"     case {r['case']:>3}  [{kind}]  {label}")

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(r1) + len(r2) + len(r3) + len(r4)
    print("\n" + "=" * 65)
    print(f"  TOTAL: {total} hero-shot images copied")
    print(f"  Output → {OUTPUT_DIR}")
    print("=" * 65)

    # ── Save manifest JSON ───────────────────────────────────────────────
    manifest = {
        "1_The_Perfect_Match":    r1,
        "2_The_Complex_Win":      r2,
        "3_The_AI_Eye_Potential":  r3,
        "4_The_Honest_Mistake":   r4,
        "total_images":           total,
    }
    manifest_path = OUTPUT_DIR / "hero_shots_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
