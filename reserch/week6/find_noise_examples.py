"""
find_noise_examples.py  –  Noise Removal Hunter

Scans all case diagnosis JSONs, re-runs connected-component noise removal
on each tooth's caries coordinates, and ranks teeth by the number of
noisy pixels removed.  Generates Before/After debug images for the Top 10.

Usage:
    python find_noise_examples.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\jaopi\Desktop\SP")
EVAL_OUT      = BASE_DIR / "week6" / "evaluation_output"
MIN_CLUSTER   = 15        # same as multi_zone_classifier.py


# ── Noise Removal (mirror of multi_zone_classifier._remove_small_clusters) ──
def remove_small_clusters(caries_pts: np.ndarray, min_cluster: int = MIN_CLUSTER):
    """Return (cleaned_pts, raw_pts, n_raw, n_cleaned, n_removed)."""
    raw_pts = caries_pts.copy()
    n_raw = len(raw_pts)

    if n_raw < min_cluster:
        return raw_pts, raw_pts, n_raw, n_raw, 0

    pts = caries_pts.astype(np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = 2
    w = x_max - x_min + 1 + 2 * pad
    h = y_max - y_min + 1 + 2 * pad
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min - pad, y_min - pad])
    mask[shifted[:, 1], shifted[:, 0]] = 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    keep = np.zeros_like(mask, dtype=np.uint8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
            keep[labels == lbl] = 255

    kept_ys, kept_xs = np.where(keep > 0)
    if len(kept_xs) == 0:
        return raw_pts, raw_pts, n_raw, n_raw, 0

    cleaned_pts = np.column_stack([
        kept_xs + x_min - pad,
        kept_ys + y_min - pad,
    ]).astype(np.float64)

    n_cleaned = len(cleaned_pts)
    n_removed = n_raw - n_cleaned
    return cleaned_pts, raw_pts, n_raw, n_cleaned, n_removed


# ── Debug image generator ────────────────────────────────────────────────────
def save_debug_image(raw_pts, cleaned_pts, tooth_id, case_num, save_dir):
    """Save a Before/After noise-removal comparison image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _to_mask(pts, x_min, y_min, w, h):
        m = np.zeros((h, w), dtype=np.uint8)
        p = pts.astype(np.int32)
        xs = np.clip(p[:, 0] - x_min, 0, w - 1)
        ys = np.clip(p[:, 1] - y_min, 0, h - 1)
        m[ys, xs] = 255
        return m

    all_pts = np.vstack([raw_pts, cleaned_pts]).astype(np.int32)
    x_min, y_min = all_pts.min(axis=0) - 5
    x_max, y_max = all_pts.max(axis=0) + 5
    w = max(x_max - x_min + 1, 1)
    h = max(y_max - y_min + 1, 1)

    raw_mask     = _to_mask(raw_pts, x_min, y_min, w, h)
    cleaned_mask = _to_mask(cleaned_pts, x_min, y_min, w, h)

    n_raw     = len(raw_pts)
    n_cleaned = len(cleaned_pts)
    n_removed = n_raw - n_cleaned

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_mask, cmap="Reds", interpolation="nearest")
    axes[0].set_title(f"BEFORE  ({n_raw} pts)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cleaned_mask, cmap="Reds", interpolation="nearest")
    axes[1].set_title(f"AFTER  ({n_cleaned} pts, −{n_removed})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    ratio = n_removed / n_raw * 100 if n_raw else 0
    fig.suptitle(
        f"Case {case_num} · Tooth {tooth_id} — Noise Removal  "
        f"({n_removed} px removed, {ratio:.1f}%)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"noise_debug_tooth_{tooth_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main scan ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Noise Removal Hunter — Finding Best Before/After Examples")
    print("=" * 70)

    candidates = []

    for case_num in range(1, 501):
        case_dir = EVAL_OUT / f"case {case_num}"
        json_path = case_dir / f"case_{case_num}_diagnosis.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            data = json.load(f)

        for tooth in data.get("teeth_data", []):
            coords = tooth.get("caries_coordinates")
            if not coords or len(coords) < 2:
                continue

            tooth_id = str(tooth.get("tooth_id", "?"))
            pts = np.array(coords, dtype=np.float64)

            cleaned, raw, n_raw, n_cleaned, n_removed = remove_small_clusters(pts)

            if n_removed == 0:
                continue                       # no noise to show

            ratio = n_removed / n_raw

            # Filter: 5% < ratio < 50%  (meaningful noise, not total wipeout)
            if ratio <= 0.05 or ratio >= 0.50:
                continue

            # Also skip if cleaned == 0 (total removal)
            if n_cleaned == 0:
                continue

            debug_img = case_dir / f"noise_debug_tooth_{tooth_id}.png"

            candidates.append({
                "case":        case_num,
                "tooth_id":    tooth_id,
                "n_raw":       n_raw,
                "n_cleaned":   n_cleaned,
                "n_removed":   n_removed,
                "ratio":       ratio,
                "case_dir":    str(case_dir),
                "debug_img":   str(debug_img),
                "raw_pts":     raw,
                "cleaned_pts": cleaned,
            })

    # ── Deduplicate (same case+tooth can appear if listed twice) ────────
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["case"], c["tooth_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    candidates = deduped

    # ── Sort by n_removed descending ─────────────────────────────────────
    candidates.sort(key=lambda x: x["n_removed"], reverse=True)

    top_n = 10
    top = candidates[:top_n]

    # ── Print results ────────────────────────────────────────────────────
    print(f"\nScanned 500 cases → {len(candidates)} teeth with meaningful noise removal")
    print(f"Filter: 5% < Noise Ratio < 50%  &  n_cleaned > 0\n")

    if not top:
        print("No candidates found.")
        return

    print(f"{'Rank':<5} {'Case':>5} {'Tooth':>6} {'Raw':>6} {'Clean':>6} "
          f"{'Removed':>8} {'Ratio':>7}  Debug Image Path")
    print("-" * 100)

    for i, c in enumerate(top, 1):
        # Generate debug image for top candidates
        img_path = save_debug_image(
            c["raw_pts"], c["cleaned_pts"],
            c["tooth_id"], c["case"],
            Path(c["case_dir"]),
        )
        print(f"{i:<5} {c['case']:>5} {c['tooth_id']:>6} {c['n_raw']:>6} "
              f"{c['n_cleaned']:>6} {c['n_removed']:>8} "
              f"{c['ratio']:>6.1%}  {img_path}")

    # ── Summary visualization ────────────────────────────────────────────
    print(f"\n  Generating summary visualization ...")
    viz_path = generate_summary_viz(top, candidates)
    print(f"\n{'=' * 70}")
    print(f"  Generated {len(top)} noise-debug images")
    print(f"  Summary viz → {viz_path}")
    print(f"  Top removal: Case {top[0]['case']} Tooth {top[0]['tooth_id']} "
          f"→ {top[0]['n_removed']} px removed ({top[0]['ratio']:.1%})")
    print(f"{'=' * 70}")


# ── Summary Visualization ────────────────────────────────────────────────────
def generate_summary_viz(top: list[dict], all_candidates: list[dict]) -> Path:
    """Create a single presentation-ready figure with:
    - Top row: bar chart of removed pixels + ratio distribution
    - Bottom rows: Before/After grids for Top 5 cases
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch

    top5 = top[:5]

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor("#0D1117")

    # Master grid: 2 rows — charts on top, before/after grid on bottom
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.25,
                              top=0.92, bottom=0.03, left=0.06, right=0.97)

    # ─── Title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Noise Removal Analysis — Connected-Component Filtering (min 15 px)",
        fontsize=20, fontweight="bold", color="white", y=0.97,
    )
    fig.text(0.5, 0.935,
             f"{len(all_candidates)} teeth with meaningful noise (5%–50% ratio) across 500 cases",
             ha="center", fontsize=12, color="#8B949E")

    # ═══════════════════════════════════════════════════════════════════════
    # TOP ROW: Two charts side by side
    # ═══════════════════════════════════════════════════════════════════════
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                              wspace=0.30)

    # ── Chart 1: Horizontal bar — Top 10 by pixels removed ──────────────
    ax_bar = fig.add_subplot(top_gs[0])
    ax_bar.set_facecolor("#161B22")

    labels_bar = [f"Case {c['case']}\nTooth {c['tooth_id']}" for c in top[:10]][::-1]
    removed    = [c["n_removed"] for c in top[:10]][::-1]
    ratios     = [c["ratio"] for c in top[:10]][::-1]

    colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(removed)))[::-1]
    bars = ax_bar.barh(range(len(removed)), removed, color=colors, edgecolor="white",
                       linewidth=0.5, height=0.7)

    for i, (bar, rm, rt) in enumerate(zip(bars, removed, ratios)):
        ax_bar.text(bar.get_width() + max(removed) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{rm} px  ({rt:.0%})", va="center", fontsize=9,
                    color="white", fontweight="bold")

    ax_bar.set_yticks(range(len(labels_bar)))
    ax_bar.set_yticklabels(labels_bar, fontsize=9, color="white")
    ax_bar.set_xlabel("Pixels Removed", fontsize=11, color="white")
    ax_bar.set_title("Top 10 — Pixels Removed", fontsize=13,
                     fontweight="bold", color="white", pad=10)
    ax_bar.tick_params(colors="white")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for sp in ax_bar.spines.values():
        sp.set_color("#30363D")

    # ── Chart 2: Histogram — Noise Ratio Distribution ────────────────────
    ax_hist = fig.add_subplot(top_gs[1])
    ax_hist.set_facecolor("#161B22")

    all_ratios = [c["ratio"] * 100 for c in all_candidates]
    bins = np.arange(5, 52, 3)
    ax_hist.hist(all_ratios, bins=bins, color="#F85149", edgecolor="white",
                 linewidth=0.5, alpha=0.85)
    ax_hist.axvline(np.median(all_ratios), color="#58A6FF", linestyle="--",
                    linewidth=2, label=f"Median = {np.median(all_ratios):.1f}%")
    ax_hist.legend(fontsize=10, facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="white")

    ax_hist.set_xlabel("Noise Ratio (%)", fontsize=11, color="white")
    ax_hist.set_ylabel("Count (teeth)", fontsize=11, color="white")
    ax_hist.set_title("Noise Ratio Distribution (5%–50%)", fontsize=13,
                      fontweight="bold", color="white", pad=10)
    ax_hist.tick_params(colors="white")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    for sp in ax_hist.spines.values():
        sp.set_color("#30363D")

    # ═══════════════════════════════════════════════════════════════════════
    # BOTTOM ROWS: Before / After grids for Top 5
    # ═══════════════════════════════════════════════════════════════════════
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        len(top5), 3, subplot_spec=outer[1],
        width_ratios=[1, 1, 0.08], wspace=0.08, hspace=0.35,
    )

    def _to_mask(pts, x_min, y_min, w, h):
        m = np.zeros((h, w, 3), dtype=np.uint8)
        p = pts.astype(np.int32)
        xs = np.clip(p[:, 0] - x_min, 0, w - 1)
        ys = np.clip(p[:, 1] - y_min, 0, h - 1)
        return m, xs, ys

    def _make_diff_image(raw_pts, cleaned_pts):
        """RGB image: Kept=green, Removed=red."""
        all_pts = np.vstack([raw_pts, cleaned_pts]).astype(np.int32)
        x_min, y_min = all_pts.min(axis=0) - 3
        x_max, y_max = all_pts.max(axis=0) + 3
        w = max(x_max - x_min + 1, 1)
        h = max(y_max - y_min + 1, 1)

        img_before = np.full((h, w, 3), 22, dtype=np.uint8)  # dark bg
        img_after  = np.full((h, w, 3), 22, dtype=np.uint8)

        # Before: all points in warm orange
        rp = raw_pts.astype(np.int32)
        rxs = np.clip(rp[:, 0] - x_min, 0, w - 1)
        rys = np.clip(rp[:, 1] - y_min, 0, h - 1)
        img_before[rys, rxs] = [248, 81, 73]  # red

        # After: kept = green, show removed as dim red ghost
        cp = cleaned_pts.astype(np.int32)
        cxs = np.clip(cp[:, 0] - x_min, 0, w - 1)
        cys = np.clip(cp[:, 1] - y_min, 0, h - 1)

        # Ghost removed first (dim)
        img_after[rys, rxs] = [80, 30, 30]
        # Kept on top (bright green)
        img_after[cys, cxs] = [63, 185, 80]

        return img_before, img_after

    for row_i, c in enumerate(top5):
        img_before, img_after = _make_diff_image(c["raw_pts"], c["cleaned_pts"])

        # BEFORE
        ax_b = fig.add_subplot(bottom_gs[row_i, 0])
        ax_b.imshow(img_before, interpolation="nearest", aspect="equal")
        ax_b.set_title(
            f"BEFORE — {c['n_raw']} pts",
            fontsize=10, fontweight="bold", color="#F85149", pad=4,
        )
        ax_b.axis("off")
        # Row label on left
        ax_b.text(-0.02, 0.5,
                  f"#{row_i+1}  Case {c['case']}\nTooth {c['tooth_id']}",
                  transform=ax_b.transAxes, ha="right", va="center",
                  fontsize=10, fontweight="bold", color="white")

        # AFTER
        ax_a = fig.add_subplot(bottom_gs[row_i, 1])
        ax_a.imshow(img_after, interpolation="nearest", aspect="equal")
        ratio_pct = c["ratio"] * 100
        ax_a.set_title(
            f"AFTER — {c['n_cleaned']} pts  (−{c['n_removed']} removed, {ratio_pct:.1f}%)",
            fontsize=10, fontweight="bold", color="#3FB950", pad=4,
        )
        ax_a.axis("off")

    # Legend text
    fig.text(0.50, 0.01,
             "■ Red = raw caries pixels     ■ Green = kept after cleaning     "
             "■ Dim red = removed noise",
             ha="center", fontsize=10, color="#8B949E",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22",
                       edgecolor="#30363D"))

    out_path = EVAL_OUT / "noise_removal_summary.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] {out_path}")
    return out_path


if __name__ == "__main__":
    main()
