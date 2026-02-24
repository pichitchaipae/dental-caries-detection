# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Week 9 — F1-Score Dashboard Visualization
============================================

Reads ``per_class_dashboard.csv`` (aggregated from all PCA methods) and
produces an academic-quality grouped bar chart comparing F1-Scores across
surface classes and PCA methods.

Usage
-----
    cd week9_pca_evaluation
    python plot_dashboard.py                          # default behaviour
    python plot_dashboard.py --metric precision       # switch to precision
    python plot_dashboard.py --no-show                # save only, skip plt.show()
    python plot_dashboard.py --csv custom_dash.csv    # alternate CSV

Output
------
    week9_pca_evaluation/f1_score_comparison.png

Author: Expert Data Scientist — Dental AI / CAD
Date:   2026-02-23
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEEK9_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = WEEK9_DIR / "per_class_dashboard.csv"
DEFAULT_OUT = WEEK9_DIR / "f1_score_comparison.png"

# ---------------------------------------------------------------------------
# PCA method display labels  (nicer than raw column values)
# ---------------------------------------------------------------------------
METHOD_LABELS = {
    0: "M0 — Baseline\n(OpenCV)",
    1: "M1 — Square\nHeuristic",
    2: "M2 — Max\nSpan",
    3: "M3 — Split\nCentroid",
    5: "M5 — Vertical\nPrior",
}

# Colour palette — one colour per method (colour-blind friendly)
METHOD_PALETTE = {
    0: "#4C72B0",   # steel blue
    1: "#DD8452",   # warm orange
    2: "#55A868",   # sage green
    3: "#C44E52",   # muted red
    5: "#8172B3",   # soft purple
}

SURFACE_ORDER = ["Distal", "Mesial", "Occlusal", "Macro Avg"]


# =============================================================================
# Data loading & macro average computation
# =============================================================================

def load_dashboard(csv_path: Path) -> pd.DataFrame:
    """
    Load the per-class dashboard CSV and append a *Macro Avg* row per method.
    """
    df = pd.read_csv(csv_path)

    # Compute macro average per method
    macro_rows = []
    for method, grp in df.groupby("pca_method"):
        method_name = grp["pca_method_name"].iloc[0]
        macro_rows.append({
            "pca_method": method,
            "pca_method_name": method_name,
            "class": "Macro Avg",
            "precision": grp["precision"].mean(),
            "recall":    grp["recall"].mean(),
            "f1":        grp["f1"].mean(),
            "support":   grp["support"].sum(),
            "TP":        grp["TP"].sum(),
            "FP":        grp["FP"].sum(),
            "FN":        grp["FN"].sum(),
        })

    df_macro = pd.DataFrame(macro_rows)
    df_all = pd.concat([df, df_macro], ignore_index=True)

    # Friendly method label column
    df_all["Method"] = df_all["pca_method"].map(METHOD_LABELS)

    return df_all


# =============================================================================
# Plotting
# =============================================================================

def plot_grouped_bars(
    df: pd.DataFrame,
    metric: str = "f1",
    out_path: Path = DEFAULT_OUT,
    show: bool = True,
    dpi: int = 200,
):
    """
    Draw a grouped bar chart: X = surface class, Y = *metric*, hue = PCA method.
    """
    metric_label = {
        "f1": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
    }.get(metric, metric.capitalize())

    # Enforce ordering
    df["class"] = pd.Categorical(df["class"], categories=SURFACE_ORDER, ordered=True)
    df = df.sort_values(["class", "pca_method"])

    # ── Seaborn theme ────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    methods_present = sorted(df["pca_method"].unique())
    palette = [METHOD_PALETTE.get(m, "#999999") for m in methods_present]

    bars = sns.barplot(
        data=df,
        x="class",
        y=metric,
        hue="Method",
        order=SURFACE_ORDER,
        hue_order=[METHOD_LABELS[m] for m in methods_present],
        palette=palette,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    # ── Value annotations on each bar ────────────────────────────────
    for container in bars.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#333333",
                )

    # ── Axis styling ─────────────────────────────────────────────────
    ax.set_xlabel("Surface Class", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel(metric_label, fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Week 9 — {metric_label} Comparison Across PCA Methods\n"
        f"(500 Cases × 5 Methods — 3-Class Surface Classification)",
        fontsize=13, fontweight="bold", pad=12,
    )

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.5)
    ax.grid(axis="y", which="minor", linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # ── Add a vertical dashed separator before "Macro Avg" ───────────
    # Macro Avg is the 4th category (index 3); separator at x=2.5
    ax.axvline(x=2.5, color="grey", linestyle="--", linewidth=0.9, alpha=0.5)

    # ── Legend ────────────────────────────────────────────────────────
    legend = ax.legend(
        title="PCA Method",
        title_fontsize=9,
        fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    legend.get_frame().set_linewidth(0.5)

    # ── Tight layout + save ──────────────────────────────────────────
    fig.tight_layout()

    # Update filename if metric is not f1
    if metric != "f1":
        out_path = out_path.parent / f"{metric}_score_comparison.png"

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  [Saved] {metric_label} chart -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Week 9 — Plot per-class F1-Score dashboard from aggregated CSV."
    )
    parser.add_argument(
        "--csv", type=str, default=str(DEFAULT_CSV),
        help=f"Path to per_class_dashboard.csv (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--metric", type=str, default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric to plot (default: f1).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure without calling plt.show().",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output DPI (default: 200).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"  [ERROR] CSV not found: {csv_path}")
        print("  Run the full evaluation first: python run_evaluation.py")
        return

    df = load_dashboard(csv_path)
    plot_grouped_bars(
        df,
        metric=args.metric,
        show=not args.no_show,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
