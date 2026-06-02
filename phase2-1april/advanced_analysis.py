#!/usr/bin/env python3

"""
advanced_analysis.py
====================
Advanced analysis pipeline for the dental caries surface classification model.

Tasks:
    1. Model Interpretability (Feature Importance + SHAP)
    2. Deep Error Analysis (misclassified case visualization)
    3. Confidence & Probability Analysis
    4. Robustness Check (5-Fold Cross Validation)
    5. Deployment / Inference Speed Benchmark

Usage:
    python advanced_analysis.py              # Run all tasks
    python advanced_analysis.py --task 1     # Run only task 1
    python advanced_analysis.py --task 1 2   # Run tasks 1 and 2
"""

# =========================================================
# Imports
# =========================================================
import os
import sys
import time
import math
import json
import warnings
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve

# Import from existing pipeline
from pipeline_run3_final import (
    FEATURE_COLS, VALID_SURFACES, MODEL_PATH,
    SEG_DIR, CARIES_DIR, GT_ROOT,
    _load_seg_case, _load_caries_case,
    build_seg_map, parse_case_ground_truth,
    _extract_ml_feature_dict, remove_small_clusters,
    perform_pca, rotate, get_bbox,
    _progress_bar,
    LEFT_BOUND, RIGHT_BOUND, get_quadrant,
    create_ml_dataset,
)


# =========================================================
# Configuration
# =========================================================
try:
    _THIS_DIR = Path(__file__).resolve().parent
except NameError:
    _THIS_DIR = Path.cwd()

OUTPUT_DIR = _THIS_DIR / "Analysis_Output"
ERROR_DIR = OUTPUT_DIR / "Error_Analysis"
MAX_ERROR_PLOTS_PER_PAIR = 20

# Plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def setup_output_dirs():
    """Create all required output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Data Loading Helpers
# =========================================================

def load_dataset_and_model():
    """
    Load the trained RF model and build the full feature dataset.

    Reproduces the exact same GroupShuffleSplit(test_size=0.2, random_state=42)
    train/test split that was used during training.

    Returns:
        tuple: (model, feature_df, train_df, test_df).
    """
    print("\n[LOADING] กำลังโหลดข้อมูลและโมเดล...", flush=True)

    # Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")
    model = joblib.load(str(MODEL_PATH))
    print(f"  ✓ โหลดโมเดลจาก {MODEL_PATH}")

    # Build full dataset (500 cases)
    case_ids = list(range(1, 501))
    feature_df = create_ml_dataset(case_ids)
    print(f"  ✓ สกัด features เสร็จ: {len(feature_df)} ซี่")

    # Reproduce the exact same train/test split
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(feature_df, groups=feature_df["case_id"]))
    train_df = feature_df.iloc[train_idx].reset_index(drop=True)
    test_df = feature_df.iloc[test_idx].reset_index(drop=True)

    print(f"  ✓ Train: {len(train_df)} ซี่ | Test: {len(test_df)} ซี่")
    return model, feature_df, train_df, test_df


def get_test_predictions(model, test_df):
    """Get predictions and probabilities for the test set."""
    X_test = test_df[FEATURE_COLS]
    y_true = test_df["label"].values
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return y_true, y_pred, y_proba


# =============================================================================
# ██████╗  TASK 1 — Model Interpretability
# =============================================================================

def task1_model_interpretability(model, train_df, test_df):
    """Task 1: Global Feature Importance + SHAP Analysis."""
    print("\n" + "=" * 60)
    print("  TASK 1: Model Interpretability")
    print("=" * 60)

    task1a_feature_importance(model, test_df)
    task1b_shap_analysis(model, test_df)


# ---- 1A: Gini vs Permutation Feature Importance ----

def task1a_feature_importance(model, test_df):
    """Compare Gini vs Permutation Feature Importance."""
    print("\n[1A] Computing Feature Importance (Gini + Permutation)...", flush=True)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]

    gini_imp = model.feature_importances_

    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=42, n_jobs=-1,
    )
    perm_imp = perm_result.importances_mean
    perm_std = perm_result.importances_std

    imp_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Gini_Importance": gini_imp,
        "Permutation_Importance": perm_imp,
        "Permutation_Std": perm_std,
    }).sort_values("Gini_Importance", ascending=False).reset_index(drop=True)

    # Console table
    print("\n" + "=" * 70)
    print(f"  {'Feature':<20s}  {'Gini':>10s}  {'Permutation':>14s}  {'± Std':>8s}")
    print("=" * 70)
    for _, row in imp_df.iterrows():
        print(f"  {row['Feature']:<20s}  {row['Gini_Importance']:>10.4f}  "
              f"{row['Permutation_Importance']:>14.4f}  {row['Permutation_Std']:>8.4f}")
    print("=" * 70)

    # --- Side-by-side horizontal bar chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    n = len(FEATURE_COLS)
    gini_sorted = imp_df.sort_values("Gini_Importance", ascending=True)
    perm_sorted = imp_df.sort_values("Permutation_Importance", ascending=True)

    colors_g = plt.cm.Blues(np.linspace(0.25, 0.90, n))
    colors_p = plt.cm.Oranges(np.linspace(0.25, 0.90, n))

    # Gini
    ax1.barh(range(n), gini_sorted["Gini_Importance"],
             color=colors_g, edgecolor="white", height=0.7)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(gini_sorted["Feature"], fontsize=9)
    ax1.set_xlabel("Importance")
    ax1.set_title("Gini Importance (MDI)", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for i, v in enumerate(gini_sorted["Gini_Importance"]):
        ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8, color="#333")

    # Permutation
    ax2.barh(range(n), perm_sorted["Permutation_Importance"],
             xerr=perm_sorted["Permutation_Std"],
             color=colors_p, edgecolor="white", height=0.7, capsize=3)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(perm_sorted["Feature"], fontsize=9)
    ax2.set_xlabel("Importance")
    ax2.set_title("Permutation Importance", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for i, v in enumerate(perm_sorted["Permutation_Importance"]):
        ax2.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8, color="#333")

    fig.suptitle("Feature Importance Comparison — Random Forest",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "feature_importance_comparison.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {save_path}")

    csv_path = OUTPUT_DIR / "feature_importance_comparison.csv"
    imp_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")


# ---- 1B: SHAP Values ----

def task1b_shap_analysis(model, test_df):
    """SHAP Values Analysis: Summary, Waterfall, and Dependence plots."""
    print("\n[1B] Computing SHAP Values...", flush=True)

    try:
        import shap
    except ImportError:
        print("[WARNING] shap ไม่ได้ติดตั้ง — ข้าม SHAP analysis")
        print("  ติดตั้งด้วย: pip install shap")
        return

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"].values

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)
    class_names = list(model.classes_)

    # Normalise shape: SHAP ≥0.50 may return a 3D array (n_samples, n_features, n_classes)
    # instead of a list of 2D arrays.  Convert to list-of-arrays for uniform access.
    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        shap_values = shap_values_raw

    # expected_value may also be an array — ensure it is indexable per class.
    base_values = explainer.expected_value
    if not hasattr(base_values, "__len__"):
        base_values = [base_values] * len(class_names)

    # ---- Global Summary Plot ----
    print("  Creating SHAP Summary Plot (Global)...", flush=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=FEATURE_COLS,
        class_names=class_names,
        show=False,
        plot_size=(12, 8),
    )
    plt.title("SHAP Summary Plot — All Classes",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "shap_summary_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  [SAVED] {save_path}")

    # ---- Per-class Summary Plot ----
    for cls_idx, cls_name in enumerate(class_names):
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values[cls_idx], X_test,
            feature_names=FEATURE_COLS,
            show=False,
            plot_size=(10, 7),
        )
        plt.title(f"SHAP Summary — {cls_name}",
                  fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"shap_summary_{cls_name.lower()}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"  [SAVED] {save_path}")

    # ---- Waterfall Plots (Local Explanation — 5 diverse samples) ----
    print("  Creating SHAP Waterfall Plots (5 samples)...", flush=True)
    y_pred = model.predict(X_test)

    samples = []
    for cls_name in class_names:
        if cls_name == "Other":
            continue
        # One correct prediction
        correct_mask = (y_test == cls_name) & (y_pred == cls_name)
        if np.sum(correct_mask) > 0:
            samples.append((np.where(correct_mask)[0][0], "correct"))
        # One incorrect prediction
        incorrect_mask = (y_test == cls_name) & (y_pred != cls_name)
        if np.sum(incorrect_mask) > 0:
            samples.append((np.where(incorrect_mask)[0][0], "incorrect"))

    for sample_num, (sample_idx, status) in enumerate(samples[:5]):
        true_label = y_test[sample_idx]
        pred_label = y_pred[sample_idx]
        pred_class_idx = list(model.classes_).index(pred_label)

        case_id = test_df.iloc[sample_idx]["case_id"]
        tooth_id = test_df.iloc[sample_idx]["tooth_id"]

        plt.figure(figsize=(10, 6))
        shap_explanation = shap.Explanation(
            values=shap_values[pred_class_idx][sample_idx],
            base_values=base_values[pred_class_idx],
            data=X_test.iloc[sample_idx].values,
            feature_names=FEATURE_COLS,
        )
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title(
            f"SHAP Waterfall — Case {int(case_id)}, Tooth {tooth_id}\n"
            f"GT: {true_label} → Pred: {pred_label} ({status.upper()})",
            fontsize=12, fontweight="bold", pad=10,
        )
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"shap_waterfall_sample_{sample_num+1}_{status}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"  [SAVED] {save_path}")

    # ---- Dependence Plots (Top 3 features) ----
    print("  Creating SHAP Dependence Plots (Top 3 features)...", flush=True)

    mean_abs_shap = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
    )
    top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat_idx in enumerate(top3_idx):
        shap_for_feat = [shap_values[c][:, feat_idx] for c in range(len(class_names))]
        best_class = np.argmax([np.std(s) for s in shap_for_feat])

        ax = axes[i]
        scatter = ax.scatter(
            X_test.iloc[:, feat_idx],
            shap_values[best_class][:, feat_idx],
            c=X_test.iloc[:, feat_idx],
            cmap="coolwarm", alpha=0.6, s=20, edgecolors="none",
        )
        ax.set_xlabel(FEATURE_COLS[feat_idx], fontweight="bold")
        ax.set_ylabel(f"SHAP value ({class_names[best_class]})")
        ax.set_title(f"#{i+1}: {FEATURE_COLS[feat_idx]}", fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("SHAP Dependence Plots — Top 3 Features",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "shap_dependence_top3.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# =============================================================================
# ██████╗  TASK 2 — Deep Error Analysis
# =============================================================================

def task2_deep_error_analysis(model, test_df):
    """Task 2: Analyze and visualize misclassified cases."""
    print("\n" + "=" * 60)
    print("  TASK 2: Deep Error Analysis")
    print("=" * 60)

    X_test = test_df[FEATURE_COLS]
    y_true = test_df["label"].values
    y_pred = model.predict(X_test)

    # Identify errors
    misclassified_mask = y_true != y_pred
    n_errors = np.sum(misclassified_mask)
    n_total = len(y_true)
    print(f"\n  Total test samples : {n_total}")
    print(f"  Misclassified      : {n_errors} ({n_errors/n_total*100:.1f}%)")

    # Build error DataFrame
    error_df = test_df[misclassified_mask].copy()
    error_df["predicted"] = y_pred[misclassified_mask]
    error_df["true_label"] = y_true[misclassified_mask]

    y_proba = model.predict_proba(X_test)
    error_proba = y_proba[misclassified_mask]
    error_df["max_probability"] = np.max(error_proba, axis=1)

    # Save error summary CSV
    csv_cols = ["case_id", "tooth_id", "true_label", "predicted",
                "max_probability"] + FEATURE_COLS
    error_csv_path = OUTPUT_DIR / "error_summary.csv"
    error_df[csv_cols].to_csv(error_csv_path, index=False)
    print(f"  [SAVED] {error_csv_path}")

    # Confusion pair counts
    confusion_pairs = (
        error_df.groupby(["true_label", "predicted"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\n  {'True Label':<12s}  {'Predicted':<12s}  {'Count':>6s}")
    print("  " + "-" * 35)
    for _, row in confusion_pairs.iterrows():
        print(f"  {row['true_label']:<12s}  {row['predicted']:<12s}  {row['count']:>6d}")

    # ---- Generate error case plots (Top-20 per confusion pair) ----
    print(f"\n  Generating error visualizations "
          f"(max {MAX_ERROR_PLOTS_PER_PAIR} per pair)...", flush=True)

    total_plots = 0
    for _, pair_row in confusion_pairs.iterrows():
        true_label = pair_row["true_label"]
        pred_label = pair_row["predicted"]
        pair_mask = ((error_df["true_label"] == true_label)
                     & (error_df["predicted"] == pred_label))
        pair_cases = error_df[pair_mask].copy()

        # Sort by probability ascending → most confused first
        pair_cases = pair_cases.sort_values("max_probability", ascending=True)
        pair_cases = pair_cases.head(MAX_ERROR_PLOTS_PER_PAIR)

        folder_name = f"{true_label}_Predicted_As_{pred_label}"
        pair_dir = ERROR_DIR / folder_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        for plot_num, (_, case_row) in enumerate(pair_cases.iterrows(), start=1):
            _plot_single_error(
                case_id=int(case_row["case_id"]),
                tooth_id=str(case_row["tooth_id"]),
                true_label=true_label,
                pred_label=pred_label,
                max_prob=case_row["max_probability"],
                save_dir=pair_dir,
                plot_num=plot_num,
            )
            total_plots += 1

        print(f"    ✓ {folder_name}: {len(pair_cases)} plots")

    print(f"\n  [DONE] Total error plots generated: {total_plots}")

    # Summary bar chart
    _plot_error_summary(confusion_pairs)


def _plot_single_error(case_id, tooth_id, true_label, pred_label,
                       max_prob, save_dir, plot_num):
    """
    Plot a single misclassified case showing tooth outline, caries region,
    PCA-aligned bounding box, and the three surface zones.
    """
    try:
        seg_data = _load_seg_case(case_id)
        caries_data = _load_caries_case(case_id)
        if seg_data is None or caries_data is None:
            return

        seg_map = build_seg_map(seg_data)
        tooth_pts = seg_map.get(tooth_id, [])

        # Find caries coordinates for this tooth
        caries_pts = []
        for tooth in caries_data.get("teeth_caries_data", []):
            if str(tooth.get("tooth_id", "")) == tooth_id:
                caries_pts = tooth.get("caries_coordinates", [])
                break

        if len(tooth_pts) < 10 or len(caries_pts) == 0:
            return

        tooth_arr = np.array(tooth_pts, dtype=np.float64)
        caries_arr = np.array(caries_pts, dtype=np.float64)
        caries_clean = remove_small_clusters(caries_arr)

        # PCA alignment
        center, angle, _ = perform_pca(tooth_arr, tooth_id)
        tooth_rot = rotate(tooth_arr, center, angle)
        caries_rot = rotate(caries_clean, center, angle)

        bbox_x, bbox_y, w, h = get_bbox(tooth_rot)
        if w <= 0 or h <= 0:
            return

        # Relative coordinates
        tooth_x_rel = (tooth_rot[:, 0] - bbox_x) / w
        tooth_y_rel = (tooth_rot[:, 1] - bbox_y) / h
        caries_x_rel = (caries_rot[:, 0] - bbox_x) / w
        caries_y_rel = (caries_rot[:, 1] - bbox_y) / h

        # ---- Draw ----
        fig, ax = plt.subplots(figsize=(6, 8))

        ax.scatter(tooth_x_rel, tooth_y_rel, c="#d1d5db",
                   s=0.5, alpha=0.4, label="Tooth", rasterized=True)
        ax.scatter(caries_x_rel, caries_y_rel, c="#ef4444",
                   s=2, alpha=0.7, label="Caries", rasterized=True)

        # Zone separator lines
        ax.axvline(x=LEFT_BOUND, color="#3b82f6", linestyle="--",
                   alpha=0.7, linewidth=1.5)
        ax.axvline(x=RIGHT_BOUND, color="#3b82f6", linestyle="--",
                   alpha=0.7, linewidth=1.5)

        # Zone labels (depends on quadrant)
        quadrant = get_quadrant(tooth_id)
        if quadrant in [1, 4]:
            zone_labels = {"Distal": 0.20, "Occlusal": 0.50, "Mesial": 0.80}
        else:
            zone_labels = {"Mesial": 0.20, "Occlusal": 0.50, "Distal": 0.80}

        for zone_name, x_pos in zone_labels.items():
            color = "#22c55e" if zone_name == true_label else "#94a3b8"
            weight = "bold" if zone_name == true_label else "normal"
            ax.text(x_pos, -0.05, zone_name, ha="center", fontsize=9,
                    fontweight=weight, color=color)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.1)
        ax.invert_yaxis()
        ax.set_xlabel("Relative X (PCA-aligned)", fontsize=10)
        ax.set_ylabel("Relative Y (PCA-aligned)", fontsize=10)
        ax.set_title(
            f"GT: {true_label} → Pred: {pred_label}\n"
            f"Case {case_id}, Tooth {tooth_id} | Conf: {max_prob:.1%}",
            fontsize=11, fontweight="bold", pad=10, color="#dc2626",
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        fname = f"error_{plot_num:02d}_case{case_id}_tooth{tooth_id}.png"
        fig.savefig(save_dir / fname, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    except Exception as e:
        print(f"    [SKIP] Case {case_id}, Tooth {tooth_id}: {e}")


def _plot_error_summary(confusion_pairs):
    """Horizontal bar chart summarising error counts by confusion pair."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{r['true_label']} → {r['predicted']}"
              for _, r in confusion_pairs.iterrows()]
    counts = confusion_pairs["count"].values

    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(labels)))
    bars = ax.barh(range(len(labels)), counts, color=colors,
                   edgecolor="white", height=0.6)

    for i, (bar_obj, cnt) in enumerate(zip(bars, counts)):
        ax.text(bar_obj.get_width() + 0.5, i, str(cnt),
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Misclassified Cases", fontsize=12)
    ax.set_title("Error Pattern Summary — Confusion Pairs",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    save_path = OUTPUT_DIR / "error_pattern_summary.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# =============================================================================
# ██████╗  TASK 3 — Confidence & Probability Analysis
# =============================================================================

def task3_confidence_analysis(model, test_df):
    """Task 3: Analyze prediction confidence and probabilities."""
    print("\n" + "=" * 60)
    print("  TASK 3: Confidence & Probability Analysis")
    print("=" * 60)

    X_test = test_df[FEATURE_COLS]
    y_true = test_df["label"].values
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    max_proba = np.max(y_proba, axis=1)
    correct_mask = y_true == y_pred

    # Console summary
    print(f"\n  Confidence Statistics:")
    print(f"  {'':20s}  {'Correct':>10s}  {'Incorrect':>10s}")
    print(f"  {'Mean':20s}  {np.mean(max_proba[correct_mask]):>10.4f}  "
          f"{np.mean(max_proba[~correct_mask]):>10.4f}")
    print(f"  {'Median':20s}  {np.median(max_proba[correct_mask]):>10.4f}  "
          f"{np.median(max_proba[~correct_mask]):>10.4f}")
    print(f"  {'Min':20s}  {np.min(max_proba[correct_mask]):>10.4f}  "
          f"{np.min(max_proba[~correct_mask]):>10.4f}")
    print(f"  {'Max':20s}  {np.max(max_proba[correct_mask]):>10.4f}  "
          f"{np.max(max_proba[~correct_mask]):>10.4f}")

    task3a_confidence_histogram(max_proba, correct_mask)
    task3b_confidence_boxplot(max_proba, correct_mask, y_true, y_pred)
    task3c_calibration_curve(model, X_test, y_true)
    task3d_uncertainty_threshold(max_proba, correct_mask, y_true, y_pred)


def task3a_confidence_histogram(max_proba, correct_mask):
    """Histogram of max probability for correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 30)
    ax.hist(max_proba[correct_mask], bins=bins, alpha=0.7, color="#22c55e",
            label=f"Correct (n={np.sum(correct_mask)})", edgecolor="white")
    ax.hist(max_proba[~correct_mask], bins=bins, alpha=0.7, color="#ef4444",
            label=f"Incorrect (n={np.sum(~correct_mask)})", edgecolor="white")

    ax.axvline(x=np.mean(max_proba[correct_mask]), color="#16a34a",
               linestyle="--", linewidth=2,
               label=f"Mean Correct: {np.mean(max_proba[correct_mask]):.3f}")
    ax.axvline(x=np.mean(max_proba[~correct_mask]), color="#dc2626",
               linestyle="--", linewidth=2,
               label=f"Mean Incorrect: {np.mean(max_proba[~correct_mask]):.3f}")

    ax.set_xlabel("Max Prediction Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Confidence Distribution — Correct vs Incorrect Predictions",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "confidence_distribution.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def task3b_confidence_boxplot(max_proba, correct_mask, y_true, y_pred):
    """Box plot of confidence by class and correctness."""
    data = []
    for i in range(len(y_true)):
        data.append({
            "Max Probability": max_proba[i],
            "True Class": y_true[i],
            "Prediction": "Correct" if correct_mask[i] else "Incorrect",
        })
    plot_df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Overall
    sns.boxplot(data=plot_df, x="Prediction", y="Max Probability", ax=ax1,
                palette={"Correct": "#22c55e", "Incorrect": "#ef4444"},
                width=0.5)
    ax1.set_title("Confidence by Prediction Outcome", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Per class
    surface_order = [s for s in VALID_SURFACES
                     if s in plot_df["True Class"].unique()]
    sns.boxplot(data=plot_df, x="True Class", y="Max Probability",
                hue="Prediction", ax=ax2,
                palette={"Correct": "#22c55e", "Incorrect": "#ef4444"},
                order=surface_order, width=0.6)
    ax2.set_title("Confidence by Class & Outcome", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Prediction Confidence Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "confidence_boxplot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def task3c_calibration_curve(model, X_test, y_true):
    """Calibration curve (reliability diagram) per class."""
    class_names = [c for c in model.classes_ if c != "Other"]
    y_proba = model.predict_proba(X_test)

    n_cls = len(class_names)
    fig, axes = plt.subplots(1, n_cls, figsize=(6 * n_cls, 5))
    if n_cls == 1:
        axes = [axes]

    colors = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"]

    for idx, (cls_name, ax) in enumerate(zip(class_names, axes)):
        cls_idx = list(model.classes_).index(cls_name)
        y_binary = (y_true == cls_name).astype(int)
        prob_cls = y_proba[:, cls_idx]
        color = colors[idx % len(colors)]

        try:
            prob_true, prob_pred = calibration_curve(
                y_binary, prob_cls, n_bins=10, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, marker="o", color=color,
                    linewidth=2, label=cls_name)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5,
                    label="Perfectly Calibrated")
            ax.fill_between(prob_pred, prob_true, prob_pred,
                            alpha=0.15, color=color)
        except Exception as e:
            ax.text(0.5, 0.5, f"Insufficient data\n{e}",
                    transform=ax.transAxes, ha="center", va="center")

        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives", fontsize=11)
        ax.set_title(f"Calibration — {cls_name}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Calibration Curves (Reliability Diagrams)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "calibration_curve.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def task3d_uncertainty_threshold(max_proba, correct_mask, y_true, y_pred):
    """
    Analyze how accuracy / F1 change when uncertain predictions
    (max_prob < threshold) are rejected.
    """
    thresholds = np.arange(0.30, 0.95, 0.05)

    results = []
    for thresh in thresholds:
        confident_mask = max_proba >= thresh
        n_confident = np.sum(confident_mask)
        if n_confident == 0:
            continue

        acc = accuracy_score(y_true[confident_mask], y_pred[confident_mask])
        f1 = f1_score(y_true[confident_mask], y_pred[confident_mask],
                      average="macro", zero_division=0)
        coverage = n_confident / len(max_proba)

        results.append({
            "Threshold": thresh,
            "Accuracy": acc,
            "F1_Score": f1,
            "Coverage": coverage,
            "Rejected": int(np.sum(~confident_mask)),
            "Remaining": int(n_confident),
        })

    thresh_df = pd.DataFrame(results)

    print("\n  Uncertainty Threshold Analysis:")
    print(f"  {'Threshold':>10s}  {'Accuracy':>10s}  {'F1':>8s}  "
          f"{'Coverage':>10s}  {'Rejected':>10s}")
    print("  " + "-" * 55)
    for _, r in thresh_df.iterrows():
        print(f"  {r['Threshold']:>10.2f}  {r['Accuracy']:>10.4f}  "
              f"{r['F1_Score']:>8.4f}  {r['Coverage']:>10.1%}  "
              f"{r['Rejected']:>10.0f}")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(thresh_df["Threshold"], thresh_df["Accuracy"], "o-",
             color="#3b82f6", linewidth=2, markersize=6, label="Accuracy")
    ax1.plot(thresh_df["Threshold"], thresh_df["F1_Score"], "s-",
             color="#f59e0b", linewidth=2, markersize=6, label="F1 Score")
    ax1.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax1.set_title("Performance vs Confidence Threshold", fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.3)

    # Annotate best F1 threshold
    if len(thresh_df) > 0:
        best_idx = thresh_df["F1_Score"].idxmax()
        best_thresh = thresh_df.loc[best_idx, "Threshold"]
        best_f1 = thresh_df.loc[best_idx, "F1_Score"]
        ax1.axvline(x=best_thresh, color="#dc2626", linestyle=":",
                    alpha=0.7, label=f"Best F1 @ {best_thresh:.2f}")
        ax1.legend(fontsize=9)

    ax2.fill_between(thresh_df["Threshold"], thresh_df["Coverage"],
                     alpha=0.3, color="#10b981")
    ax2.plot(thresh_df["Threshold"], thresh_df["Coverage"], "o-",
             color="#10b981", linewidth=2, markersize=6)
    ax2.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Coverage (% cases kept)", fontsize=12, fontweight="bold")
    ax2.set_title("Data Coverage vs Confidence Threshold", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Uncertainty Threshold Analysis — \"When to Trust the Model\"",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = OUTPUT_DIR / "uncertainty_threshold_analysis.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")

    csv_path = OUTPUT_DIR / "uncertainty_threshold_results.csv"
    thresh_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")


# =============================================================================
# ██████╗  TASK 4 — Robustness Check (K-Fold CV)
# =============================================================================

def task4_robustness_check(feature_df):
    """Task 4: 5-Fold GroupKFold Cross Validation."""
    print("\n" + "=" * 60)
    print("  TASK 4: Robustness Check (5-Fold GroupKFold CV)")
    print("=" * 60)

    X = feature_df[FEATURE_COLS]
    y = feature_df["label"]
    groups = feature_df["case_id"]

    gkf = GroupKFold(n_splits=5)

    fold_results = []
    per_class_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_results.append({
            "Fold": fold + 1,
            "Train_Size": len(train_idx),
            "Test_Size": len(test_idx),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
        })

        report = classification_report(
            y_test, y_pred, labels=VALID_SURFACES,
            output_dict=True, zero_division=0,
        )
        for cls in VALID_SURFACES:
            if cls in report:
                per_class_results.append({
                    "Fold": fold + 1,
                    "Class": cls,
                    "Precision": report[cls]["precision"],
                    "Recall": report[cls]["recall"],
                    "F1_Score": report[cls]["f1-score"],
                    "Support": report[cls]["support"],
                })

        print(f"  Fold {fold+1}: Acc={acc:.4f} | "
              f"Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

    fold_df = pd.DataFrame(fold_results)
    per_class_df = pd.DataFrame(per_class_results)

    # Summary
    print("\n" + "=" * 60)
    print("  K-Fold Summary:")
    for metric in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        vals = fold_df[metric].values
        print(f"  {metric:<12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"(range: {np.min(vals):.4f} – {np.max(vals):.4f})")
    print("=" * 60)

    _plot_kfold_results(fold_df)
    _plot_kfold_per_class(per_class_df)

    csv_path = OUTPUT_DIR / "kfold_summary.csv"
    fold_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    csv_path2 = OUTPUT_DIR / "kfold_per_class.csv"
    per_class_df.to_csv(csv_path2, index=False)
    print(f"  [SAVED] {csv_path2}")

    return fold_df


def _plot_kfold_results(fold_df):
    """Bar chart of macro metrics across folds."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    colors = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"]
    x = np.arange(len(fold_df))
    bar_width = 0.18

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = fold_df[metric].values
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=metric.replace("_", " "),
                      color=color, edgecolor="white", linewidth=0.6)
        for bar_obj in bars:
            h = bar_obj.get_height()
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    # Mean reference lines
    for metric, color in zip(metrics, colors):
        ax.axhline(y=fold_df[metric].mean(), color=color,
                   linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([f"Fold {f}" for f in fold_df["Fold"]], fontsize=11)
    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("5-Fold GroupKFold Cross Validation Results",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    mean_f1 = fold_df["F1_Score"].mean()
    std_f1 = fold_df["F1_Score"].std()
    ax.text(0.02, 0.98, f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#dbeafe",
                      edgecolor="#3b82f6", alpha=0.8))

    plt.tight_layout()
    save_path = OUTPUT_DIR / "kfold_results.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def _plot_kfold_per_class(per_class_df):
    """Per-class metric line plots across folds."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ["Precision", "Recall", "F1_Score"]
    colors_map = {
        "Occlusal": "#3b82f6", "Mesial": "#f59e0b",
        "Distal": "#10b981", "Other": "#94a3b8",
    }

    for ax, metric in zip(axes, metrics):
        classes_in_data = [c for c in VALID_SURFACES
                           if c in per_class_df["Class"].unique()]
        for cls in classes_in_data:
            cls_data = per_class_df[per_class_df["Class"] == cls]
            ax.plot(cls_data["Fold"], cls_data[metric], "o-",
                    color=colors_map.get(cls, "#666"), label=cls,
                    linewidth=2, markersize=6)

        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel(metric.replace("_", " "), fontsize=11)
        ax.set_title(f"Per-Class {metric.replace('_', ' ')}",
                     fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-Class Metrics Across 5 Folds",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "kfold_per_class.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# =============================================================================
# ██████╗  TASK 5 — Deployment / Inference Speed
# =============================================================================

def task5_deployment_optimization(model, test_df):
    """Task 5: Inference speed benchmark."""
    print("\n" + "=" * 60)
    print("  TASK 5: Deployment / Inference Speed Benchmark")
    print("=" * 60)

    X_test = test_df[FEATURE_COLS]

    # Benchmark sample
    sample_size = min(100, len(test_df))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(test_df), sample_size, replace=False)

    # ---- 5A: Feature Extraction Speed ----
    print("\n  [5A] Measuring feature extraction speed...", flush=True)

    extraction_times = []
    for idx in sample_indices:
        row = test_df.iloc[idx]
        case_id = int(row["case_id"])
        tooth_id = str(row["tooth_id"])

        seg_data = _load_seg_case(case_id)
        caries_data = _load_caries_case(case_id)
        if seg_data is None or caries_data is None:
            continue

        seg_map = build_seg_map(seg_data)
        tooth_pts = seg_map.get(tooth_id, [])

        caries_pts = []
        for tooth in caries_data.get("teeth_caries_data", []):
            if str(tooth.get("tooth_id", "")) == tooth_id:
                caries_pts = tooth.get("caries_coordinates", [])
                break

        if len(tooth_pts) < 10 or len(caries_pts) == 0:
            continue

        start = time.perf_counter()
        _extract_ml_feature_dict(tooth_id, tooth_pts, caries_pts)
        extraction_times.append((time.perf_counter() - start) * 1000)

    # ---- 5B: Prediction Speed ----
    print("  [5B] Measuring prediction speed...", flush=True)

    prediction_times = []
    for i in range(sample_size):
        idx = sample_indices[i] if i < len(sample_indices) else 0
        row_data = X_test.iloc[[idx]]

        start = time.perf_counter()
        model.predict(row_data)
        model.predict_proba(row_data)
        prediction_times.append((time.perf_counter() - start) * 1000)

    # ---- 5C: End-to-end ----
    total_times = [
        extraction_times[i] + prediction_times[i]
        for i in range(min(len(extraction_times), len(prediction_times)))
    ]

    # ---- Report ----
    report_lines = [
        "=" * 60,
        "  INFERENCE SPEED BENCHMARK REPORT",
        "=" * 60,
        f"  Sample size: {sample_size} teeth",
        "",
    ]

    for name, times in [
        ("Feature Extraction", extraction_times),
        ("Model Prediction", prediction_times),
        ("End-to-End (Total)", total_times),
    ]:
        if not times:
            continue
        t = np.array(times)
        report_lines += [
            f"  {name}:",
            f"    Mean:   {np.mean(t):>8.3f} ms",
            f"    Median: {np.median(t):>8.3f} ms",
            f"    P95:    {np.percentile(t, 95):>8.3f} ms",
            f"    P99:    {np.percentile(t, 99):>8.3f} ms",
            f"    Min:    {np.min(t):>8.3f} ms",
            f"    Max:    {np.max(t):>8.3f} ms",
            "",
        ]

    if total_times:
        throughput = 1000.0 / np.mean(total_times)
        report_lines.append(f"  Throughput: ~{throughput:.0f} teeth/second")

    report_lines.append("=" * 60)
    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = OUTPUT_DIR / "inference_speed_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [SAVED] {report_path}")

    # ---- Latency histogram ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, times, color) in zip(axes, [
        ("Feature Extraction", extraction_times, "#3b82f6"),
        ("Model Prediction", prediction_times, "#f59e0b"),
        ("End-to-End", total_times, "#10b981"),
    ]):
        if not times:
            continue
        t = np.array(times)
        ax.hist(t, bins=25, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(x=np.mean(t), color="#dc2626", linestyle="--",
                   linewidth=2, label=f"Mean: {np.mean(t):.2f}ms")
        ax.axvline(x=np.median(t), color="#7c3aed", linestyle=":",
                   linewidth=2, label=f"Median: {np.median(t):.2f}ms")
        ax.set_xlabel("Latency (ms)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Inference Latency Distribution",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "inference_latency_histogram.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run selected (or all) analysis tasks."""
    parser = argparse.ArgumentParser(
        description="Advanced Analysis Pipeline for Dental Caries Model"
    )
    parser.add_argument(
        "--task", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
        help="Run specific tasks (e.g. --task 1 3). Default: all tasks.",
    )
    args = parser.parse_args()
    tasks_to_run = args.task if args.task else [1, 2, 3, 4, 5]

    print("=" * 60)
    print("  ADVANCED ANALYSIS PIPELINE")
    print(f"  Tasks to run: {tasks_to_run}")
    print("=" * 60)

    setup_output_dirs()

    # Load data (shared across all tasks)
    model, feature_df, train_df, test_df = load_dataset_and_model()

    total_start = time.time()

    if 1 in tasks_to_run:
        task1_model_interpretability(model, train_df, test_df)

    if 2 in tasks_to_run:
        task2_deep_error_analysis(model, test_df)

    if 3 in tasks_to_run:
        task3_confidence_analysis(model, test_df)

    if 4 in tasks_to_run:
        task4_robustness_check(feature_df)

    if 5 in tasks_to_run:
        task5_deployment_optimization(model, test_df)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE! Total time: {elapsed:.1f}s")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
