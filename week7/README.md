# Week 7 — Pipeline Hardening & Final Evaluation

Step-by-step guide to reproduce the **Week 7** results from scratch.

---

## Prerequisites

### 1. Environment Setup

```bash
# Create conda environment
conda create -n sp_project python=3.10 -y
conda activate sp_project

# Install dependencies (from repo root)
pip install -r requirements.txt
```

### 2. Required Upstream Data

Week 7 scripts depend on outputs from earlier weeks and raw material files.
All paths are relative to the project root (`SP/`).

| Source | Path | Contents | Produced By |
|--------|------|----------|-------------|
| **Material: X-rays** | `material/500 cases with annotation/case {N}/case_{N}.png` | Panoramic dental X-ray images | Provided (raw data) |
| **Material: GT annotations** | `material/500 cases with annotation/case {N}/*.xml` | AIM-XML ground truth (tooth ID + surface + ROI) | Provided (expert annotations) |
| **Material: ROI masks** | `material/500-roi/case_{N}.png` | Binary caries masks (white = caries region) | Provided (pre-generated) |
| **Week 2: Tooth segmentation** | `week2/500-segmentation+recognition/case {N}/case_{N}_results.json` | Per-tooth polygon coordinates + `pixel_coordinates` | `week2/process_500_cases.py` |
| **Week 5: Surface diagnosis** | `week5/surface_classification_output/case {N}/case_{N}_diagnosis.json` | Initial surface classification (used as prediction input) | `week5/process_surface_classification.py` |
| **Week 6: Shared modules** | `week6/xml_ground_truth_parser.py` | AIM-XML parser (re-exported by week7 wrapper) | Standalone module |
| **Week 6: Shared modules** | `week6/snodent_tooth_map.py` | SNODENT ↔ FDI mapping tables (re-exported by week7 wrapper) | Standalone module |

> **Verify** before running: all 500 `case {N}` folders must exist under `material/500 cases with annotation/`, `material/500-roi/`, `week2/500-segmentation+recognition/`, and `week5/surface_classification_output/`.

---

## Week 7 Scripts Overview

| Script | Purpose |
|--------|---------|
| `dental_caries_analysis.py` | **Step 1** — Caries-to-tooth mapping with boundary erosion + unassigned blob detection |
| `evaluation_engine.py` | **Step 2** — GT-vs-prediction matching with phantom FP filter + soft surface matching |
| `validation_dashboard.py` | **Step 3** — Per-case visualization: panoramic overview + per-tooth PCA/M-C-D panels |
| `select_hero_shots_v2.py` | **Step 4** — Curated hero-shot selector (5 win categories) |
| `multi_zone_classifier.py` | Shared module — PCA orientation + M/C/D zone voting (4-rule fix) |
| `xml_ground_truth_parser.py` | Wrapper — imports `week6/xml_ground_truth_parser.py` via importlib |
| `snodent_tooth_map.py` | Wrapper — imports `week6/snodent_tooth_map.py` via importlib |

---

## Step-by-Step Reproduction

All commands must be run from the `week7/` directory with the conda environment activated:

```bash
cd week7
conda activate sp_project
```

### Step 1: Caries Mapping (500 Cases)

Reads week2 tooth polygons + material ROI masks → writes per-case JSON with caries coordinates, `has_caries` flag, and unassigned blobs.

```bash
python dental_caries_analysis.py --cases 500
```

**Input:**
- `week2/500-segmentation+recognition/case {N}/case_{N}_results.json`
- `material/500-roi/case_{N}.png`

**Output:**
- `week7/dental_analysis_output/case {N}/case_{N}_caries_mapping.json`

**Runtime:** ~20 minutes (500 cases @ ~2.3 s/case)

**Key features (vs. week3):**
- Task 2: **Boundary erosion** — 5 px morphological erosion eliminates overlap-induced false positives
- Task 3: **Unassigned caries** — blobs that fall outside all tooth polygons are logged separately

> To process specific cases only: `python dental_caries_analysis.py --sample 1,3,33,311`

---

### Step 2: Evaluation Engine (500 Cases)

Reads GT XML annotations + week5 predictions + week7 caries mapping → computes TP/FP/FN, runs multi-zone reclassification, and applies soft surface matching.

```bash
python evaluation_engine.py
```

**Input:**
- `material/500 cases with annotation/case {N}/*.xml` (ground truth)
- `week5/surface_classification_output/case {N}/case_{N}_diagnosis.json` (predictions)
- `week7/dental_analysis_output/case {N}/case_{N}_caries_mapping.json` (from Step 1)
- `week2/500-segmentation+recognition/case {N}/case_{N}_results.json` (tooth polygons)

**Output:**
- `week7/evaluation_output/evaluation_results.csv` — per-tooth row with match type, surfaces, zone fractions
- `week7/evaluation_output/evaluation_summary.json` — aggregate metrics (TP/FP/FN/F1/accuracy)
- `week7/evaluation_output/confusion_matrix_coarse.png`
- `week7/evaluation_output/confusion_matrix_fine.png`
- `week7/evaluation_output/case {N}/case_{N}_diagnosis.json` — enriched prediction JSON

**Runtime:** ~35 minutes (500 cases with MZ reclassification)

**Key features (vs. week6):**
- Task 1: **PCA 4-rule orientation fix** — eigenvector swap, occlusal/apical, mesial/distal, ±45° clamp
- Task 4: **Soft surface matching** — Proximal ↔ Mesial/Distal counts as partial match (≥30% zone fraction)
- Task 5: **Phantom FP filter** — teeth with `has_caries=False` are excluded from predictions before matching

> To evaluate a single case: `python evaluation_engine.py --case 1`
> To evaluate specific cases: `python evaluation_engine.py --sample 1,3,33,311`

---

### Step 3: Validation Dashboards (500 Cases)

Generates per-case PNG images with panoramic X-ray overview (GT polygons + prediction scatter + TP/FP/FN labels) and per-tooth PCA panels showing M-C-D zone classification.

```bash
python validation_dashboard.py
```

**Input:**
- `material/500 cases with annotation/case {N}/case_{N}.png` (X-ray image)
- `material/500 cases with annotation/case {N}/*.xml` (ground truth)
- `week2/500-segmentation+recognition/case {N}/case_{N}_results.json` (tooth polygons)
- `week5/surface_classification_output/case {N}/case_{N}_diagnosis.json` (predictions)
- `week7/dental_analysis_output/case {N}/case_{N}_caries_mapping.json` (from Step 1)

**Output:**
- `week7/dental_analysis_output/case {N}/validation_case_{N}.png`

**Runtime:** ~60 minutes (500 cases)

> To generate specific cases: `python validation_dashboard.py --sample 1,3,33,311`
> To adjust quality: `python validation_dashboard.py --dpi 200`

---

### Step 4: Hero Shots (Curated Validation Images)

Selects 25 representative dashboard images across 5 "win" categories from the evaluation CSV.

```bash
python select_hero_shots_v2.py
```

**Input:**
- `week7/evaluation_output/evaluation_results.csv` (from Step 2)
- `week7/dental_analysis_output/case {N}/validation_case_{N}.png` (from Step 3)

**Output:**
- `week7/hero_shots/1_Fallback_Angle_Clamp_Win/` (5 images)
- `week7/hero_shots/2_Rescued_TP_Soft_Match_Win/` (5 images)
- `week7/hero_shots/3_Squarish_Molar_PCA_Fix_Win/` (5 images)
- `week7/hero_shots/4_Missing_Upstream_Data_Logging/` (5 images)
- `week7/hero_shots/5_Overall_Dashboard_Summary/` (5 images)
- `week7/hero_shots/hero_shots_manifest.json`

**Runtime:** < 1 minute

---

## Full Reproduction Script (Copy-Paste)

Run all 4 steps sequentially from the repo root:

```powershell
cd week7
conda activate sp_project

# Clean previous outputs
Remove-Item "dental_analysis_output\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "evaluation_output\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "hero_shots\*" -Recurse -Force -ErrorAction SilentlyContinue

# Step 1: Caries mapping (~20 min)
python dental_caries_analysis.py --cases 500

# Step 2: Evaluation (~35 min)
python evaluation_engine.py

# Step 3: Dashboards (~60 min)
python validation_dashboard.py

# Step 4: Hero shots (<1 min)
python select_hero_shots_v2.py
```

Total runtime: **~2 hours** on a modern machine.

---

## Expected Final Metrics

| Metric | Value |
|--------|-------|
| Cases Processed | 500 |
| Ground Truth Annotations | 1,979 |
| True Positives (TP) | 1,424 |
| False Positives (FP) | 13 |
| False Negatives (FN) | 555 |
| **Precision** | **0.9910** |
| **Recall** | **0.7196** |
| **F1-Score** | **0.8337** |
| Strict Surface Accuracy | 0.7219 |
| **Soft Surface Accuracy** | **1.0000** |

---

## Week 7 Bug Fixes Summary

| Task | Bug | Fix |
|------|-----|-----|
| **Task 1** | PCA eigenvectors swapped for near-square molars → wrong M/D orientation | 4-rule PCA: verticality check, occlusal/apical flip, mesial/distal flip, ±45° angle clamp |
| **Task 2** | Overlapping tooth masks cause boundary leakage → false caries | 5 px morphological erosion on each tooth mask before caries overlap |
| **Task 3** | Caries outside all tooth polygons silently dropped | Unassigned blobs logged in JSON under `unassigned_caries` |
| **Task 4** | Strict surface matching penalizes GT "Proximal" vs predicted "Mesial" | Soft matching: if any zone fraction ≥ 30%, count as partial match |
| **Task 5** | Teeth with `has_caries=False` counted as positive predictions | Filter predictions list before matching — only teeth with actual caries participate |
| **M/D Flip** | Mesial/Distal labels inverted for Q1/Q4 teeth in dashboard | FDI quadrant-aware zone assignment: Q1/Q4 M=+X, Q2/Q3 M=−X |
