# Week 7 — Pipeline Hardening & Final Evaluation

## Getting Started

This guide walks you through **everything** needed to run the Week 7 pipeline from a fresh clone. Most upstream data is `.gitignore`'d, so you must prepare it before Week 7 can run.

---

## Table of Contents

1. [Quick Overview](#1-quick-overview)
2. [Environment Setup](#2-environment-setup)
3. [Prepare Upstream Data (Before Week 7)](#3-prepare-upstream-data-before-week-7)
4. [Verify All Prerequisites](#4-verify-all-prerequisites)
5. [Run Week 7 Pipeline (4 Steps)](#5-run-week-7-pipeline-4-steps)
6. [Quick-Start: Single Case Test](#6-quick-start-single-case-test)
7. [Full 500-Case Reproduction Script](#7-full-500-case-reproduction-script)
8. [File & Directory Reference](#8-file--directory-reference)
9. [Expected Metrics](#9-expected-metrics)
10. [Bug Fixes Applied in Week 7](#10-bug-fixes-applied-in-week-7)

---

## 1. Quick Overview

Week 7 is the **final evaluation stage**. It reads outputs from earlier weeks, applies 6 bug fixes, and produces the final metrics + dashboards.

```text
Week 2 outputs ──┐
Material files ──┤──→  Week 7  ──→  Metrics / Dashboards / Hero Shots
Week 5 outputs ──┤
Week 6 modules ──┘
```

**Week 7 does NOT train any models.** It only performs post-processing, evaluation, and visualization.

---

## 2. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/pichitchaipae/dental-caries-detection.git
cd dental-caries-detection

# 2. Create conda environment
conda create -n sp_project python=3.10 -y
conda activate sp_project

# 3. Install dependencies
pip install -r requirements.txt
```

Required packages (auto-installed by `requirements.txt`):

| Package | Used For |
| --------- | ---------- |
| `opencv-python` | Image processing, PCA, morphological erosion |
| `numpy` | Array operations |
| `pandas` | CSV/DataFrame handling |
| `matplotlib` | Dashboard visualization |
| `seaborn` | Confusion matrix plots |
| `tqdm` | Progress bars |
| `torch` + `torchvision` | (indirect dependency from ultralytics) |
| `ultralytics` | YOLO model loading (week 1–4 only) |

> **Note:** `sklearn` is optional — only used for confusion matrix plotting. The pipeline will skip it gracefully if missing.

---

## 3. Prepare Upstream Data (Before Week 7)

These files are **not** in git (`.gitignore`'d). You must obtain or generate them first.

### 3.1 Material Files (Provided by Advisor)

Copy the raw dataset into `material/` at the repo root:

```text
SP/
└── material/
    ├── 500 cases with annotation/       ← 500 folders
    │   ├── case 1/
    │   │   ├── case_1.png               ← Panoramic X-ray image
    │   │   ├── *.xml                    ← AIM-XML ground truth annotations
    │   │   └── ...
    │   ├── case 2/
    │   └── ... (up to case 500)
    │
    └── 500-roi/                         ← 500 binary mask images
        ├── case_1.png                   ← White pixels = caries region
        ├── case_2.png
        └── ... (up to case_500.png)
```

### 3.2 Week 2 Output — Tooth Segmentation (Run Once)

Week 2 produces per-tooth polygon coordinates via YOLO + Detectron2.

```bash
cd week2
python process_500_cases.py
```

**Produces:**

```text
SP/week2/500-segmentation+recognition/
├── case 1/
│   └── case_1_results.json    ← contains teeth_data[].polygon, pixel_coordinates
├── case 2/
└── ... (500 folders)
```

> **Requires:** `material/500 cases with annotation/` and pre-trained model weights in `material/Tooth Segmentation + Recognition model/weights/`.

### 3.3 Week 5 Output — Surface Classification v1 (Run Once)

Week 5 produces the initial surface diagnosis per tooth.

```bash
cd week5
python process_surface_classification.py
```

**Produces:**

```text
SP/week5/surface_classification_output/
├── case 1/
│   └── case_1_diagnosis.json  ← contains teeth_data[].caries_surface, tooth_id
├── case 2/
└── ... (500 folders)
```

> **Requires:** Week 2 output + `material/500-roi/`.

### 3.4 Week 6 Modules (Already in Git ✓)

These two files are tracked in git and are imported by Week 7 via `importlib` wrappers:

- `week6/xml_ground_truth_parser.py` — parses AIM-XML ground truth
- `week6/snodent_tooth_map.py` — SNODENT code ↔ FDI notation mapping

**No action needed** — they are already in the repository.

---

## 4. Verify All Prerequisites

Run this checklist before starting Week 7. Every check must pass.

### PowerShell verification script

```powershell
cd C:\Users\jaopi\Desktop\SP

# Check material folders exist
Write-Output "=== Material ==="
Write-Output "X-rays:       $(Test-Path 'material/500 cases with annotation/case 1/case_1.png')"
Write-Output "GT XML:       $((Get-ChildItem 'material/500 cases with annotation/case 1/*.xml').Count) XML files"
Write-Output "ROI masks:    $(Test-Path 'material/500-roi/case_1.png')"

# Check week2 output
Write-Output "`n=== Week 2 ==="
Write-Output "Segmentation: $(Test-Path 'week2/500-segmentation+recognition/case 1/case_1_results.json')"

# Check week5 output
Write-Output "`n=== Week 5 ==="
Write-Output "Diagnosis:    $(Test-Path 'week5/surface_classification_output/case 1/case_1_diagnosis.json')"

# Check week6 modules
Write-Output "`n=== Week 6 ==="
Write-Output "XML parser:   $(Test-Path 'week6/xml_ground_truth_parser.py')"
Write-Output "SNODENT map:  $(Test-Path 'week6/snodent_tooth_map.py')"

# Count total cases available
Write-Output "`n=== Case Count ==="
Write-Output "Material:     $((Get-ChildItem 'material/500 cases with annotation' -Directory).Count) cases"
Write-Output "Week2:        $((Get-ChildItem 'week2/500-segmentation+recognition' -Directory).Count) cases"
Write-Output "Week5:        $((Get-ChildItem 'week5/surface_classification_output' -Directory).Count) cases"
```

### Bash verification script (Linux/macOS)

```bash
cd ~/Desktop/SP

echo "=== Material ==="
echo "X-rays:       $(test -f 'material/500 cases with annotation/case 1/case_1.png' && echo OK || echo MISSING)"
echo "GT XML:       $(ls material/500\ cases\ with\ annotation/case\ 1/*.xml 2>/dev/null | wc -l) XML files"
echo "ROI masks:    $(test -f material/500-roi/case_1.png && echo OK || echo MISSING)"

echo -e "\n=== Week 2 ==="
echo "Segmentation: $(test -f week2/500-segmentation+recognition/case\ 1/case_1_results.json && echo OK || echo MISSING)"

echo -e "\n=== Week 5 ==="
echo "Diagnosis:    $(test -f week5/surface_classification_output/case\ 1/case_1_diagnosis.json && echo OK || echo MISSING)"

echo -e "\n=== Week 6 ==="
echo "XML parser:   $(test -f week6/xml_ground_truth_parser.py && echo OK || echo MISSING)"
echo "SNODENT map:  $(test -f week6/snodent_tooth_map.py && echo OK || echo MISSING)"

echo -e "\n=== Case Count ==="
echo "Material:     $(ls -d material/500\ cases\ with\ annotation/case\ * 2>/dev/null | wc -l) cases"
echo "Week2:        $(ls -d week2/500-segmentation+recognition/case\ * 2>/dev/null | wc -l) cases"
echo "Week5:        $(ls -d week5/surface_classification_output/case\ * 2>/dev/null | wc -l) cases"
```

**Expected output** (all must show `True`/`OK` and 500 cases):

```text
=== Material ===
X-rays:       True
GT XML:       4 XML files
ROI masks:    True

=== Week 2 ===
Segmentation: True

=== Week 5 ===
Diagnosis:    True

=== Week 6 ===
XML parser:   True
SNODENT map:  True

=== Case Count ===
Material:     500 cases
Week2:        500 cases
Week5:        500 cases
```

---

## 5. Run Week 7 Pipeline (4 Steps)

All commands run from `week7/` with conda activated:

```bash
cd week7
conda activate sp_project
```

### Step 1: Caries Mapping

Maps caries pixels to teeth using boundary erosion (5 px) and detects unassigned blobs.

```bash
python dental_caries_analysis.py --cases 500
```

| | Detail |
| --- | -------- |
| **Reads** | `week2/.../case_{N}_results.json` + `material/500-roi/case_{N}.png` |
| **Writes** | `week7/dental_analysis_output/case {N}/case_{N}_caries_mapping.json` |
| **Runtime** | ~20 min (500 cases) |
| **Key fixes** | Task 2: boundary erosion, Task 3: unassigned caries logging |

### Step 2: Evaluation

Matches predictions against GT, filters phantom FPs, runs PCA-based multi-zone reclassification, applies soft surface matching.

```bash
python evaluation_engine.py
```

| | Detail |
| --- | -------- |
| **Reads** | GT XMLs + week5 diagnosis + week7 caries mapping + week2 polygons |
| **Writes** | `week7/evaluation_output/evaluation_results.csv`, `evaluation_summary.json`, confusion matrices |
| **Runtime** | ~35 min (500 cases) |
| **Key fixes** | Task 1: PCA fix, Task 4: soft match, Task 5: phantom FP filter |

### Step 3: Dashboards

Generates per-case PNG with panoramic X-ray overview + per-tooth PCA/M-C-D panels.

```bash
python validation_dashboard.py
```

| | Detail |
| --- | -------- |
| **Reads** | X-ray images + GT XMLs + week2 polygons + week5 diagnosis + week7 mapping |
| **Writes** | `week7/dental_analysis_output/case {N}/validation_case_{N}.png` |
| **Runtime** | ~60 min (500 cases) |
| **Key fixes** | M/D flip fix, phantom FP filter (consistent with evaluation) |

### Step 4: Hero Shots

Selects 25 representative dashboards across 5 "win" categories.

```bash
python select_hero_shots_v2.py
```

| | Detail |
|---|--------|
| **Reads** | `evaluation_results.csv` + dashboard PNGs from Step 3 |
| **Writes** | `week7/hero_shots/` (5 category folders + manifest JSON) |
| **Runtime** | < 1 min |

---

## 6. Quick-Start: Single Case Test

Want to test on just one or two cases before running all 500? Use `--sample`:

```bash
cd week7
conda activate sp_project

# Process only Cases 1 and 311
python dental_caries_analysis.py --sample 1,311
python evaluation_engine.py --sample 1,311
python validation_dashboard.py --sample 1,311
```

Or use the built-in test runner (Cases 311 & 33):

```bash
python run_test_sample.py
```

This runs both caries mapping and evaluation on Cases 311 + 33, and prints a per-tooth summary with soft-match details.

---

## 7. Full 500-Case Reproduction Script

Copy-paste this entire block to reproduce all results from scratch:

### PowerShell (Windows):

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

### Bash (Linux/macOS):

```bash
cd week7
conda activate sp_project

# Clean previous outputs
rm -rf dental_analysis_output/* evaluation_output/* hero_shots/*

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

## 8. File & Directory Reference

### Source Files (in git)

| File | Role | Depends On |
|------|------|------------|
| `dental_caries_analysis.py` | Step 1 — caries mapping | week2, material/500-roi |
| `evaluation_engine.py` | Step 2 — evaluation | material/GT XMLs, week2, week5, Step 1 output |
| `validation_dashboard.py` | Step 3 — dashboards | material/X-rays + GT, week2, week5, Step 1 output |
| `select_hero_shots_v2.py` | Step 4 — hero shots | Step 2 CSV, Step 3 PNGs |
| `multi_zone_classifier.py` | Shared — PCA + M-C-D voting | (imported by Steps 2 & 3) |
| `xml_ground_truth_parser.py` | Wrapper → `week6/xml_ground_truth_parser.py` | week6 module |
| `snodent_tooth_map.py` | Wrapper → `week6/snodent_tooth_map.py` | week6 module |
| `run_test_sample.py` | Quick test on Cases 311 & 33 | Same as Steps 1 & 2 |
| `README.md` | This file | — |

### Output Directories (gitignored)

| Directory | Created By | Contents |
|-----------|------------|----------|
| `dental_analysis_output/` | Steps 1 & 3 | 500 case folders with `*_caries_mapping.json` + `validation_*.png` |
| `evaluation_output/` | Step 2 | `evaluation_results.csv`, `evaluation_summary.json`, confusion matrices |
| `hero_shots/` | Step 4 | 5 category folders (25 PNGs) + `hero_shots_manifest.json` |

### Dependency Flow

```
material/500-roi/          ─┐
week2/500-segmentation+    ─┤──→ Step 1: dental_caries_analysis.py
recognition/                │         ↓
                            │    dental_analysis_output/case {N}/
                            │    case_{N}_caries_mapping.json
                            │         ↓
material/500 cases with    ─┤──→ Step 2: evaluation_engine.py
annotation/ (*.xml)         │         ↓
week5/surface_             ─┤    evaluation_output/
classification_output/      │    evaluation_results.csv
                            │         ↓
material/500 cases with    ─┤──→ Step 3: validation_dashboard.py
annotation/ (*.png)         │         ↓
                            │    dental_analysis_output/case {N}/
                            │    validation_case_{N}.png
                            │         ↓
                            └──→ Step 4: select_hero_shots_v2.py
                                      ↓
                                 hero_shots/
```

---

## 9. Expected Metrics

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

## 10. Bug Fixes Applied in Week 7

| Task | Bug | Fix |
|------|-----|-----|
| **Task 1** | PCA eigenvectors swapped for near-square molars → wrong M/D orientation | 4-rule PCA: verticality check, occlusal/apical flip, mesial/distal flip, ±45° angle clamp |
| **Task 2** | Overlapping tooth masks cause boundary leakage → false caries | 5 px morphological erosion on each tooth mask before caries overlap |
| **Task 3** | Caries outside all tooth polygons silently dropped | Unassigned blobs logged in JSON under `unassigned_caries` |
| **Task 4** | Strict surface matching penalizes GT "Proximal" vs predicted "Mesial" | Soft matching: if any zone fraction ≥ 30%, count as partial match |
| **Task 5** | Teeth with `has_caries=False` counted as positive predictions | Filter predictions list before matching — only teeth with actual caries participate |
| **M/D Flip** | Mesial/Distal labels inverted for Q1/Q4 teeth in dashboard | FDI quadrant-aware zone assignment: Q1/Q4 M=+X, Q2/Q3 M=−X |
