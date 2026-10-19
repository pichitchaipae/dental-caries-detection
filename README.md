<div align="center">

<img src="https://blogger.googleusercontent.com/img/a/AVvXsEiDtzyVuqT0gM-UvARYmqjZ_RV63zD0aPq2mwBMlhybB5gRG_CdT9ASjD4UUb9yP-enBHiv_x9pCXa5gOCm2v_jQX7FkYJpfa75JwYcwG3RjnKaZXpN0CJsUVVP5mV7MSH8BkBVLzxCpF1K9Heyw5kHluqGLqRkdVuHVPi8DOi-LSbW2_I9TKSjbfJqAQ_m" alt="Project Cover Banner" width="100%">

<br>

<img src="https://www.ict.mahidol.ac.th/wp-content/uploads/2022/05/MUICT-LOGO-en-scaled.png" alt="ICT Mahidol Logo" width="220">

<br><br>

# Automated Dental Caries Surface Classification

**Multi-Stage Computer Vision Pipeline for Panoramic Dental X-Rays (OPG)**

<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![Detectron2](https://img.shields.io/badge/Detectron2-Facebook%20Research-7B2FBE?style=flat-square&logo=meta&logoColor=white)](https://github.com/facebookresearch/detectron2)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Phase%202%20%2B%20CNN%20deployed-2ea44f?style=flat-square)]()

<br>

[📊 Presentation Slides](https://canva.link/h7d54mdjzus4k5x) &nbsp;|&nbsp; [📄 Full Report](https://drive.google.com/file/d/19kFYW0QC-_qDAPtWYe8fHxX3QlMRT3sd/view?usp=sharing)

<br>

**ITDS491 — Senior Project I &amp; II** (Semester 1 &amp; 2, AY 2569) &nbsp;·&nbsp; **ITDS346 — Practical Data Science** (Semester 2, AY 2568)

Naris Pholpak 6687025 &nbsp;·&nbsp; Pichitchai Paecharoenchai 6687033 &nbsp;·&nbsp; Sukollapat Pisuchpen 6687052

</div>

---

## Current status (October 2026)

The repository currently holds three things:

- **`reserch/`** — the research phase: Phase 1 PCA orientation, Phase 2 spatial zoning and the weekly sprint history (sections 1–12 below).
- **`dental-caries-detection/`** — the deployed system: FastAPI ml-service (stages 1–3b), TypeScript backend, React frontend and Docker Compose.
- **`caries-detection-model/`** — the CNN training, calibration and evaluation pipeline behind the deployed surface model.

The deployed surface classifier is the **5-fold PCA-42 CNN (Stage 3b)**; the Random Forest classifier is kept as fallback — see [§13](#13-deployed-system-cnn-surface-model).

- Surface-model report: [`dental-caries-detection/docs-md/surface-cnn-full-report.md`](dental-caries-detection/docs-md/surface-cnn-full-report.md)
- Service documentation: [`dental-caries-detection/ml-service/README.md`](dental-caries-detection/ml-service/README.md)
- Pipeline and API data-flow map: [`dental-caries-detection/docs-md/ml-service-data-flow.md`](dental-caries-detection/docs-md/ml-service-data-flow.md)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Feature Engineering & Classification Logic](#3-feature-engineering--classification-logic)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Execution Workflows](#6-execution-workflows)
7. [Dataset Specifications](#7-dataset-specifications)
8. [Evaluation Results](#8-evaluation-results)
9. [Limitations & Challenges](#9-limitations--challenges)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)
12. [Development Notes](#12-development-notes)
13. [Deployed System: CNN Surface Model](#13-deployed-system-cnn-surface-model)

---

## 1. Project Overview

This repository presents a complete pipeline for **automated dental caries surface classification** from panoramic dental X-ray images (Orthopantomogram / OPG), developed as part of the ITDS346 Practical Data Science course at Mahidol University.

The system combines Computer Vision and Deep Learning techniques to analyze panoramic radiographs and accurately identify regions at risk of tooth decay. The dataset was provided by the **Faculty of Dentistry, Mahidol University**, consisting of 500 real-world clinical cases with structured XML annotations.

The pipeline spans detection, segmentation, mapping, and geometric surface classification — progressing through iterative weekly sprints across two phases:

- **Phase 1:** Explored 7 PCA-based orientation methods for tooth normalization, selecting the *Split-Centroid Anatomy* method (Macro F1: 0.41) as the most robust.
- **Phase 2:** Refined surface classification through spatial zoning strategies, with the *X-Thirds Dominant Zone* selected as the final model (Accuracy: 70.54%, Macro F1: 0.4714).
- **Deployment (2026):** in the deployed service the geometric zoning classifier was replaced by a per-tooth CNN trained on the same annotations, with the Random Forest kept as fallback — see [§13](#13-deployed-system-cnn-surface-model).

Caries are classified onto three anatomical surfaces:

| Surface | Description |
|---|---|
| **Occlusal** | Biting/chewing surface (top) |
| **Mesial** | Surface facing toward the midline |
| **Distal** | Surface facing away from the midline |

---

## 2. Pipeline Architecture

```
  OPG Radiograph Input
         │
         ▼
┌─────────────────────┐
│  1. Tooth Detection  │  Detectron2 instance segmentation → pixel-perfect tooth masks
│   & Segmentation    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Caries Detection │  Localize regions of decay within the panoramic radiograph
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│  3. Caries-to-Tooth  │  Establish spatial correlations between caries and teeth
│      Mapping         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. Surface          │  PCA-based geometric alignment + spatial partitioning
│     Classification   │  → Occlusal / Mesial / Distal label per caries region
└──────────────────────┘
```

---

## 3. Feature Engineering & Classification Logic

### Tooth Orientation Normalization (Phase 1)

Seven PCA-based orientation strategies were explored to normalize tooth geometry before classification:

| # | Method | Macro F1 | Notes |
|:---:|---|:---:|---|
| 1 | Standard PCA (Baseline) | 0.30 | Axis flipping in symmetric molars |
| 2 | Conditional Ratio | 0.40 | Reduces 90° misalignment |
| 3 | Maximum Projected Span | 0.40 | More stable than Conditional Ratio |
| **4** | **Split-Centroid Anatomy** ✦ | **0.41** | **Selected — encodes biological growth direction** |
| 5 | Furthest Point Alignment | — | Unstable with incomplete masks |
| 6 | Absolute Vertical Prior | 0.40 | Sensitive to image tilt |
| 7 | Multi-Class Segmentation | — | Requires re-annotation; not implemented |

**Split-Centroid Anatomy** splits the tooth mask into upper/lower halves, computes each centroid, and constructs an anatomical vector from root to crown — significantly reducing symmetric ambiguity.

### Surface Classification Logic (Phase 2)

After PCA orientation, lesion pixels are projected into the normalized coordinate frame and classified via **Multi-Zone Point-Cloud Voting**:

1. Divide the tooth into **Mesial / Occlusal / Distal** zones.
2. Compute the proportion of caries pixels per zone.
3. Mark a zone positive if it contains ≥ 5% of total lesion pixels.
4. Apply connected-component filtering (min 15 px) to remove noise.
5. Map zone combination → surface label (M, D, O, MO, DO, MOD).

Two zoning strategies were evaluated:

| Strategy | Zone Split | Accuracy | Macro F1 |
|---|---|:---:|:---:|
| **X-Thirds Dominant** ✦ | Mesial 40% / Occlusal 20% / Distal 40% | **70.54%** | **0.4714** |
| Diagonal-from-Centroid | Diagonal regions from centroid | 60.28% | 0.4601 |

---

## 4. Repository Structure

Current tracked layout of the repository (the research history now lives under `reserch/`):

```text
SP/                                          # repository root
├── README.md                                # This document
│
├── dental-caries-detection/                 # ★ Deployed system
│   ├── ml-service/                          # FastAPI inference service: stages 1–3b + test harnesses
│   ├── backend/                             # API service and result adapter (TypeScript)
│   ├── frontend/                            # Web UI (React + Vite)
│   ├── docs-md/                             # API contract, data-flow map, surface-CNN report
│   └── docker-compose.yml                   # db + ml-service + backend + frontend
│
├── caries-detection-model/                  # CNN training / calibration / evaluation
│   ├── pipeline.py                          # build → train → eval → calibrate → report (locked protocol)
│   ├── deploy/                              # Staging service + parity / smoke tests
│   └── outputs/                             # PROTOCOL_LOCK.md, calibration, OOF comparison
│
├── ml_model/                                # Model weights and research artifacts
│
├── reserch/                                 # Research phase 1–2 (sprint history)
│   ├── phase2-1april/                       # ★ Final research pipeline + report_v4.5 / report_v4.6
│   ├── week1/ … week7/ · weekphase2/        # Weekly sprint work
│   ├── assets/ · images/ · scripts/ · src/  # Figures, tooling and helper code
│   └── requirements.txt · cleanup-log.md · README.md
│
├── caries_surface_classification/           # Earlier standalone surface-classifier prototype
├── z-claude-md/ · _trash/ · .github/workflows/
└── data/                                    # Dataset (~1.63 GB) — NOT tracked in git
```

> **Not tracked in git** (large datasets and generated artifacts — present in a developer's working tree only): `data/`, `week2-Tooth Detection & Segmentation/`, `week3-Caries-to-Tooth Mapping/`, `week5-Surface Classification v1/`, `week6/`, `week7-Surface Classification v3/`, `week8-Surface Classification v4/`, `week9_pca_evaluation/`, `_non-using/`, model weights and `ml-service/weights/`.

### Key Files

| File | Description |
|---|---|
| `dental-caries-detection/ml-service/app/pipeline/orchestrator.py` | Inference pipeline: stages 1–3 (detection, PCA, RF) plus the Stage 3b CNN hook |
| `dental-caries-detection/ml-service/app/pipeline/surface_cnn.py` | Stage 3b CNN surface model (`run_shadow()` / `primary_findings()`) |
| `caries-detection-model/pipeline.py` | CNN build → train → evaluate → calibrate → report pipeline |
| `reserch/phase2-1april/pipeline-sp-ml-optimize.ipynb` | Phase 2 integration notebook (also `pipeline-phase1-v3.ipynb`) |
| `reserch/phase2-1april/report_v4.5.txt` | Evaluation report for the **v4.5 X-Thirds Dominant** algorithm |
| `reserch/phase2-1april/report_v4.6.txt` | Evaluation report for the **v4.6 Diagonal-from-Centroid** algorithm |
| `reserch/week2/process_500_cases.py` | Batch processor for full 500-case tooth segmentation |
| `reserch/week4/run_full_batch_v3_advanced.py` | Advanced batch runner for caries detection inference |
| `reserch/week6/evaluation_engine.py` | Standardized evaluation framework used across all sprint iterations |

---

## 5. Installation & Setup

> **Recommendation:** Use an isolated Conda environment to avoid dependency conflicts, especially for Detectron2.

```bash
# 1. Create and activate environment
conda create -n dental_caries python=3.10 -y
conda activate dental_caries

# 2. Install base dependencies
pip install -r reserch/requirements.txt

# 3. Install Detectron2 (GPU: ensure matching CUDA toolkit is installed first)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 4. (Optional) Install Week 2 legacy dependencies
pip install -r reserch/week2/requirements.txt
```

> The research scripts below keep legacy relative paths, so run them from the `reserch/` directory (or copy the folders next to the scripts).
> The deployed system has its own setup — Docker Compose, no local Python environment needed. See [§13](#13-deployed-system-cnn-surface-model).

---

## 6. Execution Workflows

### CLI — Batch Processing

```bash
# Tooth segmentation across full 500-case dataset
python reserch/week2/process_500_cases.py

# Caries detection batch inference
python reserch/week4/run_full_batch_v3_advanced.py

# Surface classification batch
python reserch/week5/process_surface_classification.py
```

### Interactive — Jupyter Notebook

For step-by-step execution with visualizations:

1. Launch Jupyter: `jupyter notebook` or `jupyter lab`
2. Open `reserch/phase2-1april/pipeline-sp-ml-optimize.ipynb` (or `pipeline-phase1-v3.ipynb`)
3. Update the data and model directory paths in the configuration cells
4. Run cells sequentially from top to bottom

---

## 7. Dataset Specifications

Data was provided by the **Faculty of Dentistry, Mahidol University**.

> The dataset is **not tracked in git** (~1.63 GB in total) — it lives at `data/` in a developer's working tree only, so the paths below are working-tree paths, not repository paths.

| Component | Path | Size | Description |
|---|---|---|---|
| Panoramic X-rays | `data/500 cases with annotation/` | ~1.25 GB | 500 cases with XML annotations (FDI notation) |
| ROI Masks | `data/500-roi/` | ~4 MB | Binary mask images — pixels > 0 indicate caries regions |
| Model Weights | `data/Tooth Segmentation + Recognition model/` | ~372 MB | Pre-trained Detectron2 config & weights |

- **Image naming:** `case_<id>.png` with a paired XML annotation file per case directory.
- **Annotation format:** XML files contain tooth IDs (FDI notation) and caries surface labels. Only valid cases (complete tooth ID + surface label) are used for evaluation.

### Dataset Statistics

| Label | Count |
|---|:---:|
| Total cases | 500 |
| Valid cases | 500 |
| **Total caries lesions** | **1,979** |
| Occlusal | 388 (19.6%) |
| Mesial | 670 (33.9%) |
| Distal | 921 (46.5%) |

> **Note:** The dataset is class-imbalanced — Distal and Mesial dominate, which directly impacts model performance on Occlusal classification.

> **⚠️ Storage Warning:** Batch processing generates large derived artifacts. The `week2-Tooth Detection & Segmentation/` directory alone requires ~**39.37 GB**. Verify available disk space before running batch jobs.

---

## 8. Evaluation Results

Evaluated on **1,979 samples** from the full 500-case dataset.

> This section reports the **research (Phase 2)** comparison of the two geometric zoning strategies. The deployed service classifies surfaces with the CNN of [§13](#13-deployed-system-cnn-surface-model) instead; its end-to-end operating point is reported there.

### Model Performance

| Metric | X-Thirds Dominant ✦ | Diagonal-from-Centroid |
|---|:---:|:---:|
| **Accuracy** | **70.54%** (1396/1979) | 60.28% (1193/1979) |
| **Macro F1-Score** | **0.4714** | 0.4601 |

### Statistical Validation

Two tests were applied to determine whether the accuracy difference is statistically significant:

**Bootstrap Resampling (Macro F1-Score)**
- Observed difference: 0.0113
- 95% CI: [−0.0091, 0.0307] &nbsp; · &nbsp; p-value: **0.2820**
- → *No statistically significant difference in F1-score* (p > 0.05)

**Two-Proportion Z-Test (Accuracy)**
- Z-statistic: 6.7837 &nbsp; · &nbsp; p-value: **1.17 × 10⁻¹¹**
- → *Accuracy difference is highly significant* (p << 0.05)

> The discrepancy arises from class imbalance: accuracy is dominated by Mesial/Distal cases, while Macro F1 treats all classes equally. Both tests together confirm **X-Thirds Dominant** as the superior and more stable choice.

### Error Analysis

| Model | Strength | Weakness |
|---|---|---|
| X-Thirds Dominant | Strong Mesial & Distal performance | Lower Occlusal recall |
| Diagonal-from-Centroid | Improved Occlusal detection | Major Distal performance drop due to geometric distortion |

Visualization outputs (PCA axis overlays, classification boundary renders):
- `reserch/phase2-1april/PCA_Output_Run3/` — PCA axis overlays for the run-3 evaluation
- `reserch/phase2-1april/classification_metrics_run3.png` · `confusion_matrix_run3.png` · `feature_importance.png`

---

## 9. Limitations & Challenges

| # | Limitation | Impact |
|:---:|---|---|
| 1 | **Class Imbalance** — Occlusal cases are significantly fewer than Mesial/Distal | Biases accuracy toward dominant classes; lowers Occlusal sensitivity |
| 2 | **Root Region in Bounding Box** — Boxes span both crown and root | Elongated shapes distort diagonal-based zoning (Occlusal zone inflates, Distal shrinks) |
| 3 | **Segmentation Quality** — Incomplete masks, noise from adjacent teeth | Errors propagate into PCA orientation and surface classification |
| 4 | **No Crown/Root Separation** — No explicit anatomical split in annotations | Limits effectiveness of geometry-based methods |
| 5 | **Limited Data Diversity** — Single-source dataset from one institution | May reduce generalizability to external clinical settings |
| 6 | **No External Validation** — Evaluated on single dataset split | Risk of overfitting to dataset-specific characteristics |

---

## 10. Conclusion

This project presents a complete, interpretable pipeline for automated dental caries surface classification from panoramic X-rays.

- **Phase 1** explored 7 PCA-based orientation methods; the **Split-Centroid Anatomy** approach (Macro F1: 0.41) was selected as the most robust.
- **Phase 2** evaluated two spatial zoning strategies on 1,979 samples. The **X-Thirds Dominant Zone** (Accuracy: **70.54%**, Macro F1: **0.4714**) outperformed Diagonal-from-Centroid (60.28%, 0.4601).
- Statistical validation confirmed the accuracy improvement is significant (Z-Test p < 0.05), while F1-scores are not significantly different (Bootstrap p = 0.28) — reflecting the influence of class imbalance.
- **Deployment (2026):** the production service replaced the geometric classifier with a per-tooth CNN — a 5-fold EfficientNet-B0 ensemble with temperature-scaled per-surface thresholds — keeping the 13-feature Random Forest as fallback. See [§13](#13-deployed-system-cnn-surface-model).

The final model demonstrates that combining geometric reasoning with statistical validation provides a reliable, clinically interpretable framework for medical image analysis.

---

## 11. References

1. Asci, E., Yilmaz, A., & Kamburoglu, K. (2024). Deep learning approach to automatic tooth caries segmentation in panoramic radiographs. *Children, 11*(6), 690. https://doi.org/10.3390/children11060690
2. Bayrakdar, I. S., et al. (2022). Deep-learning approach for caries detection and segmentation on dental bitewing radiographs. *Oral Radiology, 38*, 1–9. https://doi.org/10.1007/s11282-021-005779
3. Bui, T. H., et al. (2021). Deep fusion feature extraction for caries detection on dental panoramic radiographs. *Applied Sciences, 11*(5), 2005. https://doi.org/10.3390/app11052005
4. FDI World Dental Federation. FDI tooth numbering system. https://www.fdiworlddental.org/resources/tooth-numbering-systems
5. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A, 374*(2065), 20150202.
6. Lian, L., et al. (2021). Deep learning for caries detection and classification in panoramic radiographs. *Diagnostics, 11*(9), 1672.
7. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*. https://arxiv.org/abs/1505.04597

**GitHub Repository:** https://github.com/pichitchaipae/dental-caries-detection

---

## 12. Development Notes

| Topic | Note |
|---|---|
| **Path Compatibility** | Directories with legacy spaces (e.g., `500 cases with annotation/`) are intentionally preserved for backward compatibility with `week*` scripts. |
| **Pending Review** | `reserch/_unsorted/pipeline-phase1-v3_backup.ipynb` (relocated from the old root) is awaiting manual code review. |
| **Deduplication** | `reserch/_organizer-reports/` contains CSV/TXT duplication analysis logs. Review thoroughly before running any destructive deduplication operations. |
| **Deletion Policy** | `_trash/` is a non-destructive staging area for cache artifacts only (`__pycache__` trees). No experimental results or user-generated data have been permanently deleted. |
| **Model weights** | The service reads its weights from a bind-mounted folder (`dental-caries-detection/ml-service/weights/`, git-ignored). Weights are never committed; only their SHA-256 hashes are recorded (`docs-md/surface-cnn-full-report.md` §5). |

---

## 13. Deployed System: CNN Surface Model

The surface classifier shipped in `dental-caries-detection/` is **Stage 3b**: a per-tooth CNN trained by `caries-detection-model/pipeline.py` under the locked protocol recorded in `caries-detection-model/outputs/PROTOCOL_LOCK.md`.

| Item | Value |
|---|---|
| Architecture | `efficientnet_b0` (feature dim 1280) + `Dropout(0.3)` + `Linear(1280, 3)` sigmoid head (multi-label) |
| Input | 384×384 tooth crop, ablation `pca` (PCA axis alignment, no CLAHE), focal loss (γ = 2) |
| Ensemble | 5 folds × 14 epochs, seed 42 (seed 43 kept as a training-variance replicate) |
| Output | Distal / Mesial / Occlusal probability per tooth lesion |
| Calibration | Per-surface temperature scaling + per-surface thresholds fitted on the 100 peeked calibration cases of the locked protocol, target recall 0.85 |
| Weights | `dental-caries-detection/ml-service/weights/cnn_pca42/` (bind-mounted read-only; not tracked in git) |
| Serving switch | `SURFACE_CNN_MODE` = `off` (default) · `shadow` · `cnn`; the deployed stack is configured with `cnn` |
| Fallback | The Random Forest surface classifier (`surface_source` = `rf`), used when the CNN bundle or its calibration is unavailable |

### End-to-end operating point

Measured on the real engine path (YOLO → mask-derived crop → CNN) with the per-fold out-of-fold thresholds and **no re-tuning** after the test results were seen — 201 cases / 6,012 teeth on CPU:

| Surface | AP | Recall | Precision |
|---|:---:|:---:|:---:|
| Distal | ≈ 0.579 | 0.781 | 0.335 |
| Mesial | ≈ 0.465 | 0.726 | 0.247 |
| Occlusal | ≈ 0.582 | 0.853 | 0.164 |

- Runtime: 358.2 s for 201 cases ≈ 1.78 s/case ≈ 60 ms/tooth (CPU).
- Pre-declared end-to-end gate (ΔAP versus the cached-crop arm, budget ≤ 0.05): −0.0507 / −0.0877 / −0.0227 → **Distal and Mesial fail, Occlusal passes**. The gate outcome was accepted explicitly before deployment, and the reasoning is recorded in the report below; the Random Forest stays available as the fallback path.
- Container parity (staging host vs. container, CPU): 6/6 exact output hashes, maximum probability difference 2.272e-07.

Method, protocol, calibration table, per-fold thresholds and limitations: [`dental-caries-detection/docs-md/surface-cnn-full-report.md`](dental-caries-detection/docs-md/surface-cnn-full-report.md)

---

<div align="center">

### Team Members

| Name | Student ID |
|---|:---:|
| Naris Pholpak | 6687025 |
| Pichitchai Paecharoenchai | 6687033 |
| Sukollapat Pisuchpen | 6687052 |

### Instructors

| Name | Role |
|---|---|
| Asst. Prof. Dr. Siripen Pongpaichet | Course Instructor |
| Dr. Sirawich Vachmanus | Course Instructor |

<br>

**ITDS491 — Senior Project I &amp; II** (Semester 1 &amp; 2, AY 2569) &nbsp;·&nbsp; **ITDS346 — Practical Data Science** (Semester 2, AY 2568)<br>
Bachelor of Science in Digital Science and Technology<br>
Faculty of Information and Communication Technology, Mahidol University

</div>
