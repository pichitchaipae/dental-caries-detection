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
[![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-2ea44f?style=flat-square)]()

<br>

[📊 Presentation Slides](https://canva.link/h7d54mdjzus4k5x) &nbsp;|&nbsp; [📄 Full Report](https://drive.google.com/file/d/19kFYW0QC-_qDAPtWYe8fHxX3QlMRT3sd/view?usp=sharing)

<br>

**ITDS491 — Senior Project I &amp; II** (Semester 1 &amp; 2, AY 2569) &nbsp;·&nbsp; **ITDS346 — Practical Data Science** (Semester 2, AY 2568)

Naris Pholpak 6687025 &nbsp;·&nbsp; Pichitchai Paecharoenchai 6687033 &nbsp;·&nbsp; Sukollapat Pisuchpen 6687052

</div>

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

---

## 1. Project Overview

This repository presents a complete pipeline for **automated dental caries surface classification** from panoramic dental X-ray images (Orthopantomogram / OPG), developed as part of the ITDS346 Practical Data Science course at Mahidol University.

The system combines Computer Vision and Deep Learning techniques to analyze panoramic radiographs and accurately identify regions at risk of tooth decay. The dataset was provided by the **Faculty of Dentistry, Mahidol University**, consisting of 500 real-world clinical cases with structured XML annotations.

The pipeline spans detection, segmentation, mapping, and geometric surface classification — progressing through iterative weekly sprints across two phases:

- **Phase 1:** Explored 7 PCA-based orientation methods for tooth normalization, selecting the *Split-Centroid Anatomy* method (Macro F1: 0.41) as the most robust.
- **Phase 2:** Refined surface classification through spatial zoning strategies, with the *X-Thirds Dominant Zone* selected as the final model (Accuracy: 70.54%, Macro F1: 0.4714).

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

```text
SP/
├── README.md                              # Project documentation and setup guide
├── requirements.txt                       # Core Python dependencies
├── cleanup-log.md                         # Log of atomic restructuring operations
│
├── assets/                                # Static assets (logos, images)
├── data/                                  # Datasets and pre-trained models (~1.63 GB)
│   ├── 500 cases with annotation/         # Panoramic X-rays + XML annotations
│   ├── 500-roi/                           # Region of Interest masks
│   └── Tooth Segmentation + Recognition model/  # Pre-trained Detectron2 weights
├── docs/                                  # Architectural diagrams and documentation
├── outputs/                               # Diagnostics and system execution logs
├── scripts/                               # Utility and helper scripts
│
├── phase2-1april/                         # ★ FINAL PIPELINE: Integration notebooks & PCA outputs
├── week1/                                 # Sprint 1: Segmentation model training scripts
├── week2/                                 # Sprint 2: 500-case batch segmentation pipeline
├── week2-Tooth Detection & Segmentation/  # Generated artifacts (~39.37 GB)
├── week3/                                 # Sprint 3: Caries-to-tooth mapping logic
├── week3-Caries-to-Tooth Mapping/         # Generated mapping snapshots
├── week4/                                 # Sprint 4: Caries inference pipeline integration
├── week5/                                 # Sprint 5: Surface Classification v1
├── week5-Surface Classification v1/       # Generated v1 artifacts
├── week6/                                 # Sprint 6: Surface Classification v2 & Evaluation Engine
├── week7/                                 # Sprint 7: Evaluation workflows & dashboards
├── week7-Surface Classification v3/       # Generated v3 artifacts
├── week8-Surface Classification v4/       # Generated v4 artifacts
├── week9_pca_evaluation/                  # Sprint 9: Core PCA algorithm evaluation
│
├── _non-using/                            # Deprecated/archived experiments (~15.57 GB)
├── _organizer-reports/                    # Filesystem inventory and duplication analysis
├── _unsorted/                             # Files pending manual review
└── _trash/                                # Non-destructive staging area for cache artifacts
```

### Key Files

| File | Description |
|---|---|
| `phase2-1april/pipeline-phase1-v4.ipynb` | **Primary notebook** — orchestrates the complete Phase 2 pipeline |
| `phase2-1april/report_v4.5.txt` | Evaluation report for the **v4.5 X-Thirds Dominant** algorithm |
| `phase2-1april/report_v4.6.txt` | Evaluation report for the **v4.6 Diagonal-from-Centroid** algorithm |
| `week2/process_500_cases.py` | Batch processor for full 500-case tooth segmentation |
| `week4/run_full_batch_v3_advanced.py` | Advanced batch runner for caries detection inference |
| `week6/evaluation_engine.py` | Standardized evaluation framework used across all sprint iterations |

---

## 5. Installation & Setup

> **Recommendation:** Use an isolated Conda environment to avoid dependency conflicts, especially for Detectron2.

```bash
# 1. Create and activate environment
conda create -n dental_caries python=3.10 -y
conda activate dental_caries

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install Detectron2 (GPU: ensure matching CUDA toolkit is installed first)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 4. (Optional) Install Week 2 legacy dependencies
pip install -r week2/requirements.txt
```

---

## 6. Execution Workflows

### CLI — Batch Processing

```bash
# Tooth segmentation across full 500-case dataset
python week2/process_500_cases.py

# Caries detection batch inference
python week4/run_full_batch_v3_advanced.py

# Surface classification batch
python week5/process_surface_classification.py
```

### Interactive — Jupyter Notebook

For step-by-step execution with visualizations:

1. Launch Jupyter: `jupyter notebook` or `jupyter lab`
2. Open `phase2-1april/pipeline-phase1-v4.ipynb`
3. Update the data and model directory paths in the configuration cells
4. Run cells sequentially from top to bottom

---

## 7. Dataset Specifications

Data was provided by the **Faculty of Dentistry, Mahidol University**.

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
- `phase2-1april/PCA_Output_v4.5/`
- `phase2-1april/PCA_Output_v4.6/`

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
| **Pending Review** | `_unsorted/pipeline-phase1-v3_backup.ipynb` (relocated from root) is awaiting manual code review. |
| **Deduplication** | `_organizer-reports/` contains CSV/TXT duplication analysis logs. Review thoroughly before running any destructive deduplication operations. |
| **Deletion Policy** | `_trash/` is a non-destructive staging area for cache artifacts only (`__pycache__` trees). No experimental results or user-generated data have been permanently deleted. |

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
