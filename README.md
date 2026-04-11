# SP Dental Caries Detection Workspace

<p align="center">
  <img src="assets/logoict.png" width="200" alt="ICT Mahidol Logo">
</p>

This workspace contains a multi-stage computer vision pipeline for panoramic dental X-rays (OPG). The main flow covers tooth segmentation and recognition, caries detection, caries-to-tooth mapping, and caries surface classification. Work is organized as week-based iterations plus a phase2 integration area.

The repository also includes large intermediate outputs, historical experiments, and evaluation artifacts. Recent cleanup actions focused on non-destructive organization: moving obvious root-level utility files into standard folders, archiving cache artifacts into `_trash`, and preserving all research content.

## Folder Structure

```text
SP/
|- README.md                          # Project overview and usage
|- cleanup-log.md                     # Atomic cleanup action log
|- requirements.txt                   # Base Python dependencies
|- assets/                            # Static assets (logo)
|- data/                              # Input data + pretrained model files (~1.63 GB)
|- docs/                              # Documentation assets and images
|- outputs/                           # Diagnostics and run logs generated at root level
|- scripts/                           # Utility scripts moved from root
|- src/                               # Source-adjacent archived JSON (old notebook export)
|- phase2-1april/                     # Integrated notebooks, reports, and PCA outputs
|- week1/                             # Tooth segmentation training assets/scripts
|- week2/                             # 500-case tooth segmentation batch pipeline
|- week2-Tooth Detection & Segmentation/  # Large segmentation outputs (~39.37 GB)
|- week3/                             # Caries-to-tooth mapping (early phase)
|- week3-Caries-to-Tooth Mapping/     # Mapping output snapshots
|- week4/                             # Caries detection training/inference pipeline
|- week5/                             # Surface classification v1
|- week5-Surface Classification v1/   # Surface classification outputs
|- week6/                             # Surface classification v2 and evaluation scripts
|- week7/                             # Hardened evaluation/dashboard workflow
|- week7-Surface Classification v3/   # v3 outputs and artifacts
|- week8-Surface Classification v4/   # v4 outputs and artifacts
|- week9_pca_evaluation/              # PCA evaluation scripts/results
|- weekphase2/                        # Earlier phase2 notebook variant
|- _non-using/                        # Legacy/non-active experiments (~15.57 GB)
|- _organizer-reports/                # Inventory and duplicate analysis CSV/TXT reports
|- _trash/                            # Non-destructive trash staging (cache artifacts)
|- _unsorted/                         # Holding area for unknown/backup files pending review
```

## Key Files

| Path | Purpose |
|---|---|
| `phase2-1april/pipeline-phase1-v4.ipynb` | Main integration notebook for phase2 pipeline experimentation |
| `phase2-1april/report_v4.5.txt` | Evaluation summary for v4.5 zone strategy |
| `phase2-1april/report_v4.6.txt` | Evaluation summary for v4.6 zone strategy |
| `week2/process_500_cases.py` | Batch tooth segmentation and recognition over 500 cases |
| `week4/run_full_batch_v3_advanced.py` | Full caries detection batch runner (advanced settings) |
| `week5/process_surface_classification.py` | Surface classification batch processing |
| `week6/evaluation_engine.py` | Evaluation engine used by later weekly iterations |
| `outputs/diagnostics/tree-filtered-case-removed.txt` | Filtered tree output that suppresses noisy case folders |
| `_organizer-reports/inventory-all.csv` | Full filesystem inventory generated during organizer phase |
| `cleanup-log.md` | Action-by-action log of cleanup and restructuring |

## Setup and Usage

### 1. Environment setup (Conda recommended)

```powershell
conda create -n dental python=3.10 -y
conda activate dental
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

Optional week2 dependency file:

```powershell
pip install -r week2/requirements.txt
```

### 2. Typical execution entry points

Tooth segmentation over 500 cases:

```powershell
python week2/process_500_cases.py
```

Caries detection full batch (week4 pipeline):

```powershell
python week4/run_full_batch_v3_advanced.py
```

Surface classification batch:

```powershell
python week5/process_surface_classification.py
```

Notebook-based integration workflow:
- Open `phase2-1april/pipeline-phase1-v4.ipynb`
- Run cells in order after verifying paths for local data/model directories

## Data

### Data sources present in workspace
- `data/500 cases with annotation/` (about 1.25 GB): panoramic X-ray images + annotation files grouped per case
- `data/500-roi/` (about 0.004 GB): ROI masks for caries-related processing
- `data/Tooth Segmentation + Recognition model/` (about 0.372 GB): pretrained assets and model support files

### Expected format highlights
- Case folders follow patterns like `case 1`, `case 2`, etc.
- Per-case image naming commonly follows `case_<id>.png`
- XML annotations are co-located in case folders

### Storage warning
Large generated outputs already exist in this workspace (for example `week2-Tooth Detection & Segmentation/` is about 39.37 GB). Keep this in mind before cloning/copying or adding more derived artifacts.

## Results and Output Examples

### v4.5 summary (`phase2-1april/report_v4.5.txt`)
- Total samples: 1979
- Accuracy: 0.6741
- Precision: 0.4815
- Recall: 0.4555
- F1 score: 0.4626

### v4.6 summary (`phase2-1april/report_v4.6.txt`)
- Total samples: 1979
- Accuracy: 0.5306
- Precision: 0.5051
- Recall: 0.4364
- F1 score: 0.4139

### Main output locations
- `phase2-1april/PCA_Output_v4.5/`
- `phase2-1april/PCA_Output_v4.6/`
- `phase2-1april/caries_mapping_output/`
- `week7/` and `week7-Surface Classification v3/` evaluation dashboards/results

## Notes and Known Backlog

- Naming conventions are partially standardized for newly moved root files, but historical folder names with spaces were intentionally preserved to avoid breaking existing scripts.
- `phase2-1april/pipeline-phase1-v3_BACKUP.ipynb` was moved to `_unsorted/pipeline-phase1-v3_backup.ipynb` for manual review.
- `_organizer-reports/` contains high-volume duplicate/misplacement findings and should be reviewed before any aggressive deduplication.
- `_trash/` contains moved cache artifacts (`__pycache__` trees) for safety; nothing was permanently deleted.
