# Week 8 — Surface Classification v4 (Multi-PCA Method Evaluation)

## Overview

Week 8 inherits **all** Week 7 fixes and adds **configurable PCA method selection** from the Week 5 evaluation pipeline. This combines the best of both:

- **From Week 7**: Soft/partial surface matching, boundary erosion, unassigned caries detection, 3-rule PCA orientation
- **From Week 5**: Multiple PCA methods (1–5) for comparative evaluation

```text
Week 2 outputs ──┐
Material files ──┤──→  Week 8  ──→  Per-PCA-Method Metrics / Comparison Table
Week 5 outputs ──┤
Week 6 modules ──┘
```

## Supported PCA Methods

| Method | Name                   | Description |
|--------|------------------------|-------------|
| 1      | `square_heuristic`     | Eigenvalue ratio check for square-ish molars |
| 2      | `max_span`             | Maximum projected span along eigenvectors |
| 3      | `split_centroid`       | Anatomical vector from upper/lower centroids |
| 4      | *(placeholder)*        | Not implemented |
| 5      | `vertical_prior`       | Absolute vertical prior + 3-rule logic (default, same as week7) |

## Quick Start

```bash
# Test with default PCA method (5 = week7 behavior)
python run_test_sample.py

# Test with a specific PCA method
python run_test_sample.py --pca-method 1

# Compare all PCA methods on sample cases
python run_test_sample.py --compare-all
```

## Evaluation

```bash
# Single case evaluation
python evaluation_engine.py --case 311

# Specific cases with PCA method selection
python evaluation_engine.py --sample 311,33 --pca-method 2

# Full 500-case evaluation
python evaluation_engine.py --pca-method 5

# Compare ALL PCA methods across 500 cases
python evaluation_engine.py --compare-all
```

## Output Structure

```text
week8-Surface Classification v4/
├── evaluation_output/
│   ├── method_1_square_heuristic/
│   │   ├── evaluation_results.csv
│   │   ├── evaluation_summary.json
│   │   ├── confusion_matrix_coarse.png
│   │   └── confusion_matrix_fine.png
│   ├── method_2_max_span/
│   ├── method_3_split_centroid/
│   ├── method_5_vertical_prior/
│   └── pca_method_comparison.csv       ← summary across all methods
├── dental_analysis_output/
│   └── case N/
├── evaluation_engine.py
├── multi_zone_classifier.py
├── dental_caries_analysis.py
├── run_test_sample.py
├── xml_ground_truth_parser.py
├── snodent_tooth_map.py
└── README.md
```

## Bug Fixes (Inherited from Week 7)

| Task | Fix | Description |
|------|-----|-------------|
| 1 | PCA Eigenvector Swap | 3-rule orientation logic + angle clamp |
| 2 | Boundary Erosion | Morphological erosion + size/percentage threshold |
| 3 | Unassigned Caries | Fallback detection for missing upstream teeth |
| 4 | Soft Surface Match | Partial matching using zone fractions |
| 5 | Phantom FP Filter | Exclude teeth with `has_caries=False` from evaluation |

## New in Week 8

- **`--pca-method N`** flag on `evaluation_engine.py` and `run_test_sample.py`
- **`--compare-all`** flag to run all PCA methods and produce comparison CSV
- **`compare_all_methods()`** API function for programmatic comparison
- **`set_pca_method(N)` / `get_pca_method()`** in `multi_zone_classifier.py`
- Per-method output directories under `evaluation_output/`
- `pca_method` and `pca_method_name` columns in evaluation CSV
