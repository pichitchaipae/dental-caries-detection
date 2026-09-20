# Dental Caries Detection System - Main Scripts

## Overview
This folder contains the main production scripts for the dental caries detection pipeline with **HIGH ACCURACY** configuration.

## High Accuracy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `caries_confidence` | 0.01 | Very low threshold to maximize recall |
| `tooth_confidence` | 0.35 | Lower threshold for better tooth coverage |
| `image_size` | 1280 | Larger image for better detection |
| `augment` | True | Test-time augmentation enabled |
| `caries_iou_threshold` | 0.3 | Lower IoU for NMS to keep more detections |
| `min_iou_for_match` | 0.05 | Minimum IoU for caries-tooth matching |

## Main Scripts

### 1. `inference.py` - Core Inference Engine
The main inference script with high accuracy settings.

```bash
# Single image
python inference.py -i "path/to/image.png" -m "path/to/caries_model.pt" -t "path/to/tooth_model.pt"

# Batch processing
python inference.py --image_dir "path/to/images" -m "path/to/model.pt" -o "output_dir"

# Standard accuracy mode (faster)
python inference.py -i "image.png" -m "model.pt" --standard
```

### 2. `run_full_batch_v3.py` - Full 500 Cases Pipeline
Complete pipeline for processing all 500 cases:
1. Inference (V3.1 with high accuracy)
2. Standardization (Legacy format)
3. Visualization (Advanced overlay)

```bash
python run_full_batch_v3.py
```

### 3. `run_full_inference.py` - Batch Inference Only
Runs inference on all 500 cases without standardization.

```bash
python run_full_inference.py
```

### 4. `viz_advanced.py` - Advanced Visualization
Creates medical-style visualization overlays.

```bash
python viz_advanced.py --case_path "path/to/case_folder" --case_num 1
```

### 5. `viz_week2_style.py` - Week 2 Style Visualization
Visualization compatible with Week 2 JSON format.

```bash
python viz_week2_style.py --case_path "path/to/results.json"
```

## Output Structure

```
final_500_structured/
├── case 1/
│   ├── case_1.png              # Original image
│   ├── case_1_caries_mapping.json   # Standardized results
│   └── case_1_overlay.png      # Visualization
├── case 2/
│   └── ...
└── case 500/
    └── ...
```

## Model Paths (Default)

- **Caries Model**: `../runs/caries_train/caries_3class/weights/best.pt`
- **Tooth Model**: `../../material/Tooth Segmentation + Recognition model/weights/Tooth_seg_pano_20250319.pt`
- **Crop Model**: `../../material/Tooth Segmentation + Recognition model/weights/Tooth_seg_crop_20250424.pth`

## Pipeline Stages

1. **Stage 1**: YOLO Caries Detection (3-class: Occlusal/Proximal/Lingual)
2. **Stage 2a**: YOLO Panoramic Tooth Detection
3. **Stage 2b**: Crop tooth regions with padding (25px)
4. **Stage 2c**: Detectron2 fine segmentation (if available)
5. **Stage 2d**: Map local coordinates to global space

## Caries Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Occlusal | Caries on chewing surface |
| 1 | Proximal | Caries on side surfaces |
| 2 | Lingual | Caries on tongue-facing surface |

## Version History

- **v3.1** (2025-01-28): High Accuracy Configuration
  - Lowered confidence thresholds for better recall
  - Added test-time augmentation
  - Increased image size to 1280
  - Improved caries-tooth matching logic
