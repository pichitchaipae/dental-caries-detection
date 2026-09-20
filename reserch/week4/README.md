# Week 4: Dental Caries Detection Pipeline

## 📋 Project Overview

Week 4 implements a **Two-Stage Dental Caries Detection System** with instance segmentation capabilities. The pipeline processes panoramic dental X-rays to:
1. Detect caries lesions (3 types: Occlusal, Proximal, Lingual)
2. Identify and segment individual teeth
3. Map caries to specific teeth with FDI notation

---

## 🔄 Version History

| Version | Description | Key Features |
|---------|-------------|--------------|
| **v1.0** | Basic inference | YOLO caries + Static tooth map |
| **v2.0** | Dynamic model | YOLO tooth detection + Box containment matching |
| **v3.0** | Instance Segmentation | Detectron2 fine segmentation + Pixel coordinates |

---

## 📁 File Structure & Scripts

### Core Pipeline Scripts

| Script | Version | Description |
|--------|---------|-------------|
| [inference.py](inference.py) | **v3.0** | Main detection pipeline with Detectron2 integration |
| [standardize_week4.py](standardize_week4.py) | v3.0 | Convert raw output to Week 2/3 compatible schema |
| [viz_advanced.py](viz_advanced.py) | v1.0 | Advanced matplotlib visualization with polygon rendering |

### Dataset Preparation Scripts

| Script | Description |
|--------|-------------|
| [prepare_caries_dataset.py](prepare_caries_dataset.py) | Extract caries annotations from XML → YOLO format |
| [dataset_converter.py](dataset_converter.py) | Convert 96-class labels → 3-class simplified format |
| [train_caries.py](train_caries.py) | Train YOLOv8s caries detection model |

### Batch Processing Scripts

| Script | Description |
|--------|-------------|
| [run_full_batch_v3_advanced.py](run_full_batch_v3_advanced.py) | **Production** - Process all 500 cases with full pipeline |
| [run_full_batch_v3.py](run_full_batch_v3.py) | Legacy batch processor (test version) |

---

## 🏗️ Pipeline Architecture (v3.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Panoramic X-ray                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Caries Detection (YOLO v8s)                           │
│  ─────────────────────────────────────                          │
│  Model: runs/caries_train/caries_3class/weights/best.pt         │
│  Classes: 0=Occlusal, 1=Proximal, 2=Lingual                     │
│  Output: BBox + Class + Confidence                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Tooth Detection & Segmentation                        │
│  ─────────────────────────────────────────                      │
│  Step 2a: YOLO Panoramic Detection (Tooth_seg_pano_20250319.pt) │
│           → Coarse tooth boxes + FDI IDs (11-48)                │
│                                                                 │
│  Step 2b: Crop Individual Tooth Regions                         │
│           → Extract each tooth with 20% padding                 │
│                                                                 │
│  Step 2c: Detectron2 Fine Segmentation (Tooth_seg_crop.pth)     │
│           → Real tooth contours (polygon coordinates)           │
│                                                                 │
│  Step 2d: Map Local → Global Coordinates                        │
│           → pixel_coordinates: [[x,y], [x,y], ...]              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Caries-to-Tooth Matching                              │
│  ─────────────────────────────────────                          │
│  Strategy (3-tier):                                             │
│    1. Box Containment (caries center inside tooth box)          │
│    2. IoU Overlap (intersection > 0.1)                          │
│    3. Nearest Center (Euclidean distance fallback)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                         │
│  ─────────────────────────────────────                          │
│  • JSON with caries_detections + tooth_detections               │
│  • pixel_coordinates for real tooth shapes                      │
│  • Readable class names (not Class_0)                           │
│  • Visualization overlay (optional)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Script Documentation

### 1. inference.py (v3.0)

**Two-Stage Dental Caries Detection with Instance Segmentation**

```bash
# Basic usage
python inference.py -i "image.png" -m "caries_model.pt"

# Full pipeline with Detectron2
python inference.py \
    -i "image.png" \
    -m "runs/caries_train/caries_3class/weights/best.pt" \
    --tooth_model "path/to/Tooth_seg_pano_20250319.pt" \
    --crop_model "path/to/Tooth_seg_crop_20250424.pth" \
    -c 0.01 \
    -o "./output"

# Skip fine segmentation (faster, bounding boxes only)
python inference.py -i "image.png" -m "model.pt" --no_fine_seg

# Batch processing
python inference.py --image_dir "./images" -m "model.pt" -o "./results"
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --image_path` | - | Path to input image |
| `--image_dir` | - | Directory for batch processing |
| `-m, --model_path` | **Required** | Path to caries YOLO model |
| `--tooth_model` | Auto-detect | Path to tooth YOLO model |
| `--crop_model` | Auto-detect | Path to Detectron2 crop model |
| `-c, --caries_conf` | 0.05 | Caries confidence threshold |
| `--tooth_conf` | 0.5 | Tooth confidence threshold |
| `-o, --output_dir` | ./inference_output | Output directory |
| `--no_visualize` | False | Disable visualization |
| `--no_fine_seg` | False | Skip Detectron2 (use boxes only) |
| `--use_static_only` | False | Use static map fallback only |
| `--show` | False | Display results in window |

**Output JSON Schema (v3.0):**

```json
{
  "image_path": "case_1.png",
  "image_size": {"width": 3036, "height": 1536},
  "localization_method": "DYNAMIC_MODEL",
  "num_caries": 5,
  "num_teeth_detected": 29,
  "caries_detections": [
    {
      "detection_id": 1,
      "class_id": 1,
      "class_name": "Proximal",
      "confidence": 0.87,
      "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 250},
      "tooth_id": "46",
      "tooth_name": "Lower Right First Molar",
      "localization_method": "box_containment",
      "iou_with_tooth": 0.73
    }
  ],
  "tooth_detections": [
    {
      "tooth_id": "46",
      "tooth_name": "Lower Right First Molar",
      "confidence": 0.92,
      "bbox": {"x1": 80, "y1": 180, "x2": 200, "y2": 300},
      "pixel_coordinates": [[85, 190], [90, 185], ...]
    }
  ]
}
```

---

### 2. standardize_week4.py (v3.0)

**Convert Week 4 output to Week 2/3 compatible schema**

```bash
# Convert single directory
python standardize_week4.py -s "./raw_output" -o "./standardized"

# With custom case number extraction
python standardize_week4.py -s "./input" -o "./output"
```

**Features:**
- Preserves `pixel_coordinates` for tooth segmentation
- Converts class names to readable format (never "Class_0")
- Generates caries_mapping.json per case
- Adds model info and summary statistics

**Output Schema:**

```json
{
  "case_number": 1,
  "source": "week4_inference_v3",
  "model_info": {
    "caries_model": "YOLOv8s 3-class",
    "tooth_model": "YOLO panoramic",
    "crop_seg_model": "Detectron2 Mask R-CNN"
  },
  "teeth_caries_data": [
    {
      "tooth_id": "46",
      "tooth_name": "Lower Right First Molar",
      "caries_type": "Proximal",
      "confidence": 0.87,
      "has_caries": true,
      "pixel_coordinates": [[x,y], ...]
    }
  ],
  "summary": {
    "total_teeth": 29,
    "teeth_with_caries": 5,
    "total_caries_detections": 7,
    "num_teeth_with_segmentation": 29
  }
}
```

---

### 3. viz_advanced.py (v1.0)

**Advanced Visualization with Real Tooth Contours**

```bash
# Generate visualization
python viz_advanced.py \
    -j "case_1_caries_mapping.json" \
    -i "case_1.png" \
    -o "case_1_overlay.jpg"

# High-res output without display
python viz_advanced.py -j "data.json" -i "image.png" -o "out.jpg" --dpi 200 --no_show
```

**Features:**
- Draws real tooth polygons from `pixel_coordinates`
- Color-coded caries overlays by type
- Clean medical report styling
- Fallback to bounding boxes if no coordinates

**Color Palette:**

| Element | Color | Alpha |
|---------|-------|-------|
| Tooth Fill | Dark Cyan (#00CED1) | 0.3 |
| Tooth Edge | Dark Cyan (#008B8B) | 1.0 |
| Occlusal Caries | Red (#FF4444) | 0.6 |
| Proximal Caries | Orange-Red (#FF6B00) | 0.6 |
| Lingual Caries | Deep Pink (#FF1493) | 0.6 |

---

### 4. run_full_batch_v3_advanced.py (v3.0)

**Production Batch Processing - 500 Cases**

```bash
# Process all 500 cases (skip existing)
python run_full_batch_v3_advanced.py

# Reprocess all cases
python run_full_batch_v3_advanced.py --no-skip

# Process single case
python run_full_batch_v3_advanced.py --case 42
```

**Pipeline per case:**
1. **Inference** → Raw JSON with Detectron2 segmentation
2. **Standardization** → Convert to legacy schema
3. **Visualization** → Generate advanced overlay image

**Output Structure:**

```
final_advanced_output/
├── case 1/
│   ├── case_1.png                    (Original Image)
│   ├── case_1_caries_mapping.json    (Standardized JSON)
│   └── case_1_advanced_viz.jpg       (Visualization)
├── case 2/
│   └── ...
├── _raw_inference/                   (Intermediate outputs)
│   ├── case_1_results.json
│   └── ...
└── batch_processing_log.json         (Run statistics)
```

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Caries Confidence | 0.01 (1%) |
| Tooth Confidence | 0.5 |
| Input | material/500 cases with annotation |
| Output | week4/final_advanced_output |

---

### 5. train_caries.py

**Train YOLOv8s Caries Detection Model**

```bash
python train_caries.py
```

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8s |
| Epochs | 50 |
| Image Size | 640 |
| Batch Size | 16 |
| Classes | 3 (Occlusal, Proximal, Lingual) |

**Class Distribution Notes:**
- Proximal: ~86% (dominant)
- Occlusal: ~14% (minority)
- Lingual: ~0% (empty in dataset)

---

### 6. prepare_caries_dataset.py

**Extract Caries Annotations from XML**

```bash
python prepare_caries_dataset.py
```

**Converts:**
- Input: `material/500 cases with annotation/case X/*.xml`
- Output: `week4/dataset_3class/` (YOLO format)

---

### 7. dataset_converter.py

**Convert Multi-class to 3-class Format**

```bash
python dataset_converter.py \
    -i "./original_labels" \
    -o "./converted_labels" \
    -m "data.yaml" \
    --create_yaml
```

**Mapping:**
- `*_Occlusal` → Class 0
- `*_Proximal`, `*_Mesial`, `*_Distal` → Class 1
- `*_Lingual`, `*_Buccal`, `*_Palatal` → Class 2

---

## 🔧 Models Used

| Model | Path | Purpose |
|-------|------|---------|
| Caries YOLO | `runs/caries_train/caries_3class/weights/best.pt` | 3-class caries detection |
| Tooth YOLO | `material/.../Tooth_seg_pano_20250319.pt` | Panoramic tooth detection |
| Tooth Detectron2 | `material/.../Tooth_seg_crop_20250424.pth` | Fine tooth segmentation |

---

## 📊 Output Directories

| Directory | Description |
|-----------|-------------|
| `final_advanced_output/` | **Production** - Full pipeline results |
| `test_v3_standardized/` | Testing - V3.0 pipeline validation |
| `inference_output/` | Legacy - Basic inference results |
| `dataset_3class/` | Training dataset (YOLO format) |
| `runs/` | Model training outputs |

---

## 🚀 Quick Start

### Run Full 500-Case Pipeline

```bash
# Activate environment
conda activate sp_project

# Navigate to week4
cd C:\Users\jaopi\Desktop\SP\week4

# Run batch processing
python run_full_batch_v3_advanced.py
```

### Single Image Inference

```bash
python inference.py \
    -i "../material/500 cases with annotation/case 1/case_1.png" \
    -m "runs/caries_train/caries_3class/weights/best.pt" \
    -c 0.01
```

---

## 📝 Changelog

### v3.0 (2025-01-27)
- Added Detectron2 instance segmentation for real tooth contours
- Added `pixel_coordinates` export for visualization compatibility
- Improved class name handling (never "Class_0")
- Added `run_full_batch_v3_advanced.py` for production batch processing
- Created `viz_advanced.py` for matplotlib polygon rendering

### v2.0 (2025-01-26)
- Integrated YOLO tooth detection model
- Added dynamic caries-to-tooth matching
- Implemented 3-tier matching strategy (containment → IoU → nearest)

### v1.0 (2025-01-25)
- Initial two-stage pipeline
- Static coordinate map for tooth localization
- Basic YOLO caries detection

---

## 👥 Authors

- Lead CV Engineer
- MLOps Pipeline Engineer
- Data Visualization Expert
