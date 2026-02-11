# YOLO11 Dental Segmentation - Dataset Folder Structure

## Required Directory Layout

```
week1/
│
├── data.yaml                    # Dataset configuration (class names, paths)
├── train.py                     # Training script
├── prepare_dataset.py           # Dataset preparation script (XML to YOLO)
├── FOLDER_STRUCTURE.md          # This file
│
└── dataset/                     # Root dataset directory (referenced in data.yaml)
    │
    ├── images/                  # All images (.png)
    │   ├── train/               # Training images (80% = 400 cases)
    │   │   ├── case_1.png
    │   │   ├── case_2.png
    │   │   └── ...
    │   │
    │   └── val/                 # Validation images (20% = 100 cases)
    │       ├── case_101.png
    │       └── ...
    │
    └── labels/                  # Corresponding label files (.txt)
        ├── train/               # Training labels (same names as images)
        │   ├── case_1.txt
        │   ├── case_2.txt
        │   └── ...
        │
        └── val/                 # Validation labels
            ├── case_101.txt
            └── ...
```

## Dataset Statistics (After Preparation)

| Split      | Cases | Annotations | Percentage |
|------------|-------|-------------|------------|
| Training   | 400   | ~1548       | 80%        |
| Validation | 100   | ~431        | 20%        |
| **Total**  | **500** | **~1979** | **100%**   |
```

## Important Rules

### 1. File Naming Convention
- **Images and labels must have matching names** (different extensions)
- Example: `case_001.png` → `case_001.txt`

### 2. Image Requirements
- Format: PNG (recommended) or JPG
- Resolution: 3036 x 1536 pixels (original)
- Color: Grayscale or RGB (YOLO handles both)

### 3. Label File Format (YOLO Segmentation)
Each line in a `.txt` file represents one tooth instance:

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```

- `class_id`: Integer 0-31 (see FDI mapping below)
- Coordinates: Normalized polygon points (0.0 to 1.0)

**Example label file content:**
```
0 0.123 0.456 0.134 0.467 0.145 0.478 0.156 0.489 0.123 0.456
7 0.523 0.156 0.534 0.167 0.545 0.178 0.556 0.189 0.523 0.156
15 0.823 0.356 0.834 0.367 0.845 0.378 0.856 0.389 0.823 0.356
```

### 4. FDI Class Mapping (32 Classes)

| Class ID | FDI # | Tooth Name                      | Quadrant    |
|----------|-------|--------------------------------|-------------|
| 0        | 18    | Upper Right Third Molar        | Q1 (UR)     |
| 1        | 17    | Upper Right Second Molar       | Q1 (UR)     |
| 2        | 16    | Upper Right First Molar        | Q1 (UR)     |
| 3        | 15    | Upper Right Second Premolar    | Q1 (UR)     |
| 4        | 14    | Upper Right First Premolar     | Q1 (UR)     |
| 5        | 13    | Upper Right Canine             | Q1 (UR)     |
| 6        | 12    | Upper Right Lateral Incisor    | Q1 (UR)     |
| 7        | 11    | Upper Right Central Incisor    | Q1 (UR)     |
| 8        | 21    | Upper Left Central Incisor     | Q2 (UL)     |
| 9        | 22    | Upper Left Lateral Incisor     | Q2 (UL)     |
| 10       | 23    | Upper Left Canine              | Q2 (UL)     |
| 11       | 24    | Upper Left First Premolar      | Q2 (UL)     |
| 12       | 25    | Upper Left Second Premolar     | Q2 (UL)     |
| 13       | 26    | Upper Left First Molar         | Q2 (UL)     |
| 14       | 27    | Upper Left Second Molar        | Q2 (UL)     |
| 15       | 28    | Upper Left Third Molar         | Q2 (UL)     |
| 16       | 38    | Lower Left Third Molar         | Q3 (LL)     |
| 17       | 37    | Lower Left Second Molar        | Q3 (LL)     |
| 18       | 36    | Lower Left First Molar         | Q3 (LL)     |
| 19       | 35    | Lower Left Second Premolar     | Q3 (LL)     |
| 20       | 34    | Lower Left First Premolar      | Q3 (LL)     |
| 21       | 33    | Lower Left Canine              | Q3 (LL)     |
| 22       | 32    | Lower Left Lateral Incisor     | Q3 (LL)     |
| 23       | 31    | Lower Left Central Incisor     | Q3 (LL)     |
| 24       | 41    | Lower Right Central Incisor    | Q4 (LR)     |
| 25       | 42    | Lower Right Lateral Incisor    | Q4 (LR)     |
| 26       | 43    | Lower Right Canine             | Q4 (LR)     |
| 27       | 44    | Lower Right First Premolar     | Q4 (LR)     |
| 28       | 45    | Lower Right Second Premolar    | Q4 (LR)     |
| 29       | 46    | Lower Right First Molar        | Q4 (LR)     |
| 30       | 47    | Lower Right Second Molar       | Q4 (LR)     |
| 31       | 48    | Lower Right Third Molar        | Q4 (LR)     |

### 5. Recommended Data Split

| Split      | Percentage | Purpose                              |
|------------|------------|--------------------------------------|
| Training   | 80%        | Model learning                       |
| Validation | 15%        | Hyperparameter tuning, early stop    |
| Test       | 5%         | Final evaluation (hold-out)          |

For 500 cases: Train=400, Val=75, Test=25

---

## Quick Setup Commands (PowerShell)

```powershell
# Create folder structure
$base = "C:\Users\jaopi\Desktop\SP\week1\dataset"
New-Item -ItemType Directory -Force -Path "$base\images\train"
New-Item -ItemType Directory -Force -Path "$base\images\val"
New-Item -ItemType Directory -Force -Path "$base\images\test"
New-Item -ItemType Directory -Force -Path "$base\labels\train"
New-Item -ItemType Directory -Force -Path "$base\labels\val"
New-Item -ItemType Directory -Force -Path "$base\labels\test"
```

---

## Annotation Tool Compatibility

This structure is compatible with:
- **LabelMe** → Convert JSON to YOLO format
- **CVAT** → Export as "YOLO 1.1" segmentation
- **Roboflow** → Export as "YOLOv8" (works for v11)
- **Label Studio** → Use YOLO export format

---

## Visual Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                    PANORAMIC DENTAL X-RAY                       │
│                                                                 │
│  ┌─────────────────────────────┬─────────────────────────────┐  │
│  │      Q1 (Upper Right)       │      Q2 (Upper Left)        │  │
│  │   18 17 16 15 14 13 12 11   │   21 22 23 24 25 26 27 28   │  │
│  │   Classes: 0-7              │   Classes: 8-15             │  │
│  ├─────────────────────────────┼─────────────────────────────┤  │
│  │      Q4 (Lower Right)       │      Q3 (Lower Left)        │  │
│  │   48 47 46 45 44 43 42 41   │   31 32 33 34 35 36 37 38   │  │
│  │   Classes: 31-24            │   Classes: 23-16            │  │
│  └─────────────────────────────┴─────────────────────────────┘  │
│                                                                 │
│  ← Patient's Right                    Patient's Left →          │
└─────────────────────────────────────────────────────────────────┘
```

⚠️ **Note**: The view is as if you're facing the patient (mirror view).
