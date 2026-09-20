# 500 Cases Tooth Segmentation + Recognition Processing

This folder contains scripts to process all 500 panoramic dental X-ray cases using the Tooth Segmentation + Recognition model.

## Overview

The processing pipeline performs two-stage tooth analysis:
1. **Tooth Recognition (Pano-Seg)**: Uses YOLO model to detect and segment all teeth in the panoramic X-ray
2. **Tooth Segmentation (Crop-Seg)**: Uses Detectron2 model to perform detailed segmentation on each cropped tooth

## Files

| File | Description |
|------|-------------|
| `process_500_cases.py` | Main Python script that processes all 500 cases |
| `run_500_cases_processing.bat` | Batch file to run the processing (Windows) |
| `requirements.txt` | Python dependencies |
| `500-segmentation+recognition/` | Output directory (created automatically) |

## Requirements

### Conda Environment
Use the `dental` conda environment with the following packages:

### Python Packages
```
opencv-python
numpy
ultralytics
detectron2
tqdm
matplotlib
torch
torchvision
```

### Models
The script uses pre-trained models from `../material/Tooth Segmentation + Recognition model/weights/`:
- `Tooth_seg_pano_20250319.pt` - YOLO panoramic segmentation model
- `Tooth_seg_crop_20250424.pth` - Detectron2 cropped tooth segmentation model

## Usage

### Option 1: Run with conda (Recommended)
```powershell
cd week2
conda run -n dental python process_500_cases.py
```

### Option 2: Run with batch file (Windows)
```
run_500_cases_processing.bat
```

### Options
| Flag | Description |
|------|-------------|
| `--overwrite` | Reprocess all cases even if outputs already exist |

## Input Structure

The script expects the following structure:
```
material/
  500 cases with annotation/
    case 1/
      case_1.png
      *.xml files
    case 2/
      case_2.png
      *.xml files
    ...
```

## Output Structure

For each case, a folder is created with the case name (e.g., `case 1`, `case 2`, etc.) containing:

```
500-segmentation+recognition/
  case 1/
    case_1_bounding_boxes.png       # Visualization with bounding boxes and tooth IDs
    case_1_mask_overlay.png          # Visualization with segmentation mask overlay
    case_1_results.json              # Detection and segmentation data
  case 2/
    case_2_bounding_boxes.png
    case_2_mask_overlay.png
    case_2_results.json
  ...
```

### Output Files Per Case:

1. **Bounding Box Visualization** (`case_X_bounding_boxes.png`)
   - Original panoramic X-ray with green bounding boxes
   - Tooth IDs labeled on each detected tooth (FDI notation)
   - Title: "Bounding Boxes with Class ID"

2. **Mask Overlay Visualization** (`case_X_mask_overlay.png`)
   - Panoramic X-ray with colored segmentation masks overlaid
   - Each tooth segment shown in a random color with transparency
   - Title: "Mask Overlay on Panoramic Image (No Labels, No Boxes)"

3. **Results JSON** (`case_X_results.json`)
   - Case number
   - Number of teeth detected
   - For each tooth:
     - Tooth ID (FDI notation)
     - Detection confidence
     - Bounding box coordinates
     - Crop coordinates
     - Number of segmentation masks

### Example JSON Output
```json
{
  "case_number": "1",
  "num_teeth_detected": 28,
  "teeth_data": [
    {
      "tooth_id": "11",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2],
      "crop_coords": [x1, y1, x2, y2],
      "num_segments": 1
    },
    ...
  ]
}
```

## Processing Pipeline

### 1. Model Initialization
- Load YOLO model for panoramic segmentation
- Load Detectron2 model for detailed tooth segmentation

### 2. For Each Case:
   a. Read panoramic X-ray image
   b. **Panoramic Segmentation**:
      - Detect all teeth using YOLO
      - Extract bounding boxes and masks
      - Remove duplicate detections (IoU > 0.5)
   c. **Detailed Segmentation**:
      - Crop each detected tooth (with padding)
      - Apply Detectron2 segmentation
      - Extract detailed tooth structures
   d. **Save Results**:
      - Create bounding box visualization (green boxes with tooth IDs)
      - Create mask overlay visualization (colored segmentation masks)
      - Save detection and segmentation data to JSON
      - All files saved in case-specific folder (e.g., `case 1/`, `case 2/`)

### 3. Generate Summary
- Report total cases processed
- Count successful and failed cases

## Features

- **Resume Support**: Skips already-processed cases (checks for all 3 output files)
- **Clean Output**: Suppresses verbose YOLO detection logs
- **Batch Processing**: Processes all 500 cases automatically
- **Progress Tracking**: Shows progress bar with per-case status
- **Error Handling**: Continues processing even if individual cases fail
- **Duplicate Filtering**: Removes duplicate detections based on IoU > 0.5
- **Numerical Sorting**: Cases process in order (1, 2, 3... not 1, 10, 100...)
- **GPU Acceleration**: Uses CUDA for Detectron2 inference

## Notes

- Processing time: ~4 seconds per case on GPU
- Each case typically contains 20-32 teeth
- The script uses FDI World Dental Federation notation (ISO 3950) for tooth numbering
- Padding of 20 pixels is added around each tooth crop for better segmentation
- Uses `Agg` matplotlib backend to prevent Tkinter thread crashes on Windows

## Troubleshooting

### CUDA out of memory
- Close other applications using GPU
- The script processes one image at a time, so memory usage should be minimal

### No teeth detected
- Check if the image file exists and is readable
- Verify model weights are present and valid

### Detectron2 installation issues
- Use the `dental` conda environment which has detectron2 pre-installed
- On Windows, may require Visual Studio Build Tools

### Tkinter/Thread errors
- The script forces `Agg` backend to prevent this
- If errors persist, ensure no other matplotlib imports happen before the script

## Expected Output

After successful processing:
- 500 case folders (`case 1/`, `case 2/`, ..., `case 500/`)
- Each folder contains:
  - `case_X_bounding_boxes.png` - Visualization with bounding boxes
  - `case_X_mask_overlay.png` - Visualization with segmentation masks
  - `case_X_results.json` - Detection and segmentation data
- Console summary showing: newly processed, skipped (existed), and failed counts
