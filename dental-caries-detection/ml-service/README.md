# Dental Caries Detection ML Service

This directory contains the FastAPI inference service and the scripts used to prepare, train, and evaluate the crop-based caries segmentation model. The production pipeline combines two existing tooth models, one fine-tuned caries model, and a deterministic geometry rule for assigning a detected lesion to a tooth surface.

```text
Panoramic radiograph
  → Pano-Seg YOLO: detect teeth and assign FDI numbers
  → deduplicate teeth by FDI and bounding-box IoU
  → crop each tooth with a 20-pixel margin
  → Crop-Seg Detectron2: refine the target tooth mask
  → suppress neighbouring teeth in the crop
  → Caries YOLOv8-seg: predict a caries mask (one class)
  → transform lesion geometry from crop coordinates to panorama coordinates
  → PCA-aligned geometry heuristic: assign mesial, occlusal, or distal
  → encode findings in result.json for the backend and frontend
```

The caries network predicts **one class, `caries`**. It does not predict `mesial`, `occlusal`, or `distal`; the surface name is derived after detection from lesion position, the tooth mask, FDI quadrant, and PCA alignment. XML polygons and surface names are training/evaluation labels and are not inference inputs.

## 1. Model roles and current artifacts

| Pipeline component | Current model or method | Classes | Trainable in this workflow? | Role |
|---|---|---:|---|---|
| Pano-Seg | `Tooth_seg_pano_20250319.pt` (Ultralytics YOLO segmentation) | 32 FDI tooth classes | No | Finds teeth on the full panorama and supplies tooth identity, mask, and crop location. |
| Crop-Seg | `Tooth_seg_crop_20250424.pth` (Detectron2 Mask R-CNN R50-FPN) | 1 tooth class | No | Refines the target tooth mask inside each crop. |
| Caries segmentation | `caries_crop_seg_20261028.pt` (YOLOv8s-seg) | 1 `caries` class | Yes, by fine-tuning a pretrained YOLO segmentation checkpoint | Finds and segments caries on an individual tooth crop. |
| Surface assignment | PCA-aligned deterministic geometry heuristic | mesial / occlusal / distal output labels | Not a learned model | Maps a detected lesion centroid to a surface zone. |

The first two models are existing upstream dependencies. The caries training script fine-tunes only the caries detector; it does not retrain Pano-Seg or Crop-Seg. The production model manifest, including checksums, is [models/versions.py](models/versions.py).

| Registry key | Artifact | SHA-256 | Notes |
|---|---|---|---|
| `pano_detector` | `weights/Tooth_seg_pano_20250319.pt` | `502bd58099e50d5c5bb16245aa5a135943fbd87debe9f66928a5b938798ca579` | YOLO segmentation, 32 FDI classes. |
| `crop_segmenter` | `weights/Tooth_seg_crop_20250424.pth` | `b39fef8288634e6e756d3aace628d145d0f927a76d3f7d9bfa17f919a8c9b4a9` | Mask R-CNN R50-FPN; test score threshold 0.50, resize shortest edge 800, maximum edge 1333. |
| `caries_detector` | `weights/caries_crop_seg_20261028.pt` | `c0d4768fdbd90567feb28c5df34fa760321e79a08acbb47e11ba84a5c2c67d74` | YOLOv8s-seg, one class, input size 640. |

Weight files are deliberately not committed. `weights/.gitignore` ignores every weight except `.gitkeep`; Docker Compose mounts the local weight directory into the service. The registry verifies model hashes when loading artifacts.

## 2. Repository map

```text
ml-service/
├── app/
│   ├── main.py                         FastAPI endpoints: /infer, /cancel, /health
│   ├── config.py                       Runtime environment variables and defaults
│   ├── db.py                           Job-state database access
│   ├── runner.py                       Starts and manages the inference child process
│   └── pipeline/
│       ├── orchestrator.py             Runs inference stages and assembles results
│       ├── detection.py                 Pano-Seg, dedupe, crop, Crop-Seg, and caries YOLO
│       ├── coordinate_transform.py      Crop/panorama coordinate transforms
│       ├── crop_utils.py                Polygon clipping and neighbour suppression
│       ├── pca_alignment.py             PCA axes for tooth orientation
│       ├── surface_heuristic.py         PCA-aligned surface assignment
│       ├── numbering.py                 FDI tooth-number mapping
│       ├── postprocess.py               Result JSON and geometry encoding
│       ├── xml_annotations.py            AIM XML parser for labels/evaluation
│       ├── surface_classification.py     Legacy/reference-only surface classifier
│       └── surface_cnn.py                Reference-only future CNN option
├── models/
│   ├── registry.py                      Loads models and checks artifacts
│   └── versions.py                      Production artifact names, settings, and hashes
├── scripts/
│   ├── prepare_caries_crop_dataset.py    Creates crop images and YOLO-seg labels
│   ├── train_caries_crop_seg.py          Fine-tunes, validates, and exports caries YOLO
│   ├── evaluate_caries_pipeline.py       Evaluates full serving pipeline on held-out cases
│   ├── validate_surface_heuristic.py     Checks geometry surface direction against XML
│   └── audit_inference_parity.py         Offline inference parity diagnostics
├── tests/                                Unit/API checks and reference-only experiments
├── notebooks/                            Review and practical pipeline notebooks
├── datasets/                             Generated datasets; ignored by Git
├── runs/                                 Training/evaluation artifacts; ignored by Git
├── weights/                              Local runtime/training weights; ignored by Git
├── archive/                              Local archived experiments; excluded from image
├── Dockerfile                            Python 3.10 inference image
├── .dockerignore                         Copies runtime code, not training assets
├── requirements.txt                      Runtime Python dependencies
└── requirements-detectron2.txt           Linux CPU Detectron2 wheel reference

../outputs/
├── validation-20261029.json              Crop validation metrics
└── notebook-final-review-20261029/
    ├── fresh_full_evaluation.json         Latest 100-case end-to-end validation
    └── audit_summary.json                 Counts, environment, and audit conclusions
```

`Dockerfile` copies `app/` and `models/` only. It does not include `scripts/`, datasets, runs, notebooks, or weights. Dataset preparation, training, and evaluation therefore run in a separate host/WSL Python environment; the container is for inference.

## 3. Data inputs and target format

### Source data

The preparation script expects one directory per case, conventionally:

```text
data/500 cases with annotation/
├── case 1/
│   ├── case_1.png
│   └── *.xml
├── case 2/
│   ├── case_2.png
│   └── *.xml
└── ...
```

Each PNG is a panoramic radiograph. The AIM XML files provide lesion polygons, tooth identifiers, and surface labels. The parser reads annotation structures and reports malformed or unsupported entries. The generated model input is an image crop produced from model-detected teeth; the XML polygon is clipped into that crop and converted to a YOLO segmentation polygon with normalized coordinates in `[0, 1]`.

| Information | Used as caries-model input? | Use |
|---|---:|---|
| Tooth crop pixels | Yes | Image presented to YOLOv8-seg. |
| Pano-Seg tooth mask and crop geometry | Indirectly | Defines the crop and context passed to the model. |
| Crop-Seg tooth mask | Indirectly | Refines the target tooth region before suppression. |
| XML lesion polygon / ROI | No | Converted to the segmentation target during preparation; used as GT in evaluation. |
| XML surface name | No | Used only for surface evaluation. |
| FDI / tooth number | No class input | Used to group crops and match detections during evaluation. |

The single YOLO label class is `0 caries`. Surface labels are not encoded as additional YOLO classes. PCA is not a feature vector for caries training: it is used later to align the tooth mask and locate a predicted lesion relative to the tooth's horizontal extent.

### Dataset produced

```text
datasets/caries_crop_v1/
├── data.yaml                 names: [caries], train and val image paths
├── manifest.json             split, arguments, model hashes, per-case counts
├── images/
│   ├── train/*.png           tooth crops with neighbour teeth suppressed
│   └── val/*.png
├── labels/
│   ├── train/*.txt           YOLO segmentation polygons; class 0 = caries
│   └── val/*.txt
└── qc/*.png                  optional labelled visual checks
```

An empty label file is a negative crop. Negative crops are sampled from teeth without XML lesions; a tooth with an XML lesion that is not visible in its crop is skipped instead of incorrectly becoming a negative. The split is by **case**, before making crops, so crops from one case cannot appear in both train and validation.

## 4. Step 1 — Build the dataset

Run from the `ml-service` directory. In PowerShell:

```powershell
Set-Location 'D:\Mahidol University\SP\dental-caries-detection\ml-service'

python scripts/prepare_caries_crop_dataset.py `
  --images-root 'D:\Mahidol University\SP\data\500 cases with annotation' `
  --weights-dir '.\weights' `
  --out '.\datasets\caries_crop_v1' `
  --crop-seg on `
  --val-fraction 0.20 `
  --seed 42 `
  --detection-threshold 0.25 `
  --crop-pad 20 `
  --dedupe-iou 0.50 `
  --negatives-per-case 3 `
  --min-lesion-area 4 `
  --device 0 `
  --viz 30
```

`--crop-seg on` is important for train/serve parity: it requires the Detectron2 refinement stage. `auto` may silently choose pano masks when Detectron2 cannot load; `off` explicitly uses pano masks and is suitable only for wiring experiments, not a production-equivalent dataset. The Dockerfile's Detectron2 wheel targets Linux/Python 3.10. For local preparation, use a compatible Linux/WSL environment or a verified local Detectron2 installation.

### Dataset preparation parameters

| Argument | Script default | Production dataset value | Meaning |
|---|---:|---:|---|
| `--images-root` | Required | 500-case annotated data directory | Searches `case N` folders and their PNG/XML files. |
| `--weights-dir` | `weights` | `weights` | Must contain the registered Pano-Seg and Crop-Seg artifacts. |
| `--out` | Required | `datasets/caries_crop_v1` | Output dataset directory; a non-empty directory is protected unless `--overwrite` is set. |
| `--val-fraction` | `0.20` | `0.20` | Fraction of cases held out for validation. |
| `--seed` | `42` | `42` | Seed for deterministic negative sampling. The case split uses `seed + 1`, so split seed is `43`. |
| `--limit` | `0` (all cases) | `0` | Optional small subset for a smoke run. Do not use a limited subset for final training. |
| `--detection-threshold` | `0.25` | `0.25` | Pano-Seg confidence threshold used when finding teeth. |
| `--crop-pad` | `20` | `20` | Padding in panorama pixels around each tooth crop, clamped to image bounds. |
| `--dedupe-iou` | `0.50` | `0.50` | Duplicate detections with the same FDI are merged when their bbox IoU exceeds this value. |
| `--crop-seg` | `auto` | `on` | `on` requires crop refinement; `auto` can fall back; `off` skips refinement. |
| `--device` | `cpu` | `cuda` / `0` | Device for YOLO and Detectron2 during preparation. |
| `--negatives-per-case` | `3` | `3` | Maximum sampled lesion-free tooth crops per case. This is a deliberate sampling limit, not all negative teeth. |
| `--min-lesion-area` | `4.0` | `4.0` | Minimum clipped lesion polygon area in crop pixels. Smaller clipped annotations are skipped. |
| `--viz` | `0` | `30` | Save up to this many positive crop QC images with polygons drawn. |
| `--overwrite` | Off | Off | Allow rebuilding a non-empty output folder. |

### Actual 500-case dataset

The recorded production preparation used 500 cases: **400 train cases and 100 validation cases**. It produced 3,243 crops (2,617 train, 626 validation), including 1,743 positive crops and 1,500 sampled negative crops. The saved crop labels contain 1,892 lesion instances. The script skipped 48 lesions outside the crop and 34 positive teeth with no visible lesion in their crop; 11,870 other negative candidate teeth were not sampled. The full counts and input artifact hashes are in `datasets/caries_crop_v1/manifest.json` (local generated file).

## 5. Step 2 — Pretraining and transfer learning

### What “pretrained” means here

Training starts with an Ultralytics segmentation checkpoint, not random weights. The production run used the `yolov8s-seg.pt` starting checkpoint (recorded as `/runs/pretrained/yolov8s-seg.pt` in the run arguments). When this checkpoint is not found at the supplied path, Ultralytics can fetch the checkpoint by its model name. Keep the exact checkpoint file and its hash with each future experiment if strict reproducibility is required.

### What is transferred and what changes

1. Load the pretrained YOLO segmentation model: `YOLO("yolov8s-seg.pt")`.
2. Load the generated `data.yaml`, which must declare exactly one class named `caries`. The training script rejects other class lists.
3. Initialize compatible model layers from pretrained weights and adapt the detection/segmentation class output to the one-class target. The production run recorded `pretrained: true` and `cls_remap: true`.
4. Fine-tune the model on crop pixels and XML-derived caries polygons. `freeze: null` means no layers were frozen; the model's compatible weights were updated during fine-tuning.
5. Validate during training and keep the best checkpoint according to Ultralytics training fitness/early stopping.

This is transfer learning for **caries localization and mask prediction**. It does not transfer surface classes from the pretrained model, and it does not train the separate PCA geometry heuristic.

## 6. Step 3 — Train and export a candidate model

```powershell
python scripts/train_caries_crop_seg.py `
  --data '.\datasets\caries_crop_v1\data.yaml' `
  --model 'yolov8s-seg.pt' `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --device 0 `
  --project '.\runs\caries_crop_seg' `
  --name 'candidate' `
  --patience 20 `
  --seed 42 `
  --workers 8 `
  --export-to '.\weights\caries_crop_seg_candidate.pt'
```

The training script checks the dataset class name, invokes Ultralytics segmentation training, validates on `val`, copies `best.pt` when `--export-to` is provided, calculates the exported file's SHA-256, and prints a suggested registry entry. It does **not** edit `models/versions.py` automatically.

### Training parameters: defaults and recorded production run

The CLI defaults are convenient for a CPU/smoke run and are **not** all the values used for the shipped model. The production run used the explicit values below, recorded in `runs/caries_crop_seg/v1/args.yaml`:

| Parameter | Script default | Shipped run | Practical meaning |
|---|---:|---:|---|
| `--model` | `yolov8n-seg.pt` | `yolov8s-seg.pt` | Pretrained segmentation architecture/checkpoint. `s` is larger than `n`. |
| `--epochs` | `100` | `100` maximum; stopped at epoch 88 | Maximum passes over training data. Early stopping ended the run. |
| `--imgsz` | `640` | `640` | Model input resolution in pixels. Training and serving use 640 for the caries model. |
| `--batch` | `8` | `16` | Images/crops per optimization step; constrained by GPU memory. |
| `--device` | `cpu` | `0` (GPU) | Ultralytics device selector. |
| `--patience` | `30` | `20` | Stop after this many epochs without improvement in the monitored fitness. |
| `--seed` | `42` | `42` | Random seed for training. |
| `--workers` | `4` | `8` | Data-loader worker processes. |
| `--project`, `--name` | `runs/caries_crop_seg`, `train` | `runs/caries_crop_seg`, `v1` | Run output directory. |
| `pretrained` | Ultralytics default | `true` | Start from pretrained weights. |
| `freeze` | Ultralytics default | `null` | No layers frozen in the recorded run; fine-tune all compatible layers. |
| `optimizer` | `auto` | `auto`, resolved to AdamW | Optimizer selection. |
| `lr0`, `lrf` | Ultralytics defaults | `0.01`, `0.01` | Initial learning rate and final LR fraction. |
| `weight_decay` | Ultralytics default | `0.0005` | Regularization applied by the optimizer. |
| `warmup_epochs` | Ultralytics default | `3` | Warm-up period for learning rate/momentum. |
| `deterministic`, `amp` | Ultralytics defaults | `true`, `true` | Deterministic operations where supported; automatic mixed precision on GPU. |
| `mosaic`, `close_mosaic` | Ultralytics defaults | `1.0`, `10` | Mosaic augmentation probability and epochs before the end when it is disabled. |
| `translate`, `scale`, `fliplr` | Ultralytics defaults | `0.1`, `0.5`, `0.5` | Geometric augmentations. |
| `hsv_h`, `hsv_s`, `hsv_v` | Ultralytics defaults | `0.015`, `0.7`, `0.4` | Hue, saturation, and value augmentation settings. |

Other saved loss/augmentation values are visible in `runs/caries_crop_seg/v1/args.yaml`. The current script exposes the common run settings above as CLI arguments; settings not exposed there are supplied by the installed Ultralytics version. Save the full `args.yaml`, package versions, base checkpoint, dataset manifest, and hashes for each new experiment. `--crop-pad` affects only the script's printed description; actual crop size is controlled in dataset preparation.

### Training outputs

```text
runs/caries_crop_seg/<run-name>/
├── args.yaml                  complete resolved run settings
├── results.csv                per-epoch losses and validation metrics
├── results.png                training curves
├── confusion_matrix*.png      validation class summaries
├── Mask* / Box* curves        precision/recall curves
└── weights/
    ├── best.pt                checkpoint selected by training validation
    └── last.pt                checkpoint from the last completed epoch
```

The shipped run stopped at epoch 88 of 100 with `patience=20`. A separate fresh validation run on all 626 validation crops reported mask precision 0.5373, recall 0.5084, mAP50 0.4843, and mAP50-95 0.1573. These are crop-level segmentation metrics, not end-to-end panorama metrics.

## 7. Step 4 — Evaluate the model

Use two complementary evaluations. Crop-level validation measures the learned caries segmenter on prepared crops. Full-pipeline evaluation measures Pano-Seg, Crop-Seg, the caries model, and surface assignment together on held-out panoramic cases.

### A. Crop-level segmentation validation

```powershell
python -c "from ultralytics import YOLO; m=YOLO(r'.\weights\caries_crop_seg_20261028.pt'); r=m.val(data=r'.\datasets\caries_crop_v1\data.yaml', split='val', imgsz=640, device=0); print(r.results_dict)"
```

For segmentation metrics, Ultralytics reports `M` (mask) and `B` (box) results. The mask mAP50 is the primary reported measure here; mAP50-95 averages over stricter IoU thresholds. Precision and recall depend on the validation prediction threshold. The recorded crop validation used **626 held-out crops and 354 saved lesion instances**.

### B. End-to-end evaluation on held-out panoramas

The evaluation script runs the real tooth-detection and caries pipeline on the case IDs marked `val` in `manifest.json`. It compares predicted lesion masks with raw XML lesion polygons for the same FDI tooth. A prediction can match one unmatched GT lesion if mask IoU is at least `--match-iou`; matching is one-to-one and greedy in GT order. This is not the same procedure as Ultralytics mAP.

```powershell
python scripts/evaluate_caries_pipeline.py `
  --dataset '.\datasets\caries_crop_v1' `
  --images-root 'D:\Mahidol University\SP\data\500 cases with annotation' `
  --weights-dir '.\weights' `
  --caries-weights '.\weights\caries_crop_seg_20261028.pt' `
  --device 0 `
  --detection-threshold 0.25 `
  --collect-conf 0.01 `
  --match-iou 0.30 `
  --thresholds '0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60' `
  --out '..\outputs\caries_full_pipeline_eval.json'
```

`--collect-conf` is deliberately low so one inference pass captures candidates for the offline threshold sweep. `--detection-threshold` is the pano tooth detector threshold; `--match-iou` is the minimum lesion-mask overlap used to count a TP. The script chooses the threshold with the highest F1 from the supplied sweep. If the validation set is used to choose this operating threshold, the reported result remains a validation/tuning estimate.

### Latest recorded evaluation

The latest full-pipeline validation artifact is `../outputs/notebook-final-review-20261029/fresh_full_evaluation.json`. It evaluates all **100 held-out cases** and **373 raw XML lesions** at caries confidence `0.25`, pano confidence `0.25`, and mask IoU match threshold `0.30`:

| Lesion detection metric | Result |
|---|---:|
| Ground-truth lesions | 373 |
| TP / FP / FN | 246 / 219 / 127 |
| Precision | 52.90% |
| Recall | 65.95% |
| F1 | 58.71% |
| Surface agreement, among matched lesions only | 217 / 246 = 88.21% |

The surface confusion matrix for the **246 matched lesions** is shown as GT rows and predicted surface columns:

| GT ↓ / Pred → | Mesial | Occlusal | Distal |
|---|---:|---:|---:|
| Mesial | 78 | 0 | 0 |
| Occlusal | 5 | 30 | 10 |
| Distal | 1 | 13 | 109 |

**Interpretation:** 88.21% is conditional surface agreement after a lesion has already been matched. It is not overall caries accuracy, not a per-case success rate, and not the caries detector's precision. Lesion detection itself has 219 unmatched predictions and 127 missed GT lesions at this operating point. The 100 cases are the validation partition; the project does not currently have a separate untouched test partition. Since the validation data are used during training and to select an operating threshold, a future final estimate should use new disjoint cases.

The crop-level validation has 354 labeled lesion instances while the raw panorama evaluation has 373 XML lesions; those figures use different inputs and evaluation scopes. Do not combine them into one denominator. The older `outputs/final_500_production_metrics.json` belongs to a previous pipeline/metric definition and should not be presented as the current full-pipeline score. The older 79.7% surface-direction check also used a separate 120-case/443-lesion heuristic study and a different pano inference resolution; it is not the same evaluation as the 88.21% conditional result above.

## 8. PCA and surface assignment

The tooth mask is aligned to its principal axes. The predicted lesion centroid is then projected into the aligned tooth frame and normalized to a relative horizontal position `x_rel`. With the current boundaries, left of 0.30 and right of 0.70 are proximal zones; the middle zone is occlusal. The mapping is mirrored for the patient's left/right quadrants using FDI.

```text
FDI Q1 / Q4: x_rel < 0.30 → distal; 0.30–0.70 → occlusal; > 0.70 → mesial
FDI Q2 / Q3: x_rel < 0.30 → mesial; 0.30–0.70 → occlusal; > 0.70 → distal
```

This rule does not learn from lesion surface classes and cannot use pixels to distinguish all occlusal from proximal lesions. The boundary settings are `SURFACE_LEFT_BOUND=0.30` and `SURFACE_RIGHT_BOUND=0.70`. The current serving response includes `surfaces[].probability`; this field carries the highest raw YOLO detection confidence mapped to that surface. It is not a calibrated probability that the whole surface is carious.

## 9. Runtime configuration and API

| Environment variable | Default | Purpose |
|---|---:|---|
| `ML_SERVICE_HOST` / `ML_SERVICE_PORT` | `0.0.0.0` / `8001` | FastAPI bind address and port. |
| `SHARED_DIR` | `/shared` | Shared input/result files exchanged with the backend. |
| `WEIGHTS_DIR` | `/weights` | Mounted model artifacts. |
| `ML_DEVICE` | `cpu` | Runtime device setting. |
| `DETECTION_THRESHOLD` | `0.25` | Pano-Seg tooth detection confidence threshold. |
| `ENABLE_CROP_SEGMENTER` | `true` | Enables required crop-mask refinement in the target pipeline. |
| `CROP_PAD` | `20` | Panorama pixels added around each detected tooth bbox. |
| `DEDUPE_IOU` | `0.50` | Same-FDI duplicate bbox threshold. |
| `SUPPRESS_NEIGHBOURS` | `true` | Masks neighbouring teeth before caries inference. |
| `CARIES_CONF` | `0.25` | Caries YOLO confidence operating threshold. |
| `CARIES_IOU` | `0.50` | Caries YOLO non-maximum suppression IoU. |
| `SURFACE_LEFT_BOUND` / `SURFACE_RIGHT_BOUND` | `0.30` / `0.70` | PCA-frame surface zone boundaries. |

The service exposes `POST /infer` with `{ "jobId": ... }`, `POST /cancel`, and `GET /health`. Inference is initiated by the backend, which places the image in the shared directory; `/infer` is not a direct image-upload endpoint. The `/health` artifact report should show all three production model entries as `verified` and `caries_enabled: true`. `ready` primarily reflects the tooth pipeline and should not be used alone to prove that the caries model loaded.

For result shape and the frontend probability field, see [the API contract](../docs-md/api-contract-v1.md) and [the pipeline v2 report](../docs-md/pipeline-v2-caries-crop.md).

## 10. Install and run the inference service

### Docker Compose

From the project root:

```powershell
Set-Location 'D:\Mahidol University\SP\dental-caries-detection'
docker compose up -d --build ml-service
Invoke-RestMethod 'http://localhost:8001/health' | ConvertTo-Json -Depth 6
```

The image uses Python 3.10, Torch 2.4.1, and torchvision 0.19.1. Detectron2 is installed from the CPU wheel for Linux/Python 3.10/Torch 2.4 in `Dockerfile`; it is not a normal PyPI dependency. The image runs as a non-root user and expects weights to be mounted at `/weights`.

### Local development

```powershell
Set-Location 'D:\Mahidol University\SP\dental-caries-detection\ml-service'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Install a Detectron2 build compatible with the local OS, Python, and Torch.
$env:DATABASE_URL = 'sqlite:///./dev.db'
$env:WEIGHTS_DIR = "$PWD\weights"
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Setting `ENABLE_CROP_SEGMENTER=false` is available for limited local debugging when Detectron2 is unavailable; it changes the inference pipeline and should not be used to claim parity with production.

## 11. Register and deploy a retrained checkpoint

1. Train with a unique run name and export to a **candidate** filename. Keep `best.pt`, `args.yaml`, `results.csv`, the dataset `manifest.json`, and the base checkpoint used.
2. Review QC crops and validation predictions. Compare both crop metrics and full-pipeline case-level metrics against the current baseline.
3. Verify the exported candidate SHA-256 printed by the script. Copy the candidate into the mounted `weights/` directory under a versioned name.
4. Update `caries_detector` in `models/versions.py` with the new filename, SHA-256, version, architecture, class list, and description. Keep `num_classes: 1` and `classes: ["caries"]`.
5. Restart/rebuild the service and check `/health` for `caries_detector: verified` and `caries_enabled: true`.
6. Run an end-to-end validation before switching the production alias or presenting the new result.

Do not overwrite the current production checkpoint before the candidate has been registered and validated. Keep previous artifacts recoverable for rollback.

## 12. Reproducibility notes and known limits

- **No independent test set:** the 500 cases are split 400/100 into train/validation. There is no separate test partition in the current dataset.
- **Validation crop distribution differs from inference:** preparation samples only three negative teeth per case, whereas serving processes all detected teeth. The manifest records 1,500 sampled negatives and 11,870 unsampled negative candidates. This is a known distribution difference; its causal effect should be tested with a controlled negative/hard-negative experiment.
- **Training and runtime package versions differ:** the audited training/evaluation environment recorded Python 3.10.19, Torch 2.5.1, torchvision 0.20.1, and Ultralytics 8.4.175. The Dockerfile installs Torch 2.4.1/torchvision 0.19.1, and the audited Docker runtime used Ultralytics 8.4.191. `requirements.txt` currently says `ultralytics>=8.3.0`, so a fresh install may resolve a different version. Pin and record the exact dependency set when reproducing a result.
- **Crop-Seg preprocessing matters:** Detectron2 uses configured resize and color handling before inference. Keep the serving implementation, dataset preparation, and evaluation preprocessing aligned; see the detailed notes in `../docs-md/pipeline-v2-caries-crop.md`.
- **Confidence is not clinical calibration:** YOLO confidence is a detector score. The API forwards it as `probability` for the mapped surface but does not calibrate it as a surface disease probability.
- **Surface heuristic limitations:** the assignment is based on centroid geometry. Occlusal/proximal boundary cases can be mislabeled even when the lesion mask itself is detected correctly.
- **Do not call the 88.21% value end-to-end accuracy:** it is calculated only over matched lesions, whereas the end-to-end lesion metrics are precision, recall, and F1.

## 13. Tests

The runtime-related unit tests are:

```powershell
Set-Location 'D:\Mahidol University\SP\dental-caries-detection\ml-service'
python -m pytest `
  tests/test_coordinate_transform.py `
  tests/test_surface_heuristic.py `
  tests/test_detection.py `
  tests/test_xml_annotations.py `
  tests/test_crop_utils.py `
  tests/test_postprocess.py -q
```

`tests/test_surface_cnn_parity.py`, `tests/bench_surface_cnn.py`, and `tests/e2e_offline_eval.py` relate to the reference-only CNN path and are not the current production evaluation suite.
