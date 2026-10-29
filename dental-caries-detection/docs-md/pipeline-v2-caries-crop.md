# Pipeline v2 — caries on tooth crops

Status: **implemented and serving.** A crop-trained 1-class caries model was
built on 2026-10-28 from all 500 annotated cases and is pinned in
`models/versions.py` (`caries-crop-seg-20261028`). `/health` now reports
`caries_detector=verified` and `caries_enabled=true`. Measured results are in
§4.4; limitations are in §9. Crop-Seg is verified end-to-end after fixing the
missing Detectron2 preprocessing (§7.1).

This document replaces the assumptions behind the rejected `week4` model and
the RF 13-feature classifier.

---

## 1. Target flow

```text
panoramic image
  │
  ├─ Pano-Seg YOLO ──────────────► FDI + bbox + mask per tooth
  │    Tooth_seg_pano_20250319.pt       (32 classes)
  │
  ├─ dedupe ─────────────────────► one detection per FDI
  │    same FDI and bbox IoU > DEDUPE_IOU (0.50)
  │
  ├─ crop PAD 20 ────────────────► tooth crop, clamped to the image
  │
  ├─ Crop-Seg Detectron2 ────────► refined tooth mask
  │    Tooth_seg_crop_20250424.pth      (1 class, score_thresh 0.50)
  │
  ├─ suppress other teeth ───────► black out neighbours inside the crop
  │    target mask dilated 15 px
  │
  ├─ caries YOLO-seg ────────────► 1 class: "caries", on the crop only
  │
  ├─ crop → pano transform ──────► lesion bbox, polygon, centroid
  │
  └─ surface heuristic ──────────► mesial / occlusal / distal
       PCA-aligned x position in the tooth mask
```

Implementation:

| Stage | File |
|---|---|
| Pano detection, dedupe, crop, Crop-Seg, caries on crops | `ml-service/app/pipeline/detection.py` |
| pano ↔ crop transforms | `ml-service/app/pipeline/coordinate_transform.py` |
| neighbour suppression, polygon clipping | `ml-service/app/pipeline/crop_utils.py` |
| deterministic surface rule | `ml-service/app/pipeline/surface_heuristic.py` |
| result.json encoding | `ml-service/app/pipeline/postprocess.py` |
| orchestration | `ml-service/app/pipeline/orchestrator.py` |

There is **no caries-to-tooth ownership heuristic**. A lesion detected inside a
tooth crop belongs to that tooth by construction.

---

## 2. Why the old models are gone

### 2.1 `week4` / `caries_detect.pt`

The production `caries_detect.pt` was byte-identical to
`_non-using/week4/runs/caries_train/caries_3class/weights/best.pt`
(SHA-256 `515e09fe…8dc1fd`). Week 4 was rejected by the project owner, so it
cannot be used. It was also a 3-class pano-space detector
(`Occlusal`/`Proximal`/`Lingual`), which does not match the crop-based
pipeline.

### 2.2 RF 13-feature surface classifier

`dental_caries_surface_random_forest_13_features.pkl` takes ROI-derived
geometry as input (`x_mean`, `coverage`, `x_centroid_dist`, …). Using ROI
coordinates as a model input is disallowed for this project; ROI/XML may only
be used as **training labels**. The RF was removed from the serving path.

Both artifacts are quarantined (not deleted) in
`_non-using/deprecated-serving-models/`, with the original paths, checksums,
and reasons in its `README.md`. The `ml-service` does not read that directory.

### 2.3 CNN surface model

`surface_cnn.py` and `cnn_pca42/` are kept as a documented **future option**
(Decision Log: `cnn_pca42` is not primary). `surface_cnn.py` and
`surface_classification.py` are now marked reference-only and are no longer
imported by the orchestrator. The old CNN end-to-end harness
(`tests/e2e_offline_eval.py`) is reference-only too.

---

## 3. Input contract

| Data | Used as model input? | Used as label? |
|---|---|---|
| Panoramic image pixels | yes | — |
| Pano-Seg tooth mask/bbox | yes | — |
| Crop-Seg tooth mask | yes | — |
| Caries lesion ROI / XML polygon | **no** | yes, training only |
| XML tooth name / surface name | no | yes, training/evaluation only |

At inference the caries model sees only the cropped tooth image (with other
teeth suppressed). The surface heuristic reads the predicted lesion centroid
and the predicted tooth mask; both come from image pixels.

---

## 4. Caries model: 1 class

The only output class is `caries`. The model predicts a mask and bbox; it does
not predict the surface. This avoids the rejected 3-class taxonomy and keeps
the label space identical across quadrants and surfaces.

Serving refuses a multi-class checkpoint:

```text
ValueError: caries_detector must be a 1-class model ('caries');
got 3 classes: [...]
```

### 4.1 Dataset generation

```bash
cd ml-service
python scripts/prepare_caries_crop_dataset.py \
  --images-root "D:/Mahidol University/SP/data/500 cases with annotation" \
  --weights-dir weights \
  --out datasets/caries_crop_v1 \
  --crop-seg on \
  --val-fraction 0.2 \
  --negatives-per-case 3 \
  --viz 24
```

Per case, the script runs the **same** Stage 1 order as serving:

1. Pano-Seg → dedupe → crop PAD 20;
2. Crop-Seg refinement (`--crop-seg on` is required for train/serve parity);
3. neighbour suppression;
4. XML lesion polygon → clip to crop → normalise to `0..1`;
5. write the crop image and a YOLO-seg label file (`class 0`).

Rules:

- The split is by **case**, not by crop.
- A positive tooth whose lesion is not visible in the crop is skipped, never
  saved as a negative.
- Negative crops are sampled per case (`--negatives-per-case`); they contain no
  caries label.
- `--crop-seg off` exists only for smoke tests. It uses pano masks and is **not**
  train/serve parity; do not train the shipping model that way.
- `manifest.json` records the arguments, model checksums, per-case counts, and
  data issues.

### 4.2 Training

```bash
cd ml-service
python scripts/train_caries_crop_seg.py \
  --data datasets/caries_crop_v1/data.yaml \
  --model yolov8s-seg.pt \
  --epochs 100 --imgsz 640 --batch 8 \
  --device 0 \
  --export-to weights/caries_crop_seg_YYYYMMDD.pt
```

The script checks that `data.yaml` declares exactly one class named `caries`,
trains, validates, exports the checkpoint, prints the SHA-256, and prints the
exact `models/versions.py` entry to add.

### 4.3 Enabling the model

Add the printed entry under `caries_detector` in
`ml-service/models/versions.py`:

```python
"caries_detector": {
    "version": "caries-crop-seg-YYYYMMDD",
    "filename": "caries_crop_seg_YYYYMMDD.pt",
    "sha256": "<printed digest>",
    "architecture": "yolo_segment",
    "num_classes": 1,
    "classes": ["caries"],
    "description": "1-class caries YOLO-seg trained on tooth crops (PAD 20)",
},
```

Then restart `ml-service` and check:

```bash
curl -s localhost:8001/health | jq
# {
#   "ready": true,
#   "caries_enabled": true,
#   "artifacts": {
#     "pano_detector": "verified",
#     "crop_segmenter": "verified",
#     "caries_detector": "verified"
#   }
# }
```

Until a checkpoint is pinned, `caries_enabled=false` and
`caries_detector="not_configured"`. `ready` covers the tooth pipeline only
(pano + Crop-Seg, or Crop-Seg deliberately disabled); it does not require
caries.

### 4.4 Production run — measured results (2026-10-28)

**Dataset** (`caries_crop_v1`, GPU, `--crop-seg on`, 1,113 s for 500 cases):

| item | value |
| --- | --- |
| cases used | 500 (400 train / 100 val) |
| crops | 3,243 (2,617 train, 626 val) |
| positive crops | 1,743 (1,417 train, 326 val) |
| negative crops | 1,500 (1,200 train, 300 val) |
| XML lesions used | 1,892 |
| skipped: lesion outside tooth crop | 48 |
| skipped: positive tooth with no visible lesion | 34 |
| XML issues | 0 |

**Training** (`yolov8s-seg`, 640 px, batch 16, RTX 4080 Laptop):

- 88/100 epochs before early stop (`patience=20` firing is on fitness, not on
  the mask metric alone); 2,367 s.
- Final mask metrics on the 626-crop val split: **mAP50 0.484**, mAP50-95 0.157,
  precision 0.537, recall 0.508. Box: mAP50 0.563, mAP50-95 0.212.
- Artifact: `caries_crop_seg_20261028.pt`, 23,829,933 bytes,
  SHA-256 `c0d4768fdbd90567feb28c5df34fa760321e79a08acbb47e11ba84a5c2c67d74`.
  Verified by `/health` before loading.

**Held-out lesion evaluation** (`scripts/evaluate_caries_pipeline.py`, the 100
val cases, full serving pipeline, all 30 teeth per case scored, GT = XML):

| conf | TP | FP | FN | precision | recall | F1 | surface acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 286 | 600 | 87 | 0.323 | 0.767 | 0.454 | 89.2% |
| 0.15 | 270 | 400 | 103 | 0.403 | 0.724 | 0.518 | 88.5% |
| 0.20 | 258 | 309 | 115 | 0.455 | 0.692 | 0.549 | 88.8% |
| **0.25** | **246** | **222** | **127** | **0.526** | **0.659** | **0.585** | **88.2%** |
| 0.30 | 220 | 168 | 153 | 0.567 | 0.590 | 0.578 | 86.8% |
| 0.40 | 158 | 83 | 215 | 0.656 | 0.424 | 0.515 | 84.2% |
| 0.50 | 105 | 36 | 268 | 0.745 | 0.281 | 0.409 | 80.9% |

`CARIES_CONF=0.25` is the best-F1 point and is what ships. Matching is greedy
per FDI at mask IoU ≥ 0.30 against 373 XML lesions.

Surface accuracy at conf 0.25 (only matched lesions, 245 with a mapped label):
mesial 78/78 (100%), distal 109/122 (89.3%), occlusal 30/45 (66.7%).

**Latency** (backend `POST /process` → `GET /process` done, CPU-only service
container, 29–31 teeth per case): 40.4–46.2 s per panoramic image.

**End-to-end spot checks** (through the backend, not the ML API directly):

| case | GT lesions | predicted | exact (FDI+surface) | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 10 | 2 | 2 | 1 | 1 (FDI 45 distal) | 1 (FDI 27 mesial) |
| 20 | 9 | 9 | 7 | 1 (FDI 37 occlusal) | 1 (FDI 26 mesial) |
| 26 | 5 | 5 | 4 | 1 (FDI 13 mesial) | 1 (FDI 37 mesial) |
| 42 | 4 | 1 | 0 | 1 (FDI 48 mesial) | 4 |

Lesion *count* per case is usually well calibrated (cases 10/20/26), but case 42
shows the failure mode: one unrelated prediction and all four GT lesions missed.
The residual error is tooth-level localisation and confidence, consistent with
the 0.53/0.66 P/R — not a per-case guarantee.

---

## 5. Surface heuristic

The heuristic works in the tooth's PCA-aligned frame, computed by the same
`perform_pca()` used for `teeth[].axes` (Stage 2). Rotation removes tooth tilt,
so a tilted tooth does not shift a mesial lesion into the occlusal zone.

```text
x_rel = (lesion_x − tooth_x_min) / (tooth_x_max − tooth_x_min)

FDI Q1 / Q4 (patient's right)
    x_rel < 0.30          → distal
    0.30 ≤ x_rel ≤ 0.70   → occlusal
    x_rel > 0.70          → mesial

FDI Q2 / Q3 (patient's left) — mirrored
    x_rel < 0.30          → mesial
    0.30 ≤ x_rel ≤ 0.70   → occlusal
    x_rel > 0.70          → distal
```

Bounds are configurable (`SURFACE_LEFT_BOUND`, `SURFACE_RIGHT_BOUND`).

### 5.1 Why the direction is not flipped

Validated with `scripts/validate_surface_heuristic.py` against the AIM XML
annotations:

- 120 cases, 455 labelled lesions;
- Pano-Seg at `imgsz=1024`, `conf=0.25`;
- 443 lesions matched to a detected tooth (97.4%);
- same PCA-aligned `x_rel` as serving.

Zone hit rates (left = `x_rel < 0.30`, mid, right = `x_rel > 0.70`):

| Quadrant | Surface | n | mean x_rel | left | mid | right |
|---|---|---:|---:|---:|---:|---:|
| Q1 | mesial | 41 | 0.592 | 0.12 | 0.05 | **0.83** |
| Q1 | occlusal | 18 | 0.612 | 0.06 | **0.72** | 0.22 |
| Q1 | distal | 65 | −0.067 | **0.85** | 0.14 | 0.02 |
| Q2 | mesial | 56 | 0.277 | **0.93** | 0.04 | 0.04 |
| Q2 | occlusal | 19 | 0.935 | 0.05 | **0.74** | 0.21 |
| Q2 | distal | 71 | 0.862 | 0.11 | 0.14 | **0.75** |
| Q3 | mesial | 31 | 0.383 | **0.87** | 0.10 | 0.03 |
| Q3 | occlusal | 15 | 0.538 | 0.27 | **0.47** | 0.27 |
| Q3 | distal | 40 | 1.024 | 0.00 | 0.20 | **0.80** |
| Q4 | mesial | 31 | 0.727 | 0.06 | 0.16 | **0.77** |
| Q4 | occlusal | 13 | 0.578 | 0.23 | **0.38** | 0.38 |
| Q4 | distal | 43 | 0.135 | **0.86** | 0.09 | 0.05 |

Overall 3-way agreement: **353/443 = 79.7%**.

What this does and does not mean:

- **Direction is confirmed**: Q1/Q4 distal is on the left, mesial on the right;
  Q2/Q3 are mirrored. This holds for every quadrant.
- **Proximal recall is usable**: mesial zone hit rate 77–93%, distal 75–86%.
- **Occlusal is the weak class** (38–74% mid-zone). A lesion centroid cannot
  reliably distinguish a central occlusal lesion from a proximal lesion whose
  centroid sits near the middle of the crown. This is a property of the
  deterministic rule, not of the XML labels.
- The measurement uses XML lesion centroids, not model predictions. It
  validates the direction/label convention; it is not a clinical accuracy
  estimate for the future caries model.

Reproduce:

```bash
cd ml-service
python scripts/validate_surface_heuristic.py \
  --images-root "D:/Mahidol University/SP/data/500 cases with annotation" \
  --weights-dir weights --limit 120 \
  --out surface_validation.json
```

---

## 6. Output contract

Each tooth may carry a `caries` array in pano coordinates (additive in API
v1.1):

```json
{
  "id": 0,
  "fdi": 46,
  "bbox": [10, 20, 40, 80],
  "mask": { "encoding": "polygon", "data": [[...]] },
  "axes": { "major": [...], "minor": [...], "rotation_deg": 0.0, "clamped": false },
  "surfaces": [
    {
      "name": "distal",
      "label": "caries",
      "probability": 0.77,
      "method": "Geometry_Heuristic",
      "x_rel": 0.12
    }
  ],
  "caries": [
    {
      "bbox": [11, 30, 8, 12],
      "mask": { "encoding": "polygon", "data": [[...]] },
      "confidence": 0.77,
      "centroid": [15, 34]
    }
  ]
}
```

- `bbox` is `[x, y, w, h]`; masks are polygons; both are pano coordinates.
- `surfaces[].probability` copies the caries YOLO confidence of the lesion
  assigned to that surface. If multiple lesions map to one surface, the
  highest detector confidence is used. It is not calibrated as the probability
  that the whole surface is carious.
- `surfaces` is deduplicated to one finding per surface, ordered
  mesial → occlusal → distal.
- `meta.models.classifier` is the caries filename, or `not_configured`.
- `meta.caries_lesion_count` counts lesions, while `meta.caries_count` counts
  teeth with at least one surface finding.
- The raw result also carries a top-level `surface_source: "geometry_heuristic"`
  for provenance. The backend adapter drops it; clients use
  `surfaces[].method` instead.

The backend adapter (`backend/src/services/resultAdapter.ts`) and frontend
schema (`frontend/src/domain/inference.ts`) accept `Geometry_Heuristic` and the
optional `caries` array. Historical results with `RF`/`CNN`/`XThirds_Fallback`
still parse.

---

## 7. Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `ENABLE_CROP_SEGMENTER` | `true` | run Crop-Seg; `false` only for local debugging |
| `CROP_PAD` | `20` | pano pixels added around the tooth bbox |
| `DEDUPE_IOU` | `0.50` | duplicate bbox IoU for the same FDI |
| `SUPPRESS_NEIGHBOURS` | `true` | black out other detected teeth in the crop |
| `CARIES_CONF` | `0.25` | caries YOLO confidence; recalibrate after training |
| `CARIES_IOU` | `0.50` | caries YOLO NMS IoU |
| `SURFACE_LEFT_BOUND` | `0.30` | left zone boundary |
| `SURFACE_RIGHT_BOUND` | `0.70` | right zone boundary |
| `DETECTION_THRESHOLD` | `0.25` | Pano-Seg confidence |

Crop-Seg runs as Detectron2 Mask R-CNN R50-FPN 3x with
`MODEL.ROI_HEADS.NUM_CLASSES=1`, `SCORE_THRESH_TEST=0.5`, and `cfg.freeze()`.
The Dockerfile installs the CPU wheel for torch 2.4.1/cp310. Detectron2 is not
on PyPI, so it is not in `requirements.txt`; local installs use
`requirements-detectron2.txt`.

### 7.1 Crop-Seg preprocessing is part of the contract

Detectron2 models do **not** resize their input internally. The reference
loader (`caries-detection-model/pipeline.py`) calls `DefaultPredictor`, which
performs two steps before the forward pass:

1. BGR → RGB (`image[:, :, ::-1]`);
2. `ResizeShortestEdge(MIN_SIZE_TEST=800, MAX_SIZE_TEST=1333)`.

The first implementation of `refine_with_crop_segmenter()` called
`model([{...}])` directly, skipping both steps. On `case 1` this produced
**0/29** Crop-Seg instances at `SCORE_THRESH_TEST=0.5`, so every tooth silently
fell back to its pano mask. A controlled comparison on the same crops showed:

| Input to the model | Top score (8 sampled teeth) |
|---|---|
| raw BGR crop, no resize (the bug) | 0.000 for all |
| RGB crop, no resize | 0.000 for all |
| BGR crop, resized to shortest edge 800 | 0.988–0.999 |
| RGB crop, resized to shortest edge 800 | 0.988–0.999 |

For these grayscale radiographs the channel swap does not change the score, but
serving now performs it anyway to match the reference exactly. The current
implementation:

- applies RGB + `ResizeShortestEdge` from the bundle built in
  `models/registry.py` (`MIN_SIZE_TEST`/`MAX_SIZE_TEST` are pinned in
  `models/versions.py`);
- maps `pred_masks` from the resized model space back to the original crop with
  nearest-neighbour before converting to pano coordinates;
- logs `crop_seg refined N/M tooth masks`, and emits a **warning** when `N=0`
  so a silent whole-image fallback is visible.

Verified after the fix: `case 1` → 29/29, `case 2` → 32/32, `case 3` → 27/27
masks refined. Refinement is slow on CPU: about 36–43 s per case (all tooth
crops) on the Docker CPU image.

---

## 8. Tests

```bash
cd ml-service
python -m pytest \
  tests/test_coordinate_transform.py \
  tests/test_surface_heuristic.py \
  tests/test_detection.py \
  tests/test_xml_annotations.py \
  tests/test_crop_utils.py \
  tests/test_postprocess.py -q

DATABASE_URL=sqlite:///./test.db \
WEIGHTS_DIR="$PWD/weights" \
ENABLE_CROP_SEGMENTER=false \
python -m pytest tests/test_api.py -q
```

Covered:

- pano ↔ crop transforms and bbox IoU;
- dedupe by FDI + IoU;
- caries predictions transformed back to pano coordinates;
- Crop-Seg receives RGB, resized model input and its mask is mapped back to the
  crop before the pano transform;
- rejection of multi-class caries checkpoints;
- crop-first ordering of the caries stage;
- PCA-aligned mesial/occlusal/distal direction and tilt handling;
- XML parsing (ROI only, decoy markup ignored, dedupe, issue reporting);
- neighbour suppression and margin preservation;
- `caries` output encoding;
- backend/frontend schema acceptance of `Geometry_Heuristic` + `caries`.

---

## 9. Known gaps / next steps

1. **Negative sampling is the dominant error source.** The dataset sampled only
   3 negative teeth per case (1,500 negative crops vs 1,743 positive), but
   serving runs the model on *every* tooth (~30/case, ~85% unannotated).
   At conf 0.25 that asymmetry shows up as 222 false positives against 246 true
   positives. Retraining with 8–10 negatives per case (or hard-negative mining
   on FP teeth) is the highest-value next step; it changes only
   `prepare_caries_crop_dataset.py --negatives-per-case`.
2. **Tooth-level localisation, not lesion count, is the weak axis.** Per-case
   lesion counts match GT closely, but the errors are adjacent-FDI swaps
   (case 10: 45 instead of 27; case 20: 37 instead of 26). This is inherited
   from Pano-Seg FDI assignment, not from the caries model, and should be
   addressed in dedupe/FDI logic rather than by retraining caries.
3. **Labels are XML-derived, so the ceiling is measured against annotation
   noise.** Lesions are rasterised from the XML polygons intersected with the
   Crop-Seg mask; 48 lesions fell outside their tooth crop and 34 positive teeth
   had no visible lesion and were skipped. Unannotated early caries counts as a
   false positive in §4.4, so the reported precision is a lower bound.
4. **Heuristic limits.** Occlusal recall is weak because the rule only sees a
   centroid. If the caries model later predicts multiple points per lesion, or
   a surface classifier is trained on crop pixels, the heuristic can be
   replaced without changing the model input contract.
5. **Reference-only code.** `surface_classification.py`, `surface_cnn.py`,
   `cnn_pca42/`, `tests/bench_surface_cnn.py`,
   `tests/test_surface_cnn_parity.py`, and `tests/e2e_offline_eval.py` are not
   in the serving path.
