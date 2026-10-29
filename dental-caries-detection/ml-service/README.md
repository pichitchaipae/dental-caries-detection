# Dental Caries Detection — ML Service

FastAPI microservice for the v2 pipeline:

```text
Pano-Seg YOLO → dedupe → crop PAD 20 → Crop-Seg (Detectron2)
→ 1-class caries YOLO-seg on the tooth crop → crop→pano transform
→ deterministic mesial/occlusal/distal heuristic
```

The caries checkpoint is listed in `models/versions.py` and enabled:
`caries_detector: "verified"`, `caries_enabled: true` on `/health`. If the entry
is removed, the service degrades to tooth segmentation and reports
`caries_detector: "not_configured"`.

Full design, empirical surface validation, and known gaps:
[`docs-md/pipeline-v2-caries-crop.md`](../docs-md/pipeline-v2-caries-crop.md).

## Technology

- FastAPI + Uvicorn
- Ultralytics YOLO (pano segmentation + 1-class caries segmentation)
- Detectron2 Mask R-CNN R50-FPN (Crop-Seg)
- OpenCV / NumPy
- PostgreSQL (job status only)

## Project structure

```text
ml-service/
├── app/
│   ├── main.py                 FastAPI app: POST /infer, POST /cancel, GET /health
│   ├── config.py               Runtime configuration
│   ├── db.py                   Job state database access
│   ├── runner.py               Child-process lifecycle
│   └── pipeline/
│       ├── detection.py        Stage 1: pano → dedupe → crop → Crop-Seg → caries
│       ├── coordinate_transform.py  pano ↔ crop transforms
│       ├── crop_utils.py       Neighbour suppression, polygon clipping
│       ├── pca_alignment.py    Stage 2: PCA axes
│       ├── surface_heuristic.py  Stage 3: deterministic surface rule
│       ├── surface_classification.py  REFERENCE ONLY (retired RF)
│       ├── surface_cnn.py      REFERENCE ONLY (future CNN option)
│       ├── xml_annotations.py  AIM XML parser (training labels only)
│       ├── numbering.py        FDI mapping
│       └── postprocess.py      result.json encoding
├── models/
│   ├── registry.py             Artifact verification + model loading
│   └── versions.py             Model artifact manifest
├── scripts/
│   ├── prepare_caries_crop_dataset.py
│   ├── train_caries_crop_seg.py
│   └── validate_surface_heuristic.py
├── tests/
├── Dockerfile
├── requirements.txt
├── requirements-detectron2.txt
└── README.md
```

## Runtime model artifacts

The service loads the files listed in `models/versions.py` from `WEIGHTS_DIR`
(default `/weights`):

| Logical model | Artifact | Required? | Purpose |
|---|---|---|---|
| `pano_detector` | `Tooth_seg_pano_20250319.pt` | yes | 32-class FDI tooth segmentation on the panorama |
| `crop_segmenter` | `Tooth_seg_crop_20250424.pth` | yes unless `ENABLE_CROP_SEGMENTER=false` | 1-class Detectron2 tooth mask on each crop |
| `caries_detector` | `caries_crop_seg_20261028.pt` | yes in the current deployment; optional in code | 1-class caries YOLO-seg on each tooth crop |

`models/versions.py` pins all three production artifacts and their SHA-256 hashes.
The current deployment reports all three as `verified` on `/health`. The service
can run without the caries checkpoint, but then it reports
`caries_enabled: false` and does not return caries findings.

Quarantined/rejected artifacts (`caries_detect.pt`, the RF pickle) live in
`_non-using/deprecated-serving-models/` and are not read by this service.

## Inference tuning

| Variable | Default | Purpose |
|---|---:|---|
| `DETECTION_THRESHOLD` | `0.25` | Pano-Seg confidence |
| `ENABLE_CROP_SEGMENTER` | `true` | Run Crop-Seg; `false` only for local debugging |
| `CROP_PAD` | `20` | Pixels added around the tooth bbox |
| `DEDUPE_IOU` | `0.50` | Duplicate bbox IoU for the same FDI |
| `SUPPRESS_NEIGHBOURS` | `true` | Black out other detected teeth inside the crop |
| `CARIES_CONF` | `0.25` | Caries YOLO confidence; recalibrate after training |
| `CARIES_IOU` | `0.50` | Caries YOLO NMS IoU |
| `SURFACE_LEFT_BOUND` | `0.30` | Left surface zone |
| `SURFACE_RIGHT_BOUND` | `0.70` | Right surface zone |

The surface direction table and the measured 79.7% 3-way agreement with the
XML annotations are in the pipeline doc. The heuristic is deterministic:
`method` is `"Geometry_Heuristic"`. `surfaces[].probability` carries the
highest YOLO caries-detection confidence assigned to that surface. This score
is not calibrated as a probability that the whole surface is carious.

## Setup

### Docker (recommended)

```bash
cd dental-caries-detection
docker compose up -d --build ml-service
curl http://localhost:8001/health
```

Detectron2 is installed from the CPU wheel in the Dockerfile (torch 2.4.1 /
cp310 / Linux). It is **not** in `requirements.txt` because it is not on PyPI.
For a local Linux CPU install use `requirements-detectron2.txt`.

Crop-Seg input is preprocessed exactly like Detectron2 `DefaultPredictor`:
BGR→RGB, then `ResizeShortestEdge(MIN_SIZE_TEST=800, MAX_SIZE_TEST=1333)`,
both pinned in `models/versions.py`. The model does not resize internally; the
implementation logs `crop_seg refined N/M tooth masks` and warns when `N=0`
(all teeth would otherwise fall back to pano masks).

### Local

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
# optional, Linux only:
pip install -r requirements-detectron2.txt

export DATABASE_URL=sqlite:///./dev.db
export WEIGHTS_DIR="$PWD/weights"
export ENABLE_CROP_SEGMENTER=false   # only if Detectron2 is unavailable

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Train the caries model

```bash
# 1) dataset (must use --crop-seg on for train/serve parity)
python scripts/prepare_caries_crop_dataset.py \
  --images-root "D:/Mahidol University/SP/data/500 cases with annotation" \
  --weights-dir weights --out datasets/caries_crop_v1 \
  --crop-seg on --val-fraction 0.2 --negatives-per-case 3 --viz 24

# 2) train + export + print the versions.py entry
python scripts/train_caries_crop_seg.py \
  --data datasets/caries_crop_v1/data.yaml \
  --model yolov8s-seg.pt --epochs 100 --imgsz 640 --batch 8 --device 0 \
  --export-to weights/caries_crop_seg_YYYYMMDD.pt
```

Add the printed entry to `models/versions.py`, restart the service, and verify:

```bash
curl -s http://localhost:8001/health | jq
# artifacts.caries_detector == "verified", caries_enabled == true
```

## API

- `POST /infer {jobId}` — start a job (backend only)
- `POST /cancel` — cancel the active job
- `GET /health` — artifact status, `ready`, `caries_enabled`, `active_job_id`

`ready` covers the tooth pipeline (pano + Crop-Seg, or Crop-Seg deliberately
disabled). It does **not** require caries.

The backend owns the public API contract; result shape is documented in
[`docs-md/api-contract-v1.md`](../docs-md/api-contract-v1.md) (v1.2 sends the
caries detector confidence for each geometry-assigned surface in
`surfaces[].probability`).

## Tests

```bash
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

`tests/test_surface_cnn_parity.py`, `tests/bench_surface_cnn.py`, and
`tests/e2e_offline_eval.py` belong to the reference-only CNN path and are not
part of the serving pipeline.

## Production model files and local archives

The serving container loads the three files mounted from `weights/`: `Tooth_seg_pano_20250319.pt`, `Tooth_seg_crop_20250424.pth`, and `caries_crop_seg_20261028.pt`. The optional reference-only CNN path is disabled by default; its weights and the legacy `weights.zip` bundle are kept locally under `archive/` and are excluded from Git and the Docker build context. Older RF and caries checkpoints are retained in the repository-level `_non-using/deprecated-serving-models` archive.
