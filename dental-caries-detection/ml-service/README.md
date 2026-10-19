# Dental Caries Detection - ML Service

Python microservice for the multi-stage dental caries pipeline: panoramic tooth
segmentation, caries detection, PCA alignment, and Random Forest surface
classification.

## Technology Stack

- **Framework**: FastAPI (Python 3.11+)
- **ML Framework**: Ultralytics YOLOv8
- **Image Processing**: OpenCV, Pillow
- **Server**: Uvicorn (ASGI)
- **Containerization**: Docker

## Project Structure

```text
ml-service/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Runtime configuration
│   ├── db.py                   # Job state database access
│   ├── runner.py               # Inference process lifecycle
│   └── pipeline/
│       ├── detection.py        # Tooth/caries detection and ownership mapping
│       ├── pca_alignment.py    # Tooth PCA axes
│       ├── surface_classification.py
│       ├── surface_cnn.py      # Stage 3b CNN surface model (off | shadow | cnn)
│       └── orchestrator.py     # End-to-end inference and result publishing
├── models/
│   ├── registry.py             # Artifact verification and model loading
│   └── versions.py             # Model artifact manifest
├── Dockerfile
├── requirements.txt
└── README.md
```

## Runtime model artifacts

The service loads the files listed in `models/versions.py` from `WEIGHTS_DIR`
(default `/weights`):

| Logical model | Artifact | Purpose |
|---|---|---|
| `pano_detector` | `Tooth_seg_pano_20250319.pt` | Detect and segment FDI teeth |
| `caries_detector` | `caries_detect.pt` | Detect candidate caries |
| `crop_segmenter` | `Tooth_seg_crop_20250424.pth` | Optional tooth-mask refinement |
| `surface_classifier` | `dental_caries_surface_random_forest_13_features.pkl` | Production Random Forest classifier using the verified 13-feature geometric contract |
| `surface_cnn` (Stage 3b) | `cnn_pca42/best_pca_s42_f{0..4}.pt` + `cnn_pca42/calibration_pca.json` | 5-fold EfficientNet-B0 per-tooth CNN (Distal/Mesial/Occlusal). Not listed in `versions.py`; loaded by `app/pipeline/surface_cnn.py` from `SURFACE_CNN_DIR` (default `/weights/cnn_pca42`) |

## Surface CNN (Stage 3b)

`SURFACE_CNN_MODE` selects which model produces the surface findings (environment
only; the code default is `off`):

| Mode | Behaviour |
|---|---|
| `off` | CNN not loaded/run; RF findings only |
| `shadow` | CNN probabilities are stored under the separate `shadow_cnn` result key; RF findings stay primary |
| `cnn` | Calibrated CNN findings become primary (`method: "CNN"`); the RF is kept as a fallback |

| Environment variable | Default | Purpose |
|---|---|---|
| `SURFACE_CNN_MODE` | `off` | `off` / `shadow` / `cnn` |
| `SURFACE_CNN_DIR` | `/weights/cnn_pca42` | Folder with `best_pca_s42_f[0-9].pt` + calibration |
| `SURFACE_CNN_CALIB` | `<DIR>/calibration_pca.json` | Temperature + per-surface thresholds |
| `SURFACE_CNN_DEVICE` | `ML_DEVICE` or `cpu` | `cpu` or `cuda` |
| `SURFACE_CNN_THREADS` | `4` | torch thread count |
| `SURFACE_CNN_MAX_TEETH` | `0` (no limit) | Skip the stage above this tooth count |

Result keys added by Stage 3b (unknown to the backend/frontend zod schemas and
therefore stripped, so the existing API contract is unchanged):

- `surface_source`: `"cnn"` or `"rf"` — which model produced `teeth[].surfaces[]`
- `shadow_cnn`: evidence block (`mode`, `ready`, `n_teeth`, `teeth[]`, `notes`, `info`, `timings_ms`)

Fallback to the RF findings (recorded as `surface_source: "rf"`, never a failed
job) happens when the bundle cannot load, produced no predictions, or no
calibration thresholds are available. Rollback is one line: set
`SURFACE_CNN_MODE=shadow` (or `off`) and run `docker compose up -d ml-service`.

Full method, evidence and limitations: `docs-md/surface-cnn-full-report.md`.

## Inference tuning parameters

The following values belong to the production integration pipeline in
`app/pipeline/detection.py`; they are not parameters from the production RF
training/evaluation script:

| Parameter | Value | Purpose |
|---|---:|---|
| `CARIES_CONF` | `0.02` | Minimum confidence accepted from the YOLO caries detector |
| `CARIES_IOU` | `0.30` | NMS IoU used by the caries detector |
| `MIN_OWNERSHIP_SCORE` | `0.25` | Minimum tooth-mask score for assigning caries to a tooth |
| `POINT_CLOUD_DENSITY` | `5` | Grid spacing used to approximate points inside a caries box |

The reference RF pipeline starts after caries points are already available from
annotations/XML. It trains and evaluates the surface classifier, so it does
not need to define YOLO confidence, NMS, tooth ownership, or bbox point-cloud
sampling. These production values are therefore integration assumptions and
must be calibrated against detector evaluation data; they must not be
described as values copied from the RF reference pipeline.

## Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Docker

```bash
# Build
docker build -t dental-caries-ml-service .

# Run
docker run -p 8000:8000 dental-caries-ml-service
```

## API Endpoints

The backend owns the public API contract. The ML service receives jobs through
the shared-volume workflow described in `docs-md/ml-service-data-flow.md`.

### GET /health
Health check endpoint returning model status.

### GET /model/info
Get model metadata and configuration.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_SERVICE_PORT` | `8000` | Service port |
| `ML_SERVICE_HOST` | `0.0.0.0` | Service host |
| `WEIGHTS_DIR` | `/weights` | Directory containing all model artifacts |
| `CARIES_CONF` | `0.02` | Minimum caries confidence |
| `CARIES_IOU` | `0.30` | Caries detector NMS IoU |
| `POINT_CLOUD_DENSITY` | `5` | Caries bounding-box point-cloud spacing |
| `DETECTION_THRESHOLD` | `0.25` | Minimum tooth confidence |
| `ENABLE_CROP_SEGMENTER` | `false` | Enable optional crop-mask refinement; disabled to match the reference RF pipeline |
| `LOG_LEVEL` | `info` | Logging level |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
