# Dental Caries Detection - ML Service

Python microservice for the multi-stage dental caries pipeline: panoramic tooth
segmentation, lesion detection, PCA alignment, and Random Forest surface
classification.

## Technology Stack

- **Framework**: FastAPI (Python 3.11+)
- **ML Framework**: Ultralytics YOLOv8
- **Image Processing**: OpenCV, Pillow
- **Server**: Uvicorn (ASGI)
- **Containerization**: Docker

## Runtime model artifacts

```
The service loads the files listed in `models/versions.py` from `WEIGHTS_DIR`
(default `/weights`):

| Logical model | Artifact | Purpose |
|---|---|---|
| `pano_detector` | `Tooth_seg_pano_20250319.pt` | Detect and segment FDI teeth |
| `caries_detector` | `caries_detect.pt` | Detect candidate lesions |
| `crop_segmenter` | `Tooth_seg_crop_20250424.pth` | Optional tooth-mask refinement |
| `surface_classifier` | `rf_classify_ml.pkl` | Run 3 RF classifier using 14 features |

`rf_classify_ml.pkl` is retained as a deployment-compatible legacy filename;
the logical name `surface_classifier` should be used in code and diagnostics.
```

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
| `CARIES_CONF` | `0.02` | Minimum lesion confidence |
| `DETECTION_THRESHOLD` | `0.25` | Minimum tooth confidence |
| `ENABLE_CROP_SEGMENTER` | `true` | Enable optional crop-mask refinement |
| `LOG_LEVEL` | `info` | Logging level |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
