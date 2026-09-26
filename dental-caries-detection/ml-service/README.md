# Dental Caries Detection - ML Service

Standalone Python microservice for dental caries detection using YOLOv8 object detection model.

## Technology Stack

- **Framework**: FastAPI (Python 3.11+)
- **ML Framework**: Ultralytics YOLOv8
- **Image Processing**: OpenCV, Pillow
- **Server**: Uvicorn (ASGI)
- **Containerization**: Docker

## Project Structure

```
ml-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py             # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py         # API route definitions
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic request/response schemas
│   └── services/
│       ├── __init__.py
│       └── detection.py      # YOLOv8 detection service
├── models/
│   └── best.pt               # Trained YOLOv8 model weights
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # API endpoint tests
│   └── test_detection.py     # Detection service tests
├── Dockerfile
├── requirements.txt
└── README.md
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
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build
docker build -t dental-caries-ml-service .

# Run
docker run -p 8000:8000 dental-caries-ml-service
```

## API Endpoints

### POST /predict
Run caries detection on an uploaded dental X-ray image.

**Parameters:**
- `file` (multipart): Dental X-ray image (JPEG/PNG, max 10MB)
- `confidence_threshold` (query, optional): Minimum confidence score (0.0-1.0, default: 0.25)

### GET /health
Health check endpoint returning model status.

### GET /model/info
Get model metadata and configuration.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_SERVICE_PORT` | `8000` | Service port |
| `ML_SERVICE_HOST` | `0.0.0.0` | Service host |
| `MODEL_PATH` | `./models/best.pt` | Path to YOLOv8 model |
| `CONFIDENCE_THRESHOLD` | `0.25` | Default confidence threshold |
| `MODEL_VERSION` | `v1.0.0` | Model version string |
| `LOG_LEVEL` | `info` | Logging level |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
