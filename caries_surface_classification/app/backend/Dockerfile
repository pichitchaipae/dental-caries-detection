# ──────────────────────────────────────────────────────────────
#  Backend — FastAPI + ML inference pipeline
#  Models: Tooth_seg_pano (YOLO), rf_classify_ml (RandomForest)
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies required by OpenCV (libGL) and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (detectron2 excluded — handled via fallback)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
