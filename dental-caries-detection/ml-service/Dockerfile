# ML Inference Service
# Python 3.10 slim — matches scikit-learn 1.7.2 + torch 2.4.1 + detectron2 compatibility.
#
# Detectron2 is not on PyPI with compatible pre-built wheels for torch 2.4.1;
# install from the facebookresearch release index keyed to the torch version.

FROM python:3.10-slim

# System dependencies for OpenCV, PyTorch, and Detectron2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r mluser && useradd -r -g mluser -m mluser

WORKDIR /app

# Install Python deps (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Detectron2 pre-built wheel for torch 2.4.1 / CUDA 11.8 (CPU-compatible)
# See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
RUN pip install --no-cache-dir \
    "detectron2 @ https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.4/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl" \
    || pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Shared data dir and weights mount point
RUN mkdir -p /shared /weights && chown mluser:mluser /shared /weights

USER mluser

# Single worker — runner holds in-memory process state
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--log-level", "info"]
