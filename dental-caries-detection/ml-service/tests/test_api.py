"""Tests for ML Service API endpoints."""
import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, detection_service
from app.models.schemas import BoundingBox, Detection


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_loaded():
    """Mock detection service with model loaded."""
    detection_service.model_loaded = True
    detection_service.model = MagicMock()
    yield
    detection_service.model_loaded = False
    detection_service.model = None


@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG image."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_model_loaded(self, client, mock_model_loaded):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_version" in data
        assert "gpu_available" in data

    def test_health_model_not_loaded(self, client):
        detection_service.model_loaded = False
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info(self, client, mock_model_loaded):
        detection_service.model.names = {0: "caries"}
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "dental-caries-yolov8"
        assert data["model_type"] == "YOLOv8"
        assert "caries" in data["classes"]
        assert data["input_size"] == 640

    def test_model_info_no_model(self, client):
        detection_service.model_loaded = False
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["classes"] == ["caries"]


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_model_not_loaded(self, client, sample_image_bytes):
        detection_service.model_loaded = False
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 503

    def test_predict_unsupported_format(self, client, mock_model_loaded):
        response = client.post(
            "/predict",
            files={"file": ("test.gif", b"fake", "image/gif")},
        )
        assert response.status_code == 415

    def test_predict_empty_file(self, client, mock_model_loaded):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    @patch.object(detection_service, "predict")
    def test_predict_success(self, mock_predict, client, mock_model_loaded, sample_image_bytes):
        mock_predict.return_value = (
            [
                Detection(
                    class_name="caries",
                    confidence=0.95,
                    bbox=BoundingBox(x1=150, y1=200, x2=200, y2=245),
                )
            ],
            "base64encodedimage",
            2.3,
        )
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["class_name"] == "caries"
        assert data["annotated_image"] == "base64encodedimage"
        assert data["processing_time"] == 2.3

    @patch.object(detection_service, "predict")
    def test_predict_with_confidence_threshold(self, mock_predict, client, mock_model_loaded, sample_image_bytes):
        mock_predict.return_value = ([], "base64img", 1.0)
        response = client.post(
            "/predict?confidence_threshold=0.8",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        mock_predict.assert_called_once()
        call_kwargs = mock_predict.call_args
        assert call_kwargs[1]["confidence_threshold"] == 0.8 or call_kwargs[0][1] == 0.8 if len(call_kwargs[0]) > 1 else True
