"""Tests for DetectionService."""
import io

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import BoundingBox, Detection
from app.services.detection import DetectionService


@pytest.fixture
def service():
    """Create a DetectionService instance (model not loaded)."""
    return DetectionService()


@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG image."""
    img = Image.new("RGB", (640, 640), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class TestDecodeImage:
    """Tests for _decode_image."""

    def test_valid_jpeg(self, service, sample_image_bytes):
        image = service._decode_image(sample_image_bytes)
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # BGR channels

    def test_invalid_bytes(self, service):
        with pytest.raises(ValueError, match="Invalid image data"):
            service._decode_image(b"not an image")

    def test_empty_bytes(self, service):
        with pytest.raises(ValueError):
            service._decode_image(b"")


class TestClassifySeverity:
    """Tests for severity classification."""

    def test_mild_low_confidence(self):
        det = Detection(
            class_name="caries",
            confidence=0.3,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        )
        assert DetectionService.classify_severity(det) == "mild"

    def test_mild_small_area(self):
        det = Detection(
            class_name="caries",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),  # area = 100
        )
        assert DetectionService.classify_severity(det) == "mild"

    def test_moderate(self):
        det = Detection(
            class_name="caries",
            confidence=0.6,
            bbox=BoundingBox(x1=0, y1=0, x2=80, y2=80),  # area = 6400
        )
        assert DetectionService.classify_severity(det) == "moderate"

    def test_severe(self):
        det = Detection(
            class_name="caries",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),  # area = 10000
        )
        assert DetectionService.classify_severity(det) == "severe"


class TestGenerateAnnotatedImage:
    """Tests for annotated image generation."""

    def test_generates_base64(self, service):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = [
            Detection(
                class_name="caries",
                confidence=0.95,
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            )
        ]
        result = service._generate_annotated_image(image, detections)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid base64
        import base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_empty_detections(self, service):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = service._generate_annotated_image(image, [])
        assert isinstance(result, str)
        assert len(result) > 0


class TestPredictMethod:
    """Tests for predict method."""

    def test_predict_without_model_raises(self, service, sample_image_bytes):
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            service.predict(sample_image_bytes)
