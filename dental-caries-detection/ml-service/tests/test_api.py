import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert "active_job_id" in data

def test_cancel_endpoint(client):
    response = client.post("/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"

# We skip testing /infer as it would require mocking InferenceRunner and DB.
