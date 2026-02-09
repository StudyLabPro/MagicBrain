"""
Basic API tests.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_list_models():
    """Test listing models."""
    response = client.get("/api/v1/models/")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "total" in data


def test_create_model():
    """Test creating a model."""
    response = client.post(
        "/api/v1/models/",
        json={
            "genome": "30121033102301230112332100123",
            "vocab_size": 10,
            "model_id": "test-model"
        }
    )

    # Should create or return conflict if exists
    assert response.status_code in [201, 409]

    if response.status_code == 201:
        data = response.json()
        assert data["model_id"] == "test-model"
        assert data["genome"] == "30121033102301230112332100123"


def test_get_nonexistent_model():
    """Test getting non-existent model."""
    response = client.get("/api/v1/models/nonexistent")
    assert response.status_code == 404


def test_training_request_validation():
    """Test training request validation."""
    # Missing required fields
    response = client.post(
        "/api/v1/training/start",
        json={
            "model_id": "test"
        }
    )
    assert response.status_code == 422  # Validation error


def test_inference_validation():
    """Test inference request validation."""
    # Invalid temperature
    response = client.post(
        "/api/v1/inference/sample",
        json={
            "model_id": "test",
            "temperature": 5.0  # Too high
        }
    )
    assert response.status_code == 422


def test_evolution_validation():
    """Test evolution request validation."""
    # Invalid fitness function
    response = client.post(
        "/api/v1/evolution/start",
        json={
            "text": "test text",
            "fitness_fn": "invalid"
        }
    )
    assert response.status_code == 422
