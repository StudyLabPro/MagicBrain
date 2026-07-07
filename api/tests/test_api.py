"""
Basic API tests.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.runtime import RuntimeService
from magicbrain import TextBrain
from magicbrain.io import save_model

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


def test_runtime_service_restores_auto_loaded_models(tmp_path):
    """Test runtime manifest restores auto-loaded models."""
    model_id = "persistent-model"
    brain = TextBrain("30121033102301230112332100123", 4)
    stoi = {str(i): i for i in range(4)}
    itos = {i: str(i) for i in range(4)}
    save_model(brain, stoi, itos, str(tmp_path / f"{model_id}.npz"))

    runtime = RuntimeService(storage_path=str(tmp_path))
    runtime.load(model_id)
    assert model_id in runtime.stats()["registered_models"]
    assert model_id in runtime.stats()["loaded_models"]

    restored_runtime = RuntimeService(storage_path=str(tmp_path))
    restored = restored_runtime.restore_registered_models()
    assert restored == [model_id]
    assert model_id in restored_runtime.stats()["loaded_models"]


def test_runtime_stats():
    """Test runtime stats endpoint."""
    response = client.get("/api/v1/runtime/stats")
    assert response.status_code == 200
    data = response.json()
    assert "storage_path" in data
    assert "manifest_path" in data
    assert "registered_models" in data
    assert "loaded_models" in data
    assert "registry" in data
    assert "orchestrator" in data


def test_runtime_model_listing():
    """Test runtime model listing endpoint."""
    response = client.get("/api/v1/runtime/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "loaded_models" in data


def test_runtime_load_nonexistent_model():
    """Test loading a non-existent runtime model."""
    response = client.post("/api/v1/runtime/models/nonexistent/load")
    assert response.status_code == 404


def test_inference_uses_runtime_for_missing_model():
    """Test runtime-backed inference returns 404 for missing model."""
    response = client.post(
        "/api/v1/inference/predict",
        json={
            "model_id": "nonexistent",
            "context": "x",
        },
    )
    assert response.status_code == 404


def test_model_runtime_management_endpoints():
    """Test model-level runtime register/load/unload endpoints."""
    model_id = "runtime-management"
    client.delete(f"/api/v1/models/{model_id}")

    create_response = client.post(
        "/api/v1/models/",
        json={
            "genome": "30121033102301230112332100123",
            "vocab_size": 10,
            "model_id": model_id,
        },
    )
    assert create_response.status_code == 201

    register_response = client.post(
        f"/api/v1/models/{model_id}/register",
        json={"auto_load": False},
    )
    assert register_response.status_code == 200
    assert register_response.json()["auto_load"] is False

    model_response = client.get(f"/api/v1/models/{model_id}")
    assert model_response.status_code == 200
    model_data = model_response.json()
    assert model_data["registered"] is True
    assert model_data["auto_load"] is False
    assert model_data["loaded"] is False

    load_response = client.post(f"/api/v1/models/{model_id}/load")
    assert load_response.status_code == 200

    model_response = client.get(f"/api/v1/models/{model_id}")
    model_data = model_response.json()
    assert model_data["registered"] is True
    assert model_data["auto_load"] is True
    assert model_data["loaded"] is True

    unload_response = client.delete(f"/api/v1/models/{model_id}/unload")
    assert unload_response.status_code == 204

    model_response = client.get(f"/api/v1/models/{model_id}")
    model_data = model_response.json()
    assert model_data["registered"] is True
    assert model_data["auto_load"] is False
    assert model_data["loaded"] is False

    unregister_response = client.delete(f"/api/v1/models/{model_id}/register")
    assert unregister_response.status_code == 204

    client.delete(f"/api/v1/models/{model_id}")


def test_runtime_inference_smoke():
    """Test create, load, predict and sample through runtime."""
    model_id = "runtime-smoke"
    client.delete(f"/api/v1/models/{model_id}")

    create_response = client.post(
        "/api/v1/models/",
        json={
            "genome": "30121033102301230112332100123",
            "vocab_size": 10,
            "model_id": model_id,
        },
    )
    assert create_response.status_code in [201, 409]

    load_response = client.post(f"/api/v1/runtime/models/{model_id}/load")
    assert load_response.status_code == 200
    assert load_response.json()["loaded"] is True

    predict_response = client.post(
        "/api/v1/inference/predict",
        json={
            "model_id": model_id,
            "context": "1",
            "top_k": 3,
        },
    )
    assert predict_response.status_code == 200
    predict_data = predict_response.json()
    assert len(predict_data["predictions"]) == 3
    assert predict_data["runtime"]["models_executed"] == [model_id]

    sample_response = client.post(
        "/api/v1/inference/sample",
        json={
            "model_id": model_id,
            "seed_text": "1",
            "n_tokens": 3,
            "temperature": 0.75,
        },
    )
    assert sample_response.status_code == 200
    sample_data = sample_response.json()
    assert sample_data["model_id"] == model_id
    assert len(sample_data["generated_text"]) >= 1

    client.delete(f"/api/v1/models/{model_id}")


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
