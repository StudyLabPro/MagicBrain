"""A model_id from a request must never leave the model storage directory.

The API concatenates ``model_id`` into ``MODEL_STORAGE_PATH / f"{id}.npz"``.
Before this check a caller could pass ``../../etc/x`` and make the service
create, read or delete a file outside its own directory.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

API_DIR = pathlib.Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def test_valid_identifiers_pass():
    from app.core.ids import validate_model_id

    for good in ("test-model", "train-20260908-185554-ac9708",
                 "0f3a1c2e-4b5d-6a7f-8901-234567890abc", "a" * 64, "m.v2_1"):
        assert validate_model_id(good) == good


@pytest.mark.parametrize("bad", [
    "../../etc/passwd-probe", "..", ".", "a/b", "a\\b", "",
    ".hidden", "a" * 65, "with space", "sub/../x",
])
def test_traversal_and_junk_rejected(bad):
    from app.core.ids import InvalidModelId, validate_model_id

    with pytest.raises(InvalidModelId):
        validate_model_id(bad)


def test_runtime_service_refuses_traversal(tmp_path):
    from app.core.ids import InvalidModelId
    from app.core.runtime import RuntimeService

    service = RuntimeService(storage_path=str(tmp_path))
    assert service.model_path("ok-1") == tmp_path / "ok-1.npz"
    with pytest.raises(InvalidModelId):
        service.model_path("../escape")


def test_create_model_rejects_traversal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "MODEL_STORAGE_PATH", str(tmp_path))
    from app.api.main import app

    client = TestClient(app)
    resp = client.post("/api/v1/models/", json={
        "genome": "30121033102301230112332100123",
        "vocab_size": 8,
        "model_id": "../../escape-probe",
    })
    assert resp.status_code == 422
    # Nothing was written anywhere near the storage directory.
    assert not (tmp_path.parent / "escape-probe.npz").exists()
    assert list(tmp_path.glob("*.npz")) == []
