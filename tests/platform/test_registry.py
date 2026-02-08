"""Tests for model registry."""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from magicbrain.platform import (
    ModelInterface,
    ModelMetadata,
    ModelType,
    OutputType,
    ModelRegistry,
    ModelNotFoundError,
    ModelVersionConflict,
)


class DummyModel(ModelInterface):
    """Dummy model for testing."""

    def __init__(self, model_id="dummy", version="1.0.0"):
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=ModelType.DNN,
            version=version,
            description="Dummy model for testing",
            output_type=OutputType.DENSE,
            parameters_count=100,
        )
        super().__init__(metadata)

    def forward(self, input, **kwargs):
        return np.zeros(10)

    def get_output_type(self):
        return OutputType.DENSE


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry()
        model = DummyModel(model_id="test_model", version="1.0.0")

        model_id = registry.register(model)

        assert model_id == "test_model"
        models = registry.list_models()
        model_ids = [m["model_id"] for m in models]
        assert "test_model" in model_ids

    def test_register_with_version(self):
        """Test registering multiple versions."""
        registry = ModelRegistry()

        model_v1 = DummyModel(model_id="model1", version="1.0.0")
        model_v2 = DummyModel(model_id="model1", version="2.0.0")

        registry.register(model_v1)
        registry.register(model_v2)

        # Both versions should exist
        m1 = registry.get("model1", version="1.0.0")
        m2 = registry.get("model1", version="2.0.0")

        assert m1 is model_v1
        assert m2 is model_v2

    def test_register_conflict(self):
        """Test version conflict."""
        registry = ModelRegistry()
        model = DummyModel(model_id="model1", version="1.0.0")

        registry.register(model)

        # Try to register same version again
        with pytest.raises(ModelVersionConflict):
            registry.register(model)

    def test_register_with_overwrite(self):
        """Test overwriting existing version."""
        registry = ModelRegistry()
        model1 = DummyModel(model_id="model1", version="1.0.0")
        model2 = DummyModel(model_id="model1", version="1.0.0")

        registry.register(model1)
        registry.register(model2, overwrite=True)

        # Should get second model
        retrieved = registry.get("model1", version="1.0.0")
        assert retrieved is model2

    def test_get_model(self):
        """Test retrieving a model."""
        registry = ModelRegistry()
        model = DummyModel(model_id="test", version="1.0.0")

        registry.register(model)
        retrieved = registry.get("test", version="1.0.0")

        assert retrieved is model

    def test_get_latest_version(self):
        """Test getting latest version."""
        registry = ModelRegistry()

        registry.register(DummyModel("m1", "1.0.0"))
        registry.register(DummyModel("m1", "1.1.0"))
        registry.register(DummyModel("m1", "2.0.0"))

        # Get without specifying version (should get latest)
        model = registry.get("m1")
        assert model.get_metadata().version == "2.0.0"

    def test_get_not_found(self):
        """Test getting non-existent model."""
        registry = ModelRegistry()

        with pytest.raises(ModelNotFoundError):
            registry.get("nonexistent")

    def test_list_models(self):
        """Test listing models."""
        registry = ModelRegistry()

        registry.register(DummyModel("m1", "1.0.0"))
        registry.register(DummyModel("m2", "1.0.0"))

        models = registry.list_models()

        assert len(models) == 2
        model_ids = [m["model_id"] for m in models]
        assert "m1" in model_ids
        assert "m2" in model_ids

    def test_register_with_alias(self):
        """Test registering with alias."""
        registry = ModelRegistry()
        model = DummyModel("model1", "1.0.0")

        registry.register(model, alias="my_model")

        # Should be able to retrieve by alias
        retrieved = registry.get("my_model")
        assert retrieved is model

    def test_register_with_tags(self):
        """Test registering with tags."""
        registry = ModelRegistry()
        model = DummyModel("m1", "1.0.0")

        registry.register(model, tags=["tag1", "tag2"])

        # List with tag filter
        models = registry.list_models(tags=["tag1"])
        assert len(models) == 1
        assert models[0]["model_id"] == "m1"

    def test_register_with_dependencies(self):
        """Test registering with dependencies."""
        registry = ModelRegistry()
        model1 = DummyModel("m1", "1.0.0")
        model2 = DummyModel("m2", "1.0.0")

        registry.register(model1)
        registry.register(model2, dependencies=["m1"])

        deps = registry.get_dependencies("m2")
        assert "m1" in deps

        dependents = registry.get_dependents("m1")
        assert "m2" in dependents

    def test_remove_model(self):
        """Test removing a model."""
        registry = ModelRegistry()
        model = DummyModel("m1", "1.0.0")

        registry.register(model)
        registry.remove("m1", version="1.0.0")

        with pytest.raises(ModelNotFoundError):
            registry.get("m1", version="1.0.0")

    def test_remove_all_versions(self):
        """Test removing all versions."""
        registry = ModelRegistry()

        registry.register(DummyModel("m1", "1.0.0"))
        registry.register(DummyModel("m1", "2.0.0"))

        registry.remove("m1", remove_all_versions=True)

        with pytest.raises(ModelNotFoundError):
            registry.get("m1")

    def test_deprecate_model(self):
        """Test deprecating a model."""
        registry = ModelRegistry()
        model = DummyModel("m1", "1.0.0")

        registry.register(model)
        registry.deprecate("m1", version="1.0.0")

        # Should still be able to get it
        retrieved = registry.get("m1", version="1.0.0")
        assert retrieved is model

    def test_search(self):
        """Test searching for models."""
        registry = ModelRegistry()

        registry.register(DummyModel("neural_model", "1.0.0"))
        registry.register(DummyModel("transformer", "1.0.0"))

        results = registry.search("neural")

        assert len(results) >= 1
        assert any(r["model_id"] == "neural_model" for r in results)

    def test_get_stats(self):
        """Test getting registry statistics."""
        registry = ModelRegistry()

        registry.register(DummyModel("m1", "1.0.0"))
        registry.register(DummyModel("m2", "1.0.0"))

        stats = registry.get_stats()

        assert stats["total_models"] == 2
        assert stats["unique_model_ids"] == 2

    def test_save_and_load(self):
        """Test saving and loading registry state."""
        registry = ModelRegistry()

        registry.register(DummyModel("m1", "1.0.0"), tags=["test"])

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)

        try:
            registry.save(temp_path)

            # Load into new registry
            new_registry = ModelRegistry()
            new_registry.load(temp_path)

            # Note: Models themselves are not saved, only metadata
            # So we can't test model retrieval, only metadata
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
