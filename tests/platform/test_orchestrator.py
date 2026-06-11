"""Tests for model orchestrator."""
import pytest
import numpy as np

from magicbrain.platform import (
    ModelInterface,
    ModelMetadata,
    ModelType,
    OutputType,
    ModelOrchestrator,
    ExecutionStrategy,
    OrchestratorError,
)


class SimpleModel(ModelInterface):
    """Simple model for testing."""

    def __init__(self, model_id, multiplier=2.0):
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=ModelType.DNN,
            version="1.0.0",
            output_type=OutputType.DENSE,
        )
        super().__init__(metadata)
        self.multiplier = multiplier

    def forward(self, input, **kwargs):
        if isinstance(input, (int, float)):
            return input * self.multiplier
        return input * self.multiplier

    def get_output_type(self):
        return OutputType.DENSE


class TestModelOrchestrator:
    """Tests for ModelOrchestrator."""

    def test_add_model(self):
        """Test adding a model."""
        orch = ModelOrchestrator()
        model = SimpleModel("model1")

        model_id = orch.add_model(model)

        assert model_id == "model1"
        assert "model1" in orch.list_models()

    def test_add_duplicate_model(self):
        """Test adding duplicate model."""
        orch = ModelOrchestrator()
        model1 = SimpleModel("model1")
        model2 = SimpleModel("model1")

        orch.add_model(model1)

        with pytest.raises(ValueError):
            orch.add_model(model2)

    def test_connect_models(self):
        """Test connecting models."""
        orch = ModelOrchestrator()
        model1 = SimpleModel("m1")
        model2 = SimpleModel("m2")

        orch.add_model(model1)
        orch.add_model(model2)
        orch.connect("m1", "m2")

        graph = orch.get_graph()

        assert "m2" in graph["m1"]["outputs"]
        assert "m1" in graph["m2"]["inputs"]

    def test_connect_nonexistent(self):
        """Test connecting nonexistent models."""
        orch = ModelOrchestrator()

        with pytest.raises(OrchestratorError):
            orch.connect("m1", "m2")

    def test_disconnect(self):
        """Test disconnecting models."""
        orch = ModelOrchestrator()
        m1 = SimpleModel("m1")
        m2 = SimpleModel("m2")

        orch.add_model(m1)
        orch.add_model(m2)
        orch.connect("m1", "m2")
        orch.disconnect("m1", "m2")

        graph = orch.get_graph()

        assert "m2" not in graph["m1"]["outputs"]
        assert "m1" not in graph["m2"]["inputs"]

    def test_remove_model(self):
        """Test removing a model."""
        orch = ModelOrchestrator()
        model = SimpleModel("m1")

        orch.add_model(model)
        orch.remove_model("m1")

        assert "m1" not in orch.list_models()

    def test_execute_sequential(self):
        """Test sequential execution."""
        orch = ModelOrchestrator()

        # Create chain: m1 (×2) → m2 (×3) → m3 (×4)
        m1 = SimpleModel("m1", multiplier=2.0)
        m2 = SimpleModel("m2", multiplier=3.0)
        m3 = SimpleModel("m3", multiplier=4.0)

        orch.add_model(m1)
        orch.add_model(m2)
        orch.add_model(m3)

        orch.connect("m1", "m2")
        orch.connect("m2", "m3")

        # Execute
        result = orch.execute(
            input_data=5.0,
            strategy=ExecutionStrategy.SEQUENTIAL,
            entry_model="m1"
        )

        assert result.success
        # 5 × 2 × 3 × 4 = 120
        assert result.get_final_output() == 120.0

    def test_execute_parallel(self):
        """Test parallel execution."""
        orch = ModelOrchestrator()

        m1 = SimpleModel("m1", multiplier=2.0)
        m2 = SimpleModel("m2", multiplier=3.0)

        orch.add_model(m1)
        orch.add_model(m2)

        # Execute
        result = orch.execute(
            input_data=10.0,
            strategy=ExecutionStrategy.PARALLEL
        )

        assert result.success
        assert result.get_output("m1") == 20.0
        assert result.get_output("m2") == 30.0

    def test_execute_no_models(self):
        """Test executing with no models."""
        orch = ModelOrchestrator()

        with pytest.raises(OrchestratorError):
            orch.execute(input_data=5.0, strategy=ExecutionStrategy.SEQUENTIAL)

    def test_get_model(self):
        """Test getting a model."""
        orch = ModelOrchestrator()
        model = SimpleModel("m1")

        orch.add_model(model)
        retrieved = orch.get_model("m1")

        assert retrieved is model

    def test_get_nonexistent_model(self):
        """Test getting nonexistent model."""
        orch = ModelOrchestrator()

        retrieved = orch.get_model("nonexistent")
        assert retrieved is None

    def test_list_models(self):
        """Test listing models."""
        orch = ModelOrchestrator()

        orch.add_model(SimpleModel("m1"))
        orch.add_model(SimpleModel("m2"))

        models = orch.list_models()

        assert len(models) == 2
        assert "m1" in models
        assert "m2" in models

    def test_get_graph(self):
        """Test getting graph structure."""
        orch = ModelOrchestrator()

        m1 = SimpleModel("m1")
        m2 = SimpleModel("m2")

        orch.add_model(m1)
        orch.add_model(m2)
        orch.connect("m1", "m2")

        graph = orch.get_graph()

        assert "m1" in graph
        assert "m2" in graph
        assert "m2" in graph["m1"]["outputs"]

    def test_reset_state(self):
        """Test resetting state."""
        orch = ModelOrchestrator()
        model = SimpleModel("m1")

        orch.add_model(model)

        # Execute to set state
        orch.execute(input_data=5.0, strategy=ExecutionStrategy.SEQUENTIAL)

        # Reset
        orch.reset_state()

        # State should be reset (can't easily test without StatefulModel)

    def test_get_last_execution(self):
        """Test getting last execution result."""
        orch = ModelOrchestrator()
        model = SimpleModel("m1")

        orch.add_model(model)

        # Execute
        orch.execute(input_data=5.0, strategy=ExecutionStrategy.SEQUENTIAL)

        last = orch.get_last_execution()

        assert last is not None
        assert last.success

    def test_get_stats(self):
        """Test getting statistics."""
        orch = ModelOrchestrator()

        orch.add_model(SimpleModel("m1"))
        orch.add_model(SimpleModel("m2"))

        stats = orch.get_stats()

        assert stats["models_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
