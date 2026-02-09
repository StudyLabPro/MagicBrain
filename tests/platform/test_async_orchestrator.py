"""
Tests for async parallel execution in ModelOrchestrator.

Verifies:
- True async parallel execution
- Performance speedup vs sequential
- Error resilience (one model failure doesn't break others)
- Default async_forward() implementation
"""
import pytest
import asyncio
import time
import numpy as np

from magicbrain.platform.orchestrator import (
    ModelOrchestrator,
    ExecutionStrategy,
)
from magicbrain.platform.model_interface import (
    ModelInterface,
    OutputType,
    ModelMetadata,
    ModelType,
)


class DummyModel(ModelInterface):
    """Dummy model for testing."""

    def __init__(self, model_id: str, delay_ms: int = 0):
        """
        Initialize dummy model.

        Args:
            model_id: Model ID
            delay_ms: Artificial delay in milliseconds
        """
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=ModelType.DNN,
            version="1.0.0"
        )
        super().__init__(metadata)
        self.delay_ms = delay_ms

    def forward(self, input, **kwargs):
        """Synchronous forward with delay."""
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        # Return simple output based on input
        if isinstance(input, (list, np.ndarray)):
            return np.array([1.0, 2.0, 3.0])
        else:
            return np.array([float(input)])

    def get_output_type(self):
        """Output type."""
        return OutputType.DENSE


class FailingModel(ModelInterface):
    """Model that always fails."""

    def __init__(self, model_id: str):
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=ModelType.DNN,
            version="1.0.0"
        )
        super().__init__(metadata)

    def forward(self, input, **kwargs):
        """Always fails."""
        raise ValueError("Intentional failure")

    async def async_forward(self, input, **kwargs):
        """Always fails async."""
        raise ValueError("Intentional failure")

    def get_output_type(self):
        return OutputType.DENSE


class SlowModel(ModelInterface):
    """Model with slow processing."""

    def __init__(self, model_id: str, delay_sec: float = 0.1):
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=ModelType.DNN,
            version="1.0.0"
        )
        super().__init__(metadata)
        self.delay_sec = delay_sec

    def forward(self, input, **kwargs):
        """Slow forward."""
        time.sleep(self.delay_sec)
        return np.array([1.0])

    def get_output_type(self):
        return OutputType.DENSE


class TestAsyncOrchestrator:

    def test_parallel_execution_speedup(self):
        """Test that parallel execution runs models concurrently."""
        orchestrator = ModelOrchestrator()

        # Add 4 slow models (each takes 50ms)
        num_models = 4
        delay_sec = 0.05

        for i in range(num_models):
            model = SlowModel(f"slow_model_{i}", delay_sec=delay_sec)
            orchestrator.add_model(model)

        input_data = [0, 1, 2]

        # Parallel execution
        start = time.time()
        result_par = orchestrator.execute(input_data, ExecutionStrategy.PARALLEL)
        time_par = time.time() - start

        # If truly parallel: ~50ms (all run concurrently)
        # If sequential: 4 * 50ms = 200ms
        # We expect parallel time to be close to single model time
        # Allow 2x overhead for async machinery
        max_expected_time = delay_sec * 2.0  # 100ms

        assert time_par < max_expected_time, \
            f"Parallel execution took {time_par:.3f}s, expected <{max_expected_time:.3f}s (models running sequentially?)"

        # All models should have executed
        assert len(result_par.outputs) == num_models
        assert result_par.success

    def test_parallel_error_resilience(self):
        """Test that one model failure doesn't break others."""
        orchestrator = ModelOrchestrator()

        # Add good model
        good_model = DummyModel("good_model")
        orchestrator.add_model(good_model)

        # Add failing model
        failing_model = FailingModel("failing_model")
        orchestrator.add_model(failing_model)

        # Add another good model
        good_model2 = DummyModel("good_model_2")
        orchestrator.add_model(good_model2)

        # Execute parallel
        result = orchestrator.execute([0, 1], ExecutionStrategy.PARALLEL)

        # Good models should have outputs
        assert "good_model" in result.outputs
        assert "good_model_2" in result.outputs

        # Failing model should be skipped
        assert "failing_model" not in result.outputs

        # Execution should still succeed overall
        assert result.success

    def test_async_forward_default(self):
        """Test default async_forward implementation."""
        model = DummyModel("test_model")

        # Test async_forward via asyncio.run
        async def run_test():
            output = await model.async_forward([0, 1, 2])
            assert output is not None
            assert len(output) == 3
            return output

        output = asyncio.run(run_test())
        assert output is not None

    def test_concurrent_execution_count(self):
        """Test that all models execute once in parallel mode."""
        orchestrator = ModelOrchestrator()

        # Add multiple models
        for i in range(5):
            model = DummyModel(f"model_{i}")
            orchestrator.add_model(model)

        # Execute parallel
        result = orchestrator.execute([0, 1, 2], ExecutionStrategy.PARALLEL)

        # All 5 models should have executed
        assert len(result.models_executed) == 5
        assert len(result.outputs) == 5

    def test_parallel_with_different_outputs(self):
        """Test parallel execution with models producing different outputs."""
        orchestrator = ModelOrchestrator()

        # Add models
        model1 = DummyModel("model_1")
        model2 = DummyModel("model_2")

        orchestrator.add_model(model1)
        orchestrator.add_model(model2)

        # Execute
        result = orchestrator.execute([1, 2, 3], ExecutionStrategy.PARALLEL)

        # Both models should have outputs
        assert "model_1" in result.outputs
        assert "model_2" in result.outputs

        # Outputs should be independent
        assert isinstance(result.outputs["model_1"], np.ndarray)
        assert isinstance(result.outputs["model_2"], np.ndarray)

    def test_async_gather_behavior(self):
        """Test that asyncio.gather is actually being used."""
        orchestrator = ModelOrchestrator()

        # Add multiple slow models
        for i in range(4):
            model = SlowModel(f"slow_{i}", delay_sec=0.02)
            orchestrator.add_model(model)

        start = time.time()
        result = orchestrator.execute([0], ExecutionStrategy.PARALLEL)
        duration = time.time() - start

        # If running in parallel, should take ~0.02s
        # If running sequentially, would take 4 * 0.02 = 0.08s
        # Allow overhead, but ensure it's closer to parallel time
        assert duration < 0.06, f"Duration {duration:.3f}s too slow for parallel execution"

        # All models executed
        assert len(result.models_executed) == 4

    def test_empty_orchestrator_parallel(self):
        """Test parallel execution with no models."""
        orchestrator = ModelOrchestrator()

        with pytest.raises(Exception):
            orchestrator.execute([0], ExecutionStrategy.PARALLEL)

    def test_single_model_parallel(self):
        """Test parallel execution with single model."""
        orchestrator = ModelOrchestrator()

        model = DummyModel("single_model")
        orchestrator.add_model(model)

        result = orchestrator.execute([0, 1], ExecutionStrategy.PARALLEL)

        assert result.success
        assert len(result.outputs) == 1
        assert "single_model" in result.outputs

    def test_parallel_preserves_model_state(self):
        """Test that execution count is updated after parallel execution."""
        orchestrator = ModelOrchestrator()

        model1 = DummyModel("model_1")
        model2 = DummyModel("model_2")

        orchestrator.add_model(model1)
        orchestrator.add_model(model2)

        # Execute twice
        orchestrator.execute([0], ExecutionStrategy.PARALLEL)
        orchestrator.execute([0], ExecutionStrategy.PARALLEL)

        # Check execution counts
        node1 = orchestrator._nodes["model_1"]
        node2 = orchestrator._nodes["model_2"]

        assert node1.execution_count == 2
        assert node2.execution_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
