"""Tests for SNN model adapter."""
import pytest
import numpy as np

from magicbrain.models.snn import SNNTextModel
from magicbrain.platform import OutputType, ModelType


class TestSNNTextModel:
    """Tests for SNNTextModel adapter."""

    def test_initialization(self):
        """Test model initialization."""
        genome = "30121033102301230112332100123"
        model = SNNTextModel(
            genome=genome,
            vocab_size=50,
            model_id="test_snn",
            version="1.0.0"
        )

        assert model.metadata.model_id == "test_snn"
        assert model.metadata.version == "1.0.0"
        assert model.metadata.model_type == ModelType.SNN
        assert model.vocab_size == 50
        assert model.genome == genome

    def test_forward(self):
        """Test forward pass."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="test_snn"
        )

        # Forward with single token
        logits = model.forward(0)

        assert isinstance(logits, np.ndarray)
        assert logits.shape == (50,)

    def test_output_type(self):
        """Test output type."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        assert model.get_output_type() == OutputType.LOGITS

    def test_step(self):
        """Test single step."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        logits = model.step(0)

        assert isinstance(logits, np.ndarray)
        assert logits.shape == (50,)

    def test_get_hidden_state(self):
        """Test getting hidden state."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        # Forward to set state
        model.forward(0)

        state = model.get_hidden_state()

        assert "activation" in state
        assert "trace_fast" in state
        assert "trace_slow" in state
        assert "theta" in state
        assert "w_slow" in state
        assert "w_fast" in state

    def test_set_hidden_state(self):
        """Test setting hidden state."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        # Get initial state
        model.forward(0)
        state1 = model.get_hidden_state()

        # Modify state
        state2 = {
            "activation": np.zeros_like(state1["activation"]),
            "trace_fast": np.zeros_like(state1["trace_fast"]),
            "trace_slow": np.zeros_like(state1["trace_slow"]),
            "theta": np.ones_like(state1["theta"]),
        }

        # Set new state
        model.set_hidden_state(state2)

        # Verify
        state3 = model.get_hidden_state()
        assert np.allclose(state3["activation"], state2["activation"])
        assert np.allclose(state3["theta"], state2["theta"])

    def test_get_spikes(self):
        """Test getting spike activation."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        model.forward(0)
        spikes = model.get_spikes()

        assert isinstance(spikes, np.ndarray)
        assert spikes.shape == (model.brain.N,)
        assert np.all((spikes == 0) | (spikes == 1))

    def test_get_traces(self):
        """Test getting synaptic traces."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        model.forward(0)
        trace_fast, trace_slow = model.get_traces()

        assert isinstance(trace_fast, np.ndarray)
        assert isinstance(trace_slow, np.ndarray)
        assert trace_fast.shape == (model.brain.N,)
        assert trace_slow.shape == (model.brain.N,)

    def test_reset(self):
        """Test resetting model state."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        # Forward to set state
        model.forward(0)
        model.forward(1)

        # Reset
        model.reset()

        # Check that activation is reset
        spikes = model.get_spikes()
        assert np.all(spikes == 0)

    def test_get_brain_stats(self):
        """Test getting brain statistics."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        model.forward(0)
        stats = model.get_brain_stats()

        assert "N" in stats
        assert "K" in stats
        assert "E" in stats
        assert "firing_rate" in stats
        assert "dopamine" in stats

    def test_summary(self):
        """Test model summary."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="test_model",
            description="Test SNN model"
        )

        summary = model.summary()

        assert "test_model" in summary
        assert "SNN" in summary
        assert "Neurons" in summary

    def test_parameters_count(self):
        """Test parameter count."""
        model = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        count = model.get_parameters_count()

        assert count > 0
        assert isinstance(count, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
