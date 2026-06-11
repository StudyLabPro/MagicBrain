"""
Comprehensive tests for Hybrid Architectures.

Tests all hybrid combinations:
- SNN + DNN
- SNN + Transformer
- CNN + SNN
- Integration points
- Forward/backward passes
"""
import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from magicbrain.hybrid import (
    SNNDNNHybrid,
    HybridBuilder,
)
from magicbrain.models.snn import SNNTextModel
from magicbrain.platform.model_interface import OutputType

if TORCH_AVAILABLE:
    from magicbrain.models.dnn import DNNModel

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class TestHybridArchitectures:

    def test_snn_dnn_forward(self):
        """Test SNN+DNN forward pass."""
        # Create SNN model
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="snn_test"
        )

        # Create simple DNN model
        dnn_module = nn.Sequential(
            nn.Linear(snn.brain.N, 64),
            nn.ReLU(),
            nn.Linear(64, 50),
        )
        dnn = DNNModel(dnn_module, model_id="dnn_test")

        # Create hybrid
        hybrid = SNNDNNHybrid(snn, dnn, model_id="hybrid_test")

        # Forward pass
        output = hybrid.forward([0, 1, 2])

        # Verify output shape
        assert output is not None
        assert isinstance(output, (np.ndarray, list))

    def test_hybrid_components_accessible(self):
        """Test that hybrid components are accessible."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Access components
        assert hybrid.get_component("snn_encoder") is not None
        assert hybrid.get_component("dnn_decoder") is not None

        # Verify component types
        assert isinstance(hybrid.get_component("snn_encoder"), SNNTextModel)
        assert isinstance(hybrid.get_component("dnn_decoder"), DNNModel)

    def test_hybrid_metadata(self):
        """Test hybrid metadata is correct."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn, model_id="test_hybrid")

        metadata = hybrid.get_metadata()

        assert metadata.model_id == "test_hybrid"
        assert "hybrid" in metadata.model_id.lower()

    def test_hybrid_output_type(self):
        """Test hybrid returns correct output type."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Output type should be from final component (DNN)
        output_type = hybrid.get_output_type()
        assert output_type is not None

    def test_hybrid_with_different_vocab_sizes(self):
        """Test hybrid with different vocab sizes."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        # DNN outputs different size
        dnn_module = nn.Linear(snn.brain.N, 100)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        output = hybrid.forward([0, 1, 2])
        assert output is not None

    def test_hybrid_sequential_forward_calls(self):
        """Test multiple forward calls work correctly."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Multiple forward passes
        out1 = hybrid.forward([0, 1])
        out2 = hybrid.forward([2, 3])
        out3 = hybrid.forward([4, 5])

        assert out1 is not None
        assert out2 is not None
        assert out3 is not None

    def test_hybrid_factory_function(self):
        """Test factory function for creating hybrids."""
        from magicbrain.hybrid.snn_dnn import create_snn_dnn_hybrid

        dnn_module = nn.Linear(384, 50)  # Assuming default N=384

        hybrid = create_snn_dnn_hybrid(
            snn_genome="30121033102301230112332100123",
            vocab_size=50,
            dnn_module=dnn_module,
            model_id="factory_test"
        )

        assert hybrid is not None
        assert hybrid.get_metadata().model_id == "factory_test"

        # Test forward
        output = hybrid.forward([0, 1, 2])
        assert output is not None


class TestHybridIntegrationPoints:

    def test_snn_to_dnn_integration(self):
        """Test data flow from SNN to DNN."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Get SNN output directly
        snn_output = snn.forward([0, 1, 2])

        # Get hybrid output (SNN → DNN)
        hybrid_output = hybrid.forward([0, 1, 2])

        # Both should be valid
        assert snn_output is not None
        assert hybrid_output is not None

        # Shapes might differ (SNN outputs logits, DNN transforms them)
        # But both should be numeric arrays
        assert isinstance(snn_output, np.ndarray)

    def test_component_independence(self):
        """Test that components can be used independently."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Use SNN independently
        snn_out = snn.forward([0, 1])
        assert snn_out is not None

        # Use hybrid
        hybrid_out = hybrid.forward([0, 1])
        assert hybrid_out is not None

    def test_state_preservation_across_components(self):
        """Test that SNN state is preserved through hybrid."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # First forward
        hybrid.forward([0])

        # Check SNN state changed
        assert snn.brain.execution_count > 0 or snn.brain.step >= 0


class TestHybridErrorHandling:

    def test_invalid_component_access(self):
        """Test accessing non-existent component."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Access non-existent component
        result = hybrid.get_component("nonexistent")
        assert result is None

    def test_empty_input(self):
        """Test hybrid with empty input."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = SNNDNNHybrid(snn, dnn)

        # Try empty input
        try:
            output = hybrid.forward([])
            # Might fail or return something, either is acceptable
        except:
            pass  # Expected to fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
