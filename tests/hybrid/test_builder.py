"""
Tests for HybridBuilder fluent interface.

Verifies:
- Fluent API chaining
- Component addition
- Connection creation
- Building hybrids
- Templates
- Validation
"""
import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from magicbrain.hybrid import HybridBuilder
from magicbrain.hybrid.builder import Templates
from magicbrain.models.snn import SNNTextModel

if TORCH_AVAILABLE:
    from magicbrain.models.dnn import DNNModel

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class TestHybridBuilder:

    def test_fluent_interface_chaining(self):
        """Test that fluent interface returns self for chaining."""
        builder = HybridBuilder()

        # Each method should return builder for chaining
        result = builder.add("component1", None)
        assert isinstance(result, HybridBuilder)

        result = builder.connect("comp1", "comp2")
        assert isinstance(result, HybridBuilder)

        result = builder.set_output("comp1")
        assert isinstance(result, HybridBuilder)

    def test_build_simple_hybrid(self):
        """Test building simple 2-component hybrid."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn_module = nn.Linear(snn.brain.N, 50)
        dnn = DNNModel(dnn_module)

        hybrid = (HybridBuilder()
            .add("snn", snn)
            .add("dnn", dnn)
            .connect("snn", "dnn")
            .set_output("dnn")
            .build("test_hybrid"))

        assert hybrid is not None
        assert hybrid.get_metadata().model_id == "test_hybrid"

    def test_build_without_components(self):
        """Test that building without components raises error."""
        builder = HybridBuilder()

        with pytest.raises(ValueError, match="No components"):
            builder.build()

    def test_complex_pipeline(self):
        """Test multi-stage hybrid pipeline."""
        snn1 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="snn1"
        )

        snn2 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="snn2"
        )

        dnn_module = nn.Linear(snn1.brain.N, 50)
        dnn = DNNModel(dnn_module, model_id="dnn")

        hybrid = (HybridBuilder()
            .add("stage1", snn1)
            .add("stage2", snn2)
            .add("stage3", dnn)
            .connect("stage1", "stage2")
            .connect("stage2", "stage3")
            .set_output("stage3")
            .build("three_stage"))

        assert hybrid is not None

        # Verify structure
        assert hybrid.get_component("stage1") is not None
        assert hybrid.get_component("stage2") is not None
        assert hybrid.get_component("stage3") is not None

    def test_reset_builder(self):
        """Test reset clears builder state."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        builder = HybridBuilder()
        builder.add("snn", snn)
        builder.connect("snn", "other")
        builder.set_output("snn")

        # Reset
        builder.reset()

        # Should be empty now
        assert len(builder._components) == 0
        assert len(builder._connections) == 0
        assert builder._output_component is None

    def test_builder_reuse(self):
        """Test builder can be reused after reset."""
        snn1 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        snn2 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        builder = HybridBuilder()

        # Build first hybrid
        dnn1 = DNNModel(nn.Linear(snn1.brain.N, 50))

        hybrid1 = (builder
            .add("snn", snn1)
            .add("dnn", dnn1)
            .connect("snn", "dnn")
            .build("hybrid1"))

        # Reset and build second
        dnn2 = DNNModel(nn.Linear(snn2.brain.N, 50))

        hybrid2 = (builder.reset()
            .add("snn", snn2)
            .add("dnn", dnn2)
            .connect("snn", "dnn")
            .build("hybrid2"))

        assert hybrid1.get_metadata().model_id == "hybrid1"
        assert hybrid2.get_metadata().model_id == "hybrid2"

    def test_multiple_connections(self):
        """Test component with multiple input connections."""
        snn1 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="snn1"
        )

        snn2 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="snn2"
        )

        dnn = DNNModel(nn.Linear(snn1.brain.N, 50), model_id="dnn")

        hybrid = (HybridBuilder()
            .add("snn1", snn1)
            .add("snn2", snn2)
            .add("dnn", dnn)
            .connect("snn1", "dnn")
            .connect("snn2", "dnn")
            .build("multi_input"))

        assert hybrid is not None


class TestTemplates:

    def test_snn_dnn_pipeline_template(self):
        """Test SNN→DNN pipeline template."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn = DNNModel(nn.Linear(snn.brain.N, 50))

        hybrid = Templates.snn_dnn_pipeline(snn, dnn, model_id="template_test")

        assert hybrid is not None
        assert hybrid.get_metadata().model_id == "template_test"

        # Test forward
        output = hybrid.forward([0, 1, 2])
        assert output is not None

    def test_encoder_decoder_template(self):
        """Test encoder→decoder template."""
        encoder = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        decoder = DNNModel(nn.Linear(encoder.brain.N, 50))

        hybrid = Templates.encoder_decoder(encoder, decoder, model_id="enc_dec")

        assert hybrid is not None
        output = hybrid.forward([0, 1])
        assert output is not None

    def test_three_stage_pipeline_template(self):
        """Test three-stage pipeline template."""
        m1 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="m1"
        )

        m2 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50,
            model_id="m2"
        )

        m3 = DNNModel(nn.Linear(m1.brain.N, 50), model_id="m3")

        hybrid = Templates.three_stage_pipeline(m1, m2, m3, model_id="three_stage")

        assert hybrid is not None
        assert hybrid.get_component("stage1") is not None
        assert hybrid.get_component("stage2") is not None
        assert hybrid.get_component("stage3") is not None


class TestBuilderValidation:

    def test_build_with_missing_model_id(self):
        """Test building without explicit model_id."""
        snn = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        dnn = DNNModel(nn.Linear(snn.brain.N, 50))

        # Build without model_id (should auto-generate or use default)
        hybrid = (HybridBuilder()
            .add("snn", snn)
            .add("dnn", dnn)
            .connect("snn", "dnn")
            .build())

        assert hybrid is not None
        assert hybrid.get_metadata().model_id is not None

    def test_add_duplicate_component_name(self):
        """Test adding component with duplicate name."""
        snn1 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        snn2 = SNNTextModel(
            genome="30121033102301230112332100123",
            vocab_size=50
        )

        builder = HybridBuilder()
        builder.add("snn", snn1)

        # Adding with same name should replace
        builder.add("snn", snn2)

        # Should have only one "snn" component
        assert len(builder._components) == 1

    def test_connect_before_add(self):
        """Test connecting components before adding them."""
        builder = HybridBuilder()

        # Connect before adding (should be allowed, validated at build)
        builder.connect("comp1", "comp2")

        # Connections stored
        assert len(builder._connections) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
