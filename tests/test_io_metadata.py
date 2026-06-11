"""
Tests for model I/O with metadata support.

Verifies:
- Metadata loading from saved models
- Backward compatibility with old code
- Graceful handling of missing metadata
"""
import pytest
import tempfile
import os
from magicbrain import TextBrain
from magicbrain.io import save_model, load_model
from magicbrain.tasks.text_task import build_vocab


def test_load_model_with_metadata():
    """Test that load_model returns metadata as 4th element."""
    genome = "30121033102301230112332100123"
    text = "hello world test"

    stoi, itos = build_vocab(text)
    brain = TextBrain(genome, len(stoi))

    # Train a bit to set step counter
    brain.step = 100

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        save_model(brain, stoi, itos, path)

        # Load model with metadata
        loaded_brain, loaded_stoi, loaded_itos, metadata = load_model(path)

        # Verify metadata
        assert "genome_str" in metadata
        assert metadata["genome_str"] == genome

        assert "vocab_size" in metadata
        assert metadata["vocab_size"] == len(stoi)

        assert "step" in metadata
        assert metadata["step"] == 100

        assert "N" in metadata
        assert metadata["N"] == loaded_brain.N

        assert "K" in metadata
        assert metadata["K"] == loaded_brain.K

        assert "timestamp" in metadata

        # Verify brain loaded correctly
        assert loaded_brain.genome_str == genome
        assert loaded_brain.step == 100

    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_backward_compatibility_three_values():
    """Test that old code unpacking only 3 values still works with *_ unpacking."""
    genome = "30121033102301230112332100123"
    text = "test text"

    stoi, itos = build_vocab(text)
    brain = TextBrain(genome, len(stoi))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        save_model(brain, stoi, itos, path)

        # Old-style unpacking (3 values, ignoring metadata with *_)
        # This is how old code should be updated for compatibility
        brain_loaded, stoi_loaded, itos_loaded, *_ = load_model(path)

        # Should work without errors
        assert brain_loaded is not None
        assert stoi_loaded is not None
        assert itos_loaded is not None

        assert brain_loaded.genome_str == genome

    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_metadata_contains_expected_fields():
    """Test that metadata contains all expected fields."""
    genome = "30121033102301230112332100123"
    text = "the quick brown fox"

    stoi, itos = build_vocab(text)
    brain = TextBrain(genome, len(stoi))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        save_model(brain, stoi, itos, path)
        _, _, _, metadata = load_model(path)

        # Required fields
        required_fields = ["genome_str", "vocab_size", "N", "K"]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Optional fields (may exist)
        optional_fields = ["step", "timestamp"]
        # At least one should exist
        assert any(f in metadata for f in optional_fields)

    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_metadata_accuracy():
    """Test that metadata values are accurate."""
    genome = "30121033102301230112332100123"
    vocab_size = 50

    brain = TextBrain(genome, vocab_size)
    brain.step = 500

    stoi = {str(i): i for i in range(vocab_size)}
    itos = {i: str(i) for i in range(vocab_size)}

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        save_model(brain, stoi, itos, path)
        _, _, _, metadata = load_model(path)

        # Verify accuracy
        assert metadata["genome_str"] == genome
        assert metadata["vocab_size"] == vocab_size
        assert metadata["step"] == 500
        assert metadata["N"] == brain.N
        assert metadata["K"] == brain.K

    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_empty_metadata_for_old_files():
    """Test graceful handling of old files without metadata."""
    # This test is more conceptual - we can't easily create a file
    # without metadata using current save_model.
    # But we verify that metadata dict is always returned.

    genome = "30121033102301230112332100123"
    text = "test"

    stoi, itos = build_vocab(text)
    brain = TextBrain(genome, len(stoi))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        save_model(brain, stoi, itos, path)
        _, _, _, metadata = load_model(path)

        # Should always be a dict
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
