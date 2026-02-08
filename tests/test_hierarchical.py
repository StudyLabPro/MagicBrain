"""
Tests for hierarchical architectures.
"""
import pytest
import numpy as np
from magicbrain.architectures import HierarchicalBrain, ModularBrain


def test_hierarchical_brain_creation():
    """Test HierarchicalBrain initialization."""
    genomes = [
        "30121033102301230112332100123",
        "10220133102301230112332100123",
        "20121033102301230112332100123",
    ]

    brain = HierarchicalBrain(genomes, vocab_size=10, seed=42)

    assert brain.num_layers == 3
    assert len(brain.layers) == 3
    assert brain.vocab_size == 10


def test_hierarchical_brain_timescales():
    """Test different timescales for layers."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(
        genomes,
        vocab_size=10,
        timescale_factors=[1.0, 2.0],
        seed=42
    )

    # Second layer should have slower dynamics
    assert brain.layers[1].p["trace_fast_decay"] > brain.layers[0].p["trace_fast_decay"]


def test_hierarchical_brain_forward():
    """Test forward pass through hierarchy."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(genomes, vocab_size=5, seed=42)

    probs = brain.forward(0)

    assert probs.shape == (5,)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0)


def test_hierarchical_brain_learning():
    """Test learning in hierarchical brain."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(genomes, vocab_size=5, seed=42)

    # Forward and learn
    probs = brain.forward(0)
    loss = brain.learn(1, probs)

    assert loss > 0
    assert brain.step == 1


def test_hierarchical_brain_layer_states():
    """Test getting layer states."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(genomes, vocab_size=5, seed=42)

    brain.forward(0)
    states = brain.get_layer_states()

    assert len(states) == 2
    assert all(isinstance(s, np.ndarray) for s in states)


def test_hierarchical_brain_activities():
    """Test getting layer activities."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(genomes, vocab_size=5, seed=42)

    brain.forward(0)
    activities = brain.get_layer_activities()

    assert len(activities) == 2
    assert all(isinstance(a, float) for a in activities)
    assert all(0 <= a <= 1 for a in activities)


def test_hierarchical_brain_skip_connections():
    """Test skip connections."""
    genomes = ["30121033102301230112332100123"] * 3

    # With skip connections
    brain_with_skip = HierarchicalBrain(
        genomes, vocab_size=5, skip_connections=True, seed=42
    )

    # Without skip connections
    brain_without_skip = HierarchicalBrain(
        genomes, vocab_size=5, skip_connections=False, seed=42
    )

    assert len(brain_with_skip.skip_weights) > 0
    assert len(brain_without_skip.skip_weights) == 0


def test_hierarchical_brain_reset():
    """Test state reset."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(genomes, vocab_size=5, seed=42)

    # Do some steps
    for _ in range(5):
        brain.forward(0)

    # Reset
    brain.reset_state()

    # All layers should have zero activity
    for layer in brain.layers:
        assert np.allclose(layer.a, 0.0)


def test_modular_brain_creation():
    """Test ModularBrain initialization."""
    genome = "30121033102301230112332100123"

    brain = ModularBrain(
        genome_sensory=genome,
        genome_memory=genome,
        genome_action=genome,
        genome_controller=genome,
        vocab_size=10,
        seed=42
    )

    assert hasattr(brain, "sensory")
    assert hasattr(brain, "memory")
    assert hasattr(brain, "action")
    assert hasattr(brain, "controller")


def test_modular_brain_connections():
    """Test inter-module connections."""
    genome = "30121033102301230112332100123"

    brain = ModularBrain(
        genome_sensory=genome,
        genome_memory=genome,
        genome_action=genome,
        genome_controller=genome,
        vocab_size=10,
        seed=42
    )

    # Check connection matrices exist
    assert hasattr(brain, "w_sens_mem")
    assert hasattr(brain, "w_mem_act")
    assert hasattr(brain, "w_ctrl_sens")

    # Check shapes
    assert brain.w_sens_mem.shape == (brain.sensory.N, brain.memory.N)
    assert brain.w_mem_act.shape == (brain.memory.N, brain.action.N)


def test_modular_brain_forward():
    """Test forward pass through modules."""
    genome = "30121033102301230112332100123"

    brain = ModularBrain(
        genome_sensory=genome,
        genome_memory=genome,
        genome_action=genome,
        genome_controller=genome,
        vocab_size=5,
        seed=42
    )

    probs = brain.forward(0)

    assert probs.shape == (5,)
    assert np.isclose(np.sum(probs), 1.0)


def test_modular_brain_learning():
    """Test learning in modular brain."""
    genome = "30121033102301230112332100123"

    brain = ModularBrain(
        genome_sensory=genome,
        genome_memory=genome,
        genome_action=genome,
        genome_controller=genome,
        vocab_size=5,
        seed=42
    )

    probs = brain.forward(0)
    loss = brain.learn(1, probs)

    assert loss > 0
    assert brain.step == 1


def test_hierarchical_training_stability():
    """Test that hierarchical brain trains stably."""
    genomes = ["30121033102301230112332100123"] * 2

    brain = HierarchicalBrain(genomes, vocab_size=5, seed=42)

    losses = []
    for _ in range(20):
        probs = brain.forward(0)
        loss = brain.learn(1, probs)
        losses.append(loss)

    # Should not diverge
    assert all(np.isfinite(l) for l in losses)
    assert max(losses) < 10.0
