"""
Tests for STDP learning rules.
"""
import pytest
import numpy as np
from magicbrain.learning_rules import (
    STDPRule,
    TripletSTDP,
    MultiplicativeSTDP,
    create_stdp_rule,
    STDPBrain,
)
from magicbrain.tasks.text_task import build_vocab


def test_stdp_rule_creation():
    """Test STDP rule initialization."""
    rule = STDPRule(a_plus=0.01, a_minus=0.011, tau_plus=20.0, tau_minus=20.0)

    assert rule.a_plus == 0.01
    assert rule.a_minus == 0.011
    assert rule.tau_plus == 20.0
    assert rule.tau_minus == 20.0


def test_stdp_potentiation():
    """Test potentiation (pre before post)."""
    rule = STDPRule(a_plus=0.01, tau_plus=20.0)

    # Pre spike at t=0, post spike at t=10 → dt=10 → potentiation
    dt = np.array([10.0, 15.0, 5.0])
    weights = np.array([0.5, 0.5, 0.5])

    dw = rule.compute_weight_change(dt, weights)

    # All should be positive (potentiation)
    assert np.all(dw > 0)

    # Closer spikes should have larger effect
    assert dw[2] > dw[0] > dw[1]


def test_stdp_depression():
    """Test depression (post before pre)."""
    rule = STDPRule(a_minus=0.01, tau_minus=20.0)

    # Post spike at t=0, pre spike at t=10 → dt=-10 → depression
    dt = np.array([-10.0, -5.0, -15.0])
    weights = np.array([0.5, 0.5, 0.5])

    dw = rule.compute_weight_change(dt, weights)

    # All should be negative (depression)
    assert np.all(dw < 0)


def test_stdp_weight_bounds():
    """Test weight bounding."""
    rule = STDPRule(w_min=-1.0, w_max=1.0)

    weights = np.array([0.9, -0.9])
    weight_changes = np.array([0.5, -0.5])

    new_weights = rule.apply_update(weights, weight_changes)

    # Should be clipped to bounds
    assert new_weights[0] == 1.0
    assert new_weights[1] == -1.0


def test_triplet_stdp():
    """Test triplet STDP initialization."""
    rule = TripletSTDP()

    # Initialize traces
    rule.initialize_traces(n_synapses=100)

    assert rule.r1 is not None
    assert rule.r2 is not None
    assert rule.o1 is not None
    assert rule.o2 is not None
    assert len(rule.r1) == 100


def test_triplet_stdp_traces():
    """Test trace updates in triplet STDP."""
    rule = TripletSTDP(tau_plus=20.0, tau_minus=20.0)
    rule.initialize_traces(n_synapses=10)

    # Create spike pattern
    pre_spikes = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    post_spikes = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0])

    # Update traces
    rule.update_traces(pre_spikes, post_spikes, dt=1.0)

    # Traces should be non-zero where spikes occurred
    assert rule.r1[0] > 0  # Pre spike
    assert rule.o1[1] > 0  # Post spike


def test_multiplicative_stdp():
    """Test multiplicative STDP weight dependence."""
    rule = MultiplicativeSTDP(w_min=0.0, w_max=1.0)

    # Near maximum weight
    dt_pos = np.array([10.0])
    weights_high = np.array([0.9])

    dw_high = rule.compute_weight_change(dt_pos, weights_high)

    # Near minimum weight
    weights_low = np.array([0.1])
    dw_low = rule.compute_weight_change(dt_pos, weights_low)

    # Potentiation should be stronger for lower weights
    assert dw_low[0] > dw_high[0]


def test_create_stdp_factory():
    """Test STDP factory function."""
    # Standard STDP
    rule1 = create_stdp_rule("standard")
    assert isinstance(rule1, STDPRule)

    # Triplet STDP
    rule2 = create_stdp_rule("triplet")
    assert isinstance(rule2, TripletSTDP)

    # Multiplicative STDP
    rule3 = create_stdp_rule("multiplicative")
    assert isinstance(rule3, MultiplicativeSTDP)

    # Invalid type
    with pytest.raises(ValueError):
        create_stdp_rule("invalid")


def test_stdp_brain_creation():
    """Test STDPBrain initialization."""
    genome = "30121033102301230112332100123"
    brain = STDPBrain(genome, vocab_size=10, stdp_type="standard")

    assert brain.N > 0
    assert brain.vocab_size == 10
    assert hasattr(brain, "stdp_rule")
    assert hasattr(brain, "last_spike_time")


def test_stdp_brain_forward():
    """Test STDPBrain forward pass."""
    genome = "30121033102301230112332100123"
    brain = STDPBrain(genome, vocab_size=5)

    probs = brain.forward(0)

    assert probs.shape == (5,)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0)


def test_stdp_brain_learning():
    """Test STDPBrain learning."""
    genome = "30121033102301230112332100123"
    brain = STDPBrain(genome, vocab_size=5, stdp_type="standard")

    # Forward and learn
    probs = brain.forward(0)
    loss1 = brain.learn(1, probs)

    # Should update spike times
    assert brain.current_time == 1.0
    assert np.any(brain.last_spike_time > -np.inf)

    # Train a bit more
    for _ in range(10):
        probs = brain.forward(0)
        loss = brain.learn(1, probs)

    # Loss should change (learning happening)
    assert loss != loss1


def test_stdp_brain_reset():
    """Test spike time reset."""
    genome = "30121033102301230112332100123"
    brain = STDPBrain(genome, vocab_size=5)

    # Do some steps
    for _ in range(5):
        probs = brain.forward(0)
        brain.learn(1, probs)

    # Reset
    brain.reset_spike_times()

    assert brain.current_time == 0.0
    assert np.all(brain.last_spike_time == -np.inf)


def test_stdp_vs_dopamine_comparison():
    """Test comparison between STDP and dopamine learning."""
    from magicbrain.brain import TextBrain

    genome = "30121033102301230112332100123"
    text = "abcabc" * 20

    stoi, _ = build_vocab(text)

    # Create both brains
    dopamine_brain = TextBrain(genome, len(stoi))
    stdp_brain = STDPBrain(genome, len(stoi), stdp_type="standard")

    # Train briefly
    from magicbrain.tasks.text_task import train_loop_with_history

    dopamine_losses = train_loop_with_history(dopamine_brain, text, stoi, steps=50, verbose=False)
    stdp_losses = train_loop_with_history(stdp_brain, text, stoi, steps=50, verbose=False)

    # Both should reduce loss
    assert dopamine_losses[-1] < dopamine_losses[0]
    assert stdp_losses[-1] < stdp_losses[0]
