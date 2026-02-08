"""
Tests for diagnostics and monitoring system.
"""
import pytest
import numpy as np
from magicbrain.brain import TextBrain
from magicbrain.diagnostics import (
    LiveMonitor,
    TrainingMetrics,
    SpikeRaster,
    ActivityTracker,
    SynapticAnalyzer,
    PlasticityTracker,
)


def test_live_monitor_creation():
    """Test LiveMonitor initialization."""
    monitor = LiveMonitor(log_every=50)
    assert monitor.log_every == 50
    assert len(monitor.metrics_history) == 0


def test_live_monitor_recording():
    """Test recording metrics."""
    genome = "30121033102301230112332100123"
    brain = TextBrain(genome, vocab_size=10)

    monitor = LiveMonitor(log_every=10)

    # Record metrics
    metrics = monitor.record(brain, loss=1.5, step=0)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.step == 0
    assert metrics.loss == 1.5
    assert len(monitor.metrics_history) == 1


def test_live_monitor_should_log():
    """Test logging interval."""
    monitor = LiveMonitor(log_every=100)

    assert monitor.should_log(0)
    assert monitor.should_log(100)
    assert monitor.should_log(200)
    assert not monitor.should_log(50)
    assert not monitor.should_log(101)


def test_spike_raster():
    """Test spike raster recording."""
    genome = "30121033102301230112332100123"
    brain = TextBrain(genome, vocab_size=10)

    raster = SpikeRaster(max_steps=50, subsample_neurons=50)

    # Record some steps
    for _ in range(10):
        brain.forward(0)
        raster.record(brain)

    # Check raster
    raster_data = raster.get_raster()
    assert raster_data.shape[0] == 10  # 10 time steps
    assert raster_data.shape[1] <= 50  # At most 50 neurons

    # Firing rates should be reasonable
    firing_rates = raster.get_firing_rates()
    assert len(firing_rates) == raster_data.shape[1]


def test_activity_tracker():
    """Test activity tracking."""
    genome = "30121033102301230112332100123"
    brain = TextBrain(genome, vocab_size=10)

    tracker = ActivityTracker()

    # Record activity
    for _ in range(20):
        brain.forward(0)
        tracker.record(brain)

    stats = tracker.get_activity_stats()

    assert "mean_activity" in stats
    assert "std_activity" in stats
    assert "mean_trace_fast" in stats
    assert "mean_trace_slow" in stats


def test_synaptic_analyzer():
    """Test synaptic weight analysis."""
    genome = "30121033102301230112332100123"
    brain = TextBrain(genome, vocab_size=10)

    analyzer = SynapticAnalyzer()

    stats = analyzer.analyze_weights(brain)

    # Check all expected metrics
    assert "mean_w_slow" in stats
    assert "mean_w_fast" in stats
    assert "mean_excit" in stats
    assert "mean_inhib" in stats
    assert "ei_ratio" in stats
    assert "sparsity_total" in stats


def test_ei_invariant_check():
    """Test E/I sign invariant checking."""
    genome = "30121033102301230112332100123"
    brain = TextBrain(genome, vocab_size=10)

    analyzer = SynapticAnalyzer()

    # Should hold initially
    assert analyzer.check_ei_invariant(brain)

    # Train a bit and check again
    for _ in range(10):
        probs = brain.forward(0)
        brain.learn(1, probs)

    # Should still hold
    assert analyzer.check_ei_invariant(brain)


def test_plasticity_tracker():
    """Test plasticity tracking."""
    tracker = PlasticityTracker()

    # Record events
    tracker.record_pruning(step=100, num_pruned=50, pruned_indices=np.array([1, 2, 3]))
    tracker.record_rewiring(step=100, num_rewired=25)
    tracker.record_consolidation(0.001)

    # Check stats
    assert tracker.get_total_pruned() == 50
    assert tracker.get_total_rewired() == 25

    summary = tracker.get_summary()
    assert summary["total_pruning_events"] == 1
    assert summary["total_rewiring_events"] == 1
