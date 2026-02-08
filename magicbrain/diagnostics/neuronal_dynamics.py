"""
Neuronal dynamics analysis tools.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


class SpikeRaster:
    """
    Records and analyzes spike trains (neural activity over time).
    """

    def __init__(self, max_steps: int = 1000, subsample_neurons: int = 100):
        """
        Args:
            max_steps: Maximum number of time steps to record
            subsample_neurons: Number of neurons to track (for memory efficiency)
        """
        self.max_steps = max_steps
        self.subsample_neurons = subsample_neurons
        self.spike_history: List[np.ndarray] = []
        self.neuron_indices: np.ndarray = None

    def initialize(self, brain):
        """Initialize with brain to determine which neurons to track."""
        N = brain.N
        if N <= self.subsample_neurons:
            self.neuron_indices = np.arange(N)
        else:
            # Sample neurons uniformly
            self.neuron_indices = np.linspace(0, N - 1, self.subsample_neurons, dtype=int)

    def record(self, brain):
        """Record current spike state."""
        if self.neuron_indices is None:
            self.initialize(brain)

        # Record spikes for tracked neurons
        spikes = brain.a[self.neuron_indices].copy()
        self.spike_history.append(spikes)

        # Limit history size
        if len(self.spike_history) > self.max_steps:
            self.spike_history.pop(0)

    def get_raster(self) -> np.ndarray:
        """
        Get spike raster as 2D array.

        Returns:
            Array of shape (time_steps, num_neurons)
        """
        if not self.spike_history:
            return np.array([])
        return np.array(self.spike_history)

    def get_firing_rates(self) -> np.ndarray:
        """
        Compute average firing rate for each tracked neuron.

        Returns:
            Array of firing rates, shape (num_neurons,)
        """
        raster = self.get_raster()
        if raster.size == 0:
            return np.array([])
        return np.mean(raster, axis=0)

    def get_synchrony(self) -> float:
        """
        Measure population synchrony (correlation of firing).

        Returns:
            Synchrony score between 0 (independent) and 1 (synchronous)
        """
        raster = self.get_raster()
        if raster.size == 0 or raster.shape[0] < 2:
            return 0.0

        # Population firing rate at each time step
        pop_rate = np.mean(raster, axis=1)

        # Variance of population rate
        sync = float(np.std(pop_rate))
        return sync

    def get_burstiness(self) -> np.ndarray:
        """
        Measure burstiness for each neuron (coefficient of variation of ISI).

        Returns:
            Burstiness score for each tracked neuron
        """
        raster = self.get_raster()
        if raster.size == 0:
            return np.array([])

        burstiness = []
        for neuron_idx in range(raster.shape[1]):
            spike_times = np.where(raster[:, neuron_idx] > 0)[0]

            if len(spike_times) < 2:
                burstiness.append(0.0)
                continue

            # Inter-spike intervals
            isi = np.diff(spike_times)
            cv = np.std(isi) / (np.mean(isi) + 1e-6)
            burstiness.append(cv)

        return np.array(burstiness)


class ActivityTracker:
    """Tracks aggregate activity patterns over training."""

    def __init__(self):
        self.activity_history: List[float] = []
        self.trace_fast_history: List[float] = []
        self.trace_slow_history: List[float] = []

    def record(self, brain):
        """Record current activity statistics."""
        self.activity_history.append(float(np.mean(brain.a)))
        self.trace_fast_history.append(float(np.mean(np.abs(brain.trace_fast))))
        self.trace_slow_history.append(float(np.mean(np.abs(brain.trace_slow))))

    def get_activity_stats(self) -> dict:
        """Get statistics of neural activity."""
        if not self.activity_history:
            return {}

        return {
            "mean_activity": float(np.mean(self.activity_history)),
            "std_activity": float(np.std(self.activity_history)),
            "mean_trace_fast": float(np.mean(self.trace_fast_history)),
            "mean_trace_slow": float(np.mean(self.trace_slow_history)),
        }
