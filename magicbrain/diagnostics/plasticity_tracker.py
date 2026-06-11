"""
Structural plasticity tracking (pruning/rewiring events).
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np


class PlasticityTracker:
    """Tracks structural plasticity events during training."""

    def __init__(self):
        self.pruning_events: List[Dict] = []
        self.rewiring_events: List[Dict] = []
        self.consolidation_history: List[float] = []

    def record_pruning(self, step: int, num_pruned: int, pruned_indices: np.ndarray):
        """
        Record a pruning event.

        Args:
            step: Training step
            num_pruned: Number of edges pruned
            pruned_indices: Indices of pruned edges
        """
        self.pruning_events.append({
            "step": step,
            "num_pruned": num_pruned,
            "pruned_indices": pruned_indices.tolist() if len(pruned_indices) < 100 else [],
        })

    def record_rewiring(self, step: int, num_rewired: int):
        """
        Record a rewiring event.

        Args:
            step: Training step
            num_rewired: Number of edges rewired
        """
        self.rewiring_events.append({
            "step": step,
            "num_rewired": num_rewired,
        })

    def record_consolidation(self, consolidation_amount: float):
        """
        Record weight consolidation (transfer from w_fast to w_slow).

        Args:
            consolidation_amount: Amount of weight transferred
        """
        self.consolidation_history.append(consolidation_amount)

    def get_total_pruned(self) -> int:
        """Total number of edges pruned across all events."""
        return sum(e["num_pruned"] for e in self.pruning_events)

    def get_total_rewired(self) -> int:
        """Total number of edges rewired across all events."""
        return sum(e["num_rewired"] for e in self.rewiring_events)

    def get_pruning_rate(self) -> float:
        """Average number of edges pruned per event."""
        if not self.pruning_events:
            return 0.0
        return self.get_total_pruned() / len(self.pruning_events)

    def get_rewiring_rate(self) -> float:
        """Average number of edges rewired per event."""
        if not self.rewiring_events:
            return 0.0
        return self.get_total_rewired() / len(self.rewiring_events)

    def get_consolidation_stats(self) -> Dict:
        """Statistics of consolidation process."""
        if not self.consolidation_history:
            return {}

        return {
            "mean_consolidation": float(np.mean(self.consolidation_history)),
            "std_consolidation": float(np.std(self.consolidation_history)),
            "total_consolidation": float(np.sum(self.consolidation_history)),
        }

    def get_summary(self) -> Dict:
        """Get summary of all plasticity events."""
        return {
            "total_pruning_events": len(self.pruning_events),
            "total_rewiring_events": len(self.rewiring_events),
            "total_pruned": self.get_total_pruned(),
            "total_rewired": self.get_total_rewired(),
            "avg_pruning_rate": self.get_pruning_rate(),
            "avg_rewiring_rate": self.get_rewiring_rate(),
            "consolidation_stats": self.get_consolidation_stats(),
        }


class StructuralMonitor:
    """Monitors structural changes in the network over time."""

    def __init__(self):
        self.edge_count_history: List[int] = []
        self.connectivity_history: List[float] = []

    def record(self, brain):
        """Record current structural state."""
        num_edges = len(brain.src)
        self.edge_count_history.append(num_edges)

        # Effective connectivity (considering weight magnitudes)
        w_total = np.abs(brain.w_slow + brain.w_fast)
        threshold = np.mean(w_total)
        effective_edges = int(np.sum(w_total > threshold))
        self.connectivity_history.append(effective_edges / num_edges if num_edges > 0 else 0.0)

    def get_edge_stability(self) -> float:
        """
        Measure stability of edge count over time.

        Returns:
            Coefficient of variation of edge count
        """
        if len(self.edge_count_history) < 2:
            return 0.0

        return float(np.std(self.edge_count_history) / (np.mean(self.edge_count_history) + 1e-6))

    def get_connectivity_trend(self) -> float:
        """
        Measure trend in effective connectivity.

        Returns:
            Slope of linear fit (positive = increasing, negative = decreasing)
        """
        if len(self.connectivity_history) < 2:
            return 0.0

        x = np.arange(len(self.connectivity_history))
        y = np.array(self.connectivity_history)

        # Simple linear regression
        slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-6)
        return float(slope)
