"""
Synaptic weight analysis and E/I balance monitoring.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


class SynapticAnalyzer:
    """Analyzes synaptic weights and connectivity patterns."""

    def __init__(self):
        self.weight_history: List[Dict] = []

    def analyze_weights(self, brain) -> Dict:
        """
        Analyze current weight distribution.

        Returns:
            Dictionary with weight statistics
        """
        w_total = brain.w_slow + brain.w_fast

        # Excitatory vs inhibitory
        inhib_mask = brain.is_inhib[brain.src]
        excit_mask = ~inhib_mask

        stats = {
            # Overall statistics
            "mean_w_slow": float(np.mean(brain.w_slow)),
            "std_w_slow": float(np.std(brain.w_slow)),
            "mean_w_fast": float(np.mean(brain.w_fast)),
            "std_w_fast": float(np.std(brain.w_fast)),
            "mean_w_total": float(np.mean(w_total)),
            "std_w_total": float(np.std(w_total)),

            # Absolute weights
            "mean_abs_w_slow": float(np.mean(np.abs(brain.w_slow))),
            "mean_abs_w_fast": float(np.mean(np.abs(brain.w_fast))),
            "mean_abs_w_total": float(np.mean(np.abs(w_total))),

            # Excitatory weights
            "mean_excit": float(np.mean(w_total[excit_mask])) if excit_mask.any() else 0.0,
            "std_excit": float(np.std(w_total[excit_mask])) if excit_mask.any() else 0.0,

            # Inhibitory weights
            "mean_inhib": float(np.mean(w_total[inhib_mask])) if inhib_mask.any() else 0.0,
            "std_inhib": float(np.std(w_total[inhib_mask])) if inhib_mask.any() else 0.0,

            # E/I balance
            "ei_ratio": self.compute_ei_ratio(brain),

            # Sparsity
            "sparsity_slow": self.compute_sparsity(brain.w_slow),
            "sparsity_fast": self.compute_sparsity(brain.w_fast),
            "sparsity_total": self.compute_sparsity(w_total),
        }

        return stats

    def compute_ei_ratio(self, brain) -> float:
        """
        Compute excitatory/inhibitory balance.

        Returns:
            Ratio of excitatory to inhibitory strength
        """
        w_total = brain.w_slow + brain.w_fast
        inhib_mask = brain.is_inhib[brain.src]
        excit_mask = ~inhib_mask

        if not inhib_mask.any():
            return np.inf

        excit_strength = float(np.sum(np.abs(w_total[excit_mask])))
        inhib_strength = float(np.sum(np.abs(w_total[inhib_mask])))

        if inhib_strength < 1e-9:
            return np.inf

        return excit_strength / inhib_strength

    def compute_sparsity(self, weights, threshold=1e-4) -> float:
        """
        Compute sparsity of weights (fraction near zero).

        Args:
            weights: Weight array
            threshold: Values below this are considered zero

        Returns:
            Fraction of weights that are effectively zero
        """
        return float(np.mean(np.abs(weights) < threshold))

    def record(self, brain):
        """Record current weight statistics."""
        stats = self.analyze_weights(brain)
        stats['step'] = brain.step
        self.weight_history.append(stats)

    def get_weight_evolution(self, metric: str) -> List[float]:
        """
        Get evolution of specific metric over time.

        Args:
            metric: Name of metric (e.g., 'mean_w_total', 'ei_ratio')

        Returns:
            List of values over time
        """
        return [h[metric] for h in self.weight_history if metric in h]

    def check_ei_invariant(self, brain) -> bool:
        """
        Verify E/I sign invariant (inhibitory weights should be non-positive).

        Returns:
            True if invariant holds, False otherwise
        """
        inhib_mask = brain.is_inhib[brain.src]
        w_total = brain.w_slow + brain.w_fast

        # Check if all inhibitory weights are non-positive
        if inhib_mask.any():
            max_inhib = np.max(w_total[inhib_mask])
            return max_inhib <= 1e-6
        return True


class ConnectivityAnalyzer:
    """Analyzes graph connectivity patterns."""

    def analyze_connectivity(self, brain) -> Dict:
        """
        Analyze connectivity statistics.

        Returns:
            Dictionary with connectivity metrics
        """
        N = brain.N
        num_edges = len(brain.src)

        # Degree distribution
        in_degree = np.bincount(brain.dst, minlength=N)
        out_degree = np.bincount(brain.src, minlength=N)

        # Weight-based effective connectivity
        w_total = brain.w_slow + brain.w_fast
        strong_edges = np.abs(w_total) > np.mean(np.abs(w_total))

        return {
            "num_edges": num_edges,
            "density": num_edges / (N * N),
            "mean_in_degree": float(np.mean(in_degree)),
            "std_in_degree": float(np.std(in_degree)),
            "mean_out_degree": float(np.mean(out_degree)),
            "std_out_degree": float(np.std(out_degree)),
            "strong_edges": int(np.sum(strong_edges)),
            "strong_edge_fraction": float(np.mean(strong_edges)),
        }
