"""
Spike-Timing Dependent Plasticity (STDP) learning rule.

STDP is a biologically plausible learning mechanism where synaptic strength
changes depend on the relative timing of pre- and post-synaptic spikes.

Key principle:
- If pre-synaptic spike occurs before post-synaptic spike → potentiation (strengthen)
- If post-synaptic spike occurs before pre-synaptic spike → depression (weaken)
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class STDPRule:
    """
    Standard STDP learning rule implementation.

    Based on classical exponential STDP window:
    - ΔW = A+ * exp(-Δt/τ+)  if Δt > 0 (potentiation)
    - ΔW = -A- * exp(Δt/τ-)  if Δt < 0 (depression)

    where Δt = t_post - t_pre
    """

    def __init__(
        self,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = -1.0,
        w_max: float = 1.0,
    ):
        """
        Args:
            a_plus: Amplitude of potentiation
            a_minus: Amplitude of depression
            tau_plus: Time constant for potentiation (ms)
            tau_minus: Time constant for depression (ms)
            w_min: Minimum weight value
            w_max: Maximum weight value
        """
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

    def compute_weight_change(
        self,
        dt: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weight changes based on spike time differences.

        Args:
            dt: Time differences (t_post - t_pre) for each synapse
            current_weights: Current synaptic weights

        Returns:
            Weight changes (ΔW)
        """
        dw = np.zeros_like(current_weights, dtype=np.float32)

        # Potentiation: pre before post (dt > 0)
        potentiation_mask = dt > 0
        if np.any(potentiation_mask):
            dw[potentiation_mask] = self.a_plus * np.exp(-dt[potentiation_mask] / self.tau_plus)

        # Depression: post before pre (dt < 0)
        depression_mask = dt < 0
        if np.any(depression_mask):
            dw[depression_mask] = -self.a_minus * np.exp(dt[depression_mask] / self.tau_minus)

        return dw

    def apply_update(
        self,
        weights: np.ndarray,
        weight_changes: np.ndarray,
    ) -> np.ndarray:
        """
        Apply weight changes with bounds checking.

        Args:
            weights: Current weights
            weight_changes: Computed weight changes

        Returns:
            Updated weights
        """
        new_weights = weights + weight_changes
        return np.clip(new_weights, self.w_min, self.w_max)


class TripletSTDP:
    """
    Triplet STDP - extends standard STDP with triplet interactions.

    Captures higher-order spike correlations for better long-term memory.
    Based on Pfister & Gerstner (2006).
    """

    def __init__(
        self,
        a2_plus: float = 0.005,
        a2_minus: float = 0.00525,
        a3_plus: float = 0.0001,
        a3_minus: float = 0.00015,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_x: float = 100.0,
        tau_y: float = 100.0,
        w_min: float = -1.0,
        w_max: float = 1.0,
    ):
        """
        Args:
            a2_plus: Pair-based potentiation amplitude
            a2_minus: Pair-based depression amplitude
            a3_plus: Triplet potentiation amplitude
            a3_minus: Triplet depression amplitude
            tau_plus: Fast potentiation time constant
            tau_minus: Fast depression time constant
            tau_x: Slow pre-synaptic trace time constant
            tau_y: Slow post-synaptic trace time constant
            w_min: Minimum weight
            w_max: Maximum weight
        """
        self.a2_plus = a2_plus
        self.a2_minus = a2_minus
        self.a3_plus = a3_plus
        self.a3_minus = a3_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.w_min = w_min
        self.w_max = w_max

        # Traces for triplet interactions
        self.r1 = None  # Fast pre-synaptic trace
        self.r2 = None  # Slow pre-synaptic trace
        self.o1 = None  # Fast post-synaptic trace
        self.o2 = None  # Slow post-synaptic trace

    def initialize_traces(self, n_synapses: int):
        """Initialize spike traces."""
        self.r1 = np.zeros(n_synapses, dtype=np.float32)
        self.r2 = np.zeros(n_synapses, dtype=np.float32)
        self.o1 = np.zeros(n_synapses, dtype=np.float32)
        self.o2 = np.zeros(n_synapses, dtype=np.float32)

    def update_traces(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0,
    ):
        """
        Update spike traces.

        Args:
            pre_spikes: Binary array of pre-synaptic spikes
            post_spikes: Binary array of post-synaptic spikes
            dt: Time step (ms)
        """
        if self.r1 is None:
            self.initialize_traces(len(pre_spikes))

        # Decay traces
        self.r1 *= np.exp(-dt / self.tau_plus)
        self.r2 *= np.exp(-dt / self.tau_x)
        self.o1 *= np.exp(-dt / self.tau_minus)
        self.o2 *= np.exp(-dt / self.tau_y)

        # Increment on spikes
        self.r1 += pre_spikes.astype(np.float32)
        self.r2 += pre_spikes.astype(np.float32)
        self.o1 += post_spikes.astype(np.float32)
        self.o2 += post_spikes.astype(np.float32)

    def compute_weight_change(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weight changes using triplet rule.

        Args:
            pre_spikes: Pre-synaptic spikes (binary)
            post_spikes: Post-synaptic spikes (binary)

        Returns:
            Weight changes
        """
        dw = np.zeros_like(pre_spikes, dtype=np.float32)

        # Potentiation: post-synaptic spike
        if np.any(post_spikes > 0):
            # Pair-based: depends on r1
            dw += post_spikes * (self.a2_plus * self.r1)
            # Triplet: depends on r1 and r2
            dw += post_spikes * (self.a3_plus * self.r1 * self.r2)

        # Depression: pre-synaptic spike
        if np.any(pre_spikes > 0):
            # Pair-based: depends on o1
            dw -= pre_spikes * (self.a2_minus * self.o1)
            # Triplet: depends on o1 and o2
            dw -= pre_spikes * (self.a3_minus * self.o1 * self.o2)

        return dw

    def apply_update(
        self,
        weights: np.ndarray,
        weight_changes: np.ndarray,
    ) -> np.ndarray:
        """Apply weight changes with bounds."""
        new_weights = weights + weight_changes
        return np.clip(new_weights, self.w_min, self.w_max)


class AdditiveSTDP(STDPRule):
    """
    Additive STDP - weight changes independent of current weight.
    Simple and stable, good for initial learning.
    """
    pass


class MultiplicativeSTDP(STDPRule):
    """
    Multiplicative STDP - weight changes depend on current weight.

    Potentiation: ΔW ∝ (w_max - w)
    Depression: ΔW ∝ (w - w_min)

    Leads to bimodal weight distribution (strong stability).
    """

    def compute_weight_change(
        self,
        dt: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute weight changes with multiplicative scaling."""
        dw = super().compute_weight_change(dt, current_weights)

        # Scale potentiation by distance from w_max
        potentiation_mask = dw > 0
        if np.any(potentiation_mask):
            scale = (self.w_max - current_weights[potentiation_mask]) / (self.w_max - self.w_min)
            dw[potentiation_mask] *= scale

        # Scale depression by distance from w_min
        depression_mask = dw < 0
        if np.any(depression_mask):
            scale = (current_weights[depression_mask] - self.w_min) / (self.w_max - self.w_min)
            dw[depression_mask] *= scale

        return dw


def create_stdp_rule(rule_type: str = "standard", **kwargs) -> STDPRule:
    """
    Factory function for creating STDP rules.

    Args:
        rule_type: One of 'standard', 'triplet', 'additive', 'multiplicative'
        **kwargs: Parameters for the rule

    Returns:
        STDP rule instance
    """
    if rule_type == "standard":
        return STDPRule(**kwargs)
    elif rule_type == "triplet":
        return TripletSTDP(**kwargs)
    elif rule_type == "additive":
        return AdditiveSTDP(**kwargs)
    elif rule_type == "multiplicative":
        return MultiplicativeSTDP(**kwargs)
    else:
        raise ValueError(f"Unknown STDP rule type: {rule_type}")
