"""Energy function for attractor-based neural dynamics.

Defines the energy landscape of a spiking neural network.
Attractors are local minima of the energy function.
Memory recall corresponds to convergence to an attractor.

E(s) = -1/2 * s^T W s - theta^T s + lambda * ||s||_1
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


class EnergyState(NamedTuple):
    """Energy decomposition at a given state."""
    total: float
    interaction: float   # -1/2 s^T W s (Hopfield term)
    bias: float          # -theta^T s
    sparsity: float      # lambda * ||s||_1


class EnergyFunction:
    """Computes energy of neural states in the weight landscape.

    The energy function combines three terms:
    1. Interaction energy (Hopfield): -1/2 s^T W s
    2. Bias/threshold energy: -theta^T s
    3. Sparsity penalty: lambda_sparse * ||s||_1

    Attractors are states where dE/ds = 0 and d^2E/ds^2 > 0.
    """

    def __init__(self, lambda_sparse: float = 0.01):
        self.lambda_sparse = lambda_sparse

    def energy(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
    ) -> float:
        """Compute total energy of a neural state.

        Supports both dense weight matrix and sparse edge-list formats.

        Args:
            state: Neural activation vector (N,).
            weights: Dense weight matrix (N,N) or sparse edge weights (E,).
            theta: Threshold vector (N,).
            src: Source indices for sparse format (E,).
            dst: Destination indices for sparse format (E,).

        Returns:
            Scalar energy value. Lower = more stable.
        """
        if src is not None and dst is not None:
            # Sparse edge-list format
            interaction = -0.5 * np.sum(
                state[src] * weights * state[dst]
            )
        else:
            # Dense matrix format
            interaction = -0.5 * float(state @ weights @ state)

        bias = -float(theta @ state)
        sparsity = self.lambda_sparse * float(np.sum(np.abs(state)))

        return float(interaction + bias + sparsity)

    def energy_decomposed(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
    ) -> EnergyState:
        """Compute energy with detailed decomposition."""
        if src is not None and dst is not None:
            interaction = -0.5 * float(np.sum(
                state[src] * weights * state[dst]
            ))
        else:
            interaction = -0.5 * float(state @ weights @ state)

        bias = -float(theta @ state)
        sparsity = self.lambda_sparse * float(np.sum(np.abs(state)))
        total = interaction + bias + sparsity

        return EnergyState(
            total=float(total),
            interaction=float(interaction),
            bias=float(bias),
            sparsity=float(sparsity),
        )

    def gradient(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute energy gradient dE/ds.

        For dense: dE/ds = -W s - theta + lambda * sign(s)
        For sparse: accumulate contributions from edge list.
        """
        N = state.shape[0]

        if src is not None and dst is not None:
            # Sparse: accumulate -W s per neuron
            field = np.zeros(N, dtype=np.float32)
            # Forward: contribution from s[src] * w to dst
            np.add.at(field, dst, -weights * state[src])
            # Symmetric: contribution from s[dst] * w to src
            np.add.at(field, src, -weights * state[dst])
            field *= 0.5
        else:
            field = -(weights @ state).astype(np.float32)

        grad = field - theta + self.lambda_sparse * np.sign(state)
        return grad.astype(np.float32)

    def local_field(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute local field h_i = sum_j W_ij s_j + theta_i.

        The local field drives activation: neurons with h_i > 0 tend to fire.
        """
        N = state.shape[0]

        if src is not None and dst is not None:
            h = np.zeros(N, dtype=np.float32)
            np.add.at(h, dst, weights * state[src])
        else:
            h = (weights @ state).astype(np.float32)

        return h + theta

    def is_stable(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        tolerance: float = 1e-4,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
    ) -> bool:
        """Check if state is a stable attractor (fixed point).

        A state is stable if one step of dynamics doesn't change it
        beyond tolerance.
        """
        h = self.local_field(state, weights, theta, src, dst)
        # For binary-like states: stable if sign(h) matches sign(s)
        # For continuous: stable if gradient magnitude is small
        grad = self.gradient(state, weights, theta, src, dst)
        return bool(np.max(np.abs(grad)) < tolerance)

    def basin_energy_profile(
        self,
        attractor: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        n_samples: int = 100,
        noise_scale: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Profile the energy basin around an attractor.

        Samples random perturbations and measures energy increase,
        giving a picture of how "deep" and "wide" the basin is.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        base_energy = self.energy(attractor, weights, theta)
        energies = []

        for _ in range(n_samples):
            noise = rng.normal(0, noise_scale, size=attractor.shape).astype(np.float32)
            perturbed = np.clip(attractor + noise, 0.0, 1.0)
            e = self.energy(perturbed, weights, theta)
            energies.append(e)

        energies_arr = np.array(energies)
        return {
            "attractor_energy": base_energy,
            "mean_perturbed_energy": float(np.mean(energies_arr)),
            "basin_depth": float(np.mean(energies_arr) - base_energy),
            "energy_std": float(np.std(energies_arr)),
            "min_barrier": float(np.min(energies_arr) - base_energy),
        }
