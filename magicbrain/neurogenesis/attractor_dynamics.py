"""Attractor dynamics for neurogenomic memory.

Implements continuous neural dynamics that converge to attractor states.
Each attractor encodes a stored memory pattern.

Key difference from TextBrain's forward pass:
  - TextBrain uses hard top-k selection (discrete jumps)
  - AttractorDynamics uses continuous energy minimization (smooth convergence)

s_{t+1} = (1-alpha) * s_t + alpha * sigma(W @ s_t + theta)

Convergence criterion: ||s_{t+1} - s_t|| < tolerance
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .energy import EnergyFunction


class ConvergenceResult(NamedTuple):
    """Result of attractor convergence."""
    state: np.ndarray
    converged: bool
    iterations: int
    final_energy: float
    energy_trajectory: list[float]


class Attractor(NamedTuple):
    """A discovered attractor state."""
    state: np.ndarray
    energy: float
    basin_size: int  # number of probes that converged here
    stability: float  # how stable under perturbation


class AttractorDynamics:
    """Continuous dynamics that converge to attractor states.

    Supports both dense weight matrix and sparse edge-list (as used by TextBrain).
    """

    def __init__(
        self,
        tau: float = 0.3,
        momentum: float = 0.7,
        lambda_sparse: float = 0.01,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
    ):
        self.tau = tau
        self.momentum = momentum
        self.energy_fn = EnergyFunction(lambda_sparse=lambda_sparse)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def step(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
    ) -> np.ndarray:
        """Single dynamics step: compute local field -> sigmoid -> momentum mix.

        Args:
            state: Current state (N,).
            weights: Weight matrix (N,N) or edge weights (E,).
            theta: Threshold/bias vector (N,).
            src: Source indices for sparse format.
            dst: Destination indices for sparse format.

        Returns:
            Updated state (N,).
        """
        h = self.energy_fn.local_field(state, weights, theta, src, dst)

        # Continuous activation with temperature (tau)
        new_state = _sigmoid_vec(h / self.tau)

        # Momentum mixing for smooth convergence
        mixed = (self.momentum * state + (1.0 - self.momentum) * new_state).astype(
            np.float32
        )

        return mixed

    def converge(
        self,
        cue: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        track_energy: bool = True,
    ) -> ConvergenceResult:
        """Run dynamics until convergence to an attractor.

        Args:
            cue: Initial state (partial memory cue).
            weights: Weight matrix or edge weights.
            theta: Thresholds.
            src, dst: Sparse edge indices.
            max_iterations: Override default max iterations.
            tolerance: Override default tolerance.
            track_energy: Whether to record energy at each step.

        Returns:
            ConvergenceResult with final state and diagnostics.
        """
        max_iter = max_iterations or self.max_iterations
        tol = tolerance or self.tolerance

        state = cue.copy().astype(np.float32)
        energy_trajectory: list[float] = []

        if track_energy:
            e0 = self.energy_fn.energy(state, weights, theta, src, dst)
            energy_trajectory.append(e0)

        for i in range(max_iter):
            prev = state.copy()
            state = self.step(state, weights, theta, src, dst)

            if track_energy:
                e = self.energy_fn.energy(state, weights, theta, src, dst)
                energy_trajectory.append(e)

            delta = float(np.max(np.abs(state - prev)))
            if delta < tol:
                final_e = energy_trajectory[-1] if energy_trajectory else (
                    self.energy_fn.energy(state, weights, theta, src, dst)
                )
                return ConvergenceResult(
                    state=state,
                    converged=True,
                    iterations=i + 1,
                    final_energy=final_e,
                    energy_trajectory=energy_trajectory,
                )

        final_e = energy_trajectory[-1] if energy_trajectory else (
            self.energy_fn.energy(state, weights, theta, src, dst)
        )
        return ConvergenceResult(
            state=state,
            converged=False,
            iterations=max_iter,
            final_energy=final_e,
            energy_trajectory=energy_trajectory,
        )

    def find_attractors(
        self,
        N: int,
        weights: np.ndarray,
        theta: np.ndarray,
        n_probes: int = 500,
        sparsity: float = 0.1,
        src: np.ndarray | None = None,
        dst: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        attractor_merge_threshold: float = 0.05,
    ) -> list[Attractor]:
        """Discover attractors by probing from random initial states.

        Launches convergence from random sparse states and clusters
        the resulting fixed points.

        Args:
            N: Number of neurons.
            weights: Weight matrix or edge weights.
            theta: Thresholds.
            n_probes: Number of random initial states to try.
            sparsity: Fraction of neurons active in initial probe.
            src, dst: Sparse format indices.
            rng: Random number generator.
            attractor_merge_threshold: Max L_inf distance to consider
                two attractors identical.

        Returns:
            List of unique Attractor objects, sorted by basin size.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Collect convergence results
        results: list[tuple[np.ndarray, float]] = []
        for _ in range(n_probes):
            cue = _random_sparse_state(N, sparsity, rng)
            cr = self.converge(
                cue, weights, theta, src, dst, track_energy=False
            )
            results.append((cr.state, cr.final_energy))

        # Cluster attractors by proximity
        attractors: list[dict] = []
        for state, energy in results:
            merged = False
            for att in attractors:
                dist = float(np.max(np.abs(state - att["state"])))
                if dist < attractor_merge_threshold:
                    att["basin_size"] += 1
                    # Keep lower-energy representative
                    if energy < att["energy"]:
                        att["state"] = state
                        att["energy"] = energy
                    merged = True
                    break
            if not merged:
                attractors.append({
                    "state": state,
                    "energy": energy,
                    "basin_size": 1,
                })

        # Compute stability for each attractor
        result_list: list[Attractor] = []
        for att in attractors:
            stability = self._measure_stability(
                att["state"], weights, theta, src, dst, rng
            )
            result_list.append(Attractor(
                state=att["state"],
                energy=att["energy"],
                basin_size=att["basin_size"],
                stability=stability,
            ))

        result_list.sort(key=lambda a: -a.basin_size)
        return result_list

    def _measure_stability(
        self,
        attractor: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        src: np.ndarray | None,
        dst: np.ndarray | None,
        rng: np.random.Generator,
        n_perturbations: int = 20,
        noise_scale: float = 0.05,
    ) -> float:
        """Measure attractor stability: fraction of perturbations that return."""
        returned = 0
        for _ in range(n_perturbations):
            noise = rng.normal(0, noise_scale, size=attractor.shape).astype(np.float32)
            perturbed = np.clip(attractor + noise, 0.0, 1.0)
            cr = self.converge(
                perturbed, weights, theta, src, dst, track_energy=False
            )
            dist = float(np.max(np.abs(cr.state - attractor)))
            if dist < self.tolerance * 10:
                returned += 1
        return returned / n_perturbations


def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arrays."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _random_sparse_state(
    N: int, sparsity: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate random sparse activation pattern."""
    state = np.zeros(N, dtype=np.float32)
    n_active = max(1, int(N * sparsity))
    active = rng.choice(N, size=n_active, replace=False)
    state[active] = rng.uniform(0.3, 1.0, size=n_active).astype(np.float32)
    return state
