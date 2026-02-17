"""Pattern Memory: Hopfield-style associative memory with Storkey learning rule.

Stores patterns as attractors in the weight matrix. Each pattern becomes
a fixed point of the network dynamics.

The Storkey rule provides higher capacity (~0.14N patterns) than the
classic Hebbian rule (~0.14N / sqrt(log N)).

Pipeline:
  1. Convert text tokens to sparse neural patterns
  2. Imprint patterns into weight matrix using Storkey rule
  3. Recall via attractor convergence from partial cue
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatternQualityMetrics:
    """Quality metrics for pattern memory."""
    avg_recall_fidelity: float
    interference_score: float
    capacity_ratio: float


class ImprintResult(NamedTuple):
    """Result of pattern imprinting."""
    n_patterns: int
    weight_matrix: np.ndarray  # (N, N) dense weight matrix
    patterns: np.ndarray       # (M, N) stored patterns
    capacity_ratio: float      # patterns / theoretical_max


class RecallResult(NamedTuple):
    """Result of pattern recall."""
    recalled_pattern: np.ndarray
    matched_index: int       # index of closest stored pattern (-1 if none)
    similarity: float        # cosine similarity to best match
    iterations: int
    converged: bool


class PatternMemory:
    """Hopfield-style associative memory with Storkey learning.

    Stores binary-like patterns as attractors and retrieves them
    from partial/noisy cues via dynamics convergence.
    """

    def __init__(
        self,
        N: int,
        sparsity: float = 0.1,
        max_capacity_fraction: float = 0.12,
    ):
        """Initialize pattern memory.

        Args:
            N: Number of neurons.
            sparsity: Target fraction of active neurons per pattern.
            max_capacity_fraction: Max patterns as fraction of N (safety limit).
        """
        self.N = N
        self.sparsity = sparsity
        self.max_patterns = int(N * max_capacity_fraction)

        self.W = np.zeros((N, N), dtype=np.float32)
        self.patterns: list[np.ndarray] = []

    def text_to_pattern(
        self,
        token_ids: list[int],
        vocab_size: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Convert a sequence of token IDs to a sparse neural pattern.

        Uses a deterministic hash-like mapping: token sequence -> neuron subset.

        Args:
            token_ids: Sequence of token indices.
            vocab_size: Size of vocabulary.
            rng: Optional RNG (if None, uses deterministic mapping).

        Returns:
            Sparse binary pattern (N,).
        """
        pattern = np.zeros(self.N, dtype=np.float32)
        n_active = max(1, int(self.N * self.sparsity))

        # Deterministic: hash token sequence to select active neurons
        seed_val = 0
        for i, tid in enumerate(token_ids):
            seed_val = (seed_val * 31 + tid + i * 7) & 0x7FFFFFFF

        pat_rng = np.random.default_rng(seed_val)
        active_indices = pat_rng.choice(self.N, size=n_active, replace=False)
        pattern[active_indices] = 1.0

        return pattern

    @property
    def capacity_warning(self) -> bool:
        """Return True when n_stored exceeds theoretical Storkey capacity (0.14 * N)."""
        return self.n_stored > 0.14 * self.N

    def imprint_pattern(self, pattern: np.ndarray) -> bool:
        """Imprint a single pattern using Storkey learning rule.

        The Storkey rule:
          dW_ij = (1/N) * (p_i * p_j - p_i * h_j - h_i * p_j)
        where h_i = sum_{k!=i,j} W_ik * p_k

        Args:
            pattern: Binary-like pattern to store (N,).

        Returns:
            True if successfully stored, False if at capacity.
        """
        if len(self.patterns) >= self.max_patterns:
            return False

        # Use fast vectorized implementation
        self._imprint_fast(pattern)
        self.patterns.append(pattern.copy())

        if self.capacity_warning:
            logger.warning(
                "Pattern memory capacity exceeded: %d patterns stored "
                "(theoretical capacity ~%d for N=%d)",
                self.n_stored,
                self.theoretical_capacity,
                self.N,
            )

        return True

    def imprint_patterns_batch(
        self,
        patterns: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ImprintResult:
        """Imprint multiple patterns efficiently.

        Uses vectorized Storkey rule for batch imprinting.

        Args:
            patterns: Array of patterns (M, N).
            progress_callback: Optional callback(current, total) called after
                each pattern is imprinted.

        Returns:
            ImprintResult with weight matrix and metadata.
        """
        M = patterns.shape[0]
        n_to_store = min(M, self.max_patterns - self.n_stored)

        for m in range(n_to_store):
            self._imprint_fast(patterns[m])
            self.patterns.append(patterns[m].copy())
            if progress_callback is not None:
                progress_callback(m + 1, n_to_store)

        capacity_ratio = self.n_stored / max(1, self.max_patterns)

        return ImprintResult(
            n_patterns=n_to_store,
            weight_matrix=self.W.copy(),
            patterns=np.array(self.patterns),
            capacity_ratio=capacity_ratio,
        )

    def quality_metrics(self) -> PatternQualityMetrics:
        """Compute quality metrics for the current state of pattern memory.

        Returns:
            PatternQualityMetrics with recall fidelity, interference, capacity ratio.
        """
        if not self.patterns:
            return PatternQualityMetrics(
                avg_recall_fidelity=0.0,
                interference_score=0.0,
                capacity_ratio=0.0,
            )

        # Recall fidelity: try recalling each stored pattern
        total_sim = 0.0
        for pat in self.patterns:
            result = self.recall(pat)
            total_sim += result.similarity
        avg_fidelity = total_sim / len(self.patterns)

        # Interference score: average pairwise cosine similarity of stored patterns
        if len(self.patterns) >= 2:
            total_interference = 0.0
            count = 0
            for i in range(len(self.patterns)):
                for j in range(i + 1, len(self.patterns)):
                    total_interference += abs(
                        _cosine_similarity(self.patterns[i], self.patterns[j])
                    )
                    count += 1
            interference = total_interference / count if count > 0 else 0.0
        else:
            interference = 0.0

        capacity_ratio = self.n_stored / max(1, self.theoretical_capacity)

        return PatternQualityMetrics(
            avg_recall_fidelity=avg_fidelity,
            interference_score=interference,
            capacity_ratio=capacity_ratio,
        )

    def _imprint_fast(self, pattern: np.ndarray):
        """Fast (vectorized) Storkey imprinting with mean correction.

        Uses the covariance rule for sparse patterns: subtracts the mean
        activity level so that sparse patterns (10% active) don't all
        look the same in bipolar space.

        For sparsity=0.1, bipolar mean = 2*0.1 - 1 = -0.8. Without
        correction, all patterns are highly correlated (~0.64 overlap).
        """
        # Convert to bipolar: 0 -> -1, 1 -> +1
        p_raw = (2.0 * pattern - 1.0).astype(np.float32)

        # Mean-correct (covariance rule) for sparse pattern decorrelation
        mu = float(np.mean(p_raw))
        p = (p_raw - mu).astype(np.float32).reshape(-1, 1)

        N = float(self.N)

        # h_i = sum_k W_ik * p_k
        h = (self.W @ p.ravel()).reshape(-1, 1)

        # Outer product: (p_i - μ)(p_j - μ)
        pp = p @ p.T

        # Storkey correction terms
        ph = p @ h.T
        hp = h @ p.T

        dW = (pp - ph - hp) / N

        # Symmetrize and zero diagonal
        dW = 0.5 * (dW + dW.T)
        np.fill_diagonal(dW, 0.0)

        self.W += dW.astype(np.float32)

    def recall(
        self,
        cue: np.ndarray,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        tau: float = 0.1,
    ) -> RecallResult:
        """Recall a stored pattern from a (possibly noisy/partial) cue.

        Uses annealing: starts with high temperature for exploration,
        then cools to sharpen the attractor convergence.

        Args:
            cue: Initial state (noisy/partial version of stored pattern).
            max_iterations: Max dynamics steps.
            tolerance: Convergence threshold.
            tau: Final temperature for tanh activation (lower = sharper).

        Returns:
            RecallResult with recalled pattern and match info.
        """
        # Convert cue to bipolar and mean-correct for dynamics
        state_bp = (2.0 * cue - 1.0).astype(np.float32)
        mu = float(2.0 * self.sparsity - 1.0)
        state = (state_bp - mu).astype(np.float32)

        # Annealing: start warm, cool down
        tau_start = max(tau * 5, 0.5)
        tau_end = tau

        for it in range(max_iterations):
            prev = state.copy()

            # Anneal temperature
            progress = it / max(1, max_iterations - 1)
            current_tau = tau_start + (tau_end - tau_start) * progress

            # Compute local field
            h = self.W @ state

            # Tanh activation with annealing temperature
            new_state = np.tanh(
                np.clip(h / current_tau, -20, 20)
            ).astype(np.float32)

            # Less momentum for faster convergence
            state = (0.2 * prev + 0.8 * new_state).astype(np.float32)

            delta = float(np.max(np.abs(state - prev)))
            if delta < tolerance:
                break

        # Convert back to 0/1 for comparison
        state_bp_out = state + mu
        state_binary = ((state_bp_out + 1.0) / 2.0).astype(np.float32)
        state_binary = np.clip(state_binary, 0.0, 1.0)

        # Find closest stored pattern
        best_idx = -1
        best_sim = -1.0
        for i, pat in enumerate(self.patterns):
            sim = _cosine_similarity(state_binary, pat)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return RecallResult(
            recalled_pattern=state_binary,
            matched_index=best_idx,
            similarity=best_sim,
            iterations=it + 1,
            converged=delta < tolerance if max_iterations > 0 else False,
        )

    def recall_all_patterns(
        self, noise_level: float = 0.0, rng: np.random.Generator | None = None
    ) -> list[RecallResult]:
        """Try to recall all stored patterns.

        Args:
            noise_level: Add noise to cues to test robustness.
            rng: Random number generator for noise.

        Returns:
            List of RecallResult for each stored pattern.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        results = []
        for pat in self.patterns:
            cue = pat.copy()
            if noise_level > 0:
                # Flip some bits for noise
                flip_mask = rng.random(self.N) < noise_level
                cue[flip_mask] = 1.0 - cue[flip_mask]
            result = self.recall(cue)
            results.append(result)
        return results

    def capacity_test(
        self,
        rng: np.random.Generator | None = None,
        step: int = 5,
    ) -> dict:
        """Test memory capacity by imprinting increasing numbers of patterns.

        Returns dict with capacity curve (n_patterns -> recall_accuracy).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Fresh memory for testing
        test_mem = PatternMemory(self.N, self.sparsity, max_capacity_fraction=0.2)

        n_active = max(1, int(self.N * self.sparsity))
        results = {}

        for n_patterns in range(step, self.max_patterns + step, step):
            # Generate random patterns
            test_mem.W.fill(0.0)
            test_mem.patterns.clear()

            patterns = np.zeros((n_patterns, self.N), dtype=np.float32)
            for i in range(n_patterns):
                active = rng.choice(self.N, size=n_active, replace=False)
                patterns[i, active] = 1.0
                test_mem._imprint_fast(patterns[i])
                test_mem.patterns.append(patterns[i])

            # Test recall
            correct = 0
            for i in range(n_patterns):
                result = test_mem.recall(patterns[i])
                if result.matched_index == i and result.similarity > 0.8:
                    correct += 1

            accuracy = correct / n_patterns
            results[n_patterns] = accuracy

            if accuracy < 0.5:
                break

        return results

    @property
    def n_stored(self) -> int:
        return len(self.patterns)

    @property
    def theoretical_capacity(self) -> int:
        """Theoretical Storkey capacity: ~0.14 * N."""
        return int(0.14 * self.N)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
