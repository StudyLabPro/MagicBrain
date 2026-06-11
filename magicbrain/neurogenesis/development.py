"""Development Operator: Genome -> Neural Tissue.

The development operator "grows" a 3D neural network from a genome,
analogous to biological morphogenesis.

Three stages:
  1. Morphogenesis — 3D positions and base connectivity (uses existing graph.py)
  2. Synaptogenesis — CPPN generates spatially-patterned weights
  3. Maturation — threshold calibration and E/I balance tuning
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..genome import decode_genome
from ..graph import build_graph
from .cppn import CPPN
from .genome_v2 import decode_genome_v2

logger = logging.getLogger(__name__)


@dataclass
class DevelopmentMetrics:
    """Metrics from a neural tissue development run."""
    n_neurons: int
    n_edges: int
    weight_mean: float
    weight_std: float
    time_seconds: float
    cppn_used: bool


class NeuralTissue(NamedTuple):
    """A developed neural network structure."""
    # Topology
    N: int
    K: int
    pos: np.ndarray        # (N, 3) neuron positions
    src: np.ndarray        # (E,) source indices
    dst: np.ndarray        # (E,) destination indices
    delay: np.ndarray      # (E,) axonal delays
    idx_by_delay: list     # delay-indexed edge lists

    # Weights
    w_slow: np.ndarray     # (E,) long-term weights
    w_fast: np.ndarray     # (E,) fast plasticity weights
    is_inhib: np.ndarray   # (N,) inhibitory mask

    # Thresholds
    theta: np.ndarray      # (N,) initial thresholds

    # Hyperparameters
    params: dict

    # Development metadata
    cppn: CPPN | None
    genome_str: str


class DevelopmentOperator:
    """Grows a neural network from a genome string.

    Extends the basic genome->brain pipeline with CPPN-based weight generation
    for spatially structured, genome-determined connectivity patterns.
    """

    def develop(
        self,
        genome: str,
        vocab_size: int = 50,
        use_cppn: bool = True,
        cppn_seed: int | None = None,
        timeout_seconds: float = 300,
    ) -> NeuralTissue:
        """Full development pipeline: genome -> neural tissue.

        Args:
            genome: Base-4 genome string.
            vocab_size: Size of the vocabulary (for readout layer).
            use_cppn: If True, use CPPN for weight generation.
                If False, use random initialization (standard TextBrain behavior).
            cppn_seed: Optional seed for CPPN. If None, derived from genome.
            timeout_seconds: Maximum wall-clock time for development.

        Returns:
            NeuralTissue with fully configured network.

        Raises:
            TimeoutError: If development exceeds timeout_seconds.
        """
        t0 = time.time()

        # Decode genome
        if len(genome) >= 72:
            params = decode_genome_v2(genome)
            cppn_digits = params.get("cppn_digits", [])
        else:
            params = decode_genome(genome)
            cppn_digits = []

        N = int(params["N"])
        K = int(params["K"])
        seed = int(params["seed"])

        if cppn_seed is None:
            cppn_seed = seed

        # Stage 1: Morphogenesis
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError("Development timed out during morphogenesis")
        pos, src, dst, delay, idx_by_delay = self._morphogenesis(N, K, params, seed)

        # E/I assignment
        rng = np.random.default_rng(seed)
        p_inhib = float(params.get("p_inhib", 0.15))
        is_inhib = (rng.random(N) < p_inhib).astype(np.bool_)

        # Stage 2: Synaptogenesis
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError("Development timed out during synaptogenesis")

        cppn = None
        if use_cppn and len(genome) >= 24:
            try:
                cppn = self._create_cppn(cppn_digits, cppn_seed)
                w_slow = self._synaptogenesis_cppn(cppn, pos, src, dst, is_inhib)
            except Exception as e:
                logger.warning(
                    "CPPN synaptogenesis failed (%s), falling back to random weights",
                    e,
                )
                cppn = None
                w_slow = self._synaptogenesis_random(src, rng)
                inhib_src = is_inhib[src]
                w_slow[inhib_src] = -np.abs(w_slow[inhib_src])
        else:
            w_slow = self._synaptogenesis_random(src, rng)
            # Enforce E/I signs
            inhib_src = is_inhib[src]
            w_slow[inhib_src] = -np.abs(w_slow[inhib_src])

        w_fast = np.zeros_like(w_slow)

        # Stage 3: Maturation
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError("Development timed out during maturation")
        theta = self._maturation(N, w_slow, src, dst, params)

        # Validate tissue
        self._validate_tissue(w_slow, w_fast, theta)

        return NeuralTissue(
            N=N,
            K=K,
            pos=pos,
            src=src,
            dst=dst,
            delay=delay,
            idx_by_delay=idx_by_delay,
            w_slow=w_slow,
            w_fast=w_fast,
            is_inhib=is_inhib,
            theta=theta,
            params=params,
            cppn=cppn,
            genome_str=genome,
        )

    def _validate_tissue(
        self,
        w_slow: np.ndarray,
        w_fast: np.ndarray,
        theta: np.ndarray,
    ) -> None:
        """Validate tissue output: check for NaN and extreme weights."""
        for name, arr in [("w_slow", w_slow), ("w_fast", w_fast), ("theta", theta)]:
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN detected in {name} after development")
            if np.any(np.isinf(arr)):
                raise ValueError(f"Inf detected in {name} after development")
        # Weight bounds check
        max_abs_w = 100.0
        if np.any(np.abs(w_slow) > max_abs_w):
            logger.warning(
                "w_slow contains weights exceeding |%s|: max=%.4f",
                max_abs_w,
                float(np.max(np.abs(w_slow))),
            )
        if np.any(np.abs(w_fast) > max_abs_w):
            logger.warning(
                "w_fast contains weights exceeding |%s|: max=%.4f",
                max_abs_w,
                float(np.max(np.abs(w_fast))),
            )

    def develop_with_metrics(
        self,
        genome: str,
        vocab_size: int = 50,
        use_cppn: bool = True,
        cppn_seed: int | None = None,
        timeout_seconds: float = 300,
    ) -> tuple[NeuralTissue, DevelopmentMetrics]:
        """Develop tissue and return development metrics.

        Returns:
            Tuple of (NeuralTissue, DevelopmentMetrics).
        """
        t0 = time.time()
        tissue = self.develop(genome, vocab_size, use_cppn, cppn_seed, timeout_seconds)
        elapsed = time.time() - t0

        metrics = DevelopmentMetrics(
            n_neurons=tissue.N,
            n_edges=len(tissue.src),
            weight_mean=float(np.mean(tissue.w_slow)),
            weight_std=float(np.std(tissue.w_slow)),
            time_seconds=elapsed,
            cppn_used=tissue.cppn is not None,
        )

        return tissue, metrics

    def _morphogenesis(
        self, N: int, K: int, params: dict, seed: int
    ) -> tuple:
        """Stage 1: Spatial layout and base connectivity."""
        p_long = float(params.get("p_long", 0.04))
        return build_graph(N, K, p_long, seed)

    def _create_cppn(
        self, cppn_digits: list[int], seed: int
    ) -> CPPN:
        """Create CPPN from genome parameters."""
        if cppn_digits:
            return CPPN.from_genome_params(cppn_digits, seed=seed)
        return CPPN(seed=seed)

    def _synaptogenesis_cppn(
        self,
        cppn: CPPN,
        pos: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        is_inhib: np.ndarray,
    ) -> np.ndarray:
        """Stage 2a: CPPN-based weight generation."""
        return cppn.generate_weights(pos, src, dst, is_inhib)

    def _synaptogenesis_random(
        self, src: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Stage 2b: Random weight initialization (fallback)."""
        return rng.normal(0, 0.03, size=src.shape[0]).astype(np.float32)

    def _maturation(
        self,
        N: int,
        w_slow: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Stage 3: Threshold calibration.

        Sets initial thresholds based on mean incoming weight to each neuron,
        so that neurons start near their equilibrium firing rate.
        """
        theta = np.zeros(N, dtype=np.float32)

        # Compute mean absolute incoming weight per neuron
        incoming_sum = np.zeros(N, dtype=np.float32)
        incoming_count = np.zeros(N, dtype=np.float32)
        np.add.at(incoming_sum, dst, np.abs(w_slow))
        np.add.at(incoming_count, dst, 1.0)

        # Set threshold to fraction of mean incoming weight
        mask = incoming_count > 0
        theta[mask] = 0.5 * incoming_sum[mask] / incoming_count[mask]

        return theta

    def develop_and_build_brain(
        self,
        genome: str,
        vocab_size: int = 50,
        use_cppn: bool = True,
    ):
        """Develop tissue and build a TextBrain from it.

        Returns a TextBrain with CPPN-generated (or random) weights
        instead of the default random initialization.
        """
        from ..brain import TextBrain

        tissue = self.develop(genome, vocab_size, use_cppn)

        # Create TextBrain with standard init
        brain = TextBrain(genome, vocab_size)

        # Override weights with developed tissue
        brain.w_slow = tissue.w_slow.copy()
        brain.w_fast = tissue.w_fast.copy()
        brain.theta = tissue.theta.copy()

        return brain, tissue
