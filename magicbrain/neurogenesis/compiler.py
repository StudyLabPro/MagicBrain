"""GenomeCompiler: Dataset -> Genome.

Compiles a dataset into a compact neurogenome — a base-4 string that
deterministically produces a neural architecture tuned for the data.

Strategies:
  - hash:        SHA-256(data) -> base-4 (pseudorandom baseline)
  - statistical:  data statistics -> informed genome positions
  - hybrid:      statistics for key params, hash for the rest
  - auto:        pick strategy based on dataset characteristics
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

MAX_DATA_SIZE = 100 * 1024 * 1024  # 100 MB
MIN_DATA_LENGTH = 10


@dataclass
class CompilationMetrics:
    """Metrics from a genome compilation run."""
    time_seconds: float
    genome_quality_score: float
    strategy_used: str
    data_stats: dict = field(default_factory=dict)


class DatasetStats(NamedTuple):
    """Statistical fingerprint of a dataset."""
    size: int
    vocab_size: int
    entropy: float
    repetitiveness: float  # 0=unique, 1=fully repetitive
    mean_ngram_freq: float
    top_ngram_concentration: float  # fraction of text covered by top-10 bigrams


def analyze_dataset(text: str) -> DatasetStats:
    """Extract statistical features from text dataset."""
    size = len(text)
    chars = sorted(set(text))
    vocab_size = len(chars)

    # Shannon entropy
    counts = Counter(text)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Repetitiveness: ratio of unique bigrams to total bigrams
    if size >= 2:
        bigrams = [text[i:i + 2] for i in range(size - 1)]
        unique_bigrams = len(set(bigrams))
        max_possible = min(vocab_size * vocab_size, size - 1)
        repetitiveness = 1.0 - (unique_bigrams / max(1, max_possible))
    else:
        bigrams = []
        repetitiveness = 0.0

    # Mean bigram frequency
    if bigrams:
        bg_counts = Counter(bigrams)
        mean_ngram_freq = sum(bg_counts.values()) / len(bg_counts)
    else:
        mean_ngram_freq = 0.0

    # Top-10 bigram concentration
    if bigrams:
        bg_counts = Counter(bigrams)
        top10 = sum(c for _, c in bg_counts.most_common(10))
        top_ngram_concentration = top10 / len(bigrams)
    else:
        top_ngram_concentration = 0.0

    return DatasetStats(
        size=size,
        vocab_size=vocab_size,
        entropy=entropy,
        repetitiveness=repetitiveness,
        mean_ngram_freq=mean_ngram_freq,
        top_ngram_concentration=top_ngram_concentration,
    )


def _hex_to_base4(hex_str: str) -> str:
    """Convert hex string to base-4 string. Each hex digit -> 2 base-4 digits."""
    result = []
    for ch in hex_str:
        val = int(ch, 16)
        result.append(str(val // 4))
        result.append(str(val % 4))
    return "".join(result)


def _clamp_base4(value: int, max_val: int = 3) -> str:
    """Clamp integer to [0, max_val] and return as base-4 character."""
    return str(min(max_val, max(0, value)))


class GenomeCompiler:
    """Compiles a dataset into a neurogenome."""

    def compile(
        self,
        data: str | bytes,
        strategy: str = "statistical",
        genome_length: int = 28,
        hash_algorithm: str = "sha256",
        seed: Optional[int] = None,
    ) -> str:
        """Compile dataset into a genome string.

        Args:
            data: Input dataset (text or bytes).
            strategy: Compilation strategy ('hash', 'statistical', 'hybrid', 'auto').
            genome_length: Target genome length (minimum 24).
            hash_algorithm: Hash function for hash-based strategies.
            seed: Optional seed for deterministic results. When provided,
                appended to data hash for reproducible randomized sections.

        Returns:
            Base-4 genome string.
        """
        # Input validation
        if isinstance(data, str):
            if not data or not data.strip():
                raise ValueError("Input data is empty or whitespace-only")
            if len(data) < MIN_DATA_LENGTH:
                raise ValueError(
                    f"Input data too short: {len(data)} chars "
                    f"(minimum {MIN_DATA_LENGTH})"
                )
            data_bytes = data.encode("utf-8")
            text = data
        else:
            if not data:
                raise ValueError("Input data is empty")
            if len(data) < MIN_DATA_LENGTH:
                raise ValueError(
                    f"Input data too short: {len(data)} bytes "
                    f"(minimum {MIN_DATA_LENGTH})"
                )
            data_bytes = data
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")

        if len(data_bytes) > MAX_DATA_SIZE:
            raise ValueError(
                f"Input data too large: {len(data_bytes)} bytes "
                f"(maximum {MAX_DATA_SIZE})"
            )

        genome_length = max(24, genome_length)

        # Seed handling: modify data hash for reproducibility
        if seed is not None:
            seed_suffix = f"_seed{seed}".encode("utf-8")
            data_bytes_for_hash = data_bytes + seed_suffix
        else:
            data_bytes_for_hash = data_bytes

        # Auto strategy selection
        if strategy == "auto":
            strategy = self._auto_select_strategy(text)
            logger.info("Auto-selected strategy: %s", strategy)

        if strategy == "hash":
            return self._compile_hash(data_bytes_for_hash, genome_length, hash_algorithm)
        elif strategy == "statistical":
            return self._compile_statistical(text, genome_length, hash_algorithm, seed=seed)
        elif strategy == "hybrid":
            return self._compile_hybrid(text, genome_length, hash_algorithm, seed=seed)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _auto_select_strategy(self, text: str) -> str:
        """Pick compilation strategy based on dataset characteristics.

        - small data (< 1KB): hash (not enough stats signal)
        - medium data (1KB - 100KB): statistical
        - large data (> 100KB): hybrid (stats for architecture, hash for rest)
        """
        size = len(text)
        if size < 1024:
            return "hash"
        elif size < 100_000:
            return "statistical"
        else:
            return "hybrid"

    def compile_with_metrics(
        self,
        data: str | bytes,
        strategy: str = "statistical",
        genome_length: int = 28,
        hash_algorithm: str = "sha256",
        seed: Optional[int] = None,
    ) -> tuple[str, CompilationMetrics]:
        """Compile and return genome with compilation metrics.

        Returns:
            Tuple of (genome_string, CompilationMetrics).
        """
        t0 = time.time()

        if isinstance(data, str):
            text = data
        else:
            text = data.decode("utf-8", errors="replace")

        actual_strategy = strategy
        if strategy == "auto":
            actual_strategy = self._auto_select_strategy(text)

        genome = self.compile(data, strategy, genome_length, hash_algorithm, seed)

        elapsed = time.time() - t0

        stats = analyze_dataset(text)
        quality_score = self._compute_quality_score(genome, stats)

        metrics = CompilationMetrics(
            time_seconds=elapsed,
            genome_quality_score=quality_score,
            strategy_used=actual_strategy,
            data_stats=stats._asdict(),
        )

        return genome, metrics

    def _compute_quality_score(self, genome: str, stats: DatasetStats) -> float:
        """Compute a genome quality score (0-1) based on diversity and coverage."""
        if not genome:
            return 0.0
        # Digit diversity: how evenly distributed are base-4 digits
        counts = Counter(genome)
        n = len(genome)
        # Max entropy for 4 symbols: log2(4) = 2
        entropy = 0.0
        for c in "0123":
            p = counts.get(c, 0) / n
            if p > 0:
                entropy -= p * math.log2(p)
        diversity = entropy / 2.0  # normalize to 0-1

        # Length adequacy: longer genomes for larger data
        if stats.size > 0:
            ideal_length = min(128, max(24, int(math.log2(stats.size) * 4)))
            length_score = min(1.0, len(genome) / ideal_length)
        else:
            length_score = 0.5

        return 0.6 * diversity + 0.4 * length_score

    def _compile_hash(
        self, data: bytes, length: int, algorithm: str
    ) -> str:
        """Pure hash-based genome. Pseudorandom baseline."""
        h = hashlib.new(algorithm, data).hexdigest()
        base4 = _hex_to_base4(h)
        # SHA-256 gives 128 base-4 chars, more than enough
        if len(base4) < length:
            # Chain hashes if needed
            extra = hashlib.new(algorithm, data + b"_ext").hexdigest()
            base4 += _hex_to_base4(extra)
        return base4[:length]

    def _compile_statistical(
        self, text: str, length: int, algorithm: str, seed: Optional[int] = None
    ) -> str:
        """Statistics-driven genome. Key positions set from data properties."""
        stats = analyze_dataset(text)

        # Position 0-1: N (network size) — larger data -> larger network
        # N = 256 + 64 * b4(0,2), range 256-1216 for 2-digit base-4 (0-15)
        # Map data size logarithmically
        if stats.size > 0:
            size_code = min(15, int(math.log2(max(1, stats.size)) * 1.2))
        else:
            size_code = 4
        pos0 = str(size_code // 4)  # high digit
        pos1 = str(size_code % 4)   # low digit

        # Position 2: K (connectivity) — higher vocab -> more connections
        # K = 8 + b4*4, range 8-20
        k_code = min(3, stats.vocab_size // 25)
        pos2 = _clamp_base4(k_code)

        # Position 3: p_long — higher entropy -> more long-range
        # p_long = 0.02 + 0.02 * b4, range 0.02-0.08
        plong_code = min(3, int(stats.entropy / 2.0))
        pos3 = _clamp_base4(plong_code)

        # Position 4: lr — high repetitiveness -> higher lr (easier to learn)
        # lr = 0.0005 + 0.0005 * b4
        lr_code = min(3, int(stats.repetitiveness * 4))
        pos4 = _clamp_base4(lr_code)

        # Position 5: k_active — balanced: not too sparse, not too dense
        # Entropy-based: high entropy -> more active neurons needed
        kact_code = min(3, int(stats.entropy / 2.5))
        pos5 = _clamp_base4(kact_code)

        # Position 6: trace_fast_decay — repetitive data -> longer traces
        tfd_code = min(3, int(stats.repetitiveness * 3 + 0.5))
        pos6 = _clamp_base4(tfd_code)

        # Position 7: homeo — stable for structured data
        homeo_code = 1 if stats.entropy < 4.0 else 2
        pos7 = _clamp_base4(homeo_code)

        # Positions 8-11: seed from hash (topology randomization)
        hash_input = text.encode("utf-8")
        if seed is not None:
            hash_input += f"_seed{seed}".encode("utf-8")
        h = hashlib.new(algorithm, hash_input).hexdigest()
        hash_b4 = _hex_to_base4(h)
        pos8_11 = hash_b4[:4]

        # Position 10: trace_slow_decay — longer for larger datasets
        tsd_code = min(3, int(math.log2(max(1, stats.size)) / 5))
        pos10 = _clamp_base4(tsd_code)

        # Position 12: buf_decay — from hash
        pos12 = hash_b4[4]

        # Position 13-14: alpha/beta — concentration-based
        # High concentration -> rely more on fast trace
        alpha_code = min(3, int(stats.top_ngram_concentration * 6))
        beta_code = min(3, max(0, 3 - alpha_code))
        pos13 = _clamp_base4(alpha_code)
        pos14 = _clamp_base4(beta_code)

        # Position 15: p_inhib — from hash
        pos15 = hash_b4[5]

        # Position 16-17: dopamine — entropy-driven
        # Higher entropy -> higher gain (needs stronger learning signals)
        da_gain_code = min(3, int(stats.entropy / 2.0))
        da_bias_code = 1  # neutral
        pos16 = _clamp_base4(da_gain_code)
        pos17 = _clamp_base4(da_bias_code)

        # Positions 18-22: from hash (fine-tuning parameters)
        pos18_22 = hash_b4[6:11]

        # Assemble genome
        genome_chars = [
            pos0, pos1,     # N
            pos2,           # K
            pos3,           # p_long
            pos4,           # lr
            pos5,           # k_active
            pos6,           # trace_fast_decay
            pos7,           # homeo
        ]
        genome_chars.extend(list(pos8_11))  # seed (positions 8-11)
        # Override position 10 with statistical value
        genome_chars[10] = pos10
        genome_chars.append(pos12)          # buf_decay
        genome_chars.append(pos13)          # alpha
        genome_chars.append(pos14)          # beta
        genome_chars.append(pos15)          # p_inhib
        genome_chars.append(pos16)          # dopamine_gain
        genome_chars.append(pos17)          # dopamine_bias
        genome_chars.extend(list(pos18_22))  # cons_eps, w_fast_decay, prune, rewire

        genome = "".join(genome_chars)

        # Pad with hash if needed
        if len(genome) < length:
            remaining = hash_b4[11:11 + (length - len(genome))]
            genome += remaining

        return genome[:length]

    def _compile_hybrid(
        self, text: str, length: int, algorithm: str, seed: Optional[int] = None
    ) -> str:
        """Hybrid: statistics for architecture params, hash for dynamics."""
        stats = analyze_dataset(text)

        # Key architecture positions from statistics
        if stats.size > 0:
            size_code = min(15, int(math.log2(max(1, stats.size)) * 1.2))
        else:
            size_code = 4
        pos0 = str(size_code // 4)
        pos1 = str(size_code % 4)

        k_code = min(3, stats.vocab_size // 25)
        pos2 = _clamp_base4(k_code)

        plong_code = min(3, int(stats.entropy / 2.0))
        pos3 = _clamp_base4(plong_code)

        kact_code = min(3, int(stats.entropy / 2.5))
        pos5 = _clamp_base4(kact_code)

        # Rest from hash
        hash_input = text.encode("utf-8")
        if seed is not None:
            hash_input += f"_seed{seed}".encode("utf-8")
        h = hashlib.new(algorithm, hash_input).hexdigest()
        hash_b4 = _hex_to_base4(h)

        genome_chars = list(hash_b4[:length])
        # Override key positions with statistical values
        genome_chars[0] = pos0
        genome_chars[1] = pos1
        genome_chars[2] = pos2
        genome_chars[3] = pos3
        genome_chars[5] = pos5

        return "".join(genome_chars[:length])

    def compile_with_metadata(
        self,
        data: str | bytes,
        strategy: str = "statistical",
        genome_length: int = 28,
        hash_algorithm: str = "sha256",
        seed: Optional[int] = None,
    ) -> dict:
        """Compile and return genome with full metadata.

        Returns dict with genome, stats, hash, and strategy info.
        """
        if isinstance(data, str):
            text = data
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data
            text = data.decode("utf-8", errors="replace")

        genome = self.compile(data, strategy, genome_length, hash_algorithm, seed)
        stats = analyze_dataset(text)
        data_hash = hashlib.new(hash_algorithm, data_bytes).hexdigest()

        return {
            "genome": genome,
            "strategy": strategy,
            "data_hash": data_hash,
            "data_size": len(data_bytes),
            "stats": stats._asdict(),
            "genome_length": len(genome),
        }
