from __future__ import annotations
from typing import List
import numpy as np


class DataPartitioner:
    """Partitions text data for distributed training workers."""

    @staticmethod
    def round_robin(data: str, n_partitions: int) -> List[str]:
        """Distribute characters round-robin across partitions."""
        if n_partitions <= 0:
            raise ValueError("n_partitions must be positive")
        if not data:
            return [""] * n_partitions

        partitions: list[list[str]] = [[] for _ in range(n_partitions)]
        for i, ch in enumerate(data):
            partitions[i % n_partitions].append(ch)
        return ["".join(p) for p in partitions]

    @staticmethod
    def overlapping(data: str, n_partitions: int, overlap_chars: int = 50) -> List[str]:
        """Split data into roughly equal parts with overlap at boundaries."""
        if n_partitions <= 0:
            raise ValueError("n_partitions must be positive")
        if not data:
            return [""] * n_partitions
        if n_partitions == 1:
            return [data]

        total = len(data)
        base_size = total // n_partitions
        partitions = []

        for i in range(n_partitions):
            start = max(0, i * base_size - overlap_chars)
            if i == n_partitions - 1:
                end = total
            else:
                end = min(total, (i + 1) * base_size + overlap_chars)
            partitions.append(data[start:end])

        return partitions

    @staticmethod
    def shuffle_partition(
        data: str, n_partitions: int, chunk_size: int = 100, seed: int = 42
    ) -> List[str]:
        """Shuffle data chunks then distribute across partitions."""
        if n_partitions <= 0:
            raise ValueError("n_partitions must be positive")
        if not data:
            return [""] * n_partitions

        rng = np.random.default_rng(seed)

        # Split into chunks
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i : i + chunk_size])

        # Shuffle chunks
        indices = rng.permutation(len(chunks))
        shuffled = [chunks[i] for i in indices]

        # Distribute to partitions
        partitions: list[list[str]] = [[] for _ in range(n_partitions)]
        for i, chunk in enumerate(shuffled):
            partitions[i % n_partitions].append(chunk)

        return ["".join(p) for p in partitions]
