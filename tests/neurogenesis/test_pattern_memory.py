"""Tests for Hopfield-style pattern memory with Storkey learning rule."""

import pytest
import numpy as np
from magicbrain.neurogenesis.pattern_memory import PatternMemory, _cosine_similarity


class TestPatternMemory:
    def setup_method(self):
        self.N = 64
        self.mem = PatternMemory(N=self.N, sparsity=0.15)
        self.rng = np.random.default_rng(42)

    def _random_pattern(self) -> np.ndarray:
        p = np.zeros(self.N, dtype=np.float32)
        n_active = max(1, int(self.N * 0.15))
        active = self.rng.choice(self.N, size=n_active, replace=False)
        p[active] = 1.0
        return p

    def test_imprint_single(self):
        pattern = self._random_pattern()
        success = self.mem.imprint_pattern(pattern)
        assert success
        assert self.mem.n_stored == 1

    def test_imprint_up_to_capacity(self):
        for _ in range(self.mem.max_patterns):
            p = self._random_pattern()
            self.mem.imprint_pattern(p)
        assert self.mem.n_stored == self.mem.max_patterns

    def test_imprint_beyond_capacity_fails(self):
        for _ in range(self.mem.max_patterns):
            self.mem.imprint_pattern(self._random_pattern())
        # One more should fail
        result = self.mem.imprint_pattern(self._random_pattern())
        assert not result

    def test_weight_matrix_symmetric(self):
        self.mem.imprint_pattern(self._random_pattern())
        W = self.mem.W
        np.testing.assert_allclose(W, W.T, atol=1e-6)

    def test_weight_matrix_zero_diagonal(self):
        self.mem.imprint_pattern(self._random_pattern())
        assert np.all(np.diag(self.mem.W) == 0)

    def test_recall_perfect_cue(self):
        """Recalling with the exact stored pattern should match."""
        pattern = self._random_pattern()
        self.mem.imprint_pattern(pattern)
        result = self.mem.recall(pattern)
        assert result.matched_index == 0
        assert result.similarity > 0.5

    def test_recall_noisy_cue(self):
        """Recalling with a noisy cue should still find the stored pattern."""
        pattern = self._random_pattern()
        self.mem.imprint_pattern(pattern)

        # Add noise
        noisy = pattern.copy()
        flip_idx = self.rng.choice(self.N, size=5, replace=False)
        noisy[flip_idx] = 1.0 - noisy[flip_idx]

        result = self.mem.recall(noisy)
        assert result.matched_index == 0

    def test_recall_multiple_patterns(self):
        """Store 2 patterns in larger memory and recall each correctly."""
        # Use larger N for reliable multi-pattern recall
        big_mem = PatternMemory(N=256, sparsity=0.1)
        big_rng = np.random.default_rng(123)

        patterns = []
        n_active = max(1, int(256 * 0.1))
        for _ in range(2):
            p = np.zeros(256, dtype=np.float32)
            active = big_rng.choice(256, size=n_active, replace=False)
            p[active] = 1.0
            patterns.append(p)
            big_mem.imprint_pattern(p)

        for i, p in enumerate(patterns):
            result = big_mem.recall(p)
            assert result.matched_index == i

    def test_recall_convergence(self):
        pattern = self._random_pattern()
        self.mem.imprint_pattern(pattern)
        result = self.mem.recall(pattern)
        assert result.converged

    def test_text_to_pattern(self):
        pattern = self.mem.text_to_pattern([1, 2, 3], vocab_size=50)
        assert pattern.shape == (self.N,)
        n_active = int(np.sum(pattern > 0))
        expected = max(1, int(self.N * self.mem.sparsity))
        assert n_active == expected

    def test_text_to_pattern_deterministic(self):
        p1 = self.mem.text_to_pattern([1, 2, 3], vocab_size=50)
        p2 = self.mem.text_to_pattern([1, 2, 3], vocab_size=50)
        np.testing.assert_array_equal(p1, p2)

    def test_text_to_pattern_different_sequences(self):
        p1 = self.mem.text_to_pattern([1, 2, 3], vocab_size=50)
        p2 = self.mem.text_to_pattern([4, 5, 6], vocab_size=50)
        assert not np.array_equal(p1, p2)

    def test_batch_imprint(self):
        patterns = np.array([self._random_pattern() for _ in range(5)])
        result = self.mem.imprint_patterns_batch(patterns)
        assert result.n_patterns == 5
        assert result.weight_matrix.shape == (self.N, self.N)
        assert result.patterns.shape[0] == 5

    def test_recall_all_patterns(self):
        patterns = [self._random_pattern() for _ in range(3)]
        for p in patterns:
            self.mem.imprint_pattern(p)
        results = self.mem.recall_all_patterns()
        assert len(results) == 3

    def test_theoretical_capacity(self):
        assert self.mem.theoretical_capacity > 0
        assert self.mem.theoretical_capacity == int(0.14 * self.N)


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 0.0, 1.0])
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.zeros(3)
        b = np.ones(3)
        assert _cosine_similarity(a, b) == 0.0
