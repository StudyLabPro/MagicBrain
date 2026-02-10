"""Tests for CPPN weight generation."""

import pytest
import numpy as np
from magicbrain.neurogenesis.cppn import CPPN, CPPNLayer, BASIS_FUNCTIONS


class TestCPPN:
    def setup_method(self):
        self.cppn = CPPN(hidden_dims=[8, 4], seed=42)
        self.N = 32
        self.rng = np.random.default_rng(42)
        self.pos = self.rng.random((self.N, 3)).astype(np.float32)

    def test_query_shape(self):
        E = 100
        pos_src = self.rng.random((E, 3)).astype(np.float32)
        pos_dst = self.rng.random((E, 3)).astype(np.float32)
        dist = np.linalg.norm(pos_src - pos_dst, axis=1).astype(np.float32)
        type_src = self.rng.integers(0, 2, size=E).astype(np.float32)
        type_dst = self.rng.integers(0, 2, size=E).astype(np.float32)

        weights = self.cppn.query(pos_src, pos_dst, dist, type_src, type_dst)
        assert weights.shape == (E,)
        assert np.all(np.isfinite(weights))

    def test_generate_weights(self):
        # Build simple graph
        K = 4
        src = np.repeat(np.arange(self.N, dtype=np.int32), K)
        dst = self.rng.integers(0, self.N, size=self.N * K, dtype=np.int32)
        is_inhib = (self.rng.random(self.N) < 0.2).astype(np.bool_)

        weights = self.cppn.generate_weights(self.pos, src, dst, is_inhib)
        assert weights.shape == src.shape
        assert np.all(np.isfinite(weights))

        # E/I constraint: inhibitory source weights should be non-positive
        inhib_mask = is_inhib[src]
        if np.any(inhib_mask):
            assert np.all(weights[inhib_mask] <= 0)

    def test_deterministic(self):
        cppn1 = CPPN(seed=42)
        cppn2 = CPPN(seed=42)
        E = 20
        pos_s = self.rng.random((E, 3)).astype(np.float32)
        pos_d = self.rng.random((E, 3)).astype(np.float32)
        dist = np.ones(E, dtype=np.float32)
        types = np.zeros(E, dtype=np.float32)

        w1 = cppn1.query(pos_s, pos_d, dist, types, types)
        w2 = cppn2.query(pos_s, pos_d, dist, types, types)
        np.testing.assert_array_equal(w1, w2)

    def test_different_seeds_different_weights(self):
        cppn1 = CPPN(seed=1)
        cppn2 = CPPN(seed=2)
        E = 20
        pos_s = self.rng.random((E, 3)).astype(np.float32)
        pos_d = self.rng.random((E, 3)).astype(np.float32)
        dist = np.ones(E, dtype=np.float32)
        types = np.zeros(E, dtype=np.float32)

        w1 = cppn1.query(pos_s, pos_d, dist, types, types)
        w2 = cppn2.query(pos_s, pos_d, dist, types, types)
        assert not np.array_equal(w1, w2)

    def test_from_genome_params(self):
        digits = [1, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2]
        cppn = CPPN.from_genome_params(digits, seed=42)
        assert len(cppn.layers) > 0

        # Should be able to query
        E = 10
        pos_s = self.rng.random((E, 3)).astype(np.float32)
        pos_d = self.rng.random((E, 3)).astype(np.float32)
        dist = np.ones(E, dtype=np.float32)
        types = np.zeros(E, dtype=np.float32)
        w = cppn.query(pos_s, pos_d, dist, types, types)
        assert w.shape == (E,)

    def test_from_genome_params_empty(self):
        cppn = CPPN.from_genome_params([], seed=42)
        assert len(cppn.layers) > 0


class TestBasisFunctions:
    def test_all_basis_functions_valid(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        for fn in BASIS_FUNCTIONS:
            y = fn(x)
            assert y.shape == x.shape
            assert np.all(np.isfinite(y))
