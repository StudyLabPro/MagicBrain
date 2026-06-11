"""Tests for DevelopmentOperator: Genome -> Neural Tissue."""

import pytest
import numpy as np
from magicbrain.neurogenesis.development import DevelopmentOperator, NeuralTissue
from magicbrain.genome import decode_genome


DEFAULT_GENOME = "30121033102301230112332100123"


class TestDevelopmentOperator:
    def setup_method(self):
        self.dev = DevelopmentOperator()

    def test_develop_returns_tissue(self):
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=False)
        assert isinstance(tissue, NeuralTissue)

    def test_tissue_dimensions(self):
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=False)
        params = decode_genome(DEFAULT_GENOME)
        expected_N = params["N"]
        assert tissue.N == expected_N
        assert tissue.pos.shape == (expected_N, 3)
        assert tissue.src.shape[0] > 0
        assert tissue.dst.shape[0] == tissue.src.shape[0]
        assert tissue.w_slow.shape[0] == tissue.src.shape[0]
        assert tissue.w_fast.shape[0] == tissue.src.shape[0]
        assert tissue.theta.shape == (expected_N,)
        assert tissue.is_inhib.shape == (expected_N,)

    def test_develop_with_cppn(self):
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=True)
        assert tissue.cppn is not None
        assert np.all(np.isfinite(tissue.w_slow))

    def test_develop_without_cppn(self):
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=False)
        assert tissue.cppn is None

    def test_ei_constraint(self):
        """Inhibitory neurons should have non-positive weights."""
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=True)
        inhib_src = tissue.is_inhib[tissue.src]
        if np.any(inhib_src):
            assert np.all(tissue.w_slow[inhib_src] <= 0)

    def test_delays_valid(self):
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=False)
        assert np.all(tissue.delay >= 1)
        assert np.all(tissue.delay <= 5)

    def test_theta_nonnegative(self):
        tissue = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=True)
        assert np.all(tissue.theta >= 0)

    def test_deterministic(self):
        t1 = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=True)
        t2 = self.dev.develop(DEFAULT_GENOME, vocab_size=50, use_cppn=True)
        np.testing.assert_array_equal(t1.w_slow, t2.w_slow)
        np.testing.assert_array_equal(t1.pos, t2.pos)

    def test_develop_and_build_brain(self):
        brain, tissue = self.dev.develop_and_build_brain(
            DEFAULT_GENOME, vocab_size=50, use_cppn=True
        )
        assert brain.N == tissue.N
        # Brain should have the CPPN-generated weights
        np.testing.assert_array_equal(brain.w_slow, tissue.w_slow)

    def test_develop_and_build_brain_forward(self):
        """Brain built from development should support forward pass."""
        brain, _ = self.dev.develop_and_build_brain(
            DEFAULT_GENOME, vocab_size=50, use_cppn=True
        )
        probs = brain.forward(0)
        assert probs.shape == (50,)
        assert abs(float(np.sum(probs)) - 1.0) < 1e-5
