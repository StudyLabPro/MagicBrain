"""Tests for ACT-MagicBrain integration."""
from __future__ import annotations

import numpy as np
import pytest

from magicbrain.integration.act_backend import ACTBackend


# ---------------------------------------------------------------------------
# Minimal genome string for TextBrain tests
# ---------------------------------------------------------------------------
MINIMAL_GENOME = (
    "N64_K8_seed42_p_long0.1_p_inhib0.2_alpha0.3_beta0.1_"
    "trace_fast_decay0.8_trace_slow_decay0.95_k_active8_"
    "dopamine_gain5.0_dopamine_bias0.0_lr0.01_"
    "cons_eps0.001_w_fast_decay0.99_prune_frac0.0_"
    "rewire_frac0.0_prune_every0_homeo0.001_buf_decay0.9"
)
VOCAB = 16


# ---------------------------------------------------------------------------
# ACTBackend unit tests
# ---------------------------------------------------------------------------
class TestACTBackendFallback:
    """Test that ACTBackend works gracefully without Balansis."""

    def test_act_backend_creates_without_error(self):
        backend = ACTBackend()
        assert isinstance(backend.available, bool)

    def test_act_backend_fallback_weight_update(self):
        backend = ACTBackend()
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = backend.weight_update(w, delta, 0.5)
        expected = w + 0.5 * delta
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_act_backend_fallback_softmax(self):
        backend = ACTBackend()
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        probs = backend.softmax(logits)
        assert probs.shape == logits.shape
        assert abs(float(np.sum(probs)) - 1.0) < 1e-5

    def test_act_backend_fallback_outer_product(self):
        backend = ACTBackend()
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = backend.outer_product(a, b)
        expected = np.outer(a, b)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_act_backend_fallback_dot(self):
        backend = ACTBackend()
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = backend.dot(a, b)
        assert abs(result - 32.0) < 1e-6


class TestACTBackendWithBalansis:
    """Test ACTBackend with Balansis available."""

    @pytest.fixture(autouse=True)
    def _require_balansis(self):
        backend = ACTBackend()
        if not backend.available:
            pytest.skip("Balansis not installed")
        self.backend = backend

    def test_act_backend_weight_update(self):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = self.backend.weight_update(w, delta, 0.5)
        expected = w + 0.5 * delta
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_act_backend_softmax(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        probs = self.backend.softmax(logits)
        assert probs.shape == logits.shape
        assert abs(float(np.sum(probs)) - 1.0) < 1e-4
        # Probabilities must be non-negative
        assert np.all(probs >= 0)

    def test_act_backend_outer_product(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = self.backend.outer_product(a, b)
        expected = np.outer(a, b)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_act_backend_dot(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = self.backend.dot(a, b)
        assert abs(result - 32.0) < 1e-4


# ---------------------------------------------------------------------------
# TextBrain integration tests
# ---------------------------------------------------------------------------
class TestBrainWithACT:
    """Test TextBrain with use_act flag."""

    def _make_brain(self, use_act: bool = False):
        from magicbrain.brain import TextBrain
        return TextBrain(MINIMAL_GENOME, VOCAB, seed_override=42, use_act=use_act)

    def test_brain_with_act_flag(self):
        brain = self._make_brain(use_act=True)
        assert brain._act is not None

    def test_brain_without_act_flag(self):
        brain = self._make_brain(use_act=False)
        assert brain._act is None

    def test_brain_training_without_act(self):
        brain = self._make_brain(use_act=False)
        losses = []
        for i in range(20):
            token = i % VOCAB
            probs = brain.forward(token)
            loss = brain.learn(token, probs)
            losses.append(loss)
        assert all(np.isfinite(l) for l in losses)

    def test_brain_training_with_act(self):
        brain = self._make_brain(use_act=True)
        losses = []
        for i in range(20):
            token = i % VOCAB
            probs = brain.forward(token)
            loss = brain.learn(token, probs)
            losses.append(loss)
        assert all(np.isfinite(l) for l in losses)

    def test_act_no_nan(self):
        """Training with ACT must never produce NaN even with extreme weights."""
        brain = self._make_brain(use_act=True)
        # Push weights to extreme values
        brain.w_slow[:] = 0.49
        brain.w_fast[:] = 0.49
        brain.R[:] = 0.99

        for i in range(30):
            token = i % VOCAB
            probs = brain.forward(token)
            loss = brain.learn(token, probs)
            assert np.isfinite(loss), f"NaN/Inf loss at step {i}"
            assert not np.any(np.isnan(brain.w_fast)), f"NaN in w_fast at step {i}"
            assert not np.any(np.isnan(brain.R)), f"NaN in R at step {i}"

    def test_both_modes_produce_similar_results(self):
        """ACT and non-ACT should produce numerically close results."""
        brain_std = self._make_brain(use_act=False)
        brain_act = self._make_brain(use_act=True)

        # Skip if Balansis not available (both will use numpy fallback)
        if brain_act._act is None or not brain_act._act.available:
            pytest.skip("Balansis not installed — both paths identical")

        for i in range(10):
            token = i % VOCAB
            probs_std = brain_std.forward(token)
            probs_act = brain_act.forward(token)
            brain_std.learn(token, probs_std)
            brain_act.learn(token, probs_act)

        # Weights should be in the same ballpark (not exact due to compensation)
        assert np.allclose(brain_std.w_fast, brain_act.w_fast, atol=0.05)
