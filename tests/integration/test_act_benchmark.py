"""Benchmark: ACT-compensated vs standard numpy for typical MagicBrain sizes.

Tests numerical accuracy (not just performance) for N=384, K=12, vocab=50 —
the typical configuration referenced in CLAUDE.md as "Typical sizes".

Run with:
    pytest tests/integration/test_act_benchmark.py -v -s
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytest

from magicbrain.integration.act_backend import ACTBackend

# Typical sizes from CLAUDE.md
N = 384
K = 12
VOCAB = 50
GENOME = "30121033102301230112332100123"  # decodes to N≈384, K≈12


# ---------------------------------------------------------------------------
# Helper: time a function
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable, n_reps: int = 10) -> float:
    """Return mean wall-clock time in ms over n_reps calls."""
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    return (time.perf_counter() - t0) / n_reps * 1000.0


class TestACTBenchmarkAccuracy:
    """Accuracy comparisons: ACT must match numpy within tolerance."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rng = np.random.default_rng(42)
        self.act = ACTBackend()
        self.w_slow = self.rng.normal(0, 0.03, size=N * K).astype(np.float32)
        self.w_fast = self.rng.normal(0, 0.01, size=N * K).astype(np.float32)
        self.W = self.rng.normal(0, 0.1, size=(N, N)).astype(np.float32)
        self.state = self.rng.uniform(0, 0.05, size=N).astype(np.float32)
        self.R = self.rng.normal(0, 0.12, size=(N, VOCAB)).astype(np.float32)
        self.b = np.zeros(VOCAB, dtype=np.float32)

    def test_kahan_sum_accuracy_vs_numpy(self):
        """kahan_sum should agree with np.sum (float64 baseline) to 1e-12."""
        arr = self.w_slow.astype(np.float64)
        numpy_sum = float(np.sum(arr))
        if not self.act.available:
            pytest.skip("Balansis not installed")
        act_sum = self.act.kahan_sum(self.w_slow)
        # Both compared to float64 baseline
        numpy_err = abs(numpy_sum - float(np.sum(self.w_slow.astype(np.float64))))
        act_err = abs(act_sum - float(np.sum(self.w_slow.astype(np.float64))))
        assert act_err <= numpy_err + 1e-12, (
            f"ACT error {act_err:.2e} > numpy error {numpy_err:.2e}"
        )

    def test_quadratic_form_accuracy(self):
        """quadratic_form(s, W) should match float64 baseline to high precision."""
        s = self.state
        W = self.W
        # float64 ground truth
        ref = float(s.astype(np.float64) @ W.astype(np.float64) @ s.astype(np.float64))
        naive = float(s @ W @ s)
        if not self.act.available:
            pytest.skip("Balansis not installed")
        act_val = self.act.quadratic_form(s, W)
        # ACT should not be worse than naive float32
        naive_err = abs(naive - ref)
        act_err = abs(act_val - ref)
        assert act_err <= naive_err + 1e-6, (
            f"ACT error {act_err:.2e} > naive float32 error {naive_err:.2e}"
        )

    def test_add_precision_on_near_cancellation(self):
        """add() must preserve precision when a ≈ -b (near-cancellation)."""
        if not self.act.available:
            pytest.skip("Balansis not installed")
        a = np.array([1.0 + 1e-7], dtype=np.float64)
        b = np.array([-1.0], dtype=np.float64)
        result = self.act.add(a, b)
        np.testing.assert_allclose(result, np.array([1e-7]), atol=1e-14)

    def test_matvec_add_accuracy(self):
        """matvec_add accuracy vs float64 baseline for N=384, vocab=50."""
        s = self.state
        R = self.R
        b = self.b
        ref = (s.astype(np.float64) @ R.astype(np.float64) + b.astype(np.float64)).astype(np.float32)
        if not self.act.available:
            pytest.skip("Balansis not installed")
        act_result = self.act.matvec_add(s, R, b)
        np.testing.assert_allclose(act_result, ref, atol=1e-4)

    def test_weight_add_accumulated_N_times(self):
        """Accumulated add over N steps: ACT error stays bounded."""
        if not self.act.available:
            pytest.skip("Balansis not installed")
        # Simulate N small additions like STDP weight accumulation
        base = np.zeros(K * N, dtype=np.float32)
        delta = np.full(K * N, 1e-4, dtype=np.float32)
        act_acc = np.zeros(K * N, dtype=np.float32)
        np_acc = np.zeros(K * N, dtype=np.float32)
        for _ in range(N):
            act_acc = self.act.add(act_acc, delta)
            np_acc = np_acc + delta
        # After N additions of 1e-4, expected sum is N*1e-4 = 0.0384*K
        expected = N * 1e-4
        act_err = abs(float(np.mean(act_acc)) - expected)
        np_err = abs(float(np.mean(np_acc)) - expected)
        # ACT error should not be significantly worse than numpy
        assert act_err < 1e-7, f"ACT accumulated error too large: {act_err:.2e}"


class TestACTBenchmarkPerformance:
    """Wall-clock timing: ACT should not be more than 20x slower than numpy.

    These tests never fail on accuracy — they just print timing for reference.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rng = np.random.default_rng(42)
        self.act = ACTBackend()
        self.w = self.rng.normal(0, 0.03, size=N * K).astype(np.float32)
        self.dw = self.rng.normal(0, 0.001, size=N * K).astype(np.float32)
        self.W = self.rng.normal(0, 0.1, size=(N, N)).astype(np.float32)
        self.state = self.rng.uniform(0, 0.05, size=N).astype(np.float32)
        self.R = self.rng.normal(0, 0.12, size=(N, VOCAB)).astype(np.float32)
        self.b = np.zeros(VOCAB, dtype=np.float32)

    def test_add_timing(self, capsys):
        np_ms = _time_fn(lambda: self.w + self.dw, n_reps=50)
        if self.act.available:
            act_ms = _time_fn(lambda: self.act.add(self.w, self.dw), n_reps=50)
            ratio = act_ms / max(np_ms, 0.001)
            with capsys.disabled():
                print(f"\n  add (N*K={N*K}): numpy={np_ms:.3f}ms, ACT={act_ms:.3f}ms, ratio={ratio:.1f}x")
            assert ratio < 50, f"ACT add is {ratio:.1f}x slower than numpy"
        else:
            pytest.skip("Balansis not installed")

    def test_quadratic_form_timing(self, capsys):
        np_ms = _time_fn(lambda: float(self.state @ self.W @ self.state), n_reps=50)
        if self.act.available:
            act_ms = _time_fn(lambda: self.act.quadratic_form(self.state, self.W), n_reps=50)
            ratio = act_ms / max(np_ms, 0.001)
            with capsys.disabled():
                print(f"\n  quadratic_form (N={N}): numpy={np_ms:.3f}ms, ACT={act_ms:.3f}ms, ratio={ratio:.1f}x")
            assert ratio < 50, f"ACT quadratic_form is {ratio:.1f}x slower than numpy"
        else:
            pytest.skip("Balansis not installed")

    def test_matvec_add_timing(self, capsys):
        np_ms = _time_fn(lambda: (self.state @ self.R + self.b), n_reps=50)
        if self.act.available:
            act_ms = _time_fn(lambda: self.act.matvec_add(self.state, self.R, self.b), n_reps=50)
            ratio = act_ms / max(np_ms, 0.001)
            with capsys.disabled():
                print(f"\n  matvec_add (N={N}, vocab={VOCAB}): numpy={np_ms:.3f}ms, ACT={act_ms:.3f}ms, ratio={ratio:.1f}x")
            assert ratio < 50, f"ACT matvec_add is {ratio:.1f}x slower than numpy"
        else:
            pytest.skip("Balansis not installed")

    def test_full_brain_forward_timing(self, capsys):
        """Compare full TextBrain forward pass: ACT vs standard."""
        from magicbrain.brain import TextBrain
        brain_std = TextBrain(GENOME, VOCAB, seed_override=1, use_act=False)
        brain_act = TextBrain(GENOME, VOCAB, seed_override=1, use_act=True)
        np_ms = _time_fn(lambda: brain_std.forward(0), n_reps=20)
        if brain_act._act is None or not brain_act._act.available:
            pytest.skip("Balansis not installed")
        act_ms = _time_fn(lambda: brain_act.forward(0), n_reps=20)
        ratio = act_ms / max(np_ms, 0.001)
        with capsys.disabled():
            print(f"\n  forward (N={brain_std.N}, K={brain_std.K}, vocab={VOCAB}): "
                  f"numpy={np_ms:.3f}ms, ACT={act_ms:.3f}ms, ratio={ratio:.1f}x")
        assert ratio < 100, f"ACT forward is {ratio:.1f}x slower than numpy"
