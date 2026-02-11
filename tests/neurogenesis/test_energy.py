"""Tests for energy function and attractor dynamics."""

import pytest
import numpy as np
from magicbrain.neurogenesis.energy import EnergyFunction, EnergyState
from magicbrain.neurogenesis.attractor_dynamics import (
    AttractorDynamics,
    ConvergenceResult,
    _sigmoid_vec,
    _random_sparse_state,
)


class TestEnergyFunction:
    def setup_method(self):
        self.ef = EnergyFunction(lambda_sparse=0.01)
        self.N = 32
        self.rng = np.random.default_rng(42)
        # Symmetric weight matrix
        W = self.rng.normal(0, 0.1, size=(self.N, self.N)).astype(np.float32)
        self.W = 0.5 * (W + W.T)
        np.fill_diagonal(self.W, 0.0)
        self.theta = np.zeros(self.N, dtype=np.float32)

    def test_energy_is_scalar(self):
        state = self.rng.random(self.N).astype(np.float32)
        e = self.ef.energy(state, self.W, self.theta)
        assert isinstance(e, float)

    def test_zero_state_zero_energy(self):
        state = np.zeros(self.N, dtype=np.float32)
        e = self.ef.energy(state, self.W, self.theta)
        assert abs(e) < 1e-6

    def test_energy_decomposed(self):
        state = self.rng.random(self.N).astype(np.float32)
        decomp = self.ef.energy_decomposed(state, self.W, self.theta)
        assert isinstance(decomp, EnergyState)
        # Total should equal sum of components
        expected = decomp.interaction + decomp.bias + decomp.sparsity
        assert abs(decomp.total - expected) < 1e-5

    def test_gradient_shape(self):
        state = self.rng.random(self.N).astype(np.float32)
        grad = self.ef.gradient(state, self.W, self.theta)
        assert grad.shape == (self.N,)

    def test_gradient_numerical(self):
        """Numerical gradient should approximate analytical gradient."""
        state = self.rng.random(self.N).astype(np.float32) * 0.5
        grad = self.ef.gradient(state, self.W, self.theta)

        eps = 1e-4
        num_grad = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            sp = state.copy()
            sm = state.copy()
            sp[i] += eps
            sm[i] -= eps
            num_grad[i] = (
                self.ef.energy(sp, self.W, self.theta)
                - self.ef.energy(sm, self.W, self.theta)
            ) / (2 * eps)

        # Should be close (not exact due to sign() in sparsity term)
        np.testing.assert_allclose(grad, num_grad, atol=0.02)

    def test_local_field_shape(self):
        state = self.rng.random(self.N).astype(np.float32)
        h = self.ef.local_field(state, self.W, self.theta)
        assert h.shape == (self.N,)

    def test_sparse_format(self):
        """Test energy with sparse edge-list format."""
        # Create sparse edges
        src = np.array([0, 0, 1, 2], dtype=np.int32)
        dst = np.array([1, 2, 2, 0], dtype=np.int32)
        w = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)
        state = np.array([1.0, 0.5, 0.3], dtype=np.float32)
        theta = np.zeros(3, dtype=np.float32)

        e = self.ef.energy(state, w, theta, src=src, dst=dst)
        assert isinstance(e, float)

    def test_basin_energy_profile(self):
        state = self.rng.random(self.N).astype(np.float32) * 0.5
        profile = self.ef.basin_energy_profile(state, self.W, self.theta)
        assert "attractor_energy" in profile
        assert "basin_depth" in profile
        assert "energy_std" in profile


class TestAttractorDynamics:
    def setup_method(self):
        self.N = 32
        self.rng = np.random.default_rng(42)
        W = self.rng.normal(0, 0.3, size=(self.N, self.N)).astype(np.float32)
        self.W = 0.5 * (W + W.T)
        np.fill_diagonal(self.W, 0.0)
        self.theta = np.zeros(self.N, dtype=np.float32)
        self.dynamics = AttractorDynamics(
            tau=0.3, momentum=0.7, max_iterations=100, tolerance=1e-4
        )

    def test_step_produces_valid_state(self):
        state = self.rng.random(self.N).astype(np.float32)
        new_state = self.dynamics.step(state, self.W, self.theta)
        assert new_state.shape == (self.N,)
        assert np.all(np.isfinite(new_state))
        # Sigmoid output is in [0, 1]
        assert np.all(new_state >= 0)
        assert np.all(new_state <= 1)

    def test_converge_returns_result(self):
        cue = self.rng.random(self.N).astype(np.float32)
        result = self.dynamics.converge(cue, self.W, self.theta)
        assert isinstance(result, ConvergenceResult)
        assert result.state.shape == (self.N,)
        assert result.iterations > 0

    def test_converge_energy_decreases(self):
        """Energy should generally decrease during convergence."""
        cue = self.rng.random(self.N).astype(np.float32) * 0.5
        result = self.dynamics.converge(
            cue, self.W, self.theta, track_energy=True
        )
        if len(result.energy_trajectory) > 2:
            # Energy at end should be <= energy at start (approximately)
            assert result.energy_trajectory[-1] <= result.energy_trajectory[0] + 0.1

    def test_find_attractors(self):
        attractors = self.dynamics.find_attractors(
            self.N, self.W, self.theta, n_probes=50
        )
        assert len(attractors) > 0
        # Each attractor has required fields
        for att in attractors:
            assert att.state.shape == (self.N,)
            assert att.basin_size >= 1
            assert 0 <= att.stability <= 1

    def test_attractor_stability(self):
        """Found attractors should be valid states in [0,1] range."""
        attractors = self.dynamics.find_attractors(
            self.N, self.W, self.theta, n_probes=50
        )
        # All attractor states should be in valid range
        if attractors:
            for att in attractors:
                assert np.all(att.state >= 0)
                assert np.all(att.state <= 1)
                assert 0 <= att.stability <= 1


class TestHelpers:
    def test_sigmoid_vec(self):
        x = np.array([-100, -1, 0, 1, 100], dtype=np.float32)
        y = _sigmoid_vec(x)
        assert y.shape == x.shape
        assert np.all(y >= 0)
        assert np.all(y <= 1)
        assert abs(y[2] - 0.5) < 1e-5  # sigmoid(0) = 0.5

    def test_random_sparse_state(self):
        rng = np.random.default_rng(42)
        state = _random_sparse_state(100, 0.1, rng)
        assert state.shape == (100,)
        active = np.sum(state > 0)
        assert active == 10  # 10% of 100
