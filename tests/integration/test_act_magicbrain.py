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

    def test_act_backend_subtract(self):
        a = np.array([3.0, 1.0, -1.0], dtype=np.float32)
        b = np.array([1.0, 1.0,  1.0], dtype=np.float32)
        result = self.backend.subtract(a, b)
        np.testing.assert_allclose(result, np.array([2.0, 0.0, -2.0]), atol=1e-6)
        assert result.dtype == np.float32

    def test_act_backend_subtract_near_cancellation(self):
        """ACT subtract preserves precision when operands nearly cancel."""
        a = np.array([1.0 + 1e-7, 0.5], dtype=np.float64)
        b = np.array([1.0,        0.5], dtype=np.float64)
        result = self.backend.subtract(a, b)
        np.testing.assert_allclose(result, np.array([1e-7, 0.0]), atol=1e-14)

    def test_act_backend_add(self):
        a = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        b = np.array([3.0, -1.0, 1.0], dtype=np.float32)
        result = self.backend.add(a, b)
        np.testing.assert_allclose(result, np.array([4.0, 1.0, 0.0]), atol=1e-6)
        assert result.dtype == np.float32

    def test_act_backend_mix_identity_at_eps_zero(self):
        """mix with eps=0 should return slow unchanged."""
        slow = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fast = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        result = self.backend.mix(slow, fast, 0.0)
        np.testing.assert_allclose(result, slow, atol=1e-6)

    def test_act_backend_mix_typical(self):
        slow = np.array([1.0, 0.0], dtype=np.float32)
        fast = np.array([0.0, 1.0], dtype=np.float32)
        result = self.backend.mix(slow, fast, 0.5)
        np.testing.assert_allclose(result, np.array([0.5, 0.5]), atol=1e-6)

    def test_act_backend_matvec_add(self):
        state = np.array([1.0, 0.0], dtype=np.float32)
        W = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        b = np.array([0.1, 0.2], dtype=np.float32)
        result = self.backend.matvec_add(state, W, b)
        expected = state @ W + b
        np.testing.assert_allclose(result, expected, atol=1e-5)
        assert result.dtype == np.float32


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

    def test_effective_w_act_uses_compensated_add(self):
        """_effective_w() with ACT must produce finite results equal to w_slow + w_fast."""
        brain = self._make_brain(use_act=True)
        # Verify that ACT path is actually taken when Balansis available
        if brain._act is None or not brain._act.available:
            pytest.skip("Balansis not installed")
        w_eff = brain._effective_w()
        expected = brain.w_slow + brain.w_fast
        assert w_eff.shape == expected.shape
        assert np.all(np.isfinite(w_eff))
        np.testing.assert_allclose(w_eff, expected, atol=1e-6)

    def test_consolidate_act_preserves_shape_and_dtype(self):
        """_consolidate() with ACT must not change shape or dtype of w_slow."""
        GENOME = "30121033102301230112332100123"
        from magicbrain.brain import TextBrain
        brain = TextBrain(GENOME, 8, seed_override=42, use_act=True)
        if brain._act is None or not brain._act.available:
            pytest.skip("Balansis not installed")
        shape_before = brain.w_slow.shape
        dtype_before = brain.w_slow.dtype
        brain._consolidate()
        assert brain.w_slow.shape == shape_before
        assert brain.w_slow.dtype == dtype_before
        assert np.all(np.isfinite(brain.w_slow))

    def test_forward_near_cancellation_act(self):
        """forward() with ACT must be stable when delayed signal ≈ threshold."""
        brain = self._make_brain(use_act=True)
        if brain._act is None or not brain._act.available:
            pytest.skip("Balansis not installed")
        # Force buffers and theta to be nearly equal (near-cancellation scenario)
        brain.buffers[1][:] = brain.theta
        probs = brain.forward(0)
        assert not np.any(np.isnan(probs))
        assert abs(float(np.sum(probs)) - 1.0) < 1e-4

    def test_matvec_add_in_forward(self):
        """state @ R + b path with ACT must produce valid logit distribution."""
        brain = self._make_brain(use_act=True)
        if brain._act is None or not brain._act.available:
            pytest.skip("Balansis not installed")
        probs = brain.forward(0)
        assert probs.shape == (VOCAB,)
        assert np.all(probs >= 0)
        assert abs(float(np.sum(probs)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Neural Digital Twin with ACT
# ---------------------------------------------------------------------------

class TestDigitalTwinWithACT:
    """Tests for NeuralDigitalTwin ACT integration."""

    GENOME = "30121033102301230112332100123"

    def _make_twin(self, use_act: bool = False):
        from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin
        return NeuralDigitalTwin("student_test_42", use_act=use_act)

    def test_twin_creates_with_use_act(self):
        """NeuralDigitalTwin must accept use_act=True without error."""
        twin = self._make_twin(use_act=True)
        assert twin._act is not None

    def test_twin_fallback_without_act(self):
        """NeuralDigitalTwin without use_act must work normally."""
        twin = self._make_twin(use_act=False)
        assert twin._act is None

    def test_apply_forgetting_act_no_nan(self):
        """_apply_forgetting() with ACT must not produce NaN for small mastery."""
        twin = self._make_twin(use_act=True)
        if twin._act is None or not twin._act.available:
            pytest.skip("Balansis not installed")
        twin.register_topic("math", "Math")
        twin.mastery_scores["math"] = 0.001  # near-zero mastery (near-cancellation scenario)
        result = twin._apply_forgetting("math", 0.001)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_apply_forgetting_matches_numpy(self):
        """ACT and numpy paths must give consistent results for _apply_forgetting."""
        from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin
        twin_act = NeuralDigitalTwin("student_42a", use_act=True)
        twin_std = NeuralDigitalTwin("student_42b", use_act=False)
        for twin in (twin_act, twin_std):
            twin.register_topic("topic1", "Topic 1")
            twin.mastery_scores["topic1"] = 0.5
        result_act = twin_act._apply_forgetting("topic1", 0.5)
        result_std = twin_std._apply_forgetting("topic1", 0.5)
        assert abs(result_act - result_std) < 1e-5

    def test_process_interaction_correct_answer_act(self):
        """process_interaction_event correct answer with ACT must increase mastery."""
        twin = self._make_twin(use_act=True)
        if twin._act is None or not twin._act.available:
            pytest.skip("Balansis not installed")
        twin.register_topic("physics", "Physics")
        old_mastery = twin.mastery_scores["physics"]
        twin.process_interaction_event({
            "type": "answer_submitted",
            "topic_id": "physics",
            "is_correct": True,
            "difficulty": 0.5,
        })
        assert twin.mastery_scores["physics"] > old_mastery

    def test_process_interaction_wrong_answer_act(self):
        """process_interaction_event wrong answer with ACT must not crash."""
        twin = self._make_twin(use_act=True)
        if twin._act is None or not twin._act.available:
            pytest.skip("Balansis not installed")
        twin.register_topic("chem", "Chemistry")
        twin.mastery_scores["chem"] = 0.5
        result = twin.process_interaction_event({
            "type": "answer_submitted",
            "topic_id": "chem",
            "is_correct": False,
            "difficulty": 0.3,
        })
        assert 0.0 <= twin.mastery_scores["chem"] <= 1.0
        assert "mastery_scores" in result

    def test_get_cognitive_state_act(self):
        """get_cognitive_state() with ACT must return finite mean_weight."""
        twin = self._make_twin(use_act=True)
        if twin._act is None or not twin._act.available:
            pytest.skip("Balansis not installed")
        state = twin.get_cognitive_state()
        assert np.isfinite(state["neural_metrics"]["mean_weight"])


# ---------------------------------------------------------------------------
# ACTMetricsTracker tests
# ---------------------------------------------------------------------------

class TestACTMetricsTracker:
    """Tests for ACTMetricsTracker and LiveMonitor integration."""

    GENOME = "30121033102301230112332100123"

    def _make_brain(self, use_act: bool = False):
        from magicbrain.brain import TextBrain
        return TextBrain(self.GENOME, VOCAB, seed_override=42, use_act=use_act)

    def test_tracker_creates_without_act(self):
        """ACTMetricsTracker must work without Balansis (numpy fallback)."""
        from magicbrain.diagnostics import ACTMetricsTracker
        tracker = ACTMetricsTracker(act_backend=None)
        assert not tracker.act_available

    def test_tracker_records_snapshot(self):
        """record() must produce a valid ACTSnapshot."""
        from magicbrain.diagnostics import ACTMetricsTracker
        brain = self._make_brain(use_act=False)
        tracker = ACTMetricsTracker(act_backend=None)
        snap = tracker.record(brain, step=0)
        assert snap.step == 0
        assert np.isfinite(snap.w_slow_l2)
        assert np.isfinite(snap.combined_weight_std)

    def test_tracker_with_act_backend(self):
        """ACTMetricsTracker with live ACT backend must produce finite metrics."""
        from magicbrain.diagnostics import ACTMetricsTracker
        from magicbrain.integration.act_backend import ACTBackend
        brain = self._make_brain(use_act=True)
        act = ACTBackend()
        if not act.available:
            pytest.skip("Balansis not installed")
        tracker = ACTMetricsTracker(act_backend=act)
        snap = tracker.record(brain, step=0)
        assert snap.act_available
        assert np.isfinite(snap.w_slow_l2)
        assert np.isfinite(snap.w_fast_l2)

    def test_tracker_summary_empty(self):
        """summary() on empty tracker must return n_records=0."""
        from magicbrain.diagnostics import ACTMetricsTracker
        tracker = ACTMetricsTracker()
        s = tracker.summary()
        assert s["n_records"] == 0

    def test_tracker_summary_after_records(self):
        """summary() after multiple records must contain all expected keys."""
        from magicbrain.diagnostics import ACTMetricsTracker
        brain = self._make_brain(use_act=False)
        tracker = ACTMetricsTracker()
        for i in range(5):
            tracker.record(brain, step=i)
        s = tracker.summary()
        assert s["n_records"] == 5
        assert "final_w_slow_l2" in s
        assert "final_combined_std" in s

    def test_live_monitor_track_act_flag(self):
        """LiveMonitor with track_act=True must attach act_tracker after first record."""
        from magicbrain.diagnostics import LiveMonitor
        brain = self._make_brain(use_act=True)
        monitor = LiveMonitor(log_every=1, track_act=True)
        assert monitor.act_tracker is None
        monitor.record(brain, loss=1.0, step=0)
        assert monitor.act_tracker is not None

    def test_live_monitor_get_summary_includes_act(self):
        """get_summary() must include 'act_metrics' key when track_act=True."""
        from magicbrain.diagnostics import LiveMonitor
        brain = self._make_brain(use_act=True)
        monitor = LiveMonitor(log_every=1, track_act=True)
        monitor.record(brain, loss=1.0, step=0)
        summary = monitor.get_summary()
        assert "act_metrics" in summary
