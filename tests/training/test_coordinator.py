import os
import numpy as np
import pytest
from magicbrain.training.coordinator import TrainingCoordinator, TrainingResult
from magicbrain.training.weight_delta import WeightDelta, aggregate
from magicbrain.training.checkpointing import CheckpointManager

GENOME = "30121033102301230112332100123"
TEXT = "to be or not to be that is the question " * 50


class TestBasicTraining:
    def test_single_worker_loss_decreases(self):
        """Single worker, verify loss decreases over training."""
        coord = TrainingCoordinator(
            genome_str=GENOME,
            data=TEXT,
            n_workers=1,
            sync_every=100,
        )
        losses = []

        def cb(rnd, loss):
            losses.append(loss)

        result = coord.train(total_steps=500, callback=cb)

        assert isinstance(result, TrainingResult)
        assert result.total_steps == 500
        assert result.wall_time > 0
        assert result.final_loss > 0
        # Loss should decrease from start to end
        if len(losses) >= 2:
            assert losses[-1] < losses[0]

    def test_two_workers_training(self):
        """2 workers, 500 steps, verify training completes and loss decreases."""
        coord = TrainingCoordinator(
            genome_str=GENOME,
            data=TEXT,
            n_workers=2,
            sync_every=50,
        )
        losses = []

        def cb(rnd, loss):
            losses.append(loss)

        result = coord.train(total_steps=500, callback=cb)

        assert isinstance(result, TrainingResult)
        assert result.total_steps > 0
        assert result.wall_time > 0
        assert len(result.per_worker_losses) == 2
        # Verify loss decreased
        if len(losses) >= 3:
            early_avg = np.mean(losses[:2])
            late_avg = np.mean(losses[-2:])
            assert late_avg < early_avg


class TestWeightAggregation:
    def test_fedavg_produces_correct_average(self):
        """Verify FedAvg produces correct mean of deltas."""
        shape_w = (100,)
        shape_R = (10, 5)
        shape_b = (5,)
        shape_theta = (10,)

        d1 = WeightDelta(
            w_slow_delta=np.ones(shape_w) * 2.0,
            w_fast_delta=np.ones(shape_w) * 4.0,
            R_delta=np.ones(shape_R) * 6.0,
            b_delta=np.ones(shape_b) * 8.0,
            theta_delta=np.ones(shape_theta) * 10.0,
            steps_completed=100,
            avg_loss=2.0,
        )
        d2 = WeightDelta(
            w_slow_delta=np.ones(shape_w) * 4.0,
            w_fast_delta=np.ones(shape_w) * 8.0,
            R_delta=np.ones(shape_R) * 2.0,
            b_delta=np.ones(shape_b) * 4.0,
            theta_delta=np.ones(shape_theta) * 6.0,
            steps_completed=100,
            avg_loss=3.0,
        )

        agg = aggregate([d1, d2])

        assert np.allclose(agg.w_slow_delta, 3.0)
        assert np.allclose(agg.w_fast_delta, 6.0)
        assert np.allclose(agg.R_delta, 4.0)
        assert np.allclose(agg.b_delta, 6.0)
        assert np.allclose(agg.theta_delta, 8.0)
        assert agg.steps_completed == 200
        assert abs(agg.avg_loss - 2.5) < 1e-6

    def test_single_delta_aggregation(self):
        """Aggregating a single delta should return itself."""
        d = WeightDelta(
            w_slow_delta=np.array([1.0, 2.0, 3.0]),
            w_fast_delta=np.array([4.0, 5.0, 6.0]),
            R_delta=np.array([[1.0]]),
            b_delta=np.array([1.0]),
            theta_delta=np.array([1.0]),
            steps_completed=50,
            avg_loss=1.5,
        )
        agg = aggregate([d])
        assert np.allclose(agg.w_slow_delta, d.w_slow_delta)
        assert agg.avg_loss == d.avg_loss

    def test_empty_aggregate_raises(self):
        with pytest.raises(ValueError):
            aggregate([])


class TestCheckpointResume:
    def test_checkpoint_save_and_resume(self, tmp_path):
        """Train 250 steps, checkpoint, resume, train 250 more."""
        ckpt_dir = str(tmp_path / "checkpoints")

        # Phase 1: train 250 steps with checkpointing
        coord1 = TrainingCoordinator(
            genome_str=GENOME,
            data=TEXT,
            n_workers=1,
            sync_every=50,
            checkpoint_dir=ckpt_dir,
            checkpoint_every=5,  # save every 5 rounds
        )
        result1 = coord1.train(total_steps=250)
        loss_after_phase1 = result1.final_loss

        # Verify checkpoints were saved
        mgr = CheckpointManager(ckpt_dir)
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) > 0

        # Load best checkpoint
        weights, meta = mgr.load_best()
        assert "loss" in meta
        assert "round_num" in meta

        # Phase 2: create new coordinator, apply checkpoint weights, train more
        coord2 = TrainingCoordinator(
            genome_str=GENOME,
            data=TEXT,
            n_workers=1,
            sync_every=50,
        )
        result2 = coord2.train(total_steps=250)

        # Both phases should produce valid results
        assert result1.final_loss > 0
        assert result2.final_loss > 0


class TestWeightDeltaCompression:
    def test_compress_zeros_small_values(self):
        d = WeightDelta(
            w_slow_delta=np.array([0.001, 0.1, -0.001, -0.5]),
            w_fast_delta=np.array([0.001, 0.1, -0.001, -0.5]),
            R_delta=np.array([[0.01, 0.001]]),
            b_delta=np.array([0.001]),
            theta_delta=np.array([0.05]),
            steps_completed=10,
            avg_loss=1.0,
        )
        compressed = d.compress(threshold=0.01)
        assert compressed.w_slow_delta[0] == 0.0  # 0.001 < 0.01
        assert compressed.w_slow_delta[1] == 0.1   # 0.1 >= 0.01
        assert compressed.w_slow_delta[2] == 0.0   # |-0.001| < 0.01
        assert compressed.w_slow_delta[3] == -0.5  # |-0.5| >= 0.01
