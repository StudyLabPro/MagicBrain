import numpy as np
import pytest
from magicbrain.brain import TextBrain
from magicbrain.tasks.text_task import build_vocab
from magicbrain.training.worker import TrainingWorker
from magicbrain.training.weight_delta import WeightDelta

GENOME = "30121033102301230112332100123"
TEXT = "to be or not to be that is the question " * 20


@pytest.fixture
def vocab():
    stoi, _ = build_vocab(TEXT)
    return stoi


class TestWorkerTrainSteps:
    def test_produces_valid_delta(self, vocab):
        worker = TrainingWorker(
            worker_id=0,
            genome_str=GENOME,
            vocab=vocab,
            data_partition=TEXT,
            sync_every=50,
            rng_seed=42,
        )
        delta = worker.train_steps(50)

        assert isinstance(delta, WeightDelta)
        assert delta.steps_completed == 50
        assert delta.avg_loss > 0
        assert delta.w_slow_delta.shape == worker.brain.w_slow.shape
        assert delta.R_delta.shape == worker.brain.R.shape
        assert delta.b_delta.shape == worker.brain.b.shape
        assert delta.theta_delta.shape == worker.brain.theta.shape

    def test_delta_is_nonzero(self, vocab):
        worker = TrainingWorker(
            worker_id=0,
            genome_str=GENOME,
            vocab=vocab,
            data_partition=TEXT,
            rng_seed=42,
        )
        delta = worker.train_steps(100)
        # After training, at least some weights should have changed
        assert np.any(delta.w_slow_delta != 0)
        assert np.any(delta.R_delta != 0)


class TestWorkerApplyWeights:
    def test_weights_correctly_updated(self, vocab):
        worker = TrainingWorker(
            worker_id=0,
            genome_str=GENOME,
            vocab=vocab,
            data_partition=TEXT,
            rng_seed=42,
        )
        # Get initial weights
        initial = worker.get_weights()

        # Create new weights (zeros)
        new_weights = {
            "w_slow": np.zeros_like(initial["w_slow"]),
            "w_fast": np.zeros_like(initial["w_fast"]),
            "R": np.zeros_like(initial["R"]),
            "b": np.zeros_like(initial["b"]),
            "theta": np.ones_like(initial["theta"]) * 0.5,
        }
        worker.apply_weights(new_weights)

        current = worker.get_weights()
        assert np.allclose(current["w_slow"], 0.0)
        assert np.allclose(current["theta"], 0.5)

    def test_apply_does_not_share_memory(self, vocab):
        worker = TrainingWorker(
            worker_id=0,
            genome_str=GENOME,
            vocab=vocab,
            data_partition=TEXT,
            rng_seed=42,
        )
        weights = worker.get_weights()
        worker.apply_weights(weights)
        # Mutating the original dict should not affect the worker
        weights["w_slow"][0] = 999.0
        assert worker.brain.w_slow[0] != 999.0


class TestWorkerReproducibility:
    def test_same_seed_same_results(self, vocab):
        w1 = TrainingWorker(0, GENOME, vocab, TEXT, rng_seed=42)
        w2 = TrainingWorker(1, GENOME, vocab, TEXT, rng_seed=42)

        d1 = w1.train_steps(50)
        d2 = w2.train_steps(50)

        assert np.allclose(d1.w_slow_delta, d2.w_slow_delta)
        assert np.allclose(d1.R_delta, d2.R_delta)
        assert abs(d1.avg_loss - d2.avg_loss) < 1e-6

    def test_different_seed_different_results(self, vocab):
        w1 = TrainingWorker(0, GENOME, vocab, TEXT, rng_seed=42)
        w2 = TrainingWorker(1, GENOME, vocab, TEXT, rng_seed=99)

        d1 = w1.train_steps(50)
        d2 = w2.train_steps(50)

        # Different seeds should lead to different initializations and thus different deltas
        assert not np.allclose(d1.w_slow_delta, d2.w_slow_delta)
