from __future__ import annotations
import numpy as np
from ..brain import TextBrain
from ..tasks.text_task import build_vocab
from .weight_delta import WeightDelta


class TrainingWorker:
    """A single distributed training worker wrapping a TextBrain instance."""

    def __init__(
        self,
        worker_id: int,
        genome_str: str,
        vocab: dict,
        data_partition: str,
        sync_every: int = 100,
        rng_seed: int | None = None,
    ):
        self.worker_id = worker_id
        self.genome_str = genome_str
        self.vocab = vocab
        self.data_partition = data_partition
        self.sync_every = sync_every

        self.brain = TextBrain(genome_str, len(vocab), seed_override=rng_seed)

        self.ids = np.array([vocab[c] for c in data_partition if c in vocab], dtype=np.int32)
        if len(self.ids) < 2:
            raise ValueError(f"Worker {worker_id}: data partition too short (need >= 2 tokens)")

        self._snapshot_weights()

    def _snapshot_weights(self):
        """Save a snapshot of current weights for delta computation."""
        self._snap_w_slow = self.brain.w_slow.copy()
        self._snap_w_fast = self.brain.w_fast.copy()
        self._snap_R = self.brain.R.copy()
        self._snap_b = self.brain.b.copy()
        self._snap_theta = self.brain.theta.copy()

    def train_steps(self, n_steps: int) -> WeightDelta:
        """Train for n_steps and return the weight delta from initial snapshot."""
        self._snapshot_weights()

        n = len(self.ids) - 1
        losses = []

        for i in range(n_steps):
            idx = (self.brain.step) % n
            x = int(self.ids[idx])
            y = int(self.ids[idx + 1])

            probs = self.brain.forward(x)
            loss = self.brain.learn(y, probs)
            losses.append(loss)

        avg_loss = float(np.mean(losses)) if losses else 0.0

        return WeightDelta(
            w_slow_delta=self.brain.w_slow - self._snap_w_slow,
            w_fast_delta=self.brain.w_fast - self._snap_w_fast,
            R_delta=self.brain.R - self._snap_R,
            b_delta=self.brain.b - self._snap_b,
            theta_delta=self.brain.theta - self._snap_theta,
            steps_completed=n_steps,
            avg_loss=avg_loss,
        )

    def apply_weights(self, weights_dict: dict):
        """Replace brain weights with new values."""
        self.brain.w_slow = weights_dict["w_slow"].copy()
        self.brain.w_fast = weights_dict["w_fast"].copy()
        self.brain.R = weights_dict["R"].copy()
        self.brain.b = weights_dict["b"].copy()
        self.brain.theta = weights_dict["theta"].copy()

    def get_weights(self) -> dict:
        """Return a copy of current weights."""
        return {
            "w_slow": self.brain.w_slow.copy(),
            "w_fast": self.brain.w_fast.copy(),
            "R": self.brain.R.copy(),
            "b": self.brain.b.copy(),
            "theta": self.brain.theta.copy(),
        }

    def get_loss(self) -> float:
        """Return the current EMA loss."""
        return float(self.brain.loss_ema)


def _worker_process(
    worker_id: int,
    genome_str: str,
    vocab: dict,
    data_partition: str,
    sync_every: int,
    rng_seed: int | None,
    cmd_queue,
    result_queue,
):
    """Target function for multiprocessing.Process workers."""
    worker = TrainingWorker(
        worker_id=worker_id,
        genome_str=genome_str,
        vocab=vocab,
        data_partition=data_partition,
        sync_every=sync_every,
        rng_seed=rng_seed,
    )

    while True:
        cmd = cmd_queue.get()
        if cmd is None:
            break

        action = cmd["action"]

        if action == "train":
            n_steps = cmd["n_steps"]
            delta = worker.train_steps(n_steps)
            result_queue.put({
                "worker_id": worker_id,
                "delta": delta,
                "loss": worker.get_loss(),
            })

        elif action == "apply_weights":
            worker.apply_weights(cmd["weights"])
            result_queue.put({"worker_id": worker_id, "status": "applied"})

        elif action == "get_weights":
            result_queue.put({
                "worker_id": worker_id,
                "weights": worker.get_weights(),
            })

        elif action == "get_loss":
            result_queue.put({
                "worker_id": worker_id,
                "loss": worker.get_loss(),
            })
