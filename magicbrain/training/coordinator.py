from __future__ import annotations
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np

from ..brain import TextBrain
from ..tasks.text_task import build_vocab, train_loop
from .data_partitioner import DataPartitioner
from .weight_delta import WeightDelta, aggregate
from .worker import TrainingWorker, _worker_process
from .checkpointing import CheckpointManager


@dataclass
class TrainingResult:
    final_loss: float
    total_steps: int
    wall_time: float
    per_worker_losses: List[float]
    speedup_vs_single: float


class TrainingCoordinator:
    """Coordinates distributed training across multiple worker processes."""

    def __init__(
        self,
        genome_str: str,
        data: str,
        n_workers: int = 2,
        sync_every: int = 100,
        partition_mode: str = "overlapping",
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 10,
    ):
        self.genome_str = genome_str
        self.data = data
        self.n_workers = max(1, n_workers)
        self.sync_every = sync_every
        self.partition_mode = partition_mode

        self.vocab, self.itos = build_vocab(data)

        self.checkpoint_mgr = None
        if checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(checkpoint_dir, checkpoint_every)

        # Reference brain for weight initialization and single-worker baseline
        self._ref_brain = TextBrain(genome_str, len(self.vocab))

    def _partition_data(self) -> List[str]:
        partitioner = DataPartitioner()
        if self.partition_mode == "round_robin":
            return partitioner.round_robin(self.data, self.n_workers)
        elif self.partition_mode == "shuffle":
            return partitioner.shuffle_partition(self.data, self.n_workers)
        else:
            return partitioner.overlapping(self.data, self.n_workers)

    def train(
        self,
        total_steps: int,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingResult:
        """Main distributed training loop.

        Args:
            total_steps: Total training steps per worker.
            callback: Optional callback(round_num, avg_loss) called after each sync.

        Returns:
            TrainingResult with final metrics.
        """
        t_start = time.time()

        partitions = self._partition_data()
        n_rounds = max(1, total_steps // self.sync_every)
        steps_per_round = self.sync_every

        # Use multiprocessing for actual parallelism when >1 worker
        if self.n_workers > 1:
            result = self._train_multiprocess(
                partitions, n_rounds, steps_per_round, callback
            )
        else:
            result = self._train_single(
                partitions[0], n_rounds, steps_per_round, callback
            )

        wall_time = time.time() - t_start

        # Estimate single-worker baseline time
        single_time = self._estimate_single_time(total_steps)
        speedup = single_time / wall_time if wall_time > 0 else 1.0

        result.wall_time = wall_time
        result.speedup_vs_single = speedup
        return result

    def _train_single(
        self,
        data_partition: str,
        n_rounds: int,
        steps_per_round: int,
        callback: Optional[Callable],
    ) -> TrainingResult:
        """Single-worker training path (no multiprocessing overhead)."""
        worker = TrainingWorker(
            worker_id=0,
            genome_str=self.genome_str,
            vocab=self.vocab,
            data_partition=data_partition,
            sync_every=self.sync_every,
        )

        total_steps = 0
        last_loss = 0.0

        for rnd in range(n_rounds):
            delta = worker.train_steps(steps_per_round)
            total_steps += delta.steps_completed
            last_loss = delta.avg_loss

            if self.checkpoint_mgr and self.checkpoint_mgr.should_save(rnd):
                self.checkpoint_mgr.save(rnd, worker.get_weights(), last_loss)

            if callback:
                callback(rnd, last_loss)

        return TrainingResult(
            final_loss=last_loss,
            total_steps=total_steps,
            wall_time=0.0,
            per_worker_losses=[last_loss],
            speedup_vs_single=1.0,
        )

    def _train_multiprocess(
        self,
        partitions: List[str],
        n_rounds: int,
        steps_per_round: int,
        callback: Optional[Callable],
    ) -> TrainingResult:
        """Multi-worker training with FedAvg synchronization."""
        ctx = mp.get_context("spawn")
        cmd_queues = []
        result_queue = ctx.Queue()
        processes = []

        # Spawn workers
        for i in range(self.n_workers):
            cmd_q = ctx.Queue()
            cmd_queues.append(cmd_q)
            p = ctx.Process(
                target=_worker_process,
                args=(
                    i,
                    self.genome_str,
                    self.vocab,
                    partitions[i],
                    self.sync_every,
                    42 + i,  # deterministic per-worker seed
                    cmd_q,
                    result_queue,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        total_steps = 0
        per_worker_losses = [0.0] * self.n_workers
        avg_loss = 0.0

        try:
            for rnd in range(n_rounds):
                # 1. Tell all workers to train
                for q in cmd_queues:
                    q.put({"action": "train", "n_steps": steps_per_round})

                # 2. Collect deltas from all workers
                deltas = []
                for _ in range(self.n_workers):
                    res = result_queue.get(timeout=300)
                    deltas.append(res["delta"])
                    wid = res["worker_id"]
                    per_worker_losses[wid] = res["loss"]

                # 3. Aggregate deltas (FedAvg)
                agg = aggregate(deltas)
                avg_loss = agg.avg_loss
                total_steps += agg.steps_completed

                # 4. Compute new global weights by applying aggregated delta
                # to the reference brain's initial weights snapshot
                new_weights = self._apply_aggregate_to_ref(agg, rnd)

                # 5. Broadcast updated weights to all workers
                for q in cmd_queues:
                    q.put({"action": "apply_weights", "weights": new_weights})
                for _ in range(self.n_workers):
                    result_queue.get(timeout=60)

                # 6. Checkpoint
                if self.checkpoint_mgr and self.checkpoint_mgr.should_save(rnd):
                    self.checkpoint_mgr.save(rnd, new_weights, avg_loss)

                if callback:
                    callback(rnd, avg_loss)

        finally:
            # Shutdown workers
            for q in cmd_queues:
                q.put(None)
            for p in processes:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()

        return TrainingResult(
            final_loss=avg_loss,
            total_steps=total_steps,
            wall_time=0.0,
            per_worker_losses=per_worker_losses,
            speedup_vs_single=1.0,
        )

    def _apply_aggregate_to_ref(self, agg: WeightDelta, rnd: int) -> dict:
        """Apply aggregated delta to reference brain, return new weights dict."""
        # On first round, use the reference brain's initial weights
        # On subsequent rounds, accumulate deltas
        if not hasattr(self, "_current_weights"):
            self._current_weights = {
                "w_slow": self._ref_brain.w_slow.copy(),
                "w_fast": self._ref_brain.w_fast.copy(),
                "R": self._ref_brain.R.copy(),
                "b": self._ref_brain.b.copy(),
                "theta": self._ref_brain.theta.copy(),
            }

        self._current_weights["w_slow"] = (
            self._current_weights["w_slow"] + agg.w_slow_delta
        ).astype(np.float32)
        self._current_weights["w_fast"] = (
            self._current_weights["w_fast"] + agg.w_fast_delta
        ).astype(np.float32)
        self._current_weights["R"] = (
            self._current_weights["R"] + agg.R_delta
        ).astype(np.float32)
        self._current_weights["b"] = (
            self._current_weights["b"] + agg.b_delta
        ).astype(np.float32)
        self._current_weights["theta"] = (
            self._current_weights["theta"] + agg.theta_delta
        ).astype(np.float32)

        return {k: v.copy() for k, v in self._current_weights.items()}

    def _estimate_single_time(self, total_steps: int) -> float:
        """Quick estimate of single-worker time by benchmarking a small run."""
        brain = TextBrain(self.genome_str, len(self.vocab))
        ids = np.array([self.vocab[c] for c in self.data if c in self.vocab], dtype=np.int32)
        n = len(ids) - 1
        if n < 1:
            return 1.0

        bench_steps = min(100, total_steps)
        t0 = time.time()
        for i in range(bench_steps):
            x = int(ids[i % n])
            y = int(ids[(i + 1) % n])
            probs = brain.forward(x)
            brain.learn(y, probs)
        t1 = time.time()

        per_step = (t1 - t0) / bench_steps if bench_steps > 0 else 0.001
        return per_step * total_steps
