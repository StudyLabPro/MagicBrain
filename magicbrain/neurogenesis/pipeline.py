"""NeurogenesisPipeline: Full orchestrator for compile -> develop -> train -> find attractors.

Provides checkpointing, metrics collection, error handling, and resume support.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .compiler import GenomeCompiler
from .development import DevelopmentOperator
from .attractor_dynamics import AttractorDynamics
from .pattern_memory import PatternMemory

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the neurogenesis pipeline."""
    strategy: str = "auto"
    use_cppn: bool = True
    training_steps: int = 1000
    metrics_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    timeout_seconds: float = 600
    seed: Optional[int] = None


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    genome: str
    tissue: Any  # NeuralTissue
    brain: Any  # TextBrain
    attractors: List
    metrics: Dict[str, Any]
    checkpoints: List[str]


class NeurogenesisPipeline:
    """Full pipeline orchestrator: compile -> develop -> train -> find attractors."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._metrics: Dict[str, Any] = {}
        self._checkpoints: List[str] = []

    def run(self, data: str, vocab_size: int = 50) -> PipelineResult:
        """Execute full pipeline: compile -> develop -> train -> find attractors.

        Args:
            data: Input text data.
            vocab_size: Vocabulary size for the brain.

        Returns:
            PipelineResult with all outputs and metrics.
        """
        t0 = time.time()
        genome = None
        tissue = None
        brain = None
        attractors = []

        try:
            # Step 1: Compile genome
            self._check_timeout(t0, "compile")
            genome = self._step_compile(data)
            self._save_checkpoint("compile", {"genome": genome})

            # Step 2: Develop neural tissue
            self._check_timeout(t0, "develop")
            brain, tissue = self._step_develop(genome, vocab_size)
            self._save_checkpoint("develop", {
                "genome": genome,
                "N": tissue.N,
                "K": tissue.K,
            })

            # Step 3: Train brain
            self._check_timeout(t0, "train")
            self._step_train(brain, data, vocab_size)
            self._save_checkpoint("train", {
                "genome": genome,
                "training_steps": self.config.training_steps,
            })

            # Step 4: Find attractors
            self._check_timeout(t0, "attractors")
            attractors = self._step_find_attractors(brain)
            self._save_checkpoint("attractors", {
                "n_attractors": len(attractors),
            })

            # Step 5: Pattern memory imprinting
            self._check_timeout(t0, "patterns")
            self._step_pattern_memory(brain, data, vocab_size)
            self._save_checkpoint("patterns", {
                "complete": True,
            })

        except TimeoutError as e:
            logger.warning("Pipeline timed out: %s", e)
            self._metrics["timeout"] = True
            self._metrics["timeout_message"] = str(e)
        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            self._metrics["error"] = str(e)
            raise

        total_time = time.time() - t0
        self._metrics["total_time_seconds"] = total_time

        self._save_metrics_to_dir()

        return PipelineResult(
            genome=genome or "",
            tissue=tissue,
            brain=brain,
            attractors=attractors,
            metrics=dict(self._metrics),
            checkpoints=list(self._checkpoints),
        )

    def resume(self, checkpoint_path: str, data: str, vocab_size: int = 50) -> PipelineResult:
        """Resume pipeline from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.
            data: Original input text data.
            vocab_size: Vocabulary size.

        Returns:
            PipelineResult with all outputs.
        """
        state = self._load_checkpoint(checkpoint_path)
        step_name = state.get("step_name", "")
        step_data = state.get("step_data", {})

        t0 = time.time()
        genome = step_data.get("genome")
        tissue = None
        brain = None
        attractors = []

        # Determine which steps to skip
        completed_steps = {"compile", "develop", "train", "attractors", "patterns"}
        steps_order = ["compile", "develop", "train", "attractors", "patterns"]

        # Find the index of the completed step
        if step_name in steps_order:
            resume_from_idx = steps_order.index(step_name) + 1
        else:
            resume_from_idx = 0

        try:
            # Re-run earlier steps if needed for state
            if resume_from_idx <= 0:
                genome = self._step_compile(data)
                self._save_checkpoint("compile", {"genome": genome})
                resume_from_idx = 1

            if genome is None:
                genome = self._step_compile(data)

            if resume_from_idx <= 1:
                self._check_timeout(t0, "develop")
                brain, tissue = self._step_develop(genome, vocab_size)
                self._save_checkpoint("develop", {
                    "genome": genome, "N": tissue.N, "K": tissue.K,
                })
                resume_from_idx = 2

            if brain is None:
                brain, tissue = self._step_develop(genome, vocab_size)

            if resume_from_idx <= 2:
                self._check_timeout(t0, "train")
                self._step_train(brain, data, vocab_size)
                self._save_checkpoint("train", {
                    "genome": genome,
                    "training_steps": self.config.training_steps,
                })
                resume_from_idx = 3

            if resume_from_idx <= 3:
                self._check_timeout(t0, "attractors")
                attractors = self._step_find_attractors(brain)
                self._save_checkpoint("attractors", {
                    "n_attractors": len(attractors),
                })
                resume_from_idx = 4

            if resume_from_idx <= 4:
                self._check_timeout(t0, "patterns")
                self._step_pattern_memory(brain, data, vocab_size)
                self._save_checkpoint("patterns", {"complete": True})

        except TimeoutError as e:
            logger.warning("Pipeline timed out during resume: %s", e)
            self._metrics["timeout"] = True
        except Exception as e:
            logger.error("Pipeline failed during resume: %s", e)
            self._metrics["error"] = str(e)
            raise

        total_time = time.time() - t0
        self._metrics["total_time_seconds"] = total_time
        self._metrics["resumed_from"] = step_name
        self._save_metrics_to_dir()

        return PipelineResult(
            genome=genome or "",
            tissue=tissue,
            brain=brain,
            attractors=attractors,
            metrics=dict(self._metrics),
            checkpoints=list(self._checkpoints),
        )

    def _check_timeout(self, t0: float, step_name: str) -> None:
        """Check if pipeline has exceeded timeout."""
        elapsed = time.time() - t0
        if elapsed > self.config.timeout_seconds:
            raise TimeoutError(
                f"Pipeline timeout ({self.config.timeout_seconds}s) "
                f"exceeded before step '{step_name}' (elapsed: {elapsed:.1f}s)"
            )

    def _step_compile(self, data: str) -> str:
        """Step 1: Compile genome from data."""
        logger.info("Pipeline step 1: Compiling genome (strategy=%s)", self.config.strategy)
        t0 = time.time()

        compiler = GenomeCompiler()
        genome, metrics = compiler.compile_with_metrics(
            data,
            strategy=self.config.strategy,
            seed=self.config.seed,
        )

        elapsed = time.time() - t0
        self._collect_metrics("compile", {
            "time_seconds": elapsed,
            "genome_length": len(genome),
            "strategy_used": metrics.strategy_used,
            "quality_score": metrics.genome_quality_score,
        })

        logger.info("Compiled genome: length=%d, strategy=%s", len(genome), metrics.strategy_used)
        return genome

    def _step_develop(self, genome: str, vocab_size: int):
        """Step 2: Develop neural tissue from genome."""
        logger.info("Pipeline step 2: Developing neural tissue")
        t0 = time.time()

        dev = DevelopmentOperator()
        remaining_timeout = max(
            10, self.config.timeout_seconds - (time.time() - t0)
        )
        brain, tissue = dev.develop_and_build_brain(
            genome, vocab_size=vocab_size, use_cppn=self.config.use_cppn
        )

        elapsed = time.time() - t0
        self._collect_metrics("develop", {
            "time_seconds": elapsed,
            "n_neurons": tissue.N,
            "n_edges": len(tissue.src),
            "cppn_used": tissue.cppn is not None,
            "weight_mean": float(np.mean(tissue.w_slow)),
            "weight_std": float(np.std(tissue.w_slow)),
        })

        logger.info("Developed tissue: N=%d, edges=%d", tissue.N, len(tissue.src))
        return brain, tissue

    def _step_train(self, brain, data: str, vocab_size: int) -> None:
        """Step 3: Train the brain on data."""
        from ..tasks.text_task import build_vocab

        logger.info("Pipeline step 3: Training (%d steps)", self.config.training_steps)
        t0 = time.time()

        stoi, itos = build_vocab(data)
        chars = list(data)

        total_loss = 0.0
        loss_count = 0

        for step in range(self.config.training_steps):
            idx = step % (len(chars) - 1)
            token_id = stoi.get(chars[idx], 0)
            target_id = stoi.get(chars[idx + 1], 0)
            probs = brain.forward(token_id)
            loss = brain.learn(target_id, probs)
            total_loss += loss
            loss_count += 1

        avg_loss = total_loss / max(1, loss_count)
        elapsed = time.time() - t0

        self._collect_metrics("train", {
            "time_seconds": elapsed,
            "steps": self.config.training_steps,
            "avg_loss": avg_loss,
            "final_loss": loss,
        })

        logger.info("Training done: avg_loss=%.4f, time=%.1fs", avg_loss, elapsed)

    def _step_find_attractors(self, brain) -> list:
        """Step 4: Find attractors in the trained brain."""
        logger.info("Pipeline step 4: Finding attractors")
        t0 = time.time()

        dynamics = AttractorDynamics()

        # Build dense weight matrix
        N = brain.N
        W = np.zeros((N, N), dtype=np.float32)
        w_eff = brain.w_slow + brain.w_fast
        np.add.at(W, (brain.src, brain.dst), w_eff)

        attractors = dynamics.find_attractors(
            N=N, weights=W, theta=brain.theta, n_probes=100
        )

        elapsed = time.time() - t0
        self._collect_metrics("attractors", {
            "time_seconds": elapsed,
            "n_attractors": len(attractors),
            "top_basin_size": attractors[0].basin_size if attractors else 0,
        })

        logger.info("Found %d attractors", len(attractors))
        return attractors

    def _step_pattern_memory(self, brain, data: str, vocab_size: int) -> None:
        """Step 5: Imprint patterns into associative memory."""
        from ..tasks.text_task import build_vocab

        logger.info("Pipeline step 5: Pattern memory imprinting")
        t0 = time.time()

        stoi, itos = build_vocab(data)
        mem = PatternMemory(N=brain.N, sparsity=0.1)

        # Create patterns from text windows
        window_size = 5
        chars = list(data)
        n_patterns = 0

        for i in range(0, min(len(chars) - window_size, 50)):
            tokens = [stoi.get(chars[i + j], 0) for j in range(window_size)]
            pattern = mem.text_to_pattern(tokens, vocab_size=vocab_size)
            if mem.imprint_pattern(pattern):
                n_patterns += 1

        elapsed = time.time() - t0
        self._collect_metrics("patterns", {
            "time_seconds": elapsed,
            "n_patterns": n_patterns,
            "capacity_warning": mem.capacity_warning,
        })

        logger.info("Imprinted %d patterns", n_patterns)

    def _save_checkpoint(self, step_name: str, state: dict) -> None:
        """Save a checkpoint after a pipeline step."""
        if self.config.checkpoint_dir is None:
            return

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_data = {
            "step_name": step_name,
            "step_data": _make_json_serializable(state),
            "timestamp": time.time(),
        }

        path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{step_name}.json",
        )

        with open(path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        self._checkpoints.append(path)
        logger.info("Saved checkpoint: %s", path)

    def _load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint from disk."""
        with open(path) as f:
            return json.load(f)

    def _collect_metrics(self, step_name: str, metrics: dict) -> None:
        """Collect metrics for a pipeline step."""
        self._metrics[step_name] = metrics

    def _save_metrics_to_dir(self) -> None:
        """Save all collected metrics to metrics_dir."""
        if self.config.metrics_dir is None:
            return

        os.makedirs(self.config.metrics_dir, exist_ok=True)
        path = os.path.join(self.config.metrics_dir, "pipeline_metrics.json")

        with open(path, "w") as f:
            json.dump(_make_json_serializable(self._metrics), f, indent=2)

        logger.info("Saved metrics: %s", path)


def _make_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
