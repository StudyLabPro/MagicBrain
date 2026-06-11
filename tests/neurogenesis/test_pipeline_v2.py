"""Tests for NeurogenesisPipeline v2 production hardening."""

import os
import json
import tempfile

import pytest
import numpy as np

from magicbrain.neurogenesis.pipeline import (
    NeurogenesisPipeline,
    PipelineConfig,
    PipelineResult,
)
from magicbrain.neurogenesis.compiler import GenomeCompiler, CompilationMetrics
from magicbrain.neurogenesis.development import DevelopmentOperator, DevelopmentMetrics
from magicbrain.neurogenesis.attractor_dynamics import AttractorDynamics, AttractorMetrics
from magicbrain.neurogenesis.pattern_memory import PatternMemory, PatternQualityMetrics
from magicbrain.neurogenesis.genome_v2 import GenomeV2


SAMPLE_DATA = "To be, or not to be, that is the question. " * 20
SHORT_DATA = "Hello world! This is a test dataset for compilation."


class TestFullPipeline:
    """Test the complete pipeline runs end-to-end."""

    def test_full_pipeline(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=50,
            timeout_seconds=120,
        )
        pipeline = NeurogenesisPipeline(config)
        result = pipeline.run(SAMPLE_DATA, vocab_size=30)

        assert isinstance(result, PipelineResult)
        assert result.genome
        assert result.tissue is not None
        assert result.brain is not None
        assert isinstance(result.attractors, list)
        assert isinstance(result.metrics, dict)
        assert "compile" in result.metrics
        assert "develop" in result.metrics
        assert "train" in result.metrics
        assert "attractors" in result.metrics
        assert "total_time_seconds" in result.metrics

    def test_pipeline_with_auto_strategy(self):
        config = PipelineConfig(
            strategy="auto",
            use_cppn=False,
            training_steps=20,
        )
        pipeline = NeurogenesisPipeline(config)
        result = pipeline.run(SAMPLE_DATA, vocab_size=30)
        assert result.genome
        assert result.metrics["compile"]["strategy_used"] in ("hash", "statistical", "hybrid")

    def test_pipeline_with_seed(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=10,
            seed=42,
        )
        p1 = NeurogenesisPipeline(config)
        r1 = p1.run(SAMPLE_DATA, vocab_size=30)

        p2 = NeurogenesisPipeline(config)
        r2 = p2.run(SAMPLE_DATA, vocab_size=30)

        assert r1.genome == r2.genome


class TestPipelineCheckpointing:
    """Test checkpoint save/load."""

    def test_pipeline_with_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                strategy="hash",
                use_cppn=False,
                training_steps=20,
                checkpoint_dir=tmpdir,
            )
            pipeline = NeurogenesisPipeline(config)
            result = pipeline.run(SAMPLE_DATA, vocab_size=30)

            # Verify checkpoints were saved
            assert len(result.checkpoints) > 0
            for cp_path in result.checkpoints:
                assert os.path.exists(cp_path)
                with open(cp_path) as f:
                    data = json.load(f)
                assert "step_name" in data
                assert "step_data" in data
                assert "timestamp" in data

    def test_pipeline_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run first pipeline to get checkpoints
            config = PipelineConfig(
                strategy="hash",
                use_cppn=False,
                training_steps=20,
                checkpoint_dir=tmpdir,
            )
            pipeline = NeurogenesisPipeline(config)
            first_result = pipeline.run(SAMPLE_DATA, vocab_size=30)

            # Resume from the compile checkpoint
            compile_cp = os.path.join(tmpdir, "checkpoint_compile.json")
            assert os.path.exists(compile_cp)

            pipeline2 = NeurogenesisPipeline(config)
            resumed_result = pipeline2.resume(compile_cp, SAMPLE_DATA, vocab_size=30)

            assert resumed_result.genome == first_result.genome
            assert resumed_result.brain is not None
            assert "resumed_from" in resumed_result.metrics


class TestPipelineErrorRecovery:
    """Test error handling and recovery."""

    def test_pipeline_error_recovery_invalid_data(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=10,
        )
        pipeline = NeurogenesisPipeline(config)

        with pytest.raises(ValueError, match="too short"):
            pipeline.run("x", vocab_size=30)

    def test_pipeline_error_recovery_empty_data(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=10,
        )
        pipeline = NeurogenesisPipeline(config)

        with pytest.raises(ValueError, match="empty"):
            pipeline.run("", vocab_size=30)


class TestPipelineMetrics:
    """Test metrics collection."""

    def test_pipeline_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                strategy="hash",
                use_cppn=False,
                training_steps=20,
                metrics_dir=tmpdir,
            )
            pipeline = NeurogenesisPipeline(config)
            result = pipeline.run(SAMPLE_DATA, vocab_size=30)

            # Check metrics were saved to file
            metrics_path = os.path.join(tmpdir, "pipeline_metrics.json")
            assert os.path.exists(metrics_path)

            with open(metrics_path) as f:
                saved_metrics = json.load(f)

            assert "compile" in saved_metrics
            assert "develop" in saved_metrics
            assert "train" in saved_metrics
            assert "attractors" in saved_metrics
            assert "total_time_seconds" in saved_metrics

    def test_compile_metrics_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                strategy="hash",
                use_cppn=False,
                training_steps=10,
                metrics_dir=tmpdir,
            )
            pipeline = NeurogenesisPipeline(config)
            result = pipeline.run(SAMPLE_DATA, vocab_size=30)

            cm = result.metrics["compile"]
            assert "time_seconds" in cm
            assert "genome_length" in cm
            assert "strategy_used" in cm
            assert "quality_score" in cm
            assert cm["time_seconds"] >= 0

    def test_develop_metrics_fields(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=10,
        )
        pipeline = NeurogenesisPipeline(config)
        result = pipeline.run(SAMPLE_DATA, vocab_size=30)

        dm = result.metrics["develop"]
        assert "n_neurons" in dm
        assert "n_edges" in dm
        assert "cppn_used" in dm
        assert dm["n_neurons"] > 0
        assert dm["n_edges"] > 0

    def test_train_metrics_fields(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=20,
        )
        pipeline = NeurogenesisPipeline(config)
        result = pipeline.run(SAMPLE_DATA, vocab_size=30)

        tm = result.metrics["train"]
        assert "avg_loss" in tm
        assert "steps" in tm
        assert tm["steps"] == 20


class TestPipelineTimeout:
    """Test timeout handling."""

    def test_pipeline_timeout(self):
        config = PipelineConfig(
            strategy="hash",
            use_cppn=False,
            training_steps=10,
            timeout_seconds=0.0001,
        )
        pipeline = NeurogenesisPipeline(config)
        # Very short timeout should trigger a timeout
        # But since compile is fast, it may still complete
        # We just verify the pipeline doesn't crash
        result = pipeline.run(SAMPLE_DATA, vocab_size=30)
        # Either timed out or completed — both are OK
        assert isinstance(result, PipelineResult)


class TestCompilerHardening:
    """Test compiler validation and metrics."""

    def setup_method(self):
        self.compiler = GenomeCompiler()

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.compiler.compile("")

    def test_short_data_raises(self):
        with pytest.raises(ValueError, match="too short"):
            self.compiler.compile("abc")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.compiler.compile("   \n\t  ")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.compiler.compile(b"")

    def test_auto_strategy_small(self):
        # Small data should use "hash"
        genome = self.compiler.compile(SHORT_DATA, strategy="auto")
        assert all(c in "0123" for c in genome)

    def test_auto_strategy_medium(self):
        medium_data = "x" * 5000
        genome = self.compiler.compile(medium_data, strategy="auto")
        assert len(genome) >= 24

    def test_seed_determinism(self):
        g1 = self.compiler.compile(SAMPLE_DATA, strategy="statistical", seed=42)
        g2 = self.compiler.compile(SAMPLE_DATA, strategy="statistical", seed=42)
        assert g1 == g2

    def test_different_seeds_different_genomes(self):
        g1 = self.compiler.compile(SAMPLE_DATA, strategy="hash", seed=1)
        g2 = self.compiler.compile(SAMPLE_DATA, strategy="hash", seed=2)
        assert g1 != g2

    def test_compile_with_metrics(self):
        genome, metrics = self.compiler.compile_with_metrics(SAMPLE_DATA, strategy="hash")
        assert isinstance(metrics, CompilationMetrics)
        assert metrics.time_seconds >= 0
        assert 0 <= metrics.genome_quality_score <= 1
        assert metrics.strategy_used == "hash"
        assert "size" in metrics.data_stats


class TestDevelopmentHardening:
    """Test development operator hardening."""

    def setup_method(self):
        self.dev = DevelopmentOperator()
        self.genome = "30121033102301230112332100123"

    def test_develop_with_metrics(self):
        tissue, metrics = self.dev.develop_with_metrics(
            self.genome, vocab_size=50, use_cppn=False
        )
        assert isinstance(metrics, DevelopmentMetrics)
        assert metrics.n_neurons > 0
        assert metrics.n_edges > 0
        assert metrics.time_seconds >= 0
        assert not metrics.cppn_used

    def test_develop_with_metrics_cppn(self):
        tissue, metrics = self.dev.develop_with_metrics(
            self.genome, vocab_size=50, use_cppn=True
        )
        assert metrics.cppn_used

    def test_tissue_no_nan(self):
        tissue = self.dev.develop(self.genome, vocab_size=50, use_cppn=True)
        assert not np.any(np.isnan(tissue.w_slow))
        assert not np.any(np.isnan(tissue.w_fast))
        assert not np.any(np.isnan(tissue.theta))


class TestAttractorHardening:
    """Test attractor dynamics hardening."""

    def setup_method(self):
        self.dynamics = AttractorDynamics(max_iterations=50)
        self.N = 32
        self.rng = np.random.default_rng(42)
        self.W = self.rng.normal(0, 0.1, (self.N, self.N)).astype(np.float32)
        self.W = 0.5 * (self.W + self.W.T)
        np.fill_diagonal(self.W, 0)
        self.theta = np.zeros(self.N, dtype=np.float32)

    def test_converge_with_timeout(self):
        cue = self.rng.random(self.N).astype(np.float32) * 0.5
        result = self.dynamics.converge(
            cue, self.W, self.theta, max_time_seconds=10
        )
        assert isinstance(result.converged, bool)

    def test_find_attractors_with_metrics(self):
        attractors, metrics = self.dynamics.find_attractors_with_metrics(
            self.N, self.W, self.theta, n_probes=20
        )
        assert isinstance(metrics, AttractorMetrics)
        assert metrics.n_attractors >= 0
        assert metrics.basin_stability >= 0
        assert metrics.total_basin_coverage >= 0


class TestPatternMemoryHardening:
    """Test pattern memory hardening."""

    def setup_method(self):
        self.N = 64
        self.mem = PatternMemory(N=self.N, sparsity=0.15)
        self.rng = np.random.default_rng(42)

    def _random_pattern(self) -> np.ndarray:
        p = np.zeros(self.N, dtype=np.float32)
        n_active = max(1, int(self.N * 0.15))
        active = self.rng.choice(self.N, size=n_active, replace=False)
        p[active] = 1.0
        return p

    def test_capacity_warning(self):
        # Use larger max_capacity_fraction so we can exceed 0.14*N
        mem = PatternMemory(N=self.N, sparsity=0.15, max_capacity_fraction=0.20)
        # Initially no warning
        assert not mem.capacity_warning
        # Imprint many patterns to exceed 0.14 * N
        threshold = int(0.14 * self.N) + 1
        for _ in range(threshold):
            mem.imprint_pattern(self._random_pattern())
        assert mem.capacity_warning

    def test_quality_metrics_empty(self):
        metrics = self.mem.quality_metrics()
        assert isinstance(metrics, PatternQualityMetrics)
        assert metrics.avg_recall_fidelity == 0.0
        assert metrics.capacity_ratio == 0.0

    def test_quality_metrics_with_patterns(self):
        for _ in range(3):
            self.mem.imprint_pattern(self._random_pattern())
        metrics = self.mem.quality_metrics()
        assert metrics.avg_recall_fidelity >= 0
        assert metrics.capacity_ratio > 0

    def test_batch_imprint_with_progress(self):
        progress_log = []

        def on_progress(current, total):
            progress_log.append((current, total))

        patterns = np.array([self._random_pattern() for _ in range(5)])
        self.mem.imprint_patterns_batch(patterns, progress_callback=on_progress)

        assert len(progress_log) == 5
        assert progress_log[0] == (1, 5)
        assert progress_log[-1] == (5, 5)


class TestGenomeV2Hardening:
    """Test GenomeV2 new methods."""

    def test_checksum(self):
        g = GenomeV2("30121033102301230112332100123")
        cs = g.checksum()
        assert isinstance(cs, int)
        assert 0 <= cs <= 0xFFFFFFFF

    def test_checksum_deterministic(self):
        g1 = GenomeV2("30121033102301230112332100123")
        g2 = GenomeV2("30121033102301230112332100123")
        assert g1.checksum() == g2.checksum()

    def test_checksum_different_genomes(self):
        g1 = GenomeV2("30121033102301230112332100123")
        g2 = GenomeV2("10121033102301230112332100123")
        assert g1.checksum() != g2.checksum()

    def test_validate_valid(self):
        g = GenomeV2("30121033102301230112332100123")
        valid, errors = g.validate()
        assert valid
        assert len(errors) == 0

    def test_validate_empty(self):
        g = GenomeV2("")
        valid, errors = g.validate()
        assert not valid
        assert len(errors) > 0

    def test_validate_short(self):
        g = GenomeV2("301")
        valid, errors = g.validate()
        assert not valid
        assert any("too short" in e for e in errors)

    def test_validate_invalid_chars(self):
        g = GenomeV2("301210331023012301123321001239abc")
        valid, errors = g.validate()
        # Invalid chars should be flagged
        assert any("Invalid" in e for e in errors)

    def test_pretty_print(self):
        g = GenomeV2("30121033102301230112332100123")
        pp = g.pretty_print()
        assert "GenomeV2" in pp
        assert "Topology" in pp
        assert "Checksum" in pp

    def test_pretty_print_extended(self):
        extended = "30121033102301230112332100123" + "1" * (72 - 28)
        g = GenomeV2(extended)
        pp = g.pretty_print()
        assert "CPPN" in pp
        assert "Attractor" in pp

    def test_diff_identical(self):
        g1 = GenomeV2("30121033102301230112332100123")
        g2 = GenomeV2("30121033102301230112332100123")
        diff = g1.diff(g2)
        assert len(diff) == 0

    def test_diff_different(self):
        g1 = GenomeV2("30121033102301230112332100123")
        g2 = GenomeV2("10121033102301230112332100123")
        diff = g1.diff(g2)
        assert "topology" in diff
        assert 0 in diff["topology"]["changed_positions"]

    def test_diff_different_lengths(self):
        g1 = GenomeV2("30121033102301230112332100123")
        g2 = GenomeV2("30121033102301230112332100123" + "1" * 44)
        diff = g1.diff(g2)
        # g2 has extra CPPN and attractor sections
        assert "cppn" in diff or "attractor" in diff
