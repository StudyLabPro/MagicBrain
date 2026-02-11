"""Tests for GenomeCompiler: Dataset -> Genome compilation."""

import pytest
import numpy as np
from magicbrain.neurogenesis.compiler import (
    GenomeCompiler,
    analyze_dataset,
    DatasetStats,
    _hex_to_base4,
)
from magicbrain.genome import decode_genome


SAMPLE_TEXT = "To be, or not to be, that is the question."
SHAKESPEARE = (
    "From fairest creatures we desire increase, "
    "That thereby beauty's rose might never die."
) * 20
REPETITIVE = "abcabc" * 100
CODE_TEXT = "def foo(x):\n    return x * 2\n" * 50


class TestAnalyzeDataset:
    def test_basic_stats(self):
        stats = analyze_dataset(SAMPLE_TEXT)
        assert stats.size == len(SAMPLE_TEXT)
        assert stats.vocab_size > 0
        assert stats.entropy > 0

    def test_entropy_range(self):
        stats = analyze_dataset(SAMPLE_TEXT)
        # Shannon entropy for text: typically 2-5 bits
        assert 0 < stats.entropy < 8

    def test_repetitive_text(self):
        stats_rep = analyze_dataset(REPETITIVE)
        stats_normal = analyze_dataset(SHAKESPEARE)
        # Repetitive text should have higher top-bigram concentration
        assert stats_rep.top_ngram_concentration > stats_normal.top_ngram_concentration
        # And lower entropy
        assert stats_rep.entropy < stats_normal.entropy

    def test_vocab_size(self):
        stats = analyze_dataset("aabbcc")
        assert stats.vocab_size == 3

    def test_empty_like(self):
        stats = analyze_dataset("a")
        assert stats.size == 1
        assert stats.vocab_size == 1


class TestHexToBase4:
    def test_single_digit(self):
        assert _hex_to_base4("0") == "00"
        assert _hex_to_base4("f") == "33"
        assert _hex_to_base4("5") == "11"

    def test_full_hex(self):
        result = _hex_to_base4("ff")
        assert result == "3333"
        assert all(c in "0123" for c in result)


class TestGenomeCompiler:
    def setup_method(self):
        self.compiler = GenomeCompiler()

    def test_hash_strategy_produces_valid_genome(self):
        genome = self.compiler.compile(SAMPLE_TEXT, strategy="hash")
        assert len(genome) == 28
        assert all(c in "0123" for c in genome)

    def test_statistical_strategy_produces_valid_genome(self):
        genome = self.compiler.compile(SAMPLE_TEXT, strategy="statistical")
        assert len(genome) == 28
        assert all(c in "0123" for c in genome)

    def test_hybrid_strategy_produces_valid_genome(self):
        genome = self.compiler.compile(SAMPLE_TEXT, strategy="hybrid")
        assert len(genome) == 28
        assert all(c in "0123" for c in genome)

    def test_deterministic_hash(self):
        g1 = self.compiler.compile(SAMPLE_TEXT, strategy="hash")
        g2 = self.compiler.compile(SAMPLE_TEXT, strategy="hash")
        assert g1 == g2

    def test_deterministic_statistical(self):
        g1 = self.compiler.compile(SAMPLE_TEXT, strategy="statistical")
        g2 = self.compiler.compile(SAMPLE_TEXT, strategy="statistical")
        assert g1 == g2

    def test_different_data_different_genome(self):
        g1 = self.compiler.compile("Hello world", strategy="hash")
        g2 = self.compiler.compile("Goodbye world", strategy="hash")
        assert g1 != g2

    def test_custom_length(self):
        genome = self.compiler.compile(SAMPLE_TEXT, strategy="hash", genome_length=64)
        assert len(genome) == 64

    def test_minimum_length_enforced(self):
        genome = self.compiler.compile(SAMPLE_TEXT, strategy="hash", genome_length=10)
        assert len(genome) >= 24

    def test_genome_decodable(self):
        """Generated genome must be decodable by the standard decoder."""
        for strategy in ["hash", "statistical", "hybrid"]:
            genome = self.compiler.compile(SHAKESPEARE, strategy=strategy)
            params = decode_genome(genome)
            assert 256 <= params["N"] <= 16576
            assert 8 <= params["K"] <= 20
            assert params["lr"] > 0

    def test_statistical_larger_data_larger_N(self):
        """Statistical strategy should give larger N for larger datasets."""
        small = self.compiler.compile("hi", strategy="statistical")
        large = self.compiler.compile(SHAKESPEARE, strategy="statistical")
        params_small = decode_genome(small)
        params_large = decode_genome(large)
        assert params_large["N"] >= params_small["N"]

    def test_bytes_input(self):
        genome = self.compiler.compile(b"binary data here", strategy="hash")
        assert len(genome) == 28
        assert all(c in "0123" for c in genome)

    def test_compile_with_metadata(self):
        result = self.compiler.compile_with_metadata(SAMPLE_TEXT, strategy="statistical")
        assert "genome" in result
        assert "stats" in result
        assert "data_hash" in result
        assert result["strategy"] == "statistical"
        assert result["data_size"] > 0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            self.compiler.compile(SAMPLE_TEXT, strategy="nonexistent")


class TestGenomeCompilerDiversity:
    """Test that different datasets produce meaningfully different genomes."""

    def setup_method(self):
        self.compiler = GenomeCompiler()

    def test_code_vs_prose_different(self):
        g_code = self.compiler.compile(CODE_TEXT, strategy="statistical")
        g_prose = self.compiler.compile(SHAKESPEARE, strategy="statistical")
        # At least some positions should differ
        diffs = sum(1 for a, b in zip(g_code, g_prose) if a != b)
        assert diffs > 0

    def test_repetitive_vs_diverse_different(self):
        g_rep = self.compiler.compile(REPETITIVE, strategy="statistical")
        g_diverse = self.compiler.compile(SHAKESPEARE, strategy="statistical")
        diffs = sum(1 for a, b in zip(g_rep, g_diverse) if a != b)
        assert diffs > 0
