"""Tests for reconstruction operator."""

import pytest
import numpy as np
from magicbrain.brain import TextBrain
from magicbrain.tasks.text_task import build_vocab, train_loop
from magicbrain.neurogenesis.reconstruction import ReconstructionOperator
from magicbrain.neurogenesis.attractor_dynamics import AttractorDynamics


DEFAULT_GENOME = "30121033102301230112332100123"
SAMPLE_TEXT = "To be or not to be that is the question " * 20


class TestReconstructionOperator:
    def setup_method(self):
        self.stoi, self.itos = build_vocab(SAMPLE_TEXT)
        self.brain = TextBrain(DEFAULT_GENOME, len(self.stoi))
        # Quick training
        train_loop(self.brain, SAMPLE_TEXT, self.stoi, steps=200, verbose=False)
        self.recon = ReconstructionOperator()

    def test_autoregressive_reconstruction(self):
        result = self.recon.reconstruct_autoregressive(
            self.brain, self.stoi, self.itos, seed="To", length=50,
        )
        assert len(result.text) > 0
        assert result.method == "autoregressive"

    def test_cue_based_reconstruction(self):
        result = self.recon.reconstruct_from_cue(
            self.brain, "To be", self.stoi, self.itos, continuation_length=50,
        )
        assert len(result.text) > len("To be")
        assert result.method == "cue_based"
        assert result.text.startswith("To be")

    def test_measure_fidelity(self):
        original = "hello world this is a test"
        reconstructed = "hello world this is a best"
        fidelity = self.recon.measure_fidelity(original, reconstructed)

        assert 0 <= fidelity["char_accuracy"] <= 1
        assert 0 <= fidelity["bigram_overlap"] <= 1
        assert 0 <= fidelity["vocab_overlap"] <= 1
        assert fidelity["length_ratio"] > 0
        # One char differs: accuracy should be high
        assert fidelity["char_accuracy"] > 0.9

    def test_measure_fidelity_identical(self):
        text = "identical text"
        fidelity = self.recon.measure_fidelity(text, text)
        assert fidelity["char_accuracy"] == 1.0
        assert fidelity["bigram_overlap"] == 1.0

    def test_measure_compression(self):
        compression = self.recon.measure_compression(
            DEFAULT_GENOME, SAMPLE_TEXT,
        )
        assert compression["data_size_bytes"] > 0
        assert compression["genome_size_chars"] == len(DEFAULT_GENOME)
        assert compression["genome_ratio"] > 0

    def test_attractor_decoding(self):
        decodings = self.recon.reconstruct_from_attractors(
            self.brain, self.itos, n_probes=50,
        )
        assert len(decodings) > 0
        for dec in decodings:
            assert dec.token in self.itos.values() or dec.token == "?"
            assert 0 <= dec.confidence <= 1


class TestFidelityEdgeCases:
    def setup_method(self):
        self.recon = ReconstructionOperator()

    def test_empty_strings(self):
        fidelity = self.recon.measure_fidelity("", "")
        assert fidelity["char_accuracy"] == 0.0

    def test_completely_different(self):
        fidelity = self.recon.measure_fidelity("aaaa", "bbbb")
        assert fidelity["char_accuracy"] == 0.0
        assert fidelity["bigram_overlap"] == 0.0

    def test_different_lengths(self):
        fidelity = self.recon.measure_fidelity("short", "this is much longer")
        assert 0 <= fidelity["char_accuracy"] <= 1
        assert fidelity["length_ratio"] > 1
