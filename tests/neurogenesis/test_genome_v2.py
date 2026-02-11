"""Tests for extended genome v2 format."""

import pytest
import numpy as np
from magicbrain.neurogenesis.genome_v2 import GenomeV2, decode_genome_v2
from magicbrain.genome import decode_genome


DEFAULT_GENOME = "30121033102301230112332100123"


class TestGenomeV2:
    def test_short_genome_compatible(self):
        """Short genomes should decode identically to v1."""
        gv2 = GenomeV2(DEFAULT_GENOME)
        assert not gv2.is_extended
        assert gv2.length == len(DEFAULT_GENOME)

    def test_topology_section(self):
        gv2 = GenomeV2(DEFAULT_GENOME)
        topo = gv2.topology_section
        assert len(topo) <= 24
        assert all(0 <= d <= 3 for d in topo)

    def test_extended_genome(self):
        # Create 72-char genome
        extended = DEFAULT_GENOME + "1" * (72 - len(DEFAULT_GENOME))
        gv2 = GenomeV2(extended)
        assert gv2.is_extended
        assert len(gv2.cppn_section) == 32
        assert len(gv2.attractor_section) == 16

    def test_from_sections(self):
        gv2 = GenomeV2.from_sections(
            topology="30121033102301230112332100",
            cppn="10230123012301230123012301230123",
            attractor="0123012301230123",
        )
        assert gv2.length == 72
        assert gv2.is_extended

    def test_from_sections_with_patterns(self):
        gv2 = GenomeV2.from_sections(
            topology="301210331023012301123321",
            cppn="10230123012301230123012301230123",
            attractor="0123012301230123",
            patterns="012301230123",
        )
        assert gv2.length > 72
        assert len(gv2.pattern_section) == 12

    def test_to_string(self):
        gv2 = GenomeV2(DEFAULT_GENOME)
        assert all(c in "0123" for c in gv2.to_string())


class TestDecodeGenomeV2:
    def test_backward_compatible(self):
        """V2 decoder should produce same base params as v1 for short genomes."""
        params_v1 = decode_genome(DEFAULT_GENOME)
        params_v2 = decode_genome_v2(DEFAULT_GENOME)

        for key in params_v1:
            assert key in params_v2
            assert params_v1[key] == params_v2[key], f"Mismatch for {key}"

    def test_extended_params(self):
        extended = DEFAULT_GENOME + "1" * (72 - len(DEFAULT_GENOME))
        params = decode_genome_v2(extended)

        # Should have CPPN params
        assert "cppn_n_layers" in params
        assert "cppn_widths" in params
        assert isinstance(params["cppn_widths"], list)

        # Should have attractor params
        assert "attractor_tau" in params
        assert "attractor_momentum" in params
        assert "attractor_lambda_sparse" in params
        assert "attractor_max_iter" in params
        assert "attractor_tolerance" in params

    def test_attractor_tau_range(self):
        """Attractor tau should be in valid range."""
        for digit in "0123":
            genome = DEFAULT_GENOME + "1" * 32 + digit + "1" * 15
            params = decode_genome_v2(genome)
            assert 0.1 <= params["attractor_tau"] <= 1.0

    def test_attractor_momentum_range(self):
        for digit in "0123":
            genome = DEFAULT_GENOME + "1" * 32 + "1" + digit + "1" * 14
            params = decode_genome_v2(genome)
            assert 0.5 <= params["attractor_momentum"] <= 0.9

    def test_pattern_seeds(self):
        genome = DEFAULT_GENOME + "1" * (72 - len(DEFAULT_GENOME)) + "01230123"
        params = decode_genome_v2(genome)
        assert "pattern_seeds" in params
        assert len(params["pattern_seeds"]) == 8

    def test_cppn_layers_range(self):
        for digit in "0123":
            genome = DEFAULT_GENOME + digit + "1" * (71 - len(DEFAULT_GENOME))
            params = decode_genome_v2(genome)
            assert 1 <= params["cppn_n_layers"] <= 3
