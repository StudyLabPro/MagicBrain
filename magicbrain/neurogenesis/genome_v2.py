"""Extended Genome v2 format for NeuroGenesis Engine.

Extends the standard 24-position genome with additional sections:
  - Positions  0-23: Standard hyperparameters (compatible with genome.py)
  - Positions 24-55: CPPN architecture (hidden layers, activations)
  - Positions 56-71: Attractor dynamics (tau, momentum, energy params)
  - Positions 72+:   Pattern seeds (variable length)

The v2 genome is backward-compatible: first 24 positions decode identically
to the v1 genome. The rest is silently ignored by the v1 decoder.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..genome import decode_genome


class GenomeV2:
    """Extended genome representation with section-based structure."""

    # Section boundaries
    TOPOLOGY_START = 0
    TOPOLOGY_END = 24
    CPPN_START = 24
    CPPN_END = 56
    ATTRACTOR_START = 56
    ATTRACTOR_END = 72
    PATTERNS_START = 72

    def __init__(self, genome_str: str):
        self.raw = genome_str
        self.digits = [int(c) for c in genome_str if c in "0123"]

    @property
    def topology_section(self) -> list[int]:
        return self._section(self.TOPOLOGY_START, self.TOPOLOGY_END)

    @property
    def cppn_section(self) -> list[int]:
        return self._section(self.CPPN_START, self.CPPN_END)

    @property
    def attractor_section(self) -> list[int]:
        return self._section(self.ATTRACTOR_START, self.ATTRACTOR_END)

    @property
    def pattern_section(self) -> list[int]:
        if len(self.digits) > self.PATTERNS_START:
            return self.digits[self.PATTERNS_START:]
        return []

    def _section(self, start: int, end: int) -> list[int]:
        if len(self.digits) > start:
            return self.digits[start:min(end, len(self.digits))]
        return []

    @property
    def length(self) -> int:
        return len(self.digits)

    @property
    def is_extended(self) -> bool:
        """True if genome has CPPN/attractor sections."""
        return len(self.digits) >= self.CPPN_END

    def to_string(self) -> str:
        return "".join(str(d) for d in self.digits)

    def checksum(self) -> int:
        """Compute CRC32 checksum of the genome string for integrity verification."""
        return zlib.crc32(self.to_string().encode("ascii")) & 0xFFFFFFFF

    def validate(self) -> tuple[bool, list[str]]:
        """Validate genome format, length, and valid characters.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        if not self.raw:
            errors.append("Genome string is empty")
            return False, errors

        # Check valid characters
        invalid_chars = set(self.raw) - set("0123")
        if invalid_chars:
            errors.append(
                f"Invalid characters in genome: {invalid_chars}"
            )

        # Check minimum length
        if len(self.digits) < 24:
            errors.append(
                f"Genome too short: {len(self.digits)} digits "
                f"(minimum 24 for topology section)"
            )

        # Check section boundaries for extended genomes
        if self.is_extended:
            if len(self.cppn_section) < 32:
                errors.append(
                    f"CPPN section incomplete: {len(self.cppn_section)} digits "
                    f"(expected 32)"
                )
            if len(self.attractor_section) < 16:
                errors.append(
                    f"Attractor section incomplete: "
                    f"{len(self.attractor_section)} digits (expected 16)"
                )

        return len(errors) == 0, errors

    def pretty_print(self) -> str:
        """Decode all sections and format as human-readable text."""
        lines: list[str] = []
        lines.append(f"GenomeV2 (length={self.length}, extended={self.is_extended})")
        lines.append(f"  Raw: {self.raw[:40]}{'...' if len(self.raw) > 40 else ''}")
        lines.append(f"  Checksum: 0x{self.checksum():08X}")
        lines.append("")

        # Topology section
        topo = self.topology_section
        lines.append(f"  [Topology] positions 0-23 ({len(topo)} digits)")
        if topo:
            params = decode_genome(self.to_string()[:24])
            lines.append(f"    N={params['N']}, K={params['K']}")
            lines.append(f"    lr={params.get('lr', '?')}")
            lines.append(f"    seed={params.get('seed', '?')}")

        # CPPN section
        cppn = self.cppn_section
        if cppn:
            lines.append(f"  [CPPN] positions 24-55 ({len(cppn)} digits)")
            n_layers = min(3, max(1, cppn[0] + 1)) if cppn else 0
            lines.append(f"    n_layers={n_layers}")

        # Attractor section
        att = self.attractor_section
        if att:
            lines.append(f"  [Attractor] positions 56-71 ({len(att)} digits)")

        # Pattern section
        pat = self.pattern_section
        if pat:
            lines.append(f"  [Patterns] positions 72+ ({len(pat)} digits)")

        return "\n".join(lines)

    def diff(self, other: GenomeV2) -> dict[str, dict]:
        """Compare this genome with another, returning differences by section.

        Args:
            other: Another GenomeV2 to compare against.

        Returns:
            Dict with section names as keys, each containing 'self' and 'other'
            digit lists and a 'changed_positions' list of indices that differ.
        """
        result: dict[str, dict] = {}

        sections = {
            "topology": (self.topology_section, other.topology_section),
            "cppn": (self.cppn_section, other.cppn_section),
            "attractor": (self.attractor_section, other.attractor_section),
            "patterns": (self.pattern_section, other.pattern_section),
        }

        for name, (self_sec, other_sec) in sections.items():
            max_len = max(len(self_sec), len(other_sec))
            if max_len == 0:
                continue
            changed = []
            for i in range(max_len):
                v_self = self_sec[i] if i < len(self_sec) else None
                v_other = other_sec[i] if i < len(other_sec) else None
                if v_self != v_other:
                    changed.append(i)

            if changed:
                result[name] = {
                    "self": self_sec,
                    "other": other_sec,
                    "changed_positions": changed,
                }

        return result

    @classmethod
    def from_sections(
        cls,
        topology: str,
        cppn: str = "",
        attractor: str = "",
        patterns: str = "",
    ) -> GenomeV2:
        """Build genome from individual sections."""
        # Pad topology to 24
        topo_digits = [int(c) for c in topology if c in "0123"]
        while len(topo_digits) < 24:
            topo_digits.append(1)

        full = topo_digits[:24]

        if cppn:
            cppn_d = [int(c) for c in cppn if c in "0123"]
            while len(cppn_d) < 32:
                cppn_d.append(1)
            full.extend(cppn_d[:32])

        if attractor:
            att_d = [int(c) for c in attractor if c in "0123"]
            while len(att_d) < 16:
                att_d.append(1)
            full.extend(att_d[:16])

        if patterns:
            pat_d = [int(c) for c in patterns if c in "0123"]
            full.extend(pat_d)

        return cls("".join(str(d) for d in full))


def decode_genome_v2(genome: str) -> dict:
    """Decode extended genome into full parameter dictionary.

    Returns all v1 parameters plus CPPN and attractor parameters.
    """
    gv2 = GenomeV2(genome)
    g = np.array(gv2.digits, dtype=np.int32)

    # Base parameters from v1 decoder
    base_str = "".join(str(d) for d in gv2.digits[:24])
    params = decode_genome(base_str)

    # Helper: read base-4 with wrapping
    def b4(i: int, n: int) -> int:
        x = 0
        for k in range(n):
            idx = (i + k) % len(g)
            x = x * 4 + int(g[idx])
        return x

    # CPPN parameters (positions 24-55)
    if gv2.is_extended:
        cppn_section = gv2.cppn_section

        # Number of hidden layers (1-3)
        n_cppn_layers = min(3, max(1, cppn_section[0] + 1)) if cppn_section else 2

        # Hidden layer widths
        cppn_widths = []
        for i in range(n_cppn_layers):
            idx = 1 + i * 2
            if idx + 1 < len(cppn_section):
                w = 4 + 4 * (cppn_section[idx] * 4 + cppn_section[idx + 1])
                cppn_widths.append(min(32, max(4, w)))
            else:
                cppn_widths.append(8)

        # Activation function IDs for each hidden neuron
        act_start = 1 + n_cppn_layers * 2
        cppn_activations = cppn_section[act_start:] if act_start < len(cppn_section) else []

        # CPPN weight scale
        if len(cppn_section) > 30:
            cppn_weight_scale = 0.2 + 0.2 * cppn_section[30]
        else:
            cppn_weight_scale = 0.5

        # CPPN output scale
        if len(cppn_section) > 31:
            cppn_output_scale = 0.02 + 0.03 * cppn_section[31]
        else:
            cppn_output_scale = 0.1

        params["cppn_n_layers"] = n_cppn_layers
        params["cppn_widths"] = cppn_widths
        params["cppn_activations"] = cppn_activations
        params["cppn_weight_scale"] = cppn_weight_scale
        params["cppn_output_scale"] = cppn_output_scale
        params["cppn_digits"] = cppn_section

    # Attractor parameters (positions 56-71)
    att = gv2.attractor_section
    if att:
        # tau: temperature for sigmoid (0.1 - 1.0)
        params["attractor_tau"] = 0.1 + 0.3 * (att[0] if len(att) > 0 else 1)

        # momentum: mixing factor (0.5 - 0.9)
        params["attractor_momentum"] = 0.5 + 0.1 * (att[1] if len(att) > 1 else 2)

        # lambda_sparse: sparsity penalty (0.001 - 0.05)
        params["attractor_lambda_sparse"] = 0.001 + 0.013 * (att[2] if len(att) > 2 else 1)

        # max_iterations (50 - 350)
        params["attractor_max_iter"] = 50 + 100 * (att[3] if len(att) > 3 else 2)

        # tolerance (1e-5 to 1e-3)
        tol_idx = att[4] if len(att) > 4 else 2
        params["attractor_tolerance"] = [1e-5, 1e-4, 5e-4, 1e-3][min(3, tol_idx)]

        # Pattern sparsity (0.05 - 0.20)
        params["pattern_sparsity"] = 0.05 + 0.05 * (att[5] if len(att) > 5 else 1)

        # Energy function type (future: different energy landscapes)
        params["energy_type"] = att[6] if len(att) > 6 else 0

    # Pattern seeds (positions 72+)
    if gv2.pattern_section:
        params["pattern_seeds"] = gv2.pattern_section

    return params
