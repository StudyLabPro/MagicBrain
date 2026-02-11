"""Reconstruction Operator: Attractor State -> Data.

Extracts data from attractor states of a trained network.
Multiple reconstruction modes:
  1. Autoregressive — standard sampling from trained TextBrain
  2. Attractor decoding — decode attractor state via readout matrix
  3. Full scan — find all attractors and decode each
  4. Cue-based recall — provide partial data, reconstruct the rest
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from .attractor_dynamics import AttractorDynamics
from .pattern_memory import PatternMemory, _cosine_similarity

if TYPE_CHECKING:
    from ..brain import TextBrain


class ReconstructionResult(NamedTuple):
    """Result of data reconstruction."""
    text: str
    method: str
    fidelity: float          # similarity to original (0-1)
    n_attractors_used: int
    compression_ratio: float  # genome_size / data_size


class AttractorDecoding(NamedTuple):
    """Decoded attractor."""
    attractor_index: int
    state: np.ndarray
    token_id: int
    token: str
    confidence: float  # softmax probability of top token
    energy: float


class ReconstructionOperator:
    """Reconstructs data from trained neural network states."""

    def __init__(self, dynamics: AttractorDynamics | None = None):
        self.dynamics = dynamics or AttractorDynamics()

    def reconstruct_autoregressive(
        self,
        brain: TextBrain,
        stoi: dict,
        itos: dict,
        seed: str = "",
        length: int = 500,
        temperature: float = 0.75,
    ) -> ReconstructionResult:
        """Reconstruct text via autoregressive sampling.

        Standard next-token prediction approach. Uses the trained
        brain's forward pass to generate text.
        """
        from ..sampling import sample

        if not seed and stoi:
            seed = next(iter(stoi.keys()))

        text = sample(
            brain, stoi, itos, seed=seed, n=length,
            temperature=temperature,
        )

        return ReconstructionResult(
            text=text,
            method="autoregressive",
            fidelity=0.0,  # needs original to compute
            n_attractors_used=0,
            compression_ratio=0.0,
        )

    def reconstruct_from_attractors(
        self,
        brain: TextBrain,
        itos: dict,
        n_probes: int = 500,
    ) -> list[AttractorDecoding]:
        """Find attractors and decode each to tokens.

        Discovers the network's stable states and maps each to the
        vocabulary space via the readout matrix R.
        """
        # Build dense weight matrix for attractor search
        W_dense = self._build_dense_weights(brain)

        attractors = self.dynamics.find_attractors(
            N=brain.N,
            weights=W_dense,
            theta=brain.theta,
            n_probes=n_probes,
        )

        decodings: list[AttractorDecoding] = []
        for i, att in enumerate(attractors):
            logits = att.state @ brain.R + brain.b
            # Softmax
            logits = logits - np.max(logits)
            exp_l = np.exp(logits)
            probs = exp_l / (np.sum(exp_l) + 1e-9)

            token_id = int(np.argmax(probs))
            confidence = float(probs[token_id])
            token = itos.get(token_id, "?")

            decodings.append(AttractorDecoding(
                attractor_index=i,
                state=att.state,
                token_id=token_id,
                token=token,
                confidence=confidence,
                energy=att.energy,
            ))

        return decodings

    def reconstruct_from_cue(
        self,
        brain: TextBrain,
        cue_text: str,
        stoi: dict,
        itos: dict,
        continuation_length: int = 200,
    ) -> ReconstructionResult:
        """Reconstruct data by priming with a partial cue.

        Feeds the cue text through the brain to establish context,
        then generates continuation from the activated state.
        """
        brain.reset_state()

        # Prime with cue
        for ch in cue_text:
            if ch in stoi:
                brain.forward(stoi[ch])

        # Generate continuation
        if cue_text and cue_text[-1] in stoi:
            last_token = stoi[cue_text[-1]]
        elif stoi:
            last_token = 0
        else:
            return ReconstructionResult(
                text=cue_text, method="cue_based",
                fidelity=0.0, n_attractors_used=0, compression_ratio=0.0,
            )

        generated = list(cue_text)
        x = last_token

        for _ in range(continuation_length):
            probs = brain.forward(x)
            x = int(np.argmax(probs))
            generated.append(itos.get(x, "?"))

        return ReconstructionResult(
            text="".join(generated),
            method="cue_based",
            fidelity=0.0,
            n_attractors_used=0,
            compression_ratio=0.0,
        )

    def measure_fidelity(
        self,
        original: str,
        reconstructed: str,
    ) -> dict:
        """Measure reconstruction fidelity with multiple metrics.

        Returns:
            Dict with char_accuracy, bigram_overlap, trigram_overlap,
            vocab_overlap, and length_ratio.
        """
        # Character-level accuracy (positional)
        min_len = min(len(original), len(reconstructed))
        if min_len > 0:
            matches = sum(
                1 for a, b in zip(original[:min_len], reconstructed[:min_len])
                if a == b
            )
            char_accuracy = matches / min_len
        else:
            char_accuracy = 0.0

        # N-gram overlap (set-based)
        def ngram_overlap(text1: str, text2: str, n: int) -> float:
            if len(text1) < n or len(text2) < n:
                return 0.0
            set1 = set(text1[i:i + n] for i in range(len(text1) - n + 1))
            set2 = set(text2[i:i + n] for i in range(len(text2) - n + 1))
            if not set1 or not set2:
                return 0.0
            intersection = set1 & set2
            return len(intersection) / len(set1 | set2)

        bigram_overlap = ngram_overlap(original, reconstructed, 2)
        trigram_overlap = ngram_overlap(original, reconstructed, 3)

        # Vocabulary overlap
        vocab_orig = set(original)
        vocab_recon = set(reconstructed)
        if vocab_orig:
            vocab_overlap = len(vocab_orig & vocab_recon) / len(vocab_orig | vocab_recon)
        else:
            vocab_overlap = 0.0

        length_ratio = len(reconstructed) / max(1, len(original))

        return {
            "char_accuracy": char_accuracy,
            "bigram_overlap": bigram_overlap,
            "trigram_overlap": trigram_overlap,
            "vocab_overlap": vocab_overlap,
            "length_ratio": length_ratio,
        }

    def measure_compression(
        self,
        genome: str,
        original_data: str | bytes,
        model_size_bytes: int | None = None,
    ) -> dict:
        """Measure compression efficiency.

        Returns:
            Dict with genome_ratio, model_ratio, and total_ratio.
        """
        if isinstance(original_data, str):
            data_bytes = len(original_data.encode("utf-8"))
        else:
            data_bytes = len(original_data)

        genome_bytes = len(genome)  # base-4 chars, 2 bits each

        return {
            "data_size_bytes": data_bytes,
            "genome_size_chars": len(genome),
            "genome_size_bytes": genome_bytes,
            "genome_ratio": genome_bytes / max(1, data_bytes),
            "model_size_bytes": model_size_bytes,
            "model_ratio": (
                model_size_bytes / max(1, data_bytes)
                if model_size_bytes else None
            ),
        }

    def _build_dense_weights(self, brain: TextBrain) -> np.ndarray:
        """Build dense NxN weight matrix from brain's sparse edge list."""
        N = brain.N
        W = np.zeros((N, N), dtype=np.float32)
        w_eff = brain.w_slow + brain.w_fast
        np.add.at(W, (brain.src, brain.dst), w_eff)
        return W
