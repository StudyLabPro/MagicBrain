"""NeuroGenesis task pipeline.

Full end-to-end workflow:
  1. Compile: dataset -> genome
  2. Develop: genome -> neural tissue (optionally with CPPN)
  3. Train: standard text training loop
  4. Reconstruct: generate text from trained model
  5. Evaluate: measure fidelity and compression

Also includes benchmark mode comparing multiple strategies.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from ..brain import TextBrain
from ..neurogenesis.compiler import GenomeCompiler, analyze_dataset
from ..neurogenesis.development import DevelopmentOperator
from ..neurogenesis.reconstruction import ReconstructionOperator
from ..neurogenesis.attractor_dynamics import AttractorDynamics
from ..neurogenesis.energy import EnergyFunction
from .text_task import build_vocab, train_loop


class NeurogenesisResult(NamedTuple):
    """Result of a full neurogenesis pipeline run."""
    genome: str
    strategy: str
    final_loss: float
    reconstruction: str
    fidelity: dict
    compression: dict
    training_time: float
    attractor_count: int
    brain: TextBrain


def run_neurogenesis_pipeline(
    text: str,
    strategy: str = "statistical",
    genome_length: int = 28,
    steps: int = 10000,
    use_cppn: bool = False,
    reconstruct_length: int = 500,
    n_attractor_probes: int = 200,
    verbose: bool = True,
) -> NeurogenesisResult:
    """Run the full NeuroGenesis pipeline.

    Args:
        text: Input dataset (text).
        strategy: Genome compilation strategy ('hash', 'statistical', 'hybrid').
        genome_length: Length of generated genome.
        steps: Training steps.
        use_cppn: Use CPPN for weight generation.
        reconstruct_length: Length of reconstructed text.
        n_attractor_probes: Number of probes for attractor search.
        verbose: Print progress.

    Returns:
        NeurogenesisResult with all metrics.
    """
    # Step 1: Compile
    if verbose:
        print(f"[1/5] Compiling genome (strategy={strategy})...")

    compiler = GenomeCompiler()
    metadata = compiler.compile_with_metadata(text, strategy, genome_length)
    genome = metadata["genome"]

    if verbose:
        stats = metadata["stats"]
        print(f"  Data: {stats['size']} chars, vocab={stats['vocab_size']}, "
              f"entropy={stats['entropy']:.2f}")
        print(f"  Genome: {genome} (len={len(genome)})")

    # Step 2: Build brain
    if verbose:
        print("[2/5] Developing neural tissue...")

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    if use_cppn:
        dev = DevelopmentOperator()
        brain, tissue = dev.develop_and_build_brain(genome, vocab_size, use_cppn=True)
        if verbose:
            print(f"  CPPN-developed: N={brain.N}, K={brain.K}, "
                  f"edges={brain.src.shape[0]}")
    else:
        brain = TextBrain(genome, vocab_size)
        if verbose:
            print(f"  Standard init: N={brain.N}, K={brain.K}, "
                  f"edges={brain.src.shape[0]}")

    # Step 3: Train
    if verbose:
        print(f"[3/5] Training ({steps} steps)...")

    t0 = time.time()
    final_loss = train_loop(
        brain, text, stoi, steps=steps,
        print_every=max(1000, steps // 5),
        verbose=verbose,
    )
    training_time = time.time() - t0

    if verbose:
        print(f"  Final loss: {final_loss:.4f} ({training_time:.1f}s)")

    # Step 4: Reconstruct
    if verbose:
        print("[4/5] Reconstructing...")

    recon = ReconstructionOperator()

    # Autoregressive reconstruction
    seed_text = text[:min(10, len(text))]
    result = recon.reconstruct_autoregressive(
        brain, stoi, itos, seed=seed_text, length=reconstruct_length,
    )

    if verbose:
        preview = result.text[:80].replace("\n", " ")
        print(f"  Reconstructed: \"{preview}...\"")

    # Step 5: Evaluate
    if verbose:
        print("[5/5] Evaluating...")

    fidelity = recon.measure_fidelity(text[:reconstruct_length], result.text)
    compression = recon.measure_compression(genome, text)

    # Attractor analysis (lightweight)
    attractor_count = 0
    if n_attractor_probes > 0:
        try:
            dynamics = AttractorDynamics(max_iterations=50)
            W_dense = recon._build_dense_weights(brain)
            attractors = dynamics.find_attractors(
                brain.N, W_dense, brain.theta,
                n_probes=n_attractor_probes,
            )
            attractor_count = len(attractors)
        except Exception:
            attractor_count = -1

    if verbose:
        print(f"  Fidelity: char_acc={fidelity['char_accuracy']:.3f}, "
              f"bigram={fidelity['bigram_overlap']:.3f}, "
              f"trigram={fidelity['trigram_overlap']:.3f}")
        print(f"  Compression: genome={compression['genome_ratio']:.4f}")
        print(f"  Attractors found: {attractor_count}")

    return NeurogenesisResult(
        genome=genome,
        strategy=strategy,
        final_loss=final_loss,
        reconstruction=result.text,
        fidelity=fidelity,
        compression=compression,
        training_time=training_time,
        attractor_count=attractor_count,
        brain=brain,
    )


def run_benchmark(
    text: str,
    strategies: list[str] | None = None,
    steps: int = 5000,
    trials: int = 3,
    verbose: bool = True,
) -> dict:
    """Run benchmark comparing multiple genome compilation strategies.

    Also includes random and default genome baselines.

    Args:
        text: Input dataset.
        strategies: List of strategies to test.
        steps: Training steps per trial.
        trials: Number of trials per strategy (random gets different seeds).
        verbose: Print progress.

    Returns:
        Dict mapping strategy -> list of NeurogenesisResult.
    """
    if strategies is None:
        strategies = ["hash", "statistical", "hybrid"]

    results: dict[str, list] = {}

    # Test each compilation strategy
    for strategy in strategies:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy}")
            print(f"{'='*60}")

        results[strategy] = []
        for trial in range(trials):
            if verbose:
                print(f"\n--- Trial {trial+1}/{trials} ---")
            r = run_neurogenesis_pipeline(
                text, strategy=strategy, steps=steps,
                n_attractor_probes=100, verbose=verbose,
            )
            results[strategy].append(r)

    # Default genome baseline
    if verbose:
        print(f"\n{'='*60}")
        print("Strategy: default")
        print(f"{'='*60}")

    results["default"] = []
    default_genome = "30121033102301230112332100123"
    stoi, itos = build_vocab(text)

    for trial in range(trials):
        if verbose:
            print(f"\n--- Trial {trial+1}/{trials} ---")

        brain = TextBrain(default_genome, len(stoi))
        t0 = time.time()
        final_loss = train_loop(
            brain, text, stoi, steps=steps,
            print_every=max(1000, steps // 5), verbose=verbose,
        )
        training_time = time.time() - t0

        recon = ReconstructionOperator()
        seed_text = text[:min(10, len(text))]
        rec = recon.reconstruct_autoregressive(
            brain, stoi, itos, seed=seed_text, length=500,
        )
        fidelity = recon.measure_fidelity(text[:500], rec.text)
        compression = recon.measure_compression(default_genome, text)

        results["default"].append(NeurogenesisResult(
            genome=default_genome, strategy="default",
            final_loss=final_loss, reconstruction=rec.text,
            fidelity=fidelity, compression=compression,
            training_time=training_time, attractor_count=0, brain=brain,
        ))

    # Random genome baseline
    if verbose:
        print(f"\n{'='*60}")
        print("Strategy: random")
        print(f"{'='*60}")

    results["random"] = []
    rng = np.random.default_rng(42)
    for trial in range(trials):
        if verbose:
            print(f"\n--- Trial {trial+1}/{trials} ---")

        random_genome = "".join(str(rng.integers(0, 4)) for _ in range(28))
        brain = TextBrain(random_genome, len(stoi))
        t0 = time.time()
        final_loss = train_loop(
            brain, text, stoi, steps=steps,
            print_every=max(1000, steps // 5), verbose=verbose,
        )
        training_time = time.time() - t0

        recon = ReconstructionOperator()
        seed_text = text[:min(10, len(text))]
        rec = recon.reconstruct_autoregressive(
            brain, stoi, itos, seed=seed_text, length=500,
        )
        fidelity = recon.measure_fidelity(text[:500], rec.text)
        compression = recon.measure_compression(random_genome, text)

        results["random"].append(NeurogenesisResult(
            genome=random_genome, strategy="random",
            final_loss=final_loss, reconstruction=rec.text,
            fidelity=fidelity, compression=compression,
            training_time=training_time, attractor_count=0, brain=brain,
        ))

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for strategy, runs in results.items():
            losses = [r.final_loss for r in runs]
            fids = [r.fidelity["bigram_overlap"] for r in runs]
            times = [r.training_time for r in runs]
            print(f"  {strategy:12s}: loss={np.mean(losses):.4f}+/-{np.std(losses):.4f}  "
                  f"bigram={np.mean(fids):.3f}  time={np.mean(times):.1f}s")

    return results
