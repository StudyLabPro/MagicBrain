# MagicBrain v0.7.0 — NeuroGenesis Edition

**Release Date**: 2026-02-13
**Status**: Production Ready

## Summary

Neurogenomic memory system — data is not stored directly, but encoded as a compact generative genome that can reproduce information through dynamic neural structure activation.

Pipeline: `Dataset -> Compile -> Genome -> Develop -> Train -> Reconstruct`

## Highlights

- **NeuroGenesis Engine**: 7 new modules (compiler, CPPN, development, pattern memory, attractor dynamics, energy, reconstruction) with 93 tests
- **Pattern Memory**: Hopfield-style associative memory with Storkey learning rule, covariance rule, and annealed recall
- **CPPN**: spatial coordinates -> synaptic weights via compositional pattern-producing networks
- **Attractor Dynamics**: continuous energy-minimization convergence to stored patterns
- **True Async Parallel**: ~4x speedup for multi-model execution via asyncio.gather()
- **KnowledgeBase Integration**: HTTP client for KnowledgeBaseAI with graceful degradation
- **Model Metadata**: load_model() now returns full metadata (genome, vocab, step, timestamp)
- **REST API**: FastAPI microservice integrated as api/ module

## Stats

- 263 tests (237 passed, 26 skipped for optional deps)
- ~12,000 LOC
- 100% backward compatibility with v0.6.0

## Install

```bash
pip install -e .           # Core (NumPy only)
pip install -e ".[all]"    # All optional dependencies
```

## CLI

```bash
magicbrain neurogenesis compile --text data.txt --strategy statistical
magicbrain neurogenesis run --text data.txt --cppn --steps 10000
magicbrain neurogenesis benchmark --text data.txt --steps 5000 --trials 3
magicbrain neurogenesis attractors --model model.npz --probes 500
```

## Full Changelog

See [CHANGELOG.md](./CHANGELOG.md) for details.
