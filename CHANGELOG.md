# Changelog

## [0.7.1] — 2026-02-18 — Lean4 Formal Verification

### Added

#### Lean4 Formal Proofs (`formal/MagicBrainFormal/`)
Energy-based convergence theorem for SNN Hopfield dynamics — **21 theorems, 0 sorry, 0 errors**.

- **`DeltaE.lean`** — proves that Δ*E* ≤ 0 for every synchronous SNN weight update:
  - `dotProduct_comm`, `dotProduct_add` — vector arithmetic lemmas
  - `quadraticForm_expansion` — W symmetry → Q(v) = vᵀWv
  - `globalEnergyVec_eq_B` — global energy via bias term
  - `deltaE_le_zero` — **main theorem**: energy is non-increasing (convergence guarantee)
  - `network_converges` — corollary: network reaches attractor state
  - `convergence_bound` — energy bounded below by eigenvalues
- All `simpa` replaced with `simp` or `rw` (linter clean, 0 warnings)

#### CI/CD (StudyNinja-Eco)
- `qa-gates.yml` lean-formal matrix job: builds MagicBrain and Balansis proofs in parallel

### Changed
- Remote URL migrated to `git@github.com:XTeam-Pro/MagicBrain.git`
- `development` branch created from `main`, set as tracking branch

---

## [0.7.0] - 2026-02-13 - NEUROGENESIS EDITION

**MAJOR RELEASE**: Neurogenomic memory system — data encoded as compact generative genomes that reproduce information through dynamic neural structure activation.

### NeuroGenesis Engine
- **GenomeCompiler**: compile datasets into deterministic base-4 genomes (hash, statistical, hybrid strategies)
- **DevelopmentOperator**: three-stage morphogenesis (positions, CPPN synaptogenesis, maturation)
- **CPPN**: Compositional Pattern-Producing Networks for spatially-patterned weight generation
- **PatternMemory**: Hopfield-style associative memory with Storkey learning rule (~0.14N capacity)
- **AttractorDynamics**: continuous neural dynamics converging to attractor states via energy minimization
- **ReconstructionOperator**: autoregressive, attractor-based, and cue-based data reconstruction
- **Hopfield energy function**: E(s) = -1/2 sWs - theta*s + lambda||s||_1 with gradient computation
- **Extended genome v2**: backward-compatible 72+ position format (CPPN, attractor, pattern params)
- **CLI commands**: `magicbrain neurogenesis compile|run|benchmark|attractors`
- **93 new tests** for the NeuroGenesis subsystem

### Pattern Memory Improvements
- Covariance learning rule for better capacity scaling
- Annealed recall with temperature schedule for more reliable convergence
- Improved pattern retrieval fidelity

### Production Polish (v0.6.1 scope)
- **True async parallel execution**: `asyncio.gather()` for concurrent model execution (~4x speedup)
- **KnowledgeBase API integration**: real HTTP calls to KnowledgeBaseAI with graceful degradation
- **Model metadata loading**: `load_model()` returns full metadata (genome, vocab, step, timestamp)
- **Comprehensive hybrid tests**: 25 new tests for hybrid architectures and builder

### Other
- Integrated MagicBrainAPI as `api/` module (FastAPI REST microservice)
- Updated CLAUDE.md with full v0.7.0 architecture documentation

### Statistics
- **Tests**: 263 total (237 passed, 26 skipped for optional deps)
- **NeuroGenesis tests**: 93
- **Code**: ~12,000 LOC
- **100% backward compatibility** with v0.6.0

---

## [0.6.0] - 2026-02-08 - HYBRID EDITION

**MAJOR RELEASE**: Universal Platform for Heterogeneous and Hybrid Neural Architectures!

### 🎉 Platform Complete (Phases 1-3)

#### Phase 3: Hybrid Architectures
- **HybridArchitecture** base class for multi-model combinations
- **SNNDNNHybrid**: SNN encoder + DNN decoder pipelines
- **SNNTransformerHybrid**: SNN + Transformer integration
- **CNNSNNHybrid**: CNN feature extraction + SNN processing
- **SpikingAttention**: Attention mechanism in spike domain
- **HybridBuilder**: Compositional API with fluent interface
- Architecture templates and patterns

#### Phase 2: Multi-Model Support
- **DNNModel**: PyTorch DNN adapter (device management, layer extraction)
- **TransformerModel**: Hugging Face integration (BERT, GPT, etc)
- **CNNModel**: torchvision models (ResNet, VGG, EfficientNet)
- **RNNModel**: LSTM/GRU with stateful execution

#### Phase 1: Platform Foundation
- **ModelInterface**: Universal abstraction for all model types
- **ModelRegistry**: Versioning, metadata, dependency tracking
- **Communication Layer**: MessageBus + TypeConverters
- **ModelOrchestrator**: Multi-model execution (Sequential, Parallel, Pipeline)
- **SNNTextModel**: Platform adapter for TextBrain
- **Model Zoo**: Pretrained model management
- **57 comprehensive tests** (100% passed, >90% coverage)

### 📊 Statistics
- **Files**: 46 created
- **Code**: ~9,000 LOC
- **Tests**: 57 (100% passed)
- **Model types**: 5 (SNN, DNN, Transformer, CNN, RNN)
- **Hybrid combinations**: Unlimited
- **Documentation**: ~2,500+ lines

### 🚀 Key Features
- ✅ Universal model interface for heterogeneous architectures
- ✅ Automatic type conversion (Spikes ↔ Dense ↔ Embeddings)
- ✅ Multi-model orchestration with multiple strategies
- ✅ Hybrid architectures with compositional API
- ✅ Production-ready infrastructure

### 📦 Dependencies
**Core**: numpy>=1.24.0

**Platform (optional)**:
- torch>=2.0.0, torchvision>=0.15.0 (DNN, CNN, RNN)
- transformers>=4.30.0 (Transformers)

**Install**: `pip install magicbrain[platform]` for full platform

### 📚 Documentation
- `README.md` - Current project overview
- `PLATFORM_VISION.md` - Vision and roadmap
- `docs/FORMAL_VERIFICATION.md` - formal verification notes
- `api/README_API.md` - API guide
- `formal/README.md` - Lean4 formalization guide
- `magicbrain/platform/README.md` - Platform guide
- Working examples in `examples/platform/`

---

## [0.2.0] - 2026-02-08

### 🚀 Major Features

#### Backend System
- **Multi-backend support**: Added pluggable backend architecture
  - `NumpyBackend`: Default CPU backend (existing functionality)
  - `JAXBackend`: GPU-accelerated backend with JIT compilation
  - Auto-selection based on available hardware
  - Easy to extend with PyTorch or custom backends

#### Diagnostics & Monitoring
- **LiveMonitor**: Real-time training metrics tracking
  - Loss, dopamine, firing rate, weight statistics
  - JSON export for analysis
  - Automatic logging at configurable intervals
- **SpikeRaster**: Neural activity recording and analysis
  - Spike train recording
  - Firing rate computation
  - Synchrony and burstiness metrics
- **SynapticAnalyzer**: Weight and connectivity analysis
  - E/I balance tracking
  - Sparsity metrics
  - Connectivity statistics
- **PlasticityTracker**: Structural plasticity monitoring
  - Pruning/rewiring event tracking
  - Consolidation statistics

#### Genome Evolution
- **GenomeMutator**: Controlled genome mutations
  - Point mutations
  - Insertions/deletions
  - Adaptive mutation rates
  - Crossover operations (single-point and uniform)
- **FitnessEvaluator**: Multi-objective fitness functions
  - Loss-based fitness
  - Convergence speed
  - Training stability
  - Robustness to damage
- **SimpleGA**: Genetic algorithm for genome optimization
  - Tournament selection
  - Elitism
  - Hall of fame tracking
  - Multi-generation evolution

### 🎯 CLI Enhancements
- New command: `magicbrain evolve` - Evolve genomes using GA
- New command: `magicbrain monitor` - Train with live monitoring
- Enhanced `train` command with verbose flag
- Metrics export to JSON

### 🧪 Testing
- 26 test cases added (100% pass rate)
- Backend parity tests
- Diagnostics system tests
- Evolution system tests
- Increased coverage to ~80%

### 📦 Dependencies
- Added optional dependencies:
  - `[jax]` for GPU acceleration
  - `[viz]` for visualization (future)
  - `[dev]` for development tools
  - `[all]` for everything

### 🐛 Bug Fixes
- Fixed train_loop to support verbose flag
- Added diagnostic methods to TextBrain (avg_theta, firing_rate, etc.)

### 📝 Documentation
- Comprehensive CLAUDE.md updates
- Added CHANGELOG.md
- Inline documentation for all new modules

---

## [0.1.0] - 2026-02-08

### Initial Release
- TextBrain with DNA-encoded architecture
- Dual-weight plasticity (w_slow/w_fast)
- Neuromodulation (dopamine-based learning)
- Structural plasticity (pruning/rewiring)
- CLI tools: train, sample, repair
- Basic test suite
