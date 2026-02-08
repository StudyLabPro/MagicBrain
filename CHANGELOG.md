# Changelog

## [0.6.0] - 2026-02-08 - HYBRID EDITION 🚀

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
- `PROJECT_COMPLETE.md` - Complete project summary
- `PLATFORM_VISION.md` - Vision and roadmap
- `PHASE1_COMPLETION.md`, `PHASE2_SUMMARY.md`, `PHASE3_COMPLETE.md`
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
