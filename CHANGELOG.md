# Changelog

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
