# 🧠 MagicBrain Platform

<div align="center">

![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Универсальная платформа для гетерогенных и гибридных нейросетевых архитектур**

[Документация](./PROJECT_COMPLETE.md) • [Changelog](./CHANGELOG.md) • [Releases](https://github.com/AndrewHakmi/MagicBrain/releases) • [Issues](https://github.com/AndrewHakmi/MagicBrain/issues)

</div>

---

## 🚀 О проекте

**MagicBrain Platform** - это универсальная платформа для создания, управления и оркестрации **гетерогенных и гибридных нейросетевых архитектур**. Платформа позволяет seamless интегрировать разные типы нейронных сетей (SNN, DNN, Transformers, CNN, RNN) в единые системы.

### ✨ Ключевые возможности

- 🧩 **5 типов моделей**: SNN, DNN, Transformer, CNN, RNN
- 🔄 **Automatic type conversion**: Spikes ↔ Dense ↔ Embeddings ↔ Logits
- 🎼 **Multi-model orchestration**: Sequential, Parallel, Pipeline стратегии
- 🧬 **Unlimited hybrid combinations**: создавайте любые комбинации моделей
- 🏗️ **Compositional API**: HybridBuilder с fluent interface
- ⚡ **Spiking Attention**: attention mechanism в spike domain
- 🧪 **57 тестов** с покрытием >90%
- 📦 **Production-ready** infrastructure

---

## 📦 Установка

```bash
# Базовая установка (только SNN)
pip install magicbrain

# С поддержкой PyTorch моделей (DNN, CNN, RNN)
pip install magicbrain[torch]

# С поддержкой Transformers
pip install magicbrain[transformers]

# Полная платформа (все типы моделей)
pip install magicbrain[platform]

# Все возможности (включая JAX, визуализацию, dev tools)
pip install magicbrain[all]
```

### Разработка

```bash
git clone https://github.com/AndrewHakmi/MagicBrain.git
cd MagicBrain
pip install -e ".[dev]"
pytest
```

---

## 🎯 Quick Start

### 1. Простой Hybrid Pipeline

```python
from magicbrain.platform import ModelOrchestrator, ExecutionStrategy
from magicbrain.models.snn import SNNTextModel
from magicbrain.models.dnn import DNNModel
import torch.nn as nn

# Создаем модели
snn = SNNTextModel(genome="30121033102301230112332100123", vocab_size=50)
dnn = DNNModel(nn.Linear(384, 10))  # PyTorch модель

# Оркестрация
orch = ModelOrchestrator()
orch.add_model(snn, model_id="snn_encoder")
orch.add_model(dnn, model_id="dnn_decoder")
orch.connect("snn_encoder", "dnn_decoder")

# Выполнение
result = orch.execute(input_data, strategy=ExecutionStrategy.SEQUENTIAL)
```

### 2. Compositional Hybrid API

```python
from magicbrain.hybrid import HybridBuilder

# Fluent interface для построения гибридных архитектур
hybrid = (HybridBuilder()
    .add("encoder", snn_model)
    .add("transformer", bert_model)
    .add("decoder", dnn_model)
    .connect("encoder", "transformer")
    .connect("transformer", "decoder")
    .build("complex_hybrid"))

output = hybrid.forward(input_data)
print(hybrid.visualize_graph())
```

### 3. Vision + Language Hybrid

```python
from magicbrain.hybrid import HybridBuilder
from magicbrain.models.cnn import CNNModel
from torchvision.models import resnet50

# Multi-modal система
vision_language = (HybridBuilder()
    .add("cnn", CNNModel(resnet50()))        # Image features
    .add("snn", snn_encoder)                 # Spike encoding
    .add("transformer", bert)                # Language understanding
    .connect("cnn", "snn")
    .connect("snn", "transformer")
    .build("vision_language_system"))

result = vision_language.forward(image)
```

### 4. Spiking Neural Network

```python
from magicbrain import TextBrain
from magicbrain.tasks.text_task import train_loop

# Создание SNN с ДНК-кодированием
genome = "30121033102301230112332100123"
brain = TextBrain(genome, vocab_size=50)

# Обучение
train_loop(brain, text="Your training text", steps=10000)

# Генерация
from magicbrain.sampling import sample_text
generated = sample_text(brain, seed="Hello", n_tokens=100, temperature=0.8)
```

---

## 🏗️ Архитектура

```
╔═══════════════════════════════════════════════════════════╗
║           MagicBrain Platform v0.6.0                      ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  ┌─────────────────┐    ┌──────────────────┐            ║
║  │  Model Types    │    │   Infrastructure │            ║
║  ├─────────────────┤    ├──────────────────┤            ║
║  │ • SNN           │    │ • Registry       │            ║
║  │ • DNN           │    │ • Orchestrator   │            ║
║  │ • Transformers  │    │ • MessageBus     │            ║
║  │ • CNN           │    │ • Converters     │            ║
║  │ • RNN/LSTM      │    │ • Model Zoo      │            ║
║  └─────────────────┘    └──────────────────┘            ║
║                                                           ║
║  ┌─────────────────────────────────────────┐            ║
║  │        Hybrid Architectures             │            ║
║  ├─────────────────────────────────────────┤            ║
║  │ • SNN + DNN                              │            ║
║  │ • SNN + Transformer                      │            ║
║  │ • CNN + SNN                              │            ║
║  │ • Custom (HybridBuilder)                 │            ║
║  │ • Spiking Attention                      │            ║
║  └─────────────────────────────────────────┘            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

### Основные компоненты

**Phase 1: Platform Foundation**
- `ModelInterface` - универсальная абстракция для всех моделей
- `ModelRegistry` - версионирование, metadata, зависимости
- `Communication Layer` - MessageBus + TypeConverters
- `ModelOrchestrator` - multi-model execution
- `Model Zoo` - управление pretrained моделями

**Phase 2: Multi-Model Support**
- `DNNModel` - PyTorch DNN adapter
- `TransformerModel` - Hugging Face integration
- `CNNModel` - torchvision models
- `RNNModel` - LSTM/GRU с stateful execution

**Phase 3: Hybrid Architectures**
- `HybridArchitecture` - базовый класс для гибридов
- `SNNDNNHybrid`, `SNNTransformerHybrid`, `CNNSNNHybrid` - готовые комбинации
- `SpikingAttention` - attention в spike domain
- `HybridBuilder` - compositional API с fluent interface

---

## 💡 Use Cases

### 🧪 Neuromorphic Computing
Создание энергоэффективных систем с биологически правдоподобными спайковыми сетями.

### 🤖 Multi-Modal AI
Интеграция vision (CNN) + language (Transformer) + temporal processing (SNN/RNN).

### 🧠 Brain-Inspired Architectures
Моделирование когнитивных процессов через гибридные системы.

### 📚 Adaptive Learning Systems
Использование в StudyNinja для моделирования процессов обучения студентов.

### 🔬 AI Research
Исследование гетерогенных архитектур и neuromorphic algorithms.

---

## 📚 Документация

- **[PROJECT_COMPLETE.md](./PROJECT_COMPLETE.md)** - Полный отчет о проекте с примерами
- **[PLATFORM_VISION.md](./PLATFORM_VISION.md)** - Видение платформы и roadmap
- **[CHANGELOG.md](./CHANGELOG.md)** - История версий
- **[CLAUDE.md](./CLAUDE.md)** - Руководство для разработчиков
- **[magicbrain/platform/README.md](./magicbrain/platform/README.md)** - Детали платформы
- **[examples/](./examples/)** - Рабочие примеры кода

---

## 🧪 Тестирование

```bash
# Запуск всех тестов
pytest

# С покрытием
pytest --cov=magicbrain --cov-report=html

# Specific test
pytest tests/platform/test_orchestrator.py -v
```

**Текущее покрытие**: 57 тестов, >90% coverage, 100% pass rate

---

## 📊 Статистика проекта

| Метрика | Значение |
|---------|----------|
| **Версия** | 0.6.0 (Hybrid Edition) |
| **Файлов** | 46 |
| **Строк кода** | ~9,000 |
| **Тестов** | 57 (100% passed) |
| **Model types** | 5 базовых |
| **Hybrid combinations** | Unlimited |
| **Фаз завершено** | 3/3 (100%) |
| **Python** | 3.9+ |

---

## 🤝 Contributing

Мы приветствуем контрибуции! Пожалуйста:

1. Fork репозиторий
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Development Setup

```bash
git clone https://github.com/AndrewHakmi/MagicBrain.git
cd MagicBrain
pip install -e ".[dev]"
pytest
```

### Code Style

- Используем Ruff для форматирования и линтинга
- Type hints обязательны
- Docstrings в Google style
- 100% test coverage для новых features

---

## 🌟 Key Innovations

### 1. Universal Model Interface
Первая платформа с единым интерфейсом для SNN, DNN, Transformers, CNN, RNN.

### 2. Automatic Type Conversion
Прозрачная конвертация между spike trains, dense vectors, embeddings.

### 3. Hybrid Architecture System
Compositional API для построения arbitrary гибридных архитектур.

### 4. Spiking Attention
Attention mechanism в spike domain - foundation для neuromorphic transformers.

### 5. Multi-Model Orchestration
Seamless integration разных model types в единые pipelines.

---

## 🔮 Roadmap

### v0.7.0 (Q2 2026)
- [ ] Advanced orchestration (Mixture of Experts)
- [ ] Dynamic routing между моделями
- [ ] Feedback loops в гибридных архитектурах
- [ ] Performance optimizations

### v0.8.0 (Q3 2026)
- [ ] Joint training для гибридных систем
- [ ] Knowledge distillation
- [ ] Meta-learning capabilities
- [ ] Extended model zoo

### v1.0.0 (Q4 2026)
- [ ] Model serving infrastructure
- [ ] Distributed inference
- [ ] Monitoring dashboard
- [ ] Production deployment tools

---

## 📄 License

Этот проект является частью экосистемы [StudyNinja-Eco](https://github.com/XTeam-Pro/StudyNinja-Eco).

---

## 🙏 Acknowledgments

**Powered by**:
- PyTorch ecosystem
- Hugging Face Transformers
- NumPy & JAX
- Python 3.12

**Inspired by**:
- Biological neural networks
- Neuromorphic computing
- Multi-modal AI research
- Compositional architectures

---

## 📧 Contact

- **Repository**: [github.com/AndrewHakmi/MagicBrain](https://github.com/AndrewHakmi/MagicBrain)
- **Issues**: [GitHub Issues](https://github.com/AndrewHakmi/MagicBrain/issues)
- **Ecosystem**: [StudyNinja-Eco](https://github.com/XTeam-Pro/StudyNinja-Eco)

---

<div align="center">

**🧠 MagicBrain Platform - From single models to model ecosystems! 🌐**

Made with ❤️ for the AI & Neuromorphic Computing community

</div>
