# MagicBrain Platform

![Version](https://img.shields.io/badge/version-0.7.1-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Tests](https://img.shields.io/badge/tests-237%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25+-brightgreen.svg)
![Lean4](https://img.shields.io/badge/Lean4-21%20theorems-blueviolet.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![MAGIC Level](https://img.shields.io/badge/MAGIC-Level%202%20MetaBrain-orange.svg)

**Универсальная платформа для гетерогенных и гибридных нейросетевых архитектур**

[Документация](./PROJECT_COMPLETE.md) | [Changelog](./CHANGELOG.md) | [Formal Proofs](./formal/README.md) | [API Docs](./api/README_API.md)

---

## О проекте

**MagicBrain** — исследовательский сервис уровня **MAGIC Level 2 (MetaBrain)** в экосистеме [StudyNinja-Eco](https://github.com/XTeam-Pro/StudyNinja-Eco). Реализует живую AGI-память через биологически правдоподобные спайковые нейронные сети (SNN), геномное кодирование, нейрогенез и цифровые двойники студентов.

### Ключевые компоненты

- **TextBrain (SNN)** — спайковая нейронная сеть с разреженной активацией (~5%), дофамин-модулируемым обучением по Хеббу и гомеостатической адаптацией порогов
- **Genome System** — DNA-кодирование гиперпараметров: base-4 строка (24-72+ символа) детерминированно задаёт всю архитектуру
- **NeuroGenesis Engine** — 9 модулей: датасет → геном → 3D-морфогенез → обучение → реконструкция
- **Digital Twins** — NeuralDigitalTwin: освоение тем, адаптация сложности, прогноз когнитивного состояния студента
- **Platform Framework** — оркестрация 5 типов моделей (SNN, DNN, Transformer, CNN, RNN), 7 стратегий выполнения
- **Formal Verification** — 21 теорема в Lean4 для Hopfield-динамики (0 sorry, 0 ошибок)

---

## Установка

```bash
pip install magicbrain               # только SNN + NumPy
pip install magicbrain[torch]        # + PyTorch (DNN, CNN, RNN, Hybrid)
pip install magicbrain[transformers] # + Hugging Face
pip install magicbrain[balansis]     # + ACT-компенсированная арифметика
pip install magicbrain[platform]     # полная платформа
pip install magicbrain[all]          # всё, включая JAX и dev-инструменты
```

### Разработка

```bash
git clone https://github.com/XTeam-Pro/MagicBrain.git
cd MagicBrain
pip install -e ".[dev]"
pytest
```

### REST API

```bash
uvicorn api.app.api.main:app --host 0.0.0.0 --port 8004
# Swagger: http://localhost:8004/docs
```

---

## Быстрый старт

### SNN через геном

```python
from magicbrain import TextBrain
from magicbrain.tasks.text_task import train_loop
from magicbrain.sampling import sample_text

genome = "30121033102301230112332100123"
brain = TextBrain(genome, vocab_size=50)
train_loop(brain, text="Your training text", steps=10000)
generated = sample_text(brain, seed="Hello", n_tokens=100, temperature=0.8)
```

### Hybrid Pipeline

```python
from magicbrain.platform import ModelOrchestrator, ExecutionStrategy
from magicbrain.models.snn import SNNTextModel
from magicbrain.models.dnn import DNNModel
import torch.nn as nn

snn = SNNTextModel(genome="30121033102301230112332100123", vocab_size=50)
dnn = DNNModel(nn.Linear(384, 10))

orch = ModelOrchestrator()
orch.add_model(snn, model_id="encoder")
orch.add_model(dnn, model_id="decoder")
orch.connect("encoder", "decoder")
result = orch.execute(input_data, strategy=ExecutionStrategy.SEQUENTIAL)
```

### HybridBuilder (fluent API)

```python
from magicbrain.hybrid import HybridBuilder

hybrid = (HybridBuilder()
    .add("encoder", snn_model)
    .add("transformer", bert_model)
    .add("decoder", dnn_model)
    .connect("encoder", "transformer")
    .connect("transformer", "decoder")
    .build("complex_hybrid"))

output = hybrid.forward(input_data)
```

### NeuroGenesis

```python
from magicbrain.neurogenesis.compiler import GenomeCompiler
from magicbrain.neurogenesis.pipeline import NeurogenesisPipeline

compiler = GenomeCompiler(strategy="hybrid")
genome = compiler.compile(dataset="path/to/text.txt")

pipeline = NeurogenesisPipeline()
result = pipeline.run(dataset, genome_v2=True)
# result: обученный мозг + метрики реконструкции
```

### Digital Twins

```python
from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin

twin = NeuralDigitalTwin(student_id="student_42", learning_style="visual")
twin.learn_topic(topic_id="algebra_linear", steps=200, difficulty=0.6)
mastery = twin.get_mastery(topic_id="algebra_linear")
state = twin.get_cognitive_state()  # attention, confusion, fatigue
```

---

## Структура модулей

### Ядро SNN

| Модуль | Назначение |
|--------|-----------|
| `brain.py` | TextBrain: sparse top-k, dual weights (slow/fast), dopamine, axonal delays |
| `genome.py` | Геномное кодирование, `decode_genome()` |
| `sampling.py` | Генерация текста (temperature, top-k, top-p) |
| `graph.py` | 3D граф нейронов с аксональными задержками (1-5 шагов) |
| `io.py` | Сохранение/загрузка моделей (.npz) |

### NeuroGenesis Engine (`neurogenesis/`)

| Модуль | Назначение |
|--------|-----------|
| `compiler.py` | GenomeCompiler: датасет → геном (hash / statistical / hybrid) |
| `energy.py` | Hopfield energy: E(s) = -1/2 s'Ws - theta's + lambda||s||_1 |
| `attractor_dynamics.py` | Динамика к аттракторам, поиск бассейнов притяжения |
| `cppn.py` | CPPN: координаты → веса (8 базисных функций: sin, cos, gaussian, ...) |
| `development.py` | 3D морфогенез + синаптогенез + созревание ткани |
| `pattern_memory.py` | Ассоциативная память Хопфилда (Storkey rule, ~0.14N паттернов) |
| `reconstruction.py` | 4 режима: autoregressive, attractor, cue-based, generation |
| `genome_v2.py` | Расширенный формат 72+: Topology + CPPN + Attractor + Patterns |
| `pipeline.py` | E2E пайплайн: датасет → компиляция → развитие → обучение → оценка |

### Platform Framework (`platform/`)

| Компонент | Назначение |
|-----------|-----------|
| `ModelInterface` | Универсальная абстракция: SNN, DNN, CNN, RNN, Transformer, Hybrid, Ensemble |
| `ModelOrchestrator` | 7 стратегий: Sequential, Parallel, Pipeline, Hierarchical, Feedback, Cascaded, MoE |
| `ModelRegistry` | Версионирование, метаданные, dependency tracking |
| `MessageBus` | Маршрутизация между моделями с автоматической конвертацией типов |
| TypeConverters | Spikes <-> Dense <-> Embeddings <-> Logits |

### Digital Twins (`integration/`)

| Модуль | Назначение |
|--------|-----------|
| `neural_digital_twin.py` | NeuralDigitalTwin: геном из student_id, 4 стиля, forgetting dynamics |
| `redis_twin_store.py` | Redis-персистентность состояния двойников |
| `knowledgebase_client.py` | HTTP-клиент для синхронизации с KnowledgeBaseAI |
| `act_backend.py` | Интеграция с Balansis (ACT-компенсированная арифметика весов) |

### Дополнительные модули

| Каталог | Содержание |
|---------|-----------|
| `evolution/` | SimpleGA (tournament selection), GenomeMutator (6 операторов), FitnessEvaluator |
| `diagnostics/` | LiveMonitor, NeuronalDynamics, PlasticityTracker, SynapticMetrics |
| `learning_rules/` | STDP, STDPBrain |
| `backends/` | NumPy (default), JAX (optional) |
| `hybrid/` | HybridArchitecture, HybridBuilder, SNN+DNN/Transformer/CNN, SpikingAttention |
| `architectures/` | HierarchicalBrain (multi-layer SNN) |
| `training/` | Coordinator, Checkpointing, DataPartitioner, Worker |
| `zoo/` | Model Zoo — управление предобученными моделями |

---

## REST API

FastAPI-микросервис, порт `8004` в экосистеме.

| Группа | Основные эндпоинты |
|--------|-------------------|
| `/api/v1/models` | CRUD моделей, запуск обучения, генерация, forward pass |
| `/api/v1/training` | Фоновые задачи обучения, статус и результат |
| `/api/v1/inference` | Forward pass, text sampling |
| `/api/v1/diagnostics` | Метрики мозга, текущее состояние активации |
| `/api/v1/evolution` | Запуск генетической оптимизации генома |
| `/api/v1/twins` | CRUD двойников; learn, mastery, cognitive-state, interaction, predict |
| `/api/v1/auto-evolution` | Автоматическая оптимизация без ручного задания fitness |

---

## Формальная верификация (Lean4)

**Инструментарий**: Lean 4 v4.28.0 + Mathlib v4.28.0

| Файл | Содержание |
|------|-----------|
| `DeltaE.lean` | Delta_E <= 0 при обновлении одного нейрона |
| `Convergence.lean` | Конечная сходимость: <= 2^n шагов (принцип Дирихле) |
| `Hopfield.lean` | Базовые определения: sigma, spin, updateCoord, globalEnergy |
| `SumLemmas.lean` | Вспомогательные леммы для скалярных произведений |
| `Glue.lean` | globalEnergy_nonincreasing_updateCoord |

**21 теорема, 0 sorry, 0 ошибок, 0 axioms.**

```bash
cd formal && lake build
```

---

## Тестирование

```bash
pytest                                    # все тесты
pytest --cov=magicbrain --cov-report=html # с покрытием
pytest tests/neurogenesis/ -v             # NeuroGenesis (93 теста)
pytest tests/platform/ -v                 # Platform
pytest tests/integration/ -v              # Integration
```

| Категория | Файлов |
|-----------|--------|
| Core SNN | 6 |
| NeuroGenesis | 8 (93 теста) |
| Platform | 5 |
| Hybrid | 2 |
| Evolution, Diagnostics, STDP | 3 |
| Training | 3 |
| Integration | 3 |

**34 тестовых файла, 237+ passed, 90%+ coverage.**

---

## Геномное кодирование

Base-4 строка 24+ символов. Каждая позиция детерминированно задаёт гиперпараметр:

| Позиция | Параметр | Диапазон |
|---------|----------|---------|
| 0-1 | N (нейронов) | 256-1216 |
| 2 | K (связность) | 8-20 |
| 4 | lr | 5e-4 - 2e-3 |
| 5 | k_active | 4-7% от N |
| 15 | p_inhib | 10-25% |
| 16 | dopamine_gain | 0.8-2.0 |
| 20 | prune_every | 800-1400 шагов |

Геном по умолчанию: `"30121033102301230112332100123"`

---

## Статистика

| Метрика | Значение |
|---------|----------|
| Версия | 0.7.1 (Lean4 Formal Verification Edition) |
| Python | 3.9, 3.10, 3.11, 3.12 |
| Лицензия | Apache 2.0 |
| Python-модулей | 88 (magicbrain/) + 16 (api/) |
| Тестовых файлов | 34 |
| Passed тестов | 237+ |
| Покрытие | 90%+ |
| Lean4 теорем | 21 (0 sorry, 0 errors) |
| Типов моделей | 5 |
| Стратегий оркестрации | 7 |
| API эндпоинтов | 30+ |
| MAGIC Level | 2 (MetaBrain) |

---

## Roadmap

### v0.7.2
- [ ] Lean4 верификация STDP-правил обучения
- [ ] Покрытие тестами до 95%+

### v0.8.0 (Q2 2026)
- [ ] Joint training гибридных архитектур (SNN <-> DNN градиенты)
- [ ] Knowledge distillation из больших моделей в SNN
- [ ] Расширенный Model Zoo с предобученными геномами

### v1.0.0 (Q4 2026)
- [ ] Production inference server
- [ ] Distributed training
- [ ] Stable public API с гарантиями совместимости

---

## Место в MAGIC Ecosystem

```
Level 4 (MetaKnowledge): KnowledgeBaseAI  <- синхронизация освоения тем
Level 3 (MetaAgent):     xteam-agents     <- когнитивные модели студентов
Level 2 (MetaBrain):     MagicBrain       <- живая AGI-память (этот проект)
Level 1 (MetaBalansis):  Balansis         <- ACT-компенсированная арифметика
Applied:                 StudyNinja-API   <- circuit breaker -> /health
```

---

## Лицензия

**Apache License 2.0** — см. [LICENSE](./LICENSE).

Коммерческое использование: [COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md).

---

## Contributing

1. Fork репозиторий
2. Создайте feature branch
3. Добавьте тесты (coverage >= 90% для новых модулей)
4. Откройте Pull Request

Code style: Ruff, type hints обязательны, docstrings в Google style.

---

**MagicBrain — биологически правдоподобная память для AGI-систем**

Часть [StudyNinja-Eco](https://github.com/XTeam-Pro/StudyNinja-Eco) | MAGIC Level 2: MetaBrain
