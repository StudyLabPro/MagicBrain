# MagicBrain Platform

**Универсальная платформа для создания, управления и оркестрации гетерогенных нейросетевых архитектур.**

## Обзор

MagicBrain Platform позволяет:
- Объединять модели разных типов (SNN, DNN, Transformers, CNN, RNN)
- Автоматически преобразовывать типы выходов между моделями
- Оркестровать выполнение с различными стратегиями (Sequential, Parallel, Pipeline)
- Управлять версиями моделей и зависимостями
- Организовывать коммуникацию через message bus

## Архитектура

```
┌─────────────────────────────────────────────┐
│         MagicBrain Platform                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐    ┌──────────────┐      │
│  │   Registry   │    │ Orchestrator │      │
│  │              │    │              │      │
│  │  - Models    │    │  - Execute   │      │
│  │  - Versions  │    │  - Strategies│      │
│  │  - Metadata  │    │  - Graph     │      │
│  └──────────────┘    └──────────────┘      │
│                                             │
│  ┌──────────────┐    ┌──────────────┐      │
│  │ MessageBus   │    │  Converters  │      │
│  │              │    │              │      │
│  │  - Topics    │    │  - Spikes    │      │
│  │  - Routes    │    │  - Dense     │      │
│  │  - Pub/Sub   │    │  - Logits    │      │
│  └──────────────┘    └──────────────┘      │
│                                             │
└─────────────────────────────────────────────┘
```

## Основные компоненты

### 1. Model Interface

Базовые абстракции для всех моделей:

```python
from magicbrain.platform import ModelInterface, OutputType

class MyModel(ModelInterface):
    def forward(self, input, **kwargs):
        # Your forward pass
        return output

    def get_output_type(self):
        return OutputType.DENSE
```

**Доступные типы моделей:**
- `ModelInterface` - базовый интерфейс
- `StatefulModel` - для моделей с внутренним состоянием (RNN, SNN)
- `EnsembleModel` - для ensemble моделей
- `HybridModel` - для гибридных архитектур

### 2. Model Registry

Централизованный репозиторий моделей:

```python
from magicbrain.platform import ModelRegistry

registry = ModelRegistry()

# Регистрация модели
registry.register(
    model=my_model,
    model_id="my_model_v1",
    version="1.0.0",
    tags=["production", "snn"],
    dependencies=["base_model"]
)

# Получение модели
model = registry.get("my_model_v1", version="1.0.0")

# Поиск
results = registry.search("snn")

# Список моделей
models = registry.list_models(tags=["production"])
```

**Возможности:**
- Версионирование моделей
- Dependency tracking
- Поиск по тегам и метаданным
- Сохранение/загрузка состояния registry

### 3. Communication Layer

Система сообщений для коммуникации между моделями:

```python
from magicbrain.platform import MessageBus, Message, MessageType

bus = MessageBus()

# Pub/Sub коммуникация
def callback(msg):
    print(f"Received: {msg.data}")

bus.subscribe("model1", "results_topic", callback)

# Отправка сообщения
message = Message(
    source="model2",
    target="model1",
    data=output,
    topic="results_topic"
)
bus.publish(message)

# Direct routing
bus.route("model_a", "model_b", callback)
```

**Type Converters** - автоматическое преобразование типов:

```python
from magicbrain.platform import ConverterRegistry, OutputType

registry = ConverterRegistry()

# Преобразование spikes → dense
dense = registry.convert(
    spike_data,
    source_type=OutputType.SPIKES,
    target_type=OutputType.DENSE
)
```

### 4. Model Orchestrator

Оркестрация multi-model execution:

```python
from magicbrain.platform import ModelOrchestrator, ExecutionStrategy

orch = ModelOrchestrator()

# Добавление моделей
orch.add_model(model1, model_id="m1")
orch.add_model(model2, model_id="m2")
orch.add_model(model3, model_id="m3")

# Соединение моделей
orch.connect("m1", "m2")
orch.connect("m2", "m3")

# Выполнение
result = orch.execute(
    input_data=input_tensor,
    strategy=ExecutionStrategy.SEQUENTIAL,
    entry_model="m1"
)

# Результаты
final_output = result.get_final_output()
m2_output = result.get_output("m2")
print(f"Execution time: {result.execution_time_ms:.2f}ms")
```

**Стратегии выполнения:**
- `SEQUENTIAL` - последовательное выполнение
- `PARALLEL` - параллельное выполнение
- `PIPELINE` - pipeline с этапами
- `HIERARCHICAL` - supervisor-worker (TODO)
- `FEEDBACK` - итеративное взаимодействие (TODO)
- `MIXTURE_OF_EXPERTS` - роутер + эксперты (TODO)

### 5. Model Zoo

Управление pretrained моделями:

```python
from magicbrain.zoo import ZooManager

zoo = ZooManager()

# Добавление модели в zoo
zoo.add_model(
    model_id="snn_shakespeare",
    version="1.0.0",
    model_type="snn",
    description="SNN trained on Shakespeare",
    weights_path=Path("model.npz"),
    author="MagicBrain Team",
    tags=["text", "snn", "shakespeare"],
    metrics={"loss": 1.23, "accuracy": 0.85}
)

# Получение пути к весам
weights_path = zoo.get_weights_path("snn_shakespeare")

# Список моделей
models = zoo.list_models(model_type="snn")
```

## Примеры использования

### Пример 1: Простая Sequential Pipeline

```python
from magicbrain.platform import ModelOrchestrator, ExecutionStrategy
from magicbrain.models.snn import SNNTextModel

# Создание моделей
snn1 = SNNTextModel(
    genome="30121033102301230112332100123",
    vocab_size=50,
    model_id="snn_encoder"
)

snn2 = SNNTextModel(
    genome="30121033102301230112332100123",
    vocab_size=50,
    model_id="snn_decoder"
)

# Оркестрация
orch = ModelOrchestrator()
orch.add_model(snn1)
orch.add_model(snn2)
orch.connect("snn_encoder", "snn_decoder")

# Выполнение
result = orch.execute(
    input_data=token_id,
    strategy=ExecutionStrategy.SEQUENTIAL
)

print(f"Result: {result.get_final_output()}")
print(f"Time: {result.execution_time_ms:.2f}ms")
```

### Пример 2: Parallel Ensemble

```python
from magicbrain.platform import ModelOrchestrator, ExecutionStrategy
from magicbrain.models.snn import SNNTextModel

# Создание ensemble из нескольких SNN
models = [
    SNNTextModel(f"genome_{i}", 50, model_id=f"snn_{i}")
    for i in range(3)
]

# Оркестрация
orch = ModelOrchestrator()
for model in models:
    orch.add_model(model)

# Параллельное выполнение
result = orch.execute(
    input_data=token_id,
    strategy=ExecutionStrategy.PARALLEL
)

# Усреднение выходов
import numpy as np
outputs = [result.get_output(f"snn_{i}") for i in range(3)]
ensemble_output = np.mean(outputs, axis=0)
```

### Пример 3: Type Conversion между SNN и DNN

```python
from magicbrain.platform import (
    ModelOrchestrator,
    ConverterRegistry,
    OutputType
)

# SNN модель с spike output
snn = SNNTextModel(...)  # OutputType.SPIKES

# DNN модель ожидает dense input
class DenseModel(ModelInterface):
    def get_output_type(self):
        return OutputType.DENSE

    def forward(self, input):
        # Expects dense input
        return process(input)

# Оркестрация с автоматической конверсией
orch = ModelOrchestrator()
orch.add_model(snn, "snn")
orch.add_model(dense_model, "dnn")
orch.connect("snn", "dnn")

# Converters автоматически применяются
result = orch.execute(input_data, strategy=ExecutionStrategy.SEQUENTIAL)
```

### Пример 4: Model Registry с версионированием

```python
from magicbrain.platform import ModelRegistry
from magicbrain.models.snn import SNNTextModel

registry = ModelRegistry()

# Регистрация версии 1.0
model_v1 = SNNTextModel(genome="...", vocab_size=50)
registry.register(
    model_v1,
    model_id="text_predictor",
    version="1.0.0",
    tags=["production"]
)

# Регистрация версии 2.0 (улучшенная)
model_v2 = SNNTextModel(genome="improved_genome", vocab_size=50)
registry.register(
    model_v2,
    model_id="text_predictor",
    version="2.0.0",
    tags=["production", "improved"]
)

# Получение latest version
latest = registry.get("text_predictor")  # Returns v2.0.0

# Получение конкретной версии
v1 = registry.get("text_predictor", version="1.0.0")
```

### Пример 5: Message Bus для async коммуникации

```python
from magicbrain.platform import MessageBus, Message

bus = MessageBus()
results = []

# Подписка на результаты
def handle_result(msg):
    results.append(msg.data)
    print(f"Received result: {msg.data}")

bus.subscribe("collector", "results", handle_result)

# Модели публикуют результаты
for i, model in enumerate(models):
    output = model.forward(input)

    msg = Message(
        source=f"model_{i}",
        target="collector",
        data=output,
        topic="results"
    )
    bus.publish(msg)

print(f"Collected {len(results)} results")
```

## Интеграция с существующим кодом

### Оборачивание TextBrain в Platform

```python
from magicbrain.brain import TextBrain
from magicbrain.models.snn import create_from_existing_brain

# Существующий TextBrain
brain = TextBrain(genome="...", vocab_size=50)
# ... training ...

# Оборачивание в platform-compatible модель
platform_model = create_from_existing_brain(
    brain=brain,
    vocab_size=50,
    model_id="trained_brain",
    version="1.0.0"
)

# Теперь можно использовать в registry и orchestrator
registry.register(platform_model)
```

## Best Practices

1. **Всегда указывайте model_id и version** при регистрации моделей
2. **Используйте tags** для категоризации моделей
3. **Определяйте dependencies** для моделей, зависящих от других
4. **Используйте aliases** для удобного доступа к часто используемым моделям
5. **Сохраняйте registry state** для persistence между сессиями
6. **Используйте type converters** вместо ручного преобразования
7. **Проверяйте output_type** моделей перед подключением

## Производительность

- **Type conversion**: ~0.1-1ms для типичных размеров
- **Message routing**: <0.01ms для direct routes
- **Registry lookup**: O(1) для get by ID
- **Orchestration overhead**: ~0.5-2ms для Sequential execution

## Roadmap

### Phase 1 (Completed ✅)
- Model Interface abstraction
- Model Registry
- Communication Protocol
- Basic Orchestrator (Sequential, Parallel)
- SNN Adapter
- Model Zoo structure

### Phase 2 (Upcoming)
- DNN Integration (PyTorch/TensorFlow)
- Transformer Integration (Hugging Face)
- Type converters (SNN ↔ DNN)
- Advanced orchestration patterns

### Phase 3 (Future)
- Hybrid architectures (SNN+DNN, SNN+Transformer)
- Attention mechanisms for SNNs
- Compositional API
- Visual architecture builder

## Contributing

При добавлении новых компонентов:
1. Наследуйте от `ModelInterface` или его подклассов
2. Реализуйте `forward()` и `get_output_type()`
3. Добавьте тесты в `tests/platform/`
4. Обновите документацию

## Примеры кода

Больше примеров в директории `examples/platform/`:
- `basic_usage.py` - Базовое использование
- `ensemble.py` - Ensemble моделей
- `type_conversion.py` - Преобразование типов
- `message_bus.py` - Async коммуникация

## License

Part of MagicBrain project.
