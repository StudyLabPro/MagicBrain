# Platform Examples

Примеры использования MagicBrain Platform.

## Список примеров

### 1. `basic_usage.py`
Базовое использование platform:
- Создание и регистрация моделей
- Sequential orchestration
- Получение статистики

```bash
python examples/platform/basic_usage.py
```

### 2. `ensemble_example.py`
Ensemble нескольких SNN моделей:
- Parallel execution
- Агрегация выходов
- Diversity metrics

```bash
python examples/platform/ensemble_example.py
```

## Требования

```bash
pip install -e .
```

## Дополнительные примеры

Для более продвинутых примеров см. документацию:
- `magicbrain/platform/README.md`

## Создание своих примеров

Используйте примеры как шаблон для:
- Тестирования новых архитектур
- Экспериментов с orchestration strategies
- Интеграции разных типов моделей
