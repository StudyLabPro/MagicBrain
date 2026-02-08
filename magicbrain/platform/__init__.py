"""MagicBrain Platform - Multi-Model Neural Network Ecosystem."""
from .model_interface import (
    ModelInterface,
    StatefulModel,
    EnsembleModel,
    HybridModel,
    OutputType,
    ModelType,
    ModelMetadata,
    ModelState,
)

from .communication import (
    Message,
    MessageType,
    MessagePriority,
    MessageBus,
    Topic,
    TypeConverter,
    ConverterRegistry,
    get_global_registry as get_global_converter_registry,
)

from .registry import (
    ModelRegistry,
    ModelNotFoundError,
    ModelVersionConflict,
    get_global_registry,
)

from .orchestrator import (
    ModelOrchestrator,
    ExecutionStrategy,
    ExecutionResult,
    OrchestratorError,
    get_global_orchestrator,
)

__all__ = [
    # Model interfaces
    "ModelInterface",
    "StatefulModel",
    "EnsembleModel",
    "HybridModel",
    "OutputType",
    "ModelType",
    "ModelMetadata",
    "ModelState",
    # Communication
    "Message",
    "MessageType",
    "MessagePriority",
    "MessageBus",
    "Topic",
    "TypeConverter",
    "ConverterRegistry",
    "get_global_converter_registry",
    # Registry
    "ModelRegistry",
    "ModelNotFoundError",
    "ModelVersionConflict",
    "get_global_registry",
    # Orchestrator
    "ModelOrchestrator",
    "ExecutionStrategy",
    "ExecutionResult",
    "OrchestratorError",
    "get_global_orchestrator",
]
