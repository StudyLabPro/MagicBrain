"""Communication layer for inter-model messaging."""
from .message import Message, MessageType, MessagePriority
from .message_bus import MessageBus, Topic, get_global_bus
from .converters import (
    TypeConverter,
    ConverterRegistry,
    SpikesToDenseConverter,
    DenseToSpikesConverter,
    get_global_registry,
)

__all__ = [
    "Message",
    "MessageType",
    "MessagePriority",
    "MessageBus",
    "Topic",
    "TypeConverter",
    "ConverterRegistry",
    "SpikesToDenseConverter",
    "DenseToSpikesConverter",
    "get_global_bus",
    "get_global_registry",
]
