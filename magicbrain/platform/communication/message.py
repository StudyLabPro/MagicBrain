"""
Message classes for inter-model communication.

Provides structured messages for passing data between models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
import numpy as np


class MessageType(Enum):
    """Types of inter-model messages."""
    DATA = "data"                   # Regular data message
    CONTROL = "control"             # Control signal (start, stop, reset)
    STATE = "state"                 # State synchronization
    GRADIENT = "gradient"           # Gradient for backprop
    REWARD = "reward"               # Reward signal (RL)
    ATTENTION = "attention"         # Attention weights
    QUERY = "query"                 # Request for data
    RESPONSE = "response"           # Response to query
    ERROR = "error"                 # Error notification
    HEARTBEAT = "heartbeat"         # Health check


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """
    Inter-model message.

    Carries data and metadata between models in the platform.
    """
    # Core fields
    source: str                              # Source model ID
    target: str                              # Target model ID
    data: Any                                # Message payload

    # Message metadata
    message_type: MessageType = MessageType.DATA
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None

    # Optional fields
    topic: Optional[str] = None              # Pub/sub topic
    reply_to: Optional[str] = None           # For request-response
    correlation_id: Optional[str] = None     # For tracking related messages

    # Data type hints
    data_type: Optional[str] = None          # Type of data (e.g., "spikes", "embeddings")
    data_shape: Optional[tuple] = None       # Shape if numpy array

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Routing
    routing_key: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Generate message ID if not provided
        if self.message_id is None:
            self.message_id = f"{self.source}_{self.target}_{id(self)}"

        # Auto-detect data type and shape for numpy arrays
        if isinstance(self.data, np.ndarray):
            self.data_type = "numpy"
            self.data_shape = self.data.shape

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "source": self.source,
            "target": self.target,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "topic": self.topic,
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "data_type": self.data_type,
            "data_shape": self.data_shape,
            "routing_key": self.routing_key,
            "metadata": self.metadata,
            # Note: data field excluded for serialization complexity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], message_data: Any = None) -> Message:
        """
        Create message from dictionary.

        Args:
            data: Dictionary with message fields
            message_data: Actual data payload (not serialized)

        Returns:
            Message instance
        """
        return cls(
            source=data["source"],
            target=data["target"],
            data=message_data,
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data.get("message_id"),
            topic=data.get("topic"),
            reply_to=data.get("reply_to"),
            correlation_id=data.get("correlation_id"),
            data_type=data.get("data_type"),
            data_shape=tuple(data["data_shape"]) if data.get("data_shape") else None,
            routing_key=data.get("routing_key"),
            metadata=data.get("metadata", {}),
        )

    def is_control_message(self) -> bool:
        """Check if this is a control message."""
        return self.message_type == MessageType.CONTROL

    def is_data_message(self) -> bool:
        """Check if this is a data message."""
        return self.message_type == MessageType.DATA

    def is_error(self) -> bool:
        """Check if this is an error message."""
        return self.message_type == MessageType.ERROR

    def __repr__(self) -> str:
        """String representation."""
        data_repr = f"{type(self.data).__name__}"
        if isinstance(self.data, np.ndarray):
            data_repr = f"ndarray{self.data.shape}"

        return (
            f"Message(id={self.message_id}, "
            f"{self.source}->{self.target}, "
            f"type={self.message_type.value}, "
            f"data={data_repr})"
        )


@dataclass
class ControlMessage(Message):
    """Control message for model orchestration."""

    def __init__(self, source: str, target: str, command: str, **kwargs):
        """
        Initialize control message.

        Args:
            source: Source model ID
            target: Target model ID
            command: Control command (start, stop, reset, pause, resume)
            **kwargs: Additional message fields
        """
        super().__init__(
            source=source,
            target=target,
            data={"command": command},
            message_type=MessageType.CONTROL,
            **kwargs
        )

    @property
    def command(self) -> str:
        """Get control command."""
        return self.data["command"]


@dataclass
class ErrorMessage(Message):
    """Error message."""

    def __init__(self, source: str, target: str, error: Exception, **kwargs):
        """
        Initialize error message.

        Args:
            source: Source model ID
            target: Target model ID
            error: Exception that occurred
            **kwargs: Additional message fields
        """
        super().__init__(
            source=source,
            target=target,
            data={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            message_type=MessageType.ERROR,
            priority=MessagePriority.HIGH,
            **kwargs
        )

    @property
    def error_type(self) -> str:
        """Get error type."""
        return self.data["error_type"]

    @property
    def error_message(self) -> str:
        """Get error message."""
        return self.data["error_message"]
