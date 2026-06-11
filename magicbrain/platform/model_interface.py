"""
Model Interface - Base abstractions for all models in the platform.

Provides unified interface for heterogeneous neural network models.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


class OutputType(Enum):
    """Types of model outputs."""
    SPIKES = "spikes"           # Temporal spike trains (SNN)
    DENSE = "dense"             # Dense vectors/tensors (DNN)
    ATTENTION = "attention"     # Attention weights (Transformers)
    FEATURES = "features"       # Feature maps (CNN)
    HIDDEN = "hidden"           # Hidden states (RNN/LSTM)
    LOGITS = "logits"          # Classification logits
    EMBEDDINGS = "embeddings"   # Learned embeddings
    PROBABILITY = "probability" # Probability distributions
    SEQUENCE = "sequence"       # Sequential outputs
    GRAPH = "graph"            # Graph structures


class ModelType(Enum):
    """Types of models."""
    SNN = "spiking_neural_network"
    DNN = "deep_neural_network"
    CNN = "convolutional_neural_network"
    RNN = "recurrent_neural_network"
    TRANSFORMER = "transformer"
    GNN = "graph_neural_network"
    RL_AGENT = "reinforcement_learning_agent"
    HYBRID = "hybrid_model"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetadata:
    """Metadata for a model."""
    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    output_type: OutputType = OutputType.DENSE
    parameters_count: int = 0
    tags: List[str] = field(default_factory=list)
    author: str = ""
    framework: str = "magicbrain"

    # Dependencies
    parent_models: List[str] = field(default_factory=list)
    required_models: List[str] = field(default_factory=list)

    # Performance metrics
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    accuracy: Optional[float] = None

    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelState:
    """State of a model at a point in time."""
    timestamp: datetime = field(default_factory=datetime.now)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    hidden_states: Optional[np.ndarray] = None
    memory: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class ModelInterface(ABC):
    """
    Base interface for all models in the platform.

    All models must implement this interface to be compatible
    with the platform's orchestration and communication systems.
    """

    def __init__(self, metadata: Optional[ModelMetadata] = None):
        """
        Initialize model.

        Args:
            metadata: Model metadata
        """
        self.metadata = metadata or ModelMetadata(
            model_id="unknown",
            model_type=ModelType.DNN,
            version="0.1.0"
        )
        self._state = ModelState()

    @abstractmethod
    def forward(self, input: Any, **kwargs) -> Any:
        """
        Forward pass through the model.

        Args:
            input: Input data (type depends on model)
            **kwargs: Additional parameters

        Returns:
            Model output
        """
        pass

    async def async_forward(self, input: Any, **kwargs) -> Any:
        """
        Async forward pass (optional).

        Default implementation runs sync forward in executor.
        Override for true async models (e.g., API-based models).

        Args:
            input: Input data
            **kwargs: Additional parameters

        Returns:
            Model output (awaitable)
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.forward(input, **kwargs))

    @abstractmethod
    def get_output_type(self) -> OutputType:
        """
        Get the type of output this model produces.

        Returns:
            Output type enum
        """
        pass

    def get_state(self) -> ModelState:
        """
        Get current model state.

        Returns:
            Model state snapshot
        """
        return self._state

    def set_state(self, state: ModelState):
        """
        Set model state.

        Args:
            state: Model state to restore
        """
        self._state = state

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        return self.metadata

    def update_metadata(self, **kwargs):
        """
        Update model metadata fields.

        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

    def reset(self):
        """Reset model to initial state."""
        self._state = ModelState()

    def get_parameters_count(self) -> int:
        """
        Get number of trainable parameters.

        Returns:
            Parameter count
        """
        return self.metadata.parameters_count

    def summary(self) -> str:
        """
        Get model summary string.

        Returns:
            Human-readable summary
        """
        return f"""
Model: {self.metadata.model_id}
Type: {self.metadata.model_type.value}
Version: {self.metadata.version}
Output Type: {self.get_output_type().value}
Parameters: {self.get_parameters_count():,}
Input Shape: {self.metadata.input_shape}
Output Shape: {self.metadata.output_shape}
Description: {self.metadata.description}
""".strip()


class StatefulModel(ModelInterface):
    """
    Base class for models with internal state (RNN, SNN, etc).
    """

    @abstractmethod
    def step(self, input: Any, **kwargs) -> Any:
        """
        Single timestep forward pass (for temporal models).

        Args:
            input: Input for this timestep
            **kwargs: Additional parameters

        Returns:
            Output for this timestep
        """
        pass

    @abstractmethod
    def get_hidden_state(self) -> Any:
        """
        Get current hidden state.

        Returns:
            Hidden state tensor/array
        """
        pass

    @abstractmethod
    def set_hidden_state(self, hidden: Any):
        """
        Set hidden state.

        Args:
            hidden: Hidden state to restore
        """
        pass


class EnsembleModel(ModelInterface):
    """
    Base class for ensemble models (multiple sub-models).
    """

    def __init__(self, models: List[ModelInterface], metadata: Optional[ModelMetadata] = None):
        """
        Initialize ensemble.

        Args:
            models: List of sub-models
            metadata: Ensemble metadata
        """
        super().__init__(metadata)
        self.models = models

        # Update metadata
        self.metadata.model_type = ModelType.ENSEMBLE
        self.metadata.parameters_count = sum(m.get_parameters_count() for m in models)

    def get_models(self) -> List[ModelInterface]:
        """
        Get sub-models in the ensemble.

        Returns:
            List of models
        """
        return self.models

    @abstractmethod
    def aggregate(self, outputs: List[Any]) -> Any:
        """
        Aggregate outputs from sub-models.

        Args:
            outputs: List of outputs from sub-models

        Returns:
            Aggregated output
        """
        pass


class HybridModel(ModelInterface):
    """
    Base class for hybrid models (combination of different model types).
    """

    def __init__(self,
                 components: Dict[str, ModelInterface],
                 metadata: Optional[ModelMetadata] = None):
        """
        Initialize hybrid model.

        Args:
            components: Dictionary of model components
            metadata: Hybrid model metadata
        """
        super().__init__(metadata)
        self.components = components

        # Update metadata
        self.metadata.model_type = ModelType.HYBRID
        self.metadata.parameters_count = sum(
            m.get_parameters_count() for m in components.values()
        )

    def get_component(self, name: str) -> Optional[ModelInterface]:
        """
        Get a specific component by name.

        Args:
            name: Component name

        Returns:
            Model component or None
        """
        return self.components.get(name)

    def get_components(self) -> Dict[str, ModelInterface]:
        """
        Get all components.

        Returns:
            Dictionary of components
        """
        return self.components


# Type hints for model inputs/outputs
ModelInput = Union[np.ndarray, Dict[str, Any], List[Any]]
ModelOutput = Union[np.ndarray, Dict[str, Any], List[Any]]
