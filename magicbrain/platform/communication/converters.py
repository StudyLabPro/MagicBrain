"""
Type converters for transforming data between different model output types.

Enables communication between heterogeneous models (SNN, DNN, Transformers, etc).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from ..model_interface import OutputType


class TypeConverter(ABC):
    """
    Base class for type converters.

    Converts between different model output types.
    """

    def __init__(self, source_type: OutputType, target_type: OutputType):
        """
        Initialize converter.

        Args:
            source_type: Source output type
            target_type: Target output type
        """
        self.source_type = source_type
        self.target_type = target_type

    @abstractmethod
    def convert(self, data: Any, **kwargs) -> Any:
        """
        Convert data from source type to target type.

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Converted data
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.source_type.value} -> {self.target_type.value})"


class SpikesToDenseConverter(TypeConverter):
    """
    Convert spike trains to dense vectors.

    Methods:
    - rate: Firing rate over time window
    - sum: Sum of spikes
    - last: Last spike value
    - weighted_sum: Weighted sum with temporal kernel
    """

    def __init__(self, method: str = "rate", time_window: Optional[int] = None):
        """
        Initialize spike-to-dense converter.

        Args:
            method: Conversion method (rate, sum, last, weighted_sum)
            time_window: Time window for rate calculation (if method='rate')
        """
        super().__init__(OutputType.SPIKES, OutputType.DENSE)
        self.method = method
        self.time_window = time_window

    def convert(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Convert spike trains to dense vector.

        Args:
            data: Spike train array (T, N) or (N,)
            **kwargs: Additional parameters

        Returns:
            Dense vector (N,)
        """
        if self.method == "rate":
            # Firing rate over time window
            if data.ndim == 1:
                return data  # Already single timestep
            window = self.time_window or data.shape[0]
            return np.mean(data[-window:], axis=0)

        elif self.method == "sum":
            # Sum of spikes
            if data.ndim == 1:
                return data
            return np.sum(data, axis=0)

        elif self.method == "last":
            # Last spike values
            if data.ndim == 1:
                return data
            return data[-1]

        elif self.method == "weighted_sum":
            # Weighted sum with exponential kernel
            if data.ndim == 1:
                return data
            T = data.shape[0]
            weights = np.exp(-np.arange(T)[::-1] / (T / 3))
            weights = weights / weights.sum()
            return np.sum(data * weights[:, None], axis=0)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class DenseToSpikesConverter(TypeConverter):
    """
    Convert dense vectors to spike trains.

    Methods:
    - rate: Poisson spike generation based on rates
    - threshold: Threshold-based spiking
    - latency: Rate-to-latency encoding
    """

    def __init__(self, method: str = "rate", duration: int = 10):
        """
        Initialize dense-to-spikes converter.

        Args:
            method: Conversion method (rate, threshold, latency)
            duration: Duration of spike train in timesteps
        """
        super().__init__(OutputType.DENSE, OutputType.SPIKES)
        self.method = method
        self.duration = duration

    def convert(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Convert dense vector to spike trains.

        Args:
            data: Dense vector (N,)
            **kwargs: Additional parameters

        Returns:
            Spike train array (T, N)
        """
        # Ensure positive rates
        rates = np.maximum(data, 0)
        rates = rates / (rates.max() + 1e-8)  # Normalize to [0, 1]

        if self.method == "rate":
            # Poisson spike generation
            spikes = np.random.rand(self.duration, len(rates)) < rates
            return spikes.astype(np.float32)

        elif self.method == "threshold":
            # Threshold-based: spike if rate > threshold
            threshold = kwargs.get("threshold", 0.5)
            spikes = np.zeros((self.duration, len(rates)), dtype=np.float32)
            spikes[0] = (rates > threshold).astype(np.float32)
            return spikes

        elif self.method == "latency":
            # Rate-to-latency encoding: higher rate = earlier spike
            spikes = np.zeros((self.duration, len(rates)), dtype=np.float32)
            for i, rate in enumerate(rates):
                if rate > 0:
                    # Spike time inversely proportional to rate
                    spike_time = int((1 - rate) * (self.duration - 1))
                    spikes[spike_time, i] = 1.0
            return spikes

        else:
            raise ValueError(f"Unknown method: {self.method}")


class DenseToEmbeddingsConverter(TypeConverter):
    """Convert dense vectors to embeddings (typically just pass-through or normalization)."""

    def __init__(self, normalize: bool = True):
        """
        Initialize converter.

        Args:
            normalize: Whether to L2-normalize embeddings
        """
        super().__init__(OutputType.DENSE, OutputType.EMBEDDINGS)
        self.normalize = normalize

    def convert(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Convert dense to embeddings.

        Args:
            data: Dense vector/matrix
            **kwargs: Additional parameters

        Returns:
            Embeddings (potentially normalized)
        """
        if self.normalize:
            norm = np.linalg.norm(data, axis=-1, keepdims=True)
            return data / (norm + 1e-8)
        return data


class EmbeddingsToDenseConverter(TypeConverter):
    """Convert embeddings to dense (typically pass-through)."""

    def __init__(self):
        super().__init__(OutputType.EMBEDDINGS, OutputType.DENSE)

    def convert(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Pass-through conversion."""
        return data


class LogitsToProbabilityConverter(TypeConverter):
    """Convert logits to probability distribution."""

    def __init__(self, temperature: float = 1.0):
        """
        Initialize converter.

        Args:
            temperature: Softmax temperature
        """
        super().__init__(OutputType.LOGITS, OutputType.PROBABILITY)
        self.temperature = temperature

    def convert(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Convert logits to probabilities via softmax.

        Args:
            data: Logits
            **kwargs: Additional parameters

        Returns:
            Probability distribution
        """
        logits = data / self.temperature
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


class IdentityConverter(TypeConverter):
    """Identity converter (no transformation)."""

    def __init__(self, output_type: OutputType):
        """
        Initialize identity converter.

        Args:
            output_type: Output type (same for source and target)
        """
        super().__init__(output_type, output_type)

    def convert(self, data: Any, **kwargs) -> Any:
        """Pass through data unchanged."""
        return data


class ConverterRegistry:
    """
    Registry for managing type converters.

    Automatically selects appropriate converter for source -> target type conversion.
    """

    def __init__(self):
        """Initialize registry."""
        self._converters: Dict[Tuple[OutputType, OutputType], TypeConverter] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default converters."""
        # Spikes <-> Dense
        self.register(SpikesToDenseConverter())
        self.register(DenseToSpikesConverter())

        # Dense <-> Embeddings
        self.register(DenseToEmbeddingsConverter())
        self.register(EmbeddingsToDenseConverter())

        # Logits -> Probability
        self.register(LogitsToProbabilityConverter())

        # Identity converters
        for output_type in OutputType:
            self.register(IdentityConverter(output_type))

    def register(self, converter: TypeConverter):
        """
        Register a converter.

        Args:
            converter: Type converter instance
        """
        key = (converter.source_type, converter.target_type)
        self._converters[key] = converter

    def get_converter(
        self,
        source_type: OutputType,
        target_type: OutputType
    ) -> Optional[TypeConverter]:
        """
        Get converter for source -> target conversion.

        Args:
            source_type: Source output type
            target_type: Target output type

        Returns:
            Converter instance or None if not found
        """
        key = (source_type, target_type)
        return self._converters.get(key)

    def convert(
        self,
        data: Any,
        source_type: OutputType,
        target_type: OutputType,
        **kwargs
    ) -> Any:
        """
        Convert data from source type to target type.

        Args:
            data: Input data
            source_type: Source output type
            target_type: Target output type
            **kwargs: Additional parameters for converter

        Returns:
            Converted data

        Raises:
            ValueError: If no converter found for this type pair
        """
        converter = self.get_converter(source_type, target_type)
        if converter is None:
            raise ValueError(
                f"No converter found for {source_type.value} -> {target_type.value}"
            )
        return converter.convert(data, **kwargs)

    def has_converter(self, source_type: OutputType, target_type: OutputType) -> bool:
        """
        Check if converter exists for source -> target.

        Args:
            source_type: Source output type
            target_type: Target output type

        Returns:
            True if converter exists
        """
        return (source_type, target_type) in self._converters

    def list_converters(self) -> list[str]:
        """
        List all registered converters.

        Returns:
            List of converter descriptions
        """
        return [
            f"{src.value} -> {tgt.value}"
            for (src, tgt) in self._converters.keys()
        ]


# Global converter registry
_global_registry = ConverterRegistry()


def get_global_registry() -> ConverterRegistry:
    """
    Get the global converter registry.

    Returns:
        Global registry instance
    """
    return _global_registry


def convert(
    data: Any,
    source_type: OutputType,
    target_type: OutputType,
    **kwargs
) -> Any:
    """
    Convert data using global registry.

    Args:
        data: Input data
        source_type: Source output type
        target_type: Target output type
        **kwargs: Additional parameters

    Returns:
        Converted data
    """
    return _global_registry.convert(data, source_type, target_type, **kwargs)
