"""
Backend interface for MagicBrain computations.
Allows switching between NumPy, JAX, PyTorch implementations.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np


class Backend(ABC):
    """Abstract backend interface for numerical computations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (numpy, jax, torch)."""
        pass

    @abstractmethod
    def zeros(self, shape, dtype=None) -> Any:
        """Create array of zeros."""
        pass

    @abstractmethod
    def ones(self, shape, dtype=None) -> Any:
        """Create array of ones."""
        pass

    @abstractmethod
    def array(self, data, dtype=None) -> Any:
        """Create array from data."""
        pass

    @abstractmethod
    def zeros_like(self, arr) -> Any:
        """Create zeros with same shape as arr."""
        pass

    @abstractmethod
    def sum(self, arr, axis=None) -> Any:
        """Sum array elements."""
        pass

    @abstractmethod
    def clip(self, arr, min_val, max_val) -> Any:
        """Clip array values."""
        pass

    @abstractmethod
    def exp(self, arr) -> Any:
        """Element-wise exponential."""
        pass

    @abstractmethod
    def sqrt(self, arr) -> Any:
        """Element-wise square root."""
        pass

    @abstractmethod
    def abs(self, arr) -> Any:
        """Element-wise absolute value."""
        pass

    @abstractmethod
    def argpartition(self, arr, k) -> Any:
        """Partial sort (for top-k)."""
        pass

    @abstractmethod
    def scatter_add(self, base, indices, values) -> Any:
        """Scatter add operation for sparse updates."""
        pass

    @abstractmethod
    def to_numpy(self, arr) -> np.ndarray:
        """Convert to numpy array."""
        pass

    @abstractmethod
    def random_normal(self, shape, mean=0.0, std=1.0, seed=None) -> Any:
        """Generate random normal values."""
        pass

    @abstractmethod
    def random_uniform(self, shape, low=0.0, high=1.0, seed=None) -> Any:
        """Generate random uniform values."""
        pass


def get_backend(backend_name: str = "numpy") -> Backend:
    """
    Get backend by name.

    Args:
        backend_name: One of 'numpy', 'jax', 'torch'

    Returns:
        Backend instance
    """
    if backend_name == "numpy":
        from .numpy_backend import NumpyBackend
        return NumpyBackend()
    elif backend_name == "jax":
        try:
            from .jax_backend import JAXBackend
            return JAXBackend()
        except ImportError:
            raise ImportError(
                "JAX backend requested but JAX not installed. "
                "Install with: pip install jax jaxlib"
            )
    elif backend_name == "torch":
        try:
            from .torch_backend import TorchBackend
            return TorchBackend()
        except ImportError:
            raise ImportError(
                "PyTorch backend requested but PyTorch not installed. "
                "Install with: pip install torch"
            )
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def auto_select_backend() -> Backend:
    """
    Automatically select best available backend.
    Priority: JAX (if GPU) > NumPy
    """
    # Try JAX first if available
    try:
        from .jax_backend import JAXBackend
        backend = JAXBackend()
        if backend.has_gpu():
            print(f"Auto-selected JAX backend with GPU")
            return backend
    except ImportError:
        pass

    # Fallback to NumPy
    from .numpy_backend import NumpyBackend
    print(f"Auto-selected NumPy backend")
    return NumpyBackend()
