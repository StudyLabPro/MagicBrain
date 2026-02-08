"""
JAX backend implementation for GPU acceleration and JIT compilation.
"""
from __future__ import annotations
from typing import Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

from .backend_interface import Backend


class JAXBackend(Backend):
    """JAX-based backend with GPU support and JIT compilation."""

    def __init__(self):
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available. Install with: pip install jax jaxlib")

    @property
    def name(self) -> str:
        return "jax"

    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            return len(jax.devices('gpu')) > 0
        except:
            return False

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = jnp.float32
        return jnp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = jnp.float32
        return jnp.ones(shape, dtype=dtype)

    def array(self, data, dtype=None):
        if dtype is None:
            dtype = jnp.float32
        return jnp.array(data, dtype=dtype)

    def zeros_like(self, arr):
        return jnp.zeros_like(arr)

    def sum(self, arr, axis=None):
        return jnp.sum(arr, axis=axis)

    def clip(self, arr, min_val, max_val):
        return jnp.clip(arr, min_val, max_val)

    def exp(self, arr):
        return jnp.exp(arr)

    def sqrt(self, arr):
        return jnp.sqrt(arr)

    def abs(self, arr):
        return jnp.abs(arr)

    def argpartition(self, arr, k):
        """JAX doesn't have argpartition, use argsort instead."""
        # For top-k, we can use argsort (less efficient but works)
        return jnp.argsort(arr)

    def scatter_add(self, base, indices, values):
        """JAX scatter add using index_add."""
        return base.at[indices].add(values)

    def to_numpy(self, arr) -> np.ndarray:
        """Convert JAX array to numpy."""
        return np.asarray(arr)

    def random_normal(self, shape, mean=0.0, std=1.0, seed=None):
        if seed is None:
            seed = 0
        key = jax.random.PRNGKey(seed)
        return mean + std * jax.random.normal(key, shape, dtype=jnp.float32)

    def random_uniform(self, shape, low=0.0, high=1.0, seed=None):
        if seed is None:
            seed = 0
        key = jax.random.PRNGKey(seed)
        return jax.random.uniform(key, shape, minval=low, maxval=high, dtype=jnp.float32)
