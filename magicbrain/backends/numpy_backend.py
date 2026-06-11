"""
NumPy backend implementation - default CPU backend.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from .backend_interface import Backend


class NumpyBackend(Backend):
    """NumPy-based backend for CPU computations."""

    @property
    def name(self) -> str:
        return "numpy"

    def zeros(self, shape, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = np.float32
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = np.float32
        return np.ones(shape, dtype=dtype)

    def array(self, data, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = np.float32
        return np.array(data, dtype=dtype)

    def zeros_like(self, arr) -> np.ndarray:
        return np.zeros_like(arr)

    def sum(self, arr, axis=None) -> Any:
        return np.sum(arr, axis=axis)

    def clip(self, arr, min_val, max_val) -> np.ndarray:
        return np.clip(arr, min_val, max_val)

    def exp(self, arr) -> np.ndarray:
        return np.exp(arr)

    def sqrt(self, arr) -> np.ndarray:
        return np.sqrt(arr)

    def abs(self, arr) -> np.ndarray:
        return np.abs(arr)

    def argpartition(self, arr, k) -> np.ndarray:
        return np.argpartition(arr, -k)

    def scatter_add(self, base, indices, values) -> np.ndarray:
        """NumPy scatter add via np.add.at."""
        result = base.copy()
        np.add.at(result, indices, values)
        return result

    def to_numpy(self, arr) -> np.ndarray:
        return np.asarray(arr)

    def random_normal(self, shape, mean=0.0, std=1.0, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(mean, std, size=shape).astype(np.float32)

    def random_uniform(self, shape, low=0.0, high=1.0, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high, size=shape).astype(np.float32)
