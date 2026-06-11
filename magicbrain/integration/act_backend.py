"""ACT-compensated computation backend for MagicBrain.

Provides numerically stable operations via Balansis Absolute Compensation Theory.
Falls back to standard numpy if Balansis is not installed.
"""
from __future__ import annotations

import numpy as np


class ACTBackend:
    """ACT-compensated computation backend.

    Falls back to numpy if Balansis is unavailable.
    """

    def __init__(self):
        self._available = False
        self._compensated_array_add = None
        self._compensated_array_multiply = None
        self._compensated_dot_product = None
        self._compensated_outer_product = None
        self._compensated_softmax = None
        try:
            from balansis.numpy_integration import (
                compensated_array_add,
                compensated_array_multiply,
                compensated_dot_product,
                compensated_outer_product,
                compensated_softmax,
            )
            self._compensated_array_add = compensated_array_add
            self._compensated_array_multiply = compensated_array_multiply
            self._compensated_dot_product = compensated_dot_product
            self._compensated_outer_product = compensated_outer_product
            self._compensated_softmax = compensated_softmax
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Whether Balansis ACT backend is available."""
        return self._available

    def weight_update(self, w: np.ndarray, delta: np.ndarray, lr: float) -> np.ndarray:
        """ACT-compensated weight update: w + lr * delta.

        Args:
            w: Current weight array.
            delta: Weight delta array.
            lr: Learning rate scalar.

        Returns:
            Updated weight array.
        """
        if self._available:
            lr_arr = np.full_like(delta, lr, dtype=np.float64)
            scaled = self._compensated_array_multiply(
                delta.astype(np.float64), lr_arr
            )
            return self._compensated_array_add(
                w.astype(np.float64), scaled
            ).astype(w.dtype)
        return w + lr * delta

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """ACT-compensated stable softmax.

        Args:
            logits: Input logit array.

        Returns:
            Probability array summing to ~1.0.
        """
        if self._available:
            result = self._compensated_softmax(logits.astype(np.float64))
            return result.astype(logits.dtype)
        x = logits - np.max(logits)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-9)

    def outer_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ACT-compensated outer product.

        Args:
            a: First input vector.
            b: Second input vector.

        Returns:
            2D outer product array.
        """
        if self._available:
            result = self._compensated_outer_product(
                a.astype(np.float64), b.astype(np.float64)
            )
            return result.astype(np.float32)
        return np.outer(a, b)

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        """ACT-compensated dot product.

        Args:
            a: First input vector.
            b: Second input vector.

        Returns:
            Dot product scalar.
        """
        if self._available:
            return self._compensated_dot_product(
                a.astype(np.float64), b.astype(np.float64)
            )
        return float(np.dot(a, b))
