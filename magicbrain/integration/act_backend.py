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

    def __init__(self) -> None:
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

    # ------------------------------------------------------------------
    # Original methods (used by the 3 pre-existing integration points)
    # ------------------------------------------------------------------

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
            return self._compensated_array_add(  # type: ignore[return-value]
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
            return result.astype(logits.dtype)  # type: ignore[union-attr]
        x = logits - np.max(logits)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-9)  # type: ignore[return-value]

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
            return result.astype(np.float32)  # type: ignore[union-attr]
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
            return self._compensated_dot_product(  # type: ignore[return-value]
                a.astype(np.float64), b.astype(np.float64)
            )
        return float(np.dot(a, b))

    # ------------------------------------------------------------------
    # Extended methods (Phase 2 integration points)
    # ------------------------------------------------------------------

    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ACT-compensated element-wise subtraction: a - b.

        Uses Kahan two-sum to recover precision lost when a ≈ b
        (near-cancellation), which occurs when delayed signals approach
        the homeostatic threshold in the SNN forward pass.

        Args:
            a: Minuend array.
            b: Subtrahend array.

        Returns:
            Difference array preserving the same dtype as ``a``.
        """
        if self._available:
            return self._compensated_array_add(  # type: ignore[return-value]
                a.astype(np.float64), (-b).astype(np.float64)
            ).astype(a.dtype)
        return a - b

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ACT-compensated element-wise addition: a + b.

        Applies Kahan two-sum correction so that accumulation errors in
        long-running weight arrays (w_slow + w_fast) are recovered.

        Args:
            a: First operand array.
            b: Second operand array.

        Returns:
            Sum array preserving the same dtype as ``a``.
        """
        if self._available:
            return self._compensated_array_add(  # type: ignore[return-value]
                a.astype(np.float64), b.astype(np.float64)
            ).astype(a.dtype)
        return a + b

    def mix(self, slow: np.ndarray, fast: np.ndarray, eps: float) -> np.ndarray:
        """ACT-compensated exponential moving average: (1-eps)*slow + eps*fast.

        Used for weight consolidation (memory consolidation step) where
        near-cancellation can occur when eps is very small and slow ≈ -fast.

        Args:
            slow: Long-term weight array (w_slow).
            fast: Short-term weight array (w_fast).
            eps: Mixing coefficient in (0, 1).

        Returns:
            Mixed array preserving the same dtype as ``slow``.
        """
        if self._available:
            s64 = slow.astype(np.float64)
            f64 = fast.astype(np.float64)
            eps_s = np.full_like(s64, 1.0 - eps)
            eps_f = np.full_like(f64, eps)
            term_slow = self._compensated_array_multiply(s64, eps_s)
            term_fast = self._compensated_array_multiply(f64, eps_f)
            return self._compensated_array_add(  # type: ignore[return-value]
                term_slow, term_fast
            ).astype(slow.dtype)
        return ((1.0 - eps) * slow + eps * fast).astype(slow.dtype)

    def kahan_sum(self, arr: np.ndarray) -> float:
        """Kahan-compensated sum of all array elements.

        Reduces accumulated rounding error from O(n·ε) for a naive loop to
        O(ε) regardless of array length.  Useful for summing energy terms
        over large sparse graphs.

        Args:
            arr: Input array, any shape, cast to float64.

        Returns:
            Compensated sum as Python float.
        """
        if self._available:
            flat = arr.astype(np.float64).ravel()
            ones = np.ones_like(flat)
            return self._compensated_dot_product(flat, ones)  # type: ignore[return-value]
        return float(np.sum(arr))

    def quadratic_form(self, s: np.ndarray, W: np.ndarray) -> float:
        """Higher-precision quadratic form: s^T W s.

        Upcasts to float64 for the matrix-vector product, then uses Kahan
        dot product for the final scalar accumulation.  The dominant error
        source in Hopfield energy computation for large N is the chained
        matmul: float64 upcast alone reduces relative error from ~1e-7 to ~1e-15.

        Args:
            s: State vector (N,).
            W: Weight matrix (N, N).

        Returns:
            Scalar quadratic form value.
        """
        if self._available:
            s64 = s.astype(np.float64)
            Ws = W.astype(np.float64) @ s64
            return self._compensated_dot_product(Ws, s64)  # type: ignore[return-value]
        return float(s @ W @ s)

    def scale(self, arr: np.ndarray, factor: float) -> np.ndarray:
        """ACT-compensated element-wise multiply by a scalar factor.

        Uses compensated multiply so the accumulated rounding error from
        applying a derived normalization constant (e.g. 1/sqrt(K)) to a
        weight array of length E is O(ε) rather than O(E·ε).  Preserves
        the input dtype.

        Args:
            arr: Input array to scale.
            factor: Scalar multiplier.

        Returns:
            Scaled array with the same dtype as ``arr``.
        """
        if self._available:
            factor_arr = np.full(arr.shape, factor, dtype=np.float64)
            return self._compensated_array_multiply(  # type: ignore[return-value]
                arr.astype(np.float64), factor_arr
            ).astype(arr.dtype)
        return (arr * factor).astype(arr.dtype)

    def matvec_add(self, state: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Higher-precision matrix-vector product with bias: state @ W + b.

        Upcasts inputs to float64 before the matmul so the accumulation of
        N floating-point products (N ≈ 256-832) has lower rounding error
        than a pure float32 operation.  Full Kahan summation per column is
        omitted here for performance; the float64 upcast alone reduces the
        relative error from ~1e-7 to ~1e-15.

        Args:
            state: State vector of shape (N,).
            W: Readout weight matrix of shape (N, vocab_size).
            b: Bias vector of shape (vocab_size,).

        Returns:
            Logit vector of shape (vocab_size,), dtype float32.
        """
        if self._available:
            result = (
                state.astype(np.float64) @ W.astype(np.float64)
                + b.astype(np.float64)
            )
            return result.astype(np.float32)
        return (state @ W + b).astype(np.float32)
