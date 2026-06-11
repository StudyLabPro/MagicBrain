"""ACT numerical quality metrics tracker for MagicBrain training.

Tracks compensation-related statistics that are only meaningful when the
Balansis ACT backend is active: Kahan-compensated weight norms, drift
between compensated and naive sums, and energy stability metrics.

Integrate by passing a brain's ACTBackend to ACTMetricsTracker and calling
``record()`` at the same cadence as LiveMonitor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ACTSnapshot:
    """Numerical quality snapshot at a training step."""
    step: int
    act_available: bool
    w_slow_l2: float
    w_fast_l2: float
    # Relative deviation: (compensated_sum - naive_sum) / (|naive_sum| + eps)
    w_slow_compensation_ratio: float
    w_fast_compensation_ratio: float
    # std(w_slow + w_fast) measured via ACT vs naive
    combined_weight_std: float


class ACTMetricsTracker:
    """Tracks ACT numerical stability metrics during training.

    Designed to be optionally attached to a TextBrain when ``use_act=True``.
    Falls back to numpy metrics gracefully when Balansis is unavailable.

    Example::

        tracker = ACTMetricsTracker(brain._act)
        # ... training loop ...
        snapshot = tracker.record(brain, step=100)
        print(tracker.summary())
    """

    def __init__(self, act_backend=None):
        """
        Args:
            act_backend: ACTBackend instance (or None for numpy-only mode).
        """
        self._act = act_backend
        self.history: List[ACTSnapshot] = []

    @property
    def act_available(self) -> bool:
        return self._act is not None and self._act.available

    def record(self, brain, step: int) -> ACTSnapshot:
        """Compute and store a numerical quality snapshot.

        Args:
            brain: TextBrain instance.
            step: Current training step.

        Returns:
            ACTSnapshot with computed metrics.
        """
        w_slow = brain.w_slow
        w_fast = brain.w_fast

        if self.act_available:
            # Compensated L2 norms via kahan_sum(arr ** 2)
            w_slow_l2 = float(np.sqrt(self._act.kahan_sum(w_slow.astype(np.float64) ** 2)))
            w_fast_l2 = float(np.sqrt(self._act.kahan_sum(w_fast.astype(np.float64) ** 2)))

            # Compensation ratio: how much the ACT sum deviates from naive numpy
            naive_slow = float(np.sum(w_slow.astype(np.float64)))
            kahan_slow = self._act.kahan_sum(w_slow)
            w_slow_comp = abs(kahan_slow - naive_slow) / (abs(naive_slow) + 1e-9)

            naive_fast = float(np.sum(w_fast.astype(np.float64)))
            kahan_fast = self._act.kahan_sum(w_fast)
            w_fast_comp = abs(kahan_fast - naive_fast) / (abs(naive_fast) + 1e-9)

            # Combined weight std via ACT
            combined = self._act.add(w_slow, w_fast)
            combined_mean = self._act.kahan_sum(combined) / max(1, combined.size)
            centered = self._act.subtract(combined, np.full_like(combined, combined_mean))
            combined_std = float(np.sqrt(self._act.kahan_sum(centered.astype(np.float64) ** 2) / max(1, combined.size)))
        else:
            w_slow_l2 = float(np.linalg.norm(w_slow))
            w_fast_l2 = float(np.linalg.norm(w_fast))
            w_slow_comp = 0.0
            w_fast_comp = 0.0
            combined_std = float(np.std(w_slow + w_fast))

        snapshot = ACTSnapshot(
            step=step,
            act_available=self.act_available,
            w_slow_l2=w_slow_l2,
            w_fast_l2=w_fast_l2,
            w_slow_compensation_ratio=w_slow_comp,
            w_fast_compensation_ratio=w_fast_comp,
            combined_weight_std=combined_std,
        )
        self.history.append(snapshot)
        return snapshot

    def summary(self) -> dict:
        """Return summary of ACT metrics over recorded history.

        Returns:
            Dict with mean/max compensation ratios and weight norms.
        """
        if not self.history:
            return {"n_records": 0, "act_available": self.act_available}

        slow_comps = [s.w_slow_compensation_ratio for s in self.history]
        fast_comps = [s.w_fast_compensation_ratio for s in self.history]
        slow_l2s = [s.w_slow_l2 for s in self.history]

        return {
            "n_records": len(self.history),
            "act_available": self.act_available,
            "mean_slow_compensation_ratio": float(np.mean(slow_comps)),
            "max_slow_compensation_ratio": float(np.max(slow_comps)),
            "mean_fast_compensation_ratio": float(np.mean(fast_comps)),
            "max_fast_compensation_ratio": float(np.max(fast_comps)),
            "final_w_slow_l2": float(slow_l2s[-1]) if slow_l2s else 0.0,
            "final_combined_std": float(self.history[-1].combined_weight_std),
        }

    def get_recent(self, n: int = 10) -> List[ACTSnapshot]:
        """Return n most recent snapshots."""
        return self.history[-n:]
