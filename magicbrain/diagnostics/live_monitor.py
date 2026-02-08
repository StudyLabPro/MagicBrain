"""
Live monitoring system for tracking brain training metrics.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific step."""
    step: int
    loss: float
    dopamine: float
    avg_theta: float
    firing_rate: float
    mean_w_slow: float
    mean_w_fast: float
    mean_abs_w: float
    num_active_neurons: int
    timestamp: Optional[float] = None


class LiveMonitor:
    """
    Real-time monitoring system for brain training.
    Collects and stores metrics without visualization dependencies.
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.metrics_history: List[TrainingMetrics] = []
        self.current_step = 0

    def record(self, brain, loss: float, step: int) -> TrainingMetrics:
        """
        Record current metrics from brain state.

        Args:
            brain: TextBrain instance
            loss: Current loss value
            step: Training step number

        Returns:
            TrainingMetrics object
        """
        import time

        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            dopamine=float(brain.dopamine),
            avg_theta=float(brain.avg_theta()),
            firing_rate=float(brain.firing_rate()),
            mean_w_slow=float(brain.mean_abs_w_slow()),
            mean_w_fast=float(brain.mean_abs_w_fast()),
            mean_abs_w=float(brain.mean_abs_w()),
            num_active_neurons=int(brain.a.sum()),
            timestamp=time.time()
        )

        self.metrics_history.append(metrics)
        self.current_step = step

        return metrics

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step % self.log_every == 0

    def get_recent_metrics(self, n: int = 10) -> List[TrainingMetrics]:
        """Get n most recent metrics."""
        return self.metrics_history[-n:]

    def get_summary(self) -> Dict:
        """Get summary statistics of training."""
        if not self.metrics_history:
            return {}

        recent = self.get_recent_metrics(100)
        losses = [m.loss for m in recent]
        dopamines = [m.dopamine for m in recent]
        firing_rates = [m.firing_rate for m in recent]

        return {
            "total_steps": self.current_step,
            "total_records": len(self.metrics_history),
            "recent_avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "recent_avg_dopamine": sum(dopamines) / len(dopamines) if dopamines else 0.0,
            "recent_avg_firing_rate": sum(firing_rates) / len(firing_rates) if firing_rates else 0.0,
            "final_metrics": recent[-1].__dict__ if recent else {},
        }

    def save(self, path: str):
        """Save metrics history to JSON file."""
        data = {
            "log_every": self.log_every,
            "total_steps": self.current_step,
            "metrics": [
                {k: v for k, v in m.__dict__.items()}
                for m in self.metrics_history
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> LiveMonitor:
        """Load metrics history from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        monitor = cls(log_every=data.get("log_every", 100))
        monitor.current_step = data.get("total_steps", 0)

        for m_dict in data.get("metrics", []):
            metrics = TrainingMetrics(**m_dict)
            monitor.metrics_history.append(metrics)

        return monitor

    def print_status(self, step: int, loss: float, brain):
        """Print current training status."""
        if not self.should_log(step):
            return

        print(
            f"Step {step:6d} | "
            f"Loss: {loss:.4f} | "
            f"DA: {brain.dopamine:.3f} | "
            f"θ: {brain.avg_theta():.3f} | "
            f"FR: {brain.firing_rate():.3f} | "
            f"|W|: {brain.mean_abs_w():.4f}"
        )


def add_diagnostics_methods_to_brain():
    """
    Add diagnostic methods to TextBrain class.
    This is a monkey-patch approach for backward compatibility.
    """
    from ..brain import TextBrain

    def avg_theta(self):
        """Average homeostatic threshold."""
        import numpy as np
        return np.mean(self.theta)

    def firing_rate(self):
        """Current firing rate (fraction of active neurons)."""
        import numpy as np
        return np.mean(self.a)

    def mean_abs_w_slow(self):
        """Mean absolute slow weight."""
        import numpy as np
        return np.mean(np.abs(self.w_slow))

    def mean_abs_w_fast(self):
        """Mean absolute fast weight."""
        import numpy as np
        return np.mean(np.abs(self.w_fast))

    def mean_abs_w(self):
        """Mean absolute combined weight."""
        import numpy as np
        return np.mean(np.abs(self.w_slow + self.w_fast))

    # Add methods if they don't exist
    if not hasattr(TextBrain, 'avg_theta'):
        TextBrain.avg_theta = avg_theta
    if not hasattr(TextBrain, 'firing_rate'):
        TextBrain.firing_rate = firing_rate
    if not hasattr(TextBrain, 'mean_abs_w_slow'):
        TextBrain.mean_abs_w_slow = mean_abs_w_slow
    if not hasattr(TextBrain, 'mean_abs_w_fast'):
        TextBrain.mean_abs_w_fast = mean_abs_w_fast
    if not hasattr(TextBrain, 'mean_abs_w'):
        TextBrain.mean_abs_w = mean_abs_w


# Auto-patch on import
add_diagnostics_methods_to_brain()
