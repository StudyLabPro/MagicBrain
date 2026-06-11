"""API routes."""
from . import models, training, inference, diagnostics, evolution, twins, auto_evolution

__all__ = [
    "models", "training", "inference", "diagnostics",
    "evolution", "twins", "auto_evolution",
]
