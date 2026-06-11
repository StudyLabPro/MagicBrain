"""Backend system for MagicBrain."""
from .backend_interface import Backend, get_backend, auto_select_backend
from .numpy_backend import NumpyBackend

__all__ = [
    "Backend",
    "get_backend",
    "auto_select_backend",
    "NumpyBackend",
]

try:
    from .jax_backend import JAXBackend
    __all__.append("JAXBackend")
except ImportError:
    pass
