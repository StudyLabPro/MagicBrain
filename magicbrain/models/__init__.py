"""Model adapters for different neural network types."""
from .snn import SNNTextModel

# DNN models (optional dependency)
try:
    from .dnn import DNNModel, create_from_torch_module
    __all__ = [
        "SNNTextModel",
        "DNNModel",
        "create_from_torch_module",
    ]
except ImportError:
    __all__ = [
        "SNNTextModel",
    ]
