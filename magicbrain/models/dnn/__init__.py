"""Deep Neural Network model adapters."""
from .pytorch_model import DNNModel, create_from_torch_module

__all__ = [
    "DNNModel",
    "create_from_torch_module",
]
