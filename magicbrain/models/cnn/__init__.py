"""CNN model adapters."""
from .vision_model import CNNModel, create_from_torchvision

__all__ = [
    "CNNModel",
    "create_from_torchvision",
]
