"""
CNN Model - Platform adapter for Computer Vision models.

Wraps torchvision models for MagicBrain Platform.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    class nn:
        class Module:
            pass

from ...platform.model_interface import (
    ModelInterface,
    ModelMetadata,
    ModelType,
    OutputType,
)


class CNNModel(ModelInterface):
    """Platform adapter for CNN/Vision models."""

    def __init__(
        self,
        torch_module: nn.Module,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        output_type: OutputType = OutputType.FEATURES,
        device: Optional[str] = None,
        feature_layer: Optional[str] = None,
    ):
        """Initialize CNN model."""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not installed")

        metadata = ModelMetadata(
            model_id=model_id or f"cnn_{id(self)}",
            model_type=ModelType.CNN,
            version=version,
            description=description or "CNN vision model",
            output_type=output_type,
            framework="torchvision",
        )

        super().__init__(metadata)

        self.torch_module = torch_module
        self.feature_layer = feature_layer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.torch_module = self.torch_module.to(self.device)
        self.torch_module.eval()

        self.metadata.parameters_count = sum(
            p.numel() for p in self.torch_module.parameters()
        )

    def forward(self, input: Any, **kwargs) -> np.ndarray:
        """Forward pass."""
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input).to(self.device)
        elif isinstance(input, torch.Tensor):
            x = input.to(self.device)
        else:
            x = torch.tensor(input, device=self.device)

        if x.dim() == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            if self.feature_layer:
                output = self._get_layer_features(x, self.feature_layer)
            else:
                output = self.torch_module(x)

        result = output.detach().cpu().numpy()

        if result.shape[0] == 1 and not kwargs.get("keep_batch_dim", False):
            result = result.squeeze(0)

        return result

    def _get_layer_features(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Extract features from specific layer."""
        activation = {}

        def hook(model, input, output):
            activation['output'] = output

        layer = dict(self.torch_module.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook)
        self.torch_module(x)
        handle.remove()

        return activation.get('output', x)

    def get_output_type(self) -> OutputType:
        return self.metadata.output_type

    def summary(self) -> str:
        base = super().summary()
        return f"""{base}

CNN Model Details:
  Module: {type(self.torch_module).__name__}
  Device: {self.device}
  Feature layer: {self.feature_layer or 'output'}
"""


def create_from_torchvision(
    model_name: str,
    pretrained: bool = True,
    model_id: Optional[str] = None,
    **kwargs
) -> CNNModel:
    """Create CNNModel from torchvision."""
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision not installed")

    model_func = getattr(models, model_name)
    torch_module = model_func(pretrained=pretrained)

    return CNNModel(
        torch_module=torch_module,
        model_id=model_id or f"cnn_{model_name}",
        description=f"torchvision {model_name}",
        **kwargs
    )
