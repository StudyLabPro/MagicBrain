"""
DNN Model - Platform adapter for PyTorch models.

Wraps torch.nn.Module to make it compatible with MagicBrain Platform.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass

from ...platform.model_interface import (
    ModelInterface,
    ModelMetadata,
    ModelState,
    ModelType,
    OutputType,
)


class DNNModel(ModelInterface):
    """
    Platform adapter for PyTorch DNN models.

    Wraps torch.nn.Module to make it compatible with MagicBrain Platform.
    """

    def __init__(
        self,
        torch_module: nn.Module,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        input_shape: Optional[Tuple] = None,
        output_shape: Optional[Tuple] = None,
        output_type: OutputType = OutputType.DENSE,
        device: Optional[str] = None,
    ):
        """
        Initialize DNN model.

        Args:
            torch_module: PyTorch nn.Module
            model_id: Unique model ID
            version: Model version
            description: Model description
            input_shape: Expected input shape (without batch dim)
            output_shape: Expected output shape (without batch dim)
            output_type: Type of output (DENSE, LOGITS, EMBEDDINGS, etc)
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id or f"dnn_{id(self)}",
            model_type=ModelType.DNN,
            version=version,
            description=description or "PyTorch DNN model",
            output_type=output_type,
            framework="pytorch",
            input_shape=input_shape,
            output_shape=output_shape,
        )

        super().__init__(metadata)

        # Store torch module
        self.torch_module = torch_module

        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move model to device
        self.torch_module = self.torch_module.to(self.device)

        # Count parameters
        self.metadata.parameters_count = sum(
            p.numel() for p in self.torch_module.parameters()
        )

        # Training mode flag
        self._training = False

    def forward(self, input: Any, **kwargs) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            input: Input data (numpy array or torch tensor)
            **kwargs: Additional parameters (batch_size, etc)

        Returns:
            Output as numpy array
        """
        # Convert input to tensor if needed
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input).to(self.device)
        elif isinstance(input, torch.Tensor):
            x = input.to(self.device)
        else:
            x = torch.tensor(input, device=self.device)

        # Ensure batch dimension
        if x.dim() == len(self.metadata.input_shape or ()):
            x = x.unsqueeze(0)

        # Forward pass
        with torch.set_grad_enabled(self._training):
            output = self.torch_module(x)

        # Convert to numpy
        if isinstance(output, torch.Tensor):
            result = output.detach().cpu().numpy()
        else:
            # Handle tuple/dict outputs
            result = self._convert_output_to_numpy(output)

        # Remove batch dimension if single sample
        if result.shape[0] == 1 and not kwargs.get("keep_batch_dim", False):
            result = result.squeeze(0)

        # Update state
        self._state.internal_state = {
            "last_input_shape": x.shape,
            "last_output_shape": result.shape,
        }

        return result

    def _convert_output_to_numpy(self, output: Any) -> np.ndarray:
        """Convert various output types to numpy array."""
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().numpy()
        elif isinstance(output, tuple):
            # Return first element for tuple outputs
            return self._convert_output_to_numpy(output[0])
        elif isinstance(output, dict):
            # Return 'logits' or first value for dict outputs
            if 'logits' in output:
                return self._convert_output_to_numpy(output['logits'])
            return self._convert_output_to_numpy(next(iter(output.values())))
        else:
            return np.array(output)

    def get_output_type(self) -> OutputType:
        """
        Get output type.

        Returns:
            Output type
        """
        return self.metadata.output_type

    def train(self):
        """Set model to training mode."""
        self._training = True
        self.torch_module.train()

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.torch_module.eval()

    def is_training(self) -> bool:
        """Check if model is in training mode."""
        return self._training

    def get_device(self) -> str:
        """Get current device."""
        return str(self.device)

    def to(self, device: Union[str, torch.device]):
        """
        Move model to device.

        Args:
            device: Target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.torch_module = self.torch_module.to(device)

    def get_torch_module(self) -> nn.Module:
        """
        Get underlying PyTorch module.

        Returns:
            torch.nn.Module
        """
        return self.torch_module

    def save_weights(self, path: Union[str, Path]):
        """
        Save model weights.

        Args:
            path: Path to save file
        """
        path = Path(path)
        torch.save(self.torch_module.state_dict(), path)

    def load_weights(self, path: Union[str, Path]):
        """
        Load model weights.

        Args:
            path: Path to load file
        """
        path = Path(path)
        state_dict = torch.load(path, map_location=self.device)
        self.torch_module.load_state_dict(state_dict)

    def save_model(self, path: Union[str, Path]):
        """
        Save entire model (architecture + weights).

        Args:
            path: Path to save file
        """
        path = Path(path)
        torch.save(self.torch_module, path)

    @classmethod
    def load_model(cls, path: Union[str, Path], **kwargs) -> 'DNNModel':
        """
        Load entire model from file.

        Args:
            path: Path to model file
            **kwargs: Additional arguments for DNNModel constructor

        Returns:
            DNNModel instance
        """
        path = Path(path)
        torch_module = torch.load(path)
        return cls(torch_module, **kwargs)

    def get_layer_names(self) -> list[str]:
        """
        Get names of all layers/modules.

        Returns:
            List of layer names
        """
        return [name for name, _ in self.torch_module.named_modules()]

    def get_layer_output(
        self,
        layer_name: str,
        input: Any
    ) -> np.ndarray:
        """
        Get output from a specific layer.

        Args:
            layer_name: Name of layer
            input: Input data

        Returns:
            Layer output as numpy array
        """
        activation = {}

        def hook(model, input, output):
            activation['output'] = output

        # Register hook
        layer = dict(self.torch_module.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook)

        # Forward pass
        self.forward(input)

        # Remove hook
        handle.remove()

        # Return activation
        if 'output' in activation:
            return self._convert_output_to_numpy(activation['output'])
        return None

    def reset(self):
        """Reset model state."""
        super().reset()
        # PyTorch models typically don't have persistent state
        # But we can reset batch norm stats, etc.
        if hasattr(self.torch_module, 'reset_parameters'):
            self.torch_module.reset_parameters()

    def summary(self) -> str:
        """
        Get model summary.

        Returns:
            Human-readable summary
        """
        base_summary = super().summary()

        # Get module info
        total_params = self.metadata.parameters_count
        trainable_params = sum(
            p.numel() for p in self.torch_module.parameters() if p.requires_grad
        )

        return f"""{base_summary}

DNN Model Details:
  Total parameters: {total_params:,}
  Trainable parameters: {trainable_params:,}
  Device: {self.device}
  Training mode: {self._training}
  Module type: {type(self.torch_module).__name__}
"""


def create_from_torch_module(
    module: nn.Module,
    model_id: Optional[str] = None,
    version: str = "1.0.0",
    **kwargs
) -> DNNModel:
    """
    Create DNNModel from PyTorch module.

    Args:
        module: PyTorch nn.Module
        model_id: Model ID
        version: Version
        **kwargs: Additional DNNModel parameters

    Returns:
        DNNModel instance
    """
    return DNNModel(
        torch_module=module,
        model_id=model_id,
        version=version,
        **kwargs
    )
