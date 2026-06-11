"""
RNN Model - Platform adapter for Recurrent Neural Networks.

Wraps PyTorch RNN/LSTM/GRU models.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

from ...platform.model_interface import (
    StatefulModel,
    ModelMetadata,
    ModelType,
    OutputType,
)


class RNNModel(StatefulModel):
    """Platform adapter for RNN/LSTM/GRU models."""

    def __init__(
        self,
        torch_module: nn.Module,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        output_type: OutputType = OutputType.HIDDEN,
        device: Optional[str] = None,
    ):
        """Initialize RNN model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        metadata = ModelMetadata(
            model_id=model_id or f"rnn_{id(self)}",
            model_type=ModelType.RNN,
            version=version,
            description=description or "Recurrent neural network",
            output_type=output_type,
            framework="pytorch",
        )

        super().__init__(metadata)

        self.torch_module = torch_module

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.torch_module = self.torch_module.to(self.device)

        self.metadata.parameters_count = sum(
            p.numel() for p in self.torch_module.parameters()
        )

        # Hidden state
        self.hidden = None

    def forward(self, input: Any, **kwargs) -> np.ndarray:
        """Forward pass (full sequence)."""
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input).to(self.device)
        elif isinstance(input, torch.Tensor):
            x = input.to(self.device)
        else:
            x = torch.tensor(input, device=self.device)

        # Ensure (batch, seq, features) or (seq, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch

        with torch.no_grad():
            if self.hidden is not None:
                output, self.hidden = self.torch_module(x, self.hidden)
            else:
                output, self.hidden = self.torch_module(x)

        result = output.detach().cpu().numpy()

        if result.shape[0] == 1 and not kwargs.get("keep_batch_dim", False):
            result = result.squeeze(0)

        self._state.hidden_states = self._hidden_to_numpy()

        return result

    def step(self, input: Any, **kwargs) -> np.ndarray:
        """Single timestep forward."""
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input).to(self.device)
        elif isinstance(input, torch.Tensor):
            x = input.to(self.device)
        else:
            x = torch.tensor(input, device=self.device)

        # Ensure (batch, 1, features)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        with torch.no_grad():
            if self.hidden is not None:
                output, self.hidden = self.torch_module(x, self.hidden)
            else:
                output, self.hidden = self.torch_module(x)

        result = output.detach().cpu().numpy()

        if result.shape[0] == 1:
            result = result.squeeze(0)

        return result

    def get_hidden_state(self) -> Any:
        """Get current hidden state."""
        return self._hidden_to_numpy()

    def set_hidden_state(self, hidden: Any):
        """Set hidden state."""
        if hidden is None:
            self.hidden = None
        elif isinstance(hidden, tuple):
            # LSTM (h, c)
            self.hidden = tuple(
                torch.from_numpy(h).to(self.device) if isinstance(h, np.ndarray) else h
                for h in hidden
            )
        else:
            # RNN/GRU
            if isinstance(hidden, np.ndarray):
                self.hidden = torch.from_numpy(hidden).to(self.device)
            else:
                self.hidden = hidden

    def _hidden_to_numpy(self) -> Any:
        """Convert hidden state to numpy."""
        if self.hidden is None:
            return None
        elif isinstance(self.hidden, tuple):
            return tuple(h.detach().cpu().numpy() for h in self.hidden)
        else:
            return self.hidden.detach().cpu().numpy()

    def get_output_type(self) -> OutputType:
        return self.metadata.output_type

    def reset(self):
        """Reset hidden state."""
        super().reset()
        self.hidden = None

    def summary(self) -> str:
        base = super().summary()
        return f"""{base}

RNN Model Details:
  Module: {type(self.torch_module).__name__}
  Device: {self.device}
  Has hidden state: {self.hidden is not None}
"""
