"""
SNN Text Model - Platform adapter for TextBrain.

Wraps the existing TextBrain in ModelInterface for platform compatibility.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from ...brain import TextBrain
from ...platform.model_interface import (
    StatefulModel,
    ModelMetadata,
    ModelState,
    ModelType,
    OutputType,
)


class SNNTextModel(StatefulModel):
    """
    Platform adapter for TextBrain (Spiking Neural Network).

    Wraps TextBrain to make it compatible with MagicBrain Platform.
    """

    def __init__(
        self,
        genome: str,
        vocab_size: int,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        description: str = ""
    ):
        """
        Initialize SNN text model.

        Args:
            genome: DNA-like genome string for network configuration
            vocab_size: Vocabulary size
            model_id: Unique model ID
            version: Model version
            description: Model description
        """
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id or f"snn_text_{id(self)}",
            model_type=ModelType.SNN,
            version=version,
            description=description or "Spiking Neural Network for text modeling",
            output_type=OutputType.LOGITS,
            framework="magicbrain",
        )

        super().__init__(metadata)

        # Initialize TextBrain
        self.brain = TextBrain(genome, vocab_size)
        self.genome = genome
        self.vocab_size = vocab_size

        # Update metadata with brain parameters
        self.metadata.parameters_count = self._count_parameters()
        self.metadata.input_shape = (vocab_size,)
        self.metadata.output_shape = (vocab_size,)
        self.metadata.extra = {
            "genome": genome,
            "N": self.brain.N,
            "K": self.brain.K,
            "k_active": int(self.brain.p["k_active"]),
        }

    def _count_parameters(self) -> int:
        """Count total parameters in the brain."""
        # Count edges (w_slow + w_fast)
        num_edges = len(self.brain.src)
        edge_params = num_edges * 2

        # Count output layer (R + b)
        output_params = self.brain.N * self.vocab_size + self.vocab_size

        # Count thresholds
        threshold_params = self.brain.N

        return edge_params + output_params + threshold_params

    def forward(self, input: Any, **kwargs) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            input: Token ID (int) or token IDs (array)
            **kwargs: Additional parameters

        Returns:
            Logits for next token prediction (vocab_size,)
        """
        # Handle single token or batch
        if isinstance(input, int):
            token_id = input
        elif isinstance(input, np.ndarray):
            token_id = int(input.item()) if input.size == 1 else int(input[0])
        else:
            token_id = int(input)

        # Forward through brain
        probs = self.brain.forward(token_id)

        # Convert probabilities to logits (inverse softmax)
        # logits = log(probs)
        logits = np.log(probs + 1e-10)

        # Update state
        self._state.internal_state = {
            "activation": self.brain.a.copy(),
            "trace_fast": self.brain.trace_fast.copy(),
            "trace_slow": self.brain.trace_slow.copy(),
            "theta": self.brain.theta.copy(),
        }

        # Update metrics
        self._state.metrics = {
            "firing_rate": self.brain.firing_rate(),
            "avg_theta": self.brain.avg_theta(),
            "dopamine": self.brain.dopamine,
            "mean_abs_w": self.brain.mean_abs_w(),
        }

        return logits

    def step(self, input: Any, **kwargs) -> np.ndarray:
        """
        Single timestep forward (same as forward for this model).

        Args:
            input: Token ID
            **kwargs: Additional parameters

        Returns:
            Logits for next token
        """
        return self.forward(input, **kwargs)

    def learn(self, target_id: int) -> float:
        """
        Learning step.

        Args:
            target_id: Target token ID

        Returns:
            Loss value
        """
        # Get current probabilities
        probs = np.exp(self.forward(target_id))
        probs = probs / probs.sum()

        # Learn
        loss = self.brain.learn(target_id, probs)

        # Update state metrics
        self._state.metrics["loss"] = loss

        return loss

    def get_output_type(self) -> OutputType:
        """
        Get output type.

        Returns:
            Output type (LOGITS for this model)
        """
        return OutputType.LOGITS

    def get_hidden_state(self) -> Dict[str, np.ndarray]:
        """
        Get current hidden state.

        Returns:
            Dictionary with brain internal state
        """
        return {
            "activation": self.brain.a.copy(),
            "trace_fast": self.brain.trace_fast.copy(),
            "trace_slow": self.brain.trace_slow.copy(),
            "theta": self.brain.theta.copy(),
            "w_slow": self.brain.w_slow.copy(),
            "w_fast": self.brain.w_fast.copy(),
        }

    def set_hidden_state(self, hidden: Dict[str, np.ndarray]):
        """
        Set hidden state.

        Args:
            hidden: Dictionary with state arrays
        """
        if "activation" in hidden:
            self.brain.a = hidden["activation"].copy()
        if "trace_fast" in hidden:
            self.brain.trace_fast = hidden["trace_fast"].copy()
        if "trace_slow" in hidden:
            self.brain.trace_slow = hidden["trace_slow"].copy()
        if "theta" in hidden:
            self.brain.theta = hidden["theta"].copy()
        if "w_slow" in hidden:
            self.brain.w_slow = hidden["w_slow"].copy()
        if "w_fast" in hidden:
            self.brain.w_fast = hidden["w_fast"].copy()

    def get_spikes(self) -> np.ndarray:
        """
        Get current spike activation.

        Returns:
            Binary spike array (N,)
        """
        return self.brain.a.copy()

    def get_traces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get synaptic traces.

        Returns:
            Tuple of (trace_fast, trace_slow)
        """
        return self.brain.trace_fast.copy(), self.brain.trace_slow.copy()

    def reset(self):
        """Reset model to initial state."""
        super().reset()
        # Reset brain internal state
        self.brain.a = np.zeros(self.brain.N, dtype=np.float32)
        self.brain.trace_fast = np.zeros(self.brain.N, dtype=np.float32)
        self.brain.trace_slow = np.zeros(self.brain.N, dtype=np.float32)
        # Keep learned weights and theta

    def save_weights(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save file
        """
        from ...io import save_model
        save_model(self.brain, path, vocab=None)

    def load_weights(self, path: str):
        """
        Load model weights.

        Args:
            path: Path to load file
        """
        from ...io import load_model
        loaded_brain, _, _, _ = load_model(path)

        # Copy weights
        self.brain.w_slow = loaded_brain.w_slow.copy()
        self.brain.w_fast = loaded_brain.w_fast.copy()
        self.brain.R = loaded_brain.R.copy()
        self.brain.b = loaded_brain.b.copy()
        self.brain.theta = loaded_brain.theta.copy()

    def get_brain_stats(self) -> Dict[str, Any]:
        """
        Get detailed brain statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "N": self.brain.N,
            "K": self.brain.K,
            "E": len(self.brain.src),
            "k_active": self.brain.p["k_active"],
            "firing_rate": self.brain.firing_rate(),
            "avg_theta": self.brain.avg_theta(),
            "dopamine": self.brain.dopamine,
            "mean_abs_w": self.brain.mean_abs_w(),
            "loss_ema": self.brain.loss_ema,
            "step": self.brain.step,
        }

    def summary(self) -> str:
        """
        Get model summary.

        Returns:
            Human-readable summary
        """
        base_summary = super().summary()
        brain_stats = self.get_brain_stats()

        return f"""{base_summary}

SNN Brain Statistics:
  Neurons: {brain_stats['N']}
  Connections per neuron: {brain_stats['K']}
  Total edges: {brain_stats['E']}
  Active neurons: {int(brain_stats['k_active'])}
  Firing rate: {brain_stats['firing_rate']:.4f}
  Mean |weight|: {brain_stats['mean_abs_w']:.4f}
  Training step: {brain_stats['step']}
"""


def create_from_existing_brain(
    brain: TextBrain,
    vocab_size: int,
    model_id: Optional[str] = None,
    version: str = "1.0.0"
) -> SNNTextModel:
    """
    Create SNNTextModel from existing TextBrain instance.

    Args:
        brain: Existing TextBrain
        vocab_size: Vocabulary size
        model_id: Model ID
        version: Version

    Returns:
        SNNTextModel wrapping the brain
    """
    # Extract genome from brain (if available)
    genome = getattr(brain, 'genome_str', '30121033102301230112332100123')

    model = SNNTextModel(
        genome=genome,
        vocab_size=vocab_size,
        model_id=model_id,
        version=version,
        description="SNN Text Model created from existing TextBrain"
    )

    # Replace brain with existing one
    model.brain = brain

    # Update metadata
    model.metadata.parameters_count = model._count_parameters()

    return model
