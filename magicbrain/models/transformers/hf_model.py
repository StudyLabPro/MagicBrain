"""
Transformer Model - Platform adapter for Hugging Face Transformers.

Wraps HF models to make them compatible with MagicBrain Platform.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
from pathlib import Path

try:
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Dummy classes
    class PreTrainedModel:
        pass
    class PreTrainedTokenizer:
        pass

from ...platform.model_interface import (
    ModelInterface,
    ModelMetadata,
    ModelState,
    ModelType,
    OutputType,
)


class TransformerModel(ModelInterface):
    """
    Platform adapter for Hugging Face Transformer models.

    Supports any model from Hugging Face model hub.
    """

    def __init__(
        self,
        hf_model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        output_type: OutputType = OutputType.EMBEDDINGS,
        device: Optional[str] = None,
        return_attention: bool = False,
    ):
        """
        Initialize Transformer model.

        Args:
            hf_model: Hugging Face PreTrainedModel
            tokenizer: Optional tokenizer
            model_id: Unique model ID
            version: Model version
            description: Model description
            output_type: Type of output (EMBEDDINGS, LOGITS, ATTENTION)
            device: Device to run on
            return_attention: Whether to return attention weights
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers not installed. "
                "Install with: pip install transformers"
            )

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id or f"transformer_{id(self)}",
            model_type=ModelType.TRANSFORMER,
            version=version,
            description=description or "Hugging Face Transformer model",
            output_type=output_type,
            framework="huggingface",
        )

        super().__init__(metadata)

        # Store model and tokenizer
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.return_attention = return_attention

        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move model to device
        self.hf_model = self.hf_model.to(self.device)

        # Count parameters
        self.metadata.parameters_count = sum(
            p.numel() for p in self.hf_model.parameters()
        )

        # Get model config
        self.config = self.hf_model.config

        # Update metadata
        if hasattr(self.config, 'hidden_size'):
            self.metadata.output_shape = (self.config.hidden_size,)

    def forward(self, input: Any, **kwargs) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through the model.

        Args:
            input: Input data (text string, token IDs, or tensor)
            **kwargs: Additional parameters

        Returns:
            Embeddings, logits, or attention weights as numpy array/dict
        """
        # Handle different input types
        if isinstance(input, str):
            # Text input - tokenize
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for text input")
            inputs = self.tokenizer(
                input,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif isinstance(input, dict):
            # Already tokenized
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in input.items()}
        elif isinstance(input, (list, np.ndarray)):
            # Token IDs
            if isinstance(input, np.ndarray):
                input = torch.from_numpy(input)
            if not isinstance(input, torch.Tensor):
                input = torch.tensor(input)
            inputs = {"input_ids": input.to(self.device)}
        elif isinstance(input, torch.Tensor):
            inputs = {"input_ids": input.to(self.device)}
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # Forward pass
        with torch.no_grad():
            outputs = self.hf_model(
                **inputs,
                output_attentions=self.return_attention,
                output_hidden_states=True
            )

        # Extract appropriate output based on output_type
        result = self._extract_output(outputs, kwargs.get("layer", -1))

        # Update state
        self._state.internal_state = {
            "input_shape": str(inputs.get("input_ids", torch.tensor([])).shape),
            "output_shape": str(result.shape) if isinstance(result, np.ndarray) else "dict",
        }

        if self.return_attention and hasattr(outputs, 'attentions'):
            self._state.internal_state['attention_heads'] = len(outputs.attentions)

        return result

    def _extract_output(
        self,
        outputs,
        layer: int = -1
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Extract output based on output_type."""

        if self.metadata.output_type == OutputType.EMBEDDINGS:
            # Use last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden = outputs.hidden_states[layer]
            else:
                hidden = outputs[0]

            # Mean pooling over sequence
            embeddings = hidden.mean(dim=1)  # (batch, hidden_size)
            return embeddings.detach().cpu().numpy()

        elif self.metadata.output_type == OutputType.LOGITS:
            # Classification logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            return logits.detach().cpu().numpy()

        elif self.metadata.output_type == OutputType.ATTENTION:
            # Attention weights
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Return last layer attention
                attention = outputs.attentions[-1]
                return attention.detach().cpu().numpy()
            return None

        elif self.metadata.output_type == OutputType.HIDDEN:
            # All hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Stack all layers
                hidden_states = torch.stack(outputs.hidden_states)
                return hidden_states.detach().cpu().numpy()
            return None

        else:
            # Default: return last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.detach().cpu().numpy()
            return outputs[0].detach().cpu().numpy()

    def get_output_type(self) -> OutputType:
        """Get output type."""
        return self.metadata.output_type

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embeddings.

        Args:
            text: Text string or list of strings

        Returns:
            Embeddings array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text encoding")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward
        with torch.no_grad():
            outputs = self.hf_model(**inputs, output_hidden_states=True)

        # Mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[-1]

        # Pool over sequence
        embeddings = hidden.mean(dim=1)

        return embeddings.detach().cpu().numpy()

    def get_attention_weights(
        self,
        input: Union[str, torch.Tensor],
        layer: int = -1
    ) -> np.ndarray:
        """
        Get attention weights for input.

        Args:
            input: Input text or tokens
            layer: Which layer (-1 for last)

        Returns:
            Attention weights array
        """
        # Temporarily enable attention
        old_return_attention = self.return_attention
        self.return_attention = True

        # Forward
        result = self.forward(input)

        # Restore setting
        self.return_attention = old_return_attention

        # Extract attention from state
        if hasattr(self, '_last_attentions'):
            return self._last_attentions[layer]

        return None

    def get_hidden_states(
        self,
        input: Union[str, torch.Tensor],
        layer: Optional[int] = None
    ) -> np.ndarray:
        """
        Get hidden states from specific layer.

        Args:
            input: Input text or tokens
            layer: Which layer (None for all)

        Returns:
            Hidden states array
        """
        old_output_type = self.metadata.output_type
        self.metadata.output_type = OutputType.HIDDEN

        result = self.forward(input)

        self.metadata.output_type = old_output_type

        if layer is not None and result is not None:
            return result[layer]

        return result

    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.hf_model = self.hf_model.to(device)

    def get_device(self) -> str:
        """Get current device."""
        return str(self.device)

    def save_pretrained(self, path: Union[str, Path]):
        """
        Save model using HF save_pretrained.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.hf_model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

    @classmethod
    def load_pretrained(
        cls,
        path: Union[str, Path],
        **kwargs
    ) -> 'TransformerModel':
        """
        Load model from directory.

        Args:
            path: Directory to load from
            **kwargs: Additional arguments

        Returns:
            TransformerModel instance
        """
        path = Path(path)
        hf_model = AutoModel.from_pretrained(path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except:
            tokenizer = None

        return cls(hf_model, tokenizer, **kwargs)

    def summary(self) -> str:
        """Get model summary."""
        base_summary = super().summary()

        config_info = f"""
Transformer Model Details:
  Model type: {self.config.model_type if hasattr(self.config, 'model_type') else 'unknown'}
  Hidden size: {self.config.hidden_size if hasattr(self.config, 'hidden_size') else 'unknown'}
  Num layers: {self.config.num_hidden_layers if hasattr(self.config, 'num_hidden_layers') else 'unknown'}
  Num attention heads: {self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else 'unknown'}
  Vocab size: {self.config.vocab_size if hasattr(self.config, 'vocab_size') else 'unknown'}
  Device: {self.device}
  Has tokenizer: {self.tokenizer is not None}
"""
        return base_summary + config_info


def create_from_pretrained(
    model_name: str,
    model_id: Optional[str] = None,
    version: str = "1.0.0",
    load_tokenizer: bool = True,
    **kwargs
) -> TransformerModel:
    """
    Create TransformerModel from pretrained HF model.

    Args:
        model_name: HF model name (e.g., 'bert-base-uncased')
        model_id: Platform model ID
        version: Version
        load_tokenizer: Whether to load tokenizer
        **kwargs: Additional TransformerModel parameters

    Returns:
        TransformerModel instance
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers library not installed")

    # Load model
    hf_model = AutoModel.from_pretrained(model_name)

    # Load tokenizer if requested
    tokenizer = None
    if load_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            print(f"Warning: Could not load tokenizer for {model_name}")

    return TransformerModel(
        hf_model=hf_model,
        tokenizer=tokenizer,
        model_id=model_id or f"hf_{model_name.replace('/', '_')}",
        version=version,
        description=f"Hugging Face {model_name}",
        **kwargs
    )
