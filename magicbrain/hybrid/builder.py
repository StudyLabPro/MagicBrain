"""Compositional API for building hybrid architectures."""
from __future__ import annotations
from typing import Dict, List, Optional
from .base import HybridArchitecture, create_hybrid
from ..platform.model_interface import ModelInterface


class HybridBuilder:
    """
    Fluent API for building hybrid architectures.

    Example:
        builder = HybridBuilder()
        hybrid = (builder
            .add("snn", snn_model)
            .add("dnn", dnn_model)
            .connect("snn", "dnn")
            .set_output("dnn")
            .build("my_hybrid"))
    """

    def __init__(self):
        """Initialize builder."""
        self._components: Dict[str, ModelInterface] = {}
        self._connections: List[tuple[str, str]] = []
        self._output_component: Optional[str] = None

    def add(self, name: str, model: ModelInterface) -> 'HybridBuilder':
        """
        Add a component.

        Args:
            name: Component name
            model: Model instance

        Returns:
            Self for chaining
        """
        self._components[name] = model
        return self

    def connect(self, source: str, target: str) -> 'HybridBuilder':
        """
        Connect two components.

        Args:
            source: Source component name
            target: Target component name

        Returns:
            Self for chaining
        """
        self._connections.append((source, target))
        return self

    def set_output(self, component_name: str) -> 'HybridBuilder':
        """
        Set output component.

        Args:
            component_name: Name of output component

        Returns:
            Self for chaining
        """
        self._output_component = component_name
        return self

    def build(
        self,
        model_id: Optional[str] = None,
        **kwargs
    ) -> HybridArchitecture:
        """
        Build the hybrid architecture.

        Args:
            model_id: Model ID
            **kwargs: Additional parameters

        Returns:
            HybridArchitecture instance
        """
        if not self._components:
            raise ValueError("No components added")

        return create_hybrid(
            components=self._components,
            connections=self._connections,
            model_id=model_id,
            output_component=self._output_component,
            **kwargs
        )

    def reset(self) -> 'HybridBuilder':
        """Reset builder to empty state."""
        self._components.clear()
        self._connections.clear()
        self._output_component = None
        return self


# Template architectures
class Templates:
    """Pre-defined hybrid architecture templates."""

    @staticmethod
    def snn_dnn_pipeline(snn_model, dnn_model, model_id: Optional[str] = None):
        """SNN → DNN pipeline template."""
        return (HybridBuilder()
                .add("snn", snn_model)
                .add("dnn", dnn_model)
                .connect("snn", "dnn")
                .build(model_id))

    @staticmethod
    def encoder_decoder(encoder, decoder, model_id: Optional[str] = None):
        """Encoder → Decoder template."""
        return (HybridBuilder()
                .add("encoder", encoder)
                .add("decoder", decoder)
                .connect("encoder", "decoder")
                .build(model_id))

    @staticmethod
    def three_stage_pipeline(m1, m2, m3, model_id: Optional[str] = None):
        """Three-stage pipeline: M1 → M2 → M3."""
        return (HybridBuilder()
                .add("stage1", m1)
                .add("stage2", m2)
                .add("stage3", m3)
                .connect("stage1", "stage2")
                .connect("stage2", "stage3")
                .build(model_id))
