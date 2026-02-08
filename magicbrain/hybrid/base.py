"""
Base classes for hybrid model architectures.

Provides infrastructure for combining different model types.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np

from ..platform.model_interface import (
    HybridModel,
    ModelInterface,
    ModelMetadata,
    ModelType,
    OutputType,
)
from ..platform.communication import ConverterRegistry


@dataclass
class Component:
    """
    Component in a hybrid architecture.

    Represents a model and its connections.
    """
    name: str
    model: ModelInterface
    inputs: List[str]  # Names of input components
    outputs: List[str]  # Names of output components
    converter: Optional[Callable] = None  # Input converter

    def __repr__(self) -> str:
        return f"Component(name={self.name}, type={self.model.metadata.model_type})"


class HybridArchitecture(HybridModel):
    """
    Base class for hybrid architectures.

    Combines multiple models with automatic type conversion and data flow.
    """

    def __init__(
        self,
        components: Dict[str, ModelInterface],
        connections: List[tuple[str, str]],
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        output_component: Optional[str] = None,
    ):
        """
        Initialize hybrid architecture.

        Args:
            components: Dictionary of model components {name: model}
            connections: List of (source, target) connections
            model_id: Unique model ID
            version: Version
            description: Description
            output_component: Which component provides final output
        """
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id or f"hybrid_{id(self)}",
            model_type=ModelType.HYBRID,
            version=version,
            description=description or "Hybrid model architecture",
            output_type=self._infer_output_type(components, output_component),
        )

        super().__init__(components, metadata)

        # Build component graph
        self._components = self._build_component_graph(components, connections)
        self._output_component = output_component or self._find_output_component()

        # Converter registry for automatic type conversion
        self._converter_registry = ConverterRegistry()

        # Execution order (topological sort)
        self._execution_order = self._compute_execution_order()

        # Cache for intermediate outputs
        self._cache = {}

    def _build_component_graph(
        self,
        components: Dict[str, ModelInterface],
        connections: List[tuple[str, str]]
    ) -> Dict[str, Component]:
        """Build component graph from models and connections."""
        comp_dict = {}

        # Initialize components
        for name, model in components.items():
            comp_dict[name] = Component(
                name=name,
                model=model,
                inputs=[],
                outputs=[]
            )

        # Add connections
        for source, target in connections:
            if source not in comp_dict or target not in comp_dict:
                raise ValueError(f"Invalid connection: {source} -> {target}")

            comp_dict[source].outputs.append(target)
            comp_dict[target].inputs.append(source)

        return comp_dict

    def _find_output_component(self) -> str:
        """Find component that has no outputs (terminal node)."""
        for name, comp in self._components.items():
            if not comp.outputs:
                return name
        # If all have outputs, return last one
        return list(self._components.keys())[-1]

    def _infer_output_type(
        self,
        components: Dict[str, ModelInterface],
        output_component: Optional[str]
    ) -> OutputType:
        """Infer output type from output component."""
        if output_component and output_component in components:
            return components[output_component].get_output_type()
        # Return output type of last component
        return list(components.values())[-1].get_output_type()

    def _compute_execution_order(self) -> List[str]:
        """
        Compute execution order using topological sort.

        Returns:
            List of component names in execution order
        """
        # Find components with no inputs (entry points)
        in_degree = {name: len(comp.inputs) for name, comp in self._components.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]

        order = []
        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            order.append(current)

            # Reduce in-degree of downstream components
            for output_name in self._components[current].outputs:
                in_degree[output_name] -= 1
                if in_degree[output_name] == 0:
                    queue.append(output_name)

        if len(order) != len(self._components):
            raise ValueError("Cycle detected in hybrid architecture graph")

        return order

    def forward(self, input: Any, **kwargs) -> Any:
        """
        Forward pass through hybrid architecture.

        Args:
            input: Input data (goes to first component)
            **kwargs: Additional parameters

        Returns:
            Output from final component
        """
        self._cache.clear()

        # Execute components in order
        for comp_name in self._execution_order:
            comp = self._components[comp_name]

            # Get input for this component
            if not comp.inputs:
                # Entry component - use provided input
                comp_input = input
            elif len(comp.inputs) == 1:
                # Single input - get from cache
                source_name = comp.inputs[0]
                comp_input = self._cache[source_name]

                # Convert types if needed
                source_type = self._components[source_name].model.get_output_type()
                target_type = comp.model.get_output_type()

                if source_type != target_type:
                    comp_input = self._converter_registry.convert(
                        comp_input,
                        source_type,
                        OutputType.DENSE  # Use DENSE as intermediate
                    )
            else:
                # Multiple inputs - concatenate or aggregate
                inputs = [self._cache[src] for src in comp.inputs]
                comp_input = self._aggregate_inputs(inputs)

            # Forward through component
            output = comp.model.forward(comp_input, **kwargs)

            # Cache output
            self._cache[comp_name] = output

        # Return output from designated output component
        return self._cache[self._output_component]

    def _aggregate_inputs(self, inputs: List[Any]) -> Any:
        """
        Aggregate multiple inputs.

        Default: concatenate numpy arrays.

        Args:
            inputs: List of inputs

        Returns:
            Aggregated input
        """
        # Convert all to numpy
        arrays = []
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                arrays.append(inp)
            else:
                arrays.append(np.array(inp))

        # Concatenate along last axis
        return np.concatenate(arrays, axis=-1)

    def get_output_type(self) -> OutputType:
        """Get output type from final component."""
        return self._components[self._output_component].model.get_output_type()

    def get_component_output(self, component_name: str) -> Optional[Any]:
        """
        Get cached output from specific component.

        Args:
            component_name: Component name

        Returns:
            Cached output or None
        """
        return self._cache.get(component_name)

    def get_component_names(self) -> List[str]:
        """
        Get names of all components.

        Returns:
            List of component names
        """
        return list(self._components.keys())

    def get_execution_order(self) -> List[str]:
        """
        Get execution order of components.

        Returns:
            List of component names in execution order
        """
        return self._execution_order.copy()

    def visualize_graph(self) -> str:
        """
        Get text visualization of architecture graph.

        Returns:
            Graph visualization string
        """
        lines = ["Hybrid Architecture Graph:", ""]

        for comp_name in self._execution_order:
            comp = self._components[comp_name]
            model_type = comp.model.metadata.model_type.value

            # Inputs
            if comp.inputs:
                inputs_str = ", ".join(comp.inputs)
                lines.append(f"  [{inputs_str}]")
                lines.append(f"    ↓")

            # Component
            lines.append(f"  {comp_name} ({model_type})")

            # Outputs
            if comp.outputs:
                lines.append(f"    ↓")
                outputs_str = ", ".join(comp.outputs)
                lines.append(f"  [{outputs_str}]")
                lines.append("")

        # Mark output
        lines.append(f"Output: {self._output_component}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Get hybrid architecture summary."""
        base_summary = super().summary()

        components_info = "\n".join([
            f"  - {name}: {comp.model.metadata.model_type.value} "
            f"({comp.model.get_parameters_count():,} params)"
            for name, comp in self._components.items()
        ])

        return f"""{base_summary}

Hybrid Architecture:
  Components: {len(self._components)}
{components_info}

  Execution order: {' → '.join(self._execution_order)}
  Output component: {self._output_component}

Graph:
{self.visualize_graph()}
"""


def create_hybrid(
    components: Dict[str, ModelInterface],
    connections: List[tuple[str, str]],
    model_id: Optional[str] = None,
    **kwargs
) -> HybridArchitecture:
    """
    Create a hybrid architecture.

    Args:
        components: Dictionary of components {name: model}
        connections: List of (source, target) tuples
        model_id: Model ID
        **kwargs: Additional parameters

    Returns:
        HybridArchitecture instance
    """
    return HybridArchitecture(
        components=components,
        connections=connections,
        model_id=model_id,
        **kwargs
    )
