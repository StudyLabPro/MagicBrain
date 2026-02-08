"""
Model Orchestrator - Manage multi-model execution.

Supports various execution strategies for heterogeneous model graphs.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import threading
from collections import defaultdict, deque

from ..model_interface import ModelInterface, OutputType
from ..communication import Message, MessageBus, ConverterRegistry
from ..registry import ModelRegistry


class ExecutionStrategy(Enum):
    """Execution strategies for model orchestration."""
    SEQUENTIAL = "sequential"          # Models execute one after another
    PARALLEL = "parallel"              # Models execute simultaneously
    PIPELINE = "pipeline"              # Data flows through stages
    HIERARCHICAL = "hierarchical"      # Supervisor-worker pattern
    FEEDBACK = "feedback"              # Models communicate iteratively
    CASCADED = "cascaded"             # Fast model → accurate model
    MIXTURE_OF_EXPERTS = "moe"        # Router → experts


class OrchestratorError(Exception):
    """Raised when orchestration fails."""
    pass


@dataclass
class ExecutionResult:
    """
    Result of model orchestration execution.

    Contains outputs from all models and execution metadata.
    """
    outputs: Dict[str, Any]                    # model_id -> output
    execution_time_ms: float
    strategy: ExecutionStrategy
    models_executed: List[str]
    success: bool = True
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_final_output(self) -> Any:
        """Get final output (from last model executed)."""
        if not self.models_executed:
            return None
        return self.outputs.get(self.models_executed[-1])

    def get_output(self, model_id: str) -> Optional[Any]:
        """Get output from specific model."""
        return self.outputs.get(model_id)


class ModelNode:
    """
    Node in the model execution graph.

    Represents a model and its connections.
    """

    def __init__(self, model_id: str, model: ModelInterface):
        """
        Initialize model node.

        Args:
            model_id: Unique model ID
            model: Model instance
        """
        self.model_id = model_id
        self.model = model

        # Graph connections
        self.inputs: Set[str] = set()          # Input from these models
        self.outputs: Set[str] = set()         # Output to these models

        # Type converters
        self.converters: Dict[str, Callable] = {}  # target_id -> converter

        # Execution state
        self.last_output: Optional[Any] = None
        self.execution_count: int = 0
        self.total_time_ms: float = 0.0


class ModelOrchestrator:
    """
    Orchestrates execution of multiple heterogeneous models.

    Features:
    - Multiple execution strategies
    - Automatic type conversion
    - Error handling and fallbacks
    - State synchronization
    - Message-based communication
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        message_bus: Optional[MessageBus] = None,
        converter_registry: Optional[ConverterRegistry] = None
    ):
        """
        Initialize orchestrator.

        Args:
            registry: Model registry (creates new if None)
            message_bus: Message bus (creates new if None)
            converter_registry: Type converter registry (creates new if None)
        """
        self.registry = registry or ModelRegistry()
        self.message_bus = message_bus or MessageBus()
        self.converter_registry = converter_registry or ConverterRegistry()

        # Model graph
        self._nodes: Dict[str, ModelNode] = {}

        # Execution state
        self._last_execution: Optional[ExecutionResult] = None

        # Thread safety
        self._lock = threading.RLock()

        # Execution strategy handlers
        self._strategy_handlers = {
            ExecutionStrategy.SEQUENTIAL: self._execute_sequential,
            ExecutionStrategy.PARALLEL: self._execute_parallel,
            ExecutionStrategy.PIPELINE: self._execute_pipeline,
        }

    def add_model(
        self,
        model: ModelInterface,
        model_id: Optional[str] = None
    ) -> str:
        """
        Add a model to the orchestrator.

        Args:
            model: Model instance
            model_id: Unique model ID (uses model.metadata.model_id if None)

        Returns:
            Model ID

        Raises:
            ValueError: If model_id already exists
        """
        with self._lock:
            model_id = model_id or model.get_metadata().model_id

            if model_id in self._nodes:
                raise ValueError(f"Model {model_id} already exists")

            # Create node
            node = ModelNode(model_id, model)
            self._nodes[model_id] = node

            # Register in registry only if not already registered
            try:
                self.registry.get(model_id)
            except:
                # Not registered, so register it
                self.registry.register(model, model_id=model_id)

            return model_id

    def connect(
        self,
        source: str,
        target: str,
        converter: Optional[Callable] = None
    ):
        """
        Connect two models in the execution graph.

        Args:
            source: Source model ID
            target: Target model ID
            converter: Optional custom converter function

        Raises:
            OrchestratorError: If models not found
        """
        with self._lock:
            if source not in self._nodes:
                raise OrchestratorError(f"Source model {source} not found")
            if target not in self._nodes:
                raise OrchestratorError(f"Target model {target} not found")

            # Add connection
            self._nodes[source].outputs.add(target)
            self._nodes[target].inputs.add(source)

            # Set converter
            if converter:
                self._nodes[source].converters[target] = converter
            else:
                # Auto-select converter based on output types
                source_type = self._nodes[source].model.get_output_type()
                target_model = self._nodes[target].model
                # We'll apply converter during execution
                pass

    def disconnect(self, source: str, target: str):
        """
        Disconnect two models.

        Args:
            source: Source model ID
            target: Target model ID
        """
        with self._lock:
            if source in self._nodes:
                self._nodes[source].outputs.discard(target)
                self._nodes[source].converters.pop(target, None)

            if target in self._nodes:
                self._nodes[target].inputs.discard(source)

    def remove_model(self, model_id: str):
        """
        Remove a model from orchestrator.

        Args:
            model_id: Model ID

        Raises:
            OrchestratorError: If model not found
        """
        with self._lock:
            if model_id not in self._nodes:
                raise OrchestratorError(f"Model {model_id} not found")

            node = self._nodes[model_id]

            # Disconnect from all other models
            for input_id in list(node.inputs):
                self.disconnect(input_id, model_id)

            for output_id in list(node.outputs):
                self.disconnect(model_id, output_id)

            # Remove node
            del self._nodes[model_id]

    def execute(
        self,
        input_data: Any,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        entry_model: Optional[str] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute the model graph.

        Args:
            input_data: Input data for first model
            strategy: Execution strategy
            entry_model: Entry model ID (auto-detect if None)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Execution result

        Raises:
            OrchestratorError: If execution fails
        """
        start_time = datetime.now()

        try:
            # Get execution handler
            handler = self._strategy_handlers.get(strategy)
            if not handler:
                raise OrchestratorError(f"Strategy {strategy} not implemented")

            # Execute
            outputs, models_executed = handler(input_data, entry_model, **kwargs)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Create result
            result = ExecutionResult(
                outputs=outputs,
                execution_time_ms=execution_time,
                strategy=strategy,
                models_executed=models_executed,
                success=True,
            )

            self._last_execution = result
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result = ExecutionResult(
                outputs={},
                execution_time_ms=execution_time,
                strategy=strategy,
                models_executed=[],
                success=False,
                error=e,
            )
            self._last_execution = result
            raise OrchestratorError(f"Execution failed: {e}") from e

    def _execute_sequential(
        self,
        input_data: Any,
        entry_model: Optional[str],
        **kwargs
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute models sequentially.

        Each model receives output from previous model.
        """
        with self._lock:
            if not self._nodes:
                raise OrchestratorError("No models in orchestrator")

            # Find execution order (topological sort)
            execution_order = self._topological_sort(entry_model)

            outputs = {}
            current_input = input_data

            for model_id in execution_order:
                node = self._nodes[model_id]

                # Execute model
                output = node.model.forward(current_input)
                outputs[model_id] = output
                node.last_output = output
                node.execution_count += 1

                # Convert for next model if needed
                if node.outputs:
                    next_model_id = list(node.outputs)[0]
                    if next_model_id in self._nodes:
                        current_input = self._convert_output(
                            output,
                            node.model.get_output_type(),
                            self._nodes[next_model_id].model.get_output_type()
                        )
                else:
                    current_input = output

            return outputs, execution_order

    def _execute_parallel(
        self,
        input_data: Any,
        entry_model: Optional[str],
        **kwargs
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute all models in parallel on same input.

        Returns aggregated outputs.
        """
        with self._lock:
            if not self._nodes:
                raise OrchestratorError("No models in orchestrator")

            outputs = {}
            models_executed = []

            # Execute all models in parallel (synchronously for now)
            # TODO: Implement true async parallel execution
            for model_id, node in self._nodes.items():
                output = node.model.forward(input_data)
                outputs[model_id] = output
                node.last_output = output
                node.execution_count += 1
                models_executed.append(model_id)

            return outputs, models_executed

    def _execute_pipeline(
        self,
        input_data: Any,
        entry_model: Optional[str],
        **kwargs
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute models in pipeline fashion.

        Data flows through stages with automatic conversion.
        """
        # Similar to sequential but with explicit stage boundaries
        return self._execute_sequential(input_data, entry_model, **kwargs)

    def _topological_sort(
        self,
        start_node: Optional[str] = None
    ) -> List[str]:
        """
        Topological sort of model graph.

        Args:
            start_node: Starting node (uses node with no inputs if None)

        Returns:
            Ordered list of model IDs
        """
        # Find start node if not specified
        if start_node is None:
            # Find node with no inputs
            for model_id, node in self._nodes.items():
                if not node.inputs:
                    start_node = model_id
                    break

            if start_node is None:
                # Just use first node if no clear entry point
                start_node = next(iter(self._nodes.keys()))

        # BFS traversal
        visited = set()
        order = []
        queue = deque([start_node])

        while queue:
            model_id = queue.popleft()

            if model_id in visited:
                continue

            visited.add(model_id)
            order.append(model_id)

            # Add connected models
            node = self._nodes[model_id]
            for next_id in node.outputs:
                if next_id not in visited:
                    queue.append(next_id)

        return order

    def _convert_output(
        self,
        data: Any,
        source_type: OutputType,
        target_type: OutputType
    ) -> Any:
        """
        Convert output between model types.

        Args:
            data: Output data
            source_type: Source output type
            target_type: Target output type

        Returns:
            Converted data
        """
        if source_type == target_type:
            return data

        return self.converter_registry.convert(data, source_type, target_type)

    def get_model(self, model_id: str) -> Optional[ModelInterface]:
        """
        Get model by ID.

        Args:
            model_id: Model ID

        Returns:
            Model instance or None
        """
        node = self._nodes.get(model_id)
        return node.model if node else None

    def list_models(self) -> List[str]:
        """
        List all model IDs.

        Returns:
            List of model IDs
        """
        return list(self._nodes.keys())

    def get_graph(self) -> Dict[str, Any]:
        """
        Get graph structure.

        Returns:
            Graph representation
        """
        return {
            model_id: {
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
                "execution_count": node.execution_count,
                "output_type": node.model.get_output_type().value,
            }
            for model_id, node in self._nodes.items()
        }

    def reset_state(self):
        """Reset execution state for all models."""
        with self._lock:
            for node in self._nodes.values():
                node.model.reset()
                node.last_output = None

    def get_last_execution(self) -> Optional[ExecutionResult]:
        """
        Get last execution result.

        Returns:
            Last execution result or None
        """
        return self._last_execution

    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_executions = sum(node.execution_count for node in self._nodes.values())

            return {
                "models_count": len(self._nodes),
                "total_executions": total_executions,
                "last_execution": self._last_execution.to_dict() if self._last_execution else None,
            }


# Global orchestrator instance
_global_orchestrator = ModelOrchestrator()


def get_global_orchestrator() -> ModelOrchestrator:
    """
    Get global orchestrator instance.

    Returns:
        Global orchestrator
    """
    return _global_orchestrator
