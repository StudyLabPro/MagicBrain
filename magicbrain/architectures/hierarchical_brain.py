"""
Hierarchical Spiking Neural Network Architecture.

Multi-layer SNNs with different timescales for temporal hierarchy.
Lower layers: fast dynamics, sensory processing
Upper layers: slow dynamics, abstract representations
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from ..brain import TextBrain


class HierarchicalBrain:
    """
    Multi-layer hierarchical spiking network.

    Architecture:
        Input → Layer1 (fast) → Layer2 (medium) → Layer3 (slow) → Output

    Each layer:
    - Has its own TextBrain with different timescales
    - Receives input from previous layer
    - Can have skip connections
    """

    def __init__(
        self,
        genomes: List[str],
        vocab_size: int,
        layer_sizes: Optional[List[int]] = None,
        timescale_factors: Optional[List[float]] = None,
        skip_connections: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            genomes: List of genome strings for each layer
            vocab_size: Input vocabulary size
            layer_sizes: Optional list of layer sizes (if None, use genome defaults)
            timescale_factors: Speed factors for each layer (1.0 = normal, 2.0 = 2x slower)
            skip_connections: Whether to add skip connections
            seed: Random seed
        """
        self.num_layers = len(genomes)
        self.vocab_size = vocab_size
        self.skip_connections = skip_connections

        if timescale_factors is None:
            # Default: layers get progressively slower
            timescale_factors = [1.0, 2.0, 4.0][:self.num_layers]

        self.timescale_factors = timescale_factors

        # Create layers
        self.layers: List[TextBrain] = []
        for i, genome in enumerate(genomes):
            # For first layer, use vocab_size
            # For others, use previous layer's N (neuron count)
            if i == 0:
                input_size = vocab_size
            else:
                input_size = self.layers[-1].N

            layer = TextBrain(genome, input_size, seed_override=seed + i)

            # Adjust timescales
            if timescale_factors[i] != 1.0:
                self._adjust_timescales(layer, timescale_factors[i])

            self.layers.append(layer)

        # Output layer maps last layer to vocab
        self.output_layer = np.random.normal(
            0, 0.12, size=(self.layers[-1].N, vocab_size)
        ).astype(np.float32)
        self.output_bias = np.zeros(vocab_size, dtype=np.float32)

        # Skip connection weights (if enabled)
        self.skip_weights = []
        if skip_connections and self.num_layers > 2:
            for i in range(self.num_layers - 1):
                skip_w = np.random.normal(
                    0, 0.05, size=(self.layers[i].N, self.layers[-1].N)
                ).astype(np.float32)
                self.skip_weights.append(skip_w)

        self.step = 0

    def _adjust_timescales(self, brain: TextBrain, factor: float):
        """Adjust brain timescales by factor."""
        # Slower trace decay (longer memory)
        brain.p["trace_fast_decay"] = 1.0 - (1.0 - brain.p["trace_fast_decay"]) / factor
        brain.p["trace_slow_decay"] = 1.0 - (1.0 - brain.p["trace_slow_decay"]) / factor

        # Slower buffer decay
        brain.p["buf_decay"] = 1.0 - (1.0 - brain.p["buf_decay"]) / factor

    def forward(self, token_id: int) -> np.ndarray:
        """
        Forward pass through hierarchy.

        Args:
            token_id: Input token

        Returns:
            Output probabilities
        """
        # Layer 1: process input token
        layer_outputs = []
        probs_l1 = self.layers[0].forward(token_id)
        layer_outputs.append(self.layers[0].compute_state())

        # Subsequent layers: process previous layer's activity
        for i in range(1, self.num_layers):
            # Use previous layer's state as "input"
            # We'll use the state as a sparse input pattern
            prev_state = layer_outputs[-1]

            # Map to current layer's input space
            # Simple approach: take top-k active neurons from previous layer
            k_input = min(20, len(prev_state))
            top_indices = np.argpartition(prev_state, -k_input)[-k_input:]

            # Create pseudo "token" by hashing active pattern
            # In practice, we forward with a synthetic input
            pseudo_token = int(np.sum(top_indices) % self.layers[i].vocab_size)

            probs = self.layers[i].forward(pseudo_token)
            layer_outputs.append(self.layers[i].compute_state())

        # Final output: combine last layer + optional skip connections
        final_state = layer_outputs[-1]

        if self.skip_connections and len(self.skip_weights) > 0:
            # Add skip contributions
            for i, skip_w in enumerate(self.skip_weights):
                skip_contrib = layer_outputs[i] @ skip_w
                final_state = final_state + 0.1 * skip_contrib

        # Output layer
        logits = final_state @ self.output_layer + self.output_bias
        logits = np.clip(logits, -10, 10)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)

        return probs.astype(np.float32)

    def learn(self, target_id: int, probs: np.ndarray) -> float:
        """
        Learning step for hierarchical network.

        Args:
            target_id: Target token
            probs: Predicted probabilities

        Returns:
            Loss
        """
        loss = -np.log(np.clip(probs[target_id], 1e-8, 1.0))

        # Backprop through output layer
        grad = probs.copy()
        grad[target_id] -= 1.0

        final_state = self.layers[-1].compute_state()
        lr = 0.001

        # Update output layer
        output_grad = np.outer(final_state, grad)
        self.output_layer -= lr * output_grad
        self.output_bias -= lr * grad

        # Train each layer with its local loss
        # Layer-wise learning (simplified)
        for i, layer in enumerate(self.layers):
            # Each layer learns based on local prediction
            # (In full implementation, would use error signals from above)
            layer_probs = np.ones(layer.vocab_size) / layer.vocab_size  # Placeholder
            layer.learn(0, layer_probs)  # Simplified

        self.step += 1
        return float(loss)

    def reset_state(self):
        """Reset state of all layers."""
        for layer in self.layers:
            layer.reset_state()

    def get_layer_states(self) -> List[np.ndarray]:
        """Get current state of each layer."""
        return [layer.compute_state() for layer in self.layers]

    def get_layer_activities(self) -> List[float]:
        """Get firing rate of each layer."""
        return [float(np.mean(layer.a)) for layer in self.layers]


class ModularBrain:
    """
    Modular architecture with specialized subnetworks.

    Modules:
    - Sensory: Input processing
    - Memory: Context maintenance
    - Action: Output generation
    - Controller: Coordinates modules
    """

    def __init__(
        self,
        genome_sensory: str,
        genome_memory: str,
        genome_action: str,
        genome_controller: str,
        vocab_size: int,
        seed: int = 42,
    ):
        """
        Args:
            genome_sensory: Genome for sensory module
            genome_memory: Genome for memory module
            genome_action: Genome for action module
            genome_controller: Genome for controller
            vocab_size: Vocabulary size
            seed: Random seed
        """
        self.vocab_size = vocab_size

        # Create modules
        self.sensory = TextBrain(genome_sensory, vocab_size, seed_override=seed)
        self.memory = TextBrain(genome_memory, vocab_size, seed_override=seed + 1)
        self.action = TextBrain(genome_action, vocab_size, seed_override=seed + 2)
        self.controller = TextBrain(genome_controller, vocab_size, seed_override=seed + 3)

        # Inter-module connections
        self._init_connections()

        self.step = 0

    def _init_connections(self):
        """Initialize inter-module connections."""
        # Sensory → Memory
        self.w_sens_mem = np.random.normal(
            0, 0.05, size=(self.sensory.N, self.memory.N)
        ).astype(np.float32)

        # Memory → Action
        self.w_mem_act = np.random.normal(
            0, 0.05, size=(self.memory.N, self.action.N)
        ).astype(np.float32)

        # Controller influences all modules
        self.w_ctrl_sens = np.random.normal(
            0, 0.03, size=(self.controller.N, self.sensory.N)
        ).astype(np.float32)

        self.w_ctrl_mem = np.random.normal(
            0, 0.03, size=(self.controller.N, self.memory.N)
        ).astype(np.float32)

        self.w_ctrl_act = np.random.normal(
            0, 0.03, size=(self.controller.N, self.action.N)
        ).astype(np.float32)

    def forward(self, token_id: int) -> np.ndarray:
        """Forward pass through modular architecture."""
        # Sensory processing
        self.sensory.forward(token_id)
        sens_state = self.sensory.compute_state()

        # Controller processes sensory summary
        controller_input = int(np.sum(sens_state * np.arange(len(sens_state))) % self.vocab_size)
        self.controller.forward(controller_input)
        ctrl_state = self.controller.compute_state()

        # Memory module: receives sensory + controller modulation
        mem_input = sens_state @ self.w_sens_mem + ctrl_state @ self.w_ctrl_mem
        mem_token = int(np.argmax(mem_input) % self.vocab_size)
        self.memory.forward(mem_token)
        mem_state = self.memory.compute_state()

        # Action module: receives memory + controller modulation
        act_input = mem_state @ self.w_mem_act + ctrl_state @ self.w_ctrl_act
        act_token = int(np.argmax(act_input) % self.vocab_size)
        probs = self.action.forward(act_token)

        return probs

    def learn(self, target_id: int, probs: np.ndarray) -> float:
        """Learning step."""
        loss = -np.log(np.clip(probs[target_id], 1e-8, 1.0))

        # Train action module (main loss)
        self.action.learn(target_id, probs)

        # Train other modules with local objectives
        # (Simplified - full version would use error signals)
        dummy_probs = np.ones(self.vocab_size) / self.vocab_size

        self.sensory.learn(0, dummy_probs)
        self.memory.learn(0, dummy_probs)
        self.controller.learn(0, dummy_probs)

        self.step += 1
        return float(loss)
