"""Compositional Pattern-Producing Network (CPPN).

Generates synaptic weights as a function of neuron spatial coordinates.
Instead of storing weights directly, the CPPN is a compact program
that produces them: w_ij = CPPN(pos_i, pos_j, dist_ij, type_i, type_j).

The CPPN is parameterized by the genome, making it a developmental operator
that "grows" the weight structure from a compact representation.

Reference: Stanley, K. O. (2007). Compositional pattern producing networks.
"""

from __future__ import annotations

import numpy as np


# Basis functions for CPPN nodes
def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def _cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def _gaussian(x: np.ndarray) -> np.ndarray:
    return np.exp(-x * x * 0.5)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _step(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


BASIS_FUNCTIONS = [_sin, _cos, _gaussian, _sigmoid, _tanh, _abs, _identity, _step]
BASIS_NAMES = ["sin", "cos", "gaussian", "sigmoid", "tanh", "abs", "identity", "step"]


class CPPNLayer:
    """Single layer of a CPPN."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_ids: list[int],
        rng: np.random.Generator,
        weight_scale: float = 1.0,
    ):
        self.weights = (
            rng.normal(0, weight_scale, size=(input_dim, output_dim)).astype(np.float32)
        )
        self.bias = rng.normal(0, 0.1, size=output_dim).astype(np.float32)
        self.activations = [
            BASIS_FUNCTIONS[aid % len(BASIS_FUNCTIONS)] for aid in activation_ids
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: linear transform + per-neuron activation."""
        z = (x @ self.weights + self.bias).astype(np.float32)
        out = np.empty_like(z)
        for i, act in enumerate(self.activations):
            out[:, i] = act(z[:, i])
        return out


class CPPN:
    """Compositional Pattern-Producing Network.

    Maps spatial coordinates to synaptic weights:
        (x1,y1,z1, x2,y2,z2, dist, type_src, type_dst) -> weight

    Architecture is determined by genome parameters.
    """

    # Input features: pos_src(3) + pos_dst(3) + distance(1) + types(2) = 9
    INPUT_DIM = 9

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation_pattern: list[int] | None = None,
        seed: int = 42,
        weight_scale: float = 0.5,
        output_scale: float = 0.1,
    ):
        """Initialize CPPN.

        Args:
            hidden_dims: Sizes of hidden layers. Default [16, 8].
            activation_pattern: Base-4 codes for activation functions
                per hidden neuron. Maps to basis functions.
            seed: Random seed for weight initialization.
            weight_scale: Scale of random weight initialization.
            output_scale: Scale multiplier for output weights.
        """
        self.rng = np.random.default_rng(seed)
        self.output_scale = output_scale

        if hidden_dims is None:
            hidden_dims = [16, 8]

        # Build layers
        self.layers: list[CPPNLayer] = []
        dims = [self.INPUT_DIM] + hidden_dims + [1]

        act_idx = 0
        for i in range(len(dims) - 1):
            in_d = dims[i]
            out_d = dims[i + 1]

            # Determine activation functions per output neuron
            acts = []
            for j in range(out_d):
                if activation_pattern and act_idx < len(activation_pattern):
                    acts.append(activation_pattern[act_idx])
                    act_idx += 1
                else:
                    acts.append(self.rng.integers(0, len(BASIS_FUNCTIONS)))
                    act_idx += 1

            # Last layer uses tanh for bounded output
            if i == len(dims) - 2:
                acts = [4]  # tanh

            scale = weight_scale if i < len(dims) - 2 else weight_scale * 0.5
            self.layers.append(CPPNLayer(in_d, out_d, acts, self.rng, scale))

    def query(
        self,
        pos_src: np.ndarray,
        pos_dst: np.ndarray,
        dist: np.ndarray,
        type_src: np.ndarray,
        type_dst: np.ndarray,
    ) -> np.ndarray:
        """Compute weights for a batch of connections.

        Args:
            pos_src: Source neuron 3D positions (E, 3).
            pos_dst: Destination neuron positions (E, 3).
            dist: Euclidean distances (E,).
            type_src: Source neuron type (0=excitatory, 1=inhibitory) (E,).
            type_dst: Destination neuron type (E,).

        Returns:
            Weight values (E,).
        """
        # Build input features (E, 9)
        features = np.column_stack([
            pos_src,                          # 3D position of source
            pos_dst,                          # 3D position of destination
            dist.reshape(-1, 1),              # distance
            type_src.reshape(-1, 1),          # type source
            type_dst.reshape(-1, 1),          # type destination
        ]).astype(np.float32)

        # Forward through layers
        x = features
        for layer in self.layers:
            x = layer.forward(x)

        return (x.ravel() * self.output_scale).astype(np.float32)

    def generate_weights(
        self,
        pos: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        is_inhib: np.ndarray,
    ) -> np.ndarray:
        """Generate all synaptic weights for a network graph.

        Args:
            pos: Neuron 3D positions (N, 3).
            src: Source indices (E,).
            dst: Destination indices (E,).
            is_inhib: Inhibitory neuron mask (N,).

        Returns:
            Weight array (E,), respecting E/I signs.
        """
        pos_src = pos[src]
        pos_dst = pos[dst]
        dist = np.linalg.norm(pos_src - pos_dst, axis=1).astype(np.float32)
        type_src = is_inhib[src].astype(np.float32)
        type_dst = is_inhib[dst].astype(np.float32)

        weights = self.query(pos_src, pos_dst, dist, type_src, type_dst)

        # Enforce E/I constraint
        inhib_mask = is_inhib[src]
        weights[inhib_mask] = -np.abs(weights[inhib_mask])

        return weights

    @classmethod
    def from_genome_params(
        cls,
        genome_digits: list[int],
        seed: int = 42,
    ) -> CPPN:
        """Create a CPPN from genome-encoded parameters.

        Args:
            genome_digits: List of base-4 integers that encode CPPN structure.
                Expected layout:
                  [0]:    number of hidden layers (1-3, mapped from 0-3)
                  [1-2]:  first hidden layer width (4 + 4*val, range 4-16)
                  [3-4]:  second hidden layer width (4 + 4*val, range 4-16)
                  [5+]:   activation function IDs for each neuron (mod 8)
            seed: Random seed.

        Returns:
            Configured CPPN instance.
        """
        if not genome_digits:
            return cls(seed=seed)

        n_layers = min(3, max(1, genome_digits[0] + 1))

        hidden_dims = []
        for i in range(n_layers):
            idx = 1 + i * 2
            if idx + 1 < len(genome_digits):
                width = 4 + 4 * (genome_digits[idx] * 4 + genome_digits[idx + 1])
                width = min(32, max(4, width))
            else:
                width = 8
            hidden_dims.append(width)

        # Remaining digits encode activation patterns
        act_start = 1 + n_layers * 2
        activation_pattern = genome_digits[act_start:] if act_start < len(genome_digits) else None

        return cls(
            hidden_dims=hidden_dims,
            activation_pattern=activation_pattern,
            seed=seed,
        )
