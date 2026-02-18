"""Genome decoding: base-4 string to spiking neural network hyperparameters.

A genome is a compact string over the alphabet {0, 1, 2, 3} (base-4 digits).
Each contiguous slice of 1-4 digits encodes one hyperparameter via positional
base-4 interpretation.  The mapping is deterministic: one genome string
uniquely defines one SNN architecture and its training dynamics.

Encoding schema (minimum 24 positions)::

    Position  Digits  Parameter           Range
    --------  ------  ------------------  ----------------------------
    0-1       2       N (neuron count)    256 + 64 * b4  -> [256, 1216]
    2         1       K (connectivity)    8 + 4 * b4     -> [8, 20]
    3         1       p_long              0.02 + 0.02*b4 -> [0.02, 0.08]
    4         1       lr (learning rate)  5e-4 + 5e-4*b4 -> [5e-4, 2e-3]
    5         1       k_active fraction   0.04 + 0.01*b4 -> [0.04, 0.07] of N
    6         1       trace_fast_decay    0.92 + 0.02*b4 -> [0.92, 0.98]
    7         1       homeo (threshold)   1e-3 + 1e-3*b4 -> [1e-3, 4e-3]
    8-11      4       seed                b4(8,4)        -> [0, 255]
    10        1       trace_slow_decay    0.985+0.003*b4 -> [0.985, 0.994]
    12        1       buf_decay           0.92 + 0.02*b4 -> [0.92, 0.98]
    13        1       alpha (fast mix)    0.25 + 0.15*b4 -> [0.25, 0.70]
    14        1       beta (slow mix)     0.05 + 0.05*b4 -> [0.05, 0.20]
    15        1       p_inhib             0.10 + 0.05*b4 -> [0.10, 0.25]
    16        1       dopamine_gain       0.8 + 0.4*b4   -> [0.8, 2.0]
    17        1       dopamine_bias       -0.2 + 0.2*b4  -> [-0.2, 0.4]
    18        1       cons_eps            5e-4 + 5e-4*b4 -> [5e-4, 2e-3]
    19        1       w_fast_decay        0.999+3e-4*b4  -> [0.999, 1.0]
    20        1       prune_every         800 + 200*b4   -> [800, 1400]
    21        1       prune_frac          0.02 + 0.01*b4 -> [0.02, 0.05]
    22        1       rewire_frac         0.50 + 0.10*b4 -> [0.50, 0.80]

Genomes shorter than 24 characters are zero-padded.  Positions wrap
cyclically for genomes of any length.
"""

from __future__ import annotations
import numpy as np

def decode_genome(genome: str) -> dict:
    """Decode a base-4 genome string into a dictionary of SNN hyperparameters.

    Args:
        genome: String of characters in {'0','1','2','3'}, minimum 24 chars.
            Characters outside this set are silently dropped.

    Returns:
        Dictionary mapping parameter names to numeric values.  Keys:
        N, K, p_long, lr, k_active, trace_fast_decay, trace_slow_decay,
        homeo, buf_decay, seed, alpha, beta, p_inhib, dopamine_gain,
        dopamine_bias, cons_eps, w_fast_decay, prune_every, prune_frac,
        rewire_frac.
    """
    g = np.array([ord(c) - 48 for c in genome if c in "0123"], dtype=np.int32)
    if len(g) < 24:
        g = np.pad(g, (0, 24 - len(g)), constant_values=1)

    def b4(i: int, n: int) -> int:
        x = 0
        for k in range(n):
            x = x * 4 + int(g[(i + k) % len(g)])
        return x

    N = 256 + 64 * b4(0, 2)
    K = 8 + b4(2, 1) * 4
    p_long = 0.02 + 0.02 * b4(3, 1)

    lr = 0.0005 + 0.0005 * b4(4, 1)

    k_active = max(48, int(N * (0.04 + 0.01 * b4(5, 1))))

    trace_fast_decay = 0.92 + 0.02 * b4(6, 1)
    trace_slow_decay = 0.985 + 0.003 * b4(10, 1)

    homeo = 0.001 + 0.001 * b4(7, 1)
    buf_decay = 0.92 + 0.02 * b4(12, 1)

    seed = b4(8, 4)

    alpha = 0.25 + 0.15 * b4(13, 1)
    beta = 0.05 + 0.05 * b4(14, 1)

    p_inhib = 0.10 + 0.05 * b4(15, 1)

    dopamine_gain = 0.8 + 0.4 * b4(16, 1)
    dopamine_bias = -0.2 + 0.2 * b4(17, 1)

    cons_eps = 0.0005 + 0.0005 * b4(18, 1)
    w_fast_decay = 0.9990 + 0.0003 * b4(19, 1)

    prune_every = 800 + 200 * b4(20, 1)
    prune_frac = 0.02 + 0.01 * b4(21, 1)
    rewire_frac = 0.50 + 0.10 * b4(22, 1)

    return dict(
        N=N,
        K=K,
        p_long=p_long,
        lr=lr,
        k_active=k_active,
        trace_fast_decay=trace_fast_decay,
        trace_slow_decay=trace_slow_decay,
        homeo=homeo,
        buf_decay=buf_decay,
        seed=seed,
        alpha=alpha,
        beta=beta,
        p_inhib=p_inhib,
        dopamine_gain=dopamine_gain,
        dopamine_bias=dopamine_bias,
        cons_eps=cons_eps,
        w_fast_decay=w_fast_decay,
        prune_every=prune_every,
        prune_frac=prune_frac,
        rewire_frac=rewire_frac,
    )
