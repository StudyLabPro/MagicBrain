from __future__ import annotations
import numpy as np
import json
import time
from .brain import TextBrain

# Bumped whenever the on-disk layout changes. Older files (without a
# ``format_version`` key) are still readable — see ``load_model``.
FORMAT_VERSION = 2


def save_model(brain: TextBrain, stoi: dict, itos: dict, path: str):
    """Save the brain state, genome, topology and vocab to a compressed npz file.

    The graph is stored explicitly (``src``/``dst``/``delay``/``is_inhib``/``pos``)
    because structural plasticity rewires edges during training: rebuilding the
    graph from the genome alone reproduces the *initial* topology, not the trained
    one, and the saved weight vector would then be attached to the wrong synapses.
    """
    data = {
        "genome_str": str(brain.genome_str),
        "vocab": json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}),
        "w_slow": brain.w_slow,
        "w_fast": brain.w_fast,
        "R": brain.R,
        "b": brain.b,
        "theta": brain.theta,
        # Topology — mutated by prune/rewire, therefore not derivable from the genome.
        "src": brain.src,
        "dst": brain.dst,
        "delay": brain.delay,
        "is_inhib": brain.is_inhib,
        "pos": brain.pos,
        "meta": json.dumps({
            "format_version": FORMAT_VERSION,
            "step": brain.step,
            "timestamp": time.time(),
            "N": brain.N,
            "K": brain.K,
            "vocab_size": int(brain.vocab_size),
            "loss_ema": float(brain.loss_ema),
            "dopamine": float(brain.dopamine),
            # The forward pass injects noise from ``brain.rng``; without the
            # generator state a reloaded model is only statistically, not
            # bit-wise, the model that was saved.
            "rng_state": _rng_state_to_json(brain.rng),
        })
    }
    np.savez_compressed(path, **data)


def load_model(path: str) -> tuple[TextBrain, dict, dict, dict]:
    """
    Loads a brain from npz.

    Returns:
        Tuple of (brain, stoi, itos, metadata)
        - brain: Restored TextBrain instance
        - stoi: String to int vocab mapping
        - itos: Int to string vocab mapping
        - metadata: Model metadata dict (genome, vocab_size, step, N, K, timestamp,
          format_version, topology_restored)
    """
    data = np.load(path)

    genome_str = str(data["genome_str"])
    vocab_data = json.loads(str(data["vocab"]))
    stoi = vocab_data["stoi"]
    itos = {int(k): v for k, v in vocab_data["itos"].items()}

    brain = TextBrain(genome_str, len(stoi))

    brain.w_slow = data["w_slow"]
    brain.w_fast = data["w_fast"]
    brain.R = data["R"]
    brain.b = data["b"]
    brain.theta = data["theta"]

    topology_restored = _restore_topology(brain, data)

    # Extract metadata
    metadata = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))
        brain.step = meta.get("step", 0)
        if "loss_ema" in meta:
            brain.loss_ema = float(meta["loss_ema"])
        if "dopamine" in meta:
            brain.dopamine = float(meta["dopamine"])
        if meta.get("rng_state"):
            _rng_state_from_json(brain.rng, meta["rng_state"])
        metadata.update(meta)

    # Add genome and vocab info to metadata
    metadata["genome_str"] = genome_str
    metadata["vocab_size"] = len(stoi)
    metadata.setdefault("format_version", 1)
    metadata["topology_restored"] = topology_restored

    return brain, stoi, itos, metadata


def _rng_state_to_json(rng) -> dict:
    """Serialise a numpy Generator state; big integers become decimal strings."""
    state = rng.bit_generator.state
    inner = state.get("state", {})
    return {
        "bit_generator": state.get("bit_generator"),
        "state": {k: str(v) for k, v in inner.items()},
        "has_uint32": int(state.get("has_uint32", 0)),
        "uinteger": int(state.get("uinteger", 0)),
    }


def _rng_state_from_json(rng, payload: dict) -> bool:
    """Restore a serialised generator state; return False when incompatible."""
    try:
        if payload.get("bit_generator") != rng.bit_generator.state.get("bit_generator"):
            return False
        rng.bit_generator.state = {
            "bit_generator": payload["bit_generator"],
            "state": {k: int(v) for k, v in payload["state"].items()},
            "has_uint32": int(payload.get("has_uint32", 0)),
            "uinteger": int(payload.get("uinteger", 0)),
        }
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _restore_topology(brain: TextBrain, data) -> bool:
    """Restore the saved graph onto ``brain``; return True when it was applied.

    Files written before ``FORMAT_VERSION`` 2 carry no topology: for those the
    genome-derived graph is kept, which is only correct while no rewiring has
    happened. A shape mismatch is treated the same way as a missing array.
    """
    keys = ("src", "dst", "delay", "is_inhib", "pos")
    if any(k not in data for k in keys):
        return False

    src = data["src"]
    if src.shape != brain.src.shape:
        return False
    if data["is_inhib"].shape != brain.is_inhib.shape:
        return False

    brain.src = src.astype(np.int32, copy=False)
    brain.dst = data["dst"].astype(np.int32, copy=False)
    brain.delay = data["delay"].astype(np.int32, copy=False)
    brain.is_inhib = data["is_inhib"].astype(np.bool_, copy=False)
    brain.pos = data["pos"]

    idx_by_delay = [np.array([], dtype=np.int32)]
    for d in range(1, 6):
        idx_by_delay.append(np.where(brain.delay == d)[0].astype(np.int32))
    brain.idx_by_delay = idx_by_delay
    return True
