from __future__ import annotations
import numpy as np
import json
import time
from .brain import TextBrain

def save_model(brain: TextBrain, stoi: dict, itos: dict, path: str):
    """
    Saves the brain state, genome, and vocab to a compressed npz file.
    """
    data = {
        "genome_str": str(brain.genome_str),
        "vocab": json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}),
        "w_slow": brain.w_slow,
        "w_fast": brain.w_fast,
        "R": brain.R,
        "b": brain.b,
        "theta": brain.theta,
        "meta": json.dumps({
            "step": brain.step,
            "timestamp": time.time(),
            "N": brain.N,
            "K": brain.K
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
        - metadata: Model metadata dict (genome, vocab_size, step, N, K, timestamp)
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

    # Extract metadata
    metadata = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))
        brain.step = meta.get("step", 0)
        metadata.update(meta)

    # Add genome and vocab info to metadata
    metadata["genome_str"] = genome_str
    metadata["vocab_size"] = len(stoi)

    return brain, stoi, itos, metadata
