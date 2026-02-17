from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Optional, List
import numpy as np


class CheckpointManager:
    """Manages training checkpoints (weights + metadata) on disk."""

    def __init__(self, checkpoint_dir: str, save_every_rounds: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_rounds = save_every_rounds
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, round_num: int) -> bool:
        return self.save_every_rounds > 0 and (round_num % self.save_every_rounds == 0)

    def save(self, round_num: int, weights: dict, loss: float, metadata: Optional[dict] = None):
        """Save checkpoint as .npz (weights) + .json (metadata)."""
        prefix = f"checkpoint_round_{round_num:06d}"
        npz_path = self.checkpoint_dir / f"{prefix}.npz"
        json_path = self.checkpoint_dir / f"{prefix}.json"

        np.savez_compressed(str(npz_path), **weights)

        meta = {
            "round_num": round_num,
            "loss": loss,
            "timestamp": time.time(),
        }
        if metadata:
            meta.update(metadata)

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, round_num: int) -> tuple[dict, dict]:
        """Load a specific checkpoint by round number."""
        prefix = f"checkpoint_round_{round_num:06d}"
        npz_path = self.checkpoint_dir / f"{prefix}.npz"
        json_path = self.checkpoint_dir / f"{prefix}.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {npz_path}")

        data = np.load(str(npz_path))
        weights = {key: data[key] for key in data.files}

        metadata = {}
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)

        return weights, metadata

    def load_latest(self) -> tuple[dict, dict]:
        """Resume from the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        latest = max(checkpoints, key=lambda c: c["round_num"])
        return self.load(latest["round_num"])

    def load_best(self) -> tuple[dict, dict]:
        """Load the checkpoint with lowest validation loss."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        best = min(checkpoints, key=lambda c: c["loss"])
        return self.load(best["round_num"])

    def list_checkpoints(self) -> List[dict]:
        """List all checkpoints with their metadata."""
        results = []
        for json_path in sorted(self.checkpoint_dir.glob("checkpoint_round_*.json")):
            with open(json_path) as f:
                meta = json.load(f)
            results.append(meta)
        return results
