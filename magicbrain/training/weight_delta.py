from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional
import numpy as np


@dataclass
class WeightDelta:
    w_slow_delta: np.ndarray
    w_fast_delta: np.ndarray
    R_delta: np.ndarray
    b_delta: np.ndarray
    theta_delta: np.ndarray
    steps_completed: int
    avg_loss: float

    def compress(self, threshold: float) -> WeightDelta:
        """Zero out changes below threshold (sparse representation)."""
        return WeightDelta(
            w_slow_delta=np.where(np.abs(self.w_slow_delta) < threshold, 0.0, self.w_slow_delta),
            w_fast_delta=np.where(np.abs(self.w_fast_delta) < threshold, 0.0, self.w_fast_delta),
            R_delta=np.where(np.abs(self.R_delta) < threshold, 0.0, self.R_delta),
            b_delta=np.where(np.abs(self.b_delta) < threshold, 0.0, self.b_delta),
            theta_delta=np.where(np.abs(self.theta_delta) < threshold, 0.0, self.theta_delta),
            steps_completed=self.steps_completed,
            avg_loss=self.avg_loss,
        )

    def to_shared_memory(self) -> tuple[list[shared_memory.SharedMemory], dict]:
        """Serialize arrays to shared memory blocks. Returns (shm_list, metadata)."""
        shm_list = []
        metadata = {
            "steps_completed": self.steps_completed,
            "avg_loss": self.avg_loss,
            "arrays": {},
        }
        for name in ("w_slow_delta", "w_fast_delta", "R_delta", "b_delta", "theta_delta"):
            arr = getattr(self, name)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            buf[:] = arr
            shm_list.append(shm)
            metadata["arrays"][name] = {
                "shm_name": shm.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        return shm_list, metadata

    @staticmethod
    def from_shared_memory(metadata: dict) -> WeightDelta:
        """Deserialize from shared memory blocks described in metadata."""
        arrays = {}
        for name, info in metadata["arrays"].items():
            shm = shared_memory.SharedMemory(name=info["shm_name"], create=False)
            shape = tuple(info["shape"])
            dtype = np.dtype(info["dtype"])
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            shm.close()
            arrays[name] = arr
        return WeightDelta(
            w_slow_delta=arrays["w_slow_delta"],
            w_fast_delta=arrays["w_fast_delta"],
            R_delta=arrays["R_delta"],
            b_delta=arrays["b_delta"],
            theta_delta=arrays["theta_delta"],
            steps_completed=metadata["steps_completed"],
            avg_loss=metadata["avg_loss"],
        )


def aggregate(deltas: List[WeightDelta]) -> WeightDelta:
    """FedAvg aggregation: simple mean of all deltas."""
    if not deltas:
        raise ValueError("Cannot aggregate empty list of deltas")

    n = len(deltas)
    total_steps = sum(d.steps_completed for d in deltas)
    avg_loss = sum(d.avg_loss for d in deltas) / n

    return WeightDelta(
        w_slow_delta=np.mean([d.w_slow_delta for d in deltas], axis=0),
        w_fast_delta=np.mean([d.w_fast_delta for d in deltas], axis=0),
        R_delta=np.mean([d.R_delta for d in deltas], axis=0),
        b_delta=np.mean([d.b_delta for d in deltas], axis=0),
        theta_delta=np.mean([d.theta_delta for d in deltas], axis=0),
        steps_completed=total_steps,
        avg_loss=avg_loss,
    )
