"""Redis-backed persistence for Neural Digital Twins.

Keys:
    twin:meta:{student_id}  -> JSON metadata (mastery_scores, learning_events, config)
    twin:brain:{student_id} -> bytes (NPZ-serialized brain state)
TTL: 24 hours (refreshed on access)
"""
from __future__ import annotations
import json
import io
from typing import Optional

import numpy as np


class RedisTwinStore:
    """Persist NeuralDigitalTwin state in Redis (DB index 3)."""

    def __init__(self, redis_url: str = "redis://localhost:6379/3"):
        self._redis = None
        self._url = redis_url
        self._ttl = 86400  # 24 hours

    async def connect(self):
        """Connect to Redis."""
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(self._url, decode_responses=False)

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    async def save_twin(self, twin) -> None:
        """
        Save twin metadata and brain state to Redis.

        Args:
            twin: NeuralDigitalTwin instance to persist
        """
        student_id = twin.student_id

        # Save metadata as JSON
        meta = {
            "student_id": twin.student_id,
            "learning_style": twin.learning_style,
            "genome": twin.genome,
            "mastery_scores": {k: float(v) for k, v in twin.mastery_scores.items()},
            "topic_names": twin.topic_names,
            "topic_neurons": {
                k: [int(i) for i in v] for k, v in twin.topic_neurons.items()
            },
            "learning_events": [
                {
                    k: (
                        v.isoformat()
                        if hasattr(v, "isoformat")
                        else v
                    )
                    for k, v in event.items()
                }
                for event in twin.learning_events[-100:]  # Keep last 100
            ],
            "last_practice": {
                k: v.isoformat() if hasattr(v, "isoformat") else str(v)
                for k, v in twin.last_practice.items()
            },
        }
        await self._redis.setex(
            f"twin:meta:{student_id}",
            self._ttl,
            json.dumps(meta).encode("utf-8"),
        )

        # Save brain state as NPZ bytes
        buf = io.BytesIO()
        brain = twin.brain
        # Serialize core brain arrays using numpy
        np.savez_compressed(
            buf,
            w_slow=brain.w_slow,
            w_fast=brain.w_fast,
            theta=brain.theta,
            R=brain.R,
            b=brain.b,
            a=brain.a,
        )
        await self._redis.setex(
            f"twin:brain:{student_id}",
            self._ttl,
            buf.getvalue(),
        )

    async def load_twin_meta(self, student_id: str) -> Optional[dict]:
        """
        Load twin metadata from Redis.

        Args:
            student_id: Student identifier

        Returns:
            Metadata dict or None if not found
        """
        raw = await self._redis.get(f"twin:meta:{student_id}")
        if raw is None:
            return None
        # Refresh TTL on access
        await self._redis.expire(f"twin:meta:{student_id}", self._ttl)
        await self._redis.expire(f"twin:brain:{student_id}", self._ttl)
        return json.loads(raw)

    async def delete_twin(self, student_id: str) -> None:
        """
        Delete twin from Redis.

        Args:
            student_id: Student identifier
        """
        await self._redis.delete(f"twin:meta:{student_id}")
        await self._redis.delete(f"twin:brain:{student_id}")

    async def list_twin_ids(self) -> list[str]:
        """
        List all stored twin student IDs.

        Returns:
            List of student ID strings
        """
        keys = await self._redis.keys("twin:meta:*")
        return [k.decode().split(":")[-1] for k in keys]
