"""
Runtime service for MagicBrain platform inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import time
from typing import Any, Optional

import numpy as np

from magicbrain.io import load_model
from magicbrain.models.snn.text_model import create_from_existing_brain
from magicbrain.platform import ExecutionStrategy, ModelOrchestrator, ModelRegistry
from magicbrain.sampling import apply_sampling_filters

from .config import settings
from .ids import validate_model_id


@dataclass
class RuntimeModel:
    model_id: str
    model: Any
    stoi: dict[str, int]
    itos: dict[int, str]
    metadata: dict[str, Any]
    path: Path
    loaded_at: float = field(default_factory=time)
    mtime: float = 0.0


class RuntimeModelNotFoundError(Exception):
    pass


class RuntimeService:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.MODEL_STORAGE_PATH)
        self.manifest_path = self.storage_path / "runtime_registry.json"
        self.registry = ModelRegistry()
        self.orchestrator = ModelOrchestrator(registry=self.registry)
        self._cache: dict[str, RuntimeModel] = {}
        self._manifest: dict[str, Any] = {"models": {}}
        self._lock = RLock()
        self._load_manifest()

    def model_path(self, model_id: str) -> Path:
        # Second line of defence: the routes validate too, but the runtime
        # service is also called from background jobs and from other services.
        return self.storage_path / f"{validate_model_id(model_id)}.npz"

    def list_storage_models(self) -> list[dict[str, Any]]:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        models = []
        registered = self._manifest.get("models", {})
        for model_file in sorted(self.storage_path.glob("*.npz")):
            stat = model_file.stat()
            model_id = model_file.stem
            models.append({
                "model_id": model_id,
                "path": str(model_file),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "loaded": model_id in self._cache,
                "registered": model_id in registered,
                "auto_load": registered.get(model_id, {}).get("auto_load", False),
            })
        return models

    def load(self, model_id: str, force_reload: bool = False, persist: bool = True) -> RuntimeModel:
        path = self.model_path(model_id)
        if not path.exists():
            raise RuntimeModelNotFoundError(f"Model {model_id} not found")

        stat = path.stat()
        with self._lock:
            cached = self._cache.get(model_id)
            if cached and not force_reload and cached.mtime == stat.st_mtime:
                if persist:
                    self.register_model(model_id, auto_load=True)
                return cached

            brain, stoi, itos, metadata = load_model(str(path))
            model = create_from_existing_brain(
                brain=brain,
                vocab_size=len(stoi),
                model_id=model_id,
                version=str(metadata.get("version", "1.0.0")),
            )
            runtime_model = RuntimeModel(
                model_id=model_id,
                model=model,
                stoi=stoi,
                itos=itos,
                metadata=metadata,
                path=path,
                mtime=stat.st_mtime,
            )
            self._cache[model_id] = runtime_model
            self.registry.register(model, model_id=model_id, overwrite=True)

            if model_id in self.orchestrator.list_models():
                self.orchestrator.remove_model(model_id)
            self.orchestrator.add_model(model, model_id=model_id)

            if persist:
                self.register_model(model_id, auto_load=True)

            return runtime_model

    def unload(self, model_id: str, unregister: bool = False) -> None:
        with self._lock:
            self._cache.pop(model_id, None)
            if model_id in self.orchestrator.list_models():
                self.orchestrator.remove_model(model_id)
            try:
                self.registry.remove(model_id, remove_all_versions=True)
            except Exception:
                pass
            if unregister:
                self.unregister_model(model_id)
            elif model_id in self._manifest.get("models", {}):
                self._manifest["models"][model_id]["auto_load"] = False
                self._manifest["models"][model_id]["updated_at"] = time()
                self._save_manifest()

    def register_model(self, model_id: str, auto_load: bool = False) -> dict[str, Any]:
        path = self.model_path(model_id)
        if not path.exists():
            raise RuntimeModelNotFoundError(f"Model {model_id} not found")

        stat = path.stat()
        models = self._manifest.setdefault("models", {})
        current = models.get(model_id, {})
        current.update({
            "model_id": model_id,
            "path": str(path),
            "mtime": stat.st_mtime,
            "size_bytes": stat.st_size,
            "auto_load": bool(auto_load),
            "updated_at": time(),
        })
        current.setdefault("registered_at", time())
        models[model_id] = current
        self._save_manifest()
        return current

    def unregister_model(self, model_id: str) -> None:
        self._manifest.setdefault("models", {}).pop(model_id, None)
        self._save_manifest()

    def restore_registered_models(self) -> list[str]:
        restored = []
        for model_id, info in list(self._manifest.get("models", {}).items()):
            if not info.get("auto_load", False):
                continue
            try:
                self.load(model_id, force_reload=True, persist=False)
                restored.append(model_id)
            except RuntimeModelNotFoundError:
                self.unregister_model(model_id)
        return restored

    def reload_all(self, registered_only: bool = False) -> list[str]:
        loaded = []
        model_ids = (
            list(self._manifest.get("models", {}).keys())
            if registered_only
            else [item["model_id"] for item in self.list_storage_models()]
        )
        for model_id in model_ids:
            self.load(model_id, force_reload=True)
            loaded.append(model_id)
        return loaded

    def predict(self, model_id: str, context: str, top_k: int = 10) -> dict[str, Any]:
        runtime_model = self.load(model_id)
        if not context:
            raise ValueError("Context cannot be empty")

        last_char = context[-1]
        if last_char not in runtime_model.stoi:
            raise ValueError(f"Character '{last_char}' not in vocabulary")

        token_id = runtime_model.stoi[last_char]
        result = self.orchestrator.execute(
            input_data=token_id,
            strategy=ExecutionStrategy.SEQUENTIAL,
            entry_model=model_id,
        )
        logits = np.asarray(result.get_final_output(), dtype=np.float64)
        probs = np.exp(logits - np.max(logits))
        probs = probs / (np.sum(probs) + 1e-12)

        top_indices = np.argsort(probs)[-top_k:][::-1]
        return {
            "model_id": model_id,
            "context": context,
            "predictions": [
                {
                    "token": runtime_model.itos[int(idx)],
                    "probability": float(probs[idx]),
                }
                for idx in top_indices
            ],
            "runtime": {
                "execution_time_ms": result.execution_time_ms,
                "strategy": result.strategy.value,
                "models_executed": result.models_executed,
            },
        }

    def sample(
        self,
        model_id: str,
        seed_text: str,
        n_tokens: int,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: float = 0.92,
    ) -> dict[str, Any]:
        runtime_model = self.load(model_id)
        model = runtime_model.model
        model.reset()

        seed_chars = [ch for ch in seed_text if ch in runtime_model.stoi]
        if not seed_chars:
            if not runtime_model.stoi:
                generated = ""
                return {
                    "model_id": model_id,
                    "seed_text": seed_text,
                    "generated_text": generated,
                    "n_tokens": 0,
                    "runtime": {"models_executed": [model_id]},
                }
            seed_chars = [next(iter(runtime_model.stoi.keys()))]

        for ch in seed_chars[:-1]:
            model.forward(runtime_model.stoi[ch])

        x = runtime_model.stoi[seed_chars[-1]]
        out = list(seed_chars)
        effective_top_k = top_k if top_k is not None else 18
        start = time()

        for _ in range(n_tokens):
            logits = np.asarray(model.forward(x), dtype=np.float64)
            probs = np.exp(logits - np.max(logits))
            probs = probs / (np.sum(probs) + 1e-12)
            filtered = apply_sampling_filters(
                probs,
                temperature=temperature,
                top_k=effective_top_k,
                top_p=top_p,
            )
            x = int(model.brain.rng.choice(len(filtered), p=filtered))
            out.append(runtime_model.itos[x])

        generated = "".join(out)
        return {
            "model_id": model_id,
            "seed_text": seed_text,
            "generated_text": generated,
            "n_tokens": len(generated),
            "runtime": {
                "execution_time_ms": (time() - start) * 1000,
                "strategy": ExecutionStrategy.SEQUENTIAL.value,
                "models_executed": [model_id],
            },
        }

    def stats(self) -> dict[str, Any]:
        return {
            "storage_path": str(self.storage_path),
            "manifest_path": str(self.manifest_path),
            "registered_models": list(self._manifest.get("models", {}).keys()),
            "loaded_models": list(self._cache.keys()),
            "registry": self.registry.get_stats(),
            "orchestrator": {
                "models": self.orchestrator.list_models(),
                "graph": self.orchestrator.get_graph(),
            },
        }

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            with self.manifest_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict) and isinstance(data.get("models", {}), dict):
            self._manifest = data

    def _save_manifest(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(self._manifest, file, indent=2, sort_keys=True)


_runtime_service = RuntimeService()
_runtime_service.restore_registered_models()


def get_runtime_service() -> RuntimeService:
    return _runtime_service
