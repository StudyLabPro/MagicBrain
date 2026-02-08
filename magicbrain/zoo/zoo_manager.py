"""
Model Zoo Manager - Manage pretrained models.

Provides download, upload, and management of pretrained models.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import shutil
from datetime import datetime


@dataclass
class ModelManifest:
    """
    Manifest for a pretrained model.

    Contains metadata about the model and how to load it.
    """
    model_id: str
    version: str
    model_type: str                     # "snn", "dnn", "transformer", etc
    description: str
    author: str
    created_at: str                     # ISO format timestamp
    framework: str = "magicbrain"

    # Model files
    weights_file: Optional[str] = None  # Relative path to weights
    config_file: Optional[str] = None   # Relative path to config

    # Architecture details
    architecture: Dict[str, Any] = field(default_factory=dict)

    # Metrics and performance
    metrics: Dict[str, float] = field(default_factory=dict)

    # Training details
    training_details: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    # Download URL (if available)
    download_url: Optional[str] = None

    # Size in bytes
    size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelManifest:
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: Path):
        """
        Save manifest to JSON file.

        Args:
            path: Path to save file
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ModelManifest:
        """
        Load manifest from JSON file.

        Args:
            path: Path to load file

        Returns:
            ModelManifest instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ZooManager:
    """
    Manager for model zoo.

    Handles downloading, uploading, and managing pretrained models.
    """

    def __init__(self, zoo_path: Optional[Path] = None):
        """
        Initialize zoo manager.

        Args:
            zoo_path: Path to zoo directory (default: ~/.magicbrain/zoo)
        """
        if zoo_path is None:
            zoo_path = Path.home() / ".magicbrain" / "zoo"

        self.zoo_path = Path(zoo_path)
        self.zoo_path.mkdir(parents=True, exist_ok=True)

        # Index of available models
        self._index: Dict[str, ModelManifest] = {}
        self._load_index()

    def _load_index(self):
        """Load model index from zoo directory."""
        self._index.clear()

        # Scan for manifest files
        for manifest_path in self.zoo_path.rglob("manifest.json"):
            try:
                manifest = ModelManifest.load(manifest_path)
                key = f"{manifest.model_id}:{manifest.version}"
                self._index[key] = manifest
            except Exception as e:
                print(f"Error loading manifest {manifest_path}: {e}")

    def _save_index(self):
        """Save model index."""
        index_path = self.zoo_path / "index.json"
        index_data = {
            key: manifest.to_dict()
            for key, manifest in self._index.items()
        }
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def add_model(
        self,
        model_id: str,
        version: str,
        model_type: str,
        description: str,
        weights_path: Path,
        config_path: Optional[Path] = None,
        author: str = "",
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ModelManifest:
        """
        Add a model to the zoo.

        Args:
            model_id: Model ID
            version: Version string
            model_type: Model type (snn, dnn, etc)
            description: Model description
            weights_path: Path to weights file
            config_path: Optional path to config file
            author: Author name
            tags: Optional tags
            metrics: Optional performance metrics
            **kwargs: Additional manifest fields

        Returns:
            Created manifest
        """
        # Create model directory
        model_dir = self.zoo_path / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy weights
        weights_dest = model_dir / "weights.npz"
        shutil.copy(weights_path, weights_dest)

        # Copy config if provided
        config_dest = None
        if config_path:
            config_dest = model_dir / "config.json"
            shutil.copy(config_path, config_dest)

        # Get file size
        size_bytes = weights_dest.stat().st_size

        # Create manifest
        manifest = ModelManifest(
            model_id=model_id,
            version=version,
            model_type=model_type,
            description=description,
            author=author,
            created_at=datetime.now().isoformat(),
            weights_file="weights.npz",
            config_file="config.json" if config_dest else None,
            tags=tags or [],
            metrics=metrics or {},
            size_bytes=size_bytes,
            **kwargs
        )

        # Save manifest
        manifest.save(model_dir / "manifest.json")

        # Update index
        key = f"{model_id}:{version}"
        self._index[key] = manifest
        self._save_index()

        return manifest

    def get_model_path(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get path to model directory.

        Args:
            model_id: Model ID
            version: Version (latest if None)

        Returns:
            Path to model directory or None if not found
        """
        if version is None:
            # Find latest version
            versions = self.list_versions(model_id)
            if not versions:
                return None
            version = versions[-1]

        model_dir = self.zoo_path / model_id / version
        if model_dir.exists():
            return model_dir
        return None

    def get_weights_path(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get path to model weights file.

        Args:
            model_id: Model ID
            version: Version (latest if None)

        Returns:
            Path to weights file or None if not found
        """
        model_dir = self.get_model_path(model_id, version)
        if model_dir is None:
            return None

        weights_file = model_dir / "weights.npz"
        if weights_file.exists():
            return weights_file
        return None

    def get_manifest(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[ModelManifest]:
        """
        Get model manifest.

        Args:
            model_id: Model ID
            version: Version (latest if None)

        Returns:
            ModelManifest or None if not found
        """
        if version is None:
            versions = self.list_versions(model_id)
            if not versions:
                return None
            version = versions[-1]

        key = f"{model_id}:{version}"
        return self._index.get(key)

    def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models.

        Args:
            model_type: Filter by model type
            tags: Filter by tags

        Returns:
            List of model summaries
        """
        results = []

        for manifest in self._index.values():
            # Filter by type
            if model_type and manifest.model_type != model_type:
                continue

            # Filter by tags
            if tags:
                if not all(tag in manifest.tags for tag in tags):
                    continue

            results.append({
                "model_id": manifest.model_id,
                "version": manifest.version,
                "model_type": manifest.model_type,
                "description": manifest.description,
                "author": manifest.author,
                "tags": manifest.tags,
                "size_mb": (manifest.size_bytes or 0) / 1024 / 1024,
                "created_at": manifest.created_at,
            })

        return results

    def list_versions(self, model_id: str) -> List[str]:
        """
        List available versions for a model.

        Args:
            model_id: Model ID

        Returns:
            Sorted list of versions
        """
        versions = []
        for key in self._index.keys():
            mid, ver = key.split(":", 1)
            if mid == model_id:
                versions.append(ver)
        return sorted(versions)

    def remove_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ):
        """
        Remove a model from zoo.

        Args:
            model_id: Model ID
            version: Version (removes all versions if None)
        """
        if version:
            # Remove specific version
            model_dir = self.get_model_path(model_id, version)
            if model_dir and model_dir.exists():
                shutil.rmtree(model_dir)

            # Remove from index
            key = f"{model_id}:{version}"
            self._index.pop(key, None)

        else:
            # Remove all versions
            versions = self.list_versions(model_id)
            for ver in versions:
                self.remove_model(model_id, ver)

        self._save_index()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for models.

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []

        for manifest in self._index.values():
            # Search in model_id
            if query_lower in manifest.model_id.lower():
                results.append(self._manifest_to_summary(manifest))
                continue

            # Search in description
            if query_lower in manifest.description.lower():
                results.append(self._manifest_to_summary(manifest))
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in manifest.tags):
                results.append(self._manifest_to_summary(manifest))
                continue

        return results

    def _manifest_to_summary(self, manifest: ModelManifest) -> Dict[str, Any]:
        """Convert manifest to summary dict."""
        return {
            "model_id": manifest.model_id,
            "version": manifest.version,
            "model_type": manifest.model_type,
            "description": manifest.description,
            "author": manifest.author,
            "tags": manifest.tags,
            "size_mb": (manifest.size_bytes or 0) / 1024 / 1024,
            "created_at": manifest.created_at,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get zoo statistics.

        Returns:
            Statistics dictionary
        """
        total_size = sum(
            (m.size_bytes or 0) for m in self._index.values()
        )

        model_types = {}
        for manifest in self._index.values():
            model_types[manifest.model_type] = model_types.get(manifest.model_type, 0) + 1

        return {
            "total_models": len(self._index),
            "total_size_mb": total_size / 1024 / 1024,
            "model_types": model_types,
            "zoo_path": str(self.zoo_path),
        }


# Global zoo manager
_global_zoo = None


def get_global_zoo() -> ZooManager:
    """
    Get global zoo manager.

    Returns:
        Global zoo manager instance
    """
    global _global_zoo
    if _global_zoo is None:
        _global_zoo = ZooManager()
    return _global_zoo
