"""
Model Registry - Centralized repository for managing models.

Provides versioning, metadata management, and dependency tracking.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import threading

from ..model_interface import ModelInterface, ModelMetadata, ModelType, OutputType


class ModelNotFoundError(Exception):
    """Raised when model is not found in registry."""
    pass


class ModelVersionConflict(Exception):
    """Raised when there's a version conflict."""
    pass


@dataclass
class ModelEntry:
    """
    Entry for a registered model.

    Contains model instance, metadata, and lifecycle information.
    """
    model_id: str
    model: ModelInterface
    metadata: ModelMetadata
    registered_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    # Lifecycle state
    is_active: bool = True
    is_deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding model instance)."""
        return {
            "model_id": self.model_id,
            "metadata": {
                "model_type": self.metadata.model_type.value,
                "version": self.metadata.version,
                "description": self.metadata.description,
                "output_type": self.metadata.output_type.value,
                "parameters_count": self.metadata.parameters_count,
                "tags": self.metadata.tags,
                "author": self.metadata.author,
                "framework": self.metadata.framework,
                "input_shape": self.metadata.input_shape,
                "output_shape": self.metadata.output_shape,
            },
            "registered_at": self.registered_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "is_active": self.is_active,
            "is_deprecated": self.is_deprecated,
        }


class ModelRegistry:
    """
    Centralized registry for managing models in the platform.

    Features:
    - Model registration and retrieval
    - Version management
    - Metadata and tags
    - Dependency tracking
    - Search and filtering
    - Persistence (save/load)
    """

    def __init__(self):
        """Initialize model registry."""
        # Models: model_id -> version -> ModelEntry
        self._models: Dict[str, Dict[str, ModelEntry]] = {}

        # Aliases: alias -> (model_id, version)
        self._aliases: Dict[str, tuple[str, str]] = {}

        # Tags: tag -> set of model_ids
        self._tags: Dict[str, Set[str]] = {}

        # Dependencies: model_id -> set of required model_ids
        self._dependencies: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Hooks for lifecycle events
        self._on_register_hooks: List[Callable[[ModelEntry], None]] = []
        self._on_remove_hooks: List[Callable[[str], None]] = []

    def register(
        self,
        model: ModelInterface,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        alias: Optional[str] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: Model instance
            model_id: Unique model ID (defaults to metadata.model_id)
            version: Model version (defaults to metadata.version)
            alias: Optional alias for easy access
            tags: Optional tags for categorization
            dependencies: Optional list of required model IDs
            overwrite: Whether to overwrite existing version

        Returns:
            Registered model ID

        Raises:
            ModelVersionConflict: If version exists and overwrite=False
        """
        with self._lock:
            # Use metadata if not provided
            metadata = model.get_metadata()
            model_id = model_id or metadata.model_id
            version = version or metadata.version

            # Update metadata if custom ID/version provided
            if model_id != metadata.model_id or version != metadata.version:
                metadata.model_id = model_id
                metadata.version = version
                model.update_metadata(model_id=model_id, version=version)

            # Check for version conflict
            if model_id in self._models and version in self._models[model_id]:
                if not overwrite:
                    raise ModelVersionConflict(
                        f"Model {model_id} version {version} already exists"
                    )

            # Update tags in metadata
            if tags:
                metadata.tags = list(set(metadata.tags + tags))

            # Create entry
            entry = ModelEntry(
                model_id=model_id,
                model=model,
                metadata=metadata,
            )

            # Register
            if model_id not in self._models:
                self._models[model_id] = {}
            self._models[model_id][version] = entry

            # Register alias
            if alias:
                self._aliases[alias] = (model_id, version)

            # Register tags
            for tag in metadata.tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(model_id)

            # Register dependencies
            if dependencies:
                self._dependencies[model_id] = set(dependencies)
                metadata.required_models = dependencies

            # Call hooks
            for hook in self._on_register_hooks:
                hook(entry)

            return model_id

    def get(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> ModelInterface:
        """
        Get a model from the registry.

        Args:
            model_id: Model ID or alias
            version: Specific version (defaults to latest)

        Returns:
            Model instance

        Raises:
            ModelNotFoundError: If model not found
        """
        with self._lock:
            # Resolve alias
            if model_id in self._aliases:
                model_id, version = self._aliases[model_id]

            # Check if model exists
            if model_id not in self._models:
                raise ModelNotFoundError(f"Model {model_id} not found")

            # Get version
            if version is None:
                # Get latest version
                versions = sorted(self._models[model_id].keys())
                if not versions:
                    raise ModelNotFoundError(f"No versions for model {model_id}")
                version = versions[-1]
            elif version not in self._models[model_id]:
                raise ModelNotFoundError(
                    f"Model {model_id} version {version} not found"
                )

            # Get entry and update access stats
            entry = self._models[model_id][version]
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            return entry.model

    def get_metadata(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> ModelMetadata:
        """
        Get model metadata.

        Args:
            model_id: Model ID or alias
            version: Specific version (defaults to latest)

        Returns:
            Model metadata

        Raises:
            ModelNotFoundError: If model not found
        """
        model = self.get(model_id, version)
        return model.get_metadata()

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List models in registry.

        Args:
            model_type: Filter by model type
            tags: Filter by tags (must have all tags)
            active_only: Only include active models

        Returns:
            List of model summaries
        """
        with self._lock:
            results = []

            for model_id, versions in self._models.items():
                for version, entry in versions.items():
                    # Filter by active status
                    if active_only and not entry.is_active:
                        continue

                    # Filter by model type
                    if model_type and entry.metadata.model_type != model_type:
                        continue

                    # Filter by tags
                    if tags:
                        if not all(tag in entry.metadata.tags for tag in tags):
                            continue

                    results.append({
                        "model_id": model_id,
                        "version": version,
                        "model_type": entry.metadata.model_type.value,
                        "output_type": entry.metadata.output_type.value,
                        "description": entry.metadata.description,
                        "parameters": entry.metadata.parameters_count,
                        "tags": entry.metadata.tags,
                        "registered_at": entry.registered_at.isoformat(),
                        "access_count": entry.access_count,
                    })

            return results

    def remove(
        self,
        model_id: str,
        version: Optional[str] = None,
        remove_all_versions: bool = False
    ):
        """
        Remove a model from registry.

        Args:
            model_id: Model ID
            version: Specific version (if None, removes all versions)
            remove_all_versions: Explicitly remove all versions

        Raises:
            ModelNotFoundError: If model not found
        """
        with self._lock:
            if model_id not in self._models:
                raise ModelNotFoundError(f"Model {model_id} not found")

            if version:
                # Remove specific version
                if version not in self._models[model_id]:
                    raise ModelNotFoundError(
                        f"Model {model_id} version {version} not found"
                    )
                del self._models[model_id][version]

                # Remove model_id entry if no versions left
                if not self._models[model_id]:
                    del self._models[model_id]
            else:
                # Remove all versions
                if not remove_all_versions:
                    raise ValueError(
                        "Must specify version or set remove_all_versions=True"
                    )
                del self._models[model_id]

            # Clean up aliases
            aliases_to_remove = [
                alias for alias, (mid, _) in self._aliases.items()
                if mid == model_id
            ]
            for alias in aliases_to_remove:
                del self._aliases[alias]

            # Clean up tags
            for tag, model_ids in self._tags.items():
                model_ids.discard(model_id)

            # Clean up dependencies
            self._dependencies.pop(model_id, None)

            # Call hooks
            for hook in self._on_remove_hooks:
                hook(model_id)

    def deprecate(self, model_id: str, version: Optional[str] = None):
        """
        Mark a model as deprecated.

        Args:
            model_id: Model ID
            version: Specific version (if None, deprecates all versions)
        """
        with self._lock:
            if model_id not in self._models:
                raise ModelNotFoundError(f"Model {model_id} not found")

            if version:
                if version not in self._models[model_id]:
                    raise ModelNotFoundError(
                        f"Model {model_id} version {version} not found"
                    )
                self._models[model_id][version].is_deprecated = True
            else:
                for v in self._models[model_id].values():
                    v.is_deprecated = True

    def search(
        self,
        query: str,
        search_tags: bool = True,
        search_description: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for models.

        Args:
            query: Search query
            search_tags: Search in tags
            search_description: Search in descriptions

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []

        with self._lock:
            for model_id, versions in self._models.items():
                # Search in model_id
                if query_lower in model_id.lower():
                    for version, entry in versions.items():
                        results.append(self._entry_to_summary(model_id, version, entry))
                    continue

                # Search in metadata
                for version, entry in versions.items():
                    metadata = entry.metadata

                    # Search tags
                    if search_tags:
                        if any(query_lower in tag.lower() for tag in metadata.tags):
                            results.append(self._entry_to_summary(model_id, version, entry))
                            continue

                    # Search description
                    if search_description:
                        if query_lower in metadata.description.lower():
                            results.append(self._entry_to_summary(model_id, version, entry))
                            continue

        return results

    def get_dependencies(self, model_id: str) -> List[str]:
        """
        Get model dependencies.

        Args:
            model_id: Model ID

        Returns:
            List of required model IDs
        """
        return list(self._dependencies.get(model_id, set()))

    def get_dependents(self, model_id: str) -> List[str]:
        """
        Get models that depend on this model.

        Args:
            model_id: Model ID

        Returns:
            List of dependent model IDs
        """
        dependents = []
        for mid, deps in self._dependencies.items():
            if model_id in deps:
                dependents.append(mid)
        return dependents

    def save(self, path: Path):
        """
        Save registry state to JSON file.

        Args:
            path: Path to save file
        """
        with self._lock:
            data = {
                "models": {
                    model_id: {
                        version: entry.to_dict()
                        for version, entry in versions.items()
                    }
                    for model_id, versions in self._models.items()
                },
                "aliases": {
                    alias: {"model_id": mid, "version": ver}
                    for alias, (mid, ver) in self._aliases.items()
                },
                "dependencies": {
                    model_id: list(deps)
                    for model_id, deps in self._dependencies.items()
                },
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def load(self, path: Path):
        """
        Load registry state from JSON file.

        Note: This loads metadata only, not model instances.

        Args:
            path: Path to load file
        """
        with open(path, 'r') as f:
            data = json.load(f)

        with self._lock:
            # Load aliases
            self._aliases = {
                alias: (info["model_id"], info["version"])
                for alias, info in data.get("aliases", {}).items()
            }

            # Load dependencies
            self._dependencies = {
                model_id: set(deps)
                for model_id, deps in data.get("dependencies", {}).items()
            }

            # Note: Model instances cannot be loaded from JSON
            # This is just for metadata persistence

    def on_register(self, callback: Callable[[ModelEntry], None]):
        """
        Register callback for model registration events.

        Args:
            callback: Callback function
        """
        self._on_register_hooks.append(callback)

    def on_remove(self, callback: Callable[[str], None]):
        """
        Register callback for model removal events.

        Args:
            callback: Callback function
        """
        self._on_remove_hooks.append(callback)

    def _entry_to_summary(
        self,
        model_id: str,
        version: str,
        entry: ModelEntry
    ) -> Dict[str, Any]:
        """Convert entry to summary dict."""
        return {
            "model_id": model_id,
            "version": version,
            "model_type": entry.metadata.model_type.value,
            "output_type": entry.metadata.output_type.value,
            "description": entry.metadata.description,
            "tags": entry.metadata.tags,
            "registered_at": entry.registered_at.isoformat(),
            "access_count": entry.access_count,
            "is_deprecated": entry.is_deprecated,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_models = sum(len(versions) for versions in self._models.values())

            return {
                "total_models": total_models,
                "unique_model_ids": len(self._models),
                "aliases": len(self._aliases),
                "tags": len(self._tags),
                "dependencies": len(self._dependencies),
            }

    def clear(self):
        """Clear all registry data."""
        with self._lock:
            self._models.clear()
            self._aliases.clear()
            self._tags.clear()
            self._dependencies.clear()


# Global registry instance
_global_registry = ModelRegistry()


def get_global_registry() -> ModelRegistry:
    """
    Get global model registry.

    Returns:
        Global registry instance
    """
    return _global_registry
