"""
Model management endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
import os
from pathlib import Path

from ...core.config import settings

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    genome: str
    vocab_size: int
    created_at: str
    size_bytes: int
    metadata: dict = {}


class CreateModelRequest(BaseModel):
    """Request to create a new model."""
    genome: str
    vocab_size: int
    model_id: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of models."""
    models: List[ModelInfo]
    total: int


@router.get("/", response_model=ModelListResponse)
async def list_models(load_metadata: bool = False):
    """
    List all available models.

    Args:
        load_metadata: If True, load full metadata (slower). Default False for fast listing.

    Returns list of trained models in storage.
    """
    models_path = Path(settings.MODEL_STORAGE_PATH)
    models_path.mkdir(parents=True, exist_ok=True)

    models = []
    for model_file in models_path.glob("*.npz"):
        # Get file info
        stat = model_file.stat()

        if load_metadata:
            # Load full metadata (slower but complete)
            try:
                from magicbrain.io import load_model
                _, _, _, metadata = load_model(str(model_file))

                models.append(ModelInfo(
                    model_id=model_file.stem,
                    genome=metadata.get("genome_str", "unknown"),
                    vocab_size=metadata.get("vocab_size", 0),
                    created_at=str(stat.st_ctime),
                    size_bytes=stat.st_size,
                    metadata={
                        "steps": metadata.get("step", 0),
                        "N": metadata.get("N", 0),
                        "K": metadata.get("K", 0),
                        "timestamp": metadata.get("timestamp", 0),
                    }
                ))
            except Exception:
                # If loading fails, use minimal info
                models.append(ModelInfo(
                    model_id=model_file.stem,
                    genome="error",
                    vocab_size=0,
                    created_at=str(stat.st_ctime),
                    size_bytes=stat.st_size,
                    metadata={"error": "failed_to_load"}
                ))
        else:
            # Fast listing without loading models
            models.append(ModelInfo(
                model_id=model_file.stem,
                genome="unknown",
                vocab_size=0,
                created_at=str(stat.st_ctime),
                size_bytes=stat.st_size,
                metadata={}
            ))

    return ModelListResponse(
        models=models,
        total=len(models)
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get information about a specific model.

    Args:
        model_id: Unique model identifier

    Returns:
        Model information
    """
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    stat = model_path.stat()

    # Load model metadata from file
    try:
        from magicbrain.io import load_model

        # Load model to extract metadata (only reads, doesn't modify)
        _, _, _, metadata = load_model(str(model_path))

        return ModelInfo(
            model_id=model_id,
            genome=metadata.get("genome_str", "unknown"),
            vocab_size=metadata.get("vocab_size", 0),
            created_at=str(stat.st_ctime),
            size_bytes=stat.st_size,
            metadata={
                "steps": metadata.get("step", 0),
                "N": metadata.get("N", 0),
                "K": metadata.get("K", 0),
                "timestamp": metadata.get("timestamp", 0),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model metadata: {str(e)}"
        )


@router.post("/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(request: CreateModelRequest):
    """
    Create a new untrained model.

    Args:
        request: Model creation parameters

    Returns:
        Created model info
    """
    import uuid
    from magicbrain import TextBrain
    from magicbrain.io import save_model

    # Generate model ID if not provided
    model_id = request.model_id or str(uuid.uuid4())

    # Check if model already exists
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"
    if model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {model_id} already exists"
        )

    # Create brain
    brain = TextBrain(request.genome, request.vocab_size)

    # Create dummy vocab for now
    stoi = {str(i): i for i in range(request.vocab_size)}
    itos = {i: str(i) for i in range(request.vocab_size)}

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(brain, stoi, itos, str(model_path))

    stat = model_path.stat()

    return ModelInfo(
        model_id=model_id,
        genome=request.genome,
        vocab_size=request.vocab_size,
        created_at=str(stat.st_ctime),
        size_bytes=stat.st_size,
        metadata={"steps": 0, "untrained": True}
    )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: str):
    """
    Delete a model.

    Args:
        model_id: Model to delete
    """
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    model_path.unlink()

    return None
