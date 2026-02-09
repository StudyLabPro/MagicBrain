"""
Diagnostics and monitoring endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path

from ...core.config import settings

router = APIRouter()


class ModelDiagnostics(BaseModel):
    """Model diagnostics information."""
    model_id: str
    N: int  # Number of neurons
    K: int  # Connectivity
    vocab_size: int
    step: int
    genome: str


class WeightStatistics(BaseModel):
    """Weight statistics."""
    mean_w_slow: float
    mean_w_fast: float
    mean_abs_w_total: float
    ei_ratio: float
    sparsity: float


class ActivityMetrics(BaseModel):
    """Neural activity metrics."""
    firing_rate: float
    mean_theta: float
    mean_dopamine: Optional[float] = None


@router.get("/{model_id}", response_model=ModelDiagnostics)
async def get_diagnostics(model_id: str):
    """
    Get model diagnostics.

    Args:
        model_id: Model ID

    Returns:
        Diagnostic information
    """
    from magicbrain.io import load_model

    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    brain, stoi, itos, _ = load_model(str(model_path))

    return ModelDiagnostics(
        model_id=model_id,
        N=brain.N,
        K=brain.K,
        vocab_size=brain.vocab_size,
        step=brain.step,
        genome=brain.genome_str
    )


@router.get("/{model_id}/weights", response_model=WeightStatistics)
async def get_weight_stats(model_id: str):
    """
    Get weight statistics.

    Args:
        model_id: Model ID

    Returns:
        Weight statistics
    """
    from magicbrain.io import load_model
    from magicbrain.diagnostics import SynapticAnalyzer

    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    brain, _, _, _ = load_model(str(model_path))

    analyzer = SynapticAnalyzer()
    stats = analyzer.analyze_weights(brain)

    return WeightStatistics(
        mean_w_slow=stats["mean_w_slow"],
        mean_w_fast=stats["mean_w_fast"],
        mean_abs_w_total=stats["mean_abs_w_total"],
        ei_ratio=stats["ei_ratio"],
        sparsity=stats["sparsity_total"]
    )


@router.get("/{model_id}/activity", response_model=ActivityMetrics)
async def get_activity_metrics(model_id: str):
    """
    Get neural activity metrics.

    Args:
        model_id: Model ID

    Returns:
        Activity metrics
    """
    from magicbrain.io import load_model
    import numpy as np

    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    brain, _, _, _ = load_model(str(model_path))

    # Compute metrics
    firing_rate = float(np.mean(brain.a))
    mean_theta = float(np.mean(brain.theta))

    return ActivityMetrics(
        firing_rate=firing_rate,
        mean_theta=mean_theta,
        mean_dopamine=getattr(brain, 'dopamine', None)
    )
