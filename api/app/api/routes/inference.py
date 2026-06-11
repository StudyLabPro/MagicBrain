"""
Inference endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path

from ...core.config import settings

router = APIRouter()


class SampleRequest(BaseModel):
    """Request to generate text."""
    model_id: str
    seed_text: str = Field(default="", max_length=1000)
    n_tokens: int = Field(default=100, ge=1, le=10000)
    temperature: float = Field(default=0.75, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)


class SampleResponse(BaseModel):
    """Generated text response."""
    model_id: str
    seed_text: str
    generated_text: str
    n_tokens: int


class PredictRequest(BaseModel):
    """Request to predict next token."""
    model_id: str
    context: str


class PredictResponse(BaseModel):
    """Token prediction response."""
    model_id: str
    context: str
    predictions: List[dict]  # [{token, probability}]


@router.post("/sample", response_model=SampleResponse)
async def sample_text(request: SampleRequest):
    """
    Generate text from model.

    Args:
        request: Sampling parameters

    Returns:
        Generated text
    """
    from magicbrain.io import load_model
    from magicbrain.sampling import sample

    # Load model
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{request.model_id}.npz"
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found"
        )

    brain, stoi, itos, _ = load_model(str(model_path))

    # Generate
    generated = sample(
        brain,
        stoi,
        itos,
        seed=request.seed_text,
        n=request.n_tokens,
        temperature=request.temperature,
        top_k=request.top_k
    )

    return SampleResponse(
        model_id=request.model_id,
        seed_text=request.seed_text,
        generated_text=generated,
        n_tokens=len(generated.split())
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_next(request: PredictRequest):
    """
    Predict next token probabilities.

    Args:
        request: Prediction request

    Returns:
        Top predictions with probabilities
    """
    from magicbrain.io import load_model
    import numpy as np

    # Load model
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{request.model_id}.npz"
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found"
        )

    brain, stoi, itos, _ = load_model(str(model_path))

    # Get last character
    if not request.context:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Context cannot be empty"
        )

    last_char = request.context[-1]
    if last_char not in stoi:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Character '{last_char}' not in vocabulary"
        )

    # Forward pass
    token_id = stoi[last_char]
    probs = brain.forward(token_id)

    # Get top 10 predictions
    top_indices = np.argsort(probs)[-10:][::-1]

    predictions = [
        {
            "token": itos[int(idx)],
            "probability": float(probs[idx])
        }
        for idx in top_indices
    ]

    return PredictResponse(
        model_id=request.model_id,
        context=request.context,
        predictions=predictions
    )
