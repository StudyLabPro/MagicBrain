"""
Inference endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, List, Optional

from ...core.runtime import RuntimeModelNotFoundError, get_runtime_service

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
    runtime: dict[str, Any] = {}


class PredictRequest(BaseModel):
    """Request to predict next token."""
    model_id: str
    context: str
    top_k: int = Field(default=10, ge=1, le=100)


class PredictResponse(BaseModel):
    """Token prediction response."""
    model_id: str
    context: str
    predictions: List[dict]  # [{token, probability}]
    runtime: dict[str, Any] = {}


@router.post("/sample", response_model=SampleResponse)
async def sample_text(request: SampleRequest):
    """
    Generate text from model through the platform runtime.
    """
    runtime = get_runtime_service()
    try:
        result = runtime.sample(
            model_id=request.model_id,
            seed_text=request.seed_text,
            n_tokens=request.n_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except RuntimeModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return SampleResponse(**result)


@router.post("/predict", response_model=PredictResponse)
async def predict_next(request: PredictRequest):
    """
    Predict next token probabilities through the platform runtime.
    """
    runtime = get_runtime_service()
    try:
        result = runtime.predict(
            model_id=request.model_id,
            context=request.context,
            top_k=request.top_k,
        )
    except RuntimeModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return PredictResponse(**result)
