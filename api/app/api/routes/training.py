"""
Training endpoints.
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import uuid

from ...core.config import settings

router = APIRouter()


class TrainingRequest(BaseModel):
    """Request to start training."""
    model_id: str
    text: str = Field(..., min_length=10)
    steps: int = Field(default=1000, ge=100, le=100000)
    learning_type: str = Field(default="dopamine", pattern="^(dopamine|stdp|triplet_stdp)$")


class TrainingStatus(BaseModel):
    """Training job status."""
    job_id: str
    model_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    current_step: int = 0
    total_steps: int
    loss: Optional[float] = None
    message: Optional[str] = None


class TrainingResult(BaseModel):
    """Training result."""
    job_id: str
    model_id: str
    final_loss: float
    steps_completed: int
    time_seconds: float


# In-memory job storage (would be Redis in production)
training_jobs = {}


def train_model_task(job_id: str, model_id: str, text: str, steps: int, learning_type: str):
    """Background task for training."""
    import time
    from magicbrain.io import load_model, save_model
    from magicbrain.tasks.text_task import build_vocab, train_loop_with_history
    from magicbrain.learning_rules import STDPBrain

    try:
        # Update status
        training_jobs[job_id]["status"] = "running"

        # Load model
        model_path = Path(settings.MODEL_STORAGE_PATH) / f"{model_id}.npz"
        if not model_path.exists():
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["message"] = "Model not found"
            return

        brain, stoi, itos = load_model(str(model_path))

        # Build vocab from text
        text_stoi, text_itos = build_vocab(text)

        # Train
        start_time = time.time()

        if learning_type == "stdp":
            # Convert to STDP brain
            from magicbrain.learning_rules import STDPBrain
            stdp_brain = STDPBrain(brain.genome_str, len(text_stoi), stdp_type="standard")
            losses = train_loop_with_history(stdp_brain, text, text_stoi, steps, verbose=False)
            brain = stdp_brain
        elif learning_type == "triplet_stdp":
            from magicbrain.learning_rules import STDPBrain
            stdp_brain = STDPBrain(brain.genome_str, len(text_stoi), stdp_type="triplet")
            losses = train_loop_with_history(stdp_brain, text, text_stoi, steps, verbose=False)
            brain = stdp_brain
        else:
            losses = train_loop_with_history(brain, text, text_stoi, steps, verbose=False)

        elapsed = time.time() - start_time

        # Save model
        save_model(brain, text_stoi, text_itos, str(model_path))

        # Update status
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 1.0
        training_jobs[job_id]["current_step"] = steps
        training_jobs[job_id]["loss"] = float(losses[-1])
        training_jobs[job_id]["result"] = {
            "final_loss": float(losses[-1]),
            "steps_completed": steps,
            "time_seconds": elapsed
        }

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = str(e)


@router.post("/start", response_model=TrainingStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start asynchronous training job.

    Args:
        request: Training parameters
        background_tasks: FastAPI background tasks

    Returns:
        Job status
    """
    # Check model exists
    model_path = Path(settings.MODEL_STORAGE_PATH) / f"{request.model_id}.npz"
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found"
        )

    # Create job
    job_id = str(uuid.uuid4())

    training_jobs[job_id] = {
        "job_id": job_id,
        "model_id": request.model_id,
        "status": "pending",
        "progress": 0.0,
        "current_step": 0,
        "total_steps": request.steps,
        "loss": None,
        "message": None
    }

    # Start background task
    background_tasks.add_task(
        train_model_task,
        job_id,
        request.model_id,
        request.text,
        request.steps,
        request.learning_type
    )

    return TrainingStatus(**training_jobs[job_id])


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """
    Get training job status.

    Args:
        job_id: Training job ID

    Returns:
        Job status
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return TrainingStatus(**training_jobs[job_id])


@router.get("/result/{job_id}", response_model=TrainingResult)
async def get_training_result(job_id: str):
    """
    Get training result (only for completed jobs).

    Args:
        job_id: Training job ID

    Returns:
        Training result
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job = training_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not completed (status: {job['status']})"
        )

    return TrainingResult(
        job_id=job_id,
        model_id=job["model_id"],
        **job["result"]
    )
