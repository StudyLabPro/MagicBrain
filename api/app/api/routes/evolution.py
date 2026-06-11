"""
Genome evolution endpoints.
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid

router = APIRouter()


class EvolutionRequest(BaseModel):
    """Request to start genome evolution."""
    initial_genome: str = Field(default="30121033102301230112332100123")
    text: str = Field(..., min_length=10)
    population_size: int = Field(default=20, ge=5, le=100)
    generations: int = Field(default=10, ge=2, le=50)
    fitness_fn: str = Field(default="loss", pattern="^(loss|convergence|stability|robustness)$")
    steps_per_eval: int = Field(default=100, ge=50, le=1000)


class EvolutionStatus(BaseModel):
    """Evolution job status."""
    job_id: str
    status: str  # pending, running, completed, failed
    current_generation: int = 0
    total_generations: int
    best_fitness: Optional[float] = None
    best_genome: Optional[str] = None


class EvolutionResult(BaseModel):
    """Evolution result."""
    job_id: str
    best_genome: str
    best_fitness: float
    generation: int
    hall_of_fame: List[dict]


# In-memory storage
evolution_jobs = {}


def run_evolution_task(
    job_id: str,
    initial_genome: str,
    text: str,
    population_size: int,
    generations: int,
    fitness_fn: str,
    steps_per_eval: int
):
    """Background task for evolution."""
    from magicbrain.evolution import SimpleGA

    try:
        evolution_jobs[job_id]["status"] = "running"

        # Create GA
        ga = SimpleGA(
            population_size=population_size,
            elite_size=max(2, population_size // 10),
            seed=42
        )

        ga.initialize_population(initial_genome)

        # Run evolution
        best = ga.run_evolution(
            text=text,
            num_generations=generations,
            fitness_fn=fitness_fn,
            steps_per_eval=steps_per_eval,
            verbose=False
        )

        # Get hall of fame
        hof = ga.get_hall_of_fame(5)

        evolution_jobs[job_id]["status"] = "completed"
        evolution_jobs[job_id]["current_generation"] = generations
        evolution_jobs[job_id]["best_fitness"] = best.fitness
        evolution_jobs[job_id]["best_genome"] = best.genome
        evolution_jobs[job_id]["result"] = {
            "best_genome": best.genome,
            "best_fitness": best.fitness,
            "generation": best.generation,
            "hall_of_fame": [
                {"genome": ind.genome, "fitness": ind.fitness, "generation": ind.generation}
                for ind in hof
            ]
        }

    except Exception as e:
        evolution_jobs[job_id]["status"] = "failed"
        evolution_jobs[job_id]["message"] = str(e)


@router.post("/start", response_model=EvolutionStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_evolution(request: EvolutionRequest, background_tasks: BackgroundTasks):
    """
    Start genome evolution job.

    Args:
        request: Evolution parameters
        background_tasks: FastAPI background tasks

    Returns:
        Job status
    """
    # Create job
    job_id = str(uuid.uuid4())

    evolution_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "current_generation": 0,
        "total_generations": request.generations,
        "best_fitness": None,
        "best_genome": None
    }

    # Start background task
    background_tasks.add_task(
        run_evolution_task,
        job_id,
        request.initial_genome,
        request.text,
        request.population_size,
        request.generations,
        request.fitness_fn,
        request.steps_per_eval
    )

    return EvolutionStatus(**evolution_jobs[job_id])


@router.get("/status/{job_id}", response_model=EvolutionStatus)
async def get_evolution_status(job_id: str):
    """Get evolution job status."""
    if job_id not in evolution_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return EvolutionStatus(**evolution_jobs[job_id])


@router.get("/result/{job_id}", response_model=EvolutionResult)
async def get_evolution_result(job_id: str):
    """Get evolution result."""
    if job_id not in evolution_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job = evolution_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not completed"
        )

    return EvolutionResult(job_id=job_id, **job["result"])
