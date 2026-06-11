"""
Auto-evolution pipeline — continuous genome optimization.

Part of MAGIC Level 2 (MetaBrain) continuous learning loop.
Runs periodic evolution cycles, tracks fitness trends,
and exposes best genomes for cross-service consumption.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class AutoEvolutionConfig(BaseModel):
    """Configuration for the auto-evolution loop."""

    seed_genome: str = Field(default="30121033102301230112332100123")
    training_text: str = Field(
        default="The quick brown fox jumps over the lazy dog.",
        min_length=10,
    )
    population_size: int = Field(default=12, ge=4, le=50)
    generations_per_cycle: int = Field(default=5, ge=1, le=20)
    steps_per_eval: int = Field(default=100, ge=50, le=500)
    cycle_interval_seconds: int = Field(default=300, ge=30)
    max_cycles: int = Field(default=0, description="0 = run indefinitely")
    fitness_fn: str = Field(
        default="loss", pattern="^(loss|convergence|stability)$"
    )


class EvolutionCycleResult(BaseModel):
    """Result of a single evolution cycle."""

    cycle: int
    best_genome: str
    best_fitness: float
    mean_fitness: float
    improvement_pct: float
    duration_seconds: float
    timestamp: float


class AutoEvolutionStatus(BaseModel):
    """Status of the auto-evolution pipeline."""

    pipeline_id: str
    running: bool
    total_cycles: int
    current_best_genome: str
    current_best_fitness: float
    fitness_history: list[float]
    last_cycle_at: float | None
    config: AutoEvolutionConfig


# Pipeline state
_pipelines: dict[str, dict[str, Any]] = {}
_pipeline_tasks: dict[str, asyncio.Task[None]] = {}


async def _run_evolution_cycle(
    config: AutoEvolutionConfig,
    current_genome: str,
) -> dict[str, Any]:
    """Run a single evolution cycle in a thread pool (CPU-bound)."""
    loop = asyncio.get_event_loop()

    def _evolve() -> dict[str, Any]:
        from magicbrain.evolution import SimpleGA

        ga = SimpleGA(
            population_size=config.population_size,
            elite_size=max(2, config.population_size // 5),
            seed=int(time.time()) % 10000,
        )
        ga.initialize_population(current_genome)

        start = time.time()
        best = ga.run_evolution(
            text=config.training_text,
            num_generations=config.generations_per_cycle,
            fitness_fn=config.fitness_fn,
            steps_per_eval=config.steps_per_eval,
            verbose=False,
        )
        duration = time.time() - start

        # Compute mean fitness of final population
        fitnesses = [ind.fitness for ind in ga.population if ind.fitness is not None]
        mean_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        return {
            "best_genome": best.genome,
            "best_fitness": best.fitness,
            "mean_fitness": mean_fitness,
            "duration": duration,
        }

    return await loop.run_in_executor(None, _evolve)


async def _auto_evolution_loop(pipeline_id: str) -> None:
    """Continuous evolution loop."""
    pipeline = _pipelines[pipeline_id]
    config = AutoEvolutionConfig(**pipeline["config"])
    current_genome = config.seed_genome
    prev_fitness = 0.0

    while pipeline["running"]:
        cycle_num = pipeline["total_cycles"] + 1

        if config.max_cycles > 0 and cycle_num > config.max_cycles:
            pipeline["running"] = False
            break

        try:
            result = await _run_evolution_cycle(config, current_genome)

            improvement = 0.0
            if prev_fitness != 0.0:
                improvement = (
                    (result["best_fitness"] - prev_fitness) / abs(prev_fitness) * 100
                )

            cycle_result = EvolutionCycleResult(
                cycle=cycle_num,
                best_genome=result["best_genome"],
                best_fitness=result["best_fitness"],
                mean_fitness=result["mean_fitness"],
                improvement_pct=round(improvement, 2),
                duration_seconds=round(result["duration"], 2),
                timestamp=time.time(),
            )

            # Update pipeline state
            pipeline["total_cycles"] = cycle_num
            pipeline["current_best_genome"] = result["best_genome"]
            pipeline["current_best_fitness"] = result["best_fitness"]
            pipeline["fitness_history"].append(result["best_fitness"])
            pipeline["last_cycle_at"] = time.time()
            pipeline["cycles"].append(cycle_result.model_dump())

            # Cap stored cycles to prevent memory growth
            max_stored_cycles = 100
            if len(pipeline["cycles"]) > max_stored_cycles:
                pipeline["cycles"] = pipeline["cycles"][-max_stored_cycles:]
            if len(pipeline["fitness_history"]) > max_stored_cycles:
                pipeline["fitness_history"] = pipeline["fitness_history"][-max_stored_cycles:]

            # Use best genome as seed for next cycle
            current_genome = result["best_genome"]
            prev_fitness = result["best_fitness"]

        except Exception as e:
            logger.warning("Evolution cycle %d failed: %s", cycle_num, e)

        # Wait before next cycle
        await asyncio.sleep(config.cycle_interval_seconds)


@router.post("/start", response_model=AutoEvolutionStatus, status_code=201)
async def start_auto_evolution(config: AutoEvolutionConfig):
    """Start a continuous auto-evolution pipeline."""
    pipeline_id = str(uuid.uuid4())[:8]

    _pipelines[pipeline_id] = {
        "pipeline_id": pipeline_id,
        "running": True,
        "total_cycles": 0,
        "current_best_genome": config.seed_genome,
        "current_best_fitness": 0.0,
        "fitness_history": [],
        "last_cycle_at": None,
        "config": config.model_dump(),
        "cycles": [],
    }

    # Start the background loop
    task = asyncio.create_task(_auto_evolution_loop(pipeline_id))
    _pipeline_tasks[pipeline_id] = task

    return AutoEvolutionStatus(**_pipelines[pipeline_id])


@router.get("/status/{pipeline_id}", response_model=AutoEvolutionStatus)
async def get_pipeline_status(pipeline_id: str):
    """Get auto-evolution pipeline status."""
    if pipeline_id not in _pipelines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )
    return AutoEvolutionStatus(**_pipelines[pipeline_id])


@router.post("/stop/{pipeline_id}")
async def stop_pipeline(pipeline_id: str):
    """Stop a running auto-evolution pipeline."""
    if pipeline_id not in _pipelines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    _pipelines[pipeline_id]["running"] = False

    task = _pipeline_tasks.get(pipeline_id)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    return {"status": "stopped", "pipeline_id": pipeline_id}


@router.get("/best-genome")
async def get_best_genome():
    """Get the best genome across all pipelines."""
    best: dict[str, Any] | None = None
    for p in _pipelines.values():
        if p["current_best_fitness"] > 0:
            if best is None or p["current_best_fitness"] > best["fitness"]:
                best = {
                    "pipeline_id": p["pipeline_id"],
                    "genome": p["current_best_genome"],
                    "fitness": p["current_best_fitness"],
                    "cycles_completed": p["total_cycles"],
                }
    if not best:
        return {"genome": "30121033102301230112332100123", "fitness": 0.0, "source": "default"}
    return best


@router.get("/fitness-trend/{pipeline_id}")
async def get_fitness_trend(pipeline_id: str):
    """Get fitness trend for a pipeline."""
    if pipeline_id not in _pipelines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )
    history = _pipelines[pipeline_id]["fitness_history"]
    return {
        "pipeline_id": pipeline_id,
        "fitness_values": history,
        "trend": _compute_trend(history),
    }


def _compute_trend(values: list[float]) -> str:
    """Compute trend direction from fitness history."""
    if len(values) < 3:
        return "insufficient_data"
    recent = values[-3:]
    if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
        return "improving"
    if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
        return "declining"
    return "stable"
