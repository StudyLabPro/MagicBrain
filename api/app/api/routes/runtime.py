"""
Runtime endpoints for platform state and model execution management.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ...core.runtime import RuntimeModelNotFoundError, get_runtime_service

router = APIRouter()


class RuntimeStatsResponse(BaseModel):
    storage_path: str
    manifest_path: str
    registered_models: list[str]
    loaded_models: list[str]
    registry: dict
    orchestrator: dict


class RuntimeReloadResponse(BaseModel):
    loaded_models: list[str]
    total: int


class RuntimeRegisterRequest(BaseModel):
    auto_load: bool = False


@router.get("/stats", response_model=RuntimeStatsResponse)
async def get_runtime_stats():
    return RuntimeStatsResponse(**get_runtime_service().stats())


@router.get("/models")
async def list_runtime_models():
    runtime = get_runtime_service()
    return {
        "models": runtime.list_storage_models(),
        "loaded_models": runtime.stats()["loaded_models"],
    }


@router.post("/models/{model_id}/register")
async def register_runtime_model(model_id: str, request: RuntimeRegisterRequest):
    runtime = get_runtime_service()
    try:
        info = runtime.register_model(model_id, auto_load=request.auto_load)
    except RuntimeModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return info


@router.delete("/models/{model_id}/register", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_runtime_model(model_id: str):
    runtime = get_runtime_service()
    runtime.unload(model_id, unregister=True)
    return None


@router.post("/models/{model_id}/load")
async def load_runtime_model(model_id: str, force_reload: bool = False):
    runtime = get_runtime_service()
    try:
        model = runtime.load(model_id, force_reload=force_reload)
    except RuntimeModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return {
        "model_id": model.model_id,
        "loaded": True,
        "metadata": model.metadata,
        "loaded_at": model.loaded_at,
    }


@router.post("/models/reload", response_model=RuntimeReloadResponse)
async def reload_runtime_models(registered_only: bool = False):
    loaded = get_runtime_service().reload_all(registered_only=registered_only)
    return RuntimeReloadResponse(loaded_models=loaded, total=len(loaded))


@router.delete("/models/{model_id}/unload", status_code=status.HTTP_204_NO_CONTENT)
async def unload_runtime_model(model_id: str, unregister: bool = False):
    get_runtime_service().unload(model_id, unregister=unregister)
    return None


@router.get("/graph")
async def get_runtime_graph():
    return get_runtime_service().stats()["orchestrator"]["graph"]
