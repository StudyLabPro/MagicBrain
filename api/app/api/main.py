"""
MagicBrainAPI - FastAPI application.

Research service for MagicBrain neural network platform.
Part of the MAGIC ecosystem (Level 2: MetaBrain).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from .routes import models, training, inference, diagnostics, evolution, twins, auto_evolution, runtime

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": "magicbrain-research",
        "version": settings.VERSION,
    }


# --- MAGIC reflexive-vertical wiring (optional, gated by MAGIC_ENABLED) ---
_magic_bus = None
_magic_twins = None


@app.on_event("startup")
async def _wire_magic_vertical() -> None:
    """Wire the twin service onto the MAGIC bus (шов 6) when MAGIC_ENABLED.

    Fully optional: if the ``studyninja_magic`` SDK is absent, MAGIC is disabled,
    or no broker is reachable, this is a silent no-op — the service runs standalone.
    """
    global _magic_bus, _magic_twins
    try:
        from studyninja_magic import NullEventBus, event_bus_from_env
        from magicbrain.integration.magic_wiring import (
            default_twin_registry,
            wire_twin_service,
        )
    except Exception:  # pragma: no cover - SDK optional
        return

    bus = event_bus_from_env()
    if isinstance(bus, NullEventBus):
        return
    await bus.connect()
    _magic_twins, get_twin = default_twin_registry()
    await wire_twin_service(bus, get_twin, publish_substrate_on_update=True)
    _magic_bus = bus


# API v1 routes
app.include_router(
    models.router,
    prefix=f"{settings.API_V1_STR}/models",
    tags=["models"],
)
app.include_router(
    training.router,
    prefix=f"{settings.API_V1_STR}/training",
    tags=["training"],
)
app.include_router(
    inference.router,
    prefix=f"{settings.API_V1_STR}/inference",
    tags=["inference"],
)
app.include_router(
    runtime.router,
    prefix=f"{settings.API_V1_STR}/runtime",
    tags=["runtime"],
)
app.include_router(
    diagnostics.router,
    prefix=f"{settings.API_V1_STR}/diagnostics",
    tags=["diagnostics"],
)
app.include_router(
    evolution.router,
    prefix=f"{settings.API_V1_STR}/evolution",
    tags=["evolution"],
)
app.include_router(
    twins.router,
    prefix=f"{settings.API_V1_STR}/twins",
    tags=["neural-digital-twins"],
)
app.include_router(
    auto_evolution.router,
    prefix=f"{settings.API_V1_STR}/auto-evolution",
    tags=["auto-evolution"],
)
