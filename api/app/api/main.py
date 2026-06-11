"""
MagicBrainAPI - FastAPI application.

Research service for MagicBrain neural network platform.
Part of the MAGIC ecosystem (Level 2: MetaBrain).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from .routes import models, training, inference, diagnostics, evolution, twins, auto_evolution

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
