"""
Configuration for MagicBrainAPI service.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    PROJECT_NAME: str = "MagicBrainAPI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8001",
        "http://localhost:8002",
        "http://localhost:8004",
    ]

    # Redis (for caching student cognitive models)
    REDIS_URL: str = "redis://localhost:6379/3"

    # KnowledgeBase integration
    KB_BASE_URL: str = "http://localhost:8002"

    # Model Storage
    MODEL_STORAGE_PATH: str = "./models"
    MAX_MODELS: int = 100

    # Training
    DEFAULT_TRAINING_STEPS: int = 10000
    MAX_TRAINING_STEPS: int = 100000
    DEFAULT_GENOME: str = "30121033102301230112332100123"

    # Evolution
    DEFAULT_POPULATION_SIZE: int = 20
    DEFAULT_GENERATIONS: int = 10
    MAX_POPULATION_SIZE: int = 100

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
