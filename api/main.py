"""
Main entry point for MagicBrainAPI.
"""
import uvicorn
from app.api.main import app
from app.core.config import settings


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
