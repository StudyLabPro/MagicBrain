FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_STORAGE_PATH=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY magicbrain/ ./magicbrain/
COPY api/ ./api/

RUN pip install --no-cache-dir ".[service]"

RUN mkdir -p /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
