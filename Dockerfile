FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir "." && \
    pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic-settings>=2.0.0 \
    httpx>=0.25.0

# Copy source code
COPY magicbrain/ ./magicbrain/
COPY api/ ./api/

# Model storage directory
RUN mkdir -p /app/models

ENV PYTHONPATH=/app
ENV MODEL_STORAGE_PATH=/app/models

EXPOSE 8000

CMD ["uvicorn", "api.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
