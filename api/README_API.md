# MagicBrainAPI

REST API service for MagicBrain spiking neural networks.

## Features

- **Model Management**: Create, list, get, delete models
- **Training**: Async training with dopamine/STDP learning
- **Inference**: Text generation and next-token prediction
- **Evolution**: Genome evolution via genetic algorithms
- **Diagnostics**: Model inspection and monitoring

## Installation

```bash
cd /root/StudyNinja-Eco/projects/MagicBrainAPI
pip install -e ".[dev]"
```

## Running the API

```bash
# Development mode
cd backend
python main.py

# Production mode
uvicorn app.api.main:app --host 0.0.0.0 --port 8001
```

API will be available at: `http://localhost:8001`

API docs: `http://localhost:8001/docs`

## Quick Start

### 1. Create a Model

```bash
curl -X POST "http://localhost:8001/api/v1/models/" \
  -H "Content-Type: application/json" \
  -d '{
    "genome": "30121033102301230112332100123",
    "vocab_size": 50,
    "model_id": "my-first-model"
  }'
```

### 2. Train the Model

```bash
curl -X POST "http://localhost:8001/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-first-model",
    "text": "hello world hello world hello world",
    "steps": 1000,
    "learning_type": "dopamine"
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "my-first-model",
  "status": "pending",
  "progress": 0.0,
  "current_step": 0,
  "total_steps": 1000
}
```

### 3. Check Training Status

```bash
curl "http://localhost:8001/api/v1/training/status/{job_id}"
```

### 4. Generate Text

```bash
curl -X POST "http://localhost:8001/api/v1/inference/sample" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-first-model",
    "seed_text": "hello",
    "n_tokens": 50,
    "temperature": 0.75
  }'
```

### 5. Evolve Genomes

```bash
curl -X POST "http://localhost:8001/api/v1/evolution/start" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_genome": "30121033102301230112332100123",
    "text": "training text here...",
    "population_size": 10,
    "generations": 5,
    "fitness_fn": "loss"
  }'
```

## API Endpoints

### Models

- `GET /api/v1/models/` - List all models
- `POST /api/v1/models/` - Create new model
- `GET /api/v1/models/{model_id}` - Get model info
- `DELETE /api/v1/models/{model_id}` - Delete model

### Training

- `POST /api/v1/training/start` - Start training job
- `GET /api/v1/training/status/{job_id}` - Get job status
- `GET /api/v1/training/result/{job_id}` - Get training result

### Inference

- `POST /api/v1/inference/sample` - Generate text
- `POST /api/v1/inference/predict` - Predict next token

### Evolution

- `POST /api/v1/evolution/start` - Start evolution
- `GET /api/v1/evolution/status/{job_id}` - Get status
- `GET /api/v1/evolution/result/{job_id}` - Get result

### Diagnostics

- `GET /api/v1/diagnostics/{model_id}` - Model diagnostics
- `GET /api/v1/diagnostics/{model_id}/weights` - Weight statistics
- `GET /api/v1/diagnostics/{model_id}/activity` - Activity metrics

### Health

- `GET /health` - Health check

## Configuration

Environment variables (`.env` file):

```bash
# API Settings
PROJECT_NAME=MagicBrainAPI
VERSION=0.1.0
HOST=0.0.0.0
PORT=8001

# Model Storage
MODEL_STORAGE_PATH=./models
MAX_MODELS=100

# Training Limits
DEFAULT_TRAINING_STEPS=10000
MAX_TRAINING_STEPS=100000

# Evolution Limits
DEFAULT_POPULATION_SIZE=20
MAX_POPULATION_SIZE=100
```

## Architecture

```
MagicBrainAPI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── main.py           # FastAPI app setup
│   │   │   └── routes/
│   │   │       ├── models.py     # Model management
│   │   │       ├── training.py   # Training endpoints
│   │   │       ├── inference.py  # Inference endpoints
│   │   │       ├── evolution.py  # Evolution endpoints
│   │   │       └── diagnostics.py # Diagnostics endpoints
│   │   └── core/
│   │       └── config.py         # Configuration
│   └── main.py                   # Entry point
├── pyproject.toml
└── README.md
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

## Integration with StudyNinja

MagicBrainAPI can be used as a microservice within StudyNinja ecosystem:

```python
# From StudyNinja-API
import httpx

async def train_student_brain(student_id: str, learning_data: str):
    async with httpx.AsyncClient() as client:
        # Create model
        response = await client.post(
            "http://magicbrain-api:8001/api/v1/models/",
            json={
                "genome": generate_student_genome(student_id),
                "vocab_size": 100,
                "model_id": f"student_{student_id}"
            }
        )

        # Train
        response = await client.post(
            "http://magicbrain-api:8001/api/v1/training/start",
            json={
                "model_id": f"student_{student_id}",
                "text": learning_data,
                "steps": 5000
            }
        )

        return response.json()["job_id"]
```

## License

Part of StudyNinja-Eco project.
