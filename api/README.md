# Market Structure Backend

FastAPI backend with ARQ workers for latent structure analysis on binary purchase data.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Frontend       │────▶│  FastAPI        │────▶│  Redis          │
│  (Streamlit/    │     │  Server         │     │  (Job Queue +   │
│   React)        │◀────│                 │◀────│   Pub/Sub)      │
│                 │ SSE │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                               │                         │
                               │                         │
                               ▼                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  SQLite/        │     │  ARQ Workers    │
                        │  PostgreSQL     │     │  (Model Fitting)│
                        │  (Model Runs)   │     │                 │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

## Features

- **Async API**: Non-blocking FastAPI endpoints
- **Background Processing**: ARQ workers for long-running model fits
- **Real-time Progress**: SSE streaming with PyMC sampling callbacks
- **Model Tracking**: Timestamped runs with full metadata in SQLite/PostgreSQL
- **Multiple Models**: LCA, Factor Analysis, NMF, MCA, Bayesian models, DCM

## Quick Start

### Prerequisites

- Python 3.12+
- Redis server running on localhost:6379

### Installation

```bash
# Clone and install
cd market_structure_backend
pip install -e .

# Or with uv
uv pip install -e .
```

### Running the API

```bash
# Start the API server
uvicorn market_structure_backend.api:app --reload --port 8000

# Or use the CLI
python -m market_structure_backend
```

### Running Workers

```bash
# Start an ARQ worker
arq market_structure_backend.workers.runner.WorkerSettings

# Or use the CLI
python -m market_structure_backend worker
```

## API Endpoints

### Model Runs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/runs` | Submit a new model run |
| GET | `/api/v1/runs` | List all model runs |
| GET | `/api/v1/runs/{id}` | Get run details |
| GET | `/api/v1/runs/{id}/results` | Get full results |
| DELETE | `/api/v1/runs/{id}` | Delete a run |
| POST | `/api/v1/runs/{id}/cancel` | Cancel a run |

### Progress Tracking

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/progress/{id}` | Get current progress |
| GET | `/api/v1/progress/{id}/stream` | SSE progress stream |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| GET | `/live` | Liveness probe |

## Example Usage

### Submit a Model Run

```python
import httpx
import base64

# Read and encode data
with open("purchases.csv", "rb") as f:
    csv_base64 = base64.b64encode(f.read()).decode()

# Submit LCA run
response = httpx.post(
    "http://localhost:8000/api/v1/runs",
    json={
        "model_type": "lca",
        "name": "My LCA Analysis",
        "params": {
            "n_classes": 4,
            "max_iter": 100,
            "n_init": 10
        },
        "data": {
            "csv_base64": csv_base64
        },
        "product_columns": ["prod_a", "prod_b", "prod_c"]
    }
)

run = response.json()
print(f"Run ID: {run['id']}")
```

### Stream Progress Updates

```python
import httpx

run_id = "your-run-id"

with httpx.stream("GET", f"http://localhost:8000/api/v1/progress/{run_id}/stream") as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            progress = json.loads(line[5:])
            print(f"Progress: {progress['progress']*100:.1f}% - {progress['message']}")
            
            if progress['phase'] in ['completed', 'failed']:
                break
```

### JavaScript SSE Client

```javascript
const eventSource = new EventSource(`/api/v1/progress/${runId}/stream`);

eventSource.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    
    // Update progress bar
    progressBar.style.width = `${progress.progress * 100}%`;
    statusText.textContent = progress.message;
    
    // Show ETA
    if (progress.eta_seconds) {
        etaText.textContent = `ETA: ${Math.round(progress.eta_seconds)}s`;
    }
};

eventSource.addEventListener('close', () => {
    eventSource.close();
});

eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    eventSource.close();
};
```

## Configuration

Environment variables (or `.env` file):

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_WORKERS=1

# Redis Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Database
DATABASE_URL=sqlite+aiosqlite:///./model_runs.db

# Worker Settings
WORKER_CONCURRENCY=4
JOB_TIMEOUT_SECONDS=3600
```

## Project Structure

```
src/market_structure_backend/
├── __init__.py
├── main.py                 # Entry points
├── api/
│   ├── __init__.py
│   ├── app.py             # FastAPI application
│   └── routes/
│       ├── __init__.py
│       ├── runs.py        # Model run endpoints
│       ├── progress.py    # SSE streaming
│       └── health.py      # Health checks
├── core/
│   ├── __init__.py
│   └── config.py          # Settings management
├── db/
│   ├── __init__.py
│   └── models.py          # SQLAlchemy models
├── progress/
│   ├── __init__.py
│   └── tracker.py         # Redis pub/sub + callbacks
├── schemas/
│   ├── __init__.py
│   └── api.py             # Pydantic schemas
└── workers/
    ├── __init__.py
    ├── tasks.py           # ARQ task functions
    ├── runner.py          # Worker configuration
    └── model_implementations.py  # Model fitting logic
```

## Development

### Running Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy src/market_structure_backend
```

## Docker Deployment

```bash
# Build
docker build -t market-structure-backend .

# Run API
docker run -p 8000:8000 -e REDIS_HOST=redis market-structure-backend

# Run Worker
docker run -e REDIS_HOST=redis market-structure-backend worker
```

See `docker-compose.yml` for a complete setup.

## License

MIT