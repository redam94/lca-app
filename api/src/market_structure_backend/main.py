"""
Main entry point for the Market Structure Backend.

Provides CLI commands for running the API server and workers.
"""

import sys
import uvicorn

from .core.config import get_settings


def run_api():
    """Run the FastAPI server."""
    settings = get_settings()
    
    uvicorn.run(
        "market_structure_backend.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        workers=settings.api_workers if not settings.api_debug else 1,
    )


def run_worker():
    """Run the ARQ worker."""
    from .workers.runner import run_worker as worker_runner
    worker_runner()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        run_worker()
    else:
        run_api()