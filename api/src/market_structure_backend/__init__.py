"""
Market Structure Backend
========================

FastAPI backend with ARQ workers for market structure analysis.

This package provides:
- REST API for submitting and managing model runs
- Background workers for model fitting
- Real-time progress tracking via SSE
- Database persistence for model tracking

Quick Start:
    # Start the API server
    uvicorn market_structure_backend.api:app --reload
    
    # Start the worker
    arq market_structure_backend.workers.runner.WorkerSettings
"""

__version__ = "0.1.0"

from .api import app, create_app
from .core import Settings, get_settings

__all__ = [
    "__version__",
    "app",
    "create_app",
    "Settings",
    "get_settings",
]