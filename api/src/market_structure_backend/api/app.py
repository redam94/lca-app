"""
FastAPI application setup.

Configures the main application with:
- CORS middleware
- Exception handlers
- Route registration
- Startup/shutdown events
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.config import get_settings
from ..db import init_db
from .routes import runs_router, progress_router, health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    await init_db(settings.database_url)
    
    yield
    
    # Shutdown
    # Add cleanup here if needed


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Market Structure Analysis API",
        description="""
        Backend API for latent structure analysis on binary purchase data.
        
        Features:
        - Submit and manage model runs
        - Real-time progress tracking via SSE
        - Retrieve results for visualization
        
        Supported models:
        - Latent Class Analysis (LCA)
        - Factor Analysis (Tetrachoric)
        - Bayesian Factor Models (VI and PyMC)
        - Non-negative Matrix Factorization (NMF)
        - Multiple Correspondence Analysis (MCA)
        - Discrete Choice Models (PyMC)
        """,
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.api_debug else None,
            }
        )
    
    # Register routes
    app.include_router(health_router)
    app.include_router(runs_router, prefix="/api/v1")
    app.include_router(progress_router, prefix="/api/v1")
    
    return app


# Application instance
app = create_app()