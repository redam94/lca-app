"""
ARQ worker runner and configuration.

This module configures and runs the ARQ worker process that
executes model fitting tasks in the background.

Key design: Database and Redis connections are initialized ONCE at worker
startup and stored in the context dict. This prevents issues with aiosqlite
connections being garbage collected in PyMC's ThreadPoolExecutor threads
where there's no event loop.
"""

import asyncio
import logging
from typing import Optional

from arq import create_pool
from arq.connections import RedisSettings

from ..core.config import get_settings
from ..db import init_db, get_session_factory
from ..progress import ProgressTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def startup(ctx: dict):
    """
    Called when the worker starts.
    
    Initialize database connection and other resources ONCE.
    These are stored in ctx and reused by all tasks.
    """
    settings = get_settings()
    
    logger.info("Worker starting up...")
    
    # Initialize database ONCE at startup
    await init_db(settings.database_url)
    logger.info(f"Database initialized: {settings.database_url}")
    
    # Store session factory in context for all tasks to use
    ctx['session_factory'] = get_session_factory()
    
    # Store settings in context for tasks
    ctx['settings'] = settings
    
    # Pre-create a ProgressTracker that can be reused
    # (Each task will still create its own for isolation, but settings are cached)
    ctx['redis_url'] = settings.redis_url
    
    logger.info("Worker startup complete - resources stored in context")


async def shutdown(ctx: dict):
    """
    Called when the worker shuts down.
    
    Clean up resources.
    """
    logger.info("Worker shutting down...")
    
    # Clean up any resources if needed
    # The database engine will be garbage collected
    # Redis connections are closed per-task
    
    logger.info("Worker shutdown complete")


async def on_job_start(ctx: dict):
    """Called when a job starts."""
    logger.info("Job starting")


async def on_job_end(ctx: dict):
    """Called when a job ends."""
    logger.info("Job completed")


class WorkerSettings:
    """
    ARQ worker settings.
    
    This class is discovered by ARQ and used to configure the worker.
    """
    
    # Redis connection settings - must be a class attribute, not a method
    settings = get_settings()
    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password,
    )
    
    # Import tasks here to avoid circular imports
    from .tasks import (
        fit_lca_task,
        fit_bayesian_factor_pymc_task,
        fit_dcm_task,
        fit_factor_tetrachoric_task,
        fit_nmf_task,
        fit_mca_task,
        fit_bayesian_vi_task,
    )
    
    # Task functions to register
    functions = [
        fit_lca_task,
        fit_bayesian_factor_pymc_task,
        fit_dcm_task,
        fit_factor_tetrachoric_task,
        fit_nmf_task,
        fit_mca_task,
        fit_bayesian_vi_task,
    ]
    
    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown
    on_job_start = on_job_start
    on_job_end = on_job_end
    
    # Worker configuration
    # Reduce concurrent jobs for memory-intensive PyMC tasks
    max_jobs = 2  # Reduced from 4 to prevent OOM with PyMC
    job_timeout = 3600  # 1 hour max per job
    keep_result = 86400  # Keep results for 24 hours
    
    # Health check interval
    health_check_interval = 30


def run_worker():
    """
    Entry point for running the worker.
    
    This is called by the CLI command `backend-worker`.
    """
    import arq
    
    logger.info("Starting ARQ worker...")
    
    # Run the worker
    arq.run_worker(WorkerSettings)


async def enqueue_job(
    task_name: str,
    *args,
    job_id: Optional[str] = None,
    **kwargs
):
    """
    Enqueue a job for background processing.
    
    Args:
        task_name: Name of the task function to call
        *args: Positional arguments for the task
        job_id: Optional custom job ID
        **kwargs: Keyword arguments for the task
        
    Returns:
        The job object from ARQ
    """
    settings = get_settings()
    
    redis = await create_pool(RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password,
    ))
    
    try:
        job = await redis.enqueue_job(
            task_name,
            *args,
            _job_id=job_id,
            **kwargs
        )
        return job
    finally:
        await redis.close()


async def get_job_status(job_id: str) -> Optional[dict]:
    """
    Get the status of a queued job.
    
    Args:
        job_id: The job ID to check
        
    Returns:
        Job status dictionary or None if not found
    """
    settings = get_settings()
    
    redis = await create_pool(RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password,
    ))
    
    try:
        job = await redis.job(job_id)
        if job is None:
            return None
        
        return {
            "job_id": job_id,
            "status": job.status if hasattr(job, 'status') else "unknown",
            "result": await job.result(timeout=0) if job.status == "complete" else None,
        }
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return None
    finally:
        await redis.close()


if __name__ == "__main__":
    run_worker()