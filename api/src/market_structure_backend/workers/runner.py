"""
ARQ worker runner and configuration.

This module configures and runs the ARQ worker process that
executes model fitting tasks in the background.
"""

import asyncio
import logging
from typing import Optional

from arq import create_pool
from arq.connections import RedisSettings

from ..core.config import get_settings
from ..db import init_db
from .tasks import (
    fit_lca_task,
    fit_bayesian_factor_pymc_task,
    fit_dcm_task,
    fit_factor_tetrachoric_task,
    fit_nmf_task,
    fit_mca_task,
    fit_bayesian_vi_task,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def startup(ctx: dict):
    """
    Called when the worker starts.
    
    Initialize database connection and any other resources.
    """
    settings = get_settings()
    
    logger.info("Worker starting up...")
    
    # Initialize database
    await init_db(settings.database_url)
    logger.info(f"Database initialized: {settings.database_url}")
    
    # Store settings in context for tasks
    ctx['settings'] = settings
    
    logger.info("Worker startup complete")


async def shutdown(ctx: dict):
    """
    Called when the worker shuts down.
    
    Clean up resources.
    """
    logger.info("Worker shutting down...")
    # Add any cleanup logic here


async def on_job_start(ctx: dict):
    """Called when a job starts."""
    logger.info(f"Job starting")


async def on_job_end(ctx: dict):
    """Called when a job ends."""
    logger.info(f"Job completed")


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
    max_jobs = 4  # Concurrent jobs per worker
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