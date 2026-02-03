"""
Health check endpoints for monitoring.
"""

from fastapi import APIRouter

import redis.asyncio as aioredis
from sqlalchemy import text

from ...core.config import get_settings
from ...schemas import HealthResponse


router = APIRouter(tags=["Health"])


async def _count_active_workers(redis: aioredis.Redis) -> int:
    """Count active ARQ workers by checking health and in-progress keys."""
    try:
        # Check for worker health keys (ARQ workers with health checks enabled)
        health_keys = await redis.keys("arq:health:*")
        if health_keys:
            return len(health_keys)

        # Fallback: check for in-progress keys (workers currently processing)
        in_progress_keys = await redis.keys("arq:in-progress:*")
        if in_progress_keys:
            return len(in_progress_keys)

        # If no keys found but queue has jobs, assume at least 1 worker exists
        queued = await redis.llen("arq:queue:default")
        if queued > 0:
            return 0  # Jobs queued but no worker detected

        return 0
    except Exception:
        return 0


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and its dependencies.
    """
    settings = get_settings()

    # Check Redis and count workers
    redis_connected = False
    workers_active = 0
    redis = None
    try:
        redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
        redis_connected = True
        workers_active = await _count_active_workers(redis)
    except:
        pass
    finally:
        if redis:
            await redis.close()

    # Check database
    database_connected = False
    try:
        from ...db import get_session_factory
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
            database_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if redis_connected and database_connected else "degraded",
        version="0.1.0",
        redis_connected=redis_connected,
        database_connected=database_connected,
        workers_active=workers_active,
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes.
    
    Returns 200 if the service is ready to accept traffic.
    """
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes.
    
    Returns 200 if the service is alive.
    """
    return {"alive": True}