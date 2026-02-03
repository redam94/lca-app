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
    """Count active ARQ workers by checking various ARQ key patterns."""
    try:
        # ARQ stores worker health at arq:worker:{worker_name} as a hash
        # with fields like 'j' (jobs completed), 'f' (jobs failed), etc.
        worker_keys = await redis.keys("arq:worker:*")
        if worker_keys:
            # Verify workers are active by checking their health data
            active_count = 0
            for key in worker_keys:
                worker_data = await redis.hgetall(key)
                if worker_data:
                    active_count += 1
            if active_count > 0:
                return active_count

        # Check for worker health keys (alternative pattern)
        health_keys = await redis.keys("arq:health:*")
        if health_keys:
            return len(health_keys)

        # Check for in-progress keys (workers currently processing jobs)
        in_progress_keys = await redis.keys("arq:in-progress:*")
        if in_progress_keys:
            return len(in_progress_keys)

        # Check if there's an active queue which implies a worker is connected
        # ARQ creates a queue key when workers connect
        queue_exists = await redis.exists("arq:queue:default")
        if queue_exists:
            # Queue exists, likely a worker is connected but idle
            return 1

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