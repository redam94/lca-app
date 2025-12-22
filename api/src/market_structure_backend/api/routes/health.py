"""
Health check endpoints for monitoring.
"""

from fastapi import APIRouter

import redis.asyncio as aioredis

from ...core.config import get_settings
from ...schemas import HealthResponse


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and its dependencies.
    """
    settings = get_settings()
    
    # Check Redis
    redis_connected = False
    try:
        redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
        redis_connected = True
        await redis.close()
    except:
        pass
    
    # Check database
    database_connected = False
    try:
        from ...db import get_session_factory
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute("SELECT 1")
            database_connected = True
    except:
        pass
    
    # TODO: Check active workers count
    workers_active = 0
    
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