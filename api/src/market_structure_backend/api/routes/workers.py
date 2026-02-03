"""
Worker status and monitoring endpoints.

Provides real-time information about ARQ workers, job queues,
and background task status.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
import redis.asyncio as aioredis
from sqlalchemy import select, or_

from ...core.config import get_settings
from ...db import get_session_factory
from ...db.models import ModelRun, ModelRunStatus
from ...schemas import (
    WorkerStatusResponse,
    QueueStats,
    WorkerInfo,
    JobInfo,
)


router = APIRouter(prefix="/workers", tags=["Workers"])


async def _get_redis_connection():
    """Get an async Redis connection."""
    settings = get_settings()
    return await aioredis.from_url(settings.redis_url, decode_responses=True)


async def _get_queue_stats(redis: aioredis.Redis) -> QueueStats:
    """Get statistics about the ARQ job queue."""
    # ARQ uses specific key patterns:
    # - arq:queue:default - list of queued job IDs
    # - arq:job:{job_id} - job data hash
    # - arq:result:{job_id} - job result
    # - arq:in-progress:{worker_id} - jobs currently running

    try:
        # Count queued jobs
        queued = await redis.llen("arq:queue:default")

        # Count in-progress jobs by scanning worker keys
        in_progress_keys = await redis.keys("arq:in-progress:*")
        in_progress = len(in_progress_keys)

        # Count completed jobs (results still in Redis)
        result_keys = await redis.keys("arq:result:*")
        completed = len(result_keys)

        # Total ARQ keys
        all_arq_keys = await redis.keys("arq:*")
        total = len(all_arq_keys)

        return QueueStats(
            queued_jobs=queued,
            in_progress_jobs=in_progress,
            completed_jobs=completed,
            total_keys=total,
        )
    except Exception as e:
        return QueueStats(
            queued_jobs=0,
            in_progress_jobs=0,
            completed_jobs=0,
            total_keys=0,
        )


async def _get_worker_info(redis: aioredis.Redis) -> list[WorkerInfo]:
    """Get information about active ARQ workers."""
    workers = []

    try:
        # ARQ stores worker info at arq:worker:{worker_name} as a hash
        # with fields: 'j' (jobs completed), 'f' (jobs failed), etc.
        worker_keys = await redis.keys("arq:worker:*")

        for key in worker_keys:
            worker_id = key.replace("arq:worker:", "")
            worker_data = await redis.hgetall(key)

            if worker_data:
                # Parse worker data - ARQ uses short field names
                jobs_completed = int(worker_data.get("j", 0))
                jobs_failed = int(worker_data.get("f", 0))

                workers.append(WorkerInfo(
                    worker_id=worker_id,
                    last_health_check=datetime.now(timezone.utc),
                    jobs_completed=jobs_completed,
                    jobs_failed=jobs_failed,
                    current_job=None,
                ))

        # Check for health keys (alternative pattern)
        if not workers:
            health_keys = await redis.keys("arq:health:*")
            for key in health_keys:
                worker_id = key.replace("arq:health:", "")
                health_data = await redis.get(key)
                workers.append(WorkerInfo(
                    worker_id=worker_id,
                    last_health_check=datetime.now(timezone.utc) if health_data else None,
                    jobs_completed=0,
                    jobs_failed=0,
                    current_job=None,
                ))

        # Check for in-progress keys (workers currently processing)
        in_progress_keys = await redis.keys("arq:in-progress:*")
        for key in in_progress_keys:
            # Try to associate with existing worker or create new entry
            current_job = await redis.get(key)
            worker_id = key.replace("arq:in-progress:", "")

            # Update existing worker or add new one
            found = False
            for w in workers:
                if w.worker_id == worker_id:
                    w.current_job = current_job
                    found = True
                    break

            if not found:
                workers.append(WorkerInfo(
                    worker_id=worker_id,
                    current_job=current_job,
                ))

        # If still no workers but queue exists, indicate unknown worker
        if not workers:
            queue_exists = await redis.exists("arq:queue:default")
            if queue_exists:
                workers.append(WorkerInfo(
                    worker_id="worker (idle)",
                    last_health_check=None,
                    jobs_completed=0,
                    jobs_failed=0,
                    current_job=None,
                ))
    except Exception:
        pass

    return workers


async def _get_recent_jobs(redis: aioredis.Redis, limit: int = 20) -> list[JobInfo]:
    """Get information about recent jobs."""
    jobs = []

    try:
        # Get job keys (both pending and completed)
        job_keys = await redis.keys("arq:job:*")
        result_keys = await redis.keys("arq:result:*")

        # Process job keys
        for key in job_keys[:limit]:
            job_id = key.replace("arq:job:", "")
            job_data = await redis.hgetall(key)

            if job_data:
                jobs.append(JobInfo(
                    job_id=job_id,
                    function=job_data.get("function", "unknown"),
                    status="queued" if job_id not in [k.replace("arq:result:", "") for k in result_keys] else "complete",
                    enqueue_time=_parse_timestamp(job_data.get("enqueue_time")),
                    run_id=job_data.get("run_id"),
                ))

        # Also get queued jobs from the queue list
        queued_ids = await redis.lrange("arq:queue:default", 0, limit - 1)
        for job_id in queued_ids:
            if not any(j.job_id == job_id for j in jobs):
                job_data = await redis.hgetall(f"arq:job:{job_id}")
                if job_data:
                    jobs.append(JobInfo(
                        job_id=job_id,
                        function=job_data.get("function", "unknown"),
                        status="queued",
                        enqueue_time=_parse_timestamp(job_data.get("enqueue_time")),
                    ))
    except Exception:
        pass

    return jobs[:limit]


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp string from Redis."""
    if not ts_str:
        return None
    try:
        # ARQ stores timestamps as ISO format or Unix timestamps
        if ts_str.replace(".", "").replace("-", "").isdigit():
            return datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except:
        return None


async def _get_pending_runs() -> list[dict]:
    """Get model runs that are pending or running."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(ModelRun)
                .where(
                    or_(
                        ModelRun.status == ModelRunStatus.PENDING,
                        ModelRun.status == ModelRunStatus.QUEUED,
                        ModelRun.status == ModelRunStatus.RUNNING,
                    )
                )
                .order_by(ModelRun.created_at.desc())
                .limit(50)
            )
            runs = result.scalars().all()
            return [run.to_dict() for run in runs]
    except Exception:
        return []


@router.get("/status", response_model=WorkerStatusResponse)
async def get_worker_status():
    """
    Get comprehensive worker and queue status.

    Returns information about:
    - Redis connectivity
    - Queue statistics (queued, in-progress, completed jobs)
    - Active workers
    - Recent jobs
    - Pending model runs from database
    """
    redis = None
    redis_connected = False
    queue_stats = None
    workers = []
    recent_jobs = []

    try:
        redis = await _get_redis_connection()
        await redis.ping()
        redis_connected = True

        queue_stats = await _get_queue_stats(redis)
        workers = await _get_worker_info(redis)
        recent_jobs = await _get_recent_jobs(redis)
    except Exception:
        redis_connected = False
    finally:
        if redis:
            await redis.close()

    # Get pending runs from database
    pending_runs = await _get_pending_runs()

    return WorkerStatusResponse(
        redis_connected=redis_connected,
        queue_stats=queue_stats,
        workers=workers,
        recent_jobs=recent_jobs,
        pending_runs=pending_runs,
    )


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_info(job_id: str):
    """Get information about a specific job."""
    redis = None
    try:
        redis = await _get_redis_connection()

        # Check job data
        job_data = await redis.hgetall(f"arq:job:{job_id}")

        # Check if result exists
        result = await redis.get(f"arq:result:{job_id}")

        if not job_data and not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        status = "complete" if result else "queued"

        # Check if in progress
        in_progress_keys = await redis.keys("arq:in-progress:*")
        for key in in_progress_keys:
            current_job = await redis.get(key)
            if current_job == job_id:
                status = "in_progress"
                break

        return JobInfo(
            job_id=job_id,
            function=job_data.get("function", "unknown") if job_data else "unknown",
            status=status,
            enqueue_time=_parse_timestamp(job_data.get("enqueue_time")) if job_data else None,
            result=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if redis:
            await redis.close()


@router.get("/runs/active")
async def get_active_runs():
    """Get all active (pending, queued, running) model runs."""
    runs = await _get_pending_runs()
    return {"runs": runs, "count": len(runs)}
