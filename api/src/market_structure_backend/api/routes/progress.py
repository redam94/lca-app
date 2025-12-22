"""
Server-Sent Events (SSE) endpoint for real-time progress updates.

Provides a streaming endpoint that clients can connect to for
receiving live progress updates during model fitting.
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import get_settings
from ...db import ModelRun, ModelRunStatus, get_session
from ...progress import ProgressTracker, ProgressUpdate
from ...schemas import ProgressResponse


router = APIRouter(prefix="/progress", tags=["Progress"])


async def progress_event_generator(
    run_id: str,
    tracker: ProgressTracker,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for a model run's progress.
    
    Yields:
        Server-Sent Events formatted strings
    """
    # First, send the current state if available
    current_state = await tracker.get_current_state(run_id)
    if current_state:
        yield f"data: {current_state.to_json()}\n\n"
        
        # If already completed or failed, end the stream
        if current_state.phase in ["completed", "failed"]:
            yield f"event: close\ndata: {json.dumps({'reason': current_state.phase})}\n\n"
            return
    
    # Subscribe to updates
    try:
        async with tracker.subscribe(run_id) as subscriber:
            async for update in subscriber:
                # Send the update as an SSE event
                yield f"data: {update.to_json()}\n\n"
                
                # End stream on completion or failure
                if update.phase in ["completed", "failed"]:
                    yield f"event: close\ndata: {json.dumps({'reason': update.phase})}\n\n"
                    return
                    
    except asyncio.CancelledError:
        # Client disconnected
        yield f"event: close\ndata: {json.dumps({'reason': 'disconnected'})}\n\n"
        return
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return


@router.get("/{run_id}/stream")
async def stream_progress(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Stream progress updates for a model run using Server-Sent Events.
    
    Connect to this endpoint to receive real-time updates during model fitting.
    The stream will close automatically when the model completes or fails.
    
    Example client usage (JavaScript):
    ```javascript
    const eventSource = new EventSource('/api/v1/progress/{run_id}/stream');
    
    eventSource.onmessage = (event) => {
        const progress = JSON.parse(event.data);
        console.log(`Progress: ${progress.progress * 100}%`);
    };
    
    eventSource.addEventListener('close', (event) => {
        eventSource.close();
    });
    ```
    """
    # Verify the run exists
    run = await session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, f"Model run {run_id} not found")
    
    # If already completed or failed, return immediately
    if run.status in [ModelRunStatus.COMPLETED, ModelRunStatus.FAILED, ModelRunStatus.CANCELLED]:
        async def completed_generator():
            data = {
                "model_run_id": run_id,
                "progress": 1.0 if run.status == ModelRunStatus.COMPLETED else -1.0,
                "message": run.progress_message or str(run.status),
                "phase": run.status.value,
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield f"event: close\ndata: {json.dumps({'reason': run.status.value})}\n\n"
        
        return StreamingResponse(
            completed_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Create tracker and stream
    settings = get_settings()
    tracker = await ProgressTracker.create(settings.redis_url)
    
    async def stream_with_cleanup():
        try:
            async for event in progress_event_generator(run_id, tracker):
                yield event
        finally:
            await tracker.close()
    
    return StreamingResponse(
        stream_with_cleanup(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/{run_id}", response_model=ProgressResponse)
async def get_progress(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get the current progress for a model run.
    
    For real-time updates, use the /stream endpoint instead.
    """
    # Check database first
    run = await session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, f"Model run {run_id} not found")
    
    # Try to get live progress from Redis
    settings = get_settings()
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        current_state = await tracker.get_current_state(run_id)
        
        if current_state:
            return ProgressResponse(
                model_run_id=current_state.model_run_id,
                progress=current_state.progress,
                message=current_state.message,
                timestamp=current_state.timestamp,
                phase=current_state.phase,
                chain=current_state.chain,
                draw=current_state.draw,
                total_draws=current_state.total_draws,
                samples_per_second=current_state.samples_per_second,
                divergences=current_state.divergences,
                elapsed_seconds=current_state.elapsed_seconds,
                eta_seconds=current_state.eta_seconds,
                extra=current_state.extra,
            )
        else:
            # Fall back to database state
            from datetime import datetime, timezone
            return ProgressResponse(
                model_run_id=run_id,
                progress=run.progress,
                message=run.progress_message or str(run.status),
                timestamp=datetime.now(timezone.utc),
                phase=run.status.value,
            )
    finally:
        await tracker.close()