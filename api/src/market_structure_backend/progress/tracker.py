"""
Progress tracking system with Redis pub/sub and PyMC callbacks.

This module provides:
1. ProgressTracker - Central class for tracking and broadcasting progress
2. PyMCSamplingCallback - PyMC callback that reports MCMC sampling progress
3. Redis pub/sub for real-time progress updates to frontend
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio import Redis


@dataclass
class ProgressUpdate:
    """A single progress update message."""
    model_run_id: str
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # MCMC-specific fields
    phase: str = "running"  # "initializing", "tuning", "sampling", "completed", "failed"
    chain: Optional[int] = None
    draw: Optional[int] = None
    total_draws: Optional[int] = None
    
    # Performance metrics
    samples_per_second: Optional[float] = None
    divergences: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    
    # Extra model-specific data
    extra: Optional[dict] = None
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> "ProgressUpdate":
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


class ProgressTracker:
    """
    Central progress tracking with Redis pub/sub.
    
    Handles publishing progress updates and maintaining current state
    for each model run. Updates are published to Redis channels that
    SSE/WebSocket connections can subscribe to.
    """
    
    # Redis channel prefix for progress updates
    CHANNEL_PREFIX = "model_progress:"
    
    # Key prefix for storing latest progress state
    STATE_PREFIX = "model_state:"
    
    def __init__(self, redis: Redis):
        self.redis = redis
        self._start_times: dict[str, float] = {}
    
    @classmethod
    async def create(cls, redis_url: str) -> "ProgressTracker":
        """Create a ProgressTracker with a Redis connection."""
        redis = await aioredis.from_url(redis_url, decode_responses=True)
        return cls(redis)
    
    async def close(self):
        """Close the Redis connection."""
        await self.redis.close()
    
    def _channel_name(self, model_run_id: str) -> str:
        """Get the Redis channel name for a model run."""
        return f"{self.CHANNEL_PREFIX}{model_run_id}"
    
    def _state_key(self, model_run_id: str) -> str:
        """Get the Redis key for storing current state."""
        return f"{self.STATE_PREFIX}{model_run_id}"
    
    async def start_tracking(self, model_run_id: str):
        """Start tracking a new model run."""
        self._start_times[model_run_id] = time.time()
        
        initial_update = ProgressUpdate(
            model_run_id=model_run_id,
            progress=0.0,
            message="Initializing model...",
            phase="initializing",
        )
        await self.publish_update(initial_update)
    
    async def publish_update(self, update: ProgressUpdate):
        """Publish a progress update to Redis."""
        channel = self._channel_name(update.model_run_id)
        state_key = self._state_key(update.model_run_id)
        
        # Calculate elapsed time
        start_time = self._start_times.get(update.model_run_id)
        if start_time:
            update.elapsed_seconds = time.time() - start_time
        
        json_data = update.to_json()
        
        # Store current state (for late-joining subscribers)
        await self.redis.set(state_key, json_data, ex=3600)  # Expire after 1 hour
        
        # Publish to channel
        await self.redis.publish(channel, json_data)
    
    async def get_current_state(self, model_run_id: str) -> Optional[ProgressUpdate]:
        """Get the current progress state for a model run."""
        state_key = self._state_key(model_run_id)
        data = await self.redis.get(state_key)
        if data:
            return ProgressUpdate.from_json(data)
        return None
    
    async def complete(self, model_run_id: str, message: str = "Model completed successfully"):
        """Mark a model run as completed."""
        update = ProgressUpdate(
            model_run_id=model_run_id,
            progress=1.0,
            message=message,
            phase="completed",
        )
        await self.publish_update(update)
        
        # Clean up start time
        self._start_times.pop(model_run_id, None)
    
    async def fail(self, model_run_id: str, error_message: str):
        """Mark a model run as failed."""
        update = ProgressUpdate(
            model_run_id=model_run_id,
            progress=-1.0,  # Negative indicates failure
            message=error_message,
            phase="failed",
        )
        await self.publish_update(update)
        
        # Clean up start time
        self._start_times.pop(model_run_id, None)
    
    @asynccontextmanager
    async def subscribe(self, model_run_id: str):
        """
        Subscribe to progress updates for a model run.
        
        Usage:
            async with tracker.subscribe(run_id) as subscriber:
                async for update in subscriber:
                    print(update)
        """
        channel = self._channel_name(model_run_id)
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            yield ProgressSubscriber(pubsub, model_run_id)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()


class ProgressSubscriber:
    """Async iterator for progress updates."""
    
    def __init__(self, pubsub, model_run_id: str):
        self.pubsub = pubsub
        self.model_run_id = model_run_id
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> ProgressUpdate:
        while True:
            message = await self.pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=30.0  # 30 second timeout
            )
            if message is None:
                continue
            
            if message["type"] == "message":
                return ProgressUpdate.from_json(message["data"])


class PyMCSamplingCallback:
    """
    PyMC callback for tracking MCMC sampling progress.
    
    This callback is passed to PyMC's sample() function and reports
    progress at regular intervals. It works with both single-chain
    and multi-chain sampling.
    
    Usage:
        callback = PyMCSamplingCallback(
            model_run_id=run_id,
            tracker=progress_tracker,
            n_samples=1000,
            n_tune=500,
        )
        
        with pm.Model() as model:
            ...
            trace = pm.sample(
                draws=1000,
                tune=500,
                callback=callback,
            )
    """
    
    def __init__(
        self,
        model_run_id: str,
        tracker: ProgressTracker,
        n_samples: int,
        n_tune: int,
        n_chains: int = 4,
        update_interval: float = 1.0,  # Minimum seconds between updates
    ):
        self.model_run_id = model_run_id
        self.tracker = tracker
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.update_interval = update_interval
        
        self.total_draws = (n_tune + n_samples) * n_chains
        self.current_draw = 0
        self.divergences = 0
        self.start_time = None
        self.last_update_time = 0
        
        # Capture the main event loop at creation time (from async context)
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
    
    def _run_async(self, coro):
        """Run async code from sync context, safely across threads."""
        if self._main_loop is None:
            # No main loop captured, skip the update
            return
        
        if self._main_loop.is_running():
            # We're in a different thread, use thread-safe method
            future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
            # Don't wait for result - fire and forget
            def handle_error(f):
                try:
                    f.result()
                except Exception:
                    pass  # Silently ignore progress update errors
            future.add_done_callback(handle_error)
        else:
            # Loop exists but not running - this shouldn't happen in normal use
            pass
    
    def __call__(self, trace, draw):
        """
        Called by PyMC after each draw.
        
        Args:
            trace: Current trace object (contains samples so far)
            draw: Draw information tuple (chain, is_last, draw_info, point)
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Update draw counter
        self.current_draw += 1
        
        # Check if we should send an update (rate limiting)
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Calculate progress
        progress = self.current_draw / self.total_draws
        elapsed = current_time - self.start_time
        
        # Determine phase (tuning vs sampling)
        # PyMC provides draw info as (chain_idx, is_last, draw_idx, point)
        chain_idx = draw[0] if isinstance(draw, tuple) else 0
        draw_idx = draw[2] if isinstance(draw, tuple) and len(draw) > 2 else self.current_draw
        
        is_tuning = draw_idx < self.n_tune
        phase = "tuning" if is_tuning else "sampling"
        
        # Calculate samples per second
        samples_per_second = self.current_draw / elapsed if elapsed > 0 else 0
        
        # Estimate time remaining
        if samples_per_second > 0:
            remaining_draws = self.total_draws - self.current_draw
            eta_seconds = remaining_draws / samples_per_second
        else:
            eta_seconds = None
        
        # Count divergences if available
        try:
            if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
                self.divergences = int(trace.sample_stats['diverging'].sum())
        except:
            pass
        
        # Build progress message
        if is_tuning:
            message = f"Tuning chain {chain_idx + 1}/{self.n_chains}: {draw_idx}/{self.n_tune}"
        else:
            sample_idx = draw_idx - self.n_tune
            message = f"Sampling chain {chain_idx + 1}/{self.n_chains}: {sample_idx}/{self.n_samples}"
        
        # Create and publish update
        update = ProgressUpdate(
            model_run_id=self.model_run_id,
            progress=progress,
            message=message,
            phase=phase,
            chain=chain_idx,
            draw=draw_idx,
            total_draws=self.total_draws,
            samples_per_second=samples_per_second,
            divergences=self.divergences,
            eta_seconds=eta_seconds,
        )
        
        # Publish update (async from sync context)
        self._run_async(self.tracker.publish_update(update))


class EMProgressCallback:
    """
    Callback for tracking EM algorithm progress (LCA, Factor Analysis, etc.).
    
    Reports progress based on iteration count and convergence metrics.
    This callback is designed to be called from a thread pool executor
    and safely publishes updates to the main event loop.
    """
    
    def __init__(
        self,
        model_run_id: str,
        tracker: ProgressTracker,
        max_iter: int,
        update_interval: float = 0.5,
    ):
        self.model_run_id = model_run_id
        self.tracker = tracker
        self.max_iter = max_iter
        self.update_interval = update_interval
        
        self.current_iter = 0
        self.start_time = None
        self.last_update_time = 0
        
        # Capture the main event loop at creation time (from async context)
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
    
    def _run_async(self, coro):
        """Run async code from sync context, safely across threads."""
        if self._main_loop is None:
            # No main loop captured, skip the update
            return
        
        if self._main_loop.is_running():
            # We're in a different thread, use thread-safe method
            future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
            # Don't wait for result - fire and forget
            # But add a callback to log errors
            def handle_error(f):
                try:
                    f.result()
                except Exception as e:
                    pass  # Silently ignore progress update errors
            future.add_done_callback(handle_error)
        else:
            # Loop exists but not running - this shouldn't happen in normal use
            pass
    
    def __call__(
        self,
        iteration: int,
        log_likelihood: Optional[float] = None,
        delta: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        """
        Report progress for an EM iteration.
        
        Args:
            iteration: Current iteration number
            log_likelihood: Current log-likelihood value
            delta: Change in log-likelihood from previous iteration
            extra: Additional model-specific metrics
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self.current_iter = iteration
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Calculate progress
        progress = min(iteration / self.max_iter, 1.0) if self.max_iter > 0 else 0.0
        elapsed = current_time - self.start_time
        
        # Build message
        message_parts = [f"Iteration {iteration}/{self.max_iter}"]
        if log_likelihood is not None:
            message_parts.append(f"LL: {log_likelihood:.2f}")
        if delta is not None:
            message_parts.append(f"Δ: {delta:.6f}")
        message = " | ".join(message_parts)
        
        # ETA
        if elapsed > 0 and progress > 0:
            eta_seconds = elapsed * (1 - progress) / progress
        else:
            eta_seconds = None
        
        update = ProgressUpdate(
            model_run_id=self.model_run_id,
            progress=progress,
            message=message,
            phase="running",
            eta_seconds=eta_seconds,
            extra={
                "iteration": iteration,
                "log_likelihood": log_likelihood,
                "delta": delta,
                **(extra or {}),
            },
        )
        
        self._run_async(self.tracker.publish_update(update))