"""
FastAPI routes for model run management.

Provides endpoints for:
- Submitting new model runs
- Checking run status
- Retrieving results
- Listing runs with filtering
- Cancelling runs
"""

import base64
import io
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import select, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import ModelRun, ModelRunStatus, ModelType, get_session, get_session_factory
from ...schemas import (
    ModelRunRequest,
    ModelRunResponse,
    ModelRunListResponse,
    ModelResultsResponse,
    ModelTypeEnum,
    ModelRunStatusEnum,
    ErrorResponse,
)
from ...workers import enqueue_job, TASK_REGISTRY


router = APIRouter(prefix="/runs", tags=["Model Runs"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_data(request: ModelRunRequest) -> tuple[np.ndarray, list[str]]:
    """Parse data from request."""
    if request.data is None and request.data_id is None:
        raise HTTPException(400, "Either data or data_id must be provided")
    
    if request.data is not None:
        if request.data.csv_base64:
            # Decode base64 CSV
            csv_bytes = base64.b64decode(request.data.csv_base64)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            
            # Use product_columns if specified, otherwise use all numeric columns
            if request.product_columns:
                columns = request.product_columns
            else:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            data = df[columns].values.astype(float)
            return data, columns
        
        elif request.data.data_json:
            data = np.array(request.data.data_json)
            columns = request.product_columns or [f"Product_{i}" for i in range(data.shape[1])]
            return data, columns
    
    # TODO: Handle data_id reference to previously uploaded data
    raise HTTPException(400, "Data parsing failed")


def _model_type_to_enum(model_type: ModelTypeEnum) -> ModelType:
    """Convert schema enum to database enum."""
    return ModelType(model_type.value)


def _model_run_to_response(run: ModelRun) -> ModelRunResponse:
    """Convert database model to response schema."""
    return ModelRunResponse(
        id=run.id,
        model_type=ModelTypeEnum(run.model_type.value if isinstance(run.model_type, ModelType) else run.model_type),
        name=run.name,
        description=run.description,
        status=ModelRunStatusEnum(run.status.value if isinstance(run.status, ModelRunStatus) else run.status),
        created_at=run.created_at,
        queued_at=run.queued_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        queue_duration=run.queue_duration,
        run_duration=run.run_duration,
        model_params=run.model_params,
        data_shape=run.data_shape,
        product_columns=run.product_columns,
        progress=run.progress,
        progress_message=run.progress_message,
        results_summary=run.results_summary,
        metrics=run.metrics,
        error_message=run.error_message,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("", response_model=ModelRunResponse, status_code=201)
async def submit_model_run(
    request: ModelRunRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Submit a new model run for background processing.
    
    The run will be queued for processing by an ARQ worker.
    Use the returned run ID to check status and retrieve results.
    """
    # Parse the data
    try:
        data, product_columns = _parse_data(request)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse data: {str(e)}")
    
    # Validate model type is supported
    model_type = request.model_type.value
    if model_type not in TASK_REGISTRY:
        raise HTTPException(400, f"Unsupported model type: {model_type}")
    
    # Create database record
    run_id = str(uuid4())
    now = datetime.now(timezone.utc)
    
    model_run = ModelRun(
        id=run_id,
        model_type=_model_type_to_enum(request.model_type),
        name=request.name,
        description=request.description,
        status=ModelRunStatus.QUEUED,
        created_at=now,
        queued_at=now,
        model_params=request.params,
        data_shape={"n_obs": data.shape[0], "n_items": data.shape[1]},
        product_columns=product_columns,
        progress=0.0,
        progress_message="Queued for processing",
    )
    
    session.add(model_run)
    await session.commit()
    await session.refresh(model_run)
    
    # Enqueue the task
    try:
        task_func = TASK_REGISTRY[model_type]
        task_name = task_func.__name__
        
        await enqueue_job(
            task_name,
            run_id,
            data.tolist(),  # Convert to list for JSON serialization
            request.params,
            product_columns,
            job_id=run_id,
        )
        
        # Update with job ID
        model_run.arq_job_id = run_id
        await session.commit()
        
    except Exception as e:
        # Mark as failed if we couldn't enqueue
        model_run.status = ModelRunStatus.FAILED
        model_run.error_message = f"Failed to enqueue job: {str(e)}"
        await session.commit()
        raise HTTPException(500, f"Failed to enqueue job: {str(e)}")
    
    return _model_run_to_response(model_run)


@router.get("", response_model=ModelRunListResponse)
async def list_model_runs(
    status: Optional[ModelRunStatusEnum] = None,
    model_type: Optional[ModelTypeEnum] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    order_by: str = Query(default="created_at", pattern="^(created_at|completed_at|name)$"),
    order_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
    session: AsyncSession = Depends(get_session),
):
    """
    List model runs with optional filtering.
    
    Supports filtering by status and model type, with pagination.
    """
    # Build query
    query = select(ModelRun)
    count_query = select(func.count(ModelRun.id))
    
    # Apply filters
    if status is not None:
        query = query.where(ModelRun.status == ModelRunStatus(status.value))
        count_query = count_query.where(ModelRun.status == ModelRunStatus(status.value))
    
    if model_type is not None:
        query = query.where(ModelRun.model_type == ModelType(model_type.value))
        count_query = count_query.where(ModelRun.model_type == ModelType(model_type.value))
    
    # Get total count
    total = await session.scalar(count_query)
    
    # Apply ordering
    order_column = getattr(ModelRun, order_by)
    if order_dir == "desc":
        query = query.order_by(desc(order_column))
    else:
        query = query.order_by(asc(order_column))
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    # Execute
    result = await session.execute(query)
    runs = result.scalars().all()
    
    return ModelRunListResponse(
        runs=[_model_run_to_response(run) for run in runs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{run_id}", response_model=ModelRunResponse)
async def get_model_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get details of a specific model run."""
    run = await session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, f"Model run {run_id} not found")
    
    return _model_run_to_response(run)


@router.get("/{run_id}/results", response_model=ModelResultsResponse)
async def get_model_results(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get full results for a completed model run.
    
    This includes all computed values needed for visualization:
    - Product/household embeddings
    - Similarity matrices
    - Variance explained
    - Model-specific results
    """
    run = await session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, f"Model run {run_id} not found")
    
    if run.status != ModelRunStatus.COMPLETED:
        raise HTTPException(400, f"Model run is not completed (status: {run.status})")
    
    if run.results_path is None:
        raise HTTPException(404, "Results not available")
    
    # Load full results from disk
    results_path = Path(run.results_path)
    if not results_path.exists():
        raise HTTPException(404, "Results file not found")
    
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    
    # Convert numpy arrays to lists
    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Extract common fields
    product_embeddings = None
    household_embeddings = None
    similarity_matrix = None
    var_explained_pct = None
    
    # Model-specific extraction
    model_type = run.model_type.value if isinstance(run.model_type, ModelType) else run.model_type
    
    if model_type in ["lca", "lca_covariates"]:
        product_embeddings = to_list(results.get("item_probs", np.array([])).T)
        household_embeddings = to_list(results.get("responsibilities"))
        if "class_probs" in results:
            var_explained_pct = (np.array(results["class_probs"]) * 100).tolist()
    
    elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc"]:
        product_embeddings = to_list(results.get("loadings"))
        household_embeddings = to_list(results.get("scores"))
        var_explained_pct = to_list(results.get("var_explained_pct"))
        
        # Compute similarity from loadings
        if results.get("loadings") is not None:
            loadings = results["loadings"]
            loadings_norm = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-10)
            similarity_matrix = (loadings_norm @ loadings_norm.T).tolist()
    
    elif model_type == "nmf":
        product_embeddings = to_list(results.get("loadings"))
        household_embeddings = to_list(results.get("scores"))
        var_explained_pct = to_list(results.get("var_explained_pct"))
        
        if results.get("H") is not None:
            H = results["H"]
            H_norm = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-10)
            similarity_matrix = (H_norm.T @ H_norm).tolist()
    
    elif model_type == "dcm":
        product_embeddings = to_list(results.get("product_latent"))
        household_embeddings = to_list(results.get("household_latent"))
        
        if results.get("product_latent") is not None:
            pl = results["product_latent"]
            pl_norm = pl / (np.linalg.norm(pl, axis=1, keepdims=True) + 1e-10)
            similarity_matrix = (pl_norm @ pl_norm.T).tolist()
    
    # Clean results for JSON serialization
    clean_results = {}
    for key, value in results.items():
        if key in ["trace"]:  # Skip large objects
            continue
        if isinstance(value, np.ndarray):
            clean_results[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            clean_results[key] = float(value)
        elif hasattr(value, "__dict__"):  # Skip complex objects
            continue
        else:
            clean_results[key] = value
    
    return ModelResultsResponse(
        model_run_id=run_id,
        model_type=ModelTypeEnum(model_type),
        status=ModelRunStatusEnum(run.status.value),
        results=clean_results,
        product_embeddings=product_embeddings,
        household_embeddings=household_embeddings,
        similarity_matrix=similarity_matrix,
        var_explained_pct=var_explained_pct,
        product_columns=run.product_columns,
        metrics=run.metrics,
    )


@router.delete("/{run_id}", status_code=204)
async def delete_model_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a model run and its results."""
    run = await session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, f"Model run {run_id} not found")
    
    # Delete results file if it exists
    if run.results_path:
        results_path = Path(run.results_path)
        if results_path.exists():
            results_path.unlink()
    
    # Delete database record
    await session.delete(run)
    await session.commit()


@router.post("/{run_id}/cancel", response_model=ModelRunResponse)
async def cancel_model_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Cancel a queued or running model run.
    
    Note: If the run is already processing, it may not stop immediately.
    """
    run = await session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, f"Model run {run_id} not found")
    
    if run.status in [ModelRunStatus.COMPLETED, ModelRunStatus.FAILED, ModelRunStatus.CANCELLED]:
        raise HTTPException(400, f"Cannot cancel run with status: {run.status}")
    
    # Update status
    run.status = ModelRunStatus.CANCELLED
    run.completed_at = datetime.now(timezone.utc)
    run.progress_message = "Cancelled by user"
    
    await session.commit()
    await session.refresh(run)
    
    # TODO: Actually cancel the ARQ job if possible
    
    return _model_run_to_response(run)