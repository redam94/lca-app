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
import zipfile
import json

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
    - Model-specific results (loadings, class probs, etc.)
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
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # ==========================================
    # Initialize all fields
    # ==========================================
    product_embeddings = None
    household_embeddings = None
    similarity_matrix = None
    variance_explained = None
    loadings = None
    loadings_std = None
    item_probs = None
    class_probs = None
    alpha = None
    alpha_std = None
    product_latent = None
    household_latent = None
    
    # ==========================================
    # Model-specific extraction
    # ==========================================
    model_type = run.model_type.value if isinstance(run.model_type, ModelType) else run.model_type
    
    if model_type in ["lca", "lca_covariates"]:
        # LCA models
        item_probs_raw = results.get("item_probs")
        if item_probs_raw is not None:
            item_probs = to_list(item_probs_raw)  # (n_classes, n_items)
            # Product embeddings = transpose of item_probs for biplot
            product_embeddings = to_list(item_probs_raw.T)
        
        class_probs = to_list(results.get("class_probs"))
        household_embeddings = to_list(results.get("responsibilities"))
        
        # Similarity from residual correlations
        similarity_matrix = to_list(results.get("residual_correlations"))
        
        # For LCA, variance explained = class proportions as percentages
        if class_probs is not None:
            variance_explained = [p * 100 for p in class_probs]
    
    elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc"]:
        # Factor models
        loadings = to_list(results.get("loadings"))
        loadings_std = to_list(results.get("loadings_std"))
        variance_explained = to_list(results.get("var_explained_pct"))
        
        # Product embeddings = loadings for biplot
        product_embeddings = loadings
        household_embeddings = to_list(results.get("scores"))
        
        # Compute similarity from loadings
        if results.get("loadings") is not None:
            loadings_np = results["loadings"]
            loadings_norm = loadings_np / (np.linalg.norm(loadings_np, axis=1, keepdims=True) + 1e-10)
            similarity_matrix = (loadings_norm @ loadings_norm.T).tolist()
    
    elif model_type == "nmf":
        # NMF
        loadings = to_list(results.get("loadings"))  # H.T
        variance_explained = to_list(results.get("var_explained_pct"))
        
        product_embeddings = loadings
        household_embeddings = to_list(results.get("scores"))  # W
        
        # Similarity from H matrix
        if results.get("H") is not None:
            H = results["H"]
            H_norm = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-10)
            similarity_matrix = (H_norm.T @ H_norm).tolist()
    
    elif model_type == "mca":
        # *** FIXED: Use correct field names ***
        col_coords = results.get("column_coordinates")  # was "column_coords"
        row_coords = results.get("row_coordinates")      # was "row_coords"
        
        product_embeddings = to_list(col_coords)
        household_embeddings = to_list(row_coords)
        variance_explained = to_list(results.get("var_explained_pct"))
        similarity_matrix = to_list(results.get("similarity_matrix"))
        
        # Use column_coordinates as loadings
        loadings = product_embeddings
        
        # MCA may filter products - use product_labels if available
        mca_product_labels = results.get("product_labels")
        if mca_product_labels and len(mca_product_labels) > 0:
            product_columns = list(mca_product_labels)
    
    elif model_type == "dcm":
        # Discrete Choice Model
        alpha = to_list(results.get("alpha"))
        alpha_std = to_list(results.get("alpha_std"))
        product_latent = to_list(results.get("product_latent"))
        household_latent = to_list(results.get("household_latent"))
        
        # Use product_latent as embeddings and loadings
        product_embeddings = product_latent
        household_embeddings = household_latent
        loadings = product_latent
        
        # Similarity from product latent
        if results.get("product_latent") is not None:
            pl = results["product_latent"]
            pl_norm = pl / (np.linalg.norm(pl, axis=1, keepdims=True) + 1e-10)
            similarity_matrix = (pl_norm @ pl_norm.T).tolist()
    
    # ==========================================
    # Clean results for JSON serialization
    # ==========================================
    clean_results = {}
    for key, value in results.items():
        if key in ["trace", "waic"]:  # Skip large/complex objects
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
        # Embeddings
        product_embeddings=product_embeddings,
        household_embeddings=household_embeddings,
        similarity_matrix=similarity_matrix,
        # Renamed field: var_explained_pct -> variance_explained
        variance_explained=variance_explained,
        # Factor model fields
        loadings=loadings,
        loadings_std=loadings_std,
        # LCA fields
        item_probs=item_probs,
        class_probs=class_probs,
        # DCM fields
        alpha=alpha,
        alpha_std=alpha_std,
        product_latent=product_latent,
        household_latent=household_latent,
        # Metadata
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

@router.get("/{run_id}/export")
async def export_model_results(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Export model results as a downloadable ZIP file.
    
    The ZIP contains:
    - product_embeddings.csv
    - household_embeddings.csv (if available)
    - similarity_matrix.csv (if available)
    - model_summary.json
    - README.md
    """
    from fastapi.responses import StreamingResponse
    
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
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    model_type = run.model_type.value if isinstance(run.model_type, ModelType) else run.model_type
    product_columns = run.product_columns or []
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        # Product embeddings
        product_embeddings = None
        if model_type in ["lca", "lca_covariates"]:
            if "item_probs" in results:
                product_embeddings = results["item_probs"].T
        elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc", "nmf"]:
            product_embeddings = results.get("loadings")
        elif model_type == "dcm":
            product_embeddings = results.get("product_latent")
        elif model_type == "mca":
            product_embeddings = results.get("column_coords")
        
        if product_embeddings is not None:
            n_dims = product_embeddings.shape[1]
            header = "product," + ",".join([f"Dim_{i+1}" for i in range(n_dims)])
            rows = [header]
            for i, prod in enumerate(product_columns):
                row_vals = ",".join([f"{v:.6f}" for v in product_embeddings[i]])
                rows.append(f"{prod},{row_vals}")
            zf.writestr('product_embeddings.csv', "\n".join(rows))
        
        # Household embeddings
        household_embeddings = None
        if model_type in ["lca", "lca_covariates"]:
            household_embeddings = results.get("responsibilities")
        elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc", "nmf"]:
            household_embeddings = results.get("scores")
        elif model_type == "dcm":
            household_embeddings = results.get("household_latent")
        elif model_type == "mca":
            household_embeddings = results.get("row_coords")
        
        if household_embeddings is not None:
            n_dims = household_embeddings.shape[1]
            header = "household," + ",".join([f"Dim_{i+1}" for i in range(n_dims)])
            rows = [header]
            for i in range(len(household_embeddings)):
                row_vals = ",".join([f"{v:.6f}" for v in household_embeddings[i]])
                rows.append(f"HH_{i+1},{row_vals}")
            zf.writestr('household_embeddings.csv', "\n".join(rows))
        
        # Similarity matrix
        similarity_matrix = None
        if model_type in ["lca", "lca_covariates"]:
            similarity_matrix = results.get("residual_correlations")
        elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc"]:
            if results.get("loadings") is not None:
                loadings = results["loadings"]
                loadings_norm = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-10)
                similarity_matrix = loadings_norm @ loadings_norm.T
        elif model_type == "nmf":
            if results.get("H") is not None:
                H = results["H"]
                H_norm = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-10)
                similarity_matrix = H_norm.T @ H_norm
        elif model_type == "mca":
            similarity_matrix = results.get("similarity_matrix")
        
        if similarity_matrix is not None:
            header = "," + ",".join(product_columns)
            rows = [header]
            for i, prod in enumerate(product_columns):
                row_vals = ",".join([f"{v:.6f}" for v in similarity_matrix[i]])
                rows.append(f"{prod},{row_vals}")
            zf.writestr('similarity_matrix.csv', "\n".join(rows))
        
        # Variance explained
        var_explained = results.get("var_explained_pct")
        if var_explained is not None:
            header = "component,variance_explained_pct,cumulative_pct"
            rows = [header]
            cumulative = 0
            for i, v in enumerate(var_explained):
                cumulative += v
                rows.append(f"Component_{i+1},{v:.4f},{cumulative:.4f}")
            zf.writestr('variance_explained.csv', "\n".join(rows))
        
        # Model summary JSON
        model_summary = {
            "model_type": model_type,
            "run_id": run_id,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "parameters": run.model_params,
            "data_shape": run.data_shape,
            "product_columns": product_columns,
            "metrics": {}
        }
        
        # Add model-specific metrics
        if "log_likelihood" in results:
            model_summary["metrics"]["log_likelihood"] = float(results["log_likelihood"])
        if "bic" in results:
            model_summary["metrics"]["bic"] = float(results["bic"])
        if "aic" in results:
            model_summary["metrics"]["aic"] = float(results["aic"])
        if "n_iter" in results:
            model_summary["metrics"]["n_iterations"] = int(results["n_iter"])
        if "reconstruction_error" in results:
            model_summary["metrics"]["reconstruction_error"] = float(results["reconstruction_error"])
        if "n_divergences" in results:
            model_summary["metrics"]["n_divergences"] = int(results["n_divergences"])
        if run.metrics:
            model_summary["metrics"].update(run.metrics)
        
        zf.writestr('model_summary.json', json.dumps(model_summary, indent=2))
        
        # README
        readme = f"""# Market Structure Analysis Export

## Model: {model_type}
## Run ID: {run_id}
## Products: {len(product_columns)}

## Files Included

- `product_embeddings.csv` - Product coordinates in latent space
- `household_embeddings.csv` - Household scores/coordinates
- `similarity_matrix.csv` - Product similarity matrix
- `variance_explained.csv` - Variance explained by each component
- `model_summary.json` - Model parameters and metrics

## Usage in Python

```python
import pandas as pd

# Load embeddings
products = pd.read_csv('product_embeddings.csv', index_col='product')
households = pd.read_csv('household_embeddings.csv', index_col='household')
similarity = pd.read_csv('similarity_matrix.csv', index_col=0)

# Find similar products
def find_similar(product, top_k=5):
    return similarity[product].nlargest(top_k + 1)[1:]
```

## Usage in R

```r
library(tidyverse)

products <- read_csv('product_embeddings.csv')
similarity <- read_csv('similarity_matrix.csv') %>%
  column_to_rownames(var = names(.)[1])

# Hierarchical clustering
hc <- hclust(as.dist(1 - as.matrix(similarity)))
plot(hc)
```
"""
        zf.writestr('README.md', readme)
    
    zip_buffer.seek(0)
    
    # Generate filename
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = f"{model_type}_results_{timestamp}.zip"
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
