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
from fastapi.responses import StreamingResponse, HTMLResponse
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


# Input data storage directory (data saved to disk before sending to worker)
INPUT_DATA_DIR = Path("./model_results/input_data")
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_input_data(
    run_id: str,
    data: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    covariate_columns: Optional[list[str]] = None,
) -> str:
    """
    Save input data to disk before enqueuing to worker.

    This avoids serializing large numpy arrays through Redis,
    which can fail for large datasets due to memory limits.

    Returns:
        File path string to the saved input data.
    """
    data_path = INPUT_DATA_DIR / f"{run_id}_input.pkl"
    payload = {"data": data}
    if covariates is not None:
        payload["covariates"] = covariates
    if covariate_columns is not None:
        payload["covariate_columns"] = covariate_columns
    with open(data_path, "wb") as f:
        pickle.dump(payload, f)
    return str(data_path)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_data(request: ModelRunRequest) -> tuple[np.ndarray, list[str], np.ndarray | None, list[str] | None]:
    """
    Parse data from request.

    Returns:
        Tuple of (data, product_columns, covariates, covariate_columns)
        covariates and covariate_columns may be None if not provided
    """
    if request.data is None and request.data_id is None:
        raise HTTPException(400, "Either data or data_id must be provided")

    covariates = None
    covariate_columns = None

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

            # Parse covariates if column names provided
            if request.data.covariate_column_names:
                covariate_columns = request.data.covariate_column_names
                covariates = df[covariate_columns].values.astype(float)

            return data, columns, covariates, covariate_columns

        elif request.data.data_json:
            data = np.array(request.data.data_json)
            columns = request.product_columns or [f"Product_{i}" for i in range(data.shape[1])]

            # Parse covariates from JSON if provided
            if request.data.covariates_json:
                covariates = np.array(request.data.covariates_json)
                covariate_columns = request.data.covariate_column_names or [
                    f"Covariate_{i}" for i in range(covariates.shape[1])
                ]

            return data, columns, covariates, covariate_columns

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
        data, product_columns, covariates, covariate_columns = _parse_data(request)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse data: {str(e)}")

    # Validate model type is supported
    model_type = request.model_type.value
    if model_type not in TASK_REGISTRY:
        raise HTTPException(400, f"Unsupported model type: {model_type}")

    # Validate covariates for LCA with covariates
    if model_type == "lca_covariates" and covariates is None:
        raise HTTPException(400, "LCA with covariates requires covariate data")

    # Create database record
    run_id = str(uuid4())
    now = datetime.now(timezone.utc)

    data_shape = {"n_obs": data.shape[0], "n_items": data.shape[1]}
    if covariates is not None:
        data_shape["n_covariates"] = covariates.shape[1]

    model_run = ModelRun(
        id=run_id,
        model_type=_model_type_to_enum(request.model_type),
        name=request.name,
        description=request.description,
        status=ModelRunStatus.QUEUED,
        created_at=now,
        queued_at=now,
        model_params=request.params,
        data_shape=data_shape,
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

        # Save input data to disk to avoid serializing large arrays through Redis
        data_path = _save_input_data(
            run_id,
            data,
            covariates=covariates if model_type in ["lca_covariates", "dcm"] else None,
            covariate_columns=covariate_columns if model_type == "lca_covariates" else None,
        )

        # All model types use the same enqueue signature: (run_id, data_path, params, product_columns)
        await enqueue_job(
            task_name,
            run_id,
            data_path,
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
    
    # Additional model-specific fields
    residual_correlations = None
    tetra_corr = None
    elbo_history = None
    beta = None
    odds_ratios = None
    covariate_columns = None
    class_probs_per_hh = None

    # LDA-specific fields
    topic_product_dist = None
    household_topic_dist = None
    perplexity = None
    n_topics = None

    # Network-specific fields
    adjacency_matrix = None
    communities = None
    n_communities = None
    centrality_scores = None
    degree_centrality = None
    betweenness_centrality = None
    graph_metrics = None

    if model_type in ["lca", "lca_covariates"]:
        # LCA models
        item_probs_raw = results.get("item_probs")
        if item_probs_raw is not None:
            item_probs = to_list(item_probs_raw)  # (n_classes, n_items)
            # Product embeddings = transpose of item_probs for biplot
            product_embeddings = to_list(item_probs_raw.T)
            # Loadings = transposed item_probs (n_items, n_classes) for factor loadings heatmap
            loadings = to_list(item_probs_raw.T)

        class_probs = to_list(results.get("class_probs"))
        household_embeddings = to_list(results.get("responsibilities"))

        # Residual correlations for similarity
        residual_corr_raw = results.get("residual_correlations")
        if residual_corr_raw is not None:
            residual_correlations = to_list(residual_corr_raw)
            similarity_matrix = residual_correlations

        # For LCA, variance explained = class proportions as percentages
        if class_probs is not None:
            variance_explained = [p * 100 for p in class_probs]

        # LCA with covariates specific fields
        if model_type == "lca_covariates":
            beta = to_list(results.get("beta"))
            odds_ratios = to_list(results.get("odds_ratios"))
            covariate_columns = results.get("covariate_columns")
            class_probs_per_hh = to_list(results.get("class_probs_per_hh"))

            # For LCA with covariates, use mean class probs for variance explained
            if class_probs_per_hh is not None and class_probs is None:
                mean_class_probs = np.array(class_probs_per_hh).mean(axis=0)
                class_probs = mean_class_probs.tolist()
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

        # Tetrachoric correlation matrix (for tetrachoric FA)
        if model_type == "factor_tetrachoric":
            tetra_corr = to_list(results.get("tetra_corr"))

        # ELBO history (for Bayesian VI)
        if model_type == "bayesian_factor_vi":
            elbo_history = to_list(results.get("elbo_history"))
    
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
        
        # MCA may filter products - use product_labels only if they're real names
        # (not the fallback "item_0", "item_1" pattern from missing product_names)
        mca_product_labels = results.get("product_labels")
        if mca_product_labels and len(mca_product_labels) > 0:
            has_real_names = not all(
                label.startswith("item_") and label[5:].isdigit()
                for label in mca_product_labels
            )
            if has_real_names:
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

    elif model_type == "lda":
        # Latent Dirichlet Allocation
        topic_product_dist = to_list(results.get("topic_product_dist"))
        household_topic_dist = to_list(results.get("household_topic_dist"))
        perplexity = results.get("perplexity")
        n_topics = results.get("n_topics")

        # Use loadings and scores for biplot compatibility
        loadings = to_list(results.get("loadings"))  # topic_product_dist.T
        product_embeddings = loadings
        household_embeddings = to_list(results.get("scores"))  # household_topic_dist
        variance_explained = to_list(results.get("var_explained_pct"))

        # Compute similarity from topic distributions
        if results.get("topic_product_dist") is not None:
            tpd = results["topic_product_dist"].T  # (n_products, n_topics)
            tpd_norm = tpd / (np.linalg.norm(tpd, axis=1, keepdims=True) + 1e-10)
            similarity_matrix = (tpd_norm @ tpd_norm.T).tolist()

    elif model_type == "network":
        # Network Analysis
        adjacency_matrix = to_list(results.get("adjacency_matrix"))
        communities = results.get("communities")
        n_communities = results.get("n_communities")
        centrality_scores = to_list(results.get("centrality_scores"))
        degree_centrality = to_list(results.get("degree_centrality"))
        betweenness_centrality = to_list(results.get("betweenness_centrality"))
        graph_metrics = results.get("graph_metrics")

        # Use loadings and scores for biplot compatibility
        loadings = to_list(results.get("loadings"))  # community membership
        product_embeddings = loadings
        household_embeddings = to_list(results.get("scores"))  # household community scores
        variance_explained = to_list(results.get("var_explained_pct"))

        # Use adjacency matrix as similarity
        similarity_matrix = adjacency_matrix

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
        # LCA with covariates fields
        beta=beta,
        odds_ratios=odds_ratios,
        covariate_columns=covariate_columns,
        class_probs_per_hh=class_probs_per_hh,
        # Additional model-specific fields
        residual_correlations=residual_correlations,
        tetra_corr=tetra_corr,
        elbo_history=elbo_history,
        # LDA-specific fields
        topic_product_dist=topic_product_dist,
        household_topic_dist=household_topic_dist,
        perplexity=perplexity,
        n_topics=n_topics,
        # Network-specific fields
        adjacency_matrix=adjacency_matrix,
        communities=communities,
        centrality_scores=centrality_scores,
        degree_centrality=degree_centrality,
        betweenness_centrality=betweenness_centrality,
        graph_metrics=graph_metrics,
        n_communities=n_communities,
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


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def _get_model_display_name(model_type: str) -> str:
    """Get display name for model type."""
    names = {
        "lca": "Latent Class Analysis",
        "lca_covariates": "LCA with Covariates",
        "factor_tetrachoric": "Factor Analysis (Tetrachoric)",
        "bayesian_factor_vi": "Bayesian Factor (VI)",
        "bayesian_factor_pymc": "Bayesian Factor (PyMC)",
        "nmf": "Non-negative Matrix Factorization",
        "mca": "Multiple Correspondence Analysis",
        "dcm": "Discrete Choice Model",
        "lda": "Latent Dirichlet Allocation",
        "network": "Network Analysis",
    }
    return names.get(model_type, model_type)


def _extract_report_data(results: dict, model_type: str, product_columns: list) -> dict:
    """
    Extract and compute all data needed for the report from raw results.
    This mirrors the logic in get_model_results() endpoint.
    """
    extracted = {
        "product_columns": product_columns,
        "model_type": model_type,
        "similarity_matrix": None,
        "loadings": None,
        "loadings_std": None,
        "variance_explained": None,
        "product_embeddings": None,
        "item_probs": None,
        "class_probs": None,
        "tetra_corr": None,
        "elbo_history": None,
        "topic_product_dist": None,
        "adjacency_matrix": None,
        "communities": None,
        "centrality_scores": None,
        "degree_centrality": None,
        "betweenness_centrality": None,
        "edge_list": None,
        "alpha": None,
        "alpha_std": None,
    }

    if model_type in ["lca", "lca_covariates"]:
        # LCA models
        item_probs_raw = results.get("item_probs")
        if item_probs_raw is not None:
            extracted["item_probs"] = np.array(item_probs_raw)
            extracted["product_embeddings"] = np.array(item_probs_raw).T
            extracted["loadings"] = np.array(item_probs_raw).T  # (n_items, n_classes) for loadings heatmap

        class_probs_raw = results.get("class_probs")
        if class_probs_raw is not None:
            extracted["class_probs"] = np.array(class_probs_raw)
            extracted["variance_explained"] = np.array(class_probs_raw) * 100
        elif model_type == "lca_covariates":
            # Fallback: compute mean class probs from per-household probs
            # LCA with covariates returns class_probs_per_hh instead of global class_probs
            class_probs_per_hh = results.get("class_probs_per_hh")
            if class_probs_per_hh is not None:
                mean_class_probs = np.array(class_probs_per_hh).mean(axis=0)
                extracted["class_probs"] = mean_class_probs
                extracted["variance_explained"] = mean_class_probs * 100

        # Residual correlations for similarity
        residual_corr = results.get("residual_correlations")
        if residual_corr is not None:
            extracted["similarity_matrix"] = np.array(residual_corr)

    elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc"]:
        # Factor models
        loadings_raw = results.get("loadings")
        if loadings_raw is not None:
            loadings = np.array(loadings_raw)
            extracted["loadings"] = loadings
            extracted["product_embeddings"] = loadings

            # Compute similarity from loadings
            loadings_norm = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-10)
            extracted["similarity_matrix"] = loadings_norm @ loadings_norm.T

        loadings_std = results.get("loadings_std")
        if loadings_std is not None:
            extracted["loadings_std"] = np.array(loadings_std)

        var_explained = results.get("var_explained_pct")
        if var_explained is not None:
            extracted["variance_explained"] = np.array(var_explained)

        # Tetrachoric correlation matrix
        if model_type == "factor_tetrachoric":
            tetra = results.get("tetra_corr")
            if tetra is not None:
                extracted["tetra_corr"] = np.array(tetra)

        # ELBO history
        if model_type == "bayesian_factor_vi":
            elbo = results.get("elbo_history")
            if elbo is not None:
                extracted["elbo_history"] = list(elbo)

    elif model_type == "nmf":
        loadings_raw = results.get("loadings")
        if loadings_raw is not None:
            loadings = np.array(loadings_raw)
            extracted["loadings"] = loadings
            extracted["product_embeddings"] = loadings

        var_explained = results.get("var_explained_pct")
        if var_explained is not None:
            extracted["variance_explained"] = np.array(var_explained)

        # Similarity from H matrix
        H = results.get("H")
        if H is not None:
            H = np.array(H)
            H_norm = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-10)
            extracted["similarity_matrix"] = H_norm.T @ H_norm

    elif model_type == "mca":
        col_coords = results.get("column_coordinates")
        if col_coords is not None:
            coords = np.array(col_coords)
            extracted["loadings"] = coords
            extracted["product_embeddings"] = coords

        var_explained = results.get("var_explained_pct")
        if var_explained is not None:
            extracted["variance_explained"] = np.array(var_explained)

        sim = results.get("similarity_matrix")
        if sim is not None:
            extracted["similarity_matrix"] = np.array(sim)

        # MCA may filter products - use product_labels only if they're real names
        # (not the fallback "item_0", "item_1" pattern from missing product_names)
        mca_labels = results.get("product_labels")
        if mca_labels and len(mca_labels) > 0:
            # Check if labels are real names or fallback pattern
            has_real_names = not all(
                label.startswith("item_") and label[5:].isdigit()
                for label in mca_labels
            )
            if has_real_names:
                extracted["product_columns"] = list(mca_labels)

    elif model_type == "dcm":
        alpha = results.get("alpha")
        if alpha is not None:
            extracted["alpha"] = np.array(alpha)
        alpha_std = results.get("alpha_std")
        if alpha_std is not None:
            extracted["alpha_std"] = np.array(alpha_std)

        product_latent = results.get("product_latent")
        if product_latent is not None:
            pl = np.array(product_latent)
            extracted["loadings"] = pl
            extracted["product_embeddings"] = pl
            pl_norm = pl / (np.linalg.norm(pl, axis=1, keepdims=True) + 1e-10)
            extracted["similarity_matrix"] = pl_norm @ pl_norm.T

    elif model_type == "lda":
        topic_dist = results.get("topic_product_dist")
        if topic_dist is not None:
            tpd = np.array(topic_dist)
            extracted["topic_product_dist"] = tpd
            extracted["loadings"] = tpd.T
            extracted["product_embeddings"] = tpd.T

            # Similarity from topic distributions
            tpd_norm = tpd.T / (np.linalg.norm(tpd.T, axis=1, keepdims=True) + 1e-10)
            extracted["similarity_matrix"] = tpd_norm @ tpd_norm.T

        var_explained = results.get("var_explained_pct")
        if var_explained is not None:
            extracted["variance_explained"] = np.array(var_explained)

    elif model_type == "network":
        adj = results.get("adjacency_matrix")
        if adj is not None:
            extracted["adjacency_matrix"] = np.array(adj)
            extracted["similarity_matrix"] = np.array(adj)

        loadings_raw = results.get("loadings")
        if loadings_raw is not None:
            extracted["loadings"] = np.array(loadings_raw)
            extracted["product_embeddings"] = np.array(loadings_raw)

        var_explained = results.get("var_explained_pct")
        if var_explained is not None:
            extracted["variance_explained"] = np.array(var_explained)

        # Network-specific fields for graph and centrality figures
        extracted["communities"] = results.get("communities")
        cent = results.get("centrality_scores")
        if cent is not None:
            extracted["centrality_scores"] = np.array(cent)
        deg = results.get("degree_centrality")
        if deg is not None:
            extracted["degree_centrality"] = np.array(deg)
        betw = results.get("betweenness_centrality")
        if betw is not None:
            extracted["betweenness_centrality"] = np.array(betw)
        extracted["edge_list"] = results.get("edge_list")

    return extracted


def _generate_plotly_figures(extracted_data: dict, model_type: str) -> dict:
    """Generate Plotly figures for the report using extracted data."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def to_list(arr):
        """Convert numpy array to list for Plotly compatibility."""
        if arr is None:
            return None
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return list(arr)

    figures = {}
    product_columns = extracted_data.get("product_columns", [])

    # Similarity/Correlation Matrix
    similarity = extracted_data.get("similarity_matrix")
    if similarity is not None and len(product_columns) > 0:
        fig = go.Figure(data=go.Heatmap(
            z=to_list(similarity),
            x=product_columns,
            y=product_columns,
            colorscale="RdBu_r",
            zmid=0
        ))
        fig.update_layout(
            title="Product Similarity Matrix",
            xaxis_title="Product",
            yaxis_title="Product",
            height=max(500, len(product_columns) * 15),
            width=max(600, len(product_columns) * 15)
        )
        figures["Similarity Matrix"] = fig

    # Variance Explained
    var_explained = extracted_data.get("variance_explained")
    if var_explained is not None and len(var_explained) > 0:
        n_comp = len(var_explained)
        var_list = to_list(var_explained)
        cumulative = to_list(np.cumsum(var_explained))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Comp {i+1}" for i in range(n_comp)],
            y=var_list,
            name="Individual",
            marker_color="#667eea"
        ))
        fig.add_trace(go.Scatter(
            x=[f"Comp {i+1}" for i in range(n_comp)],
            y=cumulative,
            name="Cumulative",
            mode="lines+markers",
            marker_color="#764ba2"
        ))
        fig.update_layout(
            title="Variance Explained",
            xaxis_title="Component",
            yaxis_title="Variance Explained (%)",
            height=400,
            showlegend=True
        )
        figures["Variance Explained"] = fig

    # Loadings Heatmap (factor-type models and LCA)
    loadings = extracted_data.get("loadings")
    model_type = extracted_data.get("model_type", "")
    if loadings is not None and len(product_columns) > 0:
        if len(loadings.shape) == 2 and loadings.shape[0] == len(product_columns):
            n_factors = loadings.shape[1]
            # Use "Class" labels for LCA models, "Factor" for others
            if model_type in ["lca", "lca_covariates"]:
                factor_names = [f"Class {i+1}" for i in range(n_factors)]
                loadings_title = "Class Loadings"
                x_title = "Class"
            else:
                factor_names = [f"Factor {i+1}" for i in range(n_factors)]
                loadings_title = "Factor Loadings"
                x_title = "Factor"
            fig = go.Figure(data=go.Heatmap(
                z=to_list(loadings),
                x=factor_names,
                y=product_columns,
                colorscale="RdBu_r",
                zmid=0
            ))
            fig.update_layout(
                title=loadings_title,
                xaxis_title=x_title,
                yaxis_title="Product",
                height=max(400, len(product_columns) * 20)
            )
            figures[loadings_title] = fig

    # LCA Class Profiles
    item_probs = extracted_data.get("item_probs")
    class_probs = extracted_data.get("class_probs")
    if item_probs is not None and class_probs is not None and len(product_columns) > 0:
        n_classes = len(class_probs)
        fig = go.Figure()
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
        for c in range(n_classes):
            fig.add_trace(go.Bar(
                x=product_columns,
                y=to_list(item_probs[c]),
                name=f"Class {c+1} ({float(class_probs[c])*100:.1f}%)",
                marker_color=colors[c % len(colors)]
            ))
        fig.update_layout(
            title="LCA Class Profiles",
            xaxis_title="Product",
            yaxis_title="Purchase Probability",
            barmode="group",
            height=500
        )
        figures["Class Profiles"] = fig

    # Tetrachoric Correlation (for factor_tetrachoric)
    tetra_corr = extracted_data.get("tetra_corr")
    if tetra_corr is not None and len(product_columns) > 0:
        fig = go.Figure(data=go.Heatmap(
            z=to_list(tetra_corr),
            x=product_columns,
            y=product_columns,
            colorscale="RdBu_r",
            zmid=0
        ))
        fig.update_layout(
            title="Tetrachoric Correlation Matrix",
            xaxis_title="Product",
            yaxis_title="Product",
            height=max(500, len(product_columns) * 15)
        )
        figures["Tetrachoric Correlation"] = fig

    # ELBO Convergence (for Bayesian VI)
    elbo_history = extracted_data.get("elbo_history")
    if elbo_history is not None and len(elbo_history) > 0:
        fig = go.Figure(data=go.Scatter(
            x=list(range(1, len(elbo_history) + 1)),
            y=elbo_history,
            mode="lines",
            line=dict(color="#667eea")
        ))
        fig.update_layout(
            title="ELBO Convergence",
            xaxis_title="Iteration",
            yaxis_title="ELBO",
            height=400
        )
        figures["ELBO Convergence"] = fig

    # LDA Topic Distribution
    topic_dist = extracted_data.get("topic_product_dist")
    if topic_dist is not None and len(product_columns) > 0:
        n_topics = topic_dist.shape[0]
        topic_names = [f"Topic {i+1}" for i in range(n_topics)]
        fig = go.Figure(data=go.Heatmap(
            z=to_list(topic_dist),
            x=product_columns,
            y=topic_names,
            colorscale="Viridis"
        ))
        fig.update_layout(
            title="Topic-Product Distribution",
            xaxis_title="Product",
            yaxis_title="Topic",
            height=max(300, n_topics * 50)
        )
        figures["Topic Distribution"] = fig

    # Network Adjacency Matrix
    adj_matrix = extracted_data.get("adjacency_matrix")
    if adj_matrix is not None and len(product_columns) > 0:
        fig = go.Figure(data=go.Heatmap(
            z=to_list(adj_matrix),
            x=product_columns,
            y=product_columns,
            colorscale="Blues"
        ))
        fig.update_layout(
            title="Product Co-Purchase Network",
            xaxis_title="Product",
            yaxis_title="Product",
            height=max(500, len(product_columns) * 15)
        )
        figures["Network Matrix"] = fig

    # DCM Coefficients
    alpha = extracted_data.get("alpha")
    alpha_std = extracted_data.get("alpha_std")
    if alpha is not None and len(product_columns) > 0:
        fig = go.Figure()
        # Sort by alpha value
        sorted_idx = np.argsort(alpha)[::-1]
        sorted_products = [product_columns[i] for i in sorted_idx]
        sorted_alpha = to_list(alpha[sorted_idx])

        if alpha_std is not None:
            sorted_std = to_list(1.96 * alpha_std[sorted_idx])
            fig.add_trace(go.Bar(
                x=sorted_products,
                y=sorted_alpha,
                error_y=dict(type='data', array=sorted_std, visible=True),
                marker_color="#667eea"
            ))
        else:
            fig.add_trace(go.Bar(
                x=sorted_products,
                y=sorted_alpha,
                marker_color="#667eea"
            ))
        fig.update_layout(
            title="DCM Product Intercepts (with 95% CI)",
            xaxis_title="Product",
            yaxis_title="Intercept (α)",
            height=500
        )
        figures["Product Intercepts"] = fig

    # Biplot with dimension selector (interactive)
    product_embeddings = extracted_data.get("product_embeddings")
    if product_embeddings is not None and len(product_columns) > 0:
        if len(product_embeddings.shape) == 2 and product_embeddings.shape[1] >= 2:
            n_dims = product_embeddings.shape[1]

            # Create figure with dropdown for dimension selection
            fig = go.Figure()

            # Add all dimension combinations as separate traces
            dim_pairs = []
            for i in range(min(n_dims, 5)):
                for j in range(i + 1, min(n_dims, 5)):
                    dim_pairs.append((i, j))

            for idx, (dim_x, dim_y) in enumerate(dim_pairs):
                visible = idx == 0  # Only first pair visible by default
                fig.add_trace(go.Scatter(
                    x=to_list(product_embeddings[:, dim_x]),
                    y=to_list(product_embeddings[:, dim_y]),
                    mode="markers+text",
                    text=product_columns,
                    textposition="top center",
                    marker=dict(size=10, color="#667eea"),
                    name=f"Dim {dim_x+1} vs {dim_y+1}",
                    visible=visible
                ))

            # Create dropdown buttons
            buttons = []
            for idx, (dim_x, dim_y) in enumerate(dim_pairs):
                visibility = [i == idx for i in range(len(dim_pairs))]
                buttons.append(dict(
                    label=f"Dim {dim_x+1} vs Dim {dim_y+1}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"xaxis.title": f"Dimension {dim_x+1}",
                         "yaxis.title": f"Dimension {dim_y+1}"}
                    ]
                ))

            fig.update_layout(
                title="Product Space (Biplot)",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                height=600,
                # Dropdown in TOP-RIGHT
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        direction="down",
                        showactive=True,
                        x=1.0,
                        xanchor="right",
                        y=1.15,
                        yanchor="top"
                    )
                ] if len(dim_pairs) > 1 else []
            )
            figures["Biplot"] = fig

    # Network Graph (for network models)
    communities = extracted_data.get("communities")
    centrality_scores_data = extracted_data.get("centrality_scores")
    edge_list = extracted_data.get("edge_list")
    if (communities is not None and centrality_scores_data is not None and
            product_embeddings is not None and len(product_columns) > 0):
        import plotly.express as px

        n_prods = len(product_columns)
        x_pos = product_embeddings[:, 0] if product_embeddings.shape[1] >= 1 else np.zeros(n_prods)
        y_pos = product_embeddings[:, 1] if product_embeddings.shape[1] >= 2 else np.zeros(n_prods)

        cent = np.array(centrality_scores_data)
        if cent.max() > cent.min():
            size_norm = (cent - cent.min()) / (cent.max() - cent.min())
        else:
            size_norm = np.ones(n_prods) * 0.5
        node_sizes = 8 + size_norm * 22

        n_comms = len(set(communities))
        comm_colors = (px.colors.qualitative.Set1[:n_comms]
                       if n_comms <= 9
                       else px.colors.qualitative.Alphabet[:n_comms])

        fig = go.Figure()

        # Edges
        if edge_list is not None:
            for edge in edge_list:
                src, tgt = edge[0], edge[1]
                weight = edge[2] if len(edge) > 2 else 0.5
                if src < n_prods and tgt < n_prods:
                    fig.add_trace(go.Scatter(
                        x=[float(x_pos[src]), float(x_pos[tgt]), None],
                        y=[float(y_pos[src]), float(y_pos[tgt]), None],
                        mode="lines",
                        line=dict(width=max(0.5, float(weight) * 2), color="rgba(150,150,150,0.3)"),
                        hoverinfo="skip", showlegend=False
                    ))

        # Nodes by community
        for comm_id in range(n_comms):
            mask = [i for i, c in enumerate(communities) if c == comm_id]
            if not mask:
                continue
            hover_text = [
                f"{product_columns[i]}<br>Community: {comm_id + 1}<br>Centrality: {float(cent[i]):.3f}"
                for i in mask
            ]
            fig.add_trace(go.Scatter(
                x=[float(x_pos[i]) for i in mask],
                y=[float(y_pos[i]) for i in mask],
                mode="markers+text",
                text=[product_columns[i] for i in mask],
                textposition="top center", textfont=dict(size=9),
                hovertext=hover_text, hoverinfo="text",
                marker=dict(
                    size=[float(node_sizes[i]) for i in mask],
                    color=comm_colors[comm_id % len(comm_colors)],
                    line=dict(width=1, color="white")
                ),
                name=f"Community {comm_id + 1}"
            ))

        fig.update_layout(
            title="Product Network Graph",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            height=650, showlegend=True,
            legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(r=150), plot_bgcolor="white"
        )
        figures["Network Graph"] = fig

    # Centrality Comparison (for network models)
    degree_cent = extracted_data.get("degree_centrality")
    betweenness_cent = extracted_data.get("betweenness_centrality")
    if (centrality_scores_data is not None and degree_cent is not None and
            betweenness_cent is not None and len(product_columns) > 0):
        eigenvector = np.array(centrality_scores_data)
        degree_arr = np.array(degree_cent)
        betweenness_arr = np.array(betweenness_cent)

        # Sort by eigenvector centrality (ascending for horizontal bar)
        sorted_idx = np.argsort(eigenvector)
        sorted_products = [product_columns[i] for i in sorted_idx]

        fig = go.Figure()
        for name, vals, color in [
            ("Eigenvector", eigenvector, "#667eea"),
            ("Degree", degree_arr, "#764ba2"),
            ("Betweenness", betweenness_arr, "#f5576c"),
        ]:
            fig.add_trace(go.Bar(
                y=sorted_products,
                x=[float(vals[i]) for i in sorted_idx],
                orientation="h", name=name, marker_color=color
            ))

        fig.update_layout(
            title="Centrality Comparison",
            xaxis_title="Centrality Score",
            yaxis_title="Product",
            height=max(400, len(product_columns) * 25),
            barmode="group", showlegend=True,
            legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(r=150, l=max(80, max(len(p) for p in product_columns) * 7))
        )
        figures["Centrality Comparison"] = fig

    # Top Products per Topic (for LDA models)
    topic_dist_data = extracted_data.get("topic_product_dist")
    if topic_dist_data is not None and len(product_columns) > 0:
        n_topics = topic_dist_data.shape[0]
        n_top = 15

        fig = go.Figure()
        for topic_idx in range(n_topics):
            visible = topic_idx == 0
            probs = topic_dist_data[topic_idx]
            top_indices = np.argsort(probs)[::-1][:n_top]
            top_indices = top_indices[::-1]  # Reverse for horizontal bar

            top_prods = [product_columns[i] for i in top_indices]
            top_probs = [float(probs[i]) for i in top_indices]

            fig.add_trace(go.Bar(
                y=top_prods, x=top_probs,
                orientation="h", marker_color="#667eea",
                name=f"Topic {topic_idx + 1}", visible=visible,
                text=[f"{p:.3f}" for p in top_probs], textposition="outside"
            ))

        buttons = []
        for topic_idx in range(n_topics):
            visibility = [i == topic_idx for i in range(n_topics)]
            buttons.append(dict(
                label=f"Topic {topic_idx + 1}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Top {n_top} Products — Topic {topic_idx + 1}"}
                ]
            ))

        fig.update_layout(
            title=f"Top {n_top} Products — Topic 1",
            xaxis_title="Probability", yaxis_title="Product",
            height=max(400, n_top * 28), showlegend=False,
            margin=dict(l=max(80, max(len(p) for p in product_columns) * 7)),
            updatemenus=[
                dict(
                    active=0, buttons=buttons,
                    direction="down", showactive=True,
                    x=1.0, xanchor="right", y=1.15, yanchor="top"
                )
            ] if n_topics > 1 else []
        )
        figures["Top Products per Topic"] = fig

    # Intertopic Distance Map (for LDA models, requires >= 2 topics)
    if (topic_dist_data is not None and len(product_columns) > 0 and
            topic_dist_data.shape[0] >= 2):
        from scipy.spatial.distance import jensenshannon
        from sklearn.manifold import MDS
        import plotly.express as px

        n_topics_map = topic_dist_data.shape[0]

        # Pairwise Jensen-Shannon distance
        dist_matrix = np.zeros((n_topics_map, n_topics_map))
        for i in range(n_topics_map):
            for j in range(i + 1, n_topics_map):
                d = jensenshannon(topic_dist_data[i], topic_dist_data[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # MDS to 2D
        if n_topics_map == 2:
            coords = np.array([[-dist_matrix[0, 1] / 2, 0],
                               [dist_matrix[0, 1] / 2, 0]])
        else:
            mds = MDS(n_components=2, dissimilarity='precomputed',
                      random_state=42, normalized_stress='auto')
            coords = mds.fit_transform(dist_matrix)

        # Topic prevalence for sizing
        var_exp = extracted_data.get("variance_explained")
        if var_exp is not None and len(var_exp) == n_topics_map:
            prevalence = np.array(var_exp)
        else:
            prevalence = np.ones(n_topics_map) * (100.0 / n_topics_map)

        max_prev = prevalence.max() if prevalence.max() > 0 else 1.0
        marker_sizes = 20 + (prevalence / max_prev) * 60

        # Hover text with top 5 products
        hover_texts = []
        for t in range(n_topics_map):
            probs = topic_dist_data[t]
            top_idx = np.argsort(probs)[::-1][:5]
            lines = [f"<b>Topic {t + 1}</b> ({prevalence[t]:.1f}%)", ""]
            for idx in top_idx:
                lines.append(f"{product_columns[idx]}: {probs[idx]:.3f}")
            hover_texts.append("<br>".join(lines))

        map_colors = (px.colors.qualitative.Set1[:n_topics_map]
                      if n_topics_map <= 9
                      else px.colors.qualitative.Alphabet[:n_topics_map])

        fig = go.Figure()
        for t in range(n_topics_map):
            fig.add_trace(go.Scatter(
                x=[float(coords[t, 0])],
                y=[float(coords[t, 1])],
                mode="markers+text",
                text=[f"Topic {t + 1}"],
                textposition="top center",
                textfont=dict(size=11, color=map_colors[t % len(map_colors)]),
                hovertext=hover_texts[t], hoverinfo="text",
                marker=dict(
                    size=float(marker_sizes[t]),
                    color=map_colors[t % len(map_colors)],
                    opacity=0.7,
                    line=dict(width=2, color="white")
                ),
                name=f"Topic {t + 1} ({prevalence[t]:.1f}%)"
            ))

        fig.update_layout(
            title="Intertopic Distance Map",
            xaxis=dict(
                title="MDS Dimension 1", showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
                zeroline=True, zerolinecolor="rgba(150,150,150,0.5)"
            ),
            yaxis=dict(
                title="MDS Dimension 2", showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
                zeroline=True, zerolinecolor="rgba(150,150,150,0.5)"
            ),
            height=600, showlegend=True,
            legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02,
                        bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(r=180), plot_bgcolor="white"
        )
        figures["Intertopic Distance Map"] = fig

    # =========================================================================
    # Stakeholder-friendly competitive landscape figures (cross-model)
    # =========================================================================

    # --- Closest Competitors ---
    similarity = extracted_data.get("similarity_matrix")
    if similarity is not None and len(product_columns) > 1:
        n_prods = len(product_columns)
        pairs = []
        for i in range(n_prods):
            for j in range(i + 1, n_prods):
                score = float(similarity[i, j])
                if score > 0:
                    pairs.append((product_columns[i], product_columns[j], score))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:15]
        if top_pairs:
            top_pairs = top_pairs[::-1]
            labels_p = [f"{a} ↔ {b}" for a, b, _ in top_pairs]
            scores_p = [s for _, _, s in top_pairs]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=labels_p, x=scores_p, orientation="h",
                marker=dict(color=scores_p, colorscale="Blues", cmin=0),
                text=[f"{s:.1%}" for s in scores_p], textposition="outside",
            ))
            max_lbl = max(len(l) for l in labels_p)
            fig.update_layout(
                title="Closest Competitors",
                xaxis_title="Competitive Similarity", yaxis_title="",
                height=max(400, len(top_pairs) * 32),
                margin=dict(l=max(120, max_lbl * 6)),
                xaxis=dict(tickformat=".0%"),
            )
            figures["Closest Competitors"] = fig

    # --- Market Map ---
    if (product_embeddings is not None and len(product_columns) > 0 and
            len(product_embeddings.shape) == 2 and product_embeddings.shape[1] >= 2):
        import plotly.express as px_mm

        n_prods = len(product_columns)
        x_map = product_embeddings[:, 0]
        y_map = product_embeddings[:, 1]

        # Importance from similarity
        if similarity is not None:
            sim_abs = np.abs(np.array(similarity))
            np.fill_diagonal(sim_abs, 0)
            imp = sim_abs.mean(axis=1)
        else:
            imp = np.ones(n_prods) * 0.5
        imp_min, imp_max = imp.min(), imp.max()
        if imp_max > imp_min:
            sz = 10 + ((imp - imp_min) / (imp_max - imp_min)) * 30
        else:
            sz = np.ones(n_prods) * 20

        # Groups from loadings
        ldg = extracted_data.get("loadings")
        if ldg is not None and len(ldg.shape) == 2 and ldg.shape[1] >= 1:
            grp = np.argmax(np.abs(ldg), axis=1)
            n_g = int(grp.max()) + 1
        else:
            grp = np.zeros(n_prods, dtype=int)
            n_g = 1
        mm_colors = (px_mm.colors.qualitative.Set2[:n_g]
                     if n_g <= 8 else px_mm.colors.qualitative.Alphabet[:n_g])

        fig = go.Figure()
        for gid in range(n_g):
            mask = [i for i in range(n_prods) if grp[i] == gid]
            if not mask:
                continue
            fig.add_trace(go.Scatter(
                x=[float(x_map[i]) for i in mask],
                y=[float(y_map[i]) for i in mask],
                mode="markers+text",
                text=[product_columns[i] for i in mask],
                textposition="top center", textfont=dict(size=10),
                hovertext=[
                    f"<b>{product_columns[i]}</b><br>Group: {gid+1}<br>Importance: {imp[i]:.1%}"
                    for i in mask
                ],
                hoverinfo="text",
                marker=dict(size=[float(sz[i]) for i in mask],
                            color=mm_colors[gid % len(mm_colors)],
                            opacity=0.8, line=dict(width=1, color="white")),
                name=f"Group {gid + 1}",
            ))
        fig.update_layout(
            title="Market Map: Competitive Landscape",
            xaxis=dict(title="", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                       zeroline=False, showticklabels=False),
            yaxis=dict(title="", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                       zeroline=False, showticklabels=False),
            height=650, showlegend=True, legend=dict(title="Product Groups"),
            plot_bgcolor="white",
            annotations=[dict(
                text="Products closer together compete more directly. "
                     "Larger circles = more central to the market.",
                xref="paper", yref="paper", x=0.5, y=-0.06,
                showarrow=False, font=dict(size=10, color="gray"))],
        )
        figures["Market Map"] = fig

    # --- Product Scorecard ---
    if similarity is not None and len(product_columns) > 1:
        import plotly.express as px_sc

        n_prods = len(product_columns)
        sim_sc = np.array(similarity)
        np.fill_diagonal(sim_sc, 0)
        imp_sc = np.abs(sim_sc).mean(axis=1)

        # Groups from loadings
        ldg = extracted_data.get("loadings")
        if ldg is not None and len(ldg.shape) == 2 and ldg.shape[1] >= 1:
            grp = np.argmax(np.abs(ldg), axis=1)
            n_g = int(grp.max()) + 1
        else:
            grp = np.zeros(n_prods, dtype=int)
            n_g = 1
        sc_colors = (px_sc.colors.qualitative.Set2[:n_g]
                     if n_g <= 8 else px_sc.colors.qualitative.Alphabet[:n_g])

        sorted_idx = np.argsort(imp_sc)

        # Top 3 competitors per product
        top3_sc = {}
        for i in range(n_prods):
            row = sim_sc[i].copy()
            row[i] = -np.inf
            t3 = np.argsort(row)[::-1][:3]
            top3_sc[i] = [(product_columns[j], float(row[j])) for j in t3 if row[j] > 0]

        fig = go.Figure()
        for gid in range(n_g):
            mask = [i for i in sorted_idx if grp[i] == gid]
            if not mask:
                continue
            hover = []
            for i in mask:
                lines = [f"<b>{product_columns[i]}</b>",
                         f"Importance: {imp_sc[i]:.3f}", f"Group: {gid+1}", "",
                         "<b>Top Competitors:</b>"]
                for rk, (cn, cs) in enumerate(top3_sc.get(i, []), 1):
                    lines.append(f"  {rk}. {cn} ({cs:.1%})")
                hover.append("<br>".join(lines))
            fig.add_trace(go.Bar(
                y=[product_columns[i] for i in mask],
                x=[float(imp_sc[i]) for i in mask],
                orientation="h", name=f"Group {gid + 1}",
                marker_color=sc_colors[gid % len(sc_colors)],
                hovertext=hover, hoverinfo="text",
            ))

        fig.update_layout(
            title="Product Competitive Scorecard",
            xaxis_title="Market Importance Score", yaxis_title="",
            barmode="relative", height=max(400, n_prods * 24),
            margin=dict(l=max(100, max(len(p) for p in product_columns) * 7)),
            showlegend=True, legend=dict(title="Product Groups"),
            yaxis=dict(categoryorder="array",
                       categoryarray=[product_columns[i] for i in sorted_idx]),
        )
        figures["Product Scorecard"] = fig

    # --- Market Segments (Treemap) ---
    ldg_seg = extracted_data.get("loadings")
    if (ldg_seg is not None and len(ldg_seg.shape) == 2 and
            ldg_seg.shape[1] >= 1 and len(product_columns) > 0):
        import plotly.express as px_seg

        n_prods = len(product_columns)
        n_groups_seg = ldg_seg.shape[1]
        grp_seg = np.argmax(np.abs(ldg_seg), axis=1)
        seg_colors = (px_seg.colors.qualitative.Set2[:n_groups_seg]
                      if n_groups_seg <= 8
                      else px_seg.colors.qualitative.Alphabet[:n_groups_seg])
        var_exp_seg = extracted_data.get("variance_explained")

        tm_labels = ["Market"]
        tm_parents = [""]
        tm_values = [0]
        tm_colors = ["#ffffff"]
        tm_hover = ["Full market structure"]

        for gid in range(n_groups_seg):
            gprods = [i for i in range(n_prods) if grp_seg[i] == gid]
            gname = f"Segment {gid + 1}"
            if var_exp_seg is not None and gid < len(var_exp_seg):
                share = f"{float(var_exp_seg[gid]):.1f}%"
            else:
                share = f"{len(gprods)} products"

            tm_labels.append(gname)
            tm_parents.append("Market")
            tm_values.append(0)
            tm_colors.append(seg_colors[gid % len(seg_colors)])
            tm_hover.append(f"<b>{gname}</b><br>Products: {len(gprods)}<br>Share: {share}")

            for pi in gprods:
                fit = float(np.abs(ldg_seg[pi, gid]))
                tm_labels.append(product_columns[pi])
                tm_parents.append(gname)
                tm_values.append(max(fit, 0.01))
                tm_colors.append(seg_colors[gid % len(seg_colors)])
                tm_hover.append(
                    f"<b>{product_columns[pi]}</b><br>Segment: {gname}<br>Fit: {fit:.3f}")

        fig = go.Figure(go.Treemap(
            labels=tm_labels, parents=tm_parents, values=tm_values,
            marker=dict(colors=tm_colors, line=dict(width=2, color="white")),
            hovertext=tm_hover, hoverinfo="text", textinfo="label",
            textfont=dict(size=12), branchvalues="remainder",
        ))
        fig.update_layout(
            title="Market Segments: Natural Product Groupings",
            height=600, margin=dict(t=50, l=10, r=10, b=30),
        )
        figures["Market Segments"] = fig

    return figures


def _generate_html_report(
    run: ModelRun,
    results: dict,
    figures: dict,
    clustering_result: Optional[dict] = None
) -> str:
    """Generate standalone HTML report."""
    import plotly.io as pio

    model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
    run_name = run.name or f"Run {run.id[:8]}"
    product_columns = run.product_columns or []

    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{run_name} - Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .cluster-tag {{
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 4px;
            font-size: 0.9em;
            color: white;
        }}
        .cluster-section {{
            margin-bottom: 15px;
        }}
        .cluster-title {{
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .footer {{ text-align: center; color: #666; padding: 20px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{run_name}</h1>
        <div class="subtitle">{_get_model_display_name(model_type)} Analysis Report</div>
        <div class="subtitle">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    </div>

    <div class="card">
        <h2>Run Information</h2>
        <div class="metrics-grid">
            <div class="metric"><div class="metric-value">{run.id[:12]}...</div><div class="metric-label">Run ID</div></div>
            <div class="metric"><div class="metric-value">{_get_model_display_name(model_type)}</div><div class="metric-label">Model Type</div></div>
            <div class="metric"><div class="metric-value">{run.status.value.upper() if hasattr(run.status, 'value') else run.status}</div><div class="metric-label">Status</div></div>
            <div class="metric"><div class="metric-value">{run.created_at.strftime('%Y-%m-%d') if run.created_at else '-'}</div><div class="metric-label">Created</div></div>
            <div class="metric"><div class="metric-value">{f'{run.run_duration:.1f}s' if run.run_duration else '-'}</div><div class="metric-label">Duration</div></div>
            <div class="metric"><div class="metric-value">{len(product_columns)}</div><div class="metric-label">Products</div></div>
        </div>
    </div>
"""]

    # Model parameters
    if run.model_params:
        html_parts.append("""
    <div class="card">
        <h2>Model Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
""")
        for key, value in run.model_params.items():
            html_parts.append(f"            <tr><td>{key}</td><td>{value}</td></tr>\n")
        html_parts.append("        </table>\n    </div>\n")

    # Metrics
    if run.metrics:
        html_parts.append("""
    <div class="card">
        <h2>Model Metrics</h2>
        <div class="metrics-grid">
""")
        for key, value in run.metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            html_parts.append(f"""            <div class="metric"><div class="metric-value">{display_value}</div><div class="metric-label">{key.replace('_', ' ').title()}</div></div>\n""")
        html_parts.append("        </div>\n    </div>\n")

    # Figures
    for title, fig in figures.items():
        fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        html_parts.append(f"""
    <div class="card">
        <h2>{title}</h2>
        {fig_html}
    </div>
""")

    # Cluster membership (if clustering was performed)
    if clustering_result is not None and clustering_result.get("cluster_members"):
        cluster_members = clustering_result["cluster_members"]
        n_clusters = clustering_result.get("n_clusters", len(cluster_members))
        silhouette_score = clustering_result.get("silhouette_score")

        # Cluster colors matching the chart
        cluster_colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"
        ]

        html_parts.append(f"""
    <div class="card">
        <h2>Cluster Membership</h2>
        <div class="metrics-grid" style="margin-bottom: 20px;">
            <div class="metric"><div class="metric-value">{n_clusters}</div><div class="metric-label">Clusters</div></div>
""")
        if silhouette_score is not None:
            html_parts.append(f"""            <div class="metric"><div class="metric-value">{silhouette_score:.3f}</div><div class="metric-label">Silhouette Score</div></div>\n""")

        html_parts.append("        </div>\n")

        # List products by cluster
        for cluster_id in sorted(cluster_members.keys(), key=lambda x: int(x)):
            products = cluster_members[cluster_id]
            color = cluster_colors[int(cluster_id) % len(cluster_colors)]
            html_parts.append(f"""        <div class="cluster-section">
            <div class="cluster-title" style="color: {color};">Cluster {int(cluster_id) + 1} ({len(products)} products)</div>
            <div>
""")
            for product in products:
                html_parts.append(f"""                <span class="cluster-tag" style="background-color: {color};">{product}</span>\n""")
            html_parts.append("            </div>\n        </div>\n")

        html_parts.append("    </div>\n")

    # Products list
    if product_columns:
        html_parts.append(f"""
    <div class="card">
        <h2>Products Analyzed ({len(product_columns)})</h2>
        <p>{', '.join(product_columns)}</p>
    </div>
""")

    html_parts.append("""
    <div class="footer">
        <p>Generated by Market Structure Analysis API</p>
    </div>
</body>
</html>
""")

    return "".join(html_parts)


@router.get("/{run_id}/report/debug")
async def debug_report_data(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Debug endpoint to inspect the data being used for report generation.

    Returns information about what data is available and being extracted.
    """
    result = await session.execute(
        select(ModelRun).where(ModelRun.id == run_id)
    )
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if run.status != ModelRunStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed. Current status: {run.status.value}"
        )

    if not run.results_path:
        raise HTTPException(status_code=404, detail="Results file not found")

    results_path = Path(run.results_path)
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found on disk")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
    product_columns = run.product_columns or []

    # Get raw pickle keys and shapes
    raw_info = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            raw_info[key] = f"ndarray shape={value.shape}"
        elif isinstance(value, list):
            raw_info[key] = f"list len={len(value)}"
        elif value is None:
            raw_info[key] = "None"
        else:
            raw_info[key] = f"{type(value).__name__}"

    # Extract data
    extracted = _extract_report_data(results, model_type, product_columns)

    extracted_info = {}
    for key, value in extracted.items():
        if isinstance(value, np.ndarray):
            extracted_info[key] = f"ndarray shape={value.shape}, sample={value.flat[:3].tolist()}"
        elif isinstance(value, list):
            extracted_info[key] = f"list len={len(value)}"
        elif value is None:
            extracted_info[key] = "None"
        else:
            extracted_info[key] = f"{type(value).__name__}"

    return {
        "run_id": run_id,
        "model_type": model_type,
        "product_columns_count": len(product_columns),
        "product_columns_sample": product_columns[:5] if product_columns else [],
        "raw_pickle_info": raw_info,
        "extracted_data_info": extracted_info,
    }


def _perform_clustering_for_report(
    extracted_data: dict,
    n_clusters: Optional[int] = None,
    max_k: int = 10,
    method: str = "kmeans"
) -> Optional[dict]:
    """
    Perform clustering on product embeddings for the report.

    Returns clustering results including labels, silhouette scores, etc.
    """
    try:
        from market_structure.utils import (
            find_optimal_clusters,
            perform_kmeans_clustering,
            compute_hierarchical_clustering,
            get_hierarchical_labels,
        )
    except ImportError:
        return None

    product_embeddings = extracted_data.get("product_embeddings")
    similarity_matrix = extracted_data.get("similarity_matrix")
    product_columns = extracted_data.get("product_columns", [])

    if product_embeddings is None or len(product_columns) < 3:
        return None

    n_products = len(product_columns)
    max_k = min(max_k, n_products - 1)

    if max_k < 2:
        return None

    clustering_result = {
        "labels": None,
        "n_clusters": None,
        "silhouette_score": None,
        "optimal_k": None,
        "silhouette_scores": None,
        "k_range": None,
        "linkage_matrix": None,
        "cluster_members": {},
    }

    # Auto-detect optimal k if not specified
    if n_clusters is None:
        try:
            optimal_result = find_optimal_clusters(product_embeddings, max_k=max_k)
            clustering_result["optimal_k"] = optimal_result["optimal_k"]
            clustering_result["silhouette_scores"] = optimal_result["scores"]
            clustering_result["k_range"] = list(optimal_result["range"])
            n_clusters = optimal_result["optimal_k"]
        except Exception:
            n_clusters = min(3, max_k)

    clustering_result["n_clusters"] = n_clusters

    # Perform clustering
    if method == "kmeans":
        try:
            kmeans_result = perform_kmeans_clustering(product_embeddings, n_clusters)
            clustering_result["labels"] = kmeans_result["labels"].tolist()
            clustering_result["silhouette_score"] = kmeans_result.get("silhouette_score")
        except Exception:
            return None
    else:
        # Hierarchical clustering
        if similarity_matrix is None:
            return None
        try:
            hier_result = compute_hierarchical_clustering(similarity_matrix, method="ward")
            clustering_result["linkage_matrix"] = hier_result["linkage_matrix"].tolist()
            clustering_result["labels"] = get_hierarchical_labels(
                hier_result["linkage_matrix"], n_clusters
            ).tolist()
        except Exception:
            return None

    # Create cluster membership mapping
    if clustering_result["labels"] is not None:
        for i, (product, label) in enumerate(zip(product_columns, clustering_result["labels"])):
            cluster_id = str(label)
            if cluster_id not in clustering_result["cluster_members"]:
                clustering_result["cluster_members"][cluster_id] = []
            clustering_result["cluster_members"][cluster_id].append(product)

    return clustering_result


def _generate_clustering_figures(
    extracted_data: dict,
    clustering_result: dict
) -> dict:
    """Generate Plotly figures for clustering visualization."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    def to_list(arr):
        if arr is None:
            return None
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return list(arr)

    figures = {}
    product_columns = extracted_data.get("product_columns", [])
    product_embeddings = extracted_data.get("product_embeddings")
    labels = clustering_result.get("labels")

    if labels is None or product_embeddings is None:
        return figures

    n_clusters = clustering_result.get("n_clusters", max(labels) + 1)

    # Color palette for clusters
    colors = px.colors.qualitative.Set1[:n_clusters] if n_clusters <= 9 else px.colors.qualitative.Alphabet[:n_clusters]

    # Clustered Biplot with dimension selector
    if len(product_embeddings.shape) == 2 and product_embeddings.shape[1] >= 2:
        n_dims = product_embeddings.shape[1]
        fig = go.Figure()

        dim_pairs = []
        for i in range(min(n_dims, 5)):
            for j in range(i + 1, min(n_dims, 5)):
                dim_pairs.append((i, j))

        # For each dimension pair, add traces for each cluster
        trace_count = 0
        for pair_idx, (dim_x, dim_y) in enumerate(dim_pairs):
            visible = pair_idx == 0
            for cluster_id in range(n_clusters):
                cluster_mask = [l == cluster_id for l in labels]
                cluster_x = [product_embeddings[i, dim_x] for i, m in enumerate(cluster_mask) if m]
                cluster_y = [product_embeddings[i, dim_y] for i, m in enumerate(cluster_mask) if m]
                cluster_text = [product_columns[i] for i, m in enumerate(cluster_mask) if m]

                fig.add_trace(go.Scatter(
                    x=to_list(cluster_x),
                    y=to_list(cluster_y),
                    mode="markers+text",
                    text=cluster_text,
                    textposition="top center",
                    marker=dict(size=12, color=colors[cluster_id % len(colors)]),
                    name=f"Cluster {cluster_id + 1}",
                    legendgroup=f"cluster_{cluster_id}",
                    # Show legend for first pair's traces, dropdown will toggle for other pairs
                    showlegend=pair_idx == 0,
                    visible=visible
                ))
                trace_count += 1

        # Create dropdown buttons that also update showlegend for proper legend display
        buttons = []
        traces_per_pair = n_clusters
        for pair_idx, (dim_x, dim_y) in enumerate(dim_pairs):
            visibility = []
            showlegend_values = []
            for p_idx in range(len(dim_pairs)):
                for _ in range(traces_per_pair):
                    visibility.append(p_idx == pair_idx)
                    # Show legend for the visible traces only
                    showlegend_values.append(p_idx == pair_idx)
            buttons.append(dict(
                label=f"Dim {dim_x+1} vs Dim {dim_y+1}",
                method="update",
                args=[
                    {"visible": visibility, "showlegend": showlegend_values},
                    {"xaxis.title": f"Dimension {dim_x+1}",
                     "yaxis.title": f"Dimension {dim_y+1}"}
                ]
            ))

        fig.update_layout(
            title=f"Clustered Product Space (k={n_clusters})",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            height=600,
            # Legend to the RIGHT of the plot (outside)
            legend=dict(
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            # Add right margin to make room for legend
            margin=dict(r=150),
            # Dropdown in TOP-RIGHT
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.15,
                    yanchor="top"
                )
            ] if len(buttons) > 1 else []
        )
        figures["Clustered Biplot"] = fig

    # Silhouette scores plot (if auto-detection was used)
    silhouette_scores = clustering_result.get("silhouette_scores")
    k_range = clustering_result.get("k_range")
    optimal_k = clustering_result.get("optimal_k")

    if silhouette_scores is not None and k_range is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_range,
            y=silhouette_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#667eea"),
            marker=dict(size=8)
        ))

        # Mark optimal k
        if optimal_k is not None and optimal_k in k_range:
            opt_idx = k_range.index(optimal_k)
            fig.add_trace(go.Scatter(
                x=[optimal_k],
                y=[silhouette_scores[opt_idx]],
                mode="markers",
                name=f"Optimal k={optimal_k}",
                marker=dict(size=15, color="red", symbol="star")
            ))

        fig.update_layout(
            title="Cluster Selection: Silhouette Analysis",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            height=400
        )
        figures["Silhouette Analysis"] = fig

    # Cluster membership table/bar chart
    cluster_members = clustering_result.get("cluster_members", {})
    if cluster_members:
        cluster_sizes = [(f"Cluster {int(k)+1}", len(v)) for k, v in sorted(cluster_members.items(), key=lambda x: int(x[0]))]

        fig = go.Figure(data=[
            go.Bar(
                x=[c[0] for c in cluster_sizes],
                y=[c[1] for c in cluster_sizes],
                marker_color=[colors[i % len(colors)] for i in range(len(cluster_sizes))],
                text=[c[1] for c in cluster_sizes],
                textposition="auto"
            )
        ])
        fig.update_layout(
            title="Cluster Sizes",
            xaxis_title="Cluster",
            yaxis_title="Number of Products",
            height=400
        )
        figures["Cluster Sizes"] = fig

    # Dendrogram for hierarchical clustering
    linkage_matrix = clustering_result.get("linkage_matrix")
    if linkage_matrix is not None:
        try:
            from scipy.cluster.hierarchy import dendrogram
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Create dendrogram data
            plt.figure(figsize=(12, 6))
            dend = dendrogram(
                np.array(linkage_matrix),
                labels=product_columns,
                leaf_rotation=90,
                no_plot=True
            )

            # Convert to Plotly
            fig = go.Figure()

            # Add dendrogram lines
            icoord = dend['icoord']
            dcoord = dend['dcoord']
            for xs, ys in zip(icoord, dcoord):
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines',
                    line=dict(color='#667eea'),
                    showlegend=False
                ))

            # Add x-axis labels
            fig.update_layout(
                title="Hierarchical Clustering Dendrogram",
                xaxis=dict(
                    ticktext=dend['ivl'],
                    tickvals=list(range(5, len(dend['ivl']) * 10, 10)),
                    tickangle=45,
                    title="Product"
                ),
                yaxis_title="Distance",
                height=500
            )
            figures["Dendrogram"] = fig
            plt.close()
        except Exception:
            pass

    return figures


@router.get("/{run_id}/report")
async def get_model_report(
    run_id: str,
    include_clustering: bool = Query(default=True, description="Include clustering analysis"),
    n_clusters: Optional[int] = Query(default=None, ge=2, le=20, description="Number of clusters (None = auto-detect)"),
    clustering_method: str = Query(default="kmeans", pattern="^(kmeans|hierarchical)$", description="Clustering method"),
    max_k: int = Query(default=10, ge=2, le=20, description="Maximum k for auto-detection"),
    session: AsyncSession = Depends(get_session),
):
    """
    Generate an HTML report for a completed model run.

    Returns a standalone HTML file with embedded interactive Plotly visualizations.

    Query Parameters:
    - include_clustering: Whether to include clustering analysis (default: True)
    - n_clusters: Number of clusters (None = auto-detect optimal k)
    - clustering_method: "kmeans" or "hierarchical"
    - max_k: Maximum k to consider for auto-detection
    """
    # Get run from database
    result = await session.execute(
        select(ModelRun).where(ModelRun.id == run_id)
    )
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if run.status != ModelRunStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed. Current status: {run.status.value}"
        )

    # Load results from pickle file
    if not run.results_path:
        raise HTTPException(status_code=404, detail="Results file not found")

    results_path = Path(run.results_path)
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found on disk")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    model_type = run.model_type.value if hasattr(run.model_type, 'value') else run.model_type
    product_columns = run.product_columns or []

    # Extract data from raw results (same logic as get_model_results endpoint)
    extracted_data = _extract_report_data(results, model_type, product_columns)

    # Generate figures using extracted data
    figures = _generate_plotly_figures(extracted_data, model_type)

    # Perform clustering and add clustering figures
    clustering_result = None
    if include_clustering:
        clustering_result = _perform_clustering_for_report(
            extracted_data,
            n_clusters=n_clusters,
            max_k=max_k,
            method=clustering_method
        )
        if clustering_result is not None:
            clustering_figures = _generate_clustering_figures(extracted_data, clustering_result)
            figures.update(clustering_figures)

    # Generate HTML report
    html_content = _generate_html_report(run, results, figures, clustering_result)

    # Return as downloadable HTML file
    run_name = run.name or f"run_{run_id[:8]}"
    filename = f"{run_name.replace(' ', '_')}_{model_type}_report.html"

    return StreamingResponse(
        io.BytesIO(html_content.encode('utf-8')),
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
