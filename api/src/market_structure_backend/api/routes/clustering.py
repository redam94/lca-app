"""
FastAPI routes for clustering operations on completed model runs.

Provides endpoints for:
- Running clustering on product embeddings from completed runs
- Retrieving cached clustering results
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import ModelRun, ModelRunStatus, ModelType, get_session
from ...schemas import (
    ClusteringRequest,
    ClusteringResponse,
    ClusteringMethodEnum,
)

# Import clustering utilities from market_structure package
from market_structure.utils import (
    find_optimal_clusters,
    perform_kmeans_clustering,
    compute_hierarchical_clustering,
    get_hierarchical_labels,
    get_cluster_members,
)


router = APIRouter(prefix="/runs", tags=["Clustering"])


def _extract_product_embeddings(results: dict, model_type: str) -> Optional[np.ndarray]:
    """Extract product embeddings from model results based on model type."""
    if model_type in ["lca", "lca_covariates"]:
        # For LCA, product embeddings = transpose of item_probs
        item_probs = results.get("item_probs")
        if item_probs is not None:
            return np.array(item_probs).T
    elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc", "nmf"]:
        # For factor models, use loadings
        loadings = results.get("loadings")
        if loadings is not None:
            return np.array(loadings)
    elif model_type == "mca":
        # For MCA, use column coordinates
        col_coords = results.get("column_coordinates")
        if col_coords is not None:
            return np.array(col_coords)
    elif model_type == "dcm":
        # For DCM, use product latent features
        product_latent = results.get("product_latent")
        if product_latent is not None:
            return np.array(product_latent)

    return None


def _extract_similarity_matrix(results: dict, model_type: str, embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Extract or compute similarity matrix from model results."""
    # First check if there's a pre-computed similarity matrix
    if model_type in ["lca", "lca_covariates"]:
        residual = results.get("residual_correlations")
        if residual is not None:
            return np.array(residual)
    elif model_type == "mca":
        sim = results.get("similarity_matrix")
        if sim is not None:
            return np.array(sim)

    # Compute from embeddings if not available
    if embeddings is not None:
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings_norm = embeddings / norms
        return embeddings_norm @ embeddings_norm.T

    return None


@router.post("/{run_id}/clustering", response_model=ClusteringResponse)
async def run_clustering(
    run_id: str,
    request: ClusteringRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Run clustering on completed model results.

    Extracts product embeddings from the model results and performs
    either k-means or hierarchical clustering. If n_clusters is not
    specified, automatically detects the optimal number using silhouette scores.
    """
    # Load the model run
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

    # Get model type
    model_type = run.model_type.value if isinstance(run.model_type, ModelType) else run.model_type
    product_columns = run.product_columns or []

    # Extract product embeddings
    embeddings = _extract_product_embeddings(results, model_type)
    if embeddings is None:
        raise HTTPException(400, f"Cannot extract product embeddings for model type: {model_type}")

    # Extract or compute similarity matrix (for hierarchical clustering)
    similarity_matrix = _extract_similarity_matrix(results, model_type, embeddings)

    # Determine number of clusters
    n_products = len(product_columns)
    max_k = min(request.max_k, n_products - 1)

    optimal_k = None
    silhouette_scores = None
    k_range = None

    if request.n_clusters is None:
        # Auto-detect optimal number of clusters
        optimal_result = find_optimal_clusters(embeddings, max_k=max_k)
        optimal_k = optimal_result["optimal_k"]
        silhouette_scores = optimal_result["scores"]
        k_range = list(optimal_result["range"])
        n_clusters = optimal_k
    else:
        n_clusters = request.n_clusters

    # Perform clustering
    if request.method == ClusteringMethodEnum.KMEANS:
        cluster_result = perform_kmeans_clustering(embeddings, n_clusters)
        labels = cluster_result["labels"].tolist()
        silhouette_score = cluster_result.get("silhouette_score")
        inertia = cluster_result.get("inertia")
        linkage_matrix = None
    else:
        # Hierarchical clustering
        if similarity_matrix is None:
            raise HTTPException(400, "Cannot compute similarity matrix for hierarchical clustering")

        # Compute hierarchical clustering
        hier_result = compute_hierarchical_clustering(
            similarity_matrix,
            method=request.linkage_method
        )
        linkage_matrix = hier_result["linkage_matrix"].tolist()

        # Get labels at the requested number of clusters
        labels = get_hierarchical_labels(hier_result["linkage_matrix"], n_clusters).tolist()
        silhouette_score = None  # Can compute if needed
        inertia = None

    # Create cluster membership mapping
    cluster_df = get_cluster_members(np.array(labels), product_columns)
    cluster_members = {}
    for _, row in cluster_df.iterrows():
        print(row)
        cluster_id = str(row["Cluster"])
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(row["Product"])

    return ClusteringResponse(
        model_run_id=run_id,
        method=request.method,
        n_clusters=n_clusters,
        labels=labels,
        product_columns=product_columns,
        silhouette_score=silhouette_score,
        inertia=inertia,
        linkage_matrix=linkage_matrix,
        optimal_k=optimal_k,
        silhouette_scores=silhouette_scores,
        k_range=k_range,
        cluster_members=cluster_members,
    )
