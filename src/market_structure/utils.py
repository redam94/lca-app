"""
Utility functions for LCA Analysis package.

This module provides helper functions for:
- Model cache key generation (to avoid unnecessary re-computation)
- Hierarchical and K-means clustering on product embeddings
- Optimal cluster number detection via silhouette scores
- ZIP export of all analysis results

These utilities support the main application by handling cross-cutting concerns
that don't belong to any specific model or visualization module.
"""

import io
import json
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =============================================================================
# CACHING AND FINGERPRINTING
# =============================================================================

def _make_hashable(obj):
    """
    Recursively convert an object to a hashable form.
    
    Lists become tuples, dicts become frozensets of tuples, and other
    types are left as-is (assuming they're already hashable).
    """
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return frozenset((k, _make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    else:
        return obj


def get_model_cache_key(data_hash: str, model_type: str, 
                         model_params: dict, product_columns: tuple) -> str:
    """
    Generate a unique cache key based on data, model type, and parameters.
    
    This enables smart caching in Streamlit - we only refit models when 
    the underlying data or model parameters actually change, not when the
    user adjusts visualization settings like biplot dimensions.
    
    Args:
        data_hash: Hash of the input data matrix (from X.tobytes())
        model_type: String identifier for the model (e.g., "LCA", "NMF")
        model_params: Dictionary of model hyperparameters (can contain lists)
        product_columns: Tuple of selected product column names
        
    Returns:
        A string that uniquely identifies this model configuration.
        If two calls produce the same key, the cached model can be reused.
    """
    # Convert model params to hashable form (handles nested lists, dicts, etc.)
    hashable_params = _make_hashable(model_params)
    params_hash = hash(hashable_params)
    # Include product columns since changing the selected products invalidates the model
    return f"{model_type}_{data_hash}_{params_hash}_{hash(product_columns)}"


# =============================================================================
# CLUSTERING UTILITIES
# =============================================================================

def compute_hierarchical_clustering(similarity_matrix: np.ndarray, 
                                     method: str = 'average') -> dict:
    """
    Compute hierarchical clustering from a similarity matrix.
    
    The function converts similarity to distance (distance = 1 - similarity),
    then applies hierarchical agglomerative clustering. This is useful for
    creating dendrograms and finding natural product groupings.
    
    Args:
        similarity_matrix: Square matrix where higher values = more similar.
                          Typically bounded [-1, 1] (like correlations).
        method: Linkage method for scipy.cluster.hierarchy.linkage.
                Options include 'average', 'complete', 'ward', 'single'.
                'average' (UPGMA) is a good default for most applications.
    
    Returns:
        Dictionary with:
        - 'linkage_matrix': The scipy linkage matrix (for dendrogram plotting)
        - 'distance_matrix': The derived distance matrix
    """
    # Convert similarity to distance, ensuring values are in valid range
    distance_matrix = 1 - np.clip(similarity_matrix, -1, 1)
    # Self-distance should always be zero
    np.fill_diagonal(distance_matrix, 0)
    
    # scipy.cluster.hierarchy.linkage requires condensed distance form
    # (upper triangle of the distance matrix as a flat array)
    n = len(distance_matrix)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(distance_matrix[i, j])
    condensed = np.array(condensed)
    
    # Compute the hierarchical clustering linkage
    linkage_matrix = linkage(condensed, method=method)
    
    return {
        'linkage_matrix': linkage_matrix,
        'distance_matrix': distance_matrix
    }


def find_optimal_clusters(embeddings: np.ndarray, max_k: int = 10) -> dict:
    """
    Find optimal number of clusters using silhouette score.
    
    The silhouette score measures how similar objects are to their own cluster
    compared to other clusters. Values range from -1 to 1, where higher is better.
    We test k=2 through max_k and select the k with highest silhouette score.
    
    Args:
        embeddings: (n_products, n_dimensions) array of product coordinates
        max_k: Maximum number of clusters to consider
        
    Returns:
        Dictionary with:
        - 'optimal_k': The best number of clusters
        - 'scores': List of silhouette scores for each k tested
        - 'range': List of k values tested
    """
    scores = []
    # Start from k=2 (silhouette undefined for k=1), cap at n_products - 1
    k_range = range(2, min(max_k + 1, len(embeddings)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
    
    # Find k with maximum silhouette score
    optimal_k = list(k_range)[np.argmax(scores)]
    
    return {
        'optimal_k': optimal_k,
        'scores': scores,
        'range': list(k_range)
    }


def perform_kmeans_clustering(embeddings: np.ndarray, n_clusters: int) -> dict:
    """
    Perform K-means clustering on product embeddings.
    
    Args:
        embeddings: (n_products, n_dimensions) array
        n_clusters: Number of clusters to form
        
    Returns:
        Dictionary with cluster labels and centroids
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return {
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }


def get_hierarchical_labels(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cut a hierarchical clustering tree to get cluster labels.
    
    Args:
        linkage_matrix: Output from scipy.cluster.hierarchy.linkage
        n_clusters: Number of clusters to cut into
        
    Returns:
        Array of cluster labels (0-indexed)
    """
    # fcluster returns 1-indexed labels, we convert to 0-indexed for consistency
    labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust') - 1
    return labels


def get_cluster_members(labels: np.ndarray, product_labels: list) -> pd.DataFrame:
    """
    Create a DataFrame mapping products to their cluster assignments.
    
    Args:
        labels: Array of cluster labels (0-indexed)
        product_labels: List of product names
        
    Returns:
        DataFrame with 'Product' and 'Cluster' columns, sorted by cluster.
        Cluster numbers are 1-indexed for user display.
    """
    return pd.DataFrame({
        'Product': product_labels,
        'Cluster': labels + 1  # Convert to 1-indexed for user-facing output
    }).sort_values('Cluster')


# =============================================================================
# EXPORT FUNCTIONALITY
# =============================================================================

def create_export_zip(model_result: dict,
                      model_type: str,
                      product_columns: list,
                      product_embeddings: Optional[np.ndarray] = None,
                      household_embeddings: Optional[np.ndarray] = None,
                      var_explained: Optional[np.ndarray] = None,
                      similarity_matrix: Optional[np.ndarray] = None,
                      cluster_result: Optional[dict] = None,
                      original_data: Optional[np.ndarray] = None) -> bytes:
    """
    Create a ZIP file containing all analysis results for download.
    
    This packages everything the user might need to continue their analysis
    in Python, R, or other tools. The ZIP includes CSV data files, a JSON
    model summary, and a README with usage instructions.
    
    Args:
        model_result: Dictionary of model-specific results
        model_type: Name of the model type
        product_columns: List of product column names
        product_embeddings: Product coordinates in latent space
        household_embeddings: Household coordinates in latent space
        var_explained: Variance explained by each component (%)
        similarity_matrix: Product similarity/correlation matrix
        cluster_result: Optional clustering results
        original_data: Original binary purchase matrix
        
    Returns:
        bytes: ZIP file contents ready for download
    """
    zip_buffer = io.BytesIO()
    
    # Track which files we include for the metadata
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'product_columns': product_columns,
        'files_included': []
    }
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        # Product embeddings (coordinates in latent space)
        if product_embeddings is not None:
            n_dims = product_embeddings.shape[1]
            prod_df = pd.DataFrame(
                product_embeddings,
                index=product_columns,
                columns=[f"Dim_{i+1}" for i in range(n_dims)]
            )
            csv_buffer = io.StringIO()
            prod_df.to_csv(csv_buffer, index_label='product')
            zf.writestr('product_embeddings.csv', csv_buffer.getvalue())
            metadata['files_included'].append('product_embeddings.csv')
        
        # Household embeddings (scores/coordinates)
        if household_embeddings is not None:
            n_dims = household_embeddings.shape[1]
            hh_df = pd.DataFrame(
                household_embeddings,
                columns=[f"Dim_{i+1}" for i in range(n_dims)]
            )
            csv_buffer = io.StringIO()
            hh_df.to_csv(csv_buffer, index_label='household')
            zf.writestr('household_embeddings.csv', csv_buffer.getvalue())
            metadata['files_included'].append('household_embeddings.csv')
        
        # Variance explained (for factor-type models)
        if var_explained is not None:
            var_df = pd.DataFrame({
                'Component': [f"Dim_{i+1}" for i in range(len(var_explained))],
                'Variance_Explained_Pct': var_explained,
                'Cumulative_Pct': np.cumsum(var_explained)
            })
            csv_buffer = io.StringIO()
            var_df.to_csv(csv_buffer, index=False)
            zf.writestr('variance_explained.csv', csv_buffer.getvalue())
            metadata['files_included'].append('variance_explained.csv')
        
        # Similarity/correlation matrix
        if similarity_matrix is not None:
            sim_df = pd.DataFrame(
                similarity_matrix,
                index=product_columns,
                columns=product_columns
            )
            csv_buffer = io.StringIO()
            sim_df.to_csv(csv_buffer)
            zf.writestr('similarity_matrix.csv', csv_buffer.getvalue())
            metadata['files_included'].append('similarity_matrix.csv')
        
        # Cluster assignments
        if cluster_result is not None and 'labels' in cluster_result:
            cluster_labels = product_columns
            cluster_df = pd.DataFrame({
                'Product': cluster_labels,
                'Cluster': cluster_result['labels'] + 1  # 1-indexed for users
            })
            csv_buffer = io.StringIO()
            cluster_df.to_csv(csv_buffer, index=False)
            zf.writestr('cluster_assignments.csv', csv_buffer.getvalue())
            metadata['files_included'].append('cluster_assignments.csv')
        
        # Original data matrix
        if original_data is not None:
            orig_df = pd.DataFrame(original_data, columns=product_columns)
            csv_buffer = io.StringIO()
            orig_df.to_csv(csv_buffer, index=False)
            zf.writestr('original_data.csv', csv_buffer.getvalue())
            metadata['files_included'].append('original_data.csv')
        
        # Model summary with key metrics
        model_summary = {
            'model_type': model_type,
            'n_products': len(product_columns),
            'n_households': household_embeddings.shape[0] if household_embeddings is not None else 0,
            'n_dimensions': product_embeddings.shape[1] if product_embeddings is not None else 0,
            'total_variance_explained': float(np.sum(var_explained)) if var_explained is not None else None,
        }
        
        # Add model-specific metrics
        if model_result is not None:
            if 'log_likelihood' in model_result:
                model_summary['log_likelihood'] = float(model_result['log_likelihood'])
            if 'bic' in model_result:
                model_summary['bic'] = float(model_result['bic'])
            if 'aic' in model_result:
                model_summary['aic'] = float(model_result['aic'])
            if 'n_iter' in model_result:
                model_summary['n_iterations'] = int(model_result['n_iter'])
            if 'n_classes' in model_result:
                model_summary['n_classes'] = int(model_result['n_classes'])
            if 'total_inertia' in model_result:
                model_summary['total_inertia'] = float(model_result['total_inertia'])
            if 'waic' in model_result and model_result['waic'] is not None:
                try:
                    model_summary['waic'] = float(model_result['waic'].elpd_waic)
                except:
                    pass
            if 'n_divergences' in model_result:
                model_summary['n_divergences'] = int(model_result['n_divergences'])
        
        zf.writestr('model_summary.json', json.dumps(model_summary, indent=2))
        metadata['files_included'].append('model_summary.json')
        
        # README with usage instructions
        readme_content = _create_readme_content(model_type, product_columns, metadata)
        zf.writestr('README.md', readme_content)
        
        # Final metadata file
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def _create_readme_content(model_type: str, product_columns: list, 
                           metadata: dict) -> str:
    """Generate README content for the export ZIP."""
    
    return f"""# Latent Structure Analysis Export

## Model Type: {model_type}

## Export Date: {metadata['export_timestamp']}

## Products Analyzed: {len(product_columns)}

## Files Included

{chr(10).join(f'- {f}' for f in metadata['files_included'])}

## Usage in Python

```python
import pandas as pd
import numpy as np

# Load product embeddings
product_embeddings = pd.read_csv('product_embeddings.csv', index_col='product')

# Load household embeddings/scores
household_embeddings = pd.read_csv('household_embeddings.csv', index_col='household')

# Load similarity matrix
similarity = pd.read_csv('similarity_matrix.csv', index_col=0)

# Find most similar products to a given product
def find_similar(product_name, top_k=5):
    return similarity[product_name].nlargest(top_k + 1)[1:]  # Exclude self
```

## Usage in R

```r
library(tidyverse)

# Load embeddings
product_embeddings <- read_csv('product_embeddings.csv')
household_embeddings <- read_csv('household_embeddings.csv')

# Load similarity matrix
similarity <- read_csv('similarity_matrix.csv') |>
  column_to_rownames(var = names(.)[1])

# Cluster products using hierarchical clustering
dist_matrix <- as.dist(1 - as.matrix(similarity))
hc <- hclust(dist_matrix, method = 'average')
plot(hc)
```

## Interpreting Results

- **Product Embeddings**: Coordinates in the latent space. Products close together 
  have similar purchase patterns.
- **Household Embeddings**: Scores/coordinates for each household. Can be used for 
  segmentation or as features in downstream models.
- **Similarity Matrix**: Pairwise product similarities based on the model. Use for 
  finding substitutes, complements, or clustering.
- **Variance Explained**: How much information each dimension captures. Useful for 
  deciding how many dimensions to retain.

## Notes

Generated by LCA Analysis App
"""