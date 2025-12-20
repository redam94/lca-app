"""
Non-negative Matrix Factorization (NMF) for Purchase Data.

NMF decomposes the purchase matrix X into two non-negative factors:
    X ≈ W @ H

Where:
- W (n_households × n_components): Household coefficients/scores
  - Each row shows how much each household "belongs to" each component
- H (n_components × n_products): Component patterns/loadings
  - Each row is a "purchase archetype" or basis pattern

Unlike factor analysis, NMF enforces non-negativity, which leads to
parts-based, additive representations. Components are often more
interpretable as they represent "additive" purchase patterns that
combine to explain total purchasing behavior.

For binary purchase data, NMF provides a useful approximation even though
the data isn't strictly continuous. The resulting components can be
interpreted as purchase archetypes or "shopping baskets."
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import Dict


def fit_nmf(data: np.ndarray, n_components: int, 
            max_iter: int = 200, random_state: int = 42) -> Dict:
    """
    Fit Non-negative Matrix Factorization model.
    
    Decomposes the binary purchase matrix into non-negative factors,
    producing interpretable "parts-based" components. Each component
    represents an additive purchase pattern.
    
    The decomposition is: X ≈ W @ H
    - W: Household scores (how much each household exhibits each pattern)
    - H: Component patterns (which products define each pattern)
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        n_components: Number of NMF components to extract
        max_iter: Maximum iterations for the optimization
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with:
        - W: (n_households, n_components) household score matrix
        - H: (n_components, n_products) component pattern matrix
        - loadings: H.T, transposed for consistency with factor models
        - scores: W, for consistency with factor models
        - var_explained_pct: Approximate variance explained by each component
        - reconstruction_error: Frobenius norm of reconstruction error
        - n_iter: Number of iterations until convergence
    """
    # Initialize and fit NMF model
    # Using 'nndsvd' initialization which is deterministic and often works well
    model = NMF(
        n_components=n_components, 
        max_iter=max_iter, 
        random_state=random_state,
        init='nndsvda',  # NNDSVD with zeros replaced by small values
        solver='cd',     # Coordinate descent is fast for sparse-ish data
        beta_loss='frobenius'  # Standard squared error loss
    )
    
    # W = household scores, H = component patterns
    W = model.fit_transform(data)
    H = model.components_
    
    # Compute variance explained by each component
    # We measure this as the sum of squared loadings (column norm of H)
    # normalized by the number of items
    var_explained = np.sum(H ** 2, axis=1)
    var_explained_pct = var_explained / data.shape[1] * 100
    
    # Sort components by variance explained (descending)
    sort_idx = np.argsort(var_explained)[::-1]
    H = H[sort_idx]
    W = W[:, sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    return {
        'W': W,                           # Household scores
        'H': H,                           # Component patterns  
        'loadings': H.T,                  # Transposed for biplot compatibility
        'scores': W,                      # Alias for consistency
        'var_explained_pct': var_explained_pct,
        'reconstruction_error': model.reconstruction_err_,
        'n_iter': model.n_iter_
    }


def compute_nmf_product_similarity(H: np.ndarray) -> np.ndarray:
    """
    Compute product similarity matrix from NMF component patterns.
    
    Products with similar component loadings are considered similar.
    We normalize the H matrix and compute cosine similarity.
    
    Args:
        H: (n_components, n_products) component pattern matrix
        
    Returns:
        (n_products, n_products) similarity matrix with values in [0, 1]
        (non-negative due to NMF non-negativity)
    """
    # H.T gives us (n_products, n_components)
    # Normalize each product's loading vector to unit length
    loadings = H.T
    norms = np.linalg.norm(loadings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    loadings_normalized = loadings / norms
    
    # Cosine similarity (which equals dot product for unit vectors)
    similarity = loadings_normalized @ loadings_normalized.T
    
    return similarity