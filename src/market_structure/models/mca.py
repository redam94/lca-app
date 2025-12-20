"""
Multiple Correspondence Analysis (MCA) for Binary Purchase Data.

MCA is the extension of Principal Component Analysis (PCA) to categorical data.
For binary purchase data, it provides an excellent alternative to factor analysis
because it's specifically designed for discrete (categorical) variables.

The method works by:
1. Creating an indicator matrix from the categorical data
2. Computing a weighted PCA on this indicator matrix
3. Extracting row (household) and column (product) coordinates in the same space

Key advantages for purchase data:
- Designed for categorical/binary data (no continuous assumption)
- Produces a biplot where households and products appear in the same space
- Products close to households in the plot are likely purchased by them
- Inertia (MCA's analog to variance) has a natural interpretation

This module requires the `prince` library for MCA computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ..config import PRINCE_AVAILABLE, prince


def fit_mca(data: np.ndarray, n_components: int, 
            product_names: Optional[List[str]] = None) -> Dict:
    """
    Fit Multiple Correspondence Analysis model.
    
    MCA treats each binary variable as a categorical variable with two levels
    (purchased/not purchased). The resulting biplot shows products and households
    in the same latent space, where proximity indicates association.
    
    For the product coordinates, we extract the positions corresponding to
    "product = 1" (purchased), which gives us the location of each product
    in the space defined by purchase patterns.
    
    Args:
        data: (n_households, n_products) binary purchase matrix
        n_components: Number of MCA dimensions to extract
        product_names: Optional list of product names. If None, uses "item_0", etc.
        
    Returns:
        Dictionary with:
        - row_coordinates: (n_households, n_components) household positions
        - column_coordinates: (n_products, n_components) product positions
        - product_labels: List of product names (for exported coordinates)
        - eigenvalues: Principal inertias (eigenvalues)
        - explained_inertia: Proportion of inertia explained by each dimension
        - var_explained_pct: Same as above, as percentage
        - total_inertia: Total inertia in the data
        - contributions: Product contributions to each dimension
        - similarity_matrix: Product similarity based on MCA coordinates
        - n_components: Number of components extracted
        - mca_model: The fitted prince MCA object (for advanced usage)
        
    Raises:
        ImportError: If prince is not installed
    """
    if not PRINCE_AVAILABLE:
        raise ImportError(
            "The 'prince' library is required for MCA. "
            "Install it with: pip install prince"
        )
    
    n_obs, n_items = data.shape
    
    # Generate default product names if not provided
    if product_names is None:
        product_names = [f"item_{i}" for i in range(n_items)]
    
    # Create DataFrame with string categorical columns
    # (prince expects categorical/string data for MCA)
    df = pd.DataFrame(data.astype(int), columns=product_names)
    
    # Convert to string categories (MCA works on categorical data)
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # Fit MCA model
    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    mca.fit(df)
    
    # Extract row (household) coordinates
    row_coords = mca.row_coordinates(df).values
    
    # Extract column (category) coordinates
    # This returns coordinates for all categories (both "0" and "1" for each product)
    col_coords = mca.column_coordinates(df)
    
    # We want only the "1" (purchased) coordinates for each product
    # prince names categories as "column_name__value"
    product_coords = []
    product_labels = []
    
    for prod in product_names:
        key_1 = f"{prod}__1"  # The "purchased" category
        if key_1 in col_coords.index:
            product_coords.append(col_coords.loc[key_1].values)
            product_labels.append(prod)
    
    product_coords = np.array(product_coords)
    
    # Extract eigenvalues and inertia information
    eigenvalues = mca.eigenvalues_
    total_inertia = mca.total_inertia_
    explained_inertia = mca.percentage_of_variance_ / 100  # Convert from % to proportion
    var_explained_pct = np.array(explained_inertia) * 100
    
    # Extract column contributions (how much each product contributes to each dimension)
    col_contribs = mca.column_contributions_
    
    # Get contributions for "purchased" categories only
    product_contribs = []
    for prod in product_names:
        key_1 = f"{prod}__1"
        if key_1 in col_contribs.index:
            product_contribs.append(col_contribs.loc[key_1].values)
    product_contribs = np.array(product_contribs)
    
    # Compute product similarity from MCA coordinates
    # Products close together in MCA space have similar purchase patterns
    if len(product_coords) > 0:
        norms = np.linalg.norm(product_coords, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        coords_normalized = product_coords / norms
        similarity_matrix = coords_normalized @ coords_normalized.T
    else:
        similarity_matrix = np.eye(n_items)
    
    return {
        'row_coordinates': row_coords,
        'column_coordinates': product_coords,
        'product_labels': product_labels,
        'eigenvalues': eigenvalues,
        'explained_inertia': explained_inertia,
        'var_explained_pct': var_explained_pct,
        'total_inertia': total_inertia,
        'contributions': product_contribs,
        'similarity_matrix': similarity_matrix,
        'n_components': n_components,
        'mca_model': mca  # Keep for advanced users who want full access
    }


def interpret_mca_dimension(contributions: np.ndarray, 
                            product_labels: List[str],
                            dimension: int = 0,
                            top_n: int = 5) -> Dict:
    """
    Interpret an MCA dimension by identifying the products that define it.
    
    Products with high contributions to a dimension are the ones that
    best characterize the shopping patterns captured by that dimension.
    
    Args:
        contributions: (n_products, n_dimensions) contribution matrix
        product_labels: List of product names
        dimension: Which dimension to interpret (0-indexed)
        top_n: How many top products to return
        
    Returns:
        Dictionary with lists of top contributing products
    """
    dim_contribs = contributions[:, dimension]
    
    # Sort by contribution (highest first)
    sorted_idx = np.argsort(dim_contribs)[::-1]
    
    return {
        'top_products': [product_labels[i] for i in sorted_idx[:top_n]],
        'top_contributions': dim_contribs[sorted_idx[:top_n]].tolist()
    }