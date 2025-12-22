"""
LCA Analysis Package
====================

A comprehensive toolkit for Latent Structure Analysis on binary purchase data.

This package provides implementations of various latent structure models suitable
for analyzing consumer purchase patterns, including:

- **Latent Class Analysis (LCA)**: Discrete customer segments based on purchase patterns
- **Factor Analysis**: Continuous latent factors with tetrachoric correlations
- **Bayesian Factor Models**: Both fast VI and full MCMC implementations
- **Non-negative Matrix Factorization**: Parts-based decomposition
- **Multiple Correspondence Analysis**: PCA for categorical data
- **Discrete Choice Models**: Utility-based models with latent product features

The package is designed to work seamlessly with Streamlit for interactive
data analysis, but all functions work standalone for scripting and notebooks.

Quick Start
-----------
```python
from lca_analysis.models import fit_lca, fit_nmf
from lca_analysis.plotting import plot_biplot, plot_lca_profiles
from lca_analysis.utils import find_optimal_clusters, create_export_zip

# Fit a model
result = fit_lca(data, n_classes=3)

# Visualize results
fig = plot_lca_profiles(result['item_probs'], result['class_probs'], product_names)

# Cluster products
clusters = find_optimal_clusters(result['item_probs'].T)
```

Package Structure
-----------------
- `config`: Dependency management and available model detection
- `models`: All model fitting functions
- `plotting`: Visualization utilities
- `utils`: Caching, clustering, and export utilities
"""

__version__ = "0.1.0"

# Configuration and dependency checking
from .config import (
    PYMC_AVAILABLE,
    PRINCE_AVAILABLE,
    get_available_models,
    get_model_help_text
)

# Utility functions
from .utils import (
    get_model_cache_key,
    compute_hierarchical_clustering,
    find_optimal_clusters,
    perform_kmeans_clustering,
    get_hierarchical_labels,
    get_cluster_members,
    create_export_zip
)

# Import subpackages for easy access
from . import models
from . import plotting

# Commonly used model functions at top level for convenience
from .models import fit_lca, fit_nmf, fit_factor_analysis_tetrachoric, fit_bayesian_factor_vi

# Commonly used plotting functions at top level
from .plotting import plot_biplot, plot_correlation_matrix, plot_variance_explained

__all__ = [
    # Version
    '__version__',
    # Config
    'PYMC_AVAILABLE',
    'PRINCE_AVAILABLE',
    'get_available_models',
    'get_model_help_text',
    # Utilities
    'get_model_cache_key',
    'compute_hierarchical_clustering',
    'find_optimal_clusters',
    'perform_kmeans_clustering',
    'get_hierarchical_labels',
    'get_cluster_members',
    'create_export_zip',
    # Subpackages
    'models',
    'plotting',
    # Commonly used functions
    'fit_lca',
    'fit_nmf',
    'fit_factor_analysis_tetrachoric',
    'fit_bayesian_factor_vi',
    'plot_biplot',
    'plot_correlation_matrix',
    'plot_variance_explained',
]