"""
Plotting functions for Latent Structure Analysis.

This subpackage provides visualization utilities for the various models
in the lca_analysis package. Functions are organized into:

- core.py: General-purpose plots (correlation heatmaps, variance explained, etc.)
- model_plots.py: Model-specific visualizations (LCA profiles, biplots, etc.)

All plots use Plotly for interactive visualizations that work well in
Streamlit applications.
"""

from .core import (
    plot_correlation_matrix,
    plot_loadings_heatmap,
    plot_loadings_with_uncertainty,
    plot_variance_explained,
    plot_elbo_convergence,
    plot_silhouette_scores,
    plot_dendrogram
)

from .model_plots import (
    plot_lca_profiles,
    plot_biplot,
    plot_dcm_coefficients
)

__all__ = [
    # Core plots
    'plot_correlation_matrix',
    'plot_loadings_heatmap',
    'plot_loadings_with_uncertainty',
    'plot_variance_explained',
    'plot_elbo_convergence',
    'plot_silhouette_scores',
    'plot_dendrogram',
    # Model-specific plots
    'plot_lca_profiles',
    'plot_biplot',
    'plot_dcm_coefficients',
]