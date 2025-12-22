"""
Model fitting functions for Latent Structure Analysis.

This subpackage provides implementations of various latent structure models
for binary purchase data. Each module contains the fitting logic for a specific
model type, along with any helper functions specific to that model.

Available Models:
- LCA (lca.py): Latent Class Analysis via EM algorithm
- Factor Analysis (factor.py): Tetrachoric FA and Bayesian VI
- Bayesian Factor Model (bayesian.py): PyMC MCMC-based factor model
- NMF (nmf.py): Non-negative Matrix Factorization
- MCA (mca.py): Multiple Correspondence Analysis
- DCM (dcm.py): Discrete Choice Model with PyMC
"""

# LCA exports
from .lca import (
    fit_lca,
    compute_lca_coordinates,
    compute_residual_correlations,
    fit_lca_with_covariates,
    interpret_covariate_effects
)

# Factor Analysis exports
from .factor import (
    compute_tetrachoric_correlation,
    fit_factor_analysis_tetrachoric,
    fit_bayesian_factor_vi,
    compute_factor_scores_regression
)

# NMF exports
from .nmf import fit_nmf

# Conditional imports for optional dependencies
from ..config import PYMC_AVAILABLE, PRINCE_AVAILABLE

if PRINCE_AVAILABLE:
    from .mca import fit_mca

if PYMC_AVAILABLE:
    from .bayesian import fit_bayesian_factor_model_pymc
    from .dcm import fit_discrete_choice_model

# List of all available model fitting functions for introspection
__all__ = [
    # Always available
    'fit_lca',
    'compute_lca_coordinates',
    'compute_residual_correlations',
    'fit_lca_with_covariates',
    'interpret_covariate_effects',
    'compute_tetrachoric_correlation',
    'fit_factor_analysis_tetrachoric',
    'fit_bayesian_factor_vi',
    'compute_factor_scores_regression',
    'fit_nmf',
]

if PRINCE_AVAILABLE:
    __all__.append('fit_mca')

if PYMC_AVAILABLE:
    __all__.extend(['fit_bayesian_factor_model_pymc', 'fit_discrete_choice_model'])