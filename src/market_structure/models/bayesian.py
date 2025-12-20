"""
Bayesian Factor Model using PyMC.

This module implements a full Bayesian factor model with MCMC sampling via PyMC.
Unlike variational inference (VI), MCMC provides asymptotically exact posterior
samples, giving more reliable uncertainty estimates at the cost of longer
computation time.

Key implementation detail: Factor models have rotational invariance, meaning
many equivalent solutions exist that differ only by rotation. This creates
multimodal posteriors that are problematic for MCMC sampling. We address this
using lower triangular identification constraints:
- Diagonal elements of the loading matrix are constrained to be positive
- Below-diagonal elements of the first n_factors rows are freely estimated
- Above-diagonal elements of the first n_factors rows are fixed to zero

This identifies the model up to sign flips, enabling efficient MCMC sampling
with nutpie (a fast NUTS sampler).

Requires: pymc, pytensor, arviz
"""

import numpy as np
from typing import Dict, List, Tuple

from ..config import PYMC_AVAILABLE, pm, pt, az


def fit_bayesian_factor_model_pymc(data: np.ndarray, n_factors: int,
                                    n_samples: int = 1000, 
                                    n_tune: int = 500) -> Dict:
    """
    Fit Bayesian Factor Model using PyMC with MCMC sampling.
    
    This implements a Gaussian factor model with proper identification constraints
    to ensure a unimodal posterior. The model assumes:
    - Loadings: Lower triangular parameterization for identification
      - Diagonal: LogNormal(0, 0.5) ensuring positivity
      - Below diagonal: Normal(0, 1) 
      - Above diagonal (first n_factors rows): Fixed to 0
    - Factors: N(0, I) standard normal
    - Observations: N(Factors @ Loadings.T, sigma^2)
    
    For binary data, this treats the observed 0/1 as continuous, which is an
    approximation but often works well for exploratory analysis.
    
    Args:
        data: (n_observations, n_items) data matrix
        n_factors: Number of latent factors
        n_samples: Number of posterior samples per chain
        n_tune: Number of tuning (warmup) samples
        
    Returns:
        Dictionary with:
        - loadings: (n_items, n_factors) posterior mean loading matrix
        - loadings_std: (n_items, n_factors) posterior standard deviations
        - var_explained: Raw variance explained by each factor
        - var_explained_pct: Variance explained as percentage
        - trace: Full ArviZ InferenceData object for diagnostics
        - waic: WAIC model comparison statistic (if computable)
        - n_divergences: Number of divergent transitions (diagnostic)
        
    Raises:
        ImportError: If PyMC is not available
        ValueError: If n_items < n_factors
    """
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC is required for this model. "
            "Install with: pip install pymc arviz"
        )
    
    n_obs, n_items = data.shape
    
    if n_items < n_factors:
        raise ValueError(
            f"Need at least {n_factors} items for {n_factors} factors, "
            f"but only have {n_items} items"
        )
    
    # Compute the indices for the constrained loading matrix
    # We're building a lower triangular structure for the first n_factors rows
    diag_rows, diag_cols, lower_rows, lower_cols, remaining_rows, remaining_cols = \
        _compute_loading_indices(n_items, n_factors)
    
    n_lower = len(lower_rows)
    n_remaining = len(remaining_rows)
    
    # Pre-compute flat indices for tensor operations
    diag_flat_idx = [r * n_factors + c for r, c in zip(diag_rows, diag_cols)]
    lower_flat_idx = [r * n_factors + c for r, c in zip(lower_rows, lower_cols)] if n_lower > 0 else []
    remaining_flat_idx = [r * n_factors + c for r, c in zip(remaining_rows, remaining_cols)] if n_remaining > 0 else []
    
    # Build the PyMC model
    with pm.Model() as factor_model:
        # Diagonal loadings: positive (for identification)
        # LogNormal ensures positivity while allowing reasonable spread
        diag_loadings = pm.LogNormal('diag_loadings', mu=0, sigma=0.5, shape=n_factors)
        
        # Below-diagonal loadings in the first n_factors rows
        if n_lower > 0:
            lower_loadings = pm.Normal('lower_loadings', mu=0, sigma=1.0, shape=n_lower)
        
        # Remaining loadings (rows n_factors onwards)
        if n_remaining > 0:
            remaining_loadings = pm.Normal('remaining_loadings', mu=0, sigma=1.0, shape=n_remaining)
        
        # Reconstruct full loading matrix using pytensor operations
        loadings_flat = pt.zeros(n_items * n_factors)
        
        # Place diagonal elements
        loadings_flat = pt.set_subtensor(loadings_flat[diag_flat_idx], diag_loadings)
        
        # Place below-diagonal elements
        if n_lower > 0:
            loadings_flat = pt.set_subtensor(loadings_flat[lower_flat_idx], lower_loadings)
        
        # Place remaining elements
        if n_remaining > 0:
            loadings_flat = pt.set_subtensor(loadings_flat[remaining_flat_idx], remaining_loadings)
        
        # Reshape to matrix form
        loadings = loadings_flat.reshape((n_items, n_factors))
        
        # Latent factors (standard normal)
        factors = pm.Normal('factors', mu=0, sigma=1, shape=(n_obs, n_factors))
        
        # Observation noise (per-item)
        sigma = pm.HalfNormal('sigma', sigma=1.0, shape=n_items)
        
        # Likelihood: observations given factors and loadings
        mu = pm.math.dot(factors, loadings.T)
        likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
        
        # Sample using nutpie (fast NUTS implementation)
        trace = pm.sample(
            n_samples, 
            tune=n_tune, 
            nuts_sampler='nutpie',
            progressbar=True, 
            return_inferencedata=True,
            target_accept=0.95,  # Higher for constrained parameters
            random_seed=42
        )
        
        # Compute log-likelihood for WAIC
        trace = pm.compute_log_likelihood(trace)
    
    # Reconstruct full loading samples from constrained posterior
    loadings_samples = _reconstruct_loadings_samples(
        trace, n_items, n_factors, n_lower, n_remaining,
        diag_flat_idx, lower_flat_idx, remaining_flat_idx
    )
    
    # Compute posterior statistics
    loadings_mean = loadings_samples.mean(axis=0)
    loadings_std = loadings_samples.std(axis=0)
    
    # Variance explained by each factor (sum of squared loadings)
    var_explained = np.sum(loadings_mean ** 2, axis=0)
    var_explained_pct = var_explained / n_items * 100
    
    # Sort factors by variance explained
    sort_idx = np.argsort(var_explained)[::-1]
    loadings_mean = loadings_mean[:, sort_idx]
    loadings_std = loadings_std[:, sort_idx]
    var_explained = var_explained[sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    # Compute WAIC for model comparison
    try:
        waic = az.waic(trace)
    except Exception:
        waic = None
    
    # Count divergent transitions (diagnostic for sampling issues)
    n_divergences = int(trace.sample_stats['diverging'].sum().values)
    
    return {
        'loadings': loadings_mean,
        'loadings_std': loadings_std,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'trace': trace,
        'waic': waic,
        'n_divergences': n_divergences
    }


def _compute_loading_indices(n_items: int, n_factors: int) -> Tuple[List, ...]:
    """
    Compute row/column indices for the constrained loading matrix.
    
    The identification constraint is:
    - First n_factors rows form a lower triangular matrix with positive diagonal
    - Remaining rows (n_factors onwards) have all elements freely estimated
    
    Returns:
        Tuple of (diag_rows, diag_cols, lower_rows, lower_cols, 
                  remaining_rows, remaining_cols)
    """
    # Diagonal elements: (0,0), (1,1), ..., (n_factors-1, n_factors-1)
    diag_rows = list(range(n_factors))
    diag_cols = list(range(n_factors))
    
    # Below-diagonal elements in first n_factors rows
    # For row i, columns 0 to i-1
    lower_rows = []
    lower_cols = []
    for i in range(1, n_factors):
        for j in range(i):
            lower_rows.append(i)
            lower_cols.append(j)
    
    # Remaining rows: all columns are free
    remaining_rows = []
    remaining_cols = []
    for i in range(n_factors, n_items):
        for j in range(n_factors):
            remaining_rows.append(i)
            remaining_cols.append(j)
    
    return (diag_rows, diag_cols, lower_rows, lower_cols, 
            remaining_rows, remaining_cols)


def _reconstruct_loadings_samples(trace, n_items: int, n_factors: int,
                                   n_lower: int, n_remaining: int,
                                   diag_idx: List, lower_idx: List, 
                                   remaining_idx: List) -> np.ndarray:
    """
    Reconstruct full loading matrix samples from constrained posterior samples.
    
    The PyMC model estimates separate parameters for diagonal, below-diagonal,
    and remaining elements. This function reassembles them into the full
    (n_items × n_factors) loading matrix for each posterior sample.
    
    Args:
        trace: ArviZ InferenceData with posterior samples
        n_items, n_factors: Matrix dimensions
        n_lower, n_remaining: Number of parameters in each group
        diag_idx, lower_idx, remaining_idx: Flat indices for each group
        
    Returns:
        (n_samples, n_items, n_factors) array of loading matrices
    """
    # Extract posterior samples and flatten across chains
    diag_samples = trace.posterior['diag_loadings'].values
    n_chains, n_draws = diag_samples.shape[:2]
    n_total_samples = n_chains * n_draws
    
    diag_samples = diag_samples.reshape(n_total_samples, n_factors)
    
    if n_lower > 0:
        lower_samples = trace.posterior['lower_loadings'].values.reshape(n_total_samples, n_lower)
    
    if n_remaining > 0:
        remaining_samples = trace.posterior['remaining_loadings'].values.reshape(n_total_samples, n_remaining)
    
    # Reconstruct full loading matrix for each sample
    loadings_samples = np.zeros((n_total_samples, n_items, n_factors))
    
    for s in range(n_total_samples):
        flat = np.zeros(n_items * n_factors)
        
        # Place diagonal elements
        flat[diag_idx] = diag_samples[s]
        
        # Place below-diagonal elements
        if n_lower > 0:
            flat[lower_idx] = lower_samples[s]
        
        # Place remaining elements
        if n_remaining > 0:
            flat[remaining_idx] = remaining_samples[s]
        
        loadings_samples[s] = flat.reshape(n_items, n_factors)
    
    return loadings_samples