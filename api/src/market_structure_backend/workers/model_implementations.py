"""
Model implementations with progress callback support.

These functions wrap the model fitting functions from the lca_analysis package
and integrate with the progress tracking system. They run in a thread pool
to avoid blocking the async event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Any

import numpy as np

# Import model functions from the lca_analysis package
from market_structure.models import (
    fit_lca,
    fit_factor_analysis_tetrachoric,
    fit_bayesian_factor_vi,
    fit_nmf,
)
from market_structure.config import PYMC_AVAILABLE, PRINCE_AVAILABLE

if PRINCE_AVAILABLE:
    from market_structure.models import fit_mca

if PYMC_AVAILABLE:
    from market_structure.models import fit_bayesian_factor_model_pymc, fit_discrete_choice_model


# Thread pool for running CPU-bound model fitting
_executor = ThreadPoolExecutor(max_workers=4)


async def _run_in_executor(func, *args, **kwargs):
    """Run a sync function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: func(*args, **kwargs)
    )


# =============================================================================
# LCA WRAPPER
# =============================================================================

async def fit_lca_with_progress(
    data: np.ndarray,
    n_classes: int,
    max_iter: int = 100,
    n_init: int = 10,
    tol: float = 1e-6,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for LCA fitting with progress callbacks.
    """
    def fit_with_callback():
        # Report start
        if progress_callback:
            progress_callback(iteration=0, log_likelihood=None, delta=None, extra={"status": "starting"})
        
        result = fit_lca(
            data,
            n_classes=n_classes,
            max_iter=max_iter,
            n_init=n_init,
            tol=tol,
        )
        
        # Report completion
        if progress_callback:
            progress_callback(
                iteration=max_iter, 
                log_likelihood=result.get("log_likelihood"),
                delta=None,
                extra={"status": "completed"}
            )
        
        return result
    
    return await _run_in_executor(fit_with_callback)


# =============================================================================
# FACTOR ANALYSIS WRAPPERS
# =============================================================================

async def fit_factor_tetrachoric_with_progress(
    data: np.ndarray,
    n_factors: int,
    max_iter: int = 100,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for tetrachoric factor analysis.
    """
    def fit_with_callback():
        if progress_callback:
            progress_callback(iteration=0, log_likelihood=None, delta=None, extra={"status": "starting"})
        
        result = fit_factor_analysis_tetrachoric(
            data,
            n_factors=n_factors,
            max_iter=max_iter,
        )
        
        if progress_callback:
            progress_callback(iteration=max_iter, log_likelihood=None, delta=None, extra={"status": "completed"})
        
        return result
    
    return await _run_in_executor(fit_with_callback)


async def fit_bayesian_vi_with_progress(
    data: np.ndarray,
    n_factors: int,
    max_iter: int = 1000,
    tol: float = 1e-4,  # ← Change to tol
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for Bayesian factor analysis with VI.
    """
    def fit_with_callback():
        if progress_callback:
            progress_callback(iteration=0, log_likelihood=None, delta=None, extra={"status": "starting"})
        
        result = fit_bayesian_factor_vi(
            data,
            n_factors=n_factors,
            max_iter=max_iter,
            tol=tol,  # ← Pass tol instead
        )
        
        if progress_callback:
            progress_callback(iteration=max_iter, log_likelihood=None, delta=None, extra={"status": "completed"})
        
        return result
    
    return await _run_in_executor(fit_with_callback)


# =============================================================================
# NMF WRAPPER
# =============================================================================

async def fit_nmf_with_progress(
    data: np.ndarray,
    n_components: int,
    max_iter: int = 200,
    init: str = "nndsvda",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for NMF.
    """
    def fit_with_callback():
        if progress_callback:
            progress_callback(iteration=0, log_likelihood=None, delta=None, extra={"status": "starting"})
        
        result = fit_nmf(
            data,
            n_components=n_components,
            max_iter=max_iter,
            init=init,
        )
        
        if progress_callback:
            progress_callback(iteration=max_iter, log_likelihood=None, delta=None, extra={"status": "completed"})
        
        return result
    
    return await _run_in_executor(fit_with_callback)


# =============================================================================
# MCA WRAPPER
# =============================================================================

async def fit_mca_with_progress(
    data: np.ndarray,
    n_components: int = 5,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for MCA.
    """
    if not PRINCE_AVAILABLE:
        raise ImportError("MCA requires the 'prince' package to be installed")
    
    def fit_with_callback():
        if progress_callback:
            progress_callback(iteration=0, log_likelihood=None, delta=None, extra={"status": "starting"})
        
        result = fit_mca(data, n_components=n_components)
        
        if progress_callback:
            progress_callback(iteration=1, log_likelihood=None, delta=None, extra={"status": "completed"})
        
        return result
    
    return await _run_in_executor(fit_with_callback)


# =============================================================================
# PYMC MODEL WRAPPERS
# =============================================================================

async def fit_bayesian_factor_pymc_with_progress(
    data: np.ndarray,
    n_factors: int,
    n_samples: int = 1000,
    n_tune: int = 500,
    n_chains: int = 4,
    target_accept: float = 0.9,
    callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for PyMC Bayesian factor model.
    
    The callback is a PyMCSamplingCallback that may be passed to PyMC's sample().
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC Bayesian factor model requires PyMC to be installed")
    
    def fit_with_callback():
        # Try to pass callback if the function supports it
        try:
            result = fit_bayesian_factor_model_pymc(
                data,
                n_factors=n_factors,
                n_samples=n_samples,
                n_tune=n_tune,
                n_chains=n_chains,
                target_accept=target_accept,
                callback=callback,
            )
        except TypeError:
            # Function doesn't accept callback, call without it
            result = fit_bayesian_factor_model_pymc(
                data,
                n_factors=n_factors,
                n_samples=n_samples,
                n_tune=n_tune,
                n_chains=n_chains,
                target_accept=target_accept,
            )
        return result
    
    return await _run_in_executor(fit_with_callback)


async def fit_dcm_with_progress(
    purchase_data: np.ndarray,
    household_features: Optional[np.ndarray] = None,
    n_samples: int = 1000,
    n_tune: int = 500,
    n_chains: int = 4,
    target_accept: float = 0.9,
    include_random_effects: bool = False,
    n_latent_features: int = 0,
    latent_prior_scale: float = 1.0,
    callback: Optional[Callable] = None,
) -> dict:
    """
    Async wrapper for Discrete Choice Model.
    
    The callback is a PyMCSamplingCallback that may be passed to PyMC's sample().
    """
    if not PYMC_AVAILABLE:
        raise ImportError("DCM requires PyMC to be installed")
    
    def fit_with_callback():
        # Try to pass callback if the function supports it
        try:
            result = fit_discrete_choice_model(
                purchase_data,
                household_features=household_features,
                n_samples=n_samples,
                n_tune=n_tune,
                n_chains=n_chains,
                target_accept=target_accept,
                include_random_effects=include_random_effects,
                n_latent_features=n_latent_features,
                latent_prior_scale=latent_prior_scale,
                callback=callback,
            )
        except TypeError:
            # Function doesn't accept callback, call without it
            result = fit_discrete_choice_model(
                purchase_data,
                household_features=household_features,
                n_samples=n_samples,
                n_tune=n_tune,
                n_chains=n_chains,
                target_accept=target_accept,
                include_random_effects=include_random_effects,
                n_latent_features=n_latent_features,
                latent_prior_scale=latent_prior_scale,
            )
        return result
    
    return await _run_in_executor(fit_with_callback)