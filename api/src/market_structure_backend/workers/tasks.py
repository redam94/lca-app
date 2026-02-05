"""
ARQ worker tasks for model fitting.

Each model type has a corresponding task function that:
1. Updates database status to RUNNING
2. Executes the model fit with progress callbacks
3. Stores results and updates status to COMPLETED or FAILED

Key design: Database session_factory is obtained from the worker context (ctx)
which is initialized once at worker startup. This prevents issues with aiosqlite
connections being garbage collected in PyMC's ThreadPoolExecutor threads.
"""

import gc
import json
import traceback
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arq import ArqRedis

from ..core.config import get_settings
from ..db import ModelRun, ModelRunStatus, ModelType, get_session_factory
from ..progress import ProgressTracker, PyMCSamplingCallback, EMProgressCallback, ProgressUpdate


# Results storage directory
RESULTS_DIR = Path("./model_results")
RESULTS_DIR.mkdir(exist_ok=True)


def _get_session_factory(ctx: dict):
    """
    Get session factory from context or fall back to global.
    
    The session factory should be in ctx from worker startup,
    but we provide a fallback for safety.
    """
    if 'session_factory' in ctx:
        return ctx['session_factory']
    # Fallback - shouldn't happen if worker started correctly
    return get_session_factory()


def _get_settings(ctx: dict):
    """
    Get settings from context or fall back to global.
    """
    if 'settings' in ctx:
        return ctx['settings']
    return get_settings()


async def _update_run_status(
    session_factory,
    run_id: str,
    status: ModelRunStatus,
    **kwargs
):
    """Update model run status in database."""
    async with session_factory() as session:
        run = await session.get(ModelRun, run_id)
        if run:
            run.status = status
            for key, value in kwargs.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            await session.commit()


async def _save_results(run_id: str, results: dict) -> str:
    """Save full results to disk and return the path."""
    results_path = RESULTS_DIR / f"{run_id}.pkl"

    # Convert numpy arrays to lists for JSON serialization in summary
    # Keep original arrays in pickle for full results
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    return str(results_path)


def _load_input_data(data_path: str) -> dict:
    """
    Load input data from disk (saved by the API before enqueuing).

    Returns dict with 'data' key (numpy array) and optionally
    'covariates' and 'covariate_columns' keys.
    """
    with open(data_path, "rb") as f:
        payload = pickle.load(f)
    return payload


def _cleanup_input_data(data_path: str):
    """Remove input data file after task completes."""
    try:
        Path(data_path).unlink(missing_ok=True)
    except Exception:
        pass


def _numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _create_results_summary(results: dict, model_type: str) -> dict:
    """Create a summary of results for quick access."""
    summary = {}
    
    # Common fields
    if "n_iter" in results:
        summary["n_iter"] = results["n_iter"]
    if "log_likelihood" in results:
        summary["log_likelihood"] = float(results["log_likelihood"])
    if "bic" in results:
        summary["bic"] = float(results["bic"])
    if "aic" in results:
        summary["aic"] = float(results["aic"])
    
    # Model-specific summaries
    if model_type in ["lca", "lca_covariates"]:
        if "class_probs" in results:
            summary["class_probs"] = _numpy_to_list(results["class_probs"])
        if "n_classes" in results:
            summary["n_classes"] = results["n_classes"]
    
    elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc"]:
        if "var_explained_pct" in results:
            summary["var_explained_pct"] = _numpy_to_list(results["var_explained_pct"])
        if "n_factors" in results:
            summary["n_factors"] = results["n_factors"]
    
    elif model_type == "nmf":
        if "reconstruction_error" in results:
            summary["reconstruction_error"] = float(results["reconstruction_error"])
        if "var_explained_pct" in results:
            summary["var_explained_pct"] = _numpy_to_list(results["var_explained_pct"])
    
    elif model_type == "dcm":
        if "waic" in results and results["waic"] is not None:
            try:
                summary["waic"] = float(results["waic"].elpd_waic)
            except:
                pass
        if "n_divergences" in results:
            summary["n_divergences"] = results["n_divergences"]
    
    elif model_type == "mca":
        if "var_explained_pct" in results:
            summary["var_explained_pct"] = _numpy_to_list(results["var_explained_pct"])
        if "total_inertia" in results:
            summary["total_inertia"] = float(results["total_inertia"])
        if "n_components" in results:
            summary["n_components"] = results["n_components"]

    elif model_type == "lda":
        if "var_explained_pct" in results:
            summary["var_explained_pct"] = _numpy_to_list(results["var_explained_pct"])
        if "perplexity" in results:
            summary["perplexity"] = float(results["perplexity"])
        if "n_topics" in results:
            summary["n_topics"] = results["n_topics"]

    elif model_type == "network":
        if "n_communities" in results:
            summary["n_communities"] = results["n_communities"]
        if "graph_metrics" in results:
            gm = results["graph_metrics"]
            summary["modularity"] = float(gm.get("modularity", 0))
            summary["density"] = float(gm.get("density", 0))
            summary["n_edges"] = int(gm.get("n_edges", 0))

    return summary


# =============================================================================
# LCA TASK
# =============================================================================

async def fit_lca_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Latent Class Analysis.

    Args:
        ctx: ARQ context with session_factory and settings from worker startup
        run_id: Model run ID
        data_path: Path to pickled input data file
        params: LCA parameters (n_classes, max_iter, n_init)
        product_columns: Product column names
    """
    # Get resources from context (initialized at worker startup)
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)

    tracker = await ProgressTracker.create(settings.redis_url)

    try:
        # Update status to RUNNING
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)

        # Load data from disk
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        
        # Import the model fitting function
        from .model_implementations import fit_lca_with_progress
        
        # Create progress callback
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )
        
        # Force garbage collection before executor to clean up any lingering connections
        gc.collect()
        
        # Fit the model
        result = await fit_lca_with_progress(
            X,
            n_classes=params["n_classes"],
            max_iter=params.get("max_iter", 100),
            n_init=params.get("n_init", 10),
            tol=params.get("tol", 1e-6),
            progress_callback=progress_callback,
        )
        
        # Save results
        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "lca")
        
        # Extract metrics
        metrics = {
            "bic": float(result.get("bic", 0)),
            "aic": float(result.get("aic", 0)),
            "log_likelihood": float(result.get("log_likelihood", 0)),
        }
        
        # Update database
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="Model completed successfully",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )
        
        await tracker.complete(run_id, "LCA model completed successfully")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        error_tb = traceback.format_exc()
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=error_tb,
            progress=-1.0,
            progress_message=f"Failed: {error_msg}",
        )
        
        await tracker.fail(run_id, error_msg)

        return {"status": "failed", "run_id": run_id, "error": error_msg}

    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# LCA WITH COVARIATES TASK
# =============================================================================

async def fit_lca_covariates_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Latent Class Analysis with household covariates.

    Args:
        ctx: ARQ context with session_factory and settings from worker startup
        run_id: Model run ID
        data_path: Path to pickled input data file
        params: LCA parameters (n_classes, max_iter, n_init)
        product_columns: Product column names
    """
    # Get resources from context (initialized at worker startup)
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)

    tracker = await ProgressTracker.create(settings.redis_url)

    try:
        # Update status to RUNNING
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)

        # Load data from disk
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])

        covariates = payload.get("covariates")
        covariate_columns = payload.get("covariate_columns")

        if covariates is None or len(covariates) == 0:
            raise ValueError("LCA with covariates requires covariate data")

        Z = np.array(covariates)

        # Standardize covariates for better optimization
        covariate_means = Z.mean(axis=0)
        covariate_stds = Z.std(axis=0) + 1e-10
        Z_standardized = (Z - covariate_means) / covariate_stds

        # Import the model fitting function
        from .model_implementations import fit_lca_covariates_with_progress

        # Create progress callback
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )

        # Force garbage collection before executor
        gc.collect()

        # Fit the model
        result = await fit_lca_covariates_with_progress(
            X,
            Z_standardized,
            n_classes=params["n_classes"],
            max_iter=params.get("max_iter", 100),
            n_init=params.get("n_init", 10),
            tol=params.get("tol", 1e-6),
            progress_callback=progress_callback,
        )

        # Add covariate metadata to result
        result['covariate_columns'] = covariate_columns or [f"Covariate_{i}" for i in range(Z.shape[1])]
        result['covariate_means'] = covariate_means.tolist()
        result['covariate_stds'] = covariate_stds.tolist()
        result['feature_names'] = ['Intercept'] + result['covariate_columns']

        # Compute odds ratios for interpretation
        if 'beta' in result:
            odds_ratios = np.exp(result['beta'])
            result['odds_ratios'] = odds_ratios

        # Save results
        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "lca_covariates")

        # Extract metrics
        metrics = {
            "bic": float(result.get("bic", 0)),
            "aic": float(result.get("aic", 0)),
            "log_likelihood": float(result.get("log_likelihood", 0)),
            "n_features": int(result.get("n_features", 0)),
        }

        # Update database
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="LCA with covariates completed successfully",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )

        await tracker.complete(run_id, "LCA with covariates completed successfully")

        return {"status": "completed", "run_id": run_id}

    except Exception as e:
        error_msg = str(e)
        error_tb = traceback.format_exc()

        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=error_tb,
            progress=-1.0,
            progress_message=f"Failed: {error_msg}",
        )

        await tracker.fail(run_id, error_msg)

        return {"status": "failed", "run_id": run_id, "error": error_msg}

    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# BAYESIAN FACTOR MODEL (PYMC) TASK
# =============================================================================

async def fit_bayesian_factor_pymc_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Bayesian Factor Model with PyMC MCMC.
    
    This task demonstrates full PyMC progress tracking via callbacks.
    
    Note: PyMC runs in a ThreadPoolExecutor. To avoid aiosqlite event loop
    issues, we get the session_factory from context (initialized at startup)
    and force garbage collection before entering the executor.
    """
    # Get resources from context (initialized at worker startup)
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        
        from .model_implementations import fit_bayesian_factor_pymc_with_progress
        
        # Create PyMC sampling callback
        n_samples = params.get("n_samples", 1000)
        n_tune = params.get("n_tune", 500)
        n_chains = params.get("n_chains", 4)
        
        pymc_callback = PyMCSamplingCallback(
            model_run_id=run_id,
            tracker=tracker,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
        )
        
        # CRITICAL: Force garbage collection before entering ThreadPoolExecutor
        # This prevents aiosqlite connections from being GC'd in worker threads
        gc.collect()
        
        # Fit the model
        result = await fit_bayesian_factor_pymc_with_progress(
            X,
            n_factors=params["n_factors"],
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            target_accept=params.get("target_accept", 0.9),
            callback=pymc_callback,
        )
        
        # Save results (excluding the trace which can be huge)
        result_to_save = {k: v for k, v in result.items() if k != "trace"}
        results_path = await _save_results(run_id, result_to_save)
        results_summary = _create_results_summary(result, "bayesian_factor_pymc")
        
        metrics = {
            "n_divergences": result.get("n_divergences", 0),
        }
        if result.get("waic") is not None:
            try:
                metrics["waic"] = float(result["waic"].elpd_waic)
            except:
                pass
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="MCMC sampling completed",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )
        
        await tracker.complete(run_id, "Bayesian Factor Model completed successfully")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        error_tb = traceback.format_exc()
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=error_tb,
            progress=-1.0,
            progress_message=f"Failed: {error_msg}",
        )
        
        await tracker.fail(run_id, error_msg)
        
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# DCM TASK
# =============================================================================

async def fit_dcm_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """ARQ task for fitting Discrete Choice Model."""
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        household_features_raw = payload.get("covariates")
        hh_features = np.array(household_features_raw) if household_features_raw is not None else None
        
        from .model_implementations import fit_dcm_with_progress
        
        n_samples = params.get("n_samples", 1000)
        n_tune = params.get("n_tune", 500)
        n_chains = params.get("n_chains", 4)
        
        pymc_callback = PyMCSamplingCallback(
            model_run_id=run_id,
            tracker=tracker,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
        )
        
        # Force garbage collection before entering ThreadPoolExecutor
        gc.collect()
        
        result = await fit_dcm_with_progress(
            X,
            household_features=hh_features,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            include_random_effects=params.get("include_random_effects", False),
            n_latent_features=params.get("n_latent_features", 0),
            latent_prior_scale=params.get("latent_prior_scale", 1.0),
            callback=pymc_callback,
        )
        
        result_to_save = {k: v for k, v in result.items() if k != "trace"}
        results_path = await _save_results(run_id, result_to_save)
        results_summary = _create_results_summary(result, "dcm")
        
        metrics = {"n_divergences": result.get("n_divergences", 0)}
        if result.get("waic") is not None:
            try:
                metrics["waic"] = float(result["waic"].elpd_waic)
            except:
                pass
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="DCM completed",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )
        
        await tracker.complete(run_id, "Discrete Choice Model completed successfully")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        error_tb = traceback.format_exc()
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=error_tb,
        )
        
        await tracker.fail(run_id, error_msg)
        
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# SIMPLER MODEL TASKS
# =============================================================================

async def fit_factor_tetrachoric_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Tetrachoric Factor Analysis."""
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        
        from .model_implementations import fit_factor_tetrachoric_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )
        
        gc.collect()
        
        result = await fit_factor_tetrachoric_with_progress(
            X,
            n_factors=params["n_factors"],
            max_iter=params.get("max_iter", 100),
            progress_callback=progress_callback,
        )
        
        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "factor_tetrachoric")
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="Factor Analysis completed",
            results_path=results_path,
            results_summary=results_summary,
        )
        
        await tracker.complete(run_id, "Tetrachoric Factor Analysis completed")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=traceback.format_exc(),
        )
        await tracker.fail(run_id, error_msg)
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


async def fit_nmf_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Non-negative Matrix Factorization."""
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        
        from .model_implementations import fit_nmf_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 200),
        )
        
        gc.collect()
        
        result = await fit_nmf_with_progress(
            X,
            n_components=params["n_components"],
            max_iter=params.get("max_iter", 200),
            init=params.get("init", "nndsvda"),
            progress_callback=progress_callback,
        )
        
        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "nmf")
        
        metrics = {"reconstruction_error": float(result.get("reconstruction_error", 0))}
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="NMF completed",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )
        
        await tracker.complete(run_id, "NMF completed successfully")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=traceback.format_exc(),
        )
        await tracker.fail(run_id, error_msg)
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


async def fit_bayesian_vi_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Bayesian Factor Model with Variational Inference."""
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        
        from .model_implementations import fit_bayesian_vi_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )
        
        gc.collect()
        
        result = await fit_bayesian_vi_with_progress(
            X,
            n_factors=params["n_factors"],
            max_iter=params.get("max_iter", 100),
            progress_callback=progress_callback,
        )
        
        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "bayesian_factor_vi")
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="Bayesian VI completed",
            results_path=results_path,
            results_summary=results_summary,
        )
        
        await tracker.complete(run_id, "Bayesian Factor Model (VI) completed")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=traceback.format_exc(),
        )
        await tracker.fail(run_id, error_msg)
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


async def fit_mca_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Multiple Correspondence Analysis."""
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        payload = _load_input_data(data_path)
        X = np.array(payload["data"])
        
        from .model_implementations import fit_mca_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=1,  # MCA is not iterative
        )
        
        gc.collect()
        
        result = await fit_mca_with_progress(
            X,
            n_components=params.get("n_components", 5),
            product_names=product_columns,
            progress_callback=progress_callback,
        )
        
        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "mca")
        
        metrics = {
            "total_inertia": float(result.get("total_inertia", 0)),
            "n_components": int(result.get("n_components", 0)),
        }
        
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="MCA completed",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )
        
        await tracker.complete(run_id, "MCA completed successfully")
        
        return {"status": "completed", "run_id": run_id}
        
    except Exception as e:
        error_msg = str(e)
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=traceback.format_exc(),
        )
        await tracker.fail(run_id, error_msg)
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# LDA TASK
# =============================================================================

async def fit_lda_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Latent Dirichlet Allocation.

    Args:
        ctx: ARQ context with session_factory and settings from worker startup
        run_id: Model run ID
        data_path: Path to pickled input data file
        params: LDA parameters (n_topics, max_iter, learning_method)
        product_columns: Product column names
    """
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)

    tracker = await ProgressTracker.create(settings.redis_url)

    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)

        payload = _load_input_data(data_path)
        X = np.array(payload["data"])

        from .model_implementations import fit_lda_with_progress

        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )

        gc.collect()

        result = await fit_lda_with_progress(
            X,
            n_topics=params["n_topics"],
            max_iter=params.get("max_iter", 100),
            learning_method=params.get("learning_method", "online"),
            progress_callback=progress_callback,
        )

        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "lda")

        metrics = {
            "perplexity": float(result.get("perplexity", 0)),
            "log_likelihood": float(result.get("log_likelihood", 0)),
            "n_topics": int(result.get("n_topics", 0)),
        }

        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="LDA completed",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )

        await tracker.complete(run_id, "LDA completed successfully")

        return {"status": "completed", "run_id": run_id}

    except Exception as e:
        error_msg = str(e)
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=traceback.format_exc(),
        )
        await tracker.fail(run_id, error_msg)
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# NETWORK ANALYSIS TASK
# =============================================================================

async def fit_network_task(
    ctx: dict,
    run_id: str,
    data_path: str,
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Network Analysis.

    Args:
        ctx: ARQ context with session_factory and settings from worker startup
        run_id: Model run ID
        data_path: Path to pickled input data file
        params: Network parameters (threshold, community_method, edge_method)
        product_columns: Product column names
    """
    # Get resources from context
    settings = _get_settings(ctx)
    session_factory = _get_session_factory(ctx)

    tracker = await ProgressTracker.create(settings.redis_url)

    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)

        payload = _load_input_data(data_path)
        X = np.array(payload["data"])

        from .model_implementations import fit_network_with_progress

        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=1,  # Network analysis is not iterative
        )

        gc.collect()

        result = await fit_network_with_progress(
            X,
            threshold=params.get("threshold", 0.1),
            community_method=params.get("community_method", "louvain"),
            edge_method=params.get("edge_method", "lift"),
            progress_callback=progress_callback,
        )

        results_path = await _save_results(run_id, result)
        results_summary = _create_results_summary(result, "network")

        graph_metrics = result.get("graph_metrics", {})
        metrics = {
            "n_communities": int(result.get("n_communities", 0)),
            "modularity": float(graph_metrics.get("modularity", 0)),
            "density": float(graph_metrics.get("density", 0)),
            "n_edges": int(graph_metrics.get("n_edges", 0)),
        }

        await _update_run_status(
            session_factory, run_id, ModelRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            progress=1.0,
            progress_message="Network analysis completed",
            results_path=results_path,
            results_summary=results_summary,
            metrics=metrics,
        )

        await tracker.complete(run_id, "Network analysis completed successfully")

        return {"status": "completed", "run_id": run_id}

    except Exception as e:
        error_msg = str(e)
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_msg,
            error_traceback=traceback.format_exc(),
        )
        await tracker.fail(run_id, error_msg)
        return {"status": "failed", "run_id": run_id, "error": error_msg}
    finally:
        await tracker.close()
        _cleanup_input_data(data_path)


# =============================================================================
# TASK REGISTRY
# =============================================================================

# Map model types to task functions
TASK_REGISTRY = {
    "lca": fit_lca_task,
    "lca_covariates": fit_lca_covariates_task,  # Separate task for covariates
    "factor_tetrachoric": fit_factor_tetrachoric_task,
    "bayesian_factor_vi": fit_bayesian_vi_task,
    "bayesian_factor_pymc": fit_bayesian_factor_pymc_task,
    "nmf": fit_nmf_task,
    "mca": fit_mca_task,
    "dcm": fit_dcm_task,
    "lda": fit_lda_task,
    "network": fit_network_task,
}


def get_task_for_model_type(model_type: str):
    """Get the appropriate task function for a model type."""
    task = TASK_REGISTRY.get(model_type)
    if task is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return task