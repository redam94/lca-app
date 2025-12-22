"""
ARQ worker tasks for model fitting.

Each model type has a corresponding task function that:
1. Updates database status to RUNNING
2. Executes the model fit with progress callbacks
3. Stores results and updates status to COMPLETED or FAILED
"""

import json
import traceback
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arq import ArqRedis

from ..core.config import get_settings
from ..db import ModelRun, ModelRunStatus, ModelType, init_db, get_session_factory
from ..progress import ProgressTracker, PyMCSamplingCallback, EMProgressCallback, ProgressUpdate


# Results storage directory
RESULTS_DIR = Path("./model_results")
RESULTS_DIR.mkdir(exist_ok=True)


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
    
    return summary


# =============================================================================
# LCA TASK
# =============================================================================

async def fit_lca_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Latent Class Analysis.
    
    Args:
        ctx: ARQ context with Redis connection
        run_id: Model run ID
        data: Purchase data matrix (n_households x n_products)
        params: LCA parameters (n_classes, max_iter, n_init)
        product_columns: Product column names
    """
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        # Update status to RUNNING
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        # Convert data to numpy array
        X = np.array(data)
        
        # Import the model fitting function
        # Note: In production, you'd import from the market_structure package
        from .model_implementations import fit_lca_with_progress
        
        # Create progress callback
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )
        
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


# =============================================================================
# BAYESIAN FACTOR MODEL (PYMC) TASK
# =============================================================================

async def fit_bayesian_factor_pymc_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
):
    """
    ARQ task for fitting Bayesian Factor Model with PyMC MCMC.
    
    This task demonstrates full PyMC progress tracking via callbacks.
    """
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        X = np.array(data)
        
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


# =============================================================================
# DCM TASK
# =============================================================================

async def fit_dcm_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
    household_features: Optional[list[list[float]]] = None,
):
    """ARQ task for fitting Discrete Choice Model."""
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        X = np.array(data)
        hh_features = np.array(household_features) if household_features else None
        
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


# =============================================================================
# SIMPLER MODEL TASKS
# =============================================================================

async def fit_factor_tetrachoric_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Tetrachoric Factor Analysis."""
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        X = np.array(data)
        
        from .model_implementations import fit_factor_tetrachoric_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )
        
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


async def fit_nmf_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Non-negative Matrix Factorization."""
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        X = np.array(data)
        
        from .model_implementations import fit_nmf_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 200),
        )
        
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


async def fit_bayesian_vi_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Bayesian Factor Model with Variational Inference."""
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        X = np.array(data)
        
        from .model_implementations import fit_bayesian_vi_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=params.get("max_iter", 100),
        )
        
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


async def fit_mca_task(
    ctx: dict,
    run_id: str,
    data: list[list[float]],
    params: dict,
    product_columns: list[str],
):
    """ARQ task for Multiple Correspondence Analysis."""
    settings = get_settings()
    await init_db(settings.database_url)
    session_factory = get_session_factory()
    tracker = await ProgressTracker.create(settings.redis_url)
    
    try:
        await _update_run_status(
            session_factory, run_id, ModelRunStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        await tracker.start_tracking(run_id)
        
        X = np.array(data)
        
        from .model_implementations import fit_mca_with_progress
        
        progress_callback = EMProgressCallback(
            model_run_id=run_id,
            tracker=tracker,
            max_iter=1,  # MCA is not iterative
        )
        
        result = await fit_mca_with_progress(
            X,
            n_components=params.get("n_components", 5),
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


# =============================================================================
# TASK REGISTRY
# =============================================================================

# Map model types to task functions
TASK_REGISTRY = {
    "lca": fit_lca_task,
    "lca_covariates": fit_lca_task,  # Same task, different params
    "factor_tetrachoric": fit_factor_tetrachoric_task,
    "bayesian_factor_vi": fit_bayesian_vi_task,
    "bayesian_factor_pymc": fit_bayesian_factor_pymc_task,
    "nmf": fit_nmf_task,
    "mca": fit_mca_task,
    "dcm": fit_dcm_task,
}


def get_task_for_model_type(model_type: str):
    """Get the appropriate task function for a model type."""
    task = TASK_REGISTRY.get(model_type)
    if task is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return task