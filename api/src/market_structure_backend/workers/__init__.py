"""ARQ worker module for background model fitting."""

from .tasks import (
    fit_lca_task,
    fit_bayesian_factor_pymc_task,
    fit_dcm_task,
    fit_factor_tetrachoric_task,
    fit_nmf_task,
    fit_mca_task,
    fit_bayesian_vi_task,
    TASK_REGISTRY,
    get_task_for_model_type,
)

from .runner import (
    WorkerSettings,
    run_worker,
    enqueue_job,
    get_job_status,
)

__all__ = [
    # Tasks
    "fit_lca_task",
    "fit_bayesian_factor_pymc_task",
    "fit_dcm_task",
    "fit_factor_tetrachoric_task",
    "fit_nmf_task",
    "fit_mca_task",
    "fit_bayesian_vi_task",
    "TASK_REGISTRY",
    "get_task_for_model_type",
    # Runner
    "WorkerSettings",
    "run_worker",
    "enqueue_job",
    "get_job_status",
]