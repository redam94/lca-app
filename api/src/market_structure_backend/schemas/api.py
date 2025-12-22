"""
Pydantic schemas for API request/response validation.

These schemas define the contract between frontend and backend,
ensuring type safety and validation for all API operations.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMS
# =============================================================================

class ModelTypeEnum(str, Enum):
    """Available model types."""
    LCA = "lca"
    LCA_COVARIATES = "lca_covariates"
    FACTOR_TETRACHORIC = "factor_tetrachoric"
    BAYESIAN_FACTOR_VI = "bayesian_factor_vi"
    BAYESIAN_FACTOR_PYMC = "bayesian_factor_pymc"
    NMF = "nmf"
    MCA = "mca"
    DCM = "dcm"


class ModelRunStatusEnum(str, Enum):
    """Model run status values."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# MODEL PARAMETER SCHEMAS
# =============================================================================

class LCAParams(BaseModel):
    """Parameters for Latent Class Analysis."""
    n_classes: int = Field(ge=2, le=20, description="Number of latent classes")
    max_iter: int = Field(default=100, ge=10, le=1000, description="Maximum EM iterations")
    n_init: int = Field(default=10, ge=1, le=50, description="Number of random initializations")
    tol: float = Field(default=1e-6, ge=1e-10, le=1e-2, description="Convergence tolerance")


class LCACovariatesParams(LCAParams):
    """Parameters for LCA with household covariates."""
    covariate_columns: list[str] = Field(description="Column names for household covariates")


class FactorParams(BaseModel):
    """Parameters for Factor Analysis models."""
    n_factors: int = Field(ge=1, le=20, description="Number of latent factors")
    max_iter: int = Field(default=100, ge=10, le=1000, description="Maximum iterations")


class BayesianFactorPyMCParams(FactorParams):
    """Parameters for PyMC Bayesian Factor Model."""
    n_samples: int = Field(default=1000, ge=100, le=10000, description="Number of posterior samples")
    n_tune: int = Field(default=500, ge=100, le=5000, description="Number of tuning samples")
    n_chains: int = Field(default=4, ge=1, le=8, description="Number of MCMC chains")
    target_accept: float = Field(default=0.9, ge=0.5, le=0.99, description="Target acceptance rate")


class NMFParams(BaseModel):
    """Parameters for Non-negative Matrix Factorization."""
    n_components: int = Field(ge=1, le=20, description="Number of components")
    max_iter: int = Field(default=200, ge=10, le=1000, description="Maximum iterations")
    init: str = Field(default="nndsvda", description="Initialization method")


class MCAParams(BaseModel):
    """Parameters for Multiple Correspondence Analysis."""
    n_components: int = Field(ge=2, le=20, description="Number of components")


class DCMParams(BaseModel):
    """Parameters for Discrete Choice Model."""
    n_samples: int = Field(default=1000, ge=100, le=10000, description="Number of posterior samples")
    n_tune: int = Field(default=500, ge=100, le=5000, description="Number of tuning samples")
    n_chains: int = Field(default=4, ge=1, le=8, description="Number of MCMC chains")
    include_random_effects: bool = Field(default=False, description="Include household random effects")
    n_latent_features: int = Field(default=0, ge=0, le=10, description="Number of latent product features")
    latent_prior_scale: float = Field(default=1.0, ge=0.1, le=10.0, description="Prior scale for latent features")
    household_feature_columns: Optional[list[str]] = Field(default=None, description="Household feature columns")


# Union of all parameter types
ModelParams = Union[
    LCAParams,
    LCACovariatesParams,
    FactorParams,
    BayesianFactorPyMCParams,
    NMFParams,
    MCAParams,
    DCMParams,
]


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class DataUpload(BaseModel):
    """Schema for uploaded data."""
    # Data can be provided as base64 CSV or as a pre-parsed array
    csv_base64: Optional[str] = Field(default=None, description="Base64 encoded CSV data")
    data_json: Optional[list[list[float]]] = Field(default=None, description="Pre-parsed data matrix")
    column_names: Optional[list[str]] = Field(default=None, description="Column names")
    
    model_config = ConfigDict(extra="forbid")


class ModelRunRequest(BaseModel):
    """Request to submit a new model run."""
    model_type: ModelTypeEnum
    name: Optional[str] = Field(default=None, max_length=255, description="Optional name for this run")
    description: Optional[str] = Field(default=None, description="Optional description")
    
    # Model-specific parameters (validated based on model_type)
    params: dict[str, Any] = Field(description="Model-specific parameters")
    
    # Data reference (either upload inline or reference existing)
    data: Optional[DataUpload] = Field(default=None, description="Inline data upload")
    data_id: Optional[str] = Field(default=None, description="Reference to previously uploaded data")
    
    # Product selection
    product_columns: Optional[list[str]] = Field(default=None, description="Columns to use as products")
    
    model_config = ConfigDict(extra="forbid")


class ModelRunListParams(BaseModel):
    """Query parameters for listing model runs."""
    status: Optional[ModelRunStatusEnum] = None
    model_type: Optional[ModelTypeEnum] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    order_by: str = Field(default="created_at", pattern="^(created_at|completed_at|name)$")
    order_dir: str = Field(default="desc", pattern="^(asc|desc)$")


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class ProgressResponse(BaseModel):
    """Real-time progress update."""
    model_run_id: str
    progress: float = Field(ge=-1.0, le=1.0, description="Progress 0-1, or -1 for failure")
    message: str
    timestamp: datetime
    phase: str
    
    # MCMC-specific
    chain: Optional[int] = None
    draw: Optional[int] = None
    total_draws: Optional[int] = None
    
    # Performance
    samples_per_second: Optional[float] = None
    divergences: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    
    extra: Optional[dict[str, Any]] = None


class ModelRunResponse(BaseModel):
    """Response schema for a model run."""
    id: str
    model_type: ModelTypeEnum
    name: Optional[str]
    description: Optional[str]
    status: ModelRunStatusEnum
    
    # Timestamps
    created_at: datetime
    queued_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Duration
    queue_duration: Optional[float]
    run_duration: Optional[float]
    
    # Configuration
    model_params: dict[str, Any]
    data_shape: Optional[dict[str, int]]
    product_columns: Optional[list[str]]
    
    # Progress
    progress: float
    progress_message: Optional[str]
    
    # Results (summary only - full results via separate endpoint)
    results_summary: Optional[dict[str, Any]]
    metrics: Optional[dict[str, Any]]
    
    # Errors
    error_message: Optional[str]
    
    model_config = ConfigDict(from_attributes=True)


class ModelRunListResponse(BaseModel):
    """Response for listing model runs."""
    runs: list[ModelRunResponse]
    total: int
    limit: int
    offset: int


class ModelResultsResponse(BaseModel):
    """Full model results response."""
    model_run_id: str
    model_type: ModelTypeEnum
    status: ModelRunStatusEnum
    
    # Full results data (structure depends on model type)
    results: dict[str, Any]
    
    # Embeddings for visualization
    product_embeddings: Optional[list[list[float]]] = None
    household_embeddings: Optional[list[list[float]]] = None
    
    # Similarity/correlation matrix
    similarity_matrix: Optional[list[list[float]]] = None
    
    # Variance explained - NOTE: renamed from var_explained_pct to match frontend
    variance_explained: Optional[list[float]] = None
    
    # ==========================================
    # NEW FIELDS - Added to match frontend types
    # ==========================================
    
    # Factor loadings (for factor-type models)
    loadings: Optional[list[list[float]]] = None
    loadings_std: Optional[list[list[float]]] = None
    
    # LCA-specific fields
    item_probs: Optional[list[list[float]]] = None  # (n_classes, n_items)
    class_probs: Optional[list[float]] = None  # (n_classes,)
    
    # DCM-specific fields
    alpha: Optional[list[float]] = None  # Product intercepts
    alpha_std: Optional[list[float]] = None  # Intercept std errors
    product_latent: Optional[list[list[float]]] = None  # Latent product features
    household_latent: Optional[list[list[float]]] = None  # Latent household preferences
    
    # Product labels
    product_columns: list[str]
    
    # Metrics
    metrics: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    redis_connected: bool
    database_connected: bool
    workers_active: int


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    model_run_id: Optional[str] = None