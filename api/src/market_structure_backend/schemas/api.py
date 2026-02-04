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
    LDA = "lda"
    NETWORK = "network"


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


class LDAParams(BaseModel):
    """Parameters for Latent Dirichlet Allocation."""
    n_topics: int = Field(ge=2, le=50, description="Number of topics to discover")
    max_iter: int = Field(default=100, ge=10, le=500, description="Maximum iterations")
    learning_method: str = Field(default="online", description="Learning method: 'online' or 'batch'")


class NetworkAnalysisParams(BaseModel):
    """Parameters for Network Analysis."""
    threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum edge weight threshold")
    community_method: str = Field(default="louvain", description="Community detection method")
    edge_method: str = Field(default="lift", description="Edge weight calculation method")


# Union of all parameter types
ModelParams = Union[
    LCAParams,
    LCACovariatesParams,
    FactorParams,
    BayesianFactorPyMCParams,
    NMFParams,
    MCAParams,
    DCMParams,
    LDAParams,
    NetworkAnalysisParams,
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
    # Covariate data for LCA with covariates
    covariates_json: Optional[list[list[float]]] = Field(default=None, description="Household covariate matrix")
    covariate_column_names: Optional[list[str]] = Field(default=None, description="Covariate column names")

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

    # LCA with covariates fields
    beta: Optional[list[list[float]]] = None  # Regression coefficients (n_features x n_classes)
    odds_ratios: Optional[list[list[float]]] = None  # exp(beta) for interpretation
    covariate_columns: Optional[list[str]] = None  # Covariate column names
    class_probs_per_hh: Optional[list[list[float]]] = None  # Per-household class probs

    # Additional model-specific fields
    residual_correlations: Optional[list[list[float]]] = None  # For LCA models
    tetra_corr: Optional[list[list[float]]] = None  # Tetrachoric correlation matrix
    elbo_history: Optional[list[float]] = None  # For Bayesian VI models

    # LDA-specific fields
    topic_product_dist: Optional[list[list[float]]] = None  # (n_topics, n_products)
    household_topic_dist: Optional[list[list[float]]] = None  # (n_households, n_topics)
    perplexity: Optional[float] = None  # LDA perplexity score
    n_topics: Optional[int] = None  # Number of topics

    # Network Analysis-specific fields
    adjacency_matrix: Optional[list[list[float]]] = None  # Product co-purchase matrix
    communities: Optional[list[int]] = None  # Community assignments per product
    centrality_scores: Optional[list[float]] = None  # Eigenvector centrality
    degree_centrality: Optional[list[float]] = None  # Degree centrality
    betweenness_centrality: Optional[list[float]] = None  # Betweenness centrality
    graph_metrics: Optional[dict[str, Any]] = None  # Network statistics
    n_communities: Optional[int] = None  # Number of detected communities

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


# =============================================================================
# WORKER STATUS SCHEMAS
# =============================================================================

class JobInfo(BaseModel):
    """Information about a single ARQ job."""
    job_id: str
    function: str
    status: str  # queued, in_progress, complete, not_found
    enqueue_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    finish_time: Optional[datetime] = None
    result: Optional[Any] = None
    run_id: Optional[str] = None  # Associated model run ID


class QueueStats(BaseModel):
    """Statistics about the ARQ queue."""
    queued_jobs: int
    in_progress_jobs: int
    completed_jobs: int  # Jobs with results still in Redis
    total_keys: int


class WorkerInfo(BaseModel):
    """Information about an ARQ worker."""
    worker_id: str
    last_health_check: Optional[datetime] = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    current_job: Optional[str] = None


class WorkerStatusResponse(BaseModel):
    """Full worker status response."""
    redis_connected: bool
    queue_stats: Optional[QueueStats] = None
    workers: list[WorkerInfo] = []
    recent_jobs: list[JobInfo] = []
    pending_runs: list[dict[str, Any]] = []  # Runs in pending/running status


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    model_run_id: Optional[str] = None


# =============================================================================
# CLUSTERING SCHEMAS
# =============================================================================

class ClusteringMethodEnum(str, Enum):
    """Available clustering methods."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"


class ClusteringRequest(BaseModel):
    """Request to run clustering on model results."""
    method: ClusteringMethodEnum = Field(default=ClusteringMethodEnum.KMEANS, description="Clustering method")
    n_clusters: Optional[int] = Field(default=None, ge=2, le=20, description="Number of clusters (None = auto-detect)")
    max_k: int = Field(default=10, ge=2, le=20, description="Maximum k for auto-detection")
    linkage_method: str = Field(default="ward", description="Linkage method for hierarchical clustering")


class ClusteringResponse(BaseModel):
    """Response from clustering endpoint."""
    model_run_id: str
    method: ClusteringMethodEnum
    n_clusters: int
    labels: list[int]  # Cluster assignment for each product
    product_columns: list[str]  # Product names
    silhouette_score: Optional[float] = None
    inertia: Optional[float] = None  # For k-means
    # For hierarchical clustering
    linkage_matrix: Optional[list[list[float]]] = None
    # Auto-detection results
    optimal_k: Optional[int] = None
    silhouette_scores: Optional[list[float]] = None
    k_range: Optional[list[int]] = None
    # Cluster membership mapping
    cluster_members: Optional[dict[str, list[str]]] = None  # cluster_id -> list of products


# =============================================================================
# PRESENTATION SCHEMAS
# =============================================================================

class SlideTypeEnum(str, Enum):
    """Available slide types."""
    FIGURE = "figure"
    TEXT = "text"
    COMPARISON = "comparison"
    SUMMARY = "summary"


class FigureTypeEnum(str, Enum):
    """Available figure types for presentation slides."""
    SIMILARITY_MATRIX = "similarity_matrix"
    VARIANCE_EXPLAINED = "variance_explained"
    FACTOR_LOADINGS = "factor_loadings"
    CLASS_PROFILES = "class_profiles"
    BIPLOT = "biplot"
    LCA_CLASS_BIPLOT = "lca_class_biplot"
    TETRACHORIC = "tetrachoric"
    ELBO_HISTORY = "elbo_history"
    TOPIC_DISTRIBUTION = "topic_distribution"
    NETWORK_MATRIX = "network_matrix"
    DCM_COEFFICIENTS = "dcm_coefficients"
    CLUSTERED_BIPLOT = "clustered_biplot"
    SILHOUETTE_ANALYSIS = "silhouette_analysis"
    CLUSTER_SIZES = "cluster_sizes"
    DENDROGRAM = "dendrogram"


class PresentationSlideCreate(BaseModel):
    """Request to create a new presentation slide."""
    title: str = Field(max_length=255, description="Slide title")
    description: Optional[str] = Field(default=None, description="Slide description or commentary")
    slide_type: SlideTypeEnum = Field(default=SlideTypeEnum.FIGURE, description="Type of slide")
    model_run_id: Optional[str] = Field(default=None, description="Model run ID for figure slides")
    figure_type: Optional[FigureTypeEnum] = Field(default=None, description="Type of figure to display")
    figure_config: Optional[dict[str, Any]] = Field(default=None, description="Figure-specific configuration")
    text_content: Optional[str] = Field(default=None, description="Markdown content for text slides")
    layout: Optional[dict[str, Any]] = Field(default=None, description="Layout options")
    order: Optional[int] = Field(default=None, description="Slide order (auto-assigned if not provided)")

    model_config = ConfigDict(extra="forbid")


class PresentationSlideUpdate(BaseModel):
    """Request to update an existing slide."""
    title: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = None
    slide_type: Optional[SlideTypeEnum] = None
    model_run_id: Optional[str] = None
    figure_type: Optional[FigureTypeEnum] = None
    figure_config: Optional[dict[str, Any]] = None
    text_content: Optional[str] = None
    layout: Optional[dict[str, Any]] = None
    order: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class PresentationSlideResponse(BaseModel):
    """Response schema for a presentation slide."""
    id: str
    presentation_id: str
    order: int
    title: str
    description: Optional[str]
    slide_type: SlideTypeEnum
    model_run_id: Optional[str]
    figure_type: Optional[FigureTypeEnum]
    figure_config: Optional[dict[str, Any]]
    text_content: Optional[str]
    layout: Optional[dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


class BrandingOptions(BaseModel):
    """Branding options for presentations."""
    primary_color: Optional[str] = Field(default="#667eea", description="Primary theme color (hex)")
    secondary_color: Optional[str] = Field(default="#764ba2", description="Secondary theme color (hex)")
    logo_url: Optional[str] = Field(default=None, description="URL to logo image")
    font_family: Optional[str] = Field(default=None, description="Custom font family")


class PresentationCreate(BaseModel):
    """Request to create a new presentation."""
    name: str = Field(max_length=255, description="Presentation name")
    description: Optional[str] = Field(default=None, description="Presentation description")
    client_name: Optional[str] = Field(default=None, max_length=255, description="Client name for branding")
    project_name: Optional[str] = Field(default=None, max_length=255, description="Project name for branding")
    branding_options: Optional[BrandingOptions] = Field(default=None, description="Branding configuration")

    model_config = ConfigDict(extra="forbid")


class PresentationUpdate(BaseModel):
    """Request to update presentation metadata."""
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = None
    client_name: Optional[str] = Field(default=None, max_length=255)
    project_name: Optional[str] = Field(default=None, max_length=255)
    branding_options: Optional[BrandingOptions] = None

    model_config = ConfigDict(extra="forbid")


class PresentationResponse(BaseModel):
    """Response schema for a presentation."""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    client_name: Optional[str]
    project_name: Optional[str]
    branding_options: Optional[dict[str, Any]]
    slides: list[PresentationSlideResponse]
    slide_count: int

    model_config = ConfigDict(from_attributes=True)


class PresentationListResponse(BaseModel):
    """Response for listing presentations."""
    presentations: list[PresentationResponse]
    total: int


class SlideReorderRequest(BaseModel):
    """Request to reorder slides."""
    slide_ids: list[str] = Field(description="Ordered list of slide IDs")

    model_config = ConfigDict(extra="forbid")


class FigureInfo(BaseModel):
    """Information about an available figure type."""
    type: FigureTypeEnum
    name: str
    description: str
    available: bool = True


class RunFiguresResponse(BaseModel):
    """Available figures for a model run."""
    model_run_id: str
    model_type: ModelTypeEnum
    product_columns: list[str]
    available_figures: list[FigureInfo]


class FigureDataResponse(BaseModel):
    """Response containing figure data as Plotly JSON."""
    model_run_id: str
    figure_type: FigureTypeEnum
    figure_json: dict[str, Any]  # Plotly figure as JSON