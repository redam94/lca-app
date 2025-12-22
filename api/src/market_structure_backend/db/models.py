"""
SQLAlchemy database models for model run tracking.

Tracks all model runs with timestamps, status, parameters, and results.
Uses async SQLAlchemy for non-blocking database operations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, JSON, Enum as SQLEnum,
    Index, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


class ModelRunStatus(str, Enum):
    """Status of a model run."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """Supported model types."""
    LCA = "lca"
    LCA_COVARIATES = "lca_covariates"
    FACTOR_TETRACHORIC = "factor_tetrachoric"
    BAYESIAN_FACTOR_VI = "bayesian_factor_vi"
    BAYESIAN_FACTOR_PYMC = "bayesian_factor_pymc"
    NMF = "nmf"
    MCA = "mca"
    DCM = "dcm"


class ModelRun(Base):
    """
    Tracks a single model run with full metadata.
    
    Each run has a unique ID, timestamps for lifecycle events,
    the model configuration used, and the results when complete.
    """
    __tablename__ = "model_runs"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Model identification
    model_type: Mapped[str] = mapped_column(SQLEnum(ModelType), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Status tracking
    status: Mapped[str] = mapped_column(
        SQLEnum(ModelRunStatus), 
        default=ModelRunStatus.PENDING,
        nullable=False
    )
    
    # Job queue reference
    arq_job_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Duration tracking (in seconds)
    queue_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    run_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Input configuration
    model_params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    data_shape: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # {"n_obs": X, "n_items": Y}
    product_columns: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    
    # Progress tracking (0.0 to 1.0)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    progress_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Results (stored as JSON for flexibility)
    results_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    results_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # Path to full results
    
    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Metrics (model-specific)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # BIC, AIC, WAIC, etc.
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_model_runs_status", "status"),
        Index("ix_model_runs_model_type", "model_type"),
        Index("ix_model_runs_created_at", "created_at"),
    )
    
    def to_dict(self) -> dict:
        """Convert model run to dictionary for API responses."""
        return {
            "id": self.id,
            "model_type": self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            "name": self.name,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, ModelRunStatus) else self.status,
            "arq_job_id": self.arq_job_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "queue_duration": self.queue_duration,
            "run_duration": self.run_duration,
            "model_params": self.model_params,
            "data_shape": self.data_shape,
            "product_columns": self.product_columns,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "results_summary": self.results_summary,
            "results_path": self.results_path,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class ProgressSnapshot(Base):
    """
    Time-series record of progress updates for a model run.
    
    Captures detailed progress information at regular intervals,
    useful for debugging and understanding model behavior.
    """
    __tablename__ = "progress_snapshots"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("model_runs.id", ondelete="CASCADE"),
        nullable=False
    )
    
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    
    progress: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # MCMC-specific progress info
    chain: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    draw: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tune: Mapped[Optional[bool]] = mapped_column(Integer, nullable=True)  # 1 = tuning, 0 = sampling
    
    # Performance metrics
    samples_per_second: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    divergences: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Extra data
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    __table_args__ = (
        Index("ix_progress_snapshots_model_run_id", "model_run_id"),
        Index("ix_progress_snapshots_timestamp", "timestamp"),
    )


# Database engine and session factory
_engine = None
_async_session = None


async def init_db(database_url: str = "sqlite+aiosqlite:///./model_runs.db"):
    """Initialize the database engine and create tables."""
    global _engine, _async_session
    
    _engine = create_async_engine(
        database_url,
        echo=False,  # Set to True for SQL debugging
        future=True,
    )
    
    _async_session = async_sessionmaker(
        _engine,
        expire_on_commit=False,
    )
    
    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    return _engine


async def get_session():
    """Get an async database session."""
    if _async_session is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    
    async with _async_session() as session:
        yield session


def get_session_factory():
    """Get the async session factory."""
    if _async_session is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return _async_session