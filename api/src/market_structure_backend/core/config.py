"""
Core configuration and settings management.

Uses Pydantic settings for environment-based configuration with
sensible defaults for local development.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_workers: int = 1
    
    # Redis Settings (for ARQ job queue and progress tracking)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Database Settings
    database_url: str = "sqlite+aiosqlite:///./model_runs.db"
    
    # Worker Settings
    worker_concurrency: int = 4
    job_timeout_seconds: int = 3600  # 1 hour max per job
    job_keep_result_seconds: int = 86400  # Keep results for 24 hours
    
    # Progress Update Settings
    progress_update_interval_seconds: float = 1.0
    
    # Model defaults
    default_mcmc_samples: int = 1000
    default_mcmc_tune: int = 500
    default_lca_max_iter: int = 100
    default_lca_n_init: int = 10
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def redis_settings(self) -> dict:
        """ARQ-compatible Redis settings dictionary."""
        settings = {
            "host": self.redis_host,
            "port": self.redis_port,
            "database": self.redis_db,
        }
        if self.redis_password:
            settings["password"] = self.redis_password
        return settings


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()