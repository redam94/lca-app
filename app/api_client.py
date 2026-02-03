"""
API client for communicating with the Market Structure Analysis backend.

This module provides a synchronous client for the FastAPI backend,
designed for use within Streamlit applications.
"""

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Optional, Generator, Any

import requests
import pandas as pd
import numpy as np


@dataclass
class ProgressUpdate:
    """Real-time progress update from the API."""
    progress: float
    message: str
    phase: str
    eta_seconds: Optional[float] = None
    chain: Optional[int] = None
    draw: Optional[int] = None
    total_draws: Optional[int] = None
    samples_per_second: Optional[float] = None
    divergences: Optional[int] = None


class APIError(Exception):
    """Exception raised for API errors."""
    def __init__(self, status_code: int, message: str, detail: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(f"API Error {status_code}: {message}")


class MarketStructureApiClient:
    """
    Client for the Market Structure Analysis API.

    This is a synchronous client designed for Streamlit, which runs in a
    synchronous context. For async applications, use httpx.AsyncClient directly.
    """

    # Model type mapping from display names to API enum values
    MODEL_TYPE_MAP = {
        "Latent Class Analysis (LCA)": "lca",
        "Latent Class Analysis with Covariates": "lca_covariates",
        "Factor Analysis (Tetrachoric)": "factor_tetrachoric",
        "Bayesian Factor Model (VI)": "bayesian_factor_vi",
        "Bayesian Factor Model (PyMC)": "bayesian_factor_pymc",
        "Non-negative Matrix Factorization (NMF)": "nmf",
        "Multiple Correspondence Analysis (MCA)": "mca",
        "Discrete Choice Model (PyMC)": "dcm",
        "Latent Dirichlet Allocation (LDA)": "lda",
        "Network Analysis": "network",
    }

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server (without /api/v1 prefix)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1"
        self.timeout = timeout
        self._session = requests.Session()

    def _handle_response(self, response: requests.Response) -> dict:
        """Handle API response and raise appropriate errors."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("error", error_data.get("detail", "Unknown error"))
                detail = error_data.get("detail")
            except json.JSONDecodeError:
                message = response.text or "Unknown error"
                detail = None
            raise APIError(response.status_code, message, detail)
        return response.json()

    def check_health(self) -> dict:
        """
        Check API health status.

        Returns:
            Health status dict with redis_connected, database_connected, etc.

        Raises:
            APIError: If health check fails
        """
        response = self._session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        return self._handle_response(response)

    def is_available(self) -> bool:
        """
        Check if the API is available.

        Returns:
            True if API is healthy and responding, False otherwise
        """
        try:
            health = self.check_health()
            return health.get("status") == "healthy" or health.get("status") == "degraded"
        except Exception:
            return False

    def submit_run(
        self,
        model_type: str,
        data: pd.DataFrame,
        product_columns: list[str],
        params: dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        covariates: Optional[pd.DataFrame] = None,
        covariate_columns: Optional[list[str]] = None,
    ) -> str:
        """
        Submit a new model run to the API.

        Args:
            model_type: Model type display name (e.g., "Latent Class Analysis (LCA)")
            data: DataFrame containing the data
            product_columns: List of column names to use as products
            params: Model-specific parameters
            name: Optional name for this run
            description: Optional description
            covariates: Optional DataFrame of household covariates
            covariate_columns: Optional list of covariate column names

        Returns:
            The run ID string

        Raises:
            APIError: If submission fails
        """
        # Convert display name to API enum value
        api_model_type = self.MODEL_TYPE_MAP.get(model_type, model_type)

        # Encode data as base64 CSV
        csv_buffer = io.BytesIO()
        data.to_csv(csv_buffer, index=False)
        csv_base64 = base64.b64encode(csv_buffer.getvalue()).decode("utf-8")

        # Build data upload payload
        data_upload = {
            "csv_base64": csv_base64,
        }

        # Add covariate column names if using CSV (they'll be extracted from the DataFrame)
        if covariate_columns:
            data_upload["covariate_column_names"] = covariate_columns

        # Build request payload
        payload = {
            "model_type": api_model_type,
            "data": data_upload,
            "product_columns": product_columns,
            "params": params,
        }

        if name:
            payload["name"] = name
        if description:
            payload["description"] = description

        response = self._session.post(
            f"{self.api_url}/runs",
            json=payload,
            timeout=self.timeout
        )

        result = self._handle_response(response)
        return result["id"]

    def get_run_status(self, run_id: str) -> dict:
        """
        Get the current status of a model run.

        Args:
            run_id: The run ID

        Returns:
            Run status dict with status, progress, progress_message, etc.

        Raises:
            APIError: If run not found or request fails
        """
        response = self._session.get(
            f"{self.api_url}/runs/{run_id}",
            timeout=self.timeout
        )
        return self._handle_response(response)

    def get_results(self, run_id: str) -> dict:
        """
        Get full results for a completed model run.

        Args:
            run_id: The run ID

        Returns:
            Full results dict including embeddings, loadings, etc.

        Raises:
            APIError: If run not found, not completed, or request fails
        """
        response = self._session.get(
            f"{self.api_url}/runs/{run_id}/results",
            timeout=self.timeout * 2  # Results can be large
        )
        return self._handle_response(response)

    def stream_progress(
        self,
        run_id: str,
        poll_interval: float = 1.0,
        max_retries: int = 3,
    ) -> Generator[ProgressUpdate, None, None]:
        """
        Stream progress updates for a model run.

        This uses SSE (Server-Sent Events) for real-time updates.
        Falls back to polling if SSE fails.

        Args:
            run_id: The run ID
            poll_interval: Seconds between polls if SSE fails
            max_retries: Maximum SSE connection retries

        Yields:
            ProgressUpdate objects with current progress

        Raises:
            APIError: If run fails or is not found
        """
        retries = 0

        while retries <= max_retries:
            try:
                # Try SSE streaming first
                response = self._session.get(
                    f"{self.api_url}/progress/{run_id}/stream",
                    stream=True,
                    timeout=None,  # SSE streams indefinitely
                    headers={"Accept": "text/event-stream"}
                )

                if response.status_code == 200:
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data:"):
                            try:
                                data = json.loads(line[5:].strip())
                                update = ProgressUpdate(
                                    progress=data.get("progress", 0),
                                    message=data.get("message", ""),
                                    phase=data.get("phase", "running"),
                                    eta_seconds=data.get("eta_seconds"),
                                    chain=data.get("chain"),
                                    draw=data.get("draw"),
                                    total_draws=data.get("total_draws"),
                                    samples_per_second=data.get("samples_per_second"),
                                    divergences=data.get("divergences"),
                                )
                                yield update

                                # Check for completion
                                if update.phase in ["completed", "failed", "cancelled"]:
                                    return
                            except json.JSONDecodeError:
                                continue
                    return  # Stream ended normally

            except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError):
                retries += 1
                if retries > max_retries:
                    # Fall back to polling
                    break

        # Polling fallback
        while True:
            try:
                status = self.get_run_status(run_id)
                progress = status.get("progress", 0)
                message = status.get("progress_message", "")
                run_status = status.get("status", "running")

                # Map status to phase
                if run_status == "completed":
                    phase = "completed"
                elif run_status == "failed":
                    phase = "failed"
                elif run_status == "cancelled":
                    phase = "cancelled"
                else:
                    phase = "running"

                yield ProgressUpdate(
                    progress=progress,
                    message=message,
                    phase=phase,
                )

                if phase in ["completed", "failed", "cancelled"]:
                    return

                time.sleep(poll_interval)

            except APIError as e:
                yield ProgressUpdate(
                    progress=-1,
                    message=f"Error: {e.message}",
                    phase="failed",
                )
                return

    def cancel_run(self, run_id: str) -> dict:
        """
        Cancel a running or queued model run.

        Args:
            run_id: The run ID

        Returns:
            Updated run status

        Raises:
            APIError: If run cannot be cancelled
        """
        response = self._session.post(
            f"{self.api_url}/runs/{run_id}/cancel",
            timeout=self.timeout
        )
        return self._handle_response(response)

    def run_clustering(
        self,
        run_id: str,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        max_k: int = 10,
        linkage_method: str = "ward",
    ) -> dict:
        """
        Run clustering on a completed model run.

        Args:
            run_id: The run ID
            method: "kmeans" or "hierarchical"
            n_clusters: Number of clusters (None for auto-detect)
            max_k: Maximum k for auto-detection
            linkage_method: Linkage method for hierarchical clustering

        Returns:
            Clustering results with labels, silhouette_score, etc.

        Raises:
            APIError: If clustering fails
        """
        payload = {
            "method": method,
            "max_k": max_k,
            "linkage_method": linkage_method,
        }
        if n_clusters is not None:
            payload["n_clusters"] = n_clusters

        response = self._session.post(
            f"{self.api_url}/runs/{run_id}/clustering",
            json=payload,
            timeout=self.timeout * 2,  # Clustering can take time
        )
        return self._handle_response(response)

    def list_runs(
        self,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """
        List model runs with optional filtering.

        Args:
            status: Filter by status (e.g., "completed", "running")
            model_type: Filter by model type
            limit: Maximum number of runs to return
            offset: Offset for pagination

        Returns:
            Dict with runs list and pagination info

        Raises:
            APIError: If request fails
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if model_type:
            # Convert display name to API enum if needed
            params["model_type"] = self.MODEL_TYPE_MAP.get(model_type, model_type)

        response = self._session.get(
            f"{self.api_url}/runs",
            params=params,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def delete_run(self, run_id: str) -> None:
        """
        Delete a model run and its results.

        Args:
            run_id: The run ID

        Raises:
            APIError: If deletion fails
        """
        response = self._session.delete(
            f"{self.api_url}/runs/{run_id}",
            timeout=self.timeout
        )
        if response.status_code != 204:
            self._handle_response(response)

    def export_results(self, run_id: str) -> bytes:
        """
        Export model results as a ZIP file.

        Args:
            run_id: The run ID

        Returns:
            ZIP file contents as bytes

        Raises:
            APIError: If export fails
        """
        response = self._session.get(
            f"{self.api_url}/runs/{run_id}/export",
            timeout=self.timeout * 2,
        )
        if response.status_code >= 400:
            self._handle_response(response)
        return response.content

    def close(self):
        """Close the client session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
