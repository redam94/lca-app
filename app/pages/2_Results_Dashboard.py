"""
Results Dashboard - View and Analyze Completed Model Runs
=========================================================

A Streamlit page for browsing completed model runs, viewing detailed results,
and generating downloadable HTML reports with interactive Plotly visualizations.
"""

import streamlit as st
import requests
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import json

# API Configuration
API_BASE_URL = "http://localhost:8000"


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_completed_runs(limit: int = 50) -> Optional[Dict]:
    """Fetch completed model runs from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/runs",
            params={
                "status": "completed",
                "limit": limit,
                "order_by": "completed_at",
                "order_dir": "desc"
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching runs: {e}")
    return None


def get_run_details(run_id: str) -> Optional[Dict]:
    """Fetch detailed run information."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/runs/{run_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching run details: {e}")
    return None


def get_run_results(run_id: str) -> Optional[Dict]:
    """Fetch full results for a model run."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/runs/{run_id}/results", timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching results: {e}")
    return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_datetime(dt_str: str) -> str:
    """Format datetime string for display."""
    if not dt_str:
        return "-"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_model_display_name(model_type: str) -> str:
    """Get display name for model type."""
    names = {
        "lca": "Latent Class Analysis",
        "lca_covariates": "LCA with Covariates",
        "factor_tetrachoric": "Factor Analysis (Tetrachoric)",
        "bayesian_factor_vi": "Bayesian Factor (VI)",
        "bayesian_factor_pymc": "Bayesian Factor (PyMC)",
        "nmf": "Non-negative Matrix Factorization",
        "mca": "Multiple Correspondence Analysis",
        "dcm": "Discrete Choice Model",
        "lda": "Latent Dirichlet Allocation",
        "network": "Network Analysis",
    }
    return names.get(model_type, model_type)


def to_array(data) -> Optional[np.ndarray]:
    """Convert list to numpy array if not None."""
    if data is None:
        return None
    return np.array(data)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_figures_for_model(results: Dict, model_type: str) -> Dict[str, Any]:
    """Create all relevant Plotly figures for a model type."""
    import plotly.graph_objects as go
    from market_structure.plotting import (
        plot_correlation_matrix, plot_loadings_heatmap, plot_loadings_with_uncertainty,
        plot_variance_explained, plot_elbo_convergence, plot_lca_profiles,
        plot_biplot, plot_dcm_coefficients
    )

    figures = {}
    product_columns = results.get("product_columns", [])

    # Similarity/Correlation Matrix (most models)
    similarity = to_array(results.get("similarity_matrix"))
    if similarity is not None and len(product_columns) > 0:
        try:
            figures["Similarity Matrix"] = plot_correlation_matrix(
                similarity, product_columns, title="Product Similarity Matrix"
            )
        except Exception as e:
            st.warning(f"Could not create similarity matrix: {e}")

    # Variance Explained (factor-type models)
    var_explained = to_array(results.get("variance_explained"))
    if var_explained is not None:
        try:
            figures["Variance Explained"] = plot_variance_explained(
                var_explained, model_name=get_model_display_name(model_type)
            )
        except Exception as e:
            st.warning(f"Could not create variance plot: {e}")

    # Model-specific plots
    if model_type in ["lca", "lca_covariates"]:
        item_probs = to_array(results.get("item_probs"))
        class_probs = to_array(results.get("class_probs"))
        if item_probs is not None and class_probs is not None:
            try:
                figures["Class Profiles"] = plot_lca_profiles(
                    item_probs, class_probs, product_columns
                )
            except Exception as e:
                st.warning(f"Could not create LCA profiles: {e}")

    elif model_type in ["factor_tetrachoric", "bayesian_factor_vi", "nmf", "mca"]:
        loadings = to_array(results.get("loadings"))
        if loadings is not None and len(product_columns) > 0:
            try:
                n_factors = loadings.shape[1] if len(loadings.shape) > 1 else 1
                factor_names = [f"Factor {i+1}" for i in range(n_factors)]
                figures["Factor Loadings"] = plot_loadings_heatmap(
                    loadings, product_columns, factor_names
                )
            except Exception as e:
                st.warning(f"Could not create loadings heatmap: {e}")

        # ELBO for VI
        if model_type == "bayesian_factor_vi":
            elbo = results.get("elbo_history")
            if elbo is not None:
                try:
                    figures["ELBO Convergence"] = plot_elbo_convergence(elbo)
                except Exception as e:
                    st.warning(f"Could not create ELBO plot: {e}")

        # Tetrachoric correlation
        if model_type == "factor_tetrachoric":
            tetra = to_array(results.get("tetra_corr"))
            if tetra is not None:
                try:
                    figures["Tetrachoric Correlation"] = plot_correlation_matrix(
                        tetra, product_columns, title="Tetrachoric Correlation Matrix"
                    )
                except Exception as e:
                    st.warning(f"Could not create tetrachoric matrix: {e}")

    elif model_type == "bayesian_factor_pymc":
        loadings = to_array(results.get("loadings"))
        loadings_std = to_array(results.get("loadings_std"))
        if loadings is not None and len(product_columns) > 0:
            if loadings_std is not None:
                try:
                    figures["Factor Loadings (with uncertainty)"] = plot_loadings_with_uncertainty(
                        loadings, loadings_std, product_columns
                    )
                except Exception as e:
                    st.warning(f"Could not create uncertainty plot: {e}")
            else:
                try:
                    n_factors = loadings.shape[1] if len(loadings.shape) > 1 else 1
                    factor_names = [f"Factor {i+1}" for i in range(n_factors)]
                    figures["Factor Loadings"] = plot_loadings_heatmap(
                        loadings, product_columns, factor_names
                    )
                except Exception as e:
                    st.warning(f"Could not create loadings heatmap: {e}")

    elif model_type == "dcm":
        alpha = to_array(results.get("alpha"))
        alpha_std = to_array(results.get("alpha_std"))
        if alpha is not None and len(product_columns) > 0:
            try:
                figures["Product Intercepts"] = plot_dcm_coefficients(
                    alpha, alpha_std, product_columns
                )
            except Exception as e:
                st.warning(f"Could not create DCM coefficients: {e}")

    elif model_type == "lda":
        topic_dist = to_array(results.get("topic_product_dist"))
        if topic_dist is not None and len(product_columns) > 0:
            try:
                n_topics = topic_dist.shape[0]
                topic_names = [f"Topic {i+1}" for i in range(n_topics)]
                figures["Topic-Product Distribution"] = plot_loadings_heatmap(
                    topic_dist.T, product_columns, topic_names,
                    title="Topic-Product Distribution"
                )
            except Exception as e:
                st.warning(f"Could not create topic distribution: {e}")

    elif model_type == "network":
        adj_matrix = to_array(results.get("adjacency_matrix"))
        if adj_matrix is not None and len(product_columns) > 0:
            try:
                figures["Co-Purchase Network"] = plot_correlation_matrix(
                    adj_matrix, product_columns, title="Product Co-Purchase Network"
                )
            except Exception as e:
                st.warning(f"Could not create network matrix: {e}")

    # Biplot with dimension selector (if embeddings available)
    product_embeddings = to_array(results.get("product_embeddings"))
    if product_embeddings is not None and len(product_columns) > 0:
        if len(product_embeddings.shape) == 2 and product_embeddings.shape[1] >= 2:
            n_dims = product_embeddings.shape[1]

            # Create figure with dropdown for dimension selection
            fig = go.Figure()

            # Add all dimension combinations as separate traces
            dim_pairs = []
            for i in range(min(n_dims, 5)):
                for j in range(i + 1, min(n_dims, 5)):
                    dim_pairs.append((i, j))

            for idx, (dim_x, dim_y) in enumerate(dim_pairs):
                visible = idx == 0  # Only first pair visible by default
                fig.add_trace(go.Scatter(
                    x=product_embeddings[:, dim_x].tolist(),
                    y=product_embeddings[:, dim_y].tolist(),
                    mode="markers+text",
                    text=product_columns,
                    textposition="top center",
                    marker=dict(size=10, color="#667eea"),
                    name=f"Dim {dim_x+1} vs {dim_y+1}",
                    visible=visible
                ))

            # Create dropdown buttons
            buttons = []
            for idx, (dim_x, dim_y) in enumerate(dim_pairs):
                visibility = [i == idx for i in range(len(dim_pairs))]
                buttons.append(dict(
                    label=f"Dim {dim_x+1} vs Dim {dim_y+1}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"xaxis.title": f"Dimension {dim_x+1}",
                         "yaxis.title": f"Dimension {dim_y+1}"}
                    ]
                ))

            fig.update_layout(
                title=f"{get_model_display_name(model_type)} - Product Space (Biplot)",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                height=600,
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        direction="down",
                        showactive=True,
                        x=0.0,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    )
                ] if len(buttons) > 1 else []
            )
            figures["Biplot"] = fig

    return figures


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(run_details: Dict, results: Dict, figures: Dict[str, Any]) -> str:
    """Generate a standalone HTML report with embedded Plotly figures."""
    import plotly.io as pio

    model_type = run_details.get("model_type", "unknown")
    run_id = run_details.get("id", "unknown")
    run_name = run_details.get("name") or f"Run {run_id[:8]}"

    # Build HTML content
    html_parts = [
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{run_name} - Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{run_name}</h1>
        <div class="subtitle">{get_model_display_name(model_type)} Analysis Report</div>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
"""
    ]

    # Run metadata card
    html_parts.append("""
    <div class="card">
        <h2>Run Information</h2>
        <div class="metrics-grid">
""")

    # Add metrics
    metrics_data = [
        ("Run ID", run_id[:12] + "..."),
        ("Model Type", get_model_display_name(model_type)),
        ("Status", run_details.get("status", "unknown").upper()),
        ("Created", format_datetime(run_details.get("created_at"))),
        ("Completed", format_datetime(run_details.get("completed_at"))),
        ("Duration", format_duration(run_details.get("run_duration"))),
    ]

    for label, value in metrics_data:
        html_parts.append(f"""
            <div class="metric">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
""")

    html_parts.append("        </div>")

    # Model parameters
    params = run_details.get("model_params", {})
    if params:
        html_parts.append("""
        <h3>Model Parameters</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
""")
        for key, value in params.items():
            html_parts.append(f"            <tr><td>{key}</td><td>{value}</td></tr>\n")
        html_parts.append("        </table>")

    html_parts.append("    </div>")

    # Model metrics card
    metrics = run_details.get("metrics") or results.get("metrics")
    if metrics:
        html_parts.append("""
    <div class="card">
        <h2>Model Metrics</h2>
        <div class="metrics-grid">
""")
        for key, value in metrics.items():
            if isinstance(value, float):
                display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            html_parts.append(f"""
            <div class="metric">
                <div class="metric-value">{display_value}</div>
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
            </div>
""")
        html_parts.append("        </div>\n    </div>")

    # Add each figure
    for title, fig in figures.items():
        html_parts.append(f"""
    <div class="card">
        <h2>{title}</h2>
        <div class="plot-container">
            {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
        </div>
    </div>
""")

    # Product columns summary
    product_columns = results.get("product_columns", [])
    if product_columns:
        html_parts.append(f"""
    <div class="card">
        <h2>Products Analyzed ({len(product_columns)})</h2>
        <p>{', '.join(product_columns)}</p>
    </div>
""")

    # Footer
    html_parts.append("""
    <div class="footer">
        <p>Generated by Market Structure Analysis Dashboard</p>
        <p>Powered by Plotly for interactive visualizations</p>
    </div>
</body>
</html>
""")

    return "".join(html_parts)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Results Dashboard",
        page_icon="",
        layout="wide"
    )

    st.title("Results Dashboard")
    st.markdown("Browse completed model runs and generate detailed reports.")

    # Initialize session state
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None

    # Sidebar for run selection
    with st.sidebar:
        st.header("Completed Runs")

        if st.button("Refresh List"):
            st.rerun()

        runs_response = get_completed_runs(limit=50)

        if runs_response and runs_response.get("runs"):
            runs = runs_response["runs"]

            # Group by model type
            by_model = {}
            for run in runs:
                model_type = run.get("model_type", "unknown")
                if model_type not in by_model:
                    by_model[model_type] = []
                by_model[model_type].append(run)

            for model_type, model_runs in by_model.items():
                with st.expander(f"{get_model_display_name(model_type)} ({len(model_runs)})", expanded=True):
                    for run in model_runs:
                        run_id = run.get("id")
                        name = run.get("name") or f"Run {run_id[:8]}"
                        completed = format_datetime(run.get("completed_at"))

                        if st.button(
                            f"{name}\n{completed}",
                            key=f"run_{run_id}",
                            use_container_width=True
                        ):
                            st.session_state.selected_run_id = run_id
                            st.rerun()
        else:
            st.info("No completed runs found.")

    # Main content area
    if st.session_state.selected_run_id:
        run_id = st.session_state.selected_run_id

        # Fetch run details and results
        with st.spinner("Loading run details..."):
            run_details = get_run_details(run_id)

        if run_details is None:
            st.error(f"Could not load run {run_id}")
            return

        # Header with run info
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            run_name = run_details.get("name") or f"Run {run_id[:8]}"
            st.header(run_name)
            st.caption(f"ID: {run_id}")

        with col2:
            model_type = run_details.get("model_type", "unknown")
            st.metric("Model", get_model_display_name(model_type))

        with col3:
            duration = run_details.get("run_duration")
            st.metric("Duration", format_duration(duration))

        st.divider()

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "Export Report"])

        with tab1:
            # Run metadata
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Run Details")
                st.write(f"**Status:** {run_details.get('status', 'unknown').upper()}")
                st.write(f"**Created:** {format_datetime(run_details.get('created_at'))}")
                st.write(f"**Started:** {format_datetime(run_details.get('started_at'))}")
                st.write(f"**Completed:** {format_datetime(run_details.get('completed_at'))}")

                if run_details.get("description"):
                    st.write(f"**Description:** {run_details['description']}")

            with col2:
                st.subheader("Model Parameters")
                params = run_details.get("model_params", {})
                for key, value in params.items():
                    st.write(f"**{key}:** {value}")

            # Metrics
            st.subheader("Model Metrics")
            metrics = run_details.get("metrics", {})
            if metrics:
                cols = st.columns(min(len(metrics), 4))
                for i, (key, value) in enumerate(metrics.items()):
                    with cols[i % len(cols)]:
                        if isinstance(value, float):
                            st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                        else:
                            st.metric(key.replace("_", " ").title(), value)
            else:
                st.info("No metrics available.")

            # Data shape
            data_shape = run_details.get("data_shape", {})
            if data_shape:
                st.subheader("Data Information")
                st.write(f"**Observations:** {data_shape.get('n_obs', 'N/A')}")
                st.write(f"**Products:** {data_shape.get('n_items', 'N/A')}")

        with tab2:
            # Load full results for visualizations
            with st.spinner("Loading results and generating visualizations..."):
                results = get_run_results(run_id)

            if results is None:
                st.error("Could not load results for visualization.")
            else:
                # Generate figures
                figures = create_figures_for_model(results, model_type)

                if figures:
                    # Display each figure
                    for title, fig in figures.items():
                        st.subheader(title)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No visualizations available for this model type.")

                # Product columns
                product_columns = results.get("product_columns", [])
                if product_columns:
                    with st.expander(f"Products Analyzed ({len(product_columns)})"):
                        st.write(", ".join(product_columns))

        with tab3:
            st.subheader("Generate HTML Report")
            st.markdown("""
            Generate a standalone HTML report with all visualizations embedded as interactive Plotly charts.
            The report can be opened in any web browser without requiring an internet connection.
            """)

            # Clustering options
            st.subheader("Clustering Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                include_clustering = st.checkbox("Include Clustering Analysis", value=True)

            with col2:
                clustering_method = st.selectbox(
                    "Clustering Method",
                    options=["kmeans", "hierarchical"],
                    index=0,
                    disabled=not include_clustering
                )

            with col3:
                auto_detect = st.checkbox("Auto-detect optimal k", value=True, disabled=not include_clustering)

            col1, col2 = st.columns(2)
            with col1:
                if auto_detect or not include_clustering:
                    n_clusters = None
                    max_k = st.slider("Max clusters to consider", 2, 15, 10, disabled=not include_clustering)
                else:
                    n_clusters = st.slider("Number of clusters", 2, 15, 3)
                    max_k = 10

            st.divider()

            # Generate report using API endpoint
            if st.button("Generate Report", type="primary"):
                with st.spinner("Generating HTML report with clustering..."):
                    try:
                        # Build API URL with query parameters
                        params = {
                            "include_clustering": str(include_clustering).lower(),
                            "clustering_method": clustering_method,
                            "max_k": max_k,
                        }
                        if n_clusters is not None:
                            params["n_clusters"] = n_clusters

                        response = requests.get(
                            f"{API_BASE_URL}/api/v1/runs/{run_id}/report",
                            params=params,
                            timeout=60
                        )

                        if response.status_code == 200:
                            html_content = response.text

                            # Provide download
                            run_name = run_details.get("name") or f"run_{run_id[:8]}"
                            filename = f"{run_name.replace(' ', '_')}_{model_type}_report.html"

                            st.download_button(
                                label="Download HTML Report",
                                data=html_content,
                                file_name=filename,
                                mime="text/html",
                                type="primary"
                            )

                            st.success("Report generated! Click above to download.")

                            # Preview
                            with st.expander("Preview Report"):
                                st.components.v1.html(html_content, height=800, scrolling=True)
                        else:
                            st.error(f"Failed to generate report: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error generating report: {e}")

    else:
        # No run selected
        st.info("Select a completed run from the sidebar to view details and generate reports.")

        # Show summary statistics
        runs_response = get_completed_runs(limit=100)
        if runs_response and runs_response.get("runs"):
            runs = runs_response["runs"]

            st.subheader("Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Completed Runs", len(runs))

            with col2:
                model_types = set(r.get("model_type") for r in runs)
                st.metric("Model Types Used", len(model_types))

            with col3:
                total_duration = sum(r.get("run_duration") or 0 for r in runs)
                st.metric("Total Compute Time", format_duration(total_duration))

            # Model type breakdown
            st.subheader("Runs by Model Type")
            by_model = {}
            for run in runs:
                model_type = run.get("model_type", "unknown")
                by_model[model_type] = by_model.get(model_type, 0) + 1

            cols = st.columns(min(len(by_model), 5))
            for i, (model_type, count) in enumerate(sorted(by_model.items(), key=lambda x: -x[1])):
                with cols[i % len(cols)]:
                    st.metric(get_model_display_name(model_type), count)


if __name__ == "__main__":
    main()
