"""
Latent Structure Analysis App for Purchase Data
================================================

A Streamlit application for analyzing binary purchase data using various
latent structure models. This app provides an interactive interface for:
- Uploading and previewing purchase data
- Selecting and configuring analysis models
- Visualizing results with interactive plots
- Clustering products based on latent structure
- Exporting results for further analysis

The heavy lifting (model fitting, progress tracking) is handled by the
FastAPI backend, while this frontend handles UI and visualization using
the market_structure.plotting package.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Import API client and results adapter
from api_client import MarketStructureApiClient, APIError, ProgressUpdate
from results_adapter import (
    adapt_results_for_plotting,
    adapt_clustering_results,
    extract_model_metrics,
    get_convergence_message,
)

# Import from market_structure package for config, plotting, and utilities
from market_structure.config import (
    PYMC_AVAILABLE, PYMC_ERROR, PRINCE_AVAILABLE,
    get_available_models, get_model_help_text
)
from market_structure.models import (
    compute_lca_coordinates, compute_residual_correlations,
    interpret_covariate_effects, compute_factor_scores_regression,
)
from market_structure.plotting import (
    plot_correlation_matrix, plot_loadings_heatmap, plot_loadings_with_uncertainty,
    plot_variance_explained, plot_elbo_convergence, plot_silhouette_scores,
    plot_lca_profiles, plot_biplot, plot_dcm_coefficients, plot_dendrogram
)
from market_structure.utils import (
    get_model_cache_key, get_cluster_members,
)


# =============================================================================
# API CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables for model caching and visualization."""
    defaults = {
        'model_result': None,
        'model_cache_key': None,
        'model_type_cached': None,
        'product_columns_cached': None,
        'similarity_matrix_cached': None,
        'product_embeddings': None,
        'household_embeddings': None,
        'var_explained_cached': None,
        'cluster_result': None,
        'tetra_corr_cached': None,
        'elbo_history_cached': None,
        'convergence_msg': None,
        'model_metrics': None,
        'original_data': None,
        # API-related state
        'current_run_id': None,
        'api_client': None,
        'api_available': None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_api_client() -> MarketStructureApiClient:
    """Get or create the API client."""
    if st.session_state.api_client is None:
        st.session_state.api_client = MarketStructureApiClient(base_url=API_BASE_URL)
    return st.session_state.api_client


def check_api_availability() -> bool:
    """Check if the API is available and cache the result."""
    if st.session_state.api_available is None:
        client = get_api_client()
        st.session_state.api_available = client.is_available()
    return st.session_state.api_available


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="Latent Structure Analysis", layout="wide")

    st.title("Latent Structure Analysis for Purchase Data")
    st.markdown("""
    Discover latent customer segments and product relationships using multiple statistical methods.
    """)

    # Initialize session state
    initialize_session_state()

    # Check API availability
    if not check_api_availability():
        st.error("""
        **Backend API is not available.**

        Please ensure the FastAPI backend is running:
        ```bash
        cd api && uv run backend-api
        ```

        Also ensure Redis is running for progress tracking:
        ```bash
        redis-server
        ```
        """)
        # Add a refresh button
        if st.button("Retry Connection"):
            st.session_state.api_available = None
            st.rerun()
        return

    # Display dependency status (these affect what models are available on the backend)
    if not PYMC_AVAILABLE:
        if PYMC_ERROR:
            st.warning(f"PyMC import error: {PYMC_ERROR}. PyMC-based models will be unavailable.")
        else:
            st.warning("PyMC not installed. Install with: `pip install pymc arviz`")

    if not PRINCE_AVAILABLE:
        st.info("Install `prince` for MCA: `pip install prince`")
    
    # ==========================================================================
    # SIDEBAR: Data Upload and Model Selection
    # ==========================================================================
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        else:
            st.info("Please upload a CSV file to begin")
            _show_example_format()
            return
        
        st.markdown("---")
        st.header("🔧 Model Selection")
        
        model_type = st.selectbox(
            "Select Analysis Method",
            options=get_available_models(),
            help=get_model_help_text()
        )
    
    # ==========================================================================
    # MAIN CONTENT: Data Preview and Configuration
    # ==========================================================================
    if uploaded_file is not None:
        st.header("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column configuration
        st.header("⚙️ Configure Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            id_column = st.selectbox(
                "Select Household ID column",
                options=['(None)'] + list(df.columns)
            )
        
        with col2:
            available_cols = [c for c in df.columns if c != id_column]
            product_columns = st.multiselect(
                "Select Product columns (binary 0/1)",
                options=available_cols,
                default=[c for c in available_cols if df[c].dropna().isin([0, 1, 0.0, 1.0]).all()]
            )
        
        # DCM-specific: household features
        household_feature_columns = []
        if model_type in ["Discrete Choice Model (PyMC)", "Latent Class Analysis (LCA)"]:
            st.markdown("---")
            st.subheader("🏠 Household Features (optional)")
            remaining_cols = [c for c in available_cols if c not in product_columns]
            household_feature_columns = st.multiselect(
                "Select Household Feature columns",
                options=remaining_cols,
                help="Numeric features describing households (demographics, income, etc.). "
                     "For LCA, these can predict class membership."
            )
        
        if len(product_columns) < 2:
            st.warning("Please select at least 2 product columns to run analysis.")
            return
        
        # Prepare data
        data_subset = df[product_columns].copy()
        if data_subset.isnull().any().any():
            st.warning("Data contains missing values. Rows with missing values will be excluded.")
            data_subset = data_subset.dropna()
        
        X = data_subset.values.astype(float)
        st.session_state.original_data = X
        data_hash = str(hash(X.tobytes()))
        
        st.success(f"Ready to analyze {X.shape[0]} households across {X.shape[1]} products")
        
        # ======================================================================
        # MODEL CONFIGURATION
        # ======================================================================
        st.header("🔬 Model Configuration")
        model_params = _configure_model(model_type, product_columns)
        
        # Check cache validity
        current_cache_key = get_model_cache_key(data_hash, model_type, model_params, tuple(product_columns))
        if st.session_state.model_cache_key != current_cache_key:
            _invalidate_cache()
        
        # ======================================================================
        # RUN ANALYSIS
        # ======================================================================
        if st.button("🚀 Run Analysis", type="primary"):
            _run_analysis(X, model_type, model_params, product_columns, 
                         df, household_feature_columns, current_cache_key)
        
        # ======================================================================
        # VISUALIZATION (outside button block for persistence)
        # ======================================================================
        if st.session_state.model_result is not None and st.session_state.model_type_cached == model_type:
            _render_visualizations(model_type, product_columns)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _show_example_format():
    """Display example data format."""
    example_df = pd.DataFrame({
        'household_id': [1, 2, 3],
        'income': [50000, 75000, 60000],
        'hh_size': [2, 4, 3],
        'windex': [1, 0, 1],
        'lysol': [1, 1, 0],
        'clorox': [0, 1, 1]
    })
    st.markdown("### Expected Format")
    st.dataframe(example_df)


def _configure_model(model_type: str, product_columns: list) -> dict:
    """Configure model-specific parameters and return them as a dict."""
    params = {}
    
    if model_type == "Latent Class Analysis (LCA)":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_classes'] = st.slider("Number of Classes", 2, 10, 3)
        with col2:
            params['n_init'] = st.slider("Number of Initializations", 1, 50, 10)
        with col3:
            params['max_iter'] = st.slider("Max Iterations", 50, 500, 100)
        
        # Covariate option (only shown if household features were selected)
        params['use_covariates'] = False
        params['covariate_columns'] = []
    
    elif model_type in ["Factor Analysis (Tetrachoric)", "Bayesian Factor Model (VI)"]:
        col1, col2 = st.columns(2)
        with col1:
            params['n_factors'] = st.slider("Number of Factors", 1, min(10, len(product_columns) - 1), 2)
        with col2:
            params['max_iter'] = st.slider("Max Iterations", 50, 500, 100)
    
    elif model_type == "Bayesian Factor Model (PyMC)":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_factors'] = st.slider("Number of Factors", 1, min(10, len(product_columns) - 1), 2)
        with col2:
            params['n_samples'] = st.slider("MCMC Samples", 500, 3000, 1000)
        with col3:
            params['n_tune'] = st.slider("Tuning Samples", 200, 1000, 500)
    
    elif model_type == "Non-negative Matrix Factorization (NMF)":
        col1, col2 = st.columns(2)
        with col1:
            params['n_components'] = st.slider("Number of Components", 2, min(10, len(product_columns)), 3)
        with col2:
            params['max_iter'] = st.slider("Max Iterations", 100, 1000, 200)
    
    elif model_type == "Multiple Correspondence Analysis (MCA)":
        params['n_components'] = st.slider("Number of Components", 2, min(10, len(product_columns)), 3)
    
    elif model_type == "Discrete Choice Model (PyMC)":
        col1, col2 = st.columns(2)
        with col1:
            params['n_samples'] = st.slider("MCMC Samples", 500, 3000, 1000)
        with col2:
            params['n_tune'] = st.slider("Tuning Samples", 200, 1000, 500)

        params['include_random_effects'] = st.checkbox("Include Household Random Effects", value=False)

        include_latent = st.checkbox("Include Latent Product Features", value=True,
                                     help="Learn latent product-household interactions")
        if include_latent:
            col1, col2 = st.columns(2)
            with col1:
                params['n_latent_features'] = st.slider("Number of Latent Dimensions", 1, 5, 2)
            with col2:
                params['latent_prior_scale'] = st.slider("Regularization Scale", 0.1, 2.0, 1.0)
        else:
            params['n_latent_features'] = 0
            params['latent_prior_scale'] = 1.0

    elif model_type == "Latent Dirichlet Allocation (LDA)":
        col1, col2 = st.columns(2)
        with col1:
            params['n_topics'] = st.slider("Number of Topics", 2, 20, 5,
                                          help="Number of latent topics to discover")
        with col2:
            params['max_iter'] = st.slider("Max Iterations", 50, 300, 100)

        params['learning_method'] = st.selectbox(
            "Learning Method",
            ["online", "batch"],
            help="'online' is faster, 'batch' is more accurate for small datasets"
        )

    elif model_type == "Network Analysis":
        col1, col2 = st.columns(2)
        with col1:
            params['threshold'] = st.slider("Edge Threshold", 0.0, 0.5, 0.1, 0.05,
                                           help="Minimum co-purchase strength to include as edge")
        with col2:
            params['community_method'] = st.selectbox(
                "Community Detection",
                ["louvain", "label_propagation", "greedy_modularity"],
                help="Algorithm for finding product communities"
            )

        params['edge_method'] = st.selectbox(
            "Edge Weight Method",
            ["lift", "correlation", "cosine", "jaccard"],
            help="How to measure co-purchase strength between products"
        )

    return params


def _invalidate_cache():
    """Clear all cached model results."""
    keys = ['model_result', 'model_cache_key', 'model_type_cached',
            'product_columns_cached', 'similarity_matrix_cached',
            'product_embeddings', 'household_embeddings', 'var_explained_cached',
            'cluster_result', 'tetra_corr_cached', 'elbo_history_cached',
            'convergence_msg', 'model_metrics', 'current_run_id']
    for key in keys:
        st.session_state[key] = None


def _run_analysis(X, model_type, params, product_columns, df,
                  household_feature_columns, cache_key):
    """
    Submit model run to API and track progress.

    This function:
    1. Submits the job to the FastAPI backend
    2. Streams progress updates and displays them
    3. Fetches results when complete
    4. Stores results in session state for visualization
    """
    client = get_api_client()

    # Determine if we're using covariates
    use_covariates = len(household_feature_columns) > 0 and model_type in [
        "Latent Class Analysis (LCA)",
        "Discrete Choice Model (PyMC)"
    ]

    # Adjust model type for covariates
    api_model_type = model_type
    if model_type == "Latent Class Analysis (LCA)" and use_covariates:
        api_model_type = "Latent Class Analysis with Covariates"

    # Prepare covariate columns if needed
    covariate_columns = household_feature_columns if use_covariates else None

    if use_covariates:
        st.info(f"Using {len(household_feature_columns)} household covariates")

    # Create progress UI elements
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0, text="Submitting job...")
        status_text = st.empty()
        cancel_button_placeholder = st.empty()

    try:
        # Submit the job
        run_id = client.submit_run(
            model_type=api_model_type,
            data=df,
            product_columns=product_columns,
            params=params,
            name=f"{model_type} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            covariate_columns=covariate_columns,
        )

        st.session_state.current_run_id = run_id

        # Show cancel button
        cancelled = False
        with cancel_button_placeholder:
            if st.button("Cancel", key=f"cancel_{run_id}"):
                try:
                    client.cancel_run(run_id)
                    cancelled = True
                except APIError:
                    pass

        if cancelled:
            st.warning("Run cancelled.")
            return

        # Stream progress updates
        for update in client.stream_progress(run_id):
            if update.phase == "failed":
                progress_bar.progress(0, text="Failed")
                st.error(f"Model failed: {update.message}")
                return

            if update.phase == "cancelled":
                progress_bar.progress(0, text="Cancelled")
                st.warning("Run was cancelled.")
                return

            # Update progress bar
            progress_pct = max(0, min(1, update.progress))
            progress_bar.progress(progress_pct, text=update.message)

            # Show additional info for MCMC models
            if update.chain is not None:
                status_text.text(
                    f"Chain {update.chain + 1} | "
                    f"Draw {update.draw or 0}/{update.total_draws or '?'} | "
                    f"{update.samples_per_second or 0:.1f} samples/sec"
                )

            if update.phase == "completed":
                break

        # Clear cancel button
        cancel_button_placeholder.empty()

        # Fetch results
        progress_bar.progress(1.0, text="Fetching results...")
        api_results = client.get_results(run_id)

        # Adapt results for plotting
        result = adapt_results_for_plotting(api_results, api_model_type)

        # Store results in session state
        _store_results_from_api(result, cache_key, model_type, product_columns, api_results)

        progress_bar.progress(1.0, text="Complete!")
        st.success(st.session_state.convergence_msg)

    except APIError as e:
        st.error(f"API Error: {e.message}")
        if e.detail:
            st.error(f"Details: {e.detail}")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def _store_results_from_api(result: dict, cache_key: str, model_type: str,
                            product_columns: list, api_results: dict):
    """Store API results in session state for visualization."""
    st.session_state.model_result = result
    st.session_state.model_cache_key = cache_key
    st.session_state.model_type_cached = model_type
    st.session_state.product_columns_cached = result.get('product_columns', product_columns)
    st.session_state.similarity_matrix_cached = result.get('similarity_matrix')
    st.session_state.product_embeddings = result.get('product_embeddings')
    st.session_state.household_embeddings = result.get('household_embeddings')
    st.session_state.var_explained_cached = result.get('variance_explained')
    st.session_state.cluster_result = None

    # Model-specific cached fields
    st.session_state.tetra_corr_cached = result.get('tetra_corr')
    st.session_state.elbo_history_cached = result.get('elbo_history')

    # Convergence message and metrics
    st.session_state.convergence_msg = get_convergence_message(api_results, model_type)
    st.session_state.model_metrics = extract_model_metrics(api_results)


def _render_visualizations(model_type, product_columns):
    """Render all visualizations for the cached model results."""
    model_result = st.session_state.model_result
    product_columns_cached = st.session_state.product_columns_cached
    product_embeddings = st.session_state.product_embeddings
    household_embeddings = st.session_state.household_embeddings
    var_explained = st.session_state.var_explained_cached
    similarity_matrix = st.session_state.similarity_matrix_cached
    
    st.markdown("---")
    st.header("📊 Model Results")
    
    # Convergence message and metrics
    if st.session_state.convergence_msg:
        st.info(st.session_state.convergence_msg)
    
    if st.session_state.model_metrics:
        cols = st.columns(len(st.session_state.model_metrics))
        for i, (name, value) in enumerate(st.session_state.model_metrics.items()):
            with cols[i]:
                if value is not None:
                    if isinstance(value, float):
                        st.metric(name, f"{value:.2f}")
                    else:
                        st.metric(name, value)
    
    # Model-specific visualizations
    if model_type == "Latent Class Analysis (LCA)":
        _render_lca_viz(model_result, product_columns_cached, similarity_matrix)
    
    elif model_type == "Factor Analysis (Tetrachoric)":
        _render_factor_viz(model_result, product_columns_cached, 
                          st.session_state.tetra_corr_cached)
    
    elif model_type == "Bayesian Factor Model (VI)":
        _render_vi_viz(model_result, product_columns_cached)
    
    elif model_type == "Bayesian Factor Model (PyMC)":
        _render_pymc_viz(model_result, product_columns_cached)
    
    elif model_type == "Non-negative Matrix Factorization (NMF)":
        _render_nmf_viz(model_result, product_columns_cached)
    
    elif model_type == "Multiple Correspondence Analysis (MCA)":
        _render_mca_viz(model_result, product_columns_cached)
    
    elif model_type == "Discrete Choice Model (PyMC)":
        _render_dcm_viz(model_result, product_columns_cached)

    elif model_type == "Latent Dirichlet Allocation (LDA)":
        _render_lda_viz(model_result, product_columns_cached)

    elif model_type == "Network Analysis":
        _render_network_viz(model_result, product_columns_cached)

    # Generic visualizations (biplot, clustering)
    if product_embeddings is not None:
        _render_biplot_section(product_embeddings, product_columns_cached,
                              household_embeddings, var_explained)
        _render_clustering_section(product_embeddings, similarity_matrix, product_columns_cached)
    
    # Export section
    _render_export_section(model_result, model_type, product_columns_cached,
                          product_embeddings, household_embeddings, 
                          var_explained, similarity_matrix)


def _render_lca_viz(result, product_columns, similarity):
    """LCA-specific visualizations."""
    st.subheader("Class Profiles")

    item_probs = result.get('item_probs')
    class_probs = result.get('class_probs')
    class_probs_per_hh = result.get('class_probs_per_hh')

    if item_probs is None:
        st.warning("Item probabilities not available for visualization.")
        return

    # Handle both standard LCA and LCA with covariates
    if class_probs is not None:
        fig = plot_lca_profiles(item_probs, class_probs, product_columns)
    elif class_probs_per_hh is not None:
        # For covariate model, use mean class probabilities
        mean_probs = class_probs_per_hh.mean(axis=0)
        fig = plot_lca_profiles(item_probs, mean_probs, product_columns)
    else:
        st.warning("Class probabilities not available.")
        return

    st.plotly_chart(fig, use_container_width=True)

    # Display covariate effects if available (LCA with covariates)
    beta = result.get('beta')
    feature_names = result.get('feature_names')

    if beta is not None and feature_names:
        st.subheader("Covariate Effects on Class Membership")
        st.markdown("""
        These coefficients show how household characteristics influence the probability
        of belonging to each class. Positive values mean the feature increases odds of
        membership; negative values decrease odds. The last class serves as the reference.
        """)

        # Infer n_classes from beta shape
        n_classes = beta.shape[1] if len(beta.shape) > 1 else len(beta)

        # Get interpretation
        try:
            effects = interpret_covariate_effects(
                beta,
                feature_names,
                [f"Class {i+1}" for i in range(n_classes)]
            )

            # Display coefficients table
            st.markdown("**Regression Coefficients (log-odds):**")
            coef_display = effects['coefficients'].round(3)
            st.dataframe(coef_display, use_container_width=True)

            # Display odds ratios (more interpretable)
            st.markdown("**Odds Ratios (exp(coef)):**")
            st.markdown("_An odds ratio of 2.0 means a 1-SD increase in the feature doubles the odds of that class._")
            odds_display = effects['odds_ratios'].round(3)
            st.dataframe(odds_display, use_container_width=True)

            # Summary of key effects
            if effects.get('summary'):
                st.markdown("**Key Findings:**")
                st.info(effects['summary'])
        except Exception as e:
            st.warning(f"Could not compute covariate interpretation: {e}")

        # Show distribution of class probabilities across households
        if class_probs_per_hh is not None:
            st.subheader("Class Probability Distribution")
            st.markdown("Shows how class membership probabilities vary across households due to covariates.")

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            n_classes = class_probs_per_hh.shape[1]
            fig = make_subplots(rows=1, cols=n_classes,
                               subplot_titles=[f"Class {i+1}" for i in range(n_classes)])

            for c in range(n_classes):
                fig.add_trace(
                    go.Histogram(x=class_probs_per_hh[:, c], nbinsx=30,
                                name=f"Class {c+1}", showlegend=False,
                                marker_color='steelblue'),
                    row=1, col=c+1
                )
                fig.update_xaxes(title_text="P(class)", row=1, col=c+1)
                fig.update_yaxes(title_text="Count" if c == 0 else "", row=1, col=c+1)

            fig.update_layout(height=300, title="Distribution of Class Probabilities Across Households")
            st.plotly_chart(fig, use_container_width=True)

    # Residual correlations
    if similarity is not None:
        st.subheader("Residual Correlations (Substitution Patterns)")
        fig = plot_correlation_matrix(similarity, product_columns, "Residual Correlations")
        st.plotly_chart(fig, use_container_width=True)


def _render_factor_viz(result, product_columns, tetra_corr):
    """Tetrachoric FA visualizations."""
    if tetra_corr is not None:
        st.subheader("Tetrachoric Correlation Matrix")
        fig = plot_correlation_matrix(tetra_corr, product_columns, "Tetrachoric Correlations")
        st.plotly_chart(fig, use_container_width=True)

    loadings = result.get('loadings')
    if loadings is not None:
        st.subheader("Factor Loadings")
        fig = plot_loadings_heatmap(loadings, product_columns)
        st.plotly_chart(fig, use_container_width=True)

    var_explained = result.get('var_explained_pct')
    if var_explained is None:
        var_explained = result.get('variance_explained')
    if var_explained is not None:
        st.subheader("Variance Explained")
        fig = plot_variance_explained(var_explained, "Factor Analysis")
        st.plotly_chart(fig, use_container_width=True)


def _render_vi_viz(result, product_columns):
    """Bayesian VI visualizations."""
    loadings = result.get('loadings')
    if loadings is not None:
        st.subheader("Factor Loadings")
        fig = plot_loadings_heatmap(loadings, product_columns)
        st.plotly_chart(fig, use_container_width=True)

    var_explained = result.get('var_explained_pct') or result.get('variance_explained')
    if var_explained is not None:
        st.subheader("Variance Explained")
        fig = plot_variance_explained(var_explained, "Bayesian FA (VI)")
        st.plotly_chart(fig, use_container_width=True)

    elbo_history = st.session_state.elbo_history_cached or result.get('elbo_history')
    if elbo_history:
        st.subheader("ELBO Convergence")
        fig = plot_elbo_convergence(elbo_history)
        st.plotly_chart(fig, use_container_width=True)


def _render_pymc_viz(result, product_columns):
    """Bayesian PyMC visualizations."""
    loadings = result.get('loadings')
    loadings_std = result.get('loadings_std')

    if loadings is not None:
        st.subheader("Factor Loadings with Uncertainty")
        if loadings_std is not None:
            fig = plot_loadings_with_uncertainty(loadings, loadings_std, product_columns)
        else:
            fig = plot_loadings_heatmap(loadings, product_columns)
        st.plotly_chart(fig, use_container_width=True)

    var_explained = result.get('var_explained_pct') or result.get('variance_explained')
    if var_explained is not None:
        st.subheader("Variance Explained")
        fig = plot_variance_explained(var_explained, "Bayesian FA (PyMC)")
        st.plotly_chart(fig, use_container_width=True)


def _render_nmf_viz(result, product_columns):
    """NMF visualizations."""
    loadings = result.get('loadings')
    if loadings is not None:
        st.subheader("Component Loadings")
        fig = plot_loadings_heatmap(loadings, product_columns, title="NMF Component Loadings")
        st.plotly_chart(fig, use_container_width=True)

    var_explained = result.get('var_explained_pct') or result.get('variance_explained')
    if var_explained is not None:
        st.subheader("Variance Explained")
        fig = plot_variance_explained(var_explained, "NMF")
        st.plotly_chart(fig, use_container_width=True)


def _render_mca_viz(result, product_labels):
    """MCA visualizations."""
    var_explained = result.get('var_explained_pct')
    if var_explained is None:
        var_explained = result.get('variance_explained')
    if var_explained is not None:
        st.subheader("Variance Explained")
        fig = plot_variance_explained(var_explained, "MCA")
        st.plotly_chart(fig, use_container_width=True)

    similarity = result.get('similarity_matrix')
    if similarity is not None:
        st.subheader("Product Similarity")
        fig = plot_correlation_matrix(similarity, product_labels, "MCA-Based Similarity")
        st.plotly_chart(fig, use_container_width=True)


def _render_dcm_viz(result, product_columns):
    """DCM visualizations."""
    alpha = result.get('alpha')
    alpha_std = result.get('alpha_std')

    if alpha is not None:
        st.subheader("Product Intercepts")
        fig = plot_dcm_coefficients(alpha, alpha_std, product_columns)
        st.plotly_chart(fig, use_container_width=True)

    product_latent = result.get('product_latent')
    n_latent = result.get('n_latent_features', 0)
    if product_latent is not None and n_latent > 0:
        st.subheader("Latent Product Features")
        fig = plot_loadings_heatmap(
            product_latent,
            product_columns,
            factor_names=[f"Latent {i+1}" for i in range(n_latent)],
            title="Latent Product Features"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif product_latent is not None:
        # Infer n_latent from shape
        n_latent = product_latent.shape[1] if len(product_latent.shape) > 1 else 1
        st.subheader("Latent Product Features")
        fig = plot_loadings_heatmap(
            product_latent,
            product_columns,
            factor_names=[f"Latent {i+1}" for i in range(n_latent)],
            title="Latent Product Features"
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_lda_viz(result, product_columns):
    """LDA visualizations."""
    import plotly.express as px
    import plotly.graph_objects as go

    topic_product_dist = result.get('topic_product_dist')
    n_topics = result.get('n_topics', 0)

    if topic_product_dist is not None:
        st.subheader("Topic-Product Distributions")
        # Show as heatmap (similar to loadings)
        fig = plot_loadings_heatmap(
            topic_product_dist.T,  # Transpose to (n_products, n_topics)
            product_columns,
            factor_names=[f"Topic {i+1}" for i in range(n_topics)],
            title="Product Probability per Topic"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show perplexity metric
    perplexity = result.get('perplexity')
    if perplexity is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Perplexity", f"{perplexity:.2f}",
                     help="Lower perplexity indicates better model fit")
        log_likelihood = result.get('log_likelihood')
        if log_likelihood is not None:
            with col2:
                st.metric("Log-Likelihood", f"{log_likelihood:.2f}")

    # Variance explained
    var_explained = result.get('var_explained_pct') or result.get('variance_explained')
    if var_explained is not None:
        st.subheader("Topic Importance")
        fig = plot_variance_explained(var_explained, "LDA")
        st.plotly_chart(fig, use_container_width=True)

    # Top products per topic
    if topic_product_dist is not None:
        st.subheader("Top Products per Topic")
        n_top = st.slider("Number of top products", 3, 10, 5, key="lda_top_products")

        # Create columns for topics
        cols = st.columns(min(n_topics, 4))
        for topic_idx in range(n_topics):
            col_idx = topic_idx % len(cols)
            with cols[col_idx]:
                st.markdown(f"**Topic {topic_idx + 1}**")
                # Get top products for this topic
                topic_probs = topic_product_dist[topic_idx]
                top_indices = np.argsort(topic_probs)[::-1][:n_top]
                for idx in top_indices:
                    prob = topic_probs[idx]
                    product_name = product_columns[idx] if idx < len(product_columns) else f"Product {idx}"
                    st.write(f"- {product_name}: {prob:.3f}")


def _render_network_viz(result, product_columns):
    """Network Analysis visualizations."""
    import plotly.express as px
    import plotly.graph_objects as go

    communities = result.get('communities')
    n_communities = result.get('n_communities', 0)
    centrality_scores = result.get('centrality_scores')
    graph_metrics = result.get('graph_metrics', {})

    # Network metrics
    st.subheader("Network Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Communities", n_communities)
    with col2:
        st.metric("Modularity", f"{graph_metrics.get('modularity', 0):.3f}",
                 help="Higher modularity indicates stronger community structure")
    with col3:
        st.metric("Density", f"{graph_metrics.get('density', 0):.3f}",
                 help="Proportion of possible edges that exist")
    with col4:
        st.metric("Edges", graph_metrics.get('n_edges', 0))

    # Community membership
    if communities is not None:
        st.subheader("Product Communities")

        # Create a dataframe for display
        community_df = pd.DataFrame({
            'Product': product_columns,
            'Community': communities,
            'Centrality': centrality_scores if centrality_scores is not None else [0] * len(product_columns)
        })
        community_df = community_df.sort_values(['Community', 'Centrality'], ascending=[True, False])

        # Show as expandable sections per community
        for comm_idx in range(n_communities):
            comm_products = community_df[community_df['Community'] == comm_idx]
            with st.expander(f"Community {comm_idx + 1} ({len(comm_products)} products)", expanded=(comm_idx < 3)):
                st.dataframe(
                    comm_products[['Product', 'Centrality']].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )

    # Top central products
    if centrality_scores is not None:
        st.subheader("Most Central Products")
        n_top = st.slider("Number of products", 5, 20, 10, key="network_top_products")

        centrality_df = pd.DataFrame({
            'Product': product_columns,
            'Eigenvector Centrality': centrality_scores,
            'Degree Centrality': result.get('degree_centrality', [0] * len(product_columns)),
            'Community': communities if communities is not None else [0] * len(product_columns)
        })
        centrality_df = centrality_df.nlargest(n_top, 'Eigenvector Centrality')

        fig = px.bar(
            centrality_df,
            x='Product',
            y='Eigenvector Centrality',
            color='Community' if communities is not None else None,
            title="Top Central Products",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Adjacency matrix / similarity
    adjacency = result.get('adjacency_matrix')
    if adjacency is not None:
        st.subheader("Product Co-Purchase Network")
        fig = plot_correlation_matrix(adjacency, product_columns, "Co-Purchase Strength")
        st.plotly_chart(fig, use_container_width=True)


def _render_biplot_section(product_embeddings, product_labels, household_embeddings, var_explained):
    """Render the biplot section with dimension selectors."""
    st.markdown("---")
    st.header("🎯 Biplot")
    
    n_dims = product_embeddings.shape[1]
    col1, col2, col3 = st.columns(3)
    with col1:
        dim_x = st.selectbox("X-axis Dimension", range(1, n_dims + 1), index=0) - 1
    with col2:
        dim_y = st.selectbox("Y-axis Dimension", range(1, n_dims + 1), index=min(1, n_dims - 1)) - 1
    with col3:
        show_households = st.checkbox("Show Households", value=True)
    
    cluster_labels = None
    if st.session_state.cluster_result is not None:
        cluster_labels = st.session_state.cluster_result.get('labels')
    
    fig = plot_biplot(
        product_embeddings, product_labels,
        household_embeddings if show_households else None,
        dim_x=dim_x, dim_y=dim_y,
        var_explained=var_explained,
        cluster_labels=cluster_labels,
        title="Product Biplot"
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_clustering_section(product_embeddings, similarity_matrix, product_labels):
    """Render the product clustering section using API."""
    st.markdown("---")
    st.header("Product Clustering")

    # Check if we have a run ID for API-based clustering
    run_id = st.session_state.get('current_run_id')
    if run_id is None:
        st.warning("No model run available for clustering.")
        return

    col1, col2 = st.columns(2)
    with col1:
        cluster_method = st.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
    with col2:
        auto_k = st.checkbox("Auto-detect optimal K", value=True)

    client = get_api_client()
    api_method = "kmeans" if cluster_method == "K-Means" else "hierarchical"

    n_clusters = None
    if auto_k:
        # Run clustering with auto-detection
        with st.spinner("Finding optimal number of clusters..."):
            try:
                api_result = client.run_clustering(
                    run_id=run_id,
                    method=api_method,
                    n_clusters=None,  # Auto-detect
                    max_k=min(10, len(product_labels) - 1),
                )
                cluster_result = adapt_clustering_results(api_result)

                if cluster_result.get('optimal_k'):
                    st.info(f"Optimal number of clusters: {cluster_result['optimal_k']}")
                    n_clusters = cluster_result['optimal_k']

                if cluster_result.get('scores') and cluster_result.get('range'):
                    fig = plot_silhouette_scores(
                        cluster_result['range'],
                        cluster_result['scores'],
                        cluster_result.get('optimal_k')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Store the result
                st.session_state.cluster_result = cluster_result

            except APIError as e:
                st.error(f"Clustering failed: {e.message}")
                return
    else:
        n_clusters = st.slider("Number of Clusters", 2, min(10, len(product_labels) - 1), 3)

    # For hierarchical clustering, show dendrogram
    if cluster_method == "Hierarchical":
        cluster_result = st.session_state.get('cluster_result')
        if cluster_result and cluster_result.get('linkage_matrix') is not None:
            st.subheader("Dendrogram")
            fig = plot_dendrogram(
                cluster_result['linkage_matrix'],
                list(product_labels),
                n_clusters=n_clusters or cluster_result.get('n_clusters', 3),
                title="Product Hierarchy (cut line shows selected clusters)"
            )
            st.plotly_chart(fig, use_container_width=True)

    if st.button("Run Clustering"):
        with st.spinner("Running clustering..."):
            try:
                api_result = client.run_clustering(
                    run_id=run_id,
                    method=api_method,
                    n_clusters=n_clusters,
                    max_k=min(10, len(product_labels) - 1),
                )
                cluster_result = adapt_clustering_results(api_result)
                st.session_state.cluster_result = cluster_result

                st.subheader("Cluster Assignments")
                cluster_df = get_cluster_members(cluster_result['labels'], product_labels)
                st.dataframe(cluster_df, use_container_width=True)

                if cluster_result.get('silhouette_score'):
                    st.metric("Silhouette Score", f"{cluster_result['silhouette_score']:.3f}")

            except APIError as e:
                st.error(f"Clustering failed: {e.message}")


def _render_export_section(model_result, model_type, product_columns,
                           product_embeddings, household_embeddings,
                           var_explained, similarity_matrix):
    """Render the export section using API export endpoint."""
    st.markdown("---")
    st.header("Export Results")

    run_id = st.session_state.get('current_run_id')
    if run_id is None:
        st.warning("No model run available for export.")
        return

    client = get_api_client()

    try:
        # Use API export endpoint
        zip_data = client.export_results(run_id)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_short = model_type.split(" (")[0].lower().replace(" ", "_")
        filename = f"{model_short}_results_{timestamp}.zip"

        st.download_button(
            label="Download All Results (ZIP)",
            data=zip_data,
            file_name=filename,
            mime="application/zip"
        )
    except APIError as e:
        st.error(f"Export failed: {e.message}")
    except Exception as e:
        st.error(f"Error creating export: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()