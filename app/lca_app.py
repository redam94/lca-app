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

The heavy lifting (model fitting, plotting, utilities) is handled by the
market_structure package, keeping this file focused on UI orchestration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Import from our package
from market_structure.config import (
    PYMC_AVAILABLE, PYMC_ERROR, PRINCE_AVAILABLE,
    get_available_models, get_model_help_text
)
from market_structure.models import (
    fit_lca, compute_lca_coordinates, compute_residual_correlations,
    fit_lca_with_covariates, interpret_covariate_effects,
    fit_factor_analysis_tetrachoric, fit_bayesian_factor_vi,
    compute_factor_scores_regression, fit_nmf
)
from market_structure.plotting import (
    plot_correlation_matrix, plot_loadings_heatmap, plot_loadings_with_uncertainty,
    plot_variance_explained, plot_elbo_convergence, plot_silhouette_scores,
    plot_lca_profiles, plot_biplot, plot_dcm_coefficients, plot_dendrogram
)
from market_structure.utils import (
    get_model_cache_key, compute_hierarchical_clustering,
    find_optimal_clusters, perform_kmeans_clustering, get_hierarchical_labels,
    get_cluster_members, create_export_zip
)

# Conditional imports for optional models
if PRINCE_AVAILABLE:
    from market_structure.models import fit_mca

if PYMC_AVAILABLE:
    from market_structure.models import fit_bayesian_factor_model_pymc, fit_discrete_choice_model


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
        'original_data': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="Latent Structure Analysis", layout="wide")
    
    st.title("🛒 Latent Structure Analysis for Purchase Data")
    st.markdown("""
    Discover latent customer segments and product relationships using multiple statistical methods.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Display dependency status
    if not PYMC_AVAILABLE:
        if PYMC_ERROR:
            st.warning(f"⚠️ PyMC import error: {PYMC_ERROR}. PyMC-based models will be unavailable.")
        else:
            st.warning("⚠️ PyMC not installed. Install with: `pip install pymc arviz`")
    
    if not PRINCE_AVAILABLE:
        st.info("💡 Install `prince` for MCA: `pip install prince`")
    
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
    
    return params


def _invalidate_cache():
    """Clear all cached model results."""
    keys = ['model_result', 'model_cache_key', 'model_type_cached', 
            'product_columns_cached', 'similarity_matrix_cached',
            'product_embeddings', 'household_embeddings', 'var_explained_cached',
            'cluster_result', 'tetra_corr_cached', 'elbo_history_cached',
            'convergence_msg', 'model_metrics']
    for key in keys:
        st.session_state[key] = None


def _run_analysis(X, model_type, params, product_columns, df, 
                  household_feature_columns, cache_key):
    """Run the selected model and store results in session state."""
    
    if model_type == "Latent Class Analysis (LCA)":
        _run_lca(X, params, product_columns, df, household_feature_columns, cache_key)
    
    elif model_type == "Factor Analysis (Tetrachoric)":
        _run_tetrachoric_fa(X, params, product_columns, cache_key)
    
    elif model_type == "Bayesian Factor Model (VI)":
        _run_bayesian_vi(X, params, product_columns, cache_key)
    
    elif model_type == "Bayesian Factor Model (PyMC)":
        _run_bayesian_pymc(X, params, product_columns, cache_key)
    
    elif model_type == "Non-negative Matrix Factorization (NMF)":
        _run_nmf(X, params, product_columns, cache_key)
    
    elif model_type == "Multiple Correspondence Analysis (MCA)":
        _run_mca(X, params, product_columns, cache_key)
    
    elif model_type == "Discrete Choice Model (PyMC)":
        _run_dcm(X, params, product_columns, df, household_feature_columns, cache_key)


def _run_lca(X, params, product_columns, df, household_feature_columns, cache_key):
    """Run Latent Class Analysis, optionally with household covariates."""
    st.header("📊 Latent Class Analysis Results")
    
    # Check if we should use covariates
    use_covariates = len(household_feature_columns) > 0
    
    if use_covariates:
        # Prepare covariate matrix
        covariates_df = df[household_feature_columns].copy()
        
        # Handle missing values - need to align with X
        # Find rows that were kept after product column dropna
        product_df = df[product_columns].copy()
        valid_rows = product_df.notna().all(axis=1)
        covariates_df = covariates_df.loc[valid_rows]
        
        # Check for missing values in covariates
        if covariates_df.isnull().any().any():
            st.warning("Covariates contain missing values. Imputing with column means.")
            covariates_df = covariates_df.fillna(covariates_df.mean())
        
        # Standardize numeric covariates for better optimization
        covariates = covariates_df.values.astype(float)
        covariate_means = covariates.mean(axis=0)
        covariate_stds = covariates.std(axis=0) + 1e-10
        covariates_standardized = (covariates - covariate_means) / covariate_stds
        
        st.info(f"Using {len(household_feature_columns)} household covariates to predict class membership")
        
        with st.spinner("Fitting LCA with covariates..."):
            result = fit_lca_with_covariates(
                X, covariates_standardized, 
                params['n_classes'],
                max_iter=params['max_iter'], 
                n_init=params['n_init']
            )
        
        # Store feature names in result for interpretation
        result['feature_names'] = ['Intercept'] + list(household_feature_columns)
        result['covariate_means'] = covariate_means
        result['covariate_stds'] = covariate_stds
        
        # For LCA with covariates, class_probs varies by household
        # Use mean class probabilities for variance explained display
        mean_class_probs = result['class_probs_per_hh'].mean(axis=0)
        var_explained = mean_class_probs * 100
        
    else:
        with st.spinner("Fitting LCA model..."):
            result = fit_lca(X, params['n_classes'], 
                            max_iter=params['max_iter'], 
                            n_init=params['n_init'])
        var_explained = result['class_probs'] * 100
    
    # Compute coordinates for visualization
    # For covariate model, use mean class probs; for standard model, use class_probs
    if use_covariates:
        household_coords, product_coords = compute_lca_coordinates(
            mean_class_probs, result['item_probs'], result['responsibilities']
        )
    else:
        household_coords, product_coords = compute_lca_coordinates(
            result['class_probs'], result['item_probs'], result['responsibilities']
        )
    
    residual_corr = compute_residual_correlations(
        X, result['responsibilities'], result['item_probs']
    )
    
    _store_results(result, cache_key, "Latent Class Analysis (LCA)", 
                  product_columns, residual_corr, product_coords, household_coords,
                  var_explained,
                  f"Model converged in {result['n_iter']} iterations",
                  {'BIC': result['bic'], 'AIC': result['aic'], 
                   'Log-Likelihood': result['log_likelihood']})


def _run_tetrachoric_fa(X, params, product_columns, cache_key):
    """Run Tetrachoric Factor Analysis."""
    st.header("📊 Factor Analysis (Tetrachoric)")
    
    with st.spinner("Computing tetrachoric correlations and fitting factors..."):
        result = fit_factor_analysis_tetrachoric(X, params['n_factors'], 
                                                  max_iter=params['max_iter'])
    
    loadings = result['loadings']
    loadings_norm = loadings / (np.linalg.norm(loadings, axis=0, keepdims=True) + 1e-10)
    implied_corr = loadings_norm @ loadings_norm.T
    scores = compute_factor_scores_regression(X, loadings)
    
    st.session_state.tetra_corr_cached = result['tetra_corr']
    
    _store_results(result, cache_key, "Factor Analysis (Tetrachoric)",
                  product_columns, implied_corr, loadings, scores,
                  result['var_explained_pct'],
                  f"Model converged in {result['n_iter']} iterations", {})


def _run_bayesian_vi(X, params, product_columns, cache_key):
    """Run Bayesian Factor Model with Variational Inference."""
    st.header("📊 Bayesian Factor Model (VI)")
    
    with st.spinner("Fitting Bayesian factor model..."):
        result = fit_bayesian_factor_vi(X, params['n_factors'], 
                                        max_iter=params['max_iter'])
    
    loadings_norm = result['loadings'] / (np.linalg.norm(result['loadings'], axis=0, keepdims=True) + 1e-10)
    implied_corr = loadings_norm @ loadings_norm.T
    
    st.session_state.elbo_history_cached = result['elbo_history']
    
    _store_results(result, cache_key, "Bayesian Factor Model (VI)",
                  product_columns, implied_corr, result['loadings'], result['scores'],
                  result['var_explained_pct'],
                  f"Model converged in {result['n_iter']} iterations", {})


def _run_bayesian_pymc(X, params, product_columns, cache_key):
    """Run Bayesian Factor Model with PyMC MCMC."""
    st.header("📊 Bayesian Factor Model (PyMC)")
    
    with st.spinner("Running MCMC sampling... This may take a few minutes."):
        result = fit_bayesian_factor_model_pymc(
            X, params['n_factors'], 
            n_samples=params['n_samples'],
            n_tune=params['n_tune']
        )
    
    loadings_norm = result['loadings'] / (np.linalg.norm(result['loadings'], axis=0, keepdims=True) + 1e-10)
    implied_corr = loadings_norm @ loadings_norm.T
    scores = compute_factor_scores_regression(X, result['loadings'])
    
    try:
        waic = result['waic'].elpd_waic if result['waic'] else None
    except:
        waic = None
    
    _store_results(result, cache_key, "Bayesian Factor Model (PyMC)",
                  product_columns, implied_corr, result['loadings'], scores,
                  result['var_explained_pct'],
                  "MCMC sampling complete!",
                  {'WAIC': waic, 'Divergences': result.get('n_divergences', 0)})


def _run_nmf(X, params, product_columns, cache_key):
    """Run Non-negative Matrix Factorization."""
    st.header("📊 Non-negative Matrix Factorization")
    
    with st.spinner("Fitting NMF model..."):
        result = fit_nmf(X, params['n_components'], max_iter=params['max_iter'])
    
    H_norm = result['H'] / (np.linalg.norm(result['H'], axis=1, keepdims=True) + 1e-10)
    similarity = H_norm.T @ H_norm
    
    _store_results(result, cache_key, "Non-negative Matrix Factorization (NMF)",
                  product_columns, similarity, result['loadings'], result['scores'],
                  result['var_explained_pct'],
                  f"Model converged in {result['n_iter']} iterations",
                  {'Reconstruction Error': result['reconstruction_error']})


def _run_mca(X, params, product_columns, cache_key):
    """Run Multiple Correspondence Analysis."""
    st.header("📊 Multiple Correspondence Analysis")
    
    with st.spinner("Fitting MCA model..."):
        result = fit_mca(X, params['n_components'], product_names=product_columns)
    
    # MCA may produce different product labels (filtering for purchased categories)
    st.session_state.product_columns_cached = result.get('product_labels', product_columns)
    
    _store_results(result, cache_key, "Multiple Correspondence Analysis (MCA)",
                  result.get('product_labels', product_columns),
                  result['similarity_matrix'],
                  result['column_coordinates'], result['row_coordinates'],
                  result['var_explained_pct'],
                  "MCA completed!",
                  {'Total Inertia': result['total_inertia']})


def _run_dcm(X, params, product_columns, df, household_feature_columns, cache_key):
    """Run Discrete Choice Model."""
    st.header("📊 Discrete Choice Model")
    
    hh_features = None
    if household_feature_columns:
        hh_features = df[household_feature_columns].values.astype(float)
        hh_features = (hh_features - hh_features.mean(axis=0)) / (hh_features.std(axis=0) + 1e-10)
    
    with st.spinner("Running MCMC sampling... This may take a few minutes."):
        result = fit_discrete_choice_model(
            X,
            household_features=hh_features,
            n_samples=params['n_samples'],
            n_tune=params['n_tune'],
            include_random_effects=params.get('include_random_effects', False),
            n_latent_features=params.get('n_latent_features', 0),
            latent_prior_scale=params.get('latent_prior_scale', 1.0)
        )
    
    if params.get('n_latent_features', 0) > 0:
        product_emb = result['product_latent']
        household_emb = result['household_latent']
        prod_norm = product_emb / (np.linalg.norm(product_emb, axis=1, keepdims=True) + 1e-10)
        similarity = prod_norm @ prod_norm.T
    else:
        product_emb = None
        household_emb = None
        similarity = None
    
    try:
        waic = result['waic'].elpd_waic if result['waic'] else None
    except:
        waic = None
    
    _store_results(result, cache_key, "Discrete Choice Model (PyMC)",
                  product_columns, similarity, product_emb, household_emb,
                  None, "MCMC sampling complete!",
                  {'WAIC': waic, 'Divergences': result.get('n_divergences', 0)})


def _store_results(result, cache_key, model_type, product_columns,
                   similarity, product_emb, household_emb, var_explained,
                   conv_msg, metrics):
    """Store model results in session state."""
    st.session_state.model_result = result
    st.session_state.model_cache_key = cache_key
    st.session_state.model_type_cached = model_type
    st.session_state.product_columns_cached = product_columns
    st.session_state.similarity_matrix_cached = similarity
    st.session_state.product_embeddings = product_emb
    st.session_state.household_embeddings = household_emb
    st.session_state.var_explained_cached = var_explained
    st.session_state.cluster_result = None
    st.session_state.convergence_msg = conv_msg
    st.session_state.model_metrics = metrics
    st.success(conv_msg)


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
    
    # Handle both standard LCA and LCA with covariates
    if 'class_probs' in result:
        fig = plot_lca_profiles(result['item_probs'], result['class_probs'], product_columns)
    else:
        # For covariate model, use mean class probabilities
        mean_probs = result['class_probs_per_hh'].mean(axis=0)
        fig = plot_lca_profiles(result['item_probs'], mean_probs, product_columns)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display covariate effects if available (LCA with covariates)
    if 'beta' in result and result.get('feature_names'):
        st.subheader("🏠 Covariate Effects on Class Membership")
        st.markdown("""
        These coefficients show how household characteristics influence the probability 
        of belonging to each class. Positive values mean the feature increases odds of 
        membership; negative values decrease odds. The last class serves as the reference.
        """)
        
        # Get interpretation
        effects = interpret_covariate_effects(
            result['beta'],
            result['feature_names'],
            [f"Class {i+1}" for i in range(result['n_classes'])]
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
        if effects['summary']:
            st.markdown("**Key Findings:**")
            st.info(effects['summary'])
        
        # Show distribution of class probabilities across households
        st.subheader("Class Probability Distribution")
        st.markdown("Shows how class membership probabilities vary across households due to covariates.")
        
        class_probs = result['class_probs_per_hh']
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        n_classes = class_probs.shape[1]
        fig = make_subplots(rows=1, cols=n_classes, 
                           subplot_titles=[f"Class {i+1}" for i in range(n_classes)])
        
        for c in range(n_classes):
            fig.add_trace(
                go.Histogram(x=class_probs[:, c], nbinsx=30, 
                            name=f"Class {c+1}", showlegend=False,
                            marker_color='steelblue'),
                row=1, col=c+1
            )
            fig.update_xaxes(title_text="P(class)", row=1, col=c+1)
            fig.update_yaxes(title_text="Count" if c == 0 else "", row=1, col=c+1)
        
        fig.update_layout(height=300, title="Distribution of Class Probabilities Across Households")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Residual Correlations (Substitution Patterns)")
    fig = plot_correlation_matrix(similarity, product_columns, "Residual Correlations")
    st.plotly_chart(fig, use_container_width=True)


def _render_factor_viz(result, product_columns, tetra_corr):
    """Tetrachoric FA visualizations."""
    st.subheader("Tetrachoric Correlation Matrix")
    fig = plot_correlation_matrix(tetra_corr, product_columns, "Tetrachoric Correlations")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Factor Loadings")
    fig = plot_loadings_heatmap(result['loadings'], product_columns)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variance Explained")
    fig = plot_variance_explained(result['var_explained_pct'], "Factor Analysis")
    st.plotly_chart(fig, use_container_width=True)


def _render_vi_viz(result, product_columns):
    """Bayesian VI visualizations."""
    st.subheader("Factor Loadings")
    fig = plot_loadings_heatmap(result['loadings'], product_columns)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variance Explained")
    fig = plot_variance_explained(result['var_explained_pct'], "Bayesian FA (VI)")
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.elbo_history_cached:
        st.subheader("ELBO Convergence")
        fig = plot_elbo_convergence(st.session_state.elbo_history_cached)
        st.plotly_chart(fig, use_container_width=True)


def _render_pymc_viz(result, product_columns):
    """Bayesian PyMC visualizations."""
    st.subheader("Factor Loadings with Uncertainty")
    fig = plot_loadings_with_uncertainty(result['loadings'], result['loadings_std'], product_columns)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variance Explained")
    fig = plot_variance_explained(result['var_explained_pct'], "Bayesian FA (PyMC)")
    st.plotly_chart(fig, use_container_width=True)


def _render_nmf_viz(result, product_columns):
    """NMF visualizations."""
    st.subheader("Component Loadings")
    fig = plot_loadings_heatmap(result['loadings'], product_columns, title="NMF Component Loadings")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variance Explained")
    fig = plot_variance_explained(result['var_explained_pct'], "NMF")
    st.plotly_chart(fig, use_container_width=True)


def _render_mca_viz(result, product_labels):
    """MCA visualizations."""
    st.subheader("Variance Explained")
    fig = plot_variance_explained(result['var_explained_pct'], "MCA")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Product Similarity")
    fig = plot_correlation_matrix(result['similarity_matrix'], product_labels, "MCA-Based Similarity")
    st.plotly_chart(fig, use_container_width=True)


def _render_dcm_viz(result, product_columns):
    """DCM visualizations."""
    st.subheader("Product Intercepts")
    fig = plot_dcm_coefficients(result['alpha'], result['alpha_std'], product_columns)
    st.plotly_chart(fig, use_container_width=True)
    
    if result.get('n_latent_features', 0) > 0:
        st.subheader("Latent Product Features")
        fig = plot_loadings_heatmap(
            result['product_latent'], 
            product_columns,
            factor_names=[f"Latent {i+1}" for i in range(result['n_latent_features'])],
            title="Latent Product Features"
        )
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
    """Render the product clustering section."""
    st.markdown("---")
    st.header("🔍 Product Clustering")
    
    col1, col2 = st.columns(2)
    with col1:
        cluster_method = st.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
    with col2:
        auto_k = st.checkbox("Auto-detect optimal K", value=True)
    
    if auto_k:
        with st.spinner("Finding optimal number of clusters..."):
            optimal_result = find_optimal_clusters(product_embeddings, max_k=min(10, len(product_embeddings) - 1))
        
        st.info(f"Optimal number of clusters: {optimal_result['optimal_k']}")
        fig = plot_silhouette_scores(optimal_result['range'], optimal_result['scores'], optimal_result['optimal_k'])
        st.plotly_chart(fig, use_container_width=True)
        
        n_clusters = optimal_result['optimal_k']
    else:
        n_clusters = st.slider("Number of Clusters", 2, min(10, len(product_embeddings) - 1), 3)
    
    # For hierarchical clustering, show the dendrogram even before running
    # This helps users understand the tree structure and decide where to cut
    if cluster_method == "Hierarchical" and similarity_matrix is not None:
        with st.spinner("Computing hierarchical clustering..."):
            hier_result = compute_hierarchical_clustering(similarity_matrix)
        
        st.subheader("Dendrogram")
        fig = plot_dendrogram(
            hier_result['linkage_matrix'],
            list(product_labels),
            n_clusters=n_clusters,
            title="Product Hierarchy (cut line shows selected clusters)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Store for use in clustering button
        st.session_state['_hier_result_temp'] = hier_result
    
    if st.button("Run Clustering"):
        if cluster_method == "K-Means":
            cluster_result = perform_kmeans_clustering(product_embeddings, n_clusters)
        else:
            # Use cached hierarchical result if available
            if '_hier_result_temp' in st.session_state:
                hier_result = st.session_state['_hier_result_temp']
            else:
                hier_result = compute_hierarchical_clustering(similarity_matrix)
            labels = get_hierarchical_labels(hier_result['linkage_matrix'], n_clusters)
            cluster_result = {'labels': labels, 'linkage_matrix': hier_result['linkage_matrix']}
        
        st.session_state.cluster_result = cluster_result
        
        st.subheader("Cluster Assignments")
        cluster_df = get_cluster_members(cluster_result['labels'], product_labels)
        st.dataframe(cluster_df, use_container_width=True)


def _render_export_section(model_result, model_type, product_columns,
                           product_embeddings, household_embeddings,
                           var_explained, similarity_matrix):
    """Render the export section."""
    st.markdown("---")
    st.header("📥 Export Results")
    
    try:
        zip_data = create_export_zip(
            model_result=model_result,
            model_type=model_type,
            product_columns=product_columns,
            product_embeddings=product_embeddings,
            household_embeddings=household_embeddings,
            var_explained=var_explained,
            similarity_matrix=similarity_matrix,
            cluster_result=st.session_state.cluster_result,
            original_data=st.session_state.original_data
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_short = model_type.split(" (")[0].lower().replace(" ", "_")
        filename = f"{model_short}_results_{timestamp}.zip"
        
        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=zip_data,
            file_name=filename,
            mime="application/zip"
        )
    except Exception as e:
        st.error(f"Error creating export: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()