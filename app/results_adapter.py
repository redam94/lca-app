"""
Results adapter for converting API responses to formats expected by plotting functions.

The API returns data as JSON-serializable lists, while the plotting functions
from market_structure.plotting expect numpy arrays. This module handles the
conversion and provides a consistent interface for the frontend.
"""

from typing import Optional, Any
import numpy as np


def _to_array(data: Optional[list]) -> Optional[np.ndarray]:
    """Convert list to numpy array if not None."""
    if data is None:
        return None
    return np.array(data)


def adapt_results_for_plotting(api_results: dict, model_type: str) -> dict:
    """
    Convert API results to format expected by market_structure.plotting functions.

    This converts JSON lists back to numpy arrays and restructures the data
    to match what the plotting functions expect.

    Args:
        api_results: Response from API get_results endpoint
        model_type: The model type (e.g., "lca", "factor_tetrachoric")

    Returns:
        Dict with numpy arrays ready for plotting functions
    """
    result = {}

    # Common fields - convert to numpy arrays
    result['product_embeddings'] = _to_array(api_results.get('product_embeddings'))
    result['household_embeddings'] = _to_array(api_results.get('household_embeddings'))
    result['similarity_matrix'] = _to_array(api_results.get('similarity_matrix'))
    result['variance_explained'] = _to_array(api_results.get('variance_explained'))

    # Factor model fields
    result['loadings'] = _to_array(api_results.get('loadings'))
    result['loadings_std'] = _to_array(api_results.get('loadings_std'))

    # LCA fields
    result['item_probs'] = _to_array(api_results.get('item_probs'))
    result['class_probs'] = _to_array(api_results.get('class_probs'))

    # DCM fields
    result['alpha'] = _to_array(api_results.get('alpha'))
    result['alpha_std'] = _to_array(api_results.get('alpha_std'))
    result['product_latent'] = _to_array(api_results.get('product_latent'))
    result['household_latent'] = _to_array(api_results.get('household_latent'))

    # LCA with covariates fields
    result['beta'] = _to_array(api_results.get('beta'))
    result['odds_ratios'] = _to_array(api_results.get('odds_ratios'))
    result['covariate_columns'] = api_results.get('covariate_columns')
    result['class_probs_per_hh'] = _to_array(api_results.get('class_probs_per_hh'))

    # Additional model-specific fields
    result['residual_correlations'] = _to_array(api_results.get('residual_correlations'))
    result['tetra_corr'] = _to_array(api_results.get('tetra_corr'))
    result['elbo_history'] = api_results.get('elbo_history')  # Keep as list for plotting

    # Metadata
    result['product_columns'] = api_results.get('product_columns', [])
    result['metrics'] = api_results.get('metrics', {})
    result['model_type'] = model_type

    # Copy over raw results for any model-specific access
    if 'results' in api_results:
        result['raw_results'] = api_results['results']

    # Model-specific adaptations
    api_model_type = api_results.get('model_type', model_type)

    if api_model_type in ['lca', 'lca_covariates']:
        _adapt_lca_results(result, api_results, api_model_type)
    elif api_model_type in ['factor_tetrachoric', 'bayesian_factor_vi', 'bayesian_factor_pymc']:
        _adapt_factor_results(result, api_results, api_model_type)
    elif api_model_type == 'nmf':
        _adapt_nmf_results(result, api_results)
    elif api_model_type == 'mca':
        _adapt_mca_results(result, api_results)
    elif api_model_type == 'dcm':
        _adapt_dcm_results(result, api_results)
    elif api_model_type == 'lda':
        _adapt_lda_results(result, api_results)
    elif api_model_type == 'network':
        _adapt_network_results(result, api_results)

    return result


def _adapt_lca_results(result: dict, api_results: dict, model_type: str) -> None:
    """Adapt LCA-specific results for plotting."""
    # Ensure we have responsibilities for household embeddings
    raw = api_results.get('results', {})
    if result['household_embeddings'] is None and 'responsibilities' in raw:
        result['household_embeddings'] = _to_array(raw['responsibilities'])
        result['responsibilities'] = result['household_embeddings']

    # Store responsibilities separately for LCA coordinate computation
    result['responsibilities'] = result['household_embeddings']

    # Get convergence info
    result['n_iter'] = raw.get('n_iter')
    result['log_likelihood'] = raw.get('log_likelihood')
    result['bic'] = raw.get('bic')
    result['aic'] = raw.get('aic')
    result['n_classes'] = raw.get('n_classes')

    # For LCA with covariates, add interpretation fields
    if model_type == 'lca_covariates':
        result['feature_names'] = raw.get('feature_names', ['Intercept'] + (result.get('covariate_columns') or []))
        result['n_features'] = raw.get('n_features')


def _adapt_factor_results(result: dict, api_results: dict, model_type: str) -> None:
    """Adapt factor analysis results for plotting."""
    raw = api_results.get('results', {})

    # Get scores if not already set
    if result['household_embeddings'] is None and 'scores' in raw:
        result['household_embeddings'] = _to_array(raw['scores'])

    result['scores'] = result['household_embeddings']

    # Get convergence info
    result['n_iter'] = raw.get('n_iter')
    result['var_explained_pct'] = result['variance_explained']

    # Tetrachoric correlation matrix
    if model_type == 'factor_tetrachoric' and result['tetra_corr'] is None:
        result['tetra_corr'] = _to_array(raw.get('tetra_corr'))


def _adapt_nmf_results(result: dict, api_results: dict) -> None:
    """Adapt NMF results for plotting."""
    raw = api_results.get('results', {})

    # NMF stores H matrix for component loadings
    result['H'] = _to_array(raw.get('H'))
    result['W'] = _to_array(raw.get('W'))

    # Scores are the W matrix (households x components)
    if result['household_embeddings'] is None and result['W'] is not None:
        result['household_embeddings'] = result['W']

    result['scores'] = result['household_embeddings']

    # Get convergence info
    result['n_iter'] = raw.get('n_iter')
    result['reconstruction_error'] = raw.get('reconstruction_error')
    result['var_explained_pct'] = result['variance_explained']


def _adapt_mca_results(result: dict, api_results: dict) -> None:
    """Adapt MCA results for plotting."""
    raw = api_results.get('results', {})

    # MCA uses column_coordinates for products
    result['column_coordinates'] = result['product_embeddings']
    result['row_coordinates'] = result['household_embeddings']

    # Total inertia
    result['total_inertia'] = raw.get('total_inertia')
    result['var_explained_pct'] = result['variance_explained']

    # Product labels may have been filtered
    result['product_labels'] = raw.get('product_labels', result['product_columns'])


def _adapt_dcm_results(result: dict, api_results: dict) -> None:
    """Adapt DCM results for plotting."""
    raw = api_results.get('results', {})

    # DCM uses latent features
    result['n_latent_features'] = raw.get('n_latent_features', 0)
    result['n_divergences'] = raw.get('n_divergences', 0)

    # WAIC if available
    waic_data = raw.get('waic')
    if isinstance(waic_data, dict):
        result['waic'] = waic_data
    else:
        result['waic'] = None


def _adapt_lda_results(result: dict, api_results: dict) -> None:
    """Adapt LDA results for plotting."""
    raw = api_results.get('results', {})

    # LDA-specific fields
    result['topic_product_dist'] = _to_array(api_results.get('topic_product_dist'))
    result['household_topic_dist'] = _to_array(api_results.get('household_topic_dist'))
    result['perplexity'] = api_results.get('perplexity')
    result['n_topics'] = api_results.get('n_topics') or raw.get('n_topics')
    result['log_likelihood'] = raw.get('log_likelihood')
    result['n_iter'] = raw.get('n_iter')

    # Use loadings/scores for biplot compatibility
    if result['product_embeddings'] is None and result['loadings'] is not None:
        result['product_embeddings'] = result['loadings']

    if result['household_embeddings'] is None and 'scores' in raw:
        result['household_embeddings'] = _to_array(raw['scores'])

    result['scores'] = result['household_embeddings']


def _adapt_network_results(result: dict, api_results: dict) -> None:
    """Adapt Network Analysis results for plotting."""
    raw = api_results.get('results', {})

    # Network-specific fields
    result['adjacency_matrix'] = _to_array(api_results.get('adjacency_matrix'))
    result['communities'] = api_results.get('communities')
    result['n_communities'] = api_results.get('n_communities') or raw.get('n_communities')
    result['centrality_scores'] = _to_array(api_results.get('centrality_scores'))
    result['degree_centrality'] = _to_array(api_results.get('degree_centrality'))
    result['betweenness_centrality'] = _to_array(api_results.get('betweenness_centrality'))
    result['graph_metrics'] = api_results.get('graph_metrics') or raw.get('graph_metrics', {})
    result['edge_list'] = raw.get('edge_list', [])
    result['threshold'] = raw.get('threshold')
    result['edge_method'] = raw.get('edge_method')
    result['community_method'] = raw.get('community_method')

    # Use loadings/scores for biplot compatibility
    if result['product_embeddings'] is None and result['loadings'] is not None:
        result['product_embeddings'] = result['loadings']

    if result['household_embeddings'] is None and 'scores' in raw:
        result['household_embeddings'] = _to_array(raw['scores'])

    result['scores'] = result['household_embeddings']


def adapt_clustering_results(api_clustering: dict) -> dict:
    """
    Convert API clustering response to format expected by clustering visualization.

    Args:
        api_clustering: Response from API clustering endpoint

    Returns:
        Dict with numpy arrays and cluster assignments
    """
    result = {
        'labels': np.array(api_clustering.get('labels', [])),
        'n_clusters': api_clustering.get('n_clusters'),
        'method': api_clustering.get('method'),
        'product_columns': api_clustering.get('product_columns', []),
        'silhouette_score': api_clustering.get('silhouette_score'),
        'inertia': api_clustering.get('inertia'),
        'cluster_members': api_clustering.get('cluster_members', {}),
    }

    # Hierarchical clustering linkage matrix
    linkage = api_clustering.get('linkage_matrix')
    if linkage is not None:
        result['linkage_matrix'] = np.array(linkage)
    else:
        result['linkage_matrix'] = None

    # Auto-detection results
    result['optimal_k'] = api_clustering.get('optimal_k')
    if api_clustering.get('silhouette_scores'):
        result['scores'] = api_clustering['silhouette_scores']
        result['range'] = api_clustering.get('k_range', list(range(2, len(result['scores']) + 2)))

    return result


def extract_model_metrics(api_results: dict) -> dict:
    """
    Extract model fit metrics for display.

    Args:
        api_results: Response from API get_results endpoint

    Returns:
        Dict of metric name -> value pairs for display
    """
    metrics = api_results.get('metrics', {})
    raw = api_results.get('results', {})

    # Merge metrics from both sources
    all_metrics = {**metrics}

    # Add common metrics from raw results
    if 'bic' in raw:
        all_metrics['BIC'] = raw['bic']
    if 'aic' in raw:
        all_metrics['AIC'] = raw['aic']
    if 'log_likelihood' in raw:
        all_metrics['Log-Likelihood'] = raw['log_likelihood']
    if 'reconstruction_error' in raw:
        all_metrics['Reconstruction Error'] = raw['reconstruction_error']
    if 'n_divergences' in raw:
        all_metrics['Divergences'] = raw['n_divergences']
    if 'total_inertia' in raw:
        all_metrics['Total Inertia'] = raw['total_inertia']

    return all_metrics


def get_convergence_message(api_results: dict, model_type: str) -> str:
    """
    Generate convergence message for display.

    Args:
        api_results: Response from API get_results endpoint
        model_type: The model type

    Returns:
        Human-readable convergence message
    """
    raw = api_results.get('results', {})
    n_iter = raw.get('n_iter')

    if model_type in ['bayesian_factor_pymc', 'dcm']:
        return "MCMC sampling complete!"
    elif model_type == 'mca':
        return "MCA completed!"
    elif n_iter is not None:
        return f"Model converged in {n_iter} iterations"
    else:
        return "Model completed successfully"
