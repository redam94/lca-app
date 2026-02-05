"""
Figure generation service for presentations and reports.

Provides functions to list available figure types for a model
and generate individual Plotly figures.
"""

from enum import Enum
from typing import Optional, Any
import numpy as np
import plotly.graph_objects as go


class FigureType(str, Enum):
    """Available figure types."""
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
    NETWORK_GRAPH = "network_graph"
    CENTRALITY_COMPARISON = "centrality_comparison"
    TOP_PRODUCTS_TOPIC = "top_products_topic"
    INTERTOPIC_DISTANCE_MAP = "intertopic_distance_map"
    CLOSEST_COMPETITORS = "closest_competitors"
    MARKET_MAP = "market_map"
    PRODUCT_SCORECARD = "product_scorecard"
    PRODUCT_NEIGHBORHOOD = "product_neighborhood"
    MARKET_SEGMENTS = "market_segments"


# Figure metadata: type -> (display name, description, applicable model types)
FIGURE_METADATA = {
    FigureType.SIMILARITY_MATRIX: (
        "Similarity Matrix",
        "Heatmap showing pairwise product similarity",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
    FigureType.VARIANCE_EXPLAINED: (
        "Variance Explained",
        "Bar chart showing variance explained by each component",
        ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc",
         "nmf", "mca", "lda", "network"]
    ),
    FigureType.FACTOR_LOADINGS: (
        "Factor Loadings",
        "Heatmap of product loadings on each factor/component",
        ["factor_tetrachoric", "bayesian_factor_vi", "bayesian_factor_pymc",
         "nmf", "mca", "dcm", "lda", "network", "lca", "lca_covariates"]
    ),
    FigureType.CLASS_PROFILES: (
        "Class Profiles",
        "Purchase probability by product for each latent class",
        ["lca", "lca_covariates"]
    ),
    FigureType.BIPLOT: (
        "Biplot",
        "2D scatter plot of products in latent space",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
    FigureType.LCA_CLASS_BIPLOT: (
        "Class-Colored Biplot",
        "Biplot with products colored by dominant class assignment",
        ["lca", "lca_covariates"]
    ),
    FigureType.TETRACHORIC: (
        "Tetrachoric Correlation",
        "Tetrachoric correlation matrix between products",
        ["factor_tetrachoric"]
    ),
    FigureType.ELBO_HISTORY: (
        "ELBO Convergence",
        "Evidence Lower Bound over training iterations",
        ["bayesian_factor_vi"]
    ),
    FigureType.TOPIC_DISTRIBUTION: (
        "Topic Distribution",
        "Heatmap of topic-product associations",
        ["lda"]
    ),
    FigureType.NETWORK_MATRIX: (
        "Network Matrix",
        "Product co-purchase network adjacency matrix",
        ["network"]
    ),
    FigureType.DCM_COEFFICIENTS: (
        "Product Intercepts",
        "DCM product intercept coefficients with confidence intervals",
        ["dcm"]
    ),
    FigureType.CLUSTERED_BIPLOT: (
        "Clustered Biplot",
        "Biplot with products colored by cluster assignment",
        []  # Available when clustering is performed
    ),
    FigureType.SILHOUETTE_ANALYSIS: (
        "Silhouette Analysis",
        "Silhouette score vs number of clusters",
        []  # Available when clustering is performed
    ),
    FigureType.CLUSTER_SIZES: (
        "Cluster Sizes",
        "Bar chart of cluster sizes",
        []  # Available when clustering is performed
    ),
    FigureType.DENDROGRAM: (
        "Dendrogram",
        "Hierarchical clustering dendrogram",
        []  # Available for hierarchical clustering
    ),
    FigureType.NETWORK_GRAPH: (
        "Network Graph",
        "Interactive node-link diagram with community coloring and centrality sizing",
        ["network"]
    ),
    FigureType.CENTRALITY_COMPARISON: (
        "Centrality Comparison",
        "Bar chart comparing eigenvector, degree, and betweenness centrality",
        ["network"]
    ),
    FigureType.TOP_PRODUCTS_TOPIC: (
        "Top Products per Topic",
        "Top products for each topic by probability",
        ["lda"]
    ),
    FigureType.INTERTOPIC_DISTANCE_MAP: (
        "Intertopic Distance Map",
        "2D map of topic relationships sized by prevalence",
        ["lda"]
    ),
    FigureType.CLOSEST_COMPETITORS: (
        "Closest Competitors",
        "Ranked pairs of products that compete most directly",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
    FigureType.MARKET_MAP: (
        "Market Map",
        "Competitive landscape showing product positions, importance, and groupings",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
    FigureType.PRODUCT_SCORECARD: (
        "Product Scorecard",
        "Products ranked by market importance with top competitors on hover",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
    FigureType.PRODUCT_NEIGHBORHOOD: (
        "Product Neighborhood",
        "Competitive neighborhood around a selected product",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
    FigureType.MARKET_SEGMENTS: (
        "Market Segments",
        "Natural product groupings shown as a treemap sized by importance",
        ["lca", "lca_covariates", "factor_tetrachoric", "bayesian_factor_vi",
         "bayesian_factor_pymc", "nmf", "mca", "dcm", "lda", "network"]
    ),
}


def _to_list(arr):
    """Convert numpy array to list for Plotly compatibility."""
    if arr is None:
        return None
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return list(arr)


def get_available_figures(
    model_type: str,
    extracted_data: dict,
    clustering_result: Optional[dict] = None
) -> list[dict]:
    """
    Get list of available figure types for a model run.

    Args:
        model_type: The model type string
        extracted_data: Data extracted from model results
        clustering_result: Optional clustering results

    Returns:
        List of dicts with type, name, description, available
    """
    available = []

    for fig_type, (name, description, model_types) in FIGURE_METADATA.items():
        # Check if figure type is applicable to this model
        is_applicable = model_type in model_types

        # Check if required data is present
        has_data = _check_figure_data_available(fig_type, extracted_data, clustering_result)

        # Clustering figures are available if clustering was performed
        if fig_type in [FigureType.CLUSTERED_BIPLOT, FigureType.SILHOUETTE_ANALYSIS,
                        FigureType.CLUSTER_SIZES]:
            is_applicable = clustering_result is not None

        if fig_type == FigureType.DENDROGRAM:
            is_applicable = (clustering_result is not None and
                             clustering_result.get("linkage_matrix") is not None)

        available.append({
            "type": fig_type.value,
            "name": name,
            "description": description,
            "available": is_applicable and has_data
        })

    return available


def _check_figure_data_available(
    fig_type: FigureType,
    extracted_data: dict,
    clustering_result: Optional[dict] = None
) -> bool:
    """Check if the required data for a figure type is available."""
    product_columns = extracted_data.get("product_columns", [])

    if fig_type == FigureType.SIMILARITY_MATRIX:
        return extracted_data.get("similarity_matrix") is not None and len(product_columns) > 0

    if fig_type == FigureType.VARIANCE_EXPLAINED:
        var_exp = extracted_data.get("variance_explained")
        return var_exp is not None and len(var_exp) > 0

    if fig_type == FigureType.FACTOR_LOADINGS:
        loadings = extracted_data.get("loadings")
        return (loadings is not None and len(product_columns) > 0 and
                len(loadings.shape) == 2 and loadings.shape[0] == len(product_columns))

    if fig_type == FigureType.CLASS_PROFILES:
        return (extracted_data.get("item_probs") is not None and
                extracted_data.get("class_probs") is not None and
                len(product_columns) > 0)

    if fig_type == FigureType.BIPLOT:
        embeddings = extracted_data.get("product_embeddings")
        return (embeddings is not None and len(product_columns) > 0 and
                len(embeddings.shape) == 2 and embeddings.shape[1] >= 2)

    if fig_type == FigureType.TETRACHORIC:
        return extracted_data.get("tetra_corr") is not None and len(product_columns) > 0

    if fig_type == FigureType.ELBO_HISTORY:
        elbo = extracted_data.get("elbo_history")
        return elbo is not None and len(elbo) > 0

    if fig_type == FigureType.TOPIC_DISTRIBUTION:
        return extracted_data.get("topic_product_dist") is not None and len(product_columns) > 0

    if fig_type == FigureType.NETWORK_MATRIX:
        return extracted_data.get("adjacency_matrix") is not None and len(product_columns) > 0

    if fig_type == FigureType.DCM_COEFFICIENTS:
        return extracted_data.get("alpha") is not None and len(product_columns) > 0

    if fig_type == FigureType.CLUSTERED_BIPLOT:
        return (clustering_result is not None and
                clustering_result.get("labels") is not None and
                extracted_data.get("product_embeddings") is not None)

    if fig_type == FigureType.SILHOUETTE_ANALYSIS:
        return (clustering_result is not None and
                clustering_result.get("silhouette_scores") is not None)

    if fig_type == FigureType.CLUSTER_SIZES:
        return (clustering_result is not None and
                clustering_result.get("labels") is not None)

    if fig_type == FigureType.DENDROGRAM:
        return (clustering_result is not None and
                clustering_result.get("linkage_matrix") is not None)

    if fig_type == FigureType.LCA_CLASS_BIPLOT:
        return (extracted_data.get("item_probs") is not None and
                extracted_data.get("class_probs") is not None and
                extracted_data.get("product_embeddings") is not None and
                len(product_columns) > 0)

    if fig_type == FigureType.NETWORK_GRAPH:
        return (extracted_data.get("communities") is not None and
                extracted_data.get("centrality_scores") is not None and
                extracted_data.get("product_embeddings") is not None and
                len(product_columns) > 0)

    if fig_type == FigureType.CENTRALITY_COMPARISON:
        return (extracted_data.get("centrality_scores") is not None and
                extracted_data.get("degree_centrality") is not None and
                extracted_data.get("betweenness_centrality") is not None and
                len(product_columns) > 0)

    if fig_type == FigureType.TOP_PRODUCTS_TOPIC:
        return (extracted_data.get("topic_product_dist") is not None and
                len(product_columns) > 0)

    if fig_type == FigureType.INTERTOPIC_DISTANCE_MAP:
        tpd = extracted_data.get("topic_product_dist")
        return (tpd is not None and len(tpd.shape) == 2 and
                tpd.shape[0] >= 2 and len(product_columns) > 0)

    if fig_type == FigureType.CLOSEST_COMPETITORS:
        return (extracted_data.get("similarity_matrix") is not None and
                len(product_columns) > 1)

    if fig_type == FigureType.MARKET_MAP:
        emb = extracted_data.get("product_embeddings")
        return (emb is not None and len(emb.shape) == 2 and
                emb.shape[1] >= 2 and len(product_columns) > 0)

    if fig_type == FigureType.PRODUCT_SCORECARD:
        return (extracted_data.get("similarity_matrix") is not None and
                len(product_columns) > 1)

    if fig_type == FigureType.PRODUCT_NEIGHBORHOOD:
        return (extracted_data.get("similarity_matrix") is not None and
                len(product_columns) > 2)

    if fig_type == FigureType.MARKET_SEGMENTS:
        loadings = extracted_data.get("loadings")
        return (loadings is not None and len(loadings.shape) == 2 and
                loadings.shape[1] >= 1 and len(product_columns) > 0)

    return False


def generate_figure(
    fig_type: FigureType,
    extracted_data: dict,
    config: Optional[dict] = None,
    clustering_result: Optional[dict] = None
) -> go.Figure:
    """
    Generate a single Plotly figure.

    Args:
        fig_type: Type of figure to generate
        extracted_data: Data extracted from model results
        config: Optional figure-specific configuration
        clustering_result: Optional clustering results

    Returns:
        Plotly Figure object
    """
    config = config or {}
    product_columns = extracted_data.get("product_columns", [])

    if fig_type == FigureType.SIMILARITY_MATRIX:
        return _generate_similarity_matrix(extracted_data, product_columns, config)

    if fig_type == FigureType.VARIANCE_EXPLAINED:
        return _generate_variance_explained(extracted_data, config)

    if fig_type == FigureType.FACTOR_LOADINGS:
        return _generate_factor_loadings(extracted_data, product_columns, config)

    if fig_type == FigureType.CLASS_PROFILES:
        return _generate_class_profiles(extracted_data, product_columns, config)

    if fig_type == FigureType.BIPLOT:
        return _generate_biplot(extracted_data, product_columns, config)

    if fig_type == FigureType.TETRACHORIC:
        return _generate_tetrachoric(extracted_data, product_columns, config)

    if fig_type == FigureType.ELBO_HISTORY:
        return _generate_elbo_history(extracted_data, config)

    if fig_type == FigureType.TOPIC_DISTRIBUTION:
        return _generate_topic_distribution(extracted_data, product_columns, config)

    if fig_type == FigureType.NETWORK_MATRIX:
        return _generate_network_matrix(extracted_data, product_columns, config)

    if fig_type == FigureType.DCM_COEFFICIENTS:
        return _generate_dcm_coefficients(extracted_data, product_columns, config)

    if fig_type == FigureType.CLUSTERED_BIPLOT:
        return _generate_clustered_biplot(extracted_data, product_columns, clustering_result, config)

    if fig_type == FigureType.SILHOUETTE_ANALYSIS:
        return _generate_silhouette_analysis(clustering_result, config)

    if fig_type == FigureType.CLUSTER_SIZES:
        return _generate_cluster_sizes(clustering_result, config)

    if fig_type == FigureType.DENDROGRAM:
        return _generate_dendrogram(extracted_data, product_columns, clustering_result, config)

    if fig_type == FigureType.LCA_CLASS_BIPLOT:
        return _generate_lca_class_biplot(extracted_data, product_columns, config)

    if fig_type == FigureType.NETWORK_GRAPH:
        return _generate_network_graph(extracted_data, product_columns, config)

    if fig_type == FigureType.CENTRALITY_COMPARISON:
        return _generate_centrality_comparison(extracted_data, product_columns, config)

    if fig_type == FigureType.TOP_PRODUCTS_TOPIC:
        return _generate_top_products_topic(extracted_data, product_columns, config)

    if fig_type == FigureType.INTERTOPIC_DISTANCE_MAP:
        return _generate_intertopic_distance_map(extracted_data, product_columns, config)

    if fig_type == FigureType.CLOSEST_COMPETITORS:
        return _generate_closest_competitors(extracted_data, product_columns, config)

    if fig_type == FigureType.MARKET_MAP:
        return _generate_market_map(extracted_data, product_columns, config)

    if fig_type == FigureType.PRODUCT_SCORECARD:
        return _generate_product_scorecard(extracted_data, product_columns, config)

    if fig_type == FigureType.PRODUCT_NEIGHBORHOOD:
        return _generate_product_neighborhood(extracted_data, product_columns, config)

    if fig_type == FigureType.MARKET_SEGMENTS:
        return _generate_market_segments(extracted_data, product_columns, config)

    raise ValueError(f"Unknown figure type: {fig_type}")


# =============================================================================
# Individual figure generation functions
# =============================================================================

def _generate_similarity_matrix(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate similarity matrix heatmap."""
    similarity = extracted_data.get("similarity_matrix")

    fig = go.Figure(data=go.Heatmap(
        z=_to_list(similarity),
        x=product_columns,
        y=product_columns,
        colorscale=config.get("colorscale", "RdBu_r"),
        zmid=0
    ))
    fig.update_layout(
        title=config.get("title", "Product Similarity Matrix"),
        xaxis_title="Product",
        yaxis_title="Product",
        height=max(500, len(product_columns) * 15),
        width=max(600, len(product_columns) * 15)
    )
    return fig


def _generate_variance_explained(extracted_data: dict, config: dict) -> go.Figure:
    """Generate variance explained bar chart."""
    var_explained = extracted_data.get("variance_explained")
    n_comp = len(var_explained)
    var_list = _to_list(var_explained)
    cumulative = _to_list(np.cumsum(var_explained))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Comp {i+1}" for i in range(n_comp)],
        y=var_list,
        name="Individual",
        marker_color=config.get("bar_color", "#667eea")
    ))
    fig.add_trace(go.Scatter(
        x=[f"Comp {i+1}" for i in range(n_comp)],
        y=cumulative,
        name="Cumulative",
        mode="lines+markers",
        marker_color=config.get("line_color", "#764ba2")
    ))
    fig.update_layout(
        title=config.get("title", "Variance Explained"),
        xaxis_title="Component",
        yaxis_title="Variance Explained (%)",
        height=config.get("height", 400),
        showlegend=True
    )
    return fig


def _generate_factor_loadings(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate factor loadings heatmap."""
    loadings = extracted_data.get("loadings")
    n_factors = loadings.shape[1]
    model_type = extracted_data.get("model_type", "")

    # Use "Class" labels for LCA models, "Factor" for others
    if model_type in ["lca", "lca_covariates"]:
        factor_names = [f"Class {i+1}" for i in range(n_factors)]
        default_title = "Class Loadings"
        x_title = "Class"
    else:
        factor_names = [f"Factor {i+1}" for i in range(n_factors)]
        default_title = "Factor Loadings"
        x_title = "Factor"

    fig = go.Figure(data=go.Heatmap(
        z=_to_list(loadings),
        x=factor_names,
        y=product_columns,
        colorscale=config.get("colorscale", "RdBu_r"),
        zmid=0
    ))
    fig.update_layout(
        title=config.get("title", default_title),
        xaxis_title=x_title,
        yaxis_title="Product",
        height=max(400, len(product_columns) * 20)
    )
    return fig


def _generate_class_profiles(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate LCA class profiles bar chart."""
    item_probs = extracted_data.get("item_probs")
    class_probs = extracted_data.get("class_probs")
    n_classes = len(class_probs)

    fig = go.Figure()
    colors = config.get("colors", [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
    ])

    for c in range(n_classes):
        fig.add_trace(go.Bar(
            x=product_columns,
            y=_to_list(item_probs[c]),
            name=f"Class {c+1} ({float(class_probs[c])*100:.1f}%)",
            marker_color=colors[c % len(colors)]
        ))

    fig.update_layout(
        title=config.get("title", "LCA Class Profiles"),
        xaxis_title="Product",
        yaxis_title="Purchase Probability",
        barmode="group",
        height=config.get("height", 500)
    )
    return fig


def _generate_biplot(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate biplot with dimension selector."""
    product_embeddings = extracted_data.get("product_embeddings")
    n_dims = product_embeddings.shape[1]

    fig = go.Figure()

    # Add all dimension combinations as separate traces
    dim_pairs = []
    max_dims = config.get("max_dims", 5)
    for i in range(min(n_dims, max_dims)):
        for j in range(i + 1, min(n_dims, max_dims)):
            dim_pairs.append((i, j))

    for idx, (dim_x, dim_y) in enumerate(dim_pairs):
        visible = idx == 0
        fig.add_trace(go.Scatter(
            x=_to_list(product_embeddings[:, dim_x]),
            y=_to_list(product_embeddings[:, dim_y]),
            mode="markers+text",
            text=product_columns,
            textposition="top center",
            marker=dict(size=10, color=config.get("marker_color", "#667eea")),
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
        title=config.get("title", "Product Space (Biplot)"),
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=config.get("height", 600),
        # Dropdown in TOP-RIGHT
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ] if len(dim_pairs) > 1 else []
    )
    return fig


def _generate_tetrachoric(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate tetrachoric correlation matrix."""
    tetra_corr = extracted_data.get("tetra_corr")

    fig = go.Figure(data=go.Heatmap(
        z=_to_list(tetra_corr),
        x=product_columns,
        y=product_columns,
        colorscale=config.get("colorscale", "RdBu_r"),
        zmid=0
    ))
    fig.update_layout(
        title=config.get("title", "Tetrachoric Correlation Matrix"),
        xaxis_title="Product",
        yaxis_title="Product",
        height=max(500, len(product_columns) * 15)
    )
    return fig


def _generate_elbo_history(extracted_data: dict, config: dict) -> go.Figure:
    """Generate ELBO convergence plot."""
    elbo_history = extracted_data.get("elbo_history")

    fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(elbo_history) + 1)),
        y=elbo_history,
        mode="lines",
        line=dict(color=config.get("line_color", "#667eea"))
    ))
    fig.update_layout(
        title=config.get("title", "ELBO Convergence"),
        xaxis_title="Iteration",
        yaxis_title="ELBO",
        height=config.get("height", 400)
    )
    return fig


def _generate_topic_distribution(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate LDA topic distribution heatmap."""
    topic_dist = extracted_data.get("topic_product_dist")
    n_topics = topic_dist.shape[0]
    topic_names = [f"Topic {i+1}" for i in range(n_topics)]

    fig = go.Figure(data=go.Heatmap(
        z=_to_list(topic_dist),
        x=product_columns,
        y=topic_names,
        colorscale=config.get("colorscale", "Viridis")
    ))
    fig.update_layout(
        title=config.get("title", "Topic-Product Distribution"),
        xaxis_title="Product",
        yaxis_title="Topic",
        height=max(300, n_topics * 50)
    )
    return fig


def _generate_network_matrix(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate network adjacency matrix heatmap."""
    adj_matrix = extracted_data.get("adjacency_matrix")

    fig = go.Figure(data=go.Heatmap(
        z=_to_list(adj_matrix),
        x=product_columns,
        y=product_columns,
        colorscale=config.get("colorscale", "Blues")
    ))
    fig.update_layout(
        title=config.get("title", "Product Co-Purchase Network"),
        xaxis_title="Product",
        yaxis_title="Product",
        height=max(500, len(product_columns) * 15)
    )
    return fig


def _generate_dcm_coefficients(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """Generate DCM product intercepts bar chart."""
    alpha = extracted_data.get("alpha")
    alpha_std = extracted_data.get("alpha_std")

    # Sort by alpha value
    sorted_idx = np.argsort(alpha)[::-1]
    sorted_products = [product_columns[i] for i in sorted_idx]
    sorted_alpha = _to_list(alpha[sorted_idx])

    fig = go.Figure()

    if alpha_std is not None:
        sorted_std = _to_list(1.96 * alpha_std[sorted_idx])
        fig.add_trace(go.Bar(
            x=sorted_products,
            y=sorted_alpha,
            error_y=dict(type='data', array=sorted_std, visible=True),
            marker_color=config.get("bar_color", "#667eea")
        ))
    else:
        fig.add_trace(go.Bar(
            x=sorted_products,
            y=sorted_alpha,
            marker_color=config.get("bar_color", "#667eea")
        ))

    fig.update_layout(
        title=config.get("title", "DCM Product Intercepts (with 95% CI)"),
        xaxis_title="Product",
        yaxis_title="Intercept (α)",
        height=config.get("height", 500)
    )
    return fig


def _generate_clustered_biplot(
    extracted_data: dict,
    product_columns: list[str],
    clustering_result: dict,
    config: dict
) -> go.Figure:
    """Generate biplot colored by cluster assignment."""
    import plotly.express as px

    product_embeddings = extracted_data.get("product_embeddings")
    labels = clustering_result.get("labels")
    n_clusters = clustering_result.get("n_clusters", max(labels) + 1)
    n_dims = product_embeddings.shape[1]
    show_legend = config.get("show_legend", True)

    # Color palette
    colors = px.colors.qualitative.Set1[:n_clusters] if n_clusters <= 9 else px.colors.qualitative.Alphabet[:n_clusters]

    fig = go.Figure()

    # Add all dimension combinations
    dim_pairs = []
    max_dims = config.get("max_dims", 5)
    for i in range(min(n_dims, max_dims)):
        for j in range(i + 1, min(n_dims, max_dims)):
            dim_pairs.append((i, j))

    for pair_idx, (dim_x, dim_y) in enumerate(dim_pairs):
        visible = pair_idx == 0

        for cluster_id in range(n_clusters):
            cluster_mask = [l == cluster_id for l in labels]
            cluster_x = [product_embeddings[i, dim_x] for i, m in enumerate(cluster_mask) if m]
            cluster_y = [product_embeddings[i, dim_y] for i, m in enumerate(cluster_mask) if m]
            cluster_text = [product_columns[i] for i, m in enumerate(cluster_mask) if m]

            fig.add_trace(go.Scatter(
                x=_to_list(cluster_x),
                y=_to_list(cluster_y),
                mode="markers+text",
                text=cluster_text,
                textposition="top center",
                marker=dict(size=12, color=colors[cluster_id % len(colors)]),
                name=f"Cluster {cluster_id + 1}",
                legendgroup=f"cluster_{cluster_id}",
                # Show legend for first pair's traces, dropdown will toggle for other pairs
                showlegend=(pair_idx == 0 and show_legend),
                visible=visible
            ))

    # Create dropdown buttons that also update showlegend for proper legend display
    buttons = []
    traces_per_pair = n_clusters
    for pair_idx, (dim_x, dim_y) in enumerate(dim_pairs):
        visibility = []
        showlegend_values = []
        for p_idx in range(len(dim_pairs)):
            for _ in range(traces_per_pair):
                visibility.append(p_idx == pair_idx)
                # Show legend for the visible traces only
                showlegend_values.append(p_idx == pair_idx and show_legend)
        buttons.append(dict(
            label=f"Dim {dim_x+1} vs Dim {dim_y+1}",
            method="update",
            args=[
                {"visible": visibility, "showlegend": showlegend_values},
                {"xaxis.title": f"Dimension {dim_x+1}",
                 "yaxis.title": f"Dimension {dim_y+1}"}
            ]
        ))

    fig.update_layout(
        title=config.get("title", "Clustered Product Space"),
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=config.get("height", 600),
        showlegend=show_legend,
        # Legend to the RIGHT of the plot (outside)
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        # Add right margin to make room for legend
        margin=dict(r=150),
        # Dropdown in TOP-RIGHT
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ] if len(dim_pairs) > 1 else []
    )
    return fig


def _generate_silhouette_analysis(clustering_result: dict, config: dict) -> go.Figure:
    """Generate silhouette score analysis plot."""
    silhouette_scores = clustering_result.get("silhouette_scores")
    k_range = clustering_result.get("k_range")
    optimal_k = clustering_result.get("optimal_k")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_range,
        y=silhouette_scores,
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color=config.get("line_color", "#667eea")),
        marker=dict(size=8)
    ))

    # Mark optimal k
    if optimal_k is not None:
        optimal_idx = k_range.index(optimal_k) if optimal_k in k_range else None
        if optimal_idx is not None:
            fig.add_trace(go.Scatter(
                x=[optimal_k],
                y=[silhouette_scores[optimal_idx]],
                mode="markers",
                name=f"Optimal k={optimal_k}",
                marker=dict(size=15, color="red", symbol="star")
            ))

    fig.update_layout(
        title=config.get("title", "Silhouette Analysis"),
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=config.get("height", 400)
    )
    return fig


def _generate_cluster_sizes(clustering_result: dict, config: dict) -> go.Figure:
    """Generate cluster sizes bar chart."""
    import plotly.express as px
    from collections import Counter

    labels = clustering_result.get("labels")
    n_clusters = clustering_result.get("n_clusters", max(labels) + 1)

    # Count products per cluster
    label_counts = Counter(labels)
    cluster_ids = list(range(n_clusters))
    counts = [label_counts.get(i, 0) for i in cluster_ids]

    colors = px.colors.qualitative.Set1[:n_clusters] if n_clusters <= 9 else px.colors.qualitative.Alphabet[:n_clusters]

    fig = go.Figure(data=go.Bar(
        x=[f"Cluster {i+1}" for i in cluster_ids],
        y=counts,
        marker_color=colors[:n_clusters],
        text=counts,
        textposition="outside"
    ))

    fig.update_layout(
        title=config.get("title", "Cluster Sizes"),
        xaxis_title="Cluster",
        yaxis_title="Number of Products",
        height=config.get("height", 400)
    )
    return fig


def _generate_dendrogram(
    extracted_data: dict,
    product_columns: list[str],
    clustering_result: dict,
    config: dict
) -> go.Figure:
    """Generate hierarchical clustering dendrogram."""
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    linkage_matrix = np.array(clustering_result.get("linkage_matrix"))

    # Create matplotlib figure for dendrogram
    plt.figure(figsize=(12, 6))
    dendro = dendrogram(
        linkage_matrix,
        labels=product_columns,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=0
    )
    plt.close()

    # Convert to Plotly
    fig = go.Figure()

    # Add the dendrogram lines
    icoord = dendro['icoord']
    dcoord = dendro['dcoord']

    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color=config.get("line_color", "#667eea"), width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))

    # Add product labels
    leaves = dendro['leaves']
    leaf_labels = [product_columns[i] for i in leaves]

    fig.update_layout(
        title=config.get("title", "Hierarchical Clustering Dendrogram"),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(5, 10 * len(leaf_labels), 10)),
            ticktext=leaf_labels,
            tickangle=45
        ),
        yaxis_title="Distance",
        height=config.get("height", 500),
        margin=dict(b=150)
    )

    return fig


def _generate_lca_class_biplot(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Generate biplot for LCA with products colored by class dominance.

    Products are colored by which class they have highest purchase probability.
    Supports dimension selection and class filtering via config.
    """
    import plotly.express as px

    product_embeddings = extracted_data.get("product_embeddings")
    item_probs = extracted_data.get("item_probs")  # (n_classes, n_items)
    class_probs = extracted_data.get("class_probs")  # (n_classes,)

    # Determine product class assignments (which class each product is most associated with)
    product_class_assignment = np.argmax(item_probs, axis=0)  # (n_products,)
    n_classes = len(class_probs)
    n_dims = product_embeddings.shape[1]

    # Config options
    selected_classes = config.get("selected_classes", list(range(n_classes)))
    max_dims = config.get("max_dims", 5)
    show_legend = config.get("show_legend", True)

    # Color palette
    colors = px.colors.qualitative.Set1[:n_classes] if n_classes <= 9 else px.colors.qualitative.Alphabet[:n_classes]

    fig = go.Figure()

    # Generate dimension pairs
    dim_pairs = []
    for i in range(min(n_dims, max_dims)):
        for j in range(i + 1, min(n_dims, max_dims)):
            dim_pairs.append((i, j))

    # Create traces: one per (dimension_pair, class) combination
    for pair_idx, (dim_x, dim_y) in enumerate(dim_pairs):
        visible = pair_idx == 0

        for class_id in range(n_classes):
            # Get products assigned to this class
            products_in_class = [i for i, c in enumerate(product_class_assignment) if c == class_id]

            if not products_in_class:
                # Add empty trace to maintain consistent trace count
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode="markers+text",
                    text=[],
                    textposition="top center",
                    marker=dict(size=12, color=colors[class_id % len(colors)]),
                    name=f"Class {class_id + 1} ({float(class_probs[class_id])*100:.1f}%)",
                    legendgroup=f"class_{class_id}",
                    showlegend=(pair_idx == 0 and show_legend),
                    visible=visible
                ))
                continue

            class_x = [product_embeddings[i, dim_x] for i in products_in_class]
            class_y = [product_embeddings[i, dim_y] for i in products_in_class]
            class_text = [product_columns[i] for i in products_in_class]

            fig.add_trace(go.Scatter(
                x=_to_list(class_x),
                y=_to_list(class_y),
                mode="markers+text",
                text=class_text,
                textposition="top center",
                marker=dict(size=12, color=colors[class_id % len(colors)]),
                name=f"Class {class_id + 1} ({float(class_probs[class_id])*100:.1f}%)",
                legendgroup=f"class_{class_id}",
                showlegend=(pair_idx == 0 and show_legend),
                visible=visible
            ))

    # Dimension selector dropdown with showlegend toggling
    buttons = []
    traces_per_pair = n_classes

    for pair_idx, (dim_x, dim_y) in enumerate(dim_pairs):
        visibility = []
        showlegend_values = []
        for p_idx in range(len(dim_pairs)):
            for _ in range(traces_per_pair):
                visibility.append(p_idx == pair_idx)
                # Show legend for the visible traces only
                showlegend_values.append(p_idx == pair_idx and show_legend)

        buttons.append(dict(
            label=f"Class {dim_x+1} vs Class {dim_y+1}",
            method="update",
            args=[
                {"visible": visibility, "showlegend": showlegend_values},
                {"xaxis.title": f"Class {dim_x+1} Profile",
                 "yaxis.title": f"Class {dim_y+1} Profile"}
            ]
        ))

    fig.update_layout(
        title=config.get("title", "Product Space by Class Assignment"),
        xaxis_title="Class 1 Profile",
        yaxis_title="Class 2 Profile",
        height=config.get("height", 600),
        showlegend=show_legend,
        # Legend to the RIGHT of the plot (outside)
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        # Add right margin to make room for legend
        margin=dict(r=150),
        # Dropdown in TOP-RIGHT
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ] if len(dim_pairs) > 1 else []
    )

    return fig


def _generate_network_graph(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Generate interactive network graph with community coloring and centrality sizing.

    Uses product embeddings (first 2 dimensions) for node layout,
    colors nodes by community assignment, and sizes by centrality.
    """
    import plotly.express as px

    product_embeddings = extracted_data.get("product_embeddings")
    communities = extracted_data.get("communities")
    centrality_scores = extracted_data.get("centrality_scores")
    edge_list = extracted_data.get("edge_list")

    n_products = len(product_columns)

    # Use first 2 dims of embeddings for layout
    x_pos = product_embeddings[:, 0] if product_embeddings.shape[1] >= 1 else np.zeros(n_products)
    y_pos = product_embeddings[:, 1] if product_embeddings.shape[1] >= 2 else np.zeros(n_products)

    # Normalize centrality for sizing (min 8, max 30)
    cent = np.array(centrality_scores)
    if cent.max() > cent.min():
        size_norm = (cent - cent.min()) / (cent.max() - cent.min())
    else:
        size_norm = np.ones(n_products) * 0.5
    node_sizes = 8 + size_norm * 22

    # Community info
    n_communities = len(set(communities))
    colors = (px.colors.qualitative.Set1[:n_communities]
              if n_communities <= 9
              else px.colors.qualitative.Alphabet[:n_communities])

    fig = go.Figure()

    # Draw edges first (behind nodes)
    if edge_list is not None:
        for edge in edge_list:
            src, tgt = edge[0], edge[1]
            weight = edge[2] if len(edge) > 2 else 0.5
            # Ensure indices are valid
            if src < n_products and tgt < n_products:
                fig.add_trace(go.Scatter(
                    x=[float(x_pos[src]), float(x_pos[tgt]), None],
                    y=[float(y_pos[src]), float(y_pos[tgt]), None],
                    mode="lines",
                    line=dict(
                        width=max(0.5, float(weight) * 2),
                        color="rgba(150,150,150,0.3)"
                    ),
                    hoverinfo="skip",
                    showlegend=False
                ))

    # Draw nodes by community
    for comm_id in range(n_communities):
        mask = [i for i, c in enumerate(communities) if c == comm_id]
        if not mask:
            continue

        hover_text = [
            f"{product_columns[i]}<br>Community: {comm_id + 1}<br>"
            f"Centrality: {float(centrality_scores[i]):.3f}"
            for i in mask
        ]

        fig.add_trace(go.Scatter(
            x=[float(x_pos[i]) for i in mask],
            y=[float(y_pos[i]) for i in mask],
            mode="markers+text",
            text=[product_columns[i] for i in mask],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                size=[float(node_sizes[i]) for i in mask],
                color=colors[comm_id % len(colors)],
                line=dict(width=1, color="white")
            ),
            name=f"Community {comm_id + 1}",
            legendgroup=f"comm_{comm_id}"
        ))

    fig.update_layout(
        title=config.get("title", "Product Network Graph"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        height=config.get("height", 650),
        showlegend=True,
        legend=dict(
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=150),
        plot_bgcolor="white"
    )
    return fig


def _generate_centrality_comparison(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Generate horizontal bar chart comparing eigenvector, degree, and betweenness centrality.

    Products sorted by eigenvector centrality by default, with dropdown to change sort.
    """
    eigenvector = np.array(extracted_data.get("centrality_scores"))
    degree = np.array(extracted_data.get("degree_centrality"))
    betweenness = np.array(extracted_data.get("betweenness_centrality"))

    n_products = len(product_columns)

    # Define centrality measures
    measures = [
        ("Eigenvector", eigenvector, "#667eea"),
        ("Degree", degree, "#764ba2"),
        ("Betweenness", betweenness, "#f5576c"),
    ]

    fig = go.Figure()

    # Create traces for each sort order (one set per sort measure)
    for sort_idx, (sort_name, sort_vals, _) in enumerate(measures):
        visible = sort_idx == 0  # Default: sorted by eigenvector

        # Sort products by the selected measure
        sorted_indices = np.argsort(sort_vals)  # ascending for horizontal bar

        sorted_products = [product_columns[i] for i in sorted_indices]

        for measure_name, measure_vals, color in measures:
            sorted_vals = [float(measure_vals[i]) for i in sorted_indices]

            fig.add_trace(go.Bar(
                y=sorted_products,
                x=sorted_vals,
                orientation="h",
                name=measure_name,
                marker_color=color,
                visible=visible,
                legendgroup=measure_name,
                showlegend=(sort_idx == 0),  # Only show legend for first group
            ))

    # Dropdown to select sort order
    traces_per_sort = 3  # 3 measures per sort
    buttons = []
    for sort_idx, (sort_name, _, _) in enumerate(measures):
        visibility = []
        showlegend_vals = []
        for s_idx in range(len(measures)):
            for _ in range(traces_per_sort):
                visibility.append(s_idx == sort_idx)
                showlegend_vals.append(s_idx == sort_idx)
        buttons.append(dict(
            label=f"Sort by {sort_name}",
            method="update",
            args=[{"visible": visibility, "showlegend": showlegend_vals}]
        ))

    fig.update_layout(
        title=config.get("title", "Centrality Comparison"),
        xaxis_title="Centrality Score",
        yaxis_title="Product",
        height=max(400, n_products * 25),
        barmode="group",
        showlegend=True,
        legend=dict(
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=150, l=max(80, max(len(p) for p in product_columns) * 7)),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ]
    )
    return fig


def _generate_top_products_topic(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Generate horizontal bar chart showing top products per topic with topic dropdown.
    """
    topic_dist = extracted_data.get("topic_product_dist")
    n_topics = topic_dist.shape[0]
    n_top = config.get("n_top", 15)

    fig = go.Figure()

    for topic_idx in range(n_topics):
        visible = topic_idx == 0

        # Get top products for this topic
        probs = topic_dist[topic_idx]
        top_indices = np.argsort(probs)[::-1][:n_top]

        # Reverse for horizontal bar (top product at top)
        top_indices = top_indices[::-1]

        top_products = [product_columns[i] for i in top_indices]
        top_probs = [float(probs[i]) for i in top_indices]

        fig.add_trace(go.Bar(
            y=top_products,
            x=top_probs,
            orientation="h",
            marker_color="#667eea",
            name=f"Topic {topic_idx + 1}",
            visible=visible,
            text=[f"{p:.3f}" for p in top_probs],
            textposition="outside"
        ))

    # Dropdown for topic selection
    buttons = []
    for topic_idx in range(n_topics):
        visibility = [i == topic_idx for i in range(n_topics)]
        buttons.append(dict(
            label=f"Topic {topic_idx + 1}",
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Top {n_top} Products — Topic {topic_idx + 1}"}
            ]
        ))

    fig.update_layout(
        title=config.get("title", f"Top {n_top} Products — Topic 1"),
        xaxis_title="Probability",
        yaxis_title="Product",
        height=max(400, n_top * 28),
        showlegend=False,
        margin=dict(l=max(80, max(len(p) for p in product_columns) * 7)),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ] if n_topics > 1 else []
    )
    return fig


def _generate_intertopic_distance_map(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Generate a pyLDAvis-style intertopic distance map.

    Topics are positioned via MDS on Jensen-Shannon divergences,
    sized by prevalence, with hover showing top products per topic.
    """
    import plotly.express as px
    from scipy.spatial.distance import jensenshannon
    from sklearn.manifold import MDS

    topic_dist = extracted_data.get("topic_product_dist")  # (n_topics, n_products)
    var_explained = extracted_data.get("variance_explained")  # (n_topics,) percentages
    n_topics = topic_dist.shape[0]
    n_top_hover = config.get("n_top_hover", 5)

    # --- 1. Compute pairwise Jensen-Shannon distance matrix ---
    dist_matrix = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            d = jensenshannon(topic_dist[i], topic_dist[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # --- 2. MDS to 2D ---
    if n_topics == 2:
        # MDS with 2 points: place them on x-axis separated by their distance
        coords = np.array([[-dist_matrix[0, 1] / 2, 0],
                           [dist_matrix[0, 1] / 2, 0]])
    else:
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=42,
            normalized_stress='auto'
        )
        coords = mds.fit_transform(dist_matrix)  # (n_topics, 2)

    # --- 3. Topic prevalence for circle sizing ---
    if var_explained is not None and len(var_explained) == n_topics:
        prevalence = np.array(var_explained)
    else:
        # Fallback: equal prevalence
        prevalence = np.ones(n_topics) * (100.0 / n_topics)

    max_prev = prevalence.max() if prevalence.max() > 0 else 1.0
    marker_sizes = 20 + (prevalence / max_prev) * 60

    # --- 4. Build hover text with top products per topic ---
    hover_texts = []
    for t in range(n_topics):
        probs = topic_dist[t]
        top_idx = np.argsort(probs)[::-1][:n_top_hover]
        lines = [f"<b>Topic {t + 1}</b> ({prevalence[t]:.1f}%)", ""]
        for idx in top_idx:
            lines.append(f"{product_columns[idx]}: {probs[idx]:.3f}")
        hover_texts.append("<br>".join(lines))

    # --- 5. Color palette ---
    colors = (px.colors.qualitative.Set1[:n_topics]
              if n_topics <= 9
              else px.colors.qualitative.Alphabet[:n_topics])

    # --- 6. Create figure ---
    fig = go.Figure()

    for t in range(n_topics):
        fig.add_trace(go.Scatter(
            x=[float(coords[t, 0])],
            y=[float(coords[t, 1])],
            mode="markers+text",
            text=[f"Topic {t + 1}"],
            textposition="top center",
            textfont=dict(size=11, color=colors[t % len(colors)]),
            hovertext=hover_texts[t],
            hoverinfo="text",
            marker=dict(
                size=float(marker_sizes[t]),
                color=colors[t % len(colors)],
                opacity=0.7,
                line=dict(width=2, color="white")
            ),
            name=f"Topic {t + 1} ({prevalence[t]:.1f}%)",
            legendgroup=f"topic_{t}"
        ))

    fig.update_layout(
        title=config.get("title", "Intertopic Distance Map"),
        xaxis=dict(
            title="MDS Dimension 1",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)"
        ),
        yaxis=dict(
            title="MDS Dimension 2",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)"
        ),
        height=config.get("height", 600),
        showlegend=True,
        legend=dict(
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=180),
        plot_bgcolor="white"
    )
    return fig


# =============================================================================
# Stakeholder-friendly competitive landscape figures
# =============================================================================

def _generate_closest_competitors(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Horizontal bar chart ranking the most competitive product pairs.

    Extracts upper triangle of similarity matrix, sorts by strength,
    and shows top N pairs with business-friendly labels.
    """
    similarity = extracted_data.get("similarity_matrix")
    n = len(product_columns)
    n_top = config.get("n_top", 15)
    highlight = config.get("highlight_product", None)

    # Extract all unique pairs from upper triangle
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity[i, j])
            if highlight is not None:
                if product_columns[i] != highlight and product_columns[j] != highlight:
                    continue
            pairs.append((product_columns[i], product_columns[j], score))

    # Sort descending by score, take only positive pairs
    pairs = [p for p in pairs if p[2] > 0]
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:n_top]

    if not pairs:
        fig = go.Figure()
        fig.update_layout(title="Closest Competitors — No competitive pairs found")
        return fig

    # Reverse for horizontal bar (top pair at top of chart)
    pairs = pairs[::-1]
    total_pairs = n * (n - 1) // 2

    labels = [f"{a} ↔ {b}" for a, b, _ in pairs]
    scores = [s for _, _, s in pairs]
    hover = [
        f"<b>{a} ↔ {b}</b><br>"
        f"Similarity: {s:.1%}<br>"
        f"Rank: #{total_pairs - idx} of {total_pairs}"
        for idx, (a, b, s) in enumerate(pairs)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=scores,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale="Blues",
            cmin=0,
        ),
        text=[f"{s:.1%}" for s in scores],
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))

    max_label_len = max(len(l) for l in labels) if labels else 20
    title = f"Closest Competitors: {highlight}" if highlight else "Closest Competitors"

    fig.update_layout(
        title=config.get("title", title),
        xaxis_title="Competitive Similarity",
        yaxis_title="",
        height=max(400, len(pairs) * 32),
        margin=dict(l=max(120, max_label_len * 6)),
        xaxis=dict(tickformat=".0%"),
    )
    return fig


def _generate_market_map(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Competitive landscape scatter: products positioned by proximity,
    sized by importance, colored by product group.

    Business-friendly version of the biplot with no statistical labels.
    """
    import plotly.express as px

    embeddings = extracted_data.get("product_embeddings")
    similarity = extracted_data.get("similarity_matrix")
    loadings = extracted_data.get("loadings")
    n = len(product_columns)

    x_vals = embeddings[:, 0]
    y_vals = embeddings[:, 1]

    # Compute market importance: mean absolute similarity to all other products
    if similarity is not None:
        sim_abs = np.abs(similarity)
        np.fill_diagonal(sim_abs, 0)
        importance = sim_abs.mean(axis=1)
    else:
        importance = np.ones(n) * 0.5

    # Normalize to marker sizes [10, 40]
    imp_min, imp_max = importance.min(), importance.max()
    if imp_max > imp_min:
        size_norm = (importance - imp_min) / (imp_max - imp_min)
    else:
        size_norm = np.ones(n) * 0.5
    marker_sizes = 10 + size_norm * 30

    # Assign groups from loadings (argmax of absolute loadings per product)
    if loadings is not None and len(loadings.shape) == 2 and loadings.shape[1] >= 1:
        group_assignments = np.argmax(np.abs(loadings), axis=1)
        n_groups = int(group_assignments.max()) + 1
    else:
        group_assignments = np.zeros(n, dtype=int)
        n_groups = 1

    colors = (px.colors.qualitative.Set2[:n_groups]
              if n_groups <= 8
              else px.colors.qualitative.Alphabet[:n_groups])

    fig = go.Figure()

    for group_id in range(n_groups):
        mask = [i for i in range(n) if group_assignments[i] == group_id]
        if not mask:
            continue

        hover = [
            f"<b>{product_columns[i]}</b><br>"
            f"Group: {group_id + 1}<br>"
            f"Market Importance: {importance[i]:.1%}"
            for i in mask
        ]

        fig.add_trace(go.Scatter(
            x=[float(x_vals[i]) for i in mask],
            y=[float(y_vals[i]) for i in mask],
            mode="markers+text",
            text=[product_columns[i] for i in mask],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=[float(marker_sizes[i]) for i in mask],
                color=colors[group_id % len(colors)],
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            name=f"Group {group_id + 1}",
        ))

    fig.update_layout(
        title=config.get("title", "Market Map: Competitive Landscape"),
        xaxis=dict(
            title="", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            zeroline=False, showticklabels=False
        ),
        yaxis=dict(
            title="", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            zeroline=False, showticklabels=False
        ),
        height=config.get("height", 650),
        showlegend=True,
        legend=dict(title="Product Groups"),
        plot_bgcolor="white",
        annotations=[dict(
            text="Products closer together compete more directly. "
                 "Larger circles = more central to the market.",
            xref="paper", yref="paper", x=0.5, y=-0.06,
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )
    return fig


def _generate_product_scorecard(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Horizontal bar chart ranking products by market importance,
    colored by group, with hover showing top 3 competitors.
    """
    import plotly.express as px

    similarity = extracted_data.get("similarity_matrix")
    loadings = extracted_data.get("loadings")
    n = len(product_columns)

    # Compute importance: mean similarity to other products (excluding self)
    sim = np.array(similarity)
    np.fill_diagonal(sim, 0)
    importance = np.abs(sim).mean(axis=1)

    # Find top 3 competitors per product
    top3 = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        top_idx = np.argsort(row)[::-1][:3]
        top3[i] = [(product_columns[j], float(row[j])) for j in top_idx if row[j] > 0]

    # Group assignments from loadings
    if loadings is not None and len(loadings.shape) == 2 and loadings.shape[1] >= 1:
        group_assignments = np.argmax(np.abs(loadings), axis=1)
        n_groups = int(group_assignments.max()) + 1
    else:
        group_assignments = np.zeros(n, dtype=int)
        n_groups = 1

    colors = (px.colors.qualitative.Set2[:n_groups]
              if n_groups <= 8
              else px.colors.qualitative.Alphabet[:n_groups])

    # Sort by importance ascending (highest at top of horizontal bar)
    sorted_idx = np.argsort(importance)

    fig = go.Figure()

    for group_id in range(n_groups):
        mask = [i for i in sorted_idx if group_assignments[i] == group_id]
        if not mask:
            continue

        hover = []
        for i in mask:
            lines = [
                f"<b>{product_columns[i]}</b>",
                f"Market Importance: {importance[i]:.3f}",
                f"Group: {group_id + 1}",
                "",
                "<b>Top Competitors:</b>",
            ]
            for rank, (comp_name, comp_score) in enumerate(top3.get(i, []), 1):
                lines.append(f"  {rank}. {comp_name} ({comp_score:.1%})")
            if not top3.get(i):
                lines.append("  (none above zero)")
            hover.append("<br>".join(lines))

        fig.add_trace(go.Bar(
            y=[product_columns[i] for i in mask],
            x=[float(importance[i]) for i in mask],
            orientation="h",
            name=f"Group {group_id + 1}",
            marker_color=colors[group_id % len(colors)],
            hovertext=hover,
            hoverinfo="text",
        ))

    max_name = max(len(p) for p in product_columns) if product_columns else 10

    fig.update_layout(
        title=config.get("title", "Product Competitive Scorecard"),
        xaxis_title="Market Importance Score",
        yaxis_title="",
        barmode="relative",
        height=max(400, n * 24),
        margin=dict(l=max(100, max_name * 7)),
        showlegend=True,
        legend=dict(title="Product Groups"),
        yaxis=dict(
            categoryorder="array",
            categoryarray=[product_columns[i] for i in sorted_idx]
        ),
    )
    return fig


def _generate_product_neighborhood(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Radial target chart showing a focal product at center
    with closest competitors arranged by competitive distance.

    Dropdown to switch focal product (top 20 most important).
    """
    import plotly.express as px

    similarity = extracted_data.get("similarity_matrix")
    loadings = extracted_data.get("loadings")
    n = len(product_columns)
    n_neighbors = config.get("n_neighbors", 10)
    max_focal = config.get("max_focal_products", 20)

    # Compute importance for focal product selection
    sim = np.array(similarity)
    np.fill_diagonal(sim, 0)
    importance = np.abs(sim).mean(axis=1)

    # Group assignments
    if loadings is not None and len(loadings.shape) == 2 and loadings.shape[1] >= 1:
        group_assignments = np.argmax(np.abs(loadings), axis=1)
        n_groups = int(group_assignments.max()) + 1
    else:
        group_assignments = np.zeros(n, dtype=int)
        n_groups = 1

    colors = (px.colors.qualitative.Set2[:n_groups]
              if n_groups <= 8
              else px.colors.qualitative.Alphabet[:n_groups])

    # Select top focal products by importance
    focal_indices = np.argsort(importance)[::-1][:max_focal]

    fig = go.Figure()
    traces_per_focal = 0  # will count for first focal product

    for f_idx, focal in enumerate(focal_indices):
        visible = f_idx == 0

        # Get neighbors sorted by similarity
        row = sim[focal].copy()
        row[focal] = -np.inf
        neighbor_idx = np.argsort(row)[::-1][:n_neighbors]

        # Normalize similarity within focal row for radius scaling
        neighbor_sims = np.array([row[j] for j in neighbor_idx])
        max_sim = neighbor_sims.max() if len(neighbor_sims) > 0 and neighbor_sims.max() > 0 else 1.0

        # Connection lines from center to each neighbor
        for rank, j in enumerate(neighbor_idx):
            s = max(0, float(row[j]))
            normalized_s = s / max_sim if max_sim > 0 else 0
            radius = 1.0 - normalized_s * 0.85  # keep minimum distance from center
            angle = 2 * np.pi * rank / n_neighbors
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)

            fig.add_trace(go.Scatter(
                x=[0, x_pos], y=[0, y_pos],
                mode="lines",
                line=dict(width=max(1, normalized_s * 5), color="rgba(150,150,150,0.3)"),
                hoverinfo="skip", showlegend=False, visible=visible,
            ))

        # Neighbor markers (one trace per group for this focal product)
        for group_id in range(n_groups):
            group_mask = [
                (rank, j) for rank, j in enumerate(neighbor_idx)
                if group_assignments[j] == group_id
            ]
            if not group_mask:
                # Add empty trace to keep trace count consistent
                fig.add_trace(go.Scatter(
                    x=[], y=[], mode="markers", visible=visible,
                    showlegend=(f_idx == 0), name=f"Group {group_id + 1}",
                    legendgroup=f"group_{group_id}",
                ))
                continue

            gx, gy, gtext, ghover = [], [], [], []
            for rank, j in group_mask:
                s = max(0, float(row[j]))
                normalized_s = s / max_sim if max_sim > 0 else 0
                radius = 1.0 - normalized_s * 0.85
                angle = 2 * np.pi * rank / n_neighbors
                gx.append(radius * np.cos(angle))
                gy.append(radius * np.sin(angle))
                gtext.append(product_columns[j])
                ghover.append(
                    f"<b>{product_columns[j]}</b><br>"
                    f"Similarity: {s:.1%}<br>"
                    f"Group: {group_id + 1}<br>"
                    f"Rank: #{rank + 1}"
                )

            fig.add_trace(go.Scatter(
                x=gx, y=gy,
                mode="markers+text",
                text=gtext, textposition="top center", textfont=dict(size=9),
                hovertext=ghover, hoverinfo="text",
                marker=dict(size=14, color=colors[group_id % len(colors)],
                            line=dict(width=1, color="white")),
                name=f"Group {group_id + 1}",
                legendgroup=f"group_{group_id}",
                showlegend=(f_idx == 0),
                visible=visible,
            ))

        # Focal product star at center
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers+text",
            text=[product_columns[focal]],
            textposition="bottom center",
            textfont=dict(size=11, color="#f5576c"),
            marker=dict(size=20, color="#f5576c", symbol="star",
                        line=dict(width=2, color="white")),
            name=product_columns[focal],
            showlegend=False, visible=visible,
            hovertext=f"<b>{product_columns[focal]}</b><br>(Selected Product)",
            hoverinfo="text",
        ))

        # Count traces for first focal product
        if f_idx == 0:
            traces_per_focal = len(fig.data)

    # If we have more than one focal product, compute traces per focal
    if len(focal_indices) > 1:
        traces_per_focal = len(fig.data) // len(focal_indices)

    # Concentric guide circles (always visible, added after all focal traces)
    for threshold, label in [(0.3, "Very Close"), (0.6, "Moderate"), (0.9, "Distant")]:
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=(threshold * np.cos(theta)).tolist(),
            y=(threshold * np.sin(theta)).tolist(),
            mode="lines",
            line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # Dropdown to switch focal product
    n_guide_traces = 3  # guide circles always visible
    n_focal_traces = len(fig.data) - n_guide_traces
    buttons = []
    for f_idx, focal in enumerate(focal_indices):
        vis = [False] * n_focal_traces + [True] * n_guide_traces
        start = f_idx * traces_per_focal
        end = start + traces_per_focal
        for t in range(start, min(end, n_focal_traces)):
            vis[t] = True
        buttons.append(dict(
            label=product_columns[focal],
            method="update",
            args=[{"visible": vis},
                  {"title": f"Competitive Neighborhood: {product_columns[focal]}"}]
        ))

    fig.update_layout(
        title=config.get("title",
                         f"Competitive Neighborhood: {product_columns[focal_indices[0]]}"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="y", range=[-1.3, 1.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.3, 1.3]),
        height=config.get("height", 650),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(title="Product Groups", yanchor="top", y=1.0,
                    xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(r=150),
        updatemenus=[
            dict(
                active=0, buttons=buttons,
                direction="down", showactive=True,
                x=1.0, xanchor="right", y=1.15, yanchor="top"
            )
        ] if len(focal_indices) > 1 else [],
        annotations=[dict(
            text="Distance from center = competitive distance. "
                 "Closer = stronger competition.",
            xref="paper", yref="paper", x=0.5, y=-0.06,
            showarrow=False, font=dict(size=10, color="gray"),
        )],
    )
    return fig


def _generate_market_segments(
    extracted_data: dict,
    product_columns: list[str],
    config: dict
) -> go.Figure:
    """
    Treemap showing natural product groupings sized by segment importance.
    """
    import plotly.express as px

    loadings = extracted_data.get("loadings")
    var_explained = extracted_data.get("variance_explained")
    n = len(product_columns)
    n_groups = loadings.shape[1]

    # Assign products to dominant group
    group_assignments = np.argmax(np.abs(loadings), axis=1)

    colors = (px.colors.qualitative.Set2[:n_groups]
              if n_groups <= 8
              else px.colors.qualitative.Alphabet[:n_groups])

    # Build treemap data
    labels = ["Market"]
    parents = [""]
    values = [0]
    marker_colors = ["#ffffff"]
    hover_texts = ["Full market structure"]

    for group_id in range(n_groups):
        group_products = [i for i in range(n) if group_assignments[i] == group_id]
        group_name = f"Segment {group_id + 1}"

        if var_explained is not None and group_id < len(var_explained):
            group_weight = float(var_explained[group_id])
            share_text = f"{group_weight:.1f}%"
        else:
            group_weight = len(group_products)
            share_text = f"{len(group_products)} products"

        labels.append(group_name)
        parents.append("Market")
        values.append(0)  # auto-computed from children
        marker_colors.append(colors[group_id % len(colors)])
        hover_texts.append(
            f"<b>{group_name}</b><br>"
            f"Products: {len(group_products)}<br>"
            f"Share: {share_text}"
        )

        for prod_idx in group_products:
            fit_score = float(np.abs(loadings[prod_idx, group_id]))
            labels.append(product_columns[prod_idx])
            parents.append(group_name)
            values.append(max(fit_score, 0.01))  # min for visibility
            marker_colors.append(colors[group_id % len(colors)])
            hover_texts.append(
                f"<b>{product_columns[prod_idx]}</b><br>"
                f"Segment: {group_name}<br>"
                f"Fit Score: {fit_score:.3f}"
            )

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=marker_colors, line=dict(width=2, color="white")),
        hovertext=hover_texts,
        hoverinfo="text",
        textinfo="label",
        textfont=dict(size=12),
        branchvalues="remainder",
    ))

    fig.update_layout(
        title=config.get("title", "Market Segments: Natural Product Groupings"),
        height=config.get("height", 600),
        margin=dict(t=50, l=10, r=10, b=30),
    )
    return fig
