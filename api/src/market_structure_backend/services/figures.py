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
         "nmf", "mca", "dcm", "lda", "network"]
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
    factor_names = [f"Factor {i+1}" for i in range(n_factors)]

    fig = go.Figure(data=go.Heatmap(
        z=_to_list(loadings),
        x=factor_names,
        y=product_columns,
        colorscale=config.get("colorscale", "RdBu_r"),
        zmid=0
    ))
    fig.update_layout(
        title=config.get("title", "Factor Loadings"),
        xaxis_title="Factor",
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
        ]
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
