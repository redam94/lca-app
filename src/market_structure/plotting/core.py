"""
Core plotting functions for Latent Structure Analysis.

This module provides general-purpose visualizations that work across multiple
model types, including correlation heatmaps, loading plots, variance explained
charts, and convergence diagnostics.

All functions return Plotly Figure objects for seamless integration with
Streamlit's st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


def plot_correlation_matrix(corr_matrix: np.ndarray, 
                            item_names: List[str], 
                            title: str = "Correlation Matrix") -> go.Figure:
    """
    Create an interactive correlation matrix heatmap.
    
    Useful for visualizing similarity matrices, residual correlations, or
    any square symmetric matrix. The color scale is centered at zero with
    blue for negative and red for positive values.
    
    Args:
        corr_matrix: (n_items, n_items) square matrix of correlations/similarities
        item_names: List of labels for each item (used for both axes)
        title: Plot title
        
    Returns:
        Plotly Figure with the heatmap
    """
    # Create the heatmap with hover information and value annotations
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=item_names,
        y=item_names,
        colorscale='RdBu_r',  # Red-Blue reversed (red = positive)
        zmid=0,               # Center the color scale at zero
        text=np.round(corr_matrix, 2),  # Values to display in cells
        texttemplate='%{text}',         # Format for cell text
        textfont={'size': 9},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    # Adjust height based on number of items (more items = taller plot)
    fig.update_layout(
        title=title,
        height=max(400, 35 * len(item_names)),
        xaxis={'tickangle': 45}
    )
    
    return fig


def plot_loadings_heatmap(loadings: np.ndarray, 
                          item_names: List[str],
                          factor_names: Optional[List[str]] = None, 
                          title: str = "Factor Loadings") -> go.Figure:
    """
    Create a heatmap of factor loadings.
    
    Visualizes the loading matrix with items on the y-axis and factors on
    the x-axis. Positive loadings appear red, negative loadings appear blue,
    with the color scale centered at zero.
    
    This is the standard way to visualize factor analysis, NMF, or any
    matrix factorization results where rows are items and columns are factors.
    
    Args:
        loadings: (n_items, n_factors) loading matrix
        item_names: Labels for each item/product
        factor_names: Optional labels for each factor. Defaults to "Factor 1", etc.
        title: Plot title
        
    Returns:
        Plotly Figure with the heatmap
    """
    n_factors = loadings.shape[1]
    
    # Generate default factor names if not provided
    if factor_names is None:
        factor_names = [f"Factor {i+1}" for i in range(n_factors)]
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=factor_names,
        y=item_names,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(loadings, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='%{y} on %{x}<br>Loading: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=max(400, 30 * len(item_names)),
        xaxis_title='Factor',
        yaxis_title='Product'
    )
    
    return fig


def plot_loadings_with_uncertainty(loadings: np.ndarray, 
                                    loadings_std: np.ndarray,
                                    item_names: List[str]) -> go.Figure:
    """
    Plot factor loadings with uncertainty (error bars).
    
    This is particularly useful for Bayesian factor models where we have
    posterior uncertainty estimates. Each factor is shown in a separate
    subplot with items sorted by absolute loading magnitude.
    
    Args:
        loadings: (n_items, n_factors) posterior mean loadings
        loadings_std: (n_items, n_factors) posterior standard deviations
        item_names: Labels for each item
        
    Returns:
        Plotly Figure with bar charts and error bars for each factor
    """
    n_items, n_factors = loadings.shape
    
    # Create subplots, one per factor
    fig = make_subplots(
        rows=1, 
        cols=n_factors, 
        subplot_titles=[f"Factor {i+1}" for i in range(n_factors)]
    )
    
    for f in range(n_factors):
        # Sort items by absolute loading (highest first) for this factor
        sorted_idx = np.argsort(np.abs(loadings[:, f]))[::-1]
        
        # Create bar chart with error bars
        fig.add_trace(
            go.Bar(
                x=[item_names[i] for i in sorted_idx],
                y=loadings[sorted_idx, f],
                error_y=dict(
                    type='data', 
                    array=loadings_std[sorted_idx, f], 
                    visible=True
                ),
                # Color bars based on sign: blue for positive, coral for negative
                marker_color=[
                    'steelblue' if l > 0 else 'coral' 
                    for l in loadings[sorted_idx, f]
                ],
                showlegend=False
            ),
            row=1, 
            col=f+1
        )
    
    fig.update_layout(
        height=400, 
        title='Factor Loadings (with Posterior Std)'
    )
    
    # Rotate x-axis labels for readability
    for i in range(n_factors):
        fig.update_xaxes(tickangle=45, row=1, col=i+1)
    
    return fig


def plot_variance_explained(var_explained_pct: np.ndarray, 
                            model_name: str) -> go.Figure:
    """
    Plot variance explained by each component with cumulative overlay.
    
    Shows both individual component contributions (bars) and cumulative
    variance explained (line), helping users decide how many components
    to retain.
    
    Args:
        var_explained_pct: Array of variance explained percentages for each component
        model_name: Name of the model (for the plot title)
        
    Returns:
        Plotly Figure with dual-axis bar and line chart
    """
    n_components = len(var_explained_pct)
    cumulative = np.cumsum(var_explained_pct)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Individual variance as bars (primary y-axis)
    fig.add_trace(
        go.Bar(
            x=[f"Comp {i+1}" for i in range(n_components)],
            y=var_explained_pct,
            name='Individual',
            marker_color='steelblue'
        ),
        secondary_y=False
    )
    
    # Cumulative variance as line (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=[f"Comp {i+1}" for i in range(n_components)],
            y=cumulative,
            name='Cumulative',
            line=dict(color='coral', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'{model_name} - Variance Explained',
        height=350
    )
    fig.update_yaxes(title_text="Individual %", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    
    return fig


def plot_elbo_convergence(elbo_history: List[float]) -> go.Figure:
    """
    Plot ELBO convergence for variational inference models.
    
    The ELBO (Evidence Lower Bound) should increase monotonically during
    variational inference. This plot helps diagnose convergence issues.
    
    Args:
        elbo_history: List of ELBO values at each iteration
        
    Returns:
        Plotly Figure showing ELBO over iterations
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(elbo_history) + 1)),
        y=elbo_history,
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title='ELBO Convergence',
        xaxis_title='Iteration',
        yaxis_title='ELBO',
        height=300
    )
    
    return fig


def plot_dendrogram(linkage_matrix: np.ndarray,
                    labels: List[str],
                    n_clusters: Optional[int] = None,
                    title: str = "Hierarchical Clustering Dendrogram") -> go.Figure:
    """
    Create an interactive dendrogram visualization from hierarchical clustering.
    
    The dendrogram shows the tree structure of how products are grouped together
    based on their similarity. The y-axis represents the distance at which clusters
    merge—lower merges indicate more similar items.
    
    Uses scipy's dendrogram function to compute the tree structure, then renders
    it in Plotly for interactivity. If n_clusters is provided, a horizontal line
    is drawn showing where the tree would be cut to produce that many clusters.
    
    Args:
        linkage_matrix: Output from scipy.cluster.hierarchy.linkage
        labels: List of labels for each leaf (product names)
        n_clusters: Optional number of clusters to highlight with a cut line
        title: Plot title
        
    Returns:
        Plotly Figure with the dendrogram
    """
    from scipy.cluster.hierarchy import dendrogram
    
    # Compute dendrogram structure (but don't plot with matplotlib)
    # no_plot=True just returns the data structure we need
    dendro_data = dendrogram(
        linkage_matrix,
        labels=labels,
        no_plot=True,
        color_threshold=0  # We'll handle coloring ourselves
    )
    
    fig = go.Figure()
    
    # The dendrogram returns icoord (x) and dcoord (y) for each U-shaped segment
    # Each segment connects two branches with a horizontal bar at the merge height
    icoord = np.array(dendro_data['icoord'])
    dcoord = np.array(dendro_data['dcoord'])
    
    # Draw each U-shaped linkage segment
    for i in range(len(icoord)):
        # Each segment is a "U" shape with 4 points: left leg, across, right leg
        x_coords = icoord[i]
        y_coords = dcoord[i]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='steelblue', width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add the leaf labels on the x-axis
    # dendro_data['ivl'] contains the reordered labels
    leaf_labels = dendro_data['ivl']
    # Leaf positions are at 5, 15, 25, ... (scipy default spacing)
    leaf_positions = [5 + 10 * i for i in range(len(leaf_labels))]
    
    # If n_clusters is specified, draw a horizontal line showing the cut threshold
    if n_clusters is not None and n_clusters > 1:
        # Find the appropriate height to cut the tree for n_clusters
        # The linkage matrix has columns: [cluster1, cluster2, distance, count]
        # To get n_clusters, we need to cut at the (n-n_clusters)th merge
        n_leaves = len(labels)
        cut_idx = n_leaves - n_clusters
        if 0 <= cut_idx < len(linkage_matrix):
            cut_height = linkage_matrix[cut_idx, 2]
            # Add a small offset so the line is just above the merge
            cut_height = cut_height * 1.01
            
            fig.add_hline(
                y=cut_height,
                line_dash="dash",
                line_color="coral",
                line_width=2,
                annotation_text=f"Cut for {n_clusters} clusters",
                annotation_position="right"
            )
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode='array',
            tickvals=leaf_positions,
            ticktext=leaf_labels,
            tickangle=45,
            title=''
        ),
        yaxis=dict(
            title='Distance'
        ),
        height=max(400, 20 * len(labels)),
        showlegend=False,
        margin=dict(b=max(100, 8 * max(len(l) for l in leaf_labels)))  # Room for rotated labels
    )
    
    return fig


def plot_silhouette_scores(k_range: List[int], 
                           scores: List[float],
                           optimal_k: int) -> go.Figure:
    """
    Plot silhouette scores for different numbers of clusters.
    
    Helps users choose the optimal number of clusters by visualizing
    how cluster quality (silhouette score) varies with k. The optimal
    k is highlighted.
    
    Args:
        k_range: List of k values tested
        scores: List of silhouette scores for each k
        optimal_k: The k with highest silhouette score
        
    Returns:
        Plotly Figure showing silhouette scores vs k
    """
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=k_range,
        y=scores,
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        marker=dict(size=8),
        name='Silhouette Score'
    ))
    
    # Highlight optimal k
    optimal_idx = k_range.index(optimal_k)
    fig.add_trace(go.Scatter(
        x=[optimal_k],
        y=[scores[optimal_idx]],
        mode='markers',
        marker=dict(color='coral', size=15, symbol='star'),
        name=f'Optimal k={optimal_k}'
    ))
    
    fig.update_layout(
        title='Silhouette Score by Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        height=350
    )
    
    return fig