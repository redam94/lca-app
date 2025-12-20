"""
Model-specific plotting functions for Latent Structure Analysis.

This module provides visualizations tailored to specific model types:
- LCA profile plots (purchase probability patterns per class)
- Biplots for factor-type models (products and households in latent space)
- DCM coefficient plots with uncertainty intervals

These plots help interpret model results and communicate findings.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


def plot_lca_profiles(item_probs: np.ndarray, 
                      class_probs: np.ndarray, 
                      product_names: List[str]) -> go.Figure:
    """
    Plot LCA class profiles showing purchase probabilities for each class.
    
    This is the key visualization for interpreting LCA results. Each class
    is shown as a grouped bar representing the probability of purchasing
    each product given membership in that class. Classes are labeled with
    their population share.
    
    Interpretation tips:
    - High bars indicate products that define a class
    - Classes with similar profiles may be candidates for merging
    - Products with high probability in multiple classes are "universal"
    - Products high in one class but low in others are "discriminating"
    
    Args:
        item_probs: (n_classes, n_items) probability matrix P(purchase | class)
        class_probs: (n_classes,) prior class probabilities
        product_names: Labels for each product
        
    Returns:
        Plotly Figure with grouped bar chart of class profiles
    """
    n_classes = len(class_probs)
    
    fig = go.Figure()
    
    # Define a color palette for classes
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    # Create a bar trace for each class
    for c in range(n_classes):
        # Label includes class number and population percentage
        class_label = f"Class {c+1} ({class_probs[c]*100:.1f}%)"
        
        fig.add_trace(go.Bar(
            name=class_label,
            x=product_names,
            y=item_probs[c],
            marker_color=colors[c % len(colors)]
        ))
    
    fig.update_layout(
        title='Class Profiles: Purchase Probability by Class',
        barmode='group',
        xaxis_title='Product',
        yaxis_title='P(Purchase | Class)',
        xaxis_tickangle=45,
        height=450,
        legend_title='Latent Class'
    )
    
    # Set y-axis to probability range
    fig.update_yaxes(range=[0, 1])
    
    return fig


def plot_biplot(product_embeddings: np.ndarray,
                product_labels: List[str],
                household_embeddings: Optional[np.ndarray] = None,
                dim_x: int = 0,
                dim_y: int = 1,
                var_explained: Optional[np.ndarray] = None,
                cluster_labels: Optional[np.ndarray] = None,
                title: str = "Biplot",
                show_households: bool = True,
                max_households: int = 1000) -> go.Figure:
    """
    Create a biplot showing products and households in latent space.
    
    Biplots are fundamental for interpreting factor models. Products are
    shown as labeled points, and optionally households are shown as smaller
    unlabeled points in the same space. The position of products reveals
    their relationships: products close together have similar purchase patterns.
    
    For LCA, the latent space is defined by class membership probabilities.
    For factor models, it's defined by the factor loadings.
    
    Args:
        product_embeddings: (n_products, n_dims) product coordinates
        product_labels: List of product names
        household_embeddings: Optional (n_households, n_dims) household coordinates
        dim_x: Which dimension to plot on x-axis (0-indexed)
        dim_y: Which dimension to plot on y-axis (0-indexed)
        var_explained: Optional variance explained by each dimension (%)
        cluster_labels: Optional cluster assignments for products (0-indexed)
        title: Plot title
        show_households: Whether to show household points
        max_households: Maximum number of households to plot (for performance)
        
    Returns:
        Plotly Figure with the biplot
    """
    fig = go.Figure()
    
    # Color palette for clusters (if provided)
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    # Plot household points first (as background)
    if show_households and household_embeddings is not None:
        # Subsample households if there are too many
        n_households = len(household_embeddings)
        if n_households > max_households:
            indices = np.random.choice(n_households, max_households, replace=False)
            hh_subset = household_embeddings[indices]
        else:
            hh_subset = household_embeddings
        
        fig.add_trace(go.Scatter(
            x=hh_subset[:, dim_x],
            y=hh_subset[:, dim_y],
            mode='markers',
            marker=dict(
                size=4,
                color='lightgray',
                opacity=0.5
            ),
            name='Households',
            hoverinfo='skip'
        ))
    
    # Plot products with labels
    if cluster_labels is not None:
        # Color by cluster
        n_clusters = len(np.unique(cluster_labels))
        for c in range(n_clusters):
            mask = cluster_labels == c
            fig.add_trace(go.Scatter(
                x=product_embeddings[mask, dim_x],
                y=product_embeddings[mask, dim_y],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=colors[c % len(colors)]
                ),
                text=[product_labels[i] for i in np.where(mask)[0]],
                textposition='top center',
                textfont=dict(size=10),
                name=f'Cluster {c+1}',
                hovertemplate='%{text}<br>Dim '+f"{dim_x+1}"+': %{{x:.3f}}<br>Dim '+f"{dim_y+1}"+': %{{y:.3f}}<extra></extra>'
            ))
    else:
        # Single color for all products
        fig.add_trace(go.Scatter(
            x=product_embeddings[:, dim_x],
            y=product_embeddings[:, dim_y],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#636EFA'
            ),
            text=product_labels,
            textposition='top center',
            textfont=dict(size=10),
            name='Products',
            hovertemplate='%{text}<br>Dim ' + str(dim_x+1) + ': %{x:.3f}<br>Dim ' + str(dim_y+1) + ': %{y:.3f}<extra></extra>'
        ))
    
    # Add axis labels with variance explained if available
    if var_explained is not None:
        x_label = f"Dimension {dim_x+1} ({var_explained[dim_x]:.1f}%)"
        y_label = f"Dimension {dim_y+1} ({var_explained[dim_y]:.1f}%)"
    else:
        x_label = f"Dimension {dim_x+1}"
        y_label = f"Dimension {dim_y+1}"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        showlegend=True
    )
    
    # Add crosshairs at origin for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def plot_dcm_coefficients(alpha: np.ndarray,
                          alpha_std: np.ndarray,
                          product_names: List[str],
                          beta: Optional[np.ndarray] = None,
                          beta_std: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None) -> go.Figure:
    """
    Plot DCM coefficients with uncertainty intervals.
    
    Shows product intercepts (baseline purchase probabilities) and optionally
    household feature effects with 95% credible intervals. This helps identify
    which products have higher/lower baseline appeal and how household
    characteristics affect purchase probabilities.
    
    Args:
        alpha: (n_products,) product intercept posterior means
        alpha_std: (n_products,) posterior standard deviations
        product_names: Labels for each product
        beta: Optional (n_products, n_features) feature effect means
        beta_std: Optional feature effect standard deviations
        feature_names: Optional labels for household features
        
    Returns:
        Plotly Figure with coefficient plots
    """
    # Calculate number of subplots needed
    n_subplots = 1  # Always have intercept plot
    if beta is not None:
        n_subplots += beta.shape[1]
    
    # Create subplots
    fig = make_subplots(
        rows=n_subplots, 
        cols=1, 
        subplot_titles=['Product Intercepts (α)'] + 
                       ([f'Effect of {f}' for f in feature_names] if feature_names else
                        [f'Feature {i+1}' for i in range(n_subplots-1)])
    )
    
    # Sort products by intercept for the intercept plot
    sorted_idx = np.argsort(alpha)
    sorted_names = [product_names[i] for i in sorted_idx]
    sorted_alpha = alpha[sorted_idx]
    sorted_alpha_std = alpha_std[sorted_idx]
    
    # Plot intercepts with error bars
    fig.add_trace(
        go.Scatter(
            x=sorted_alpha,
            y=sorted_names,
            mode='markers',
            marker=dict(size=8, color='steelblue'),
            error_x=dict(
                type='data',
                array=1.96 * sorted_alpha_std,  # 95% CI
                visible=True
            ),
            name='Intercept',
            hovertemplate='%{y}<br>α = %{x:.2f} ± %{error_x.array:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Plot feature effects if provided
    if beta is not None:
        n_features = beta.shape[1]
        for f in range(n_features):
            # Sort by this feature's effect
            f_sorted_idx = np.argsort(beta[:, f])
            f_sorted_names = [product_names[i] for i in f_sorted_idx]
            f_sorted_beta = beta[f_sorted_idx, f]
            f_sorted_std = beta_std[f_sorted_idx, f]
            
            fig.add_trace(
                go.Scatter(
                    x=f_sorted_beta,
                    y=f_sorted_names,
                    mode='markers',
                    marker=dict(size=8, color='coral'),
                    error_x=dict(
                        type='data',
                        array=1.96 * f_sorted_std,
                        visible=True
                    ),
                    name=feature_names[f] if feature_names else f'Feature {f+1}',
                    showlegend=False,
                    hovertemplate='%{y}<br>β = %{x:.2f} ± %{error_x.array:.2f}<extra></extra>'
                ),
                row=f+2, col=1
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=f+2, col=1)
    
    fig.update_layout(
        height=200 + 25 * len(product_names) * n_subplots,
        title='Discrete Choice Model Coefficients (95% CI)',
        showlegend=False
    )
    
    return fig


def plot_mca_contributions(contributions: np.ndarray,
                           product_labels: List[str],
                           n_dims: int = 3) -> go.Figure:
    """
    Plot MCA product contributions to each dimension.
    
    Contributions show which products most strongly define each dimension.
    Products with high contributions on a dimension are the best markers
    for that "shopping style."
    
    Args:
        contributions: (n_products, n_dims) contribution matrix
        product_labels: Labels for each product
        n_dims: Number of dimensions to show
        
    Returns:
        Plotly Figure with stacked bar contributions
    """
    n_dims = min(n_dims, contributions.shape[1])
    
    fig = make_subplots(
        rows=1, 
        cols=n_dims,
        subplot_titles=[f'Dimension {i+1}' for i in range(n_dims)]
    )
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    for d in range(n_dims):
        # Sort by contribution to this dimension
        sorted_idx = np.argsort(contributions[:, d])[::-1]
        top_n = min(10, len(sorted_idx))  # Show top 10
        
        fig.add_trace(
            go.Bar(
                y=[product_labels[i] for i in sorted_idx[:top_n]],
                x=contributions[sorted_idx[:top_n], d],
                orientation='h',
                marker_color=colors[d % len(colors)],
                showlegend=False
            ),
            row=1, col=d+1
        )
    
    fig.update_layout(
        height=400,
        title='Product Contributions to MCA Dimensions'
    )
    
    return fig