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
from typing import List, Optional, Dict


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
                hovertemplate='%{text}<br>Dim '+f"{dim_x+1}"+': %{x:.3f}<br>Dim '+f"{dim_y+1}"+': %{y:.3f}<extra></extra>'
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

### POISSON LCA PLOTS #####
def plot_count_lca_profiles(
    item_rates: np.ndarray,
    class_probs: np.ndarray,
    product_names: List[str],
    zero_probs: Optional[np.ndarray] = None,
    title: str = "Count LCA Class Profiles"
) -> go.Figure:
    """
    Plot class profiles showing expected purchase rates for each class.
    
    For count LCA, this shows the λ (rate) parameters per class,
    representing expected purchase counts.
    
    Args:
        item_rates: (n_classes, n_items) rate parameters
        class_probs: (n_classes,) class proportions
        product_names: List of product names
        zero_probs: Optional (n_classes, n_items) zero-inflation probs
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_classes, n_items = item_rates.shape
    
    # Color palette
    colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    
    fig = go.Figure()
    
    # Bar width and positioning
    bar_width = 0.8 / n_classes
    
    for c in range(n_classes):
        class_label = f"Class {c+1} ({class_probs[c]*100:.1f}%)"
        offset = (c - n_classes/2 + 0.5) * bar_width
        
        fig.add_trace(go.Bar(
            name=class_label,
            x=[i + offset for i in range(n_items)],
            y=item_rates[c, :],
            width=bar_width,
            marker_color=colors[c % len(colors)],
            hovertemplate=(
                f"<b>{class_label}</b><br>" +
                "Product: %{customdata}<br>" +
                "Expected Count: %{y:.2f}<extra></extra>"
            ),
            customdata=product_names,
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Product",
            tickmode='array',
            tickvals=list(range(n_items)),
            ticktext=product_names,
            tickangle=45,
        ),
        yaxis=dict(title="Expected Count (λ)"),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template="plotly_white",
        height=500,
    )
    
    return fig


def plot_zero_inflation_heatmap(
    zero_probs: np.ndarray,
    class_probs: np.ndarray,
    product_names: List[str],
    title: str = "Zero-Inflation Probabilities by Class"
) -> go.Figure:
    """
    Plot heatmap of zero-inflation probabilities (ψ) for ZIP-LCA.
    
    Shows the probability of "structural zeros" (never-buyers) for
    each product-class combination. High values indicate products
    that certain segments never purchase.
    
    Args:
        zero_probs: (n_classes, n_items) zero-inflation probabilities
        class_probs: (n_classes,) class proportions
        product_names: List of product names
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_classes = len(class_probs)
    
    class_labels = [f"Class {c+1} ({class_probs[c]*100:.1f}%)" 
                    for c in range(n_classes)]
    
    fig = go.Figure(data=go.Heatmap(
        z=zero_probs,
        x=product_names,
        y=class_labels,
        colorscale='RdYlBu_r',  # Red = high zero-inflation
        zmin=0,
        zmax=1,
        colorbar=dict(title="P(structural zero)"),
        hovertemplate=(
            "Class: %{y}<br>" +
            "Product: %{x}<br>" +
            "Zero-Inflation: %{z:.2%}<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Product", tickangle=45),
        yaxis=dict(title="Latent Class"),
        template="plotly_white",
        height=400,
    )
    
    return fig


def plot_effective_rates(
    item_rates: np.ndarray,
    zero_probs: np.ndarray,
    class_probs: np.ndarray,
    product_names: List[str],
    title: str = "Effective Purchase Rates (Accounting for Zero-Inflation)"
) -> go.Figure:
    """
    Plot effective rates E[Y] = (1-ψ)*λ for ZIP model.
    
    Shows the actual expected purchase counts after accounting for
    structural zeros. Useful for comparing with observed means.
    
    Args:
        item_rates: (n_classes, n_items) Poisson rate parameters
        zero_probs: (n_classes, n_items) zero-inflation probabilities
        class_probs: (n_classes,) class proportions
        product_names: List of product names
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Effective rate = (1 - ψ) * λ
    effective_rates = (1 - zero_probs) * item_rates
    
    return plot_count_lca_profiles(
        effective_rates, class_probs, product_names,
        title=title
    )


def plot_class_radar(
    item_rates: np.ndarray,
    class_probs: np.ndarray,
    product_names: List[str],
    normalize: bool = True,
    title: str = "Class Profile Comparison"
) -> go.Figure:
    """
    Radar chart comparing class profiles across products.
    
    Useful for visualizing how classes differ in their purchase patterns
    across multiple products simultaneously.
    
    Args:
        item_rates: (n_classes, n_items) rate parameters
        class_probs: (n_classes,) class proportions
        product_names: List of product names
        normalize: If True, normalize rates to [0,1] for each product
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_classes = item_rates.shape[0]
    
    colors = [
        'rgba(99, 110, 250, 0.6)', 'rgba(239, 85, 59, 0.6)', 
        'rgba(0, 204, 150, 0.6)', 'rgba(171, 99, 250, 0.6)',
        'rgba(255, 161, 90, 0.6)', 'rgba(25, 211, 243, 0.6)'
    ]
    
    if normalize:
        # Normalize each product to [0,1] range
        rates_plot = item_rates / (item_rates.max(axis=0, keepdims=True) + 1e-10)
    else:
        rates_plot = item_rates
    
    fig = go.Figure()
    
    for c in range(n_classes):
        class_label = f"Class {c+1} ({class_probs[c]*100:.1f}%)"
        
        # Close the radar by appending first value
        r_values = list(rates_plot[c, :]) + [rates_plot[c, 0]]
        theta_values = product_names + [product_names[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            name=class_label,
            fill='toself',
            fillcolor=colors[c % len(colors)],
            line=dict(color=colors[c % len(colors)].replace('0.6', '1')),
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else [0, rates_plot.max() * 1.1]
            )
        ),
        title=dict(text=title, x=0.5),
        template="plotly_white",
        height=500,
    )
    
    return fig


def plot_observed_vs_expected(
    data: np.ndarray,
    expected: np.ndarray,
    product_names: List[str],
    sample_size: int = 1000,
    title: str = "Observed vs Expected Counts"
) -> go.Figure:
    """
    Scatter plot comparing observed counts to model predictions.
    
    Points near the diagonal indicate good fit. Systematic deviations
    reveal products or segments where the model struggles.
    
    Args:
        data: (n_obs, n_items) observed count matrix
        expected: (n_obs, n_items) expected count matrix
        product_names: List of product names
        sample_size: Number of points to plot (for large datasets)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_obs, n_items = data.shape
    
    # Flatten for scatter plot
    obs_flat = data.flatten()
    exp_flat = expected.flatten()
    
    # Product labels for each point
    product_labels = np.tile(product_names, n_obs)
    
    # Sample if too many points
    if len(obs_flat) > sample_size:
        idx = np.random.choice(len(obs_flat), sample_size, replace=False)
        obs_flat = obs_flat[idx]
        exp_flat = exp_flat[idx]
        product_labels = product_labels[idx]
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=exp_flat,
        y=obs_flat,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.5,
            color='#636EFA',
        ),
        text=product_labels,
        hovertemplate=(
            "Product: %{text}<br>" +
            "Expected: %{x:.2f}<br>" +
            "Observed: %{y}<extra></extra>"
        ),
    ))
    
    # Diagonal line
    max_val = max(obs_flat.max(), exp_flat.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Fit',
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Expected Count"),
        yaxis=dict(title="Observed Count"),
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    
    return fig


def plot_residual_heatmap(
    residuals: np.ndarray,
    product_names: List[str],
    household_labels: Optional[List[str]] = None,
    max_households: int = 50,
    title: str = "Pearson Residuals"
) -> go.Figure:
    """
    Heatmap of Pearson residuals for model diagnostics.
    
    Highlights household-product combinations with poor fit.
    Red = model underestimates, Blue = model overestimates.
    
    Args:
        residuals: (n_obs, n_items) Pearson residual matrix
        product_names: List of product names
        household_labels: Optional household identifiers
        max_households: Max households to display
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_obs = residuals.shape[0]
    
    # Sample households if too many
    if n_obs > max_households:
        # Pick households with largest absolute residuals
        max_abs_residual = np.abs(residuals).max(axis=1)
        idx = np.argsort(max_abs_residual)[-max_households:]
        residuals_plot = residuals[idx, :]
        if household_labels is not None:
            household_labels = [household_labels[i] for i in idx]
    else:
        residuals_plot = residuals
    
    if household_labels is None:
        household_labels = [f"HH {i+1}" for i in range(residuals_plot.shape[0])]
    
    # Clip for visualization
    residuals_clipped = np.clip(residuals_plot, -5, 5)
    
    fig = go.Figure(data=go.Heatmap(
        z=residuals_clipped,
        x=product_names,
        y=household_labels,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="Residual"),
        hovertemplate=(
            "Household: %{y}<br>" +
            "Product: %{x}<br>" +
            "Residual: %{z:.2f}<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Product", tickangle=45),
        yaxis=dict(title="Household"),
        template="plotly_white",
        height=600,
    )
    
    return fig


def plot_model_selection(
    results_by_k: Dict[int, Dict],
    criterion: str = 'bic',
    title: str = "Model Selection: Number of Classes"
) -> go.Figure:
    """
    Plot information criteria vs number of classes for model selection.
    
    Args:
        results_by_k: Dictionary mapping n_classes to results
        criterion: 'bic' or 'aic'
        title: Plot title
        
    Returns:
        Plotly figure
    """
    k_values = sorted(results_by_k.keys())
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Information Criteria', 'Log-Likelihood')
    )
    
    # BIC and AIC
    bic_values = [results_by_k[k]['bic'] for k in k_values]
    aic_values = [results_by_k[k]['aic'] for k in k_values]
    
    fig.add_trace(go.Scatter(
        x=k_values, y=bic_values,
        mode='lines+markers',
        name='BIC',
        marker=dict(size=10),
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=k_values, y=aic_values,
        mode='lines+markers',
        name='AIC',
        marker=dict(size=10),
    ), row=1, col=1)
    
    # Highlight optimal
    optimal_k = min(k_values, key=lambda k: results_by_k[k][criterion])
    optimal_val = results_by_k[optimal_k][criterion]
    
    fig.add_trace(go.Scatter(
        x=[optimal_k], y=[optimal_val],
        mode='markers',
        name=f'Optimal ({criterion.upper()}={optimal_k})',
        marker=dict(size=15, symbol='star', color='gold'),
    ), row=1, col=1)
    
    # Log-likelihood
    ll_values = [results_by_k[k]['log_likelihood'] for k in k_values]
    
    fig.add_trace(go.Scatter(
        x=k_values, y=ll_values,
        mode='lines+markers',
        name='Log-Likelihood',
        marker=dict(size=10, color='green'),
        showlegend=True,
    ), row=1, col=2)
    
    fig.update_xaxes(title_text="Number of Classes", row=1, col=1)
    fig.update_xaxes(title_text="Number of Classes", row=1, col=2)
    fig.update_yaxes(title_text="Information Criterion", row=1, col=1)
    fig.update_yaxes(title_text="Log-Likelihood", row=1, col=2)
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        height=400,
    )
    
    return fig


def plot_dispersion_comparison(
    dispersion: np.ndarray,
    class_probs: np.ndarray,
    title: str = "Dispersion Parameters by Class (NB-LCA)"
) -> go.Figure:
    """
    Bar plot of dispersion parameters for Negative Binomial LCA.
    
    Higher dispersion indicates more overdispersion (variance >> mean).
    
    Args:
        dispersion: (n_classes,) dispersion parameters
        class_probs: (n_classes,) class proportions
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_classes = len(dispersion)
    class_labels = [f"Class {c+1}\n({class_probs[c]*100:.1f}%)" 
                    for c in range(n_classes)]
    
    fig = go.Figure(data=go.Bar(
        x=class_labels,
        y=dispersion,
        marker_color='#636EFA',
        text=[f"α={d:.3f}" for d in dispersion],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Latent Class"),
        yaxis=dict(title="Dispersion (α)"),
        template="plotly_white",
        height=400,
    )
    
    # Add reference line for Poisson (alpha approaching 0)
    fig.add_hline(
        y=0, line_dash="dash", line_color="red",
        annotation_text="Poisson limit",
        annotation_position="bottom right"
    )
    
    return fig