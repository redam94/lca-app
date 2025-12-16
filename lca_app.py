"""
Latent Structure Analysis App for Purchase Data
================================================
Fixed version with proper session state caching to prevent plots from disappearing
when UI elements (like biplot dimensions) are changed.

Key fixes:
1. All visualization data stored in session state
2. Model-specific visualizations moved outside button block
3. Proper caching to avoid model re-runs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
import io
import json
import zipfile
from datetime import datetime

# Optional imports
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    PYMC_AVAILABLE = True
    PYMC_ERROR = None
except Exception as e:
    PYMC_AVAILABLE = False
    PYMC_ERROR = str(e)

try:
    import prince
    PRINCE_AVAILABLE = True
except ImportError:
    PRINCE_AVAILABLE = False


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_correlation_matrix(corr_matrix: np.ndarray, item_names: list, 
                            title: str = "Correlation Matrix") -> go.Figure:
    """Create correlation matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=item_names,
        y=item_names,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={'size': 9},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=max(400, 35 * len(item_names)),
        xaxis={'tickangle': 45}
    )
    
    return fig


def plot_loadings_heatmap(loadings: np.ndarray, item_names: list,
                          factor_names: list = None, title: str = "Factor Loadings") -> go.Figure:
    """Create factor loadings heatmap."""
    n_factors = loadings.shape[1]
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


def plot_loadings_with_uncertainty(loadings: np.ndarray, loadings_std: np.ndarray,
                                   item_names: list) -> go.Figure:
    """Plot factor loadings with uncertainty bars."""
    n_items, n_factors = loadings.shape
    
    fig = make_subplots(rows=1, cols=n_factors, 
                        subplot_titles=[f"Factor {i+1}" for i in range(n_factors)])
    
    for f in range(n_factors):
        sorted_idx = np.argsort(np.abs(loadings[:, f]))[::-1]
        
        fig.add_trace(
            go.Bar(
                x=[item_names[i] for i in sorted_idx],
                y=loadings[sorted_idx, f],
                error_y=dict(type='data', array=loadings_std[sorted_idx, f], visible=True),
                marker_color=['steelblue' if l > 0 else 'coral' for l in loadings[sorted_idx, f]],
                showlegend=False
            ),
            row=1, col=f+1
        )
    
    fig.update_layout(height=400, title='Factor Loadings (with Posterior Std)')
    for i in range(n_factors):
        fig.update_xaxes(tickangle=45, row=1, col=i+1)
    
    return fig


def plot_variance_explained(var_explained_pct: np.ndarray, model_name: str) -> go.Figure:
    """Plot variance explained by each component."""
    n_components = len(var_explained_pct)
    cumulative = np.cumsum(var_explained_pct)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=[f"Comp {i+1}" for i in range(n_components)],
            y=var_explained_pct,
            name='Individual',
            marker_color='steelblue'
        ),
        secondary_y=False
    )
    
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


def plot_lca_profiles(item_probs: np.ndarray, class_probs: np.ndarray, 
                      product_names: list) -> go.Figure:
    """Plot LCA class profiles."""
    n_classes = len(class_probs)
    
    fig = go.Figure()
    
    for c in range(n_classes):
        fig.add_trace(go.Bar(
            name=f"Class {c+1} ({class_probs[c]*100:.1f}%)",
            x=product_names,
            y=item_probs[c],
        ))
    
    fig.update_layout(
        title='Purchase Probability by Latent Class',
        xaxis_title='Product',
        yaxis_title='P(Purchase)',
        barmode='group',
        xaxis={'tickangle': 45},
        height=500
    )
    
    return fig


def plot_beta_coefficients(beta: np.ndarray, beta_std: np.ndarray, 
                           product_names: list, feature_names: list) -> go.Figure:
    """Plot household feature coefficients for each product."""
    n_items, n_features = beta.shape
    
    fig = go.Figure()
    
    for j, prod in enumerate(product_names):
        fig.add_trace(go.Bar(
            name=prod,
            x=feature_names,
            y=beta[j],
            error_y=dict(type='data', array=beta_std[j], visible=True),
        ))
    
    fig.update_layout(
        title='Household Feature Effects by Product (with 1 SD error bars)',
        xaxis_title='Household Feature',
        yaxis_title='Coefficient',
        barmode='group',
        height=500
    )
    
    return fig


def plot_product_intercepts(alpha: np.ndarray, alpha_std: np.ndarray, 
                            product_names: list) -> go.Figure:
    """Plot product intercepts (baseline purchase probabilities)."""
    probs = 1 / (1 + np.exp(-alpha))
    
    idx = np.argsort(probs)[::-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[product_names[i] for i in idx],
        y=probs[idx],
        error_y=dict(
            type='data',
            array=alpha_std[idx] * probs[idx] * (1 - probs[idx]),
            visible=True
        ),
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title='Baseline Purchase Probability by Product',
        xaxis_title='Product',
        yaxis_title='P(Purchase)',
        xaxis={'tickangle': 45},
        height=400
    )
    
    return fig


def plot_generic_biplot(row_coords: np.ndarray, col_coords: np.ndarray, 
                        product_labels: list, dim1: int = 0, dim2: int = 1,
                        var_explained: list = None, model_name: str = "Model",
                        cluster_labels: np.ndarray = None,
                        show_households: bool = True) -> go.Figure:
    """Create a generic biplot showing products and optionally households in latent space."""
    fig = go.Figure()
    
    # Check if household coords have enough dimensions for the selected dims
    if show_households and row_coords is not None and len(row_coords) > 0:
        if row_coords.shape[1] > max(dim1, dim2):
            fig.add_trace(go.Scatter(
                x=row_coords[:, dim1],
                y=row_coords[:, dim2],
                mode='markers',
                marker=dict(size=4, color='lightblue', opacity=0.5),
                name='Households',
                hoverinfo='skip'
            ))
    
    # Check if col_coords has enough dimensions
    if col_coords is not None and col_coords.shape[1] > max(dim1, dim2):
        if cluster_labels is not None:
            unique_clusters = np.unique(cluster_labels)
            colors = px.colors.qualitative.Set1[:len(unique_clusters)]
            
            for i, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                cluster_name = f"Cluster {cluster + 1}" if cluster >= 0 else "Noise"
                fig.add_trace(go.Scatter(
                    x=col_coords[mask, dim1],
                    y=col_coords[mask, dim2],
                    mode='markers+text',
                    marker=dict(size=14, color=colors[i % len(colors)], symbol='diamond',
                               line=dict(width=1, color='black')),
                    text=[product_labels[j] for j in np.where(mask)[0]],
                    textposition='top center',
                    name=cluster_name,
                    hovertemplate='{text}<br>Dim %d: {x:.3f}<br>Dim %d: {y:.3f}<extra></extra>' % (dim1+1, dim2+1)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=col_coords[:, dim1],
                y=col_coords[:, dim2],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='diamond'),
                text=product_labels,
                textposition='top center',
                name='Products',
                hovertemplate='{text}<br>Dim %d: {x:.3f}<br>Dim %d: {y:.3f}<extra></extra>' % (dim1+1, dim2+1)
            ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    if var_explained is not None and len(var_explained) > max(dim1, dim2):
        x_label = f"Dimension {dim1+1} ({var_explained[dim1]:.1f}%)"
        y_label = f"Dimension {dim2+1} ({var_explained[dim2]:.1f}%)"
    else:
        x_label = f"Dimension {dim1+1}"
        y_label = f"Dimension {dim2+1}"
    
    fig.update_layout(
        title=f'{model_name} Biplot: Products in Latent Space',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        showlegend=True
    )
    
    return fig


def plot_mca_contributions(contributions: np.ndarray, product_labels: list,
                           n_dims: int = 3) -> go.Figure:
    """Plot product contributions to each MCA dimension."""
    n_dims = min(n_dims, contributions.shape[1])
    
    fig = make_subplots(
        rows=1, cols=n_dims,
        subplot_titles=[f"Dimension {i+1}" for i in range(n_dims)]
    )
    
    for dim in range(n_dims):
        sorted_idx = np.argsort(contributions[:, dim])[::-1]
        
        fig.add_trace(
            go.Bar(
                x=[product_labels[i] for i in sorted_idx],
                y=contributions[sorted_idx, dim] * 100,
                marker_color='steelblue',
                showlegend=False
            ),
            row=1, col=dim+1
        )
    
    fig.update_layout(
        title='Product Contributions to MCA Dimensions (%)',
        height=400
    )
    
    for i in range(n_dims):
        fig.update_xaxes(tickangle=45, row=1, col=i+1)
        fig.update_yaxes(title_text="Contribution %" if i == 0 else "", row=1, col=i+1)
    
    return fig


def plot_cluster_summary(embeddings: np.ndarray, labels: np.ndarray, 
                         product_labels: list, title: str = "Cluster Summary") -> go.Figure:
    """Create a summary visualization of clusters."""
    n_clusters = len(np.unique(labels))
    colors = px.colors.qualitative.Set1[:n_clusters]
    
    fig = go.Figure()
    
    for cluster in range(n_clusters):
        mask = labels == cluster
        cluster_products = [product_labels[i] for i in np.where(mask)[0]]
        
        fig.add_trace(go.Bar(
            name=f"Cluster {cluster + 1}",
            x=cluster_products,
            y=[1] * len(cluster_products),
            marker_color=colors[cluster % len(colors)],
            text=cluster_products,
            textposition='auto'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Product',
        yaxis_visible=False,
        barmode='group',
        height=300,
        showlegend=True
    )
    
    return fig


def plot_dendrogram(linkage_matrix: np.ndarray, labels: list, title: str = "Dendrogram") -> go.Figure:
    """Create a dendrogram plot."""
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    import matplotlib.pyplot as plt
    
    # Create dendrogram data
    plt.figure(figsize=(10, 5))
    dend = scipy_dendrogram(linkage_matrix, labels=labels, no_plot=True)
    plt.close()
    
    # Convert to plotly
    fig = go.Figure()
    
    # Add the dendrogram lines
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])
    
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color='steelblue', width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add labels
    tick_positions = [5 + 10 * i for i in range(len(labels))]
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=dend['ivl'],
            tickangle=45
        ),
        yaxis_title='Distance',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_convergence(history: list, title: str = "Convergence", 
                     ylabel: str = "Value") -> go.Figure:
    """Plot convergence history."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history, mode='lines', name=ylabel))
    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title=ylabel,
        height=300
    )
    return fig


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_residual_correlations(data: np.ndarray, responsibilities: np.ndarray, 
                                   item_probs: np.ndarray) -> np.ndarray:
    """Compute residual correlations after accounting for class membership."""
    expected = responsibilities @ item_probs
    residuals = data - expected
    return np.corrcoef(residuals.T)


def get_model_cache_key(data_hash: str, model_type: str, model_params: dict, product_columns: tuple) -> str:
    """Generate a unique cache key for model results based on input parameters."""
    params_hash = hash(frozenset(model_params.items()))
    return f"{model_type}_{data_hash}_{params_hash}_{hash(product_columns)}"


def compute_hierarchical_clustering(similarity_matrix: np.ndarray, method: str = 'average') -> dict:
    """Compute hierarchical clustering from similarity matrix."""
    # Convert similarity to distance
    distance_matrix = 1 - np.clip(similarity_matrix, -1, 1)
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form
    n = len(distance_matrix)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(distance_matrix[i, j])
    condensed = np.array(condensed)
    
    # Compute linkage
    linkage_matrix = linkage(condensed, method=method)
    
    return {
        'linkage_matrix': linkage_matrix,
        'distance_matrix': distance_matrix
    }


def find_optimal_clusters(embeddings: np.ndarray, max_k: int = 10) -> dict:
    """Find optimal number of clusters using silhouette score."""
    scores = []
    k_range = range(2, min(max_k + 1, len(embeddings)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
    
    optimal_k = list(k_range)[np.argmax(scores)]
    
    return {
        'optimal_k': optimal_k,
        'scores': scores,
        'range': list(k_range)
    }


def get_cluster_members(labels: np.ndarray, product_labels: list) -> pd.DataFrame:
    """Get cluster membership as a DataFrame."""
    return pd.DataFrame({
        'Product': product_labels,
        'Cluster': labels + 1  # 1-indexed
    }).sort_values('Cluster')


# =============================================================================
# LATENT CLASS ANALYSIS (LCA)
# =============================================================================

def initialize_lca_parameters(n_classes: int, n_items: int, seed: int = 42) -> tuple:
    """Initialize LCA parameters randomly."""
    np.random.seed(seed)
    class_probs = np.random.dirichlet(np.ones(n_classes))
    item_probs = np.random.beta(2, 2, size=(n_classes, n_items))
    return class_probs, item_probs


def lca_e_step(data: np.ndarray, class_probs: np.ndarray, item_probs: np.ndarray) -> np.ndarray:
    """E-step: Compute posterior probability of class membership."""
    n_obs = data.shape[0]
    n_classes = len(class_probs)
    
    log_probs = np.zeros((n_obs, n_classes))
    
    for c in range(n_classes):
        log_prob = np.log(class_probs[c] + 1e-10)
        log_prob += np.sum(data * np.log(item_probs[c] + 1e-10), axis=1)
        log_prob += np.sum((1 - data) * np.log(1 - item_probs[c] + 1e-10), axis=1)
        log_probs[:, c] = log_prob
    
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    responsibilities = probs / probs.sum(axis=1, keepdims=True)
    
    return responsibilities


def lca_m_step(data: np.ndarray, responsibilities: np.ndarray) -> tuple:
    """M-step: Update parameters given responsibilities."""
    n_obs = data.shape[0]
    
    class_counts = responsibilities.sum(axis=0)
    class_probs = class_counts / n_obs
    
    item_probs = (responsibilities.T @ data) / (class_counts[:, np.newaxis] + 1e-10)
    item_probs = np.clip(item_probs, 0.01, 0.99)
    
    return class_probs, item_probs


def compute_lca_log_likelihood(data: np.ndarray, class_probs: np.ndarray, 
                                item_probs: np.ndarray) -> float:
    """Compute log-likelihood of LCA model."""
    n_obs = data.shape[0]
    n_classes = len(class_probs)
    
    log_likelihood = 0
    for i in range(n_obs):
        obs_prob = 0
        for c in range(n_classes):
            class_prob = class_probs[c]
            item_prob = np.prod(item_probs[c] ** data[i] * (1 - item_probs[c]) ** (1 - data[i]))
            obs_prob += class_prob * item_prob
        log_likelihood += np.log(obs_prob + 1e-10)
    
    return log_likelihood


def fit_lca(data: np.ndarray, n_classes: int, max_iter: int = 100, 
            tol: float = 1e-4, n_init: int = 10, seed: int = 42) -> dict:
    """Fit Latent Class Analysis model using EM algorithm."""
    best_ll = -np.inf
    best_result = None
    
    for init in range(n_init):
        class_probs, item_probs = initialize_lca_parameters(n_classes, data.shape[1], seed=seed + init)
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            responsibilities = lca_e_step(data, class_probs, item_probs)
            class_probs, item_probs = lca_m_step(data, responsibilities)
            ll = compute_lca_log_likelihood(data, class_probs, item_probs)
            
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        
        if ll > best_ll:
            best_ll = ll
            best_result = {
                'class_probs': class_probs,
                'item_probs': item_probs,
                'responsibilities': responsibilities,
                'log_likelihood': ll,
                'n_iter': iteration + 1,
                'n_classes': n_classes
            }
    
    n_obs, n_items = data.shape
    n_params = (n_classes - 1) + n_classes * n_items
    best_result['bic'] = -2 * best_ll + n_params * np.log(n_obs)
    best_result['aic'] = -2 * best_ll + 2 * n_params
    best_result['class_assignments'] = best_result['responsibilities'].argmax(axis=1)
    
    return best_result


def compute_lca_coordinates(class_probs: np.ndarray, item_probs: np.ndarray, 
                            responsibilities: np.ndarray) -> tuple:
    """Compute coordinates for LCA biplot visualization."""
    product_coords = item_probs.T  # Items x Classes
    household_coords = responsibilities  # Households x Classes
    return household_coords, product_coords


# =============================================================================
# TETRACHORIC CORRELATION & FACTOR ANALYSIS
# =============================================================================

def compute_tetrachoric_single(x: np.ndarray, y: np.ndarray) -> float:
    """Compute tetrachoric correlation between two binary variables."""
    from scipy.stats import multivariate_normal
    
    a = np.sum((x == 1) & (y == 1))
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    d = np.sum((x == 0) & (y == 0))
    n = a + b + c + d
    
    if n == 0 or min(a+b, c+d, a+c, b+d) == 0:
        return 0.0
    
    p1 = (a + b) / n
    p2 = (a + c) / n
    
    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return 0.0
    
    h1 = norm.ppf(p1)
    h2 = norm.ppf(p2)
    
    pobs = a / n
    
    def objective(rho):
        upper = np.array([h1, h2])
        cov = np.array([[1, rho], [rho, 1]])
        try:
            p = multivariate_normal.cdf(upper, mean=np.zeros(2), cov=cov)
        except:
            return 1.0
        return (p - pobs) ** 2
    
    result = minimize_scalar(objective, bounds=(-0.99, 0.99), method='bounded')
    return result.x


def compute_tetrachoric_matrix(data: np.ndarray, progress_callback=None) -> np.ndarray:
    """Compute full tetrachoric correlation matrix."""
    n_items = data.shape[1]
    corr_matrix = np.eye(n_items)
    
    total_pairs = n_items * (n_items - 1) // 2
    pair_count = 0
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            r = compute_tetrachoric_single(data[:, i], data[:, j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
            pair_count += 1
            if progress_callback:
                progress_callback(pair_count / total_pairs)
    
    return corr_matrix


def varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    """Apply varimax rotation to factor loadings."""
    n_items, n_factors = loadings.shape
    rotated = loadings.copy()
    
    for _ in range(max_iter):
        old_rotated = rotated.copy()
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                u = rotated[:, i] ** 2 - rotated[:, j] ** 2
                v = 2 * rotated[:, i] * rotated[:, j]
                
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)
                
                phi = 0.25 * np.arctan2(D - 2 * A * B / n_items, 
                                        C - (A ** 2 - B ** 2) / n_items)
                
                c, s = np.cos(phi), np.sin(phi)
                new_i = c * rotated[:, i] + s * rotated[:, j]
                new_j = -s * rotated[:, i] + c * rotated[:, j]
                rotated[:, i] = new_i
                rotated[:, j] = new_j
        
        if np.allclose(rotated, old_rotated, atol=tol):
            break
    
    return rotated


def factor_analysis_principal_axis(corr_matrix: np.ndarray, n_factors: int, 
                                   max_iter: int = 100, tol: float = 1e-4) -> dict:
    """Principal Axis Factor Analysis with varimax rotation."""
    n_items = corr_matrix.shape[0]
    
    communalities = 1 - 1 / np.diag(np.linalg.inv(corr_matrix + 0.01 * np.eye(n_items)))
    communalities = np.clip(communalities, 0.1, 0.99)
    
    for iteration in range(max_iter):
        reduced_corr = corr_matrix.copy()
        np.fill_diagonal(reduced_corr, communalities)
        
        eigenvalues, eigenvectors = np.linalg.eigh(reduced_corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        pos_eigenvalues = np.maximum(eigenvalues[:n_factors], 0.01)
        loadings = eigenvectors[:, :n_factors] * np.sqrt(pos_eigenvalues)
        
        new_communalities = np.sum(loadings ** 2, axis=1)
        new_communalities = np.clip(new_communalities, 0.1, 0.99)
        
        if np.max(np.abs(new_communalities - communalities)) < tol:
            break
        communalities = new_communalities
    
    loadings = varimax_rotation(loadings)
    
    var_explained = np.sum(loadings ** 2, axis=0)
    var_explained_pct = var_explained / n_items * 100
    
    sort_idx = np.argsort(var_explained)[::-1]
    loadings = loadings[:, sort_idx]
    var_explained = var_explained[sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    return {
        'loadings': loadings,
        'communalities': communalities,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'n_iter': iteration + 1
    }


def compute_factor_scores_regression(data: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Compute factor scores using the regression (Thurstone) method."""
    data_centered = data - data.mean(axis=0)
    
    cov_matrix = np.cov(data_centered.T)
    try:
        weights = np.linalg.solve(cov_matrix + 0.01 * np.eye(cov_matrix.shape[0]), loadings)
    except:
        weights = loadings
    
    scores = data_centered @ weights
    return scores


# =============================================================================
# BAYESIAN FACTOR MODEL (Variational Inference)
# =============================================================================

def fit_bayesian_factor_vi(data: np.ndarray, n_factors: int, max_iter: int = 100, 
                           tol: float = 1e-4, seed: int = 42) -> dict:
    """Bayesian Factor Analysis using Variational Inference."""
    np.random.seed(seed)
    n_obs, n_items = data.shape
    
    m_z = np.random.randn(n_obs, n_factors) * 0.1
    m_lambda = np.random.randn(n_items, n_factors) * 0.1
    s_lambda = np.ones((n_items, n_factors)) * 0.1
    
    a_tau = np.ones(n_items)
    b_tau = np.ones(n_items)
    
    prior_lambda_precision = 1.0
    prior_tau_a = 1.0
    prior_tau_b = 1.0
    
    elbo_history = []
    
    for iteration in range(max_iter):
        E_tau = a_tau / b_tau
        
        precision_z = np.eye(n_factors) + (m_lambda.T * E_tau) @ m_lambda
        for i in range(n_obs):
            cov_z = np.linalg.inv(precision_z + 0.01 * np.eye(n_factors))
            m_z[i] = cov_z @ (m_lambda.T * E_tau) @ data[i]
        
        E_zz = m_z.T @ m_z + n_obs * np.eye(n_factors) * 0.01
        for j in range(n_items):
            precision_lambda = prior_lambda_precision * np.eye(n_factors) + E_tau[j] * E_zz
            s_lambda[j] = np.diag(np.linalg.inv(precision_lambda + 0.01 * np.eye(n_factors)))
            m_lambda[j] = np.linalg.solve(
                precision_lambda + 0.01 * np.eye(n_factors),
                E_tau[j] * m_z.T @ data[:, j]
            )
        
        for j in range(n_items):
            residuals = data[:, j] - m_z @ m_lambda[j]
            a_tau[j] = prior_tau_a + n_obs / 2
            b_tau[j] = prior_tau_b + 0.5 * (np.sum(residuals ** 2) + 
                                            np.trace(E_zz * np.outer(m_lambda[j], m_lambda[j])))
        
        reconstruction = m_z @ m_lambda.T
        recon_error = np.sum((data - reconstruction) ** 2)
        elbo = -0.5 * recon_error - 0.5 * np.sum(m_lambda ** 2)
        elbo_history.append(elbo)
        
        if iteration > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            break
    
    m_lambda = varimax_rotation(m_lambda)
    
    var_explained = np.sum(m_lambda ** 2, axis=0)
    var_explained_pct = var_explained / n_items * 100
    
    sort_idx = np.argsort(var_explained)[::-1]
    m_lambda = m_lambda[:, sort_idx]
    var_explained = var_explained[sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    return {
        'loadings': m_lambda,
        'loadings_std': np.sqrt(s_lambda),
        'scores': m_z,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'elbo_history': elbo_history,
        'n_iter': iteration + 1
    }


# =============================================================================
# NMF
# =============================================================================

def fit_nmf(data: np.ndarray, n_components: int, max_iter: int = 200) -> dict:
    """Fit Non-negative Matrix Factorization."""
    model = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
    W = model.fit_transform(data)
    H = model.components_
    
    var_explained = np.sum(H ** 2, axis=1)
    var_explained_pct = var_explained / data.shape[1] * 100
    
    sort_idx = np.argsort(var_explained)[::-1]
    H = H[sort_idx]
    W = W[:, sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    return {
        'W': W,
        'H': H,
        'loadings': H.T,
        'scores': W,
        'var_explained_pct': var_explained_pct,
        'reconstruction_error': model.reconstruction_err_,
        'n_iter': model.n_iter_
    }


# =============================================================================
# MCA
# =============================================================================

def fit_mca(data: np.ndarray, n_components: int, product_names: list = None) -> dict:
    """Fit Multiple Correspondence Analysis."""
    if not PRINCE_AVAILABLE:
        raise ImportError("prince is not installed. Install with: pip install prince")
    
    n_obs, n_items = data.shape
    
    if product_names is None:
        product_names = [f"item_{i}" for i in range(n_items)]
    
    df = pd.DataFrame(data.astype(int), columns=product_names)
    
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    mca.fit(df)
    
    row_coords = mca.row_coordinates(df).values
    col_coords = mca.column_coordinates(df)
    
    product_coords = []
    product_labels = []
    
    for prod in product_names:
        key_1 = f"{prod}__1"
        if key_1 in col_coords.index:
            product_coords.append(col_coords.loc[key_1].values)
            product_labels.append(prod)
    
    product_coords = np.array(product_coords)
    
    eigenvalues = mca.eigenvalues_
    total_inertia = mca.total_inertia_
    explained_inertia = mca.percentage_of_variance_/100
    
    var_explained_pct = np.array(explained_inertia) * 100
    
    col_contribs = mca.column_contributions_
    
    product_contribs = []
    for prod in product_names:
        key_1 = f"{prod}__1"
        if key_1 in col_contribs.index:
            product_contribs.append(col_contribs.loc[key_1].values)
    product_contribs = np.array(product_contribs)
    
    if len(product_coords) > 0:
        norms = np.linalg.norm(product_coords, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        coords_normalized = product_coords / norms
        similarity_matrix = coords_normalized @ coords_normalized.T
    else:
        similarity_matrix = np.eye(n_items)
    
    return {
        'row_coordinates': row_coords,
        'column_coordinates': product_coords,
        'product_labels': product_labels,
        'eigenvalues': eigenvalues,
        'explained_inertia': explained_inertia,
        'var_explained_pct': var_explained_pct,
        'total_inertia': total_inertia,
        'contributions': product_contribs,
        'similarity_matrix': similarity_matrix,
        'n_components': n_components,
        'mca_model': mca
    }


# =============================================================================
# BAYESIAN FACTOR MODEL (PyMC)
# =============================================================================

def fit_bayesian_factor_model_pymc(data: np.ndarray, n_factors: int,
                                    n_samples: int = 1000, n_tune: int = 500) -> dict:
    """Bayesian Factor Model using PyMC with lower triangular identification."""
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available. Install with: pip install pymc")
    
    n_obs, n_items = data.shape
    
    if n_items < n_factors:
        raise ValueError(f"Need at least {n_factors} items for {n_factors} factors, got {n_items}")
    
    diag_rows = list(range(n_factors))
    diag_cols = list(range(n_factors))
    
    lower_rows = []
    lower_cols = []
    for i in range(1, n_factors):
        for j in range(i):
            lower_rows.append(i)
            lower_cols.append(j)
    n_lower = len(lower_rows)
    
    remaining_rows = []
    remaining_cols = []
    for i in range(n_factors, n_items):
        for j in range(n_factors):
            remaining_rows.append(i)
            remaining_cols.append(j)
    n_remaining = len(remaining_rows)
    
    diag_flat_idx = [r * n_factors + c for r, c in zip(diag_rows, diag_cols)]
    lower_flat_idx = [r * n_factors + c for r, c in zip(lower_rows, lower_cols)] if n_lower > 0 else []
    remaining_flat_idx = [r * n_factors + c for r, c in zip(remaining_rows, remaining_cols)] if n_remaining > 0 else []
    
    with pm.Model() as factor_model:
        diag_loadings = pm.LogNormal('diag_loadings', mu=0, sigma=0.5, shape=n_factors)
        
        if n_lower > 0:
            lower_loadings = pm.Normal('lower_loadings', mu=0, sigma=1.0, shape=n_lower)
        
        if n_remaining > 0:
            remaining_loadings = pm.Normal('remaining_loadings', mu=0, sigma=1.0, shape=n_remaining)
        
        loadings_flat = pt.zeros(n_items * n_factors)
        
        loadings_flat = pt.set_subtensor(loadings_flat[diag_flat_idx], diag_loadings)
        
        if n_lower > 0:
            loadings_flat = pt.set_subtensor(loadings_flat[lower_flat_idx], lower_loadings)
        
        if n_remaining > 0:
            loadings_flat = pt.set_subtensor(loadings_flat[remaining_flat_idx], remaining_loadings)
        
        loadings = loadings_flat.reshape((n_items, n_factors))
        
        factors = pm.Normal('factors', mu=0, sigma=1, shape=(n_obs, n_factors))
        
        sigma = pm.HalfNormal('sigma', sigma=1.0, shape=n_items)
        
        mu = pm.math.dot(factors, loadings.T)
        likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
        
        trace = pm.sample(
            n_samples, 
            tune=n_tune, 
            nuts_sampler='nutpie',
            progressbar=True, 
            return_inferencedata=True,
            target_accept=0.95,
            random_seed=42
        )
        
        trace = pm.compute_log_likelihood(trace)
    
    loadings_samples = _reconstruct_loadings_samples(
        trace, n_items, n_factors, n_lower, n_remaining,
        diag_flat_idx, 
        lower_flat_idx,
        remaining_flat_idx
    )
    
    loadings_mean = loadings_samples.mean(axis=0)
    loadings_std = loadings_samples.std(axis=0)
    
    var_explained = np.sum(loadings_mean ** 2, axis=0)
    var_explained_pct = var_explained / n_items * 100
    
    sort_idx = np.argsort(var_explained)[::-1]
    loadings_mean = loadings_mean[:, sort_idx]
    loadings_std = loadings_std[:, sort_idx]
    var_explained = var_explained[sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    try:
        waic = az.waic(trace)
    except Exception:
        waic = None
    
    n_divergences = int(trace.sample_stats['diverging'].sum().values)
    
    return {
        'loadings': loadings_mean,
        'loadings_std': loadings_std,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'trace': trace,
        'waic': waic,
        'n_divergences': n_divergences
    }


def _reconstruct_loadings_samples(trace, n_items, n_factors, n_lower, n_remaining,
                                   diag_idx, lower_idx, remaining_idx):
    """Reconstruct the full loadings matrix from the constrained posterior samples."""
    diag_samples = trace.posterior['diag_loadings'].values
    n_chains, n_draws = diag_samples.shape[:2]
    n_total_samples = n_chains * n_draws
    
    diag_samples = diag_samples.reshape(n_total_samples, n_factors)
    
    if n_lower > 0:
        lower_samples = trace.posterior['lower_loadings'].values.reshape(n_total_samples, n_lower)
    
    if n_remaining > 0:
        remaining_samples = trace.posterior['remaining_loadings'].values.reshape(n_total_samples, n_remaining)
    
    loadings_samples = np.zeros((n_total_samples, n_items, n_factors))
    
    for s in range(n_total_samples):
        flat = np.zeros(n_items * n_factors)
        
        flat[diag_idx] = diag_samples[s]
        
        if n_lower > 0:
            flat[lower_idx] = lower_samples[s]
        
        if n_remaining > 0:
            flat[remaining_idx] = remaining_samples[s]
        
        loadings_samples[s] = flat.reshape(n_items, n_factors)
    
    return loadings_samples


# =============================================================================
# DISCRETE CHOICE MODEL (PyMC)
# =============================================================================

def fit_discrete_choice_model(purchase_data: np.ndarray, 
                               household_features: np.ndarray = None,
                               product_features: np.ndarray = None,
                               n_samples: int = 1000, 
                               n_tune: int = 500,
                               include_random_effects: bool = False,
                               n_latent_features: int = 0,
                               latent_prior_scale: float = 1.0) -> dict:
    """Fit Discrete Choice Model using PyMC."""
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available")
    
    n_households, n_items = purchase_data.shape
    
    with pm.Model() as dcm:
        alpha = pm.Normal('alpha', mu=0, sigma=2, shape=n_items)
        
        utility = alpha
        
        if household_features is not None:
            n_hh_features = household_features.shape[1]
            beta = pm.Normal('beta', mu=0, sigma=1, shape=(n_items, n_hh_features))
            hh_effect = pm.math.dot(household_features, beta.T)
            utility = utility + hh_effect
        
        if product_features is not None:
            n_prod_features = product_features.shape[1]
            gamma = pm.Normal('gamma', mu=0, sigma=1, shape=n_prod_features)
            prod_effect = pm.math.dot(product_features, gamma)
            utility = utility + prod_effect
        
        if include_random_effects:
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=1)
            hh_random = pm.Normal('hh_random', mu=0, sigma=sigma_hh, shape=n_households)
            utility = utility + hh_random[:, None]
        
        if n_latent_features > 0:
            lambda_sd = pm.HalfNormal('lambda_sd', sigma=latent_prior_scale)
            theta_sd = pm.HalfNormal('theta_sd', sigma=latent_prior_scale)
            
            product_latent = pm.Normal('product_latent', mu=0, sigma=lambda_sd, 
                                       shape=(n_items, n_latent_features))
            household_latent = pm.Normal('household_latent', mu=0, sigma=theta_sd,
                                         shape=(n_households, n_latent_features))
            
            latent_utility = pm.math.dot(household_latent, product_latent.T)
            utility = utility + latent_utility
        
        p = pm.math.sigmoid(utility)
        likelihood = pm.Bernoulli('obs', p=p, observed=purchase_data)
        
        trace = pm.sample(
            n_samples, 
            tune=n_tune, 
            nuts_sampler='nutpie',
            progressbar=True,
            return_inferencedata=True,
            target_accept=0.9,
            random_seed=42
        )
        
        trace = pm.compute_log_likelihood(trace)
    
    result = {
        'alpha': trace.posterior['alpha'].mean(dim=['chain', 'draw']).values,
        'alpha_std': trace.posterior['alpha'].std(dim=['chain', 'draw']).values,
        'trace': trace,
        'n_latent_features': n_latent_features
    }
    
    if household_features is not None:
        result['beta'] = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        result['beta_std'] = trace.posterior['beta'].std(dim=['chain', 'draw']).values
    
    if product_features is not None:
        result['gamma'] = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        result['gamma_std'] = trace.posterior['gamma'].std(dim=['chain', 'draw']).values
    
    if n_latent_features > 0:
        result['product_latent'] = trace.posterior['product_latent'].mean(dim=['chain', 'draw']).values
        result['product_latent_std'] = trace.posterior['product_latent'].std(dim=['chain', 'draw']).values
        result['household_latent'] = trace.posterior['household_latent'].mean(dim=['chain', 'draw']).values
        result['household_latent_std'] = trace.posterior['household_latent'].std(dim=['chain', 'draw']).values
        result['lambda_sd'] = float(trace.posterior['lambda_sd'].mean().values)
        result['theta_sd'] = float(trace.posterior['theta_sd'].mean().values)
    
    try:
        result['waic'] = az.waic(trace)
    except:
        result['waic'] = None
    
    result['n_divergences'] = int(trace.sample_stats['diverging'].sum().values)
    
    return result


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def create_export_zip(model_result, model_type, product_columns, product_embeddings,
                      household_embeddings, var_explained, similarity_matrix,
                      cluster_result=None, original_data=None) -> bytes:
    """Create a ZIP file with all analysis results."""
    zip_buffer = io.BytesIO()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        metadata = {
            'model_type': model_type,
            'export_timestamp': timestamp,
            'n_products': len(product_columns),
            'product_names': list(product_columns),
            'files_included': []
        }
        
        if product_embeddings is not None:
            if model_type == "Multiple Correspondence Analysis (MCA)":
                export_product_labels = model_result.get('product_labels', product_columns)
            else:
                export_product_labels = product_columns
            
            prod_df = pd.DataFrame(
                product_embeddings,
                index=export_product_labels,
                columns=[f"Dim_{i+1}" for i in range(product_embeddings.shape[1])]
            )
            csv_buffer = io.StringIO()
            prod_df.to_csv(csv_buffer)
            zf.writestr('product_vectors.csv', csv_buffer.getvalue())
            metadata['files_included'].append('product_vectors.csv')
        
        if household_embeddings is not None:
            hh_df = pd.DataFrame(
                household_embeddings,
                columns=[f"Dim_{i+1}" for i in range(household_embeddings.shape[1])]
            )
            csv_buffer = io.StringIO()
            hh_df.to_csv(csv_buffer)
            zf.writestr('household_vectors.csv', csv_buffer.getvalue())
            metadata['files_included'].append('household_vectors.csv')
        
        if similarity_matrix is not None:
            if model_type == "Multiple Correspondence Analysis (MCA)":
                sim_labels = model_result.get('product_labels', product_columns)
            else:
                sim_labels = product_columns
            sim_df = pd.DataFrame(similarity_matrix, index=sim_labels, columns=sim_labels)
            csv_buffer = io.StringIO()
            sim_df.to_csv(csv_buffer)
            zf.writestr('similarity_matrix.csv', csv_buffer.getvalue())
            metadata['files_included'].append('similarity_matrix.csv')
        
        if var_explained is not None:
            var_df = pd.DataFrame({
                'Dimension': [f"Dim_{i+1}" for i in range(len(var_explained))],
                'Variance_Explained_Pct': var_explained,
                'Cumulative_Pct': np.cumsum(var_explained)
            })
            csv_buffer = io.StringIO()
            var_df.to_csv(csv_buffer, index=False)
            zf.writestr('variance_explained.csv', csv_buffer.getvalue())
            metadata['files_included'].append('variance_explained.csv')
        
        if cluster_result is not None:
            if model_type == "Multiple Correspondence Analysis (MCA)":
                cluster_labels = model_result.get('product_labels', product_columns)
            else:
                cluster_labels = product_columns
            cluster_df = pd.DataFrame({
                'Product': cluster_labels,
                'Cluster': cluster_result['labels'] + 1
            })
            csv_buffer = io.StringIO()
            cluster_df.to_csv(csv_buffer, index=False)
            zf.writestr('cluster_assignments.csv', csv_buffer.getvalue())
            metadata['files_included'].append('cluster_assignments.csv')
        
        if original_data is not None:
            orig_df = pd.DataFrame(original_data, columns=product_columns)
            csv_buffer = io.StringIO()
            orig_df.to_csv(csv_buffer, index=False)
            zf.writestr('original_data.csv', csv_buffer.getvalue())
            metadata['files_included'].append('original_data.csv')
        
        model_summary = {
            'model_type': model_type,
            'n_products': len(product_columns),
            'n_households': household_embeddings.shape[0] if household_embeddings is not None else 0,
            'n_dimensions': product_embeddings.shape[1] if product_embeddings is not None else 0,
            'total_variance_explained': float(np.sum(var_explained)) if var_explained is not None else None,
        }
        
        if model_result is not None:
            if 'log_likelihood' in model_result:
                model_summary['log_likelihood'] = float(model_result['log_likelihood'])
            if 'bic' in model_result:
                model_summary['bic'] = float(model_result['bic'])
            if 'aic' in model_result:
                model_summary['aic'] = float(model_result['aic'])
            if 'n_iter' in model_result:
                model_summary['n_iterations'] = int(model_result['n_iter'])
            if 'n_classes' in model_result:
                model_summary['n_classes'] = int(model_result['n_classes'])
            if 'total_inertia' in model_result:
                model_summary['total_inertia'] = float(model_result['total_inertia'])
            if 'waic' in model_result and model_result['waic'] is not None:
                try:
                    model_summary['waic'] = float(model_result['waic'].elpd_waic)
                except:
                    pass
            if 'n_divergences' in model_result:
                model_summary['n_divergences'] = int(model_result['n_divergences'])
        
        zf.writestr('model_summary.json', json.dumps(model_summary, indent=2))
        metadata['files_included'].append('model_summary.json')
        
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Latent Structure Analysis", layout="wide")
    
    st.title("🛒 Latent Structure Analysis for Purchase Data")
    st.markdown("""
    Discover latent customer segments and product relationships using multiple statistical methods.
    """)
    
    # =========== INITIALIZE ALL SESSION STATE ===========
    # Core model results
    if 'model_result' not in st.session_state:
        st.session_state.model_result = None
    if 'model_cache_key' not in st.session_state:
        st.session_state.model_cache_key = None
    if 'model_type_cached' not in st.session_state:
        st.session_state.model_type_cached = None
    if 'product_columns_cached' not in st.session_state:
        st.session_state.product_columns_cached = None
    if 'similarity_matrix_cached' not in st.session_state:
        st.session_state.similarity_matrix_cached = None
    if 'product_embeddings' not in st.session_state:
        st.session_state.product_embeddings = None
    if 'household_embeddings' not in st.session_state:
        st.session_state.household_embeddings = None
    if 'var_explained_cached' not in st.session_state:
        st.session_state.var_explained_cached = None
    if 'cluster_result' not in st.session_state:
        st.session_state.cluster_result = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    
    # Additional visualization data that needs to persist
    if 'tetra_corr_cached' not in st.session_state:
        st.session_state.tetra_corr_cached = None
    if 'elbo_history_cached' not in st.session_state:
        st.session_state.elbo_history_cached = None
    if 'convergence_msg' not in st.session_state:
        st.session_state.convergence_msg = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    
    # Check PyMC availability
    if not PYMC_AVAILABLE:
        if PYMC_ERROR:
            st.warning(f"⚠️ PyMC initialization issue: {PYMC_ERROR}. PyMC-based models will be unavailable.")
        else:
            st.warning("⚠️ PyMC not installed. PyMC-based models will be unavailable. Install with: `pip install pymc arviz`")
    
    if not PRINCE_AVAILABLE:
        st.info("💡 Install `prince` for Multiple Correspondence Analysis (MCA): `pip install prince`")
    
    # Sidebar
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
            return
        
        st.markdown("---")
        st.header("🔧 Model Selection")
        
        model_options = [
            "Latent Class Analysis (LCA)",
            "Factor Analysis (Tetrachoric)",
            "Bayesian Factor Model (VI)",
            "Non-negative Matrix Factorization (NMF)"
        ]
        
        if PRINCE_AVAILABLE:
            model_options.append("Multiple Correspondence Analysis (MCA)")
        
        if PYMC_AVAILABLE:
            model_options.extend([
                "Bayesian Factor Model (PyMC)",
                "Discrete Choice Model (PyMC)"
            ])
        
        model_type = st.selectbox(
            "Select Analysis Method",
            options=model_options,
            help="""
            **LCA**: Discrete latent segments  
            **Tetrachoric FA**: Continuous latent factors (proper for binary)  
            **Bayesian FA (VI)**: Fast variational inference  
            **NMF**: Parts-based decomposition  
            **MCA**: PCA for categorical data - ideal for binary purchase data  
            **Bayesian FA (PyMC)**: Full MCMC with uncertainty  
            **Discrete Choice (PyMC)**: Model with household/product features
            """
        )
    
    # Main content
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
        
        # For Discrete Choice Model: select feature columns
        household_feature_columns = []
        product_feature_df = None
        
        if model_type == "Discrete Choice Model (PyMC)":
            st.markdown("---")
            st.subheader("🏠 Household Features (optional)")
            
            remaining_cols = [c for c in available_cols if c not in product_columns]
            household_feature_columns = st.multiselect(
                "Select Household Feature columns",
                options=remaining_cols,
                help="Numeric features describing households (income, size, etc.)"
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
        
        # Store original data for export
        st.session_state.original_data = X
        
        # Create a hash of the data for cache invalidation
        data_hash = str(hash(X.tobytes()))
        
        st.success(f"Ready to analyze {X.shape[0]} households across {X.shape[1]} products")
        
        # Model-specific configuration
        st.header("🔬 Model Configuration")
        
        if model_type == "Latent Class Analysis (LCA)":
            col1, col2, col3 = st.columns(3)
            with col1:
                n_classes = st.slider("Number of Classes", 2, 10, 3)
            with col2:
                n_init = st.slider("Number of Initializations", 1, 50, 10)
            with col3:
                max_iter = st.slider("Max Iterations", 50, 500, 100)
            
            compare_models = st.checkbox("Compare multiple class solutions", value=False)
            if compare_models:
                class_range = st.slider("Range of classes", 2, 10, (2, 5))
        
        elif model_type in ["Factor Analysis (Tetrachoric)", "Bayesian Factor Model (VI)"]:
            col1, col2 = st.columns(2)
            with col1:
                n_factors = st.slider("Number of Factors", 1, min(10, len(product_columns) - 1), 2)
            with col2:
                max_iter = st.slider("Max Iterations", 50, 500, 100)
        
        elif model_type == "Bayesian Factor Model (PyMC)":
            col1, col2, col3 = st.columns(3)
            with col1:
                n_factors = st.slider("Number of Factors", 1, min(10, len(product_columns) - 1), 2)
            with col2:
                n_samples = st.slider("MCMC Samples", 500, 3000, 1000)
            with col3:
                n_tune = st.slider("Tuning Samples", 200, 1000, 500)
        
        elif model_type == "Non-negative Matrix Factorization (NMF)":
            col1, col2 = st.columns(2)
            with col1:
                n_components = st.slider("Number of Components", 1, min(10, len(product_columns) - 1), 2)
            with col2:
                max_iter = st.slider("Max Iterations", 50, 500, 200)
        
        elif model_type == "Multiple Correspondence Analysis (MCA)":
            n_components = st.slider("Number of Dimensions", 2, min(10, len(product_columns) - 1), 3)
        
        elif model_type == "Discrete Choice Model (PyMC)":
            col1, col2, col3 = st.columns(3)
            with col1:
                n_samples = st.slider("MCMC Samples", 500, 3000, 1000)
            with col2:
                n_tune = st.slider("Tuning Samples", 200, 1000, 500)
            with col3:
                include_random_effects = st.checkbox("Include Household Random Effects", value=False)
            
            st.markdown("##### Latent Product Features")
            include_latent_features = st.checkbox(
                "Include Latent Product Features", 
                value=False,
                help="Add latent dimensions that capture unobserved product characteristics"
            )
            if include_latent_features:
                col1, col2 = st.columns(2)
                with col1:
                    n_latent_features = st.slider("Number of Latent Dimensions", 1, min(5, len(product_columns) - 1), 2)
                with col2:
                    latent_prior_scale = st.slider("Prior Scale", 0.1, 2.0, 1.0)
            else:
                n_latent_features = 0
                latent_prior_scale = 1.0
        
        # Visualization options
        st.header("📈 Visualization Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_hierarchy = st.checkbox("Show Hierarchical Clustering", value=True)
        with col2:
            if show_hierarchy:
                linkage_method = st.selectbox("Linkage Method", 
                                             options=['average', 'complete', 'single', 'ward'],
                                             index=0)
            else:
                linkage_method = 'average'
        
        # Build model params dict for cache key
        if model_type == "Latent Class Analysis (LCA)":
            model_params = {'n_classes': n_classes, 'n_init': n_init, 'max_iter': max_iter}
        elif model_type in ["Factor Analysis (Tetrachoric)", "Bayesian Factor Model (VI)"]:
            model_params = {'n_factors': n_factors, 'max_iter': max_iter}
        elif model_type == "Bayesian Factor Model (PyMC)":
            model_params = {'n_factors': n_factors, 'n_samples': n_samples, 'n_tune': n_tune}
        elif model_type == "Non-negative Matrix Factorization (NMF)":
            model_params = {'n_components': n_components, 'max_iter': max_iter}
        elif model_type == "Multiple Correspondence Analysis (MCA)":
            model_params = {'n_components': n_components}
        elif model_type == "Discrete Choice Model (PyMC)":
            model_params = {
                'n_samples': n_samples, 
                'n_tune': n_tune, 
                'random_effects': include_random_effects,
                'n_latent_features': n_latent_features if include_latent_features else 0,
                'latent_prior_scale': latent_prior_scale if include_latent_features else 1.0
            }
        
        # Check if we need to invalidate the cache
        current_cache_key = get_model_cache_key(data_hash, model_type, model_params, tuple(product_columns))
        if st.session_state.model_cache_key != current_cache_key:
            # Parameters changed, invalidate cache
            st.session_state.model_result = None
            st.session_state.model_cache_key = None
            st.session_state.model_type_cached = None
            st.session_state.product_columns_cached = None
            st.session_state.similarity_matrix_cached = None
            st.session_state.product_embeddings = None
            st.session_state.household_embeddings = None
            st.session_state.var_explained_cached = None
            st.session_state.cluster_result = None
            st.session_state.tetra_corr_cached = None
            st.session_state.elbo_history_cached = None
            st.session_state.convergence_msg = None
            st.session_state.model_metrics = None
        
        # Run Analysis Button
        if st.button("🚀 Run Analysis", type="primary"):
            
            similarity_matrix = None
            tetra_corr = None
            elbo_history = None
            convergence_msg = None
            model_metrics = {}
            
            # =========== LCA ===========
            if model_type == "Latent Class Analysis (LCA)":
                st.header("📊 Latent Class Analysis Results")
                
                with st.spinner("Fitting LCA model..."):
                    result = fit_lca(X, n_classes, max_iter=max_iter, n_init=n_init)
                
                household_coords, product_coords = compute_lca_coordinates(
                    result['class_probs'], result['item_probs'], result['responsibilities']
                )
                
                residual_corr = compute_residual_correlations(X, result['responsibilities'], result['item_probs'])
                
                # Store everything in session state
                st.session_state.model_result = result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = residual_corr
                st.session_state.product_embeddings = product_coords
                st.session_state.household_embeddings = household_coords
                st.session_state.var_explained_cached = result['class_probs'] * 100
                st.session_state.cluster_result = None
                st.session_state.convergence_msg = f"Model converged in {result['n_iter']} iterations"
                st.session_state.model_metrics = {
                    'Log-Likelihood': result['log_likelihood'],
                    'BIC': result['bic'],
                    'AIC': result['aic']
                }
                
                st.success(st.session_state.convergence_msg)
            
            # =========== TETRACHORIC FA ===========
            elif model_type == "Factor Analysis (Tetrachoric)":
                st.header("📊 Factor Analysis (Tetrachoric Correlations)")
                
                progress_bar = st.progress(0)
                st.write("Computing tetrachoric correlations...")
                
                def update_progress(p):
                    progress_bar.progress(p)
                
                tetra_corr = compute_tetrachoric_matrix(X, progress_callback=update_progress)
                
                with st.spinner("Fitting factor model..."):
                    fa_result = factor_analysis_principal_axis(tetra_corr, n_factors, max_iter=max_iter)
                
                factor_scores = compute_factor_scores_regression(X, fa_result['loadings'])
                
                loadings_norm = fa_result['loadings'] / (np.linalg.norm(fa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                st.session_state.model_result = fa_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = implied_corr
                st.session_state.product_embeddings = fa_result['loadings']
                st.session_state.household_embeddings = factor_scores
                st.session_state.var_explained_cached = fa_result['var_explained_pct']
                st.session_state.cluster_result = None
                st.session_state.tetra_corr_cached = tetra_corr
                st.session_state.convergence_msg = f"Model converged in {fa_result['n_iter']} iterations"
                st.session_state.model_metrics = {}
                
                st.success(st.session_state.convergence_msg)
            
            # =========== BAYESIAN FA (VI) ===========
            elif model_type == "Bayesian Factor Model (VI)":
                st.header("📊 Bayesian Factor Model (Variational Inference)")
                
                with st.spinner("Fitting Bayesian factor model..."):
                    bfa_result = fit_bayesian_factor_vi(X, n_factors, max_iter=max_iter)
                
                loadings_norm = bfa_result['loadings'] / (np.linalg.norm(bfa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                st.session_state.model_result = bfa_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = implied_corr
                st.session_state.product_embeddings = bfa_result['loadings']
                st.session_state.household_embeddings = bfa_result['scores']
                st.session_state.var_explained_cached = bfa_result['var_explained_pct']
                st.session_state.cluster_result = None
                st.session_state.elbo_history_cached = bfa_result['elbo_history']
                st.session_state.convergence_msg = f"Model converged in {bfa_result['n_iter']} iterations"
                st.session_state.model_metrics = {}
                
                st.success(st.session_state.convergence_msg)
            
            # =========== BAYESIAN FA (PyMC) ===========
            elif model_type == "Bayesian Factor Model (PyMC)":
                st.header("📊 Bayesian Factor Model (PyMC MCMC)")
                
                with st.spinner("Running MCMC sampling... This may take a few minutes."):
                    bfa_result = fit_bayesian_factor_model_pymc(X, n_factors, n_samples=n_samples, n_tune=n_tune)
                
                loadings_norm = bfa_result['loadings'] / (np.linalg.norm(bfa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                factor_scores = compute_factor_scores_regression(X, bfa_result['loadings'])
                
                st.session_state.model_result = bfa_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = implied_corr
                st.session_state.product_embeddings = bfa_result['loadings']
                st.session_state.household_embeddings = factor_scores
                st.session_state.var_explained_cached = bfa_result['var_explained_pct']
                st.session_state.cluster_result = None
                st.session_state.convergence_msg = "MCMC sampling complete!"
                st.session_state.model_metrics = {
                    'WAIC': bfa_result['waic'].elpd_waic if bfa_result.get('waic') else None,
                    'Divergences': bfa_result.get('n_divergences', 0)
                }
                
                st.success(st.session_state.convergence_msg)
            
            # =========== NMF ===========
            elif model_type == "Non-negative Matrix Factorization (NMF)":
                st.header("📊 Non-negative Matrix Factorization")
                
                with st.spinner("Fitting NMF model..."):
                    nmf_result = fit_nmf(X, n_components, max_iter=max_iter)
                
                H_norm = nmf_result['H'] / (np.linalg.norm(nmf_result['H'], axis=1, keepdims=True) + 1e-10)
                similarity = H_norm.T @ H_norm
                
                st.session_state.model_result = nmf_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = similarity
                st.session_state.product_embeddings = nmf_result['loadings']
                st.session_state.household_embeddings = nmf_result['scores']
                st.session_state.var_explained_cached = nmf_result['var_explained_pct']
                st.session_state.cluster_result = None
                st.session_state.convergence_msg = f"Model converged in {nmf_result['n_iter']} iterations"
                st.session_state.model_metrics = {
                    'Reconstruction Error': nmf_result['reconstruction_error']
                }
                
                st.success(st.session_state.convergence_msg)
            
            # =========== MCA ===========
            elif model_type == "Multiple Correspondence Analysis (MCA)":
                st.header("📊 Multiple Correspondence Analysis")
                
                with st.spinner("Fitting MCA model..."):
                    mca_result = fit_mca(X, n_components, product_names=product_columns)
                
                st.session_state.model_result = mca_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = mca_result['similarity_matrix']
                st.session_state.product_embeddings = mca_result['column_coordinates']
                st.session_state.household_embeddings = mca_result['row_coordinates']
                st.session_state.var_explained_cached = mca_result['var_explained_pct']
                st.session_state.cluster_result = None
                st.session_state.convergence_msg = "MCA completed!"
                st.session_state.model_metrics = {
                    'Total Inertia': mca_result['total_inertia']
                }
                
                st.success(st.session_state.convergence_msg)
            
            # =========== DISCRETE CHOICE MODEL ===========
            elif model_type == "Discrete Choice Model (PyMC)":
                st.header("📊 Discrete Choice Model Results")
                
                hh_features = None
                if household_feature_columns:
                    hh_features = df[household_feature_columns].values.astype(float)
                    hh_features = (hh_features - hh_features.mean(axis=0)) / (hh_features.std(axis=0) + 1e-10)
                
                with st.spinner("Running MCMC sampling... This may take a few minutes."):
                    dcm_result = fit_discrete_choice_model(
                        X,
                        household_features=hh_features,
                        n_samples=n_samples,
                        n_tune=n_tune,
                        include_random_effects=include_random_effects,
                        n_latent_features=n_latent_features,
                        latent_prior_scale=latent_prior_scale
                    )
                
                if n_latent_features > 0:
                    product_embeddings = dcm_result['product_latent']
                    household_embeddings = dcm_result['household_latent']
                    
                    prod_latent_norm = dcm_result['product_latent'] / (
                        np.linalg.norm(dcm_result['product_latent'], axis=1, keepdims=True) + 1e-10
                    )
                    similarity = prod_latent_norm @ prod_latent_norm.T
                else:
                    product_embeddings = None
                    household_embeddings = None
                    similarity = None
                
                st.session_state.model_result = dcm_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = similarity
                st.session_state.product_embeddings = product_embeddings
                st.session_state.household_embeddings = household_embeddings
                st.session_state.var_explained_cached = None
                st.session_state.cluster_result = None
                st.session_state.convergence_msg = "MCMC sampling complete!"
                st.session_state.model_metrics = {
                    'WAIC': dcm_result['waic'].elpd_waic if dcm_result.get('waic') else None,
                    'Divergences': dcm_result.get('n_divergences', 0)
                }
                
                st.success(st.session_state.convergence_msg)
        
        # =========== UNIFIED VISUALIZATION SECTION (outside button block) ===========
        # This section displays results from session state, allowing UI changes without model rerun
        if st.session_state.model_result is not None and st.session_state.model_type_cached == model_type:
            model_result = st.session_state.model_result
            product_columns_cached = st.session_state.product_columns_cached
            product_embeddings = st.session_state.product_embeddings
            household_embeddings = st.session_state.household_embeddings
            var_explained = st.session_state.var_explained_cached
            similarity_matrix = st.session_state.similarity_matrix_cached
            
            st.markdown("---")
            st.header("📊 Model Results")
            
            # Show convergence message
            if st.session_state.convergence_msg:
                st.info(st.session_state.convergence_msg)
            
            # Show model metrics
            if st.session_state.model_metrics:
                cols = st.columns(len(st.session_state.model_metrics))
                for i, (name, value) in enumerate(st.session_state.model_metrics.items()):
                    with cols[i]:
                        if value is not None:
                            if isinstance(value, float):
                                st.metric(name, f"{value:.2f}")
                            else:
                                st.metric(name, value)
                        else:
                            st.metric(name, "N/A")
            
            # =========== MODEL-SPECIFIC VISUALIZATIONS ===========
            if model_type == "Latent Class Analysis (LCA)":
                st.subheader("Class Profiles")
                fig = plot_lca_profiles(model_result['item_probs'], model_result['class_probs'], product_columns_cached)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Residual Correlations (Substitution Patterns)")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Residual Correlations (negative = substitution)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Factor Analysis (Tetrachoric)":
                if st.session_state.tetra_corr_cached is not None:
                    st.subheader("Tetrachoric Correlation Matrix")
                    fig = plot_correlation_matrix(st.session_state.tetra_corr_cached, product_columns_cached, "Tetrachoric Correlations")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Factor Loadings (Varimax Rotated)")
                fig = plot_loadings_heatmap(model_result['loadings'], product_columns_cached)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Variance Explained")
                fig = plot_variance_explained(model_result['var_explained_pct'], "Factor Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Implied Correlations from Factor Model")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Bayesian Factor Model (VI)":
                if st.session_state.elbo_history_cached is not None:
                    st.subheader("Convergence (ELBO)")
                    fig = plot_convergence(st.session_state.elbo_history_cached, 
                                          "Evidence Lower Bound (ELBO) Over Iterations", "ELBO")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Factor Loadings (Posterior Means, Varimax Rotated)")
                fig = plot_loadings_heatmap(model_result['loadings'], product_columns_cached)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Variance Explained")
                fig = plot_variance_explained(model_result['var_explained_pct'], "Bayesian FA (VI)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Implied Correlations from Factor Model")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Bayesian Factor Model (PyMC)":
                st.subheader("Factor Loadings (Posterior Means ± Std)")
                fig = plot_loadings_with_uncertainty(model_result['loadings'], model_result['loadings_std'], product_columns_cached)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Variance Explained")
                fig = plot_variance_explained(model_result['var_explained_pct'], "Bayesian FA (PyMC)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Implied Correlations from Bayesian FA")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Non-negative Matrix Factorization (NMF)":
                st.subheader("Component Loadings")
                fig = plot_loadings_heatmap(model_result['loadings'], product_columns_cached, title="NMF Component Loadings")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Variance Explained")
                fig = plot_variance_explained(model_result['var_explained_pct'], "NMF")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Product Similarity (from NMF)")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Cosine Similarity (NMF)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Multiple Correspondence Analysis (MCA)":
                st.subheader("Explained Inertia (Variance)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Inertia", f"{model_result['total_inertia']:.4f}")
                with col2:
                    total_explained = sum(model_result['var_explained_pct'][:model_result['n_components']])
                    st.metric("Total Explained", f"{total_explained:.1f}%")
                
                fig = plot_variance_explained(model_result['var_explained_pct'], "MCA")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Product Contributions to Dimensions")
                st.caption("Higher contribution = product is more important in defining that dimension")
                
                fig = plot_mca_contributions(
                    model_result['contributions'],
                    model_result['product_labels'],
                    n_dims=min(3, model_result['n_components'])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Product Coordinates (Dimension Loadings)")
                dim_labels = [f"Dim {k+1}" for k in range(model_result['n_components'])]
                fig = plot_loadings_heatmap(
                    model_result['column_coordinates'],
                    model_result['product_labels'],
                    dim_labels,
                    "Product Coordinates in MCA Space"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Product Similarity (from MCA)")
                fig = plot_correlation_matrix(
                    model_result['similarity_matrix'],
                    model_result['product_labels'],
                    "Product Similarity (cosine in MCA space)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Discrete Choice Model (PyMC)":
                st.subheader("Product Intercepts (Baseline Probabilities)")
                fig = plot_product_intercepts(model_result['alpha'], model_result['alpha_std'], product_columns_cached)
                st.plotly_chart(fig, use_container_width=True)
                
                if 'beta' in model_result:
                    st.subheader("Household Feature Effects")
                    fig = plot_beta_coefficients(model_result['beta'], model_result['beta_std'], 
                                                product_columns_cached, household_feature_columns)
                    st.plotly_chart(fig, use_container_width=True)
                
                if model_result.get('n_latent_features', 0) > 0:
                    st.subheader("Product Latent Features")
                    dim_labels = [f"Latent {k+1}" for k in range(model_result['n_latent_features'])]
                    fig = plot_loadings_heatmap(
                        model_result['product_latent'],
                        product_columns_cached,
                        dim_labels,
                        "Product Latent Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if similarity_matrix is not None:
                        st.subheader("Product Similarity (from Latent Features)")
                        fig = plot_correlation_matrix(
                            similarity_matrix, 
                            product_columns_cached,
                            "Product Similarity (cosine in latent space)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Household Latent Preference Distribution")
                    fig = go.Figure()
                    for k in range(model_result['n_latent_features']):
                        fig.add_trace(go.Violin(
                            y=model_result['household_latent'][:, k],
                            name=f"Latent {k+1}",
                            box_visible=True,
                            meanline_visible=True
                        ))
                    fig.update_layout(
                        title='Household Latent Preferences by Dimension',
                        yaxis_title='Preference Value',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # =========== BIPLOT SECTION (for all models with embeddings) ===========
            if product_embeddings is not None and product_embeddings.shape[1] >= 2:
                st.markdown("---")
                st.subheader("🎯 Product Biplot")
                st.caption("Explore products in the latent space. Products close together have similar patterns.")
                
                n_dims = product_embeddings.shape[1]
                dim_options = list(range(n_dims))
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    dim1 = st.selectbox("X-axis dimension", dim_options, index=0, key="biplot_dim1")
                with col2:
                    dim2 = st.selectbox("Y-axis dimension", dim_options, 
                                       index=1 if n_dims > 1 else 0, key="biplot_dim2")
                with col3:
                    show_households = st.checkbox("Show households", value=True, key="show_hh",
                                                  help="Show household points in the biplot")
                
                # Warn if household embeddings have fewer dimensions
                if household_embeddings is not None and show_households:
                    hh_n_dims = household_embeddings.shape[1]
                    if max(dim1, dim2) >= hh_n_dims:
                        st.warning(f"⚠️ Household embeddings only have {hh_n_dims} dimensions. "
                                   f"Households will not be shown for dimensions beyond {hh_n_dims}.")
                
                # Get cluster labels if available
                cluster_labels = None
                if st.session_state.cluster_result is not None:
                    cluster_labels = st.session_state.cluster_result['labels']
                
                # Handle MCA product labels
                if model_type == "Multiple Correspondence Analysis (MCA)":
                    prod_labels = model_result['product_labels']
                else:
                    prod_labels = product_columns_cached
                
                fig = plot_generic_biplot(
                    row_coords=household_embeddings if show_households else None,
                    col_coords=product_embeddings,
                    product_labels=prod_labels,
                    dim1=dim1,
                    dim2=dim2,
                    var_explained=list(var_explained) if var_explained is not None else None,
                    model_name=model_type.split(" (")[0],
                    cluster_labels=cluster_labels,
                    show_households=show_households
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # =========== PRODUCT CLUSTERING SECTION ===========
            if product_embeddings is not None:
                st.markdown("---")
                st.subheader("🔮 Product Clustering")
                st.caption("Cluster products based on their positions in the latent space.")
                
                if len(product_columns_cached) >= 3:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        cluster_method = st.selectbox(
                            "Clustering method",
                            options=["K-Means", "Hierarchical"],
                            key="cluster_method"
                        )
                    
                    with col2:
                        max_k = min(10, len(product_columns_cached) - 1)
                        if max_k >= 2:
                            auto_k = st.checkbox("Auto-detect optimal K", value=False, key="auto_k")
                        else:
                            auto_k = False
                    
                    with col3:
                        if not auto_k and max_k >= 2:
                            n_clusters = st.slider("Number of clusters", 2, max_k, 
                                                  min(3, max_k), key="n_clusters")
                        else:
                            n_clusters = 2
                    
                    if max_k >= 2:
                        if st.button("🔄 Run Clustering", key="run_clustering"):
                            with st.spinner("Clustering products..."):
                                if auto_k:
                                    opt_result = find_optimal_clusters(product_embeddings, max_k)
                                    n_clusters = opt_result['optimal_k']
                                    st.info(f"Optimal number of clusters: {n_clusters} (based on silhouette score)")
                                    
                                    if len(opt_result['scores']) > 0:
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=opt_result['range'],
                                            y=opt_result['scores'],
                                            mode='lines+markers',
                                            name='Silhouette Score'
                                        ))
                                        fig.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                                                     annotation_text=f"Optimal K={n_clusters}")
                                        fig.update_layout(
                                            title='Silhouette Score by Number of Clusters',
                                            xaxis_title='Number of Clusters (K)',
                                            yaxis_title='Silhouette Score',
                                            height=300
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                if cluster_method == "K-Means":
                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                    labels = kmeans.fit_predict(product_embeddings)
                                    silhouette = silhouette_score(product_embeddings, labels)
                                else:
                                    hc_result = compute_hierarchical_clustering(similarity_matrix, method='average')
                                    labels = fcluster(hc_result['linkage_matrix'], n_clusters, criterion='maxclust') - 1
                                    silhouette = silhouette_score(product_embeddings, labels)
                                
                                st.session_state.cluster_result = {
                                    'labels': labels,
                                    'n_clusters': n_clusters,
                                    'silhouette': silhouette,
                                    'method': cluster_method
                                }
                                
                                st.success(f"Clustering complete! Silhouette score: {silhouette:.3f}")
                    
                    # Show cluster results if available
                    if st.session_state.cluster_result is not None:
                        cluster_result = st.session_state.cluster_result
                        
                        st.markdown(f"**{cluster_result['method']} Clustering** - "
                                   f"{cluster_result['n_clusters']} clusters, "
                                   f"Silhouette: {cluster_result['silhouette']:.3f}")
                        
                        if model_type == "Multiple Correspondence Analysis (MCA)":
                            prod_labels = model_result['product_labels']
                        else:
                            prod_labels = product_columns_cached
                        
                        cluster_df = get_cluster_members(cluster_result['labels'], prod_labels)
                        
                        n_clusters_actual = cluster_result['n_clusters']
                        cols = st.columns(min(n_clusters_actual, 4))
                        for i in range(n_clusters_actual):
                            with cols[i % len(cols)]:
                                cluster_products = cluster_df[cluster_df['Cluster'] == i + 1]['Product'].tolist()
                                st.markdown(f"**Cluster {i + 1}** ({len(cluster_products)} products)")
                                for prod in cluster_products:
                                    st.write(f"• {prod}")
                        
                        fig = plot_cluster_summary(
                            product_embeddings, 
                            cluster_result['labels'],
                            prod_labels,
                            title="Cluster Summary"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 3 products to perform clustering.")
            
            # =========== HIERARCHICAL CLUSTERING / DENDROGRAM ===========
            if show_hierarchy and similarity_matrix is not None:
                st.markdown("---")
                st.subheader("🌳 Hierarchical Product Relationships")
                hc_result = compute_hierarchical_clustering(similarity_matrix, method=linkage_method)
                
                if model_type == "Multiple Correspondence Analysis (MCA)":
                    prod_labels = model_result['product_labels']
                else:
                    prod_labels = product_columns_cached
                
                fig = plot_dendrogram(hc_result['linkage_matrix'], prod_labels,
                                      f"Product Hierarchy ({model_type.split(' (')[0]})")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Reading the Dendrogram:**
                - Products that merge at **lower heights** are more similar
                - Clusters that form early suggest strong relationships
                - Large jumps in height indicate natural groupings
                """)
            
            # =========== EXPORT / DOWNLOAD SECTION ===========
            st.markdown("---")
            st.subheader("📥 Export Results")
            st.caption("Download all analysis results as a ZIP file for later use in Python, R, or other tools.")
            
            try:
                if model_type == "Multiple Correspondence Analysis (MCA)":
                    export_product_labels = model_result.get('product_labels', product_columns_cached)
                else:
                    export_product_labels = product_columns_cached
                
                zip_data = create_export_zip(
                    model_result=model_result,
                    model_type=model_type,
                    product_columns=export_product_labels,
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
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📦 Download All Results (ZIP)",
                        data=zip_data,
                        file_name=filename,
                        mime="application/zip"
                    )
                
                with col2:
                    if product_embeddings is not None:
                        prod_df = pd.DataFrame(
                            product_embeddings,
                            index=export_product_labels,
                            columns=[f"Dim_{i+1}" for i in range(product_embeddings.shape[1])]
                        )
                        csv_buffer = io.StringIO()
                        prod_df.to_csv(csv_buffer)
                        st.download_button(
                            label="📊 Product Vectors (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"product_vectors_{timestamp}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    if similarity_matrix is not None:
                        sim_df = pd.DataFrame(
                            similarity_matrix,
                            index=export_product_labels,
                            columns=export_product_labels
                        )
                        csv_buffer = io.StringIO()
                        sim_df.to_csv(csv_buffer)
                        st.download_button(
                            label="🔗 Similarity Matrix (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"similarity_matrix_{timestamp}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error creating export: {e}")
        
        # =========== INTERPRETATION GUIDE ===========
        with st.expander("📖 How to Interpret Results"):
            st.markdown("""
            ### Model Comparison
            
            | Model | Best For | Output |
            |-------|----------|--------|
            | **LCA** | Discrete customer segments | Class memberships |
            | **Tetrachoric FA** | Continuous latent factors | Factor loadings |
            | **Bayesian FA (VI)** | Fast approximation | Point estimates |
            | **NMF** | Parts-based decomposition | Non-negative components |
            | **MCA** | Binary/categorical data (no assumptions) | Dimension coordinates |
            | **Bayesian FA (PyMC)** | Full uncertainty | Posteriors + credible intervals |
            | **Discrete Choice (PyMC)** | Understanding drivers | Feature coefficients |
            
            ### Discrete Choice Model with Latent Features
            
            When latent product features are enabled, the DCM learns:
            - **Product Latent Features (Λ)**: Unobserved characteristics of each product
            - **Household Latent Preferences (Θ)**: How much each household values each latent attribute
            - **Utility**: α + (household features) + Θ × Λᵀ
            
            **Interpretation:**
            - Products with similar latent features are substitutes
            - Households with similar preferences have correlated purchases
            - High loadings indicate defining characteristics
            - The biplot shows products and households in the same latent space
            
            ### Biplot Interpretation
            
            - **Product positions**: Products close together have similar purchase patterns
            - **Household points**: Show where individual households fall in the latent space
            - **Dimensions**: Each axis represents a latent factor or component
            - **Variance explained**: Shows how much information each dimension captures
            
            ### Product Clustering
            
            - **K-Means**: Partitions products into K non-overlapping clusters
            - **Hierarchical**: Creates a tree of clusters (use dendrogram to visualize)
            - **Silhouette Score**: Measures cluster quality (-1 to 1, higher is better)
            - **Auto-detect K**: Uses silhouette score to find optimal number of clusters
            
            ### MCA Interpretation
            
            - **Biplot**: Products close together → similar purchase patterns. Households near a product → likely buyers.
            - **Dimensions**: Each dimension captures a "shopping style" or product affinity pattern.
            - **Contributions**: Shows which products define each dimension most strongly.
            - **Inertia**: MCA's analog to variance explained. Total inertia = (n_categories/n_variables) - 1 for binary data.
            
            ### General Tips
            
            - **Hierarchical clustering**: Use to identify natural product groupings
            - **Negative residual correlations** in LCA suggest substitution patterns
            - **High loadings** on multiple factors suggest products that bridge categories
            - **Cluster products** to create actionable segments for marketing/assortment
            """)


if __name__ == "__main__":
    main()