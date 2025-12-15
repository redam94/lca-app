"""
Streamlit App for Latent Structure Analysis on Binary Purchase Data
Supports: LCA, Factor Analysis (Tetrachoric), Bayesian Factor Models (VI & PyMC), 
          NMF, MCA, Discrete Choice Model (PyMC)
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure PyTensor to avoid C compilation issues on macOS (especially Apple Silicon)
# This must be done BEFORE importing PyMC
import os
os.environ.setdefault("PYTENSOR_FLAGS", "device=cpu,floatX=float64,optimizer=fast_compile,cxx=")

# Check for prince (MCA) availability
PRINCE_AVAILABLE = False
try:
    import prince
    PRINCE_AVAILABLE = True
except ImportError:
    pass

# Check for PyMC availability
PYMC_AVAILABLE = False
PYMC_ERROR = None
try:
    import pytensor
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError as e:
    PYMC_ERROR = str(e)
except Exception as e:
    PYMC_ERROR = str(e)


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
    
    eps = 1e-10
    item_probs_clipped = np.clip(item_probs, eps, 1 - eps)
    
    log_resp = np.zeros((n_obs, n_classes))
    
    for k in range(n_classes):
        log_lik = (
            data @ np.log(item_probs_clipped[k]) +
            (1 - data) @ np.log(1 - item_probs_clipped[k])
        )
        log_resp[:, k] = np.log(class_probs[k]) + log_lik
    
    log_resp_norm = log_resp - logsumexp(log_resp, axis=1, keepdims=True)
    return np.exp(log_resp_norm)


def lca_m_step(data: np.ndarray, responsibilities: np.ndarray) -> tuple:
    """M-step: Update parameters given responsibilities."""
    n_obs = data.shape[0]
    class_counts = responsibilities.sum(axis=0)
    class_probs = class_counts / n_obs
    item_probs = (responsibilities.T @ data) / class_counts[:, np.newaxis]
    item_probs = np.clip(item_probs, 0.001, 0.999)
    return class_probs, item_probs


def compute_lca_log_likelihood(data: np.ndarray, class_probs: np.ndarray, item_probs: np.ndarray) -> float:
    """Compute total log-likelihood of the data."""
    n_obs = data.shape[0]
    n_classes = len(class_probs)
    
    eps = 1e-10
    item_probs_clipped = np.clip(item_probs, eps, 1 - eps)
    
    log_lik_matrix = np.zeros((n_obs, n_classes))
    
    for k in range(n_classes):
        log_lik = (
            data @ np.log(item_probs_clipped[k]) +
            (1 - data) @ np.log(1 - item_probs_clipped[k])
        )
        log_lik_matrix[:, k] = np.log(class_probs[k]) + log_lik
    
    return logsumexp(log_lik_matrix, axis=1).sum()


def fit_lca(data: np.ndarray, n_classes: int, max_iter: int = 100, tol: float = 1e-6, 
            n_init: int = 10, seed: int = 42) -> dict:
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
        # Use multivariate_normal.cdf for bivariate normal probability
        upper = np.array([h1, h2])
        cov = np.array([[1, rho], [rho, 1]])
        try:
            p = multivariate_normal.cdf(upper, mean=np.zeros(2), cov=cov)
        except:
            return 1.0  # Return high error if computation fails
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


def factor_analysis_principal_axis(corr_matrix: np.ndarray, n_factors: int, 
                                   max_iter: int = 100, tol: float = 1e-4) -> dict:
    """Principal Axis Factor Analysis with varimax rotation."""
    n_items = corr_matrix.shape[0]
    communalities = np.ones(n_items) * 0.5
    
    for iteration in range(max_iter):
        reduced_corr = corr_matrix.copy()
        np.fill_diagonal(reduced_corr, communalities)
        
        eigenvalues, eigenvectors = np.linalg.eigh(reduced_corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues_pos = np.maximum(eigenvalues[:n_factors], 0)
        loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues_pos)
        
        new_communalities = np.sum(loadings ** 2, axis=1)
        new_communalities = np.clip(new_communalities, 0.001, 0.999)
        
        if np.max(np.abs(new_communalities - communalities)) < tol:
            break
        communalities = new_communalities
    
    loadings = varimax_rotation(loadings)
    
    var_explained = np.sum(loadings ** 2, axis=0)
    total_var = n_items
    var_explained_pct = var_explained / total_var * 100
    
    return {
        'loadings': loadings,
        'communalities': communalities,
        'eigenvalues': eigenvalues[:n_factors],
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'n_iter': iteration + 1,
        'n_factors': n_factors
    }


def varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Apply varimax rotation to factor loadings."""
    n_items, n_factors = loadings.shape
    rotated = loadings.copy()
    
    for _ in range(max_iter):
        old_rotated = rotated.copy()
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                x = rotated[:, i]
                y = rotated[:, j]
                
                u = x ** 2 - y ** 2
                v = 2 * x * y
                
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)
                
                phi = 0.25 * np.arctan2(D - 2 * A * B / n_items, 
                                        C - (A ** 2 - B ** 2) / n_items)
                
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                rotated[:, i] = x * cos_phi + y * sin_phi
                rotated[:, j] = -x * sin_phi + y * cos_phi
        
        if np.max(np.abs(rotated - old_rotated)) < tol:
            break
    
    return rotated


# =============================================================================
# BAYESIAN FACTOR MODEL (Variational Inference)
# =============================================================================

def fit_bayesian_factor_model_vi(data: np.ndarray, n_factors: int, max_iter: int = 100,
                                  tol: float = 1e-4) -> dict:
    """Fit Bayesian factor model using variational inference."""
    n_obs, n_items = data.shape
    
    np.random.seed(42)
    lambda_mean = np.random.randn(n_items, n_factors) * 0.1
    lambda_var = np.ones((n_items, n_factors)) * 0.1
    
    z_mean = np.random.randn(n_obs, n_factors) * 0.1
    z_var = np.ones((n_obs, n_factors)) * 0.1
    
    elbo_history = []
    
    for iteration in range(max_iter):
        for i in range(n_obs):
            for k in range(n_factors):
                expected_lambda_sq = lambda_mean[:, k] ** 2 + lambda_var[:, k]
                
                precision = 1 + np.sum(expected_lambda_sq)
                z_var[i, k] = 1 / precision
                
                residual = data[i] - z_mean[i] @ lambda_mean.T
                residual += z_mean[i, k] * lambda_mean[:, k]
                
                z_mean[i, k] = z_var[i, k] * np.sum(residual * lambda_mean[:, k])
        
        for j in range(n_items):
            for k in range(n_factors):
                expected_z_sq = z_mean[:, k] ** 2 + z_var[:, k]
                
                precision = 1 + np.sum(expected_z_sq)
                lambda_var[j, k] = 1 / precision
                
                residual = data[:, j] - z_mean @ lambda_mean[j]
                residual += z_mean[:, k] * lambda_mean[j, k]
                
                lambda_mean[j, k] = lambda_var[j, k] * np.sum(residual * z_mean[:, k])
        
        reconstruction = z_mean @ lambda_mean.T
        recon_error = -0.5 * np.sum((data - reconstruction) ** 2)
        kl_z = -0.5 * np.sum(1 + np.log(z_var) - z_mean ** 2 - z_var)
        kl_lambda = -0.5 * np.sum(1 + np.log(lambda_var) - lambda_mean ** 2 - lambda_var)
        elbo = recon_error - kl_z - kl_lambda
        elbo_history.append(elbo)
        
        if iteration > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            break
    
    loadings = varimax_rotation(lambda_mean)
    
    var_explained = np.sum(loadings ** 2, axis=0)
    total_var = n_items
    var_explained_pct = var_explained / total_var * 100
    
    return {
        'loadings': loadings,
        'loadings_std': np.sqrt(lambda_var),
        'scores': z_mean,
        'scores_std': np.sqrt(z_var),
        'elbo_history': elbo_history,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'n_iter': iteration + 1,
        'n_factors': n_factors
    }


# =============================================================================
# BAYESIAN FACTOR MODEL (PyMC - Full MCMC)
# =============================================================================

def fit_bayesian_factor_model_pymc(data: np.ndarray, n_factors: int, n_samples: int = 1000,
                                    n_tune: int = 500) -> dict:
    """Fit Bayesian factor model using PyMC MCMC."""
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available. Please install with: pip install pymc arviz")
    
    n_obs, n_items = data.shape
    
    with pm.Model() as model:
        loadings = pm.Normal('loadings', mu=0, sigma=1, shape=(n_items, n_factors))
        scores = pm.Normal('scores', mu=0, sigma=1, shape=(n_obs, n_factors))
        prob = pm.math.sigmoid(pm.math.dot(scores, loadings.T))
        likelihood = pm.Bernoulli('obs', p=prob, observed=data)
        
        trace = pm.sample(n_samples, tune=n_tune, cores=1, 
                         return_inferencedata=True, progressbar=True,
                         target_accept=0.9)
    
    loadings_samples = trace.posterior['loadings'].values.reshape(-1, n_items, n_factors)
    loadings_mean = loadings_samples.mean(axis=0)
    loadings_std = loadings_samples.std(axis=0)
    
    loadings_rotated = varimax_rotation(loadings_mean)
    
    var_explained = np.sum(loadings_rotated ** 2, axis=0)
    var_explained_pct = var_explained / n_items * 100
    
    summary = az.summary(trace)
    
    try:
        waic = az.waic(trace)
    except:
        waic = None
    
    return {
        'loadings': loadings_rotated,
        'loadings_std': loadings_std,
        'trace': trace,
        'summary': summary,
        'waic': waic,
        'var_explained_pct': var_explained_pct,
        'n_factors': n_factors,
        'n_divergences': trace.sample_stats.diverging.sum().values
    }


# =============================================================================
# DISCRETE CHOICE MODEL (PyMC)
# =============================================================================

def fit_discrete_choice_model_pymc(data: np.ndarray, household_features: np.ndarray = None,
                                    product_features: np.ndarray = None,
                                    product_names: list = None,
                                    include_random_effects: bool = True,
                                    n_samples: int = 1000, n_tune: int = 500) -> dict:
    """Fit discrete choice model with household and product features using PyMC."""
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available")
    
    n_obs, n_items = data.shape
    
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=2, shape=n_items)
        
        utility = alpha
        
        if household_features is not None:
            n_hh_features = household_features.shape[1]
            hh_features_std = (household_features - household_features.mean(0)) / (household_features.std(0) + 1e-10)
            
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=(n_items, n_hh_features))
            beta = pm.Deterministic('beta', beta_raw * 0.5)
            
            utility = utility + pm.math.dot(hh_features_std, beta.T)
        
        if product_features is not None:
            n_prod_features = product_features.shape[1]
            prod_features_std = (product_features - product_features.mean(0)) / (product_features.std(0) + 1e-10)
            
            gamma = pm.Normal('gamma', mu=0, sigma=1, shape=n_prod_features)
            prod_effect = pm.math.dot(prod_features_std, gamma)
            utility = utility + prod_effect
        
        if include_random_effects:
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=0.5)
            hh_effect_raw = pm.Normal('hh_effect_raw', mu=0, sigma=1, shape=n_obs)
            hh_effect = pm.Deterministic('hh_effect', hh_effect_raw * sigma_hh)
            utility = utility + hh_effect[:, None]
        
        prob = pm.math.sigmoid(utility)
        likelihood = pm.Bernoulli('obs', p=prob, observed=data)
        
        trace = pm.sample(n_samples, tune=n_tune, cores=1,
                         return_inferencedata=True, progressbar=True,
                         target_accept=0.95)
    
    alpha_samples = trace.posterior['alpha'].values.reshape(-1, n_items)
    alpha_mean = alpha_samples.mean(axis=0)
    alpha_std = alpha_samples.std(axis=0)
    
    result = {
        'alpha': alpha_mean,
        'alpha_std': alpha_std,
        'trace': trace,
        'n_divergences': trace.sample_stats.diverging.sum().values
    }
    
    if household_features is not None:
        beta_samples = trace.posterior['beta'].values.reshape(-1, n_items, n_hh_features)
        result['beta'] = beta_samples.mean(axis=0)
        result['beta_std'] = beta_samples.std(axis=0)
    
    if product_features is not None:
        gamma_samples = trace.posterior['gamma'].values.reshape(-1, n_prod_features)
        result['gamma'] = gamma_samples.mean(axis=0)
        result['gamma_std'] = gamma_samples.std(axis=0)
    
    if include_random_effects:
        result['sigma_hh'] = trace.posterior['sigma_hh'].values.mean()
    
    try:
        result['waic'] = az.waic(trace)
    except:
        result['waic'] = None
    
    return result


# =============================================================================
# NON-NEGATIVE MATRIX FACTORIZATION (NMF)
# =============================================================================

def fit_nmf(data: np.ndarray, n_components: int, max_iter: int = 200, 
            tol: float = 1e-4, seed: int = 42) -> dict:
    """Fit NMF model using multiplicative update rules."""
    np.random.seed(seed)
    n_obs, n_items = data.shape
    
    W = np.abs(np.random.randn(n_obs, n_components)) * 0.1 + 0.1
    H = np.abs(np.random.randn(n_components, n_items)) * 0.1 + 0.1
    
    eps = 1e-10
    reconstruction_errors = []
    
    for iteration in range(max_iter):
        numerator = W.T @ data
        denominator = W.T @ W @ H + eps
        H *= numerator / denominator
        
        numerator = data @ H.T
        denominator = W @ H @ H.T + eps
        W *= numerator / denominator
        
        reconstruction = W @ H
        error = np.sum((data - reconstruction) ** 2)
        reconstruction_errors.append(error)
        
        if iteration > 0 and abs(reconstruction_errors[-1] - reconstruction_errors[-2]) < tol:
            break
    
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    H_normalized = H / norms
    W_scaled = W * norms.T
    
    total_var = np.sum(data ** 2)
    var_explained = []
    for k in range(n_components):
        recon_k = np.outer(W_scaled[:, k], H_normalized[k, :])
        var_k = np.sum(recon_k ** 2)
        var_explained.append(var_k)
    
    var_explained = np.array(var_explained)
    var_explained_pct = var_explained / total_var * 100
    
    return {
        'W': W_scaled,
        'H': H_normalized,
        'loadings': H_normalized.T,
        'scores': W_scaled,
        'reconstruction_error': reconstruction_errors[-1],
        'reconstruction_errors': reconstruction_errors,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'n_components': n_components,
        'n_iter': iteration + 1
    }


# =============================================================================
# MULTIPLE CORRESPONDENCE ANALYSIS (MCA) using prince
# =============================================================================

def fit_mca(data: np.ndarray, n_components: int, product_names: list = None) -> dict:
    """
    Multiple Correspondence Analysis for binary purchase data using prince.
    
    MCA is essentially PCA for categorical/binary data. It's appropriate for 
    0/1 purchase matrices because:
    - No normality assumptions
    - Handles binary data natively via indicator matrix expansion
    - Reveals co-purchase patterns as latent dimensions
    - Dimensions can be interpreted as shopping "styles" or product affinities
    
    Returns:
        - Column coordinates: Product positions in latent space
        - Row coordinates: Household positions in latent space  
        - Eigenvalues/inertia: Variance explained by each dimension
        - Contributions: How much each product contributes to each dimension
    """
    if not PRINCE_AVAILABLE:
        raise ImportError("prince is not installed. Install with: pip install prince")
    
    n_obs, n_items = data.shape
    
    # Create DataFrame with proper column names
    if product_names is None:
        product_names = [f"item_{i}" for i in range(n_items)]
    
    df = pd.DataFrame(data.astype(int), columns=product_names)
    
    # Convert to categorical (required for MCA)
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # Fit MCA
    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    mca.fit(df)
    
    # Get row coordinates (household positions in latent space)
    row_coords = mca.row_coordinates(df).values
    print("row_coords.shape:", row_coords.shape)
    # Get column coordinates (product/category positions)
    col_coords = mca.column_coordinates(df)
    print("col_coords.shape:", col_coords.shape)
    
    # Extract just the "1" (purchased) coordinates for each product
    # MCA creates two coordinates per binary variable: one for 0 and one for 1
    # We want the "1" coordinates to understand purchase patterns
    product_coords = []
    product_labels = []
    
    for prod in product_names:
        # Look for the "1" category coordinate
        key_1 = f"{prod}__1"
        if key_1 in col_coords.index:
            product_coords.append(col_coords.loc[key_1].values)
            product_labels.append(prod)
    
    product_coords = np.array(product_coords)
    print("product_coords.shape:", product_coords.shape)
    # Eigenvalues and explained inertia (variance)
    eigenvalues = mca.eigenvalues_
    total_inertia = mca.total_inertia_
    explained_inertia = mca.percentage_of_variance_/100
    
    # Percentage of variance explained
    var_explained_pct = np.array(explained_inertia) * 100
    
    # Column contributions (how much each variable contributes to each dimension)
    # This helps interpret what each dimension means
    col_contribs = mca.column_contributions_
    
    # Extract contributions for "purchased" categories only
    product_contribs = []
    for prod in product_names:
        key_1 = f"{prod}__1"
        if key_1 in col_contribs.index:
            product_contribs.append(col_contribs.loc[key_1].values)
    product_contribs = np.array(product_contribs)
    
    # Compute correlation-like matrix from product coordinates
    # Products close in MCA space have similar purchase patterns
    if len(product_coords) > 0:
        # Normalize coordinates
        norms = np.linalg.norm(product_coords, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        coords_normalized = product_coords / norms
        
        # Cosine similarity as correlation proxy
        similarity_matrix = coords_normalized @ coords_normalized.T
    else:
        similarity_matrix = np.eye(n_items)
    
    return {
        'row_coordinates': row_coords,  # Household scores
        'column_coordinates': product_coords,  # Product loadings
        'product_labels': product_labels,
        'eigenvalues': eigenvalues,
        'explained_inertia': explained_inertia,
        'var_explained_pct': var_explained_pct,
        'total_inertia': total_inertia,
        'contributions': product_contribs,  # Product contributions to each dim
        'similarity_matrix': similarity_matrix,
        'n_components': n_components,
        'mca_model': mca
    }


def plot_mca_biplot(row_coords: np.ndarray, col_coords: np.ndarray, 
                    product_labels: list, dim1: int = 0, dim2: int = 1,
                    var_explained: list = None) -> go.Figure:
    """
    Create MCA biplot showing products and households in latent space.
    """
    fig = go.Figure()
    print(row_coords.shape, col_coords.shape)
    #Plot households (row coordinates) as small points
    fig.add_trace(go.Scatter(
        x=row_coords[:, dim1],
        y=row_coords[:, dim2],
        mode='markers',
        marker=dict(size=4, color='lightblue', opacity=0.5),
        name='Households',
        hoverinfo='skip'
    ))
    
    # Plot products (column coordinates) as labeled points
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
    
    # Add origin lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Axis labels with variance explained
    if var_explained is not None:
        x_label = f"Dimension {dim1+1} ({var_explained[dim1]:.1f}%)"
        y_label = f"Dimension {dim2+1} ({var_explained[dim2]:.1f}%)"
    else:
        x_label = f"Dimension {dim1+1}"
        y_label = f"Dimension {dim2+1}"
    
    fig.update_layout(
        title='MCA Biplot: Products and Households in Latent Space',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        showlegend=True
    )
    
    return fig


def plot_mca_contributions(contributions: np.ndarray, product_labels: list,
                           n_dims: int = 3) -> go.Figure:
    """
    Plot product contributions to each MCA dimension.
    Higher contribution = product is more important in defining that dimension.
    """
    n_dims = min(n_dims, contributions.shape[1])
    
    fig = make_subplots(
        rows=1, cols=n_dims,
        subplot_titles=[f"Dimension {i+1}" for i in range(n_dims)]
    )
    
    for dim in range(n_dims):
        # Sort by contribution
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


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def compute_hierarchical_clustering(similarity_matrix: np.ndarray, method: str = 'average') -> dict:
    """Perform hierarchical clustering on items based on similarity matrix."""
    sim_matrix = similarity_matrix.copy()
    np.fill_diagonal(sim_matrix, 1)
    sim_matrix = np.clip(sim_matrix, -1, 1)
    
    distance_matrix = 1 - sim_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method=method)
    
    return {
        'linkage_matrix': Z,
        'distance_matrix': distance_matrix
    }


def plot_dendrogram(linkage_matrix: np.ndarray, labels: list, title: str = "Hierarchical Clustering") -> go.Figure:
    """Create a plotly dendrogram."""
    dend = dendrogram(linkage_matrix, labels=labels, no_plot=True, 
                      color_threshold=0, above_threshold_color='gray')
    
    fig = go.Figure()
    
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])
    
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(5, len(labels) * 10, 10)),
            ticktext=dend['ivl'],
            tickangle=45
        ),
        yaxis_title='Distance',
        height=500,
        showlegend=False
    )
    
    return fig


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_loadings_heatmap(loadings: np.ndarray, item_names: list, 
                          factor_labels: list = None, title: str = "Factor Loadings") -> go.Figure:
    """Create heatmap of factor loadings."""
    n_factors = loadings.shape[1]
    
    if factor_labels is None:
        factor_labels = [f"Factor {k+1}" for k in range(n_factors)]
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=factor_labels,
        y=item_names,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(loadings, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='Product: %{y}<br>%{x}<br>Loading: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Factor',
        yaxis_title='Product',
        height=max(400, 30 * len(item_names)),
    )
    
    return fig


def plot_loadings_with_uncertainty(loadings: np.ndarray, loadings_std: np.ndarray,
                                    item_names: list, title: str = "Factor Loadings with Uncertainty") -> go.Figure:
    """Create heatmap showing posterior mean and std."""
    n_items, n_factors = loadings.shape
    
    text_matrix = []
    for i in range(n_items):
        row = []
        for j in range(n_factors):
            row.append(f"{loadings[i, j]:.2f}±{loadings_std[i, j]:.2f}")
        text_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=[f"Factor {k+1}" for k in range(n_factors)],
        y=item_names,
        colorscale='RdBu_r',
        zmid=0,
        text=text_matrix,
        texttemplate='%{text}',
        textfont={'size': 9},
        hovertemplate='Product: %{y}<br>%{x}<br>Loading: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Factor',
        yaxis_title='Product',
        height=max(400, 30 * len(item_names)),
    )
    
    return fig


def plot_variance_explained(var_explained_pct: np.ndarray, model_name: str = "Model") -> go.Figure:
    """Plot variance explained by each component."""
    n_components = len(var_explained_pct)
    cumulative = np.cumsum(var_explained_pct)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=[f"Comp {i+1}" for i in range(n_components)], 
               y=var_explained_pct, name="Individual"),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=[f"Comp {i+1}" for i in range(n_components)], 
                   y=cumulative, name="Cumulative", mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"{model_name}: Variance Explained",
        height=350
    )
    fig.update_yaxes(title_text="% Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    
    return fig


def plot_lca_profiles(item_probs: np.ndarray, item_names: list, class_probs: np.ndarray) -> go.Figure:
    """Plot LCA class profiles."""
    n_classes = item_probs.shape[0]
    
    fig = go.Figure()
    
    for k in range(n_classes):
        fig.add_trace(go.Bar(
            name=f"Class {k+1} ({class_probs[k]*100:.1f}%)",
            x=item_names,
            y=item_probs[k],
        ))
    
    fig.update_layout(
        title='Purchase Probability by Latent Class',
        xaxis_title='Product',
        yaxis_title='P(Purchase | Class)',
        barmode='group',
        height=500,
        xaxis={'tickangle': 45}
    )
    
    return fig


def plot_correlation_matrix(corr_matrix: np.ndarray, item_names: list, 
                            title: str = "Correlation Matrix") -> go.Figure:
    """Create correlation heatmap."""
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
    # Convert to probabilities
    probs = 1 / (1 + np.exp(-alpha))
    
    # Sort by probability
    idx = np.argsort(probs)[::-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[product_names[i] for i in idx],
        y=probs[idx],
        error_y=dict(
            type='data',
            array=alpha_std[idx] * probs[idx] * (1 - probs[idx]),  # Delta method approx
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


def compute_residual_correlations(data: np.ndarray, responsibilities: np.ndarray, 
                                   item_probs: np.ndarray) -> np.ndarray:
    """Compute residual correlations after accounting for class membership."""
    expected = responsibilities @ item_probs
    residuals = data - expected
    return np.corrcoef(residuals.T)


# =============================================================================
# HELPER FUNCTION: Generate cache key for MCA results
# =============================================================================

def get_mca_cache_key(data_hash: str, n_components: int, product_columns: tuple) -> str:
    """Generate a unique cache key for MCA results based on input parameters."""
    return f"mca_{data_hash}_{n_components}_{hash(product_columns)}"


# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Latent Structure Analysis", layout="wide")
    
    st.title("🛒 Latent Structure Analysis for Purchase Data")
    st.markdown("""
    Discover latent customer segments and product relationships using multiple statistical methods.
    """)
    
    # Initialize session state for MCA results
    if 'mca_result' not in st.session_state:
        st.session_state.mca_result = None
    if 'mca_cache_key' not in st.session_state:
        st.session_state.mca_cache_key = None
    if 'mca_product_columns' not in st.session_state:
        st.session_state.mca_product_columns = None
    if 'mca_similarity_matrix' not in st.session_state:
        st.session_state.mca_similarity_matrix = None
    
    # Check PyMC availability
    if not PYMC_AVAILABLE:
        if PYMC_ERROR:
            st.warning(f"⚠️ PyMC initialization issue: {PYMC_ERROR}. PyMC-based models will be unavailable.")
        else:
            st.warning("⚠️ PyMC not installed. PyMC-based models will be unavailable. Install with: `pip install pymc arviz`")
    
    # Check prince availability
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
            
            st.subheader("📦 Product Features (optional)")
            use_product_features = st.checkbox("Upload separate product feature file")
            
            if use_product_features:
                product_file = st.file_uploader("Upload product features CSV", type=['csv'], key='prod_features')
                if product_file is not None:
                    product_feature_df = pd.read_csv(product_file)
                    st.dataframe(product_feature_df.head())
        
        if not product_columns:
            st.warning("Please select at least one product column")
            return
        
        # Prepare data
        data_subset = df[product_columns].copy()
        
        if data_subset.isnull().any().any():
            st.warning("Data contains missing values. Rows with missing values will be excluded.")
            data_subset = data_subset.dropna()
        
        X = data_subset.values.astype(float)
        
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
            n_components = st.slider("Number of Dimensions", 2, min(10, len(product_columns) - 1), 3,
                                    help="Number of latent dimensions to extract")
        
        elif model_type == "Discrete Choice Model (PyMC)":
            col1, col2, col3 = st.columns(3)
            with col1:
                n_samples = st.slider("MCMC Samples", 500, 3000, 1000)
            with col2:
                n_tune = st.slider("Tuning Samples", 200, 1000, 500)
            with col3:
                include_random_effects = st.checkbox("Include household random effects", value=True)
        
        # Hierarchical clustering option
        st.markdown("---")
        show_hierarchy = st.checkbox("Show hierarchical clustering of products", value=True)
        if show_hierarchy:
            linkage_method = st.selectbox(
                "Clustering linkage method",
                options=['average', 'complete', 'single']
            )
        
        # For MCA: Check if we need to invalidate the cache
        if model_type == "Multiple Correspondence Analysis (MCA)":
            current_cache_key = get_mca_cache_key(data_hash, n_components, tuple(product_columns))
            if st.session_state.mca_cache_key != current_cache_key:
                # Parameters changed, invalidate cache
                st.session_state.mca_result = None
                st.session_state.mca_cache_key = None
                st.session_state.mca_product_columns = None
                st.session_state.mca_similarity_matrix = None
        
        # Run Analysis
        if st.button("🚀 Run Analysis", type="primary"):
            
            similarity_matrix = None
            
            # =========== LCA ===========
            if model_type == "Latent Class Analysis (LCA)":
                
                if compare_models:
                    st.header("📈 Model Comparison")
                    results_list = []
                    progress_bar = st.progress(0)
                    
                    for i, k in enumerate(range(class_range[0], class_range[1] + 1)):
                        with st.spinner(f"Fitting {k}-class model..."):
                            result = fit_lca(X, k, max_iter=max_iter, n_init=n_init)
                            results_list.append({
                                'Classes': k,
                                'Log-Likelihood': result['log_likelihood'],
                                'BIC': result['bic'],
                                'AIC': result['aic']
                            })
                        progress_bar.progress((i + 1) / (class_range[1] - class_range[0] + 1))
                    
                    comparison_df = pd.DataFrame(results_list)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=comparison_df['Classes'], y=comparison_df['BIC'], 
                                           mode='lines+markers', name='BIC'))
                    fig.add_trace(go.Scatter(x=comparison_df['Classes'], y=comparison_df['AIC'], 
                                           mode='lines+markers', name='AIC'))
                    fig.update_layout(title='Model Selection Criteria', xaxis_title='Number of Classes',
                                     yaxis_title='Information Criterion', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    best_k = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Classes']
                    st.info(f"Suggested number of classes (lowest BIC): {int(best_k)}")
                
                st.header("📊 Latent Class Analysis Results")
                
                with st.spinner("Fitting LCA model..."):
                    result = fit_lca(X, n_classes, max_iter=max_iter, n_init=n_init)
                
                st.success(f"Model converged in {result['n_iter']} iterations")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Log-Likelihood", f"{result['log_likelihood']:.2f}")
                with col2:
                    st.metric("BIC", f"{result['bic']:.2f}")
                with col3:
                    st.metric("AIC", f"{result['aic']:.2f}")
                
                st.subheader("Class Profiles")
                fig = plot_lca_profiles(result['item_probs'], product_columns, result['class_probs'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Residual Correlations (Substitution Patterns)")
                residual_corr = compute_residual_correlations(X, result['responsibilities'], result['item_probs'])
                fig = plot_correlation_matrix(residual_corr, product_columns, 
                    "Residual Correlations (negative = substitution)")
                st.plotly_chart(fig, use_container_width=True)
                
                similarity_matrix = residual_corr
            
            # =========== TETRACHORIC FA ===========
            elif model_type == "Factor Analysis (Tetrachoric)":
                st.header("📊 Factor Analysis (Tetrachoric Correlations)")
                
                st.subheader("Step 1: Computing Tetrachoric Correlations")
                progress_bar = st.progress(0)
                
                with st.spinner("Computing tetrachoric correlation matrix..."):
                    tetra_corr = compute_tetrachoric_matrix(X, progress_callback=lambda p: progress_bar.progress(p))
                
                st.success("Tetrachoric correlation matrix computed!")
                
                fig = plot_correlation_matrix(tetra_corr, product_columns, "Tetrachoric Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Step 2: Factor Analysis")
                
                with st.spinner("Fitting factor analysis..."):
                    fa_result = factor_analysis_principal_axis(tetra_corr, n_factors, max_iter=max_iter)
                
                st.success(f"Factor analysis converged in {fa_result['n_iter']} iterations")
                
                st.subheader("Factor Loadings (Varimax Rotated)")
                fig = plot_loadings_heatmap(fa_result['loadings'], product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                fig = plot_variance_explained(fa_result['var_explained_pct'], "Tetrachoric FA")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Communalities")
                comm_df = pd.DataFrame({
                    'Product': product_columns,
                    'Communality': fa_result['communalities']
                }).sort_values('Communality', ascending=False)
                st.dataframe(comm_df, use_container_width=True)
                
                similarity_matrix = tetra_corr
            
            # =========== BAYESIAN FA (VI) ===========
            elif model_type == "Bayesian Factor Model (VI)":
                st.header("📊 Bayesian Factor Model (Variational Inference)")
                
                with st.spinner("Fitting Bayesian factor model..."):
                    bfa_result = fit_bayesian_factor_model_vi(X, n_factors, max_iter=max_iter)
                
                st.success(f"Model converged in {bfa_result['n_iter']} iterations")
                
                st.subheader("Convergence (ELBO)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=bfa_result['elbo_history'], mode='lines', name='ELBO'))
                fig.update_layout(title='Evidence Lower Bound (ELBO) Over Iterations',
                                  xaxis_title='Iteration', yaxis_title='ELBO', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Factor Loadings (Posterior Means, Varimax Rotated)")
                fig = plot_loadings_heatmap(bfa_result['loadings'], product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                fig = plot_variance_explained(bfa_result['var_explained_pct'], "Bayesian FA (VI)")
                st.plotly_chart(fig, use_container_width=True)
                
                loadings_norm = bfa_result['loadings'] / (np.linalg.norm(bfa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(implied_corr, product_columns, "Implied Correlations from Factor Model")
                st.plotly_chart(fig, use_container_width=True)
                
                similarity_matrix = implied_corr
            
            # =========== BAYESIAN FA (PyMC) ===========
            elif model_type == "Bayesian Factor Model (PyMC)":
                st.header("📊 Bayesian Factor Model (PyMC MCMC)")
                
                with st.spinner("Running MCMC sampling... This may take a few minutes."):
                    bfa_result = fit_bayesian_factor_model_pymc(X, n_factors, n_samples=n_samples, n_tune=n_tune)
                
                st.success("MCMC sampling complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if bfa_result.get('waic') is not None:
                        st.metric("WAIC", f"{bfa_result['waic'].elpd_waic:.2f}")
                    else:
                        st.metric("WAIC", "N/A")
                with col2:
                    n_div = bfa_result.get('n_divergences', 0)
                    if n_div > 0:
                        st.metric("⚠️ Divergences", n_div)
                    else:
                        st.metric("✅ Divergences", 0)
                
                st.subheader("Factor Loadings (Posterior Means ± Std)")
                fig = plot_loadings_with_uncertainty(bfa_result['loadings'], bfa_result['loadings_std'], product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                fig = plot_variance_explained(bfa_result['var_explained_pct'], "Bayesian FA (PyMC)")
                st.plotly_chart(fig, use_container_width=True)
                
                loadings_norm = bfa_result['loadings'] / (np.linalg.norm(bfa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(implied_corr, product_columns, "Implied Correlations from Bayesian FA")
                st.plotly_chart(fig, use_container_width=True)
                
                similarity_matrix = implied_corr
            
            # =========== NMF ===========
            elif model_type == "Non-negative Matrix Factorization (NMF)":
                st.header("📊 Non-negative Matrix Factorization")
                
                with st.spinner("Fitting NMF model..."):
                    nmf_result = fit_nmf(X, n_components, max_iter=max_iter)
                
                st.success(f"NMF converged in {nmf_result['n_iter']} iterations")
                
                st.subheader("Convergence")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=nmf_result['reconstruction_errors'], mode='lines', name='Reconstruction Error'))
                fig.update_layout(title='NMF Reconstruction Error Over Iterations',
                                  xaxis_title='Iteration', yaxis_title='Error', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Component Loadings (NMF)")
                component_labels = [f"Component {k+1}" for k in range(n_components)]
                fig = plot_loadings_heatmap(nmf_result['loadings'], product_columns, 
                                            component_labels, "NMF Components (items x components)")
                st.plotly_chart(fig, use_container_width=True)
                
                fig = plot_variance_explained(nmf_result['var_explained_pct'], "NMF")
                st.plotly_chart(fig, use_container_width=True)
                
                loadings_norm = nmf_result['loadings'] / (np.linalg.norm(nmf_result['loadings'], axis=1, keepdims=True) + 1e-10)
                cosine_sim = loadings_norm @ loadings_norm.T
                
                st.subheader("Product Similarity (from NMF)")
                fig = plot_correlation_matrix(cosine_sim, product_columns, "Cosine Similarity (NMF)")
                st.plotly_chart(fig, use_container_width=True)
                
                similarity_matrix = cosine_sim
            
            # =========== MCA ===========
            elif model_type == "Multiple Correspondence Analysis (MCA)":
                st.header("📊 Multiple Correspondence Analysis")
                
                st.info("""
                **MCA is PCA for categorical/binary data.** It's ideal for purchase matrices because:
                - No normality assumptions required
                - Handles binary 0/1 data natively
                - Reveals co-purchase patterns as latent dimensions
                - Dimensions can be interpreted as shopping "styles" or product affinities
                """)
                
                with st.spinner("Fitting MCA model..."):
                    mca_result = fit_mca(X, n_components, product_names=product_columns)
                
                # Store results in session state
                st.session_state.mca_result = mca_result
                st.session_state.mca_cache_key = get_mca_cache_key(data_hash, n_components, tuple(product_columns))
                st.session_state.mca_product_columns = product_columns
                st.session_state.mca_similarity_matrix = mca_result['similarity_matrix']
                
                st.success("MCA complete!")
                
                similarity_matrix = mca_result['similarity_matrix']
            
            # =========== DISCRETE CHOICE MODEL (PyMC) ===========
            elif model_type == "Discrete Choice Model (PyMC)":
                st.header("📊 Discrete Choice Model (PyMC)")
                
                st.info("""
                **Model uses non-centered parameterization** for hierarchical effects to reduce divergences.
                Features are standardized automatically.
                """)
                
                # Prepare household features
                hh_features = None
                if household_feature_columns:
                    hh_features = df.loc[data_subset.index, household_feature_columns].values
                    # Handle any missing values in features
                    hh_features = np.nan_to_num(hh_features, nan=0)
                
                # Prepare product features
                prod_features = None
                if product_feature_df is not None and use_product_features:
                    # Get numeric columns only (exclude 'product' column)
                    prod_feature_cols = [c for c in product_feature_df.columns if c != 'product']
                    prod_features = product_feature_df[prod_feature_cols].values.astype(float)
                
                with st.spinner("Running MCMC sampling for discrete choice model..."):
                    dcm_result = fit_discrete_choice_model_pymc(
                        X,
                        household_features=hh_features,
                        product_features=prod_features,
                        product_names=product_columns,
                        include_random_effects=include_random_effects,
                        n_samples=n_samples,
                        n_tune=n_tune
                    )
                
                st.success("MCMC sampling complete!")
                
                # Model diagnostics
                col1, col2 = st.columns(2)
                with col1:
                    if dcm_result.get('waic') is not None:
                        st.metric("WAIC", f"{dcm_result['waic'].elpd_waic:.2f}")
                    else:
                        st.metric("WAIC", "N/A")
                with col2:
                    n_div = dcm_result.get('n_divergences', 0)
                    if n_div > 0:
                        st.metric("⚠️ Divergences", n_div)
                    else:
                        st.metric("✅ Divergences", 0)
                
                if dcm_result.get('n_divergences', 0) > 0:
                    st.warning(f"Model had {dcm_result['n_divergences']} divergent transitions. "
                              "Consider: more tuning samples, higher target_accept, or simpler model.")
                
                # Product intercepts
                st.subheader("Product Baseline Probabilities")
                fig = plot_product_intercepts(dcm_result['alpha'], dcm_result['alpha_std'], product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                # Household feature effects
                if 'beta' in dcm_result:
                    st.subheader("Household Feature Effects")
                    fig = plot_beta_coefficients(dcm_result['beta'], dcm_result['beta_std'],
                                                 product_columns, household_feature_columns)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Product feature effects
                if 'gamma' in dcm_result:
                    st.subheader("Product Feature Effects")
                    prod_feature_cols = [c for c in product_feature_df.columns if c != 'product']
                    gamma_df = pd.DataFrame({
                        'Feature': prod_feature_cols,
                        'Coefficient': dcm_result['gamma'],
                        'Std': dcm_result['gamma_std']
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=gamma_df['Feature'],
                        y=gamma_df['Coefficient'],
                        error_y=dict(type='data', array=gamma_df['Std'], visible=True),
                        marker_color='coral'
                    ))
                    fig.update_layout(
                        title='Product Feature Effects (shared across households)',
                        xaxis_title='Product Feature',
                        yaxis_title='Coefficient',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Random effects
                if include_random_effects and 'sigma_hh' in dcm_result:
                    st.subheader("Household Heterogeneity")
                    st.metric("Household Random Effect SD", f"{dcm_result['sigma_hh']:.3f}")
                
                # Compute implied correlations from the model
                # Use product intercepts to compute similarity
                alpha = dcm_result['alpha']
                probs = 1 / (1 + np.exp(-alpha))
                
                # Correlation based on predicted probabilities across products
                # This is a simple approximation
                similarity_matrix = np.corrcoef(X.T)
            
            # =========== HIERARCHICAL CLUSTERING (for non-MCA models) ===========
            if model_type != "Multiple Correspondence Analysis (MCA)":
                if show_hierarchy and similarity_matrix is not None:
                    st.subheader("🌳 Hierarchical Product Relationships")
                    hc_result = compute_hierarchical_clustering(similarity_matrix, method=linkage_method)
                    fig = plot_dendrogram(hc_result['linkage_matrix'], product_columns,
                                          f"Product Hierarchy ({model_type})")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Reading the Dendrogram:**
                    - Products that merge at **lower heights** are more similar
                    - Clusters that form early suggest strong relationships
                    - Large jumps in height indicate natural groupings
                    """)
        
        # =========== MCA RESULTS DISPLAY (outside button block for state persistence) ===========
        if model_type == "Multiple Correspondence Analysis (MCA)" and st.session_state.mca_result is not None:
            mca_result = st.session_state.mca_result
            product_columns_cached = st.session_state.mca_product_columns
            
            # Variance explained
            st.subheader("Explained Inertia (Variance)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Inertia", f"{mca_result['total_inertia']:.4f}")
            with col2:
                total_explained = sum(mca_result['var_explained_pct'][:mca_result['n_components']])
                st.metric("Total Explained", f"{total_explained:.1f}%")
            
            fig = plot_variance_explained(mca_result['var_explained_pct'], "MCA")
            st.plotly_chart(fig, use_container_width=True)
            
            # Biplot with dimension selectors (these won't trigger model rerun)
            st.subheader("MCA Biplot")
            st.caption("Products close together have similar purchase patterns. Households near a product tend to buy it.")
            
            n_components_result = mca_result['n_components']
            if n_components_result >= 2:
                dim_options = list(range(n_components_result))
                col1, col2 = st.columns(2)
                with col1:
                    dim1 = st.selectbox("X-axis dimension", dim_options, index=0, key="mca_dim1")
                with col2:
                    dim2 = st.selectbox("Y-axis dimension", dim_options, index=1 if n_components_result > 1 else 0, key="mca_dim2")
                
                fig = plot_mca_biplot(
                    mca_result['row_coordinates'],
                    mca_result['column_coordinates'],
                    mca_result['product_labels'],
                    dim1=dim1,
                    dim2=dim2,
                    var_explained=list(mca_result['var_explained_pct'])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Product contributions
            st.subheader("Product Contributions to Dimensions")
            st.caption("Higher contribution = product is more important in defining that dimension")
            
            fig = plot_mca_contributions(
                mca_result['contributions'],
                mca_result['product_labels'],
                n_dims=min(3, n_components_result)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Product coordinates (loadings equivalent)
            st.subheader("Product Coordinates (Dimension Loadings)")
            dim_labels = [f"Dim {k+1}" for k in range(n_components_result)]
            loadings_df = pd.DataFrame(
                mca_result['column_coordinates'],
                index=mca_result['product_labels'],
                columns=dim_labels
            )
            fig = plot_loadings_heatmap(
                mca_result['column_coordinates'],
                mca_result['product_labels'],
                dim_labels,
                "Product Coordinates in MCA Space"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Product similarity
            st.subheader("Product Similarity (from MCA)")
            fig = plot_correlation_matrix(
                mca_result['similarity_matrix'],
                mca_result['product_labels'],
                "Product Similarity (cosine in MCA space)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Hierarchical clustering for MCA
            if show_hierarchy and st.session_state.mca_similarity_matrix is not None:
                st.subheader("🌳 Hierarchical Product Relationships")
                hc_result = compute_hierarchical_clustering(st.session_state.mca_similarity_matrix, method=linkage_method)
                fig = plot_dendrogram(hc_result['linkage_matrix'], product_columns_cached,
                                      f"Product Hierarchy (MCA)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Reading the Dendrogram:**
                - Products that merge at **lower heights** are more similar
                - Clusters that form early suggest strong relationships
                - Large jumps in height indicate natural groupings
                """)
        
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
            
            ### MCA Interpretation
            
            - **Biplot**: Products close together → similar purchase patterns. Households near a product → likely buyers.
            - **Dimensions**: Each dimension captures a "shopping style" or product affinity pattern.
            - **Contributions**: Shows which products define each dimension most strongly.
            - **Inertia**: MCA's analog to variance explained. Total inertia = (n_categories/n_variables) - 1 for binary data.
            
            ### General Tips
            
            - **Hierarchical clustering**: Use to identify natural product groupings
            - **Negative residual correlations** in LCA suggest substitution patterns
            - **High loadings** on multiple factors suggest products that bridge categories
            """)


if __name__ == "__main__":
    main()