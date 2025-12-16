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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
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
        
        trace = pm.sample(n_samples, tune=n_tune, nuts_sampler='nutpie', 
                         return_inferencedata=True, progressbar=True,
                         target_accept=0.9)
        trace = pm.compute_log_likelihood(trace)
    
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


def fit_latent_factor_dcm_pymc(data: np.ndarray, n_factors: int,
                                household_features: np.ndarray = None,
                                include_random_effects: bool = True,
                                n_samples: int = 1000, n_tune: int = 500) -> dict:
    """
    Fit a joint latent factor discrete choice model using PyMC.
    
    This model simultaneously estimates:
    1. Latent product factors (like factor analysis)
    2. Household preferences over those latent factors
    3. Optional household-level random effects
    
    Model structure:
    - Products have positions in a latent K-dimensional space (Lambda)
    - Households have preferences over each latent dimension (theta)
    - Utility = alpha + theta @ Lambda' + household_features @ beta + random_effects
    - P(purchase) = sigmoid(utility)
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available")
    
    n_obs, n_items = data.shape
    
    with pm.Model() as model:
        # Product intercepts (baseline purchase probability)
        alpha = pm.Normal('alpha', mu=0, sigma=2, shape=n_items)
        
        # Latent product factors (K-dimensional embedding for each product)
        # Using non-centered parameterization for better sampling
        Lambda_raw = pm.Normal('Lambda_raw', mu=0, sigma=1, shape=(n_items, n_factors))
        Lambda = pm.Deterministic('Lambda', Lambda_raw * 0.5)  # Scale down
        
        # Household preferences over latent factors
        # Each household has a K-dimensional preference vector
        theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=(n_obs, n_factors))
        theta = pm.Deterministic('theta', theta_raw * 0.5)
        
        # Utility from latent factors: theta @ Lambda'
        # Shape: (n_obs, n_factors) @ (n_factors, n_items) = (n_obs, n_items)
        latent_utility = pm.math.dot(theta, Lambda.T)
        
        utility = alpha + latent_utility
        
        # Add household features if provided
        if household_features is not None:
            n_hh_features = household_features.shape[1]
            hh_features_std = (household_features - household_features.mean(0)) / (household_features.std(0) + 1e-10)
            
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=(n_items, n_hh_features))
            beta = pm.Deterministic('beta', beta_raw * 0.5)
            
            utility = utility + pm.math.dot(hh_features_std, beta.T)
        
        # Household random effects (additional heterogeneity)
        if include_random_effects:
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=0.3)
            hh_effect_raw = pm.Normal('hh_effect_raw', mu=0, sigma=1, shape=n_obs)
            hh_effect = pm.Deterministic('hh_effect', hh_effect_raw * sigma_hh)
            utility = utility + hh_effect[:, None]
        
        # Likelihood
        prob = pm.math.sigmoid(utility)
        likelihood = pm.Bernoulli('obs', p=prob, observed=data)
        
        # Sample
        trace = pm.sample(n_samples, tune=n_tune, nuts_sampler='nutpie',
                         return_inferencedata=True, progressbar=True,
                         target_accept=0.95)
        trace = pm.compute_log_likelihood(trace)
    # Extract results
    alpha_samples = trace.posterior['alpha'].values.reshape(-1, n_items)
    Lambda_samples = trace.posterior['Lambda'].values.reshape(-1, n_items, n_factors)
    theta_samples = trace.posterior['theta'].values.reshape(-1, n_obs, n_factors)
    
    result = {
        'alpha': alpha_samples.mean(axis=0),
        'alpha_std': alpha_samples.std(axis=0),
        'Lambda': Lambda_samples.mean(axis=0),  # Product loadings
        'Lambda_std': Lambda_samples.std(axis=0),
        'theta': theta_samples.mean(axis=0),  # Household preferences
        'theta_std': theta_samples.std(axis=0),
        'trace': trace,
        'n_factors': n_factors,
        'n_divergences': trace.sample_stats.diverging.sum().values
    }
    
    if household_features is not None:
        beta_samples = trace.posterior['beta'].values.reshape(-1, n_items, household_features.shape[1])
        result['beta'] = beta_samples.mean(axis=0)
        result['beta_std'] = beta_samples.std(axis=0)
    
    if include_random_effects:
        result['sigma_hh'] = trace.posterior['sigma_hh'].values.mean()
    
    try:
        result['waic'] = az.waic(trace)
    except:
        result['waic'] = None
    
    # Compute variance explained by latent factors
    Lambda_mean = result['Lambda']
    var_explained = np.sum(Lambda_mean ** 2, axis=0)
    total_var = np.sum(Lambda_mean ** 2)
    result['var_explained_pct'] = var_explained / total_var * 100 if total_var > 0 else np.zeros(n_factors)
    
    return result


def fit_dcm_with_latent_features(data: np.ndarray, latent_product_features: np.ndarray,
                                  household_features: np.ndarray = None,
                                  include_random_effects: bool = True,
                                  include_interactions: bool = False,
                                  n_samples: int = 1000, n_tune: int = 500) -> dict:
    """
    Fit discrete choice model using pre-computed latent product features.
    
    This is a two-stage approach:
    1. First fit a factor model (FA, MCA, NMF) to get product embeddings
    2. Use those embeddings as product features in DCM
    
    The model estimates:
    - gamma: Coefficients for each latent dimension (shared across households)
    - Optionally: interactions between household features and latent dimensions
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available")
    
    n_obs, n_items = data.shape
    n_latent = latent_product_features.shape[1]
    
    # Standardize latent features
    latent_std = (latent_product_features - latent_product_features.mean(0)) / (latent_product_features.std(0) + 1e-10)
    
    with pm.Model() as model:
        # Product intercepts
        alpha = pm.Normal('alpha', mu=0, sigma=2, shape=n_items)
        
        # Coefficients for latent product features
        # gamma: effect of each latent dimension on utility
        gamma = pm.Normal('gamma', mu=0, sigma=1, shape=n_latent)
        latent_effect = pm.math.dot(latent_std, gamma)  # (n_items,)
        
        utility = alpha + latent_effect
        
        # Household features
        if household_features is not None:
            n_hh_features = household_features.shape[1]
            hh_features_std = (household_features - household_features.mean(0)) / (household_features.std(0) + 1e-10)
            
            # Main effects of household features
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=(n_items, n_hh_features))
            beta = pm.Deterministic('beta', beta_raw * 0.5)
            utility = utility + pm.math.dot(hh_features_std, beta.T)
            
            # Interactions: household preferences over latent dimensions
            if include_interactions:
                # delta[k, f] = interaction between latent dim k and household feature f
                delta = pm.Normal('delta', mu=0, sigma=0.5, shape=(n_latent, n_hh_features))
                
                # For each household, compute preference-weighted latent effects
                # hh_features_std: (n_obs, n_hh_features)
                # delta: (n_latent, n_hh_features)
                # latent_std: (n_items, n_latent)
                
                # Household-specific latent preferences: (n_obs, n_latent)
                hh_latent_pref = pm.math.dot(hh_features_std, delta.T)
                
                # Interaction utility: (n_obs, n_items)
                interaction_utility = pm.math.dot(hh_latent_pref, latent_std.T)
                utility = utility + interaction_utility
        
        # Random effects
        if include_random_effects:
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=0.5)
            hh_effect_raw = pm.Normal('hh_effect_raw', mu=0, sigma=1, shape=n_obs)
            hh_effect = pm.Deterministic('hh_effect', hh_effect_raw * sigma_hh)
            utility = utility + hh_effect[:, None]
        
        prob = pm.math.sigmoid(utility)
        likelihood = pm.Bernoulli('obs', p=prob, observed=data)
        
        trace = pm.sample(n_samples, tune=n_tune,
                         return_inferencedata=True, progressbar=True,
                         target_accept=0.95, nuts_sampler='nutpie')
        trace = pm.compute_log_likelihood(trace)
    # Extract results
    result = {
        'alpha': trace.posterior['alpha'].values.reshape(-1, n_items).mean(axis=0),
        'alpha_std': trace.posterior['alpha'].values.reshape(-1, n_items).std(axis=0),
        'gamma': trace.posterior['gamma'].values.reshape(-1, n_latent).mean(axis=0),
        'gamma_std': trace.posterior['gamma'].values.reshape(-1, n_latent).std(axis=0),
        'trace': trace,
        'n_latent': n_latent,
        'n_divergences': trace.sample_stats.diverging.sum().values
    }
    
    if household_features is not None:
        n_hh_features = household_features.shape[1]
        beta_samples = trace.posterior['beta'].values.reshape(-1, n_items, n_hh_features)
        result['beta'] = beta_samples.mean(axis=0)
        result['beta_std'] = beta_samples.std(axis=0)
        
        if include_interactions:
            delta_samples = trace.posterior['delta'].values.reshape(-1, n_latent, n_hh_features)
            result['delta'] = delta_samples.mean(axis=0)
            result['delta_std'] = delta_samples.std(axis=0)
    
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
# GENERIC BIPLOT FUNCTION FOR ALL MODELS
# =============================================================================

def plot_generic_biplot(row_coords: np.ndarray, col_coords: np.ndarray, 
                        product_labels: list, dim1: int = 0, dim2: int = 1,
                        var_explained: list = None, model_name: str = "Model",
                        cluster_labels: np.ndarray = None,
                        show_households: bool = True) -> go.Figure:
    """
    Create a generic biplot showing products and optionally households in latent space.
    
    Args:
        row_coords: Household/observation coordinates (n_obs x n_dims)
        col_coords: Product/variable coordinates (n_items x n_dims)
        product_labels: Names of products
        dim1, dim2: Which dimensions to plot
        var_explained: Variance explained by each dimension (for axis labels)
        model_name: Name of the model for the title
        cluster_labels: Optional cluster assignments for products
        show_households: Whether to show household points
    """
    fig = go.Figure()
    
    # Plot households (row coordinates) as small points
    if show_households and row_coords is not None and len(row_coords) > 0:
        fig.add_trace(go.Scatter(
            x=row_coords[:, dim1],
            y=row_coords[:, dim2],
            mode='markers',
            marker=dict(size=4, color='lightblue', opacity=0.5),
            name='Households',
            hoverinfo='skip'
        ))
    
    # Plot products (column coordinates) as labeled points
    if cluster_labels is not None:
        # Color by cluster
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
    
    # Add origin lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Axis labels with variance explained
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


# =============================================================================
# CLUSTERING FUNCTIONS FOR EMBEDDING SPACE
# =============================================================================

def cluster_products_kmeans(embeddings: np.ndarray, n_clusters: int) -> dict:
    """Cluster products using K-means in the embedding space."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Compute silhouette score if we have more than 1 cluster
    sil_score = None
    if n_clusters > 1 and n_clusters < len(embeddings):
        try:
            sil_score = silhouette_score(embeddings, labels)
        except:
            pass
    
    return {
        'labels': labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'silhouette': sil_score,
        'n_clusters': n_clusters
    }


def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10) -> dict:
    """Find optimal number of clusters using silhouette score."""
    max_clusters = min(max_clusters, len(embeddings) - 1)
    if max_clusters < 2:
        return {'optimal_k': 1, 'scores': [], 'range': []}
    
    scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append(score)
        except:
            scores.append(0)
    
    if len(scores) > 0:
        optimal_k = list(k_range)[np.argmax(scores)]
    else:
        optimal_k = 2
    
    return {
        'optimal_k': optimal_k,
        'scores': scores,
        'range': list(k_range)
    }


def cluster_products_hierarchical(embeddings: np.ndarray, n_clusters: int, 
                                   method: str = 'ward') -> dict:
    """Cluster products using hierarchical clustering."""
    Z = linkage(embeddings, method=method)
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-indexed
    
    # Compute silhouette score
    sil_score = None
    if n_clusters > 1 and n_clusters < len(embeddings):
        try:
            sil_score = silhouette_score(embeddings, labels)
        except:
            pass
    
    return {
        'labels': labels,
        'linkage_matrix': Z,
        'silhouette': sil_score,
        'n_clusters': n_clusters
    }


def plot_cluster_summary(embeddings: np.ndarray, labels: np.ndarray, 
                         product_labels: list, title: str = "Cluster Summary") -> go.Figure:
    """Create a summary visualization of clusters."""
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # Create subplots: cluster sizes + cluster profiles
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Cluster Sizes", "Cluster Centers (Mean Coordinates)"],
        column_widths=[0.3, 0.7]
    )
    
    # Cluster sizes
    cluster_sizes = [np.sum(labels == c) for c in unique_clusters]
    cluster_names = [f"Cluster {c+1}" for c in unique_clusters]
    
    fig.add_trace(
        go.Bar(x=cluster_names, y=cluster_sizes, marker_color='steelblue', showlegend=False),
        row=1, col=1
    )
    
    # Cluster centers (mean coordinates for first few dimensions)
    n_dims_to_show = min(5, embeddings.shape[1])
    centers = np.array([embeddings[labels == c].mean(axis=0)[:n_dims_to_show] for c in unique_clusters])
    
    for i, cluster in enumerate(unique_clusters):
        fig.add_trace(
            go.Bar(
                name=f"Cluster {cluster+1}",
                x=[f"Dim {d+1}" for d in range(n_dims_to_show)],
                y=centers[i],
                showlegend=True
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=title,
        height=400,
        barmode='group'
    )
    
    return fig


def get_cluster_members(labels: np.ndarray, product_labels: list) -> pd.DataFrame:
    """Create a DataFrame showing cluster membership."""
    df = pd.DataFrame({
        'Product': product_labels,
        'Cluster': labels + 1  # 1-indexed for display
    })
    return df.sort_values('Cluster')


# =============================================================================
# FACTOR SCORE COMPUTATION
# =============================================================================

def compute_factor_scores_regression(data: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """
    Compute factor scores using regression method.
    scores = data @ loadings @ inv(loadings.T @ loadings)
    """
    # Center the data
    data_centered = data - data.mean(axis=0)
    
    # Regression method: (L'L)^-1 L' X'
    LtL = loadings.T @ loadings
    try:
        LtL_inv = np.linalg.inv(LtL + np.eye(LtL.shape[0]) * 1e-6)  # Regularization
        scores = data_centered @ loadings @ LtL_inv
    except:
        # Fallback: simple projection
        scores = data_centered @ loadings
    
    return scores


def compute_lca_coordinates(class_probs: np.ndarray, item_probs: np.ndarray, 
                            responsibilities: np.ndarray) -> tuple:
    """
    Compute coordinates for LCA visualization.
    Products: use item_probs across classes
    Households: use responsibilities (posterior class probabilities)
    """
    # For products: transpose item_probs so each product has coordinates across classes
    # item_probs shape: (n_classes, n_items) -> product_coords: (n_items, n_classes)
    product_coords = item_probs.T
    
    # For households: use responsibilities directly
    # responsibilities shape: (n_obs, n_classes)
    household_coords = responsibilities
    
    return household_coords, product_coords


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
# HELPER FUNCTION: Generate cache key for model results
# =============================================================================

def get_model_cache_key(data_hash: str, model_type: str, model_params: dict, product_columns: tuple) -> str:
    """Generate a unique cache key for model results based on input parameters."""
    params_hash = hash(frozenset(model_params.items()))
    return f"{model_type}_{data_hash}_{params_hash}_{hash(product_columns)}"


# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Latent Structure Analysis", layout="wide")
    
    st.title("🛒 Latent Structure Analysis for Purchase Data")
    st.markdown("""
    Discover latent customer segments and product relationships using multiple statistical methods.
    """)
    
    # Initialize session state for model results (prevents rerun on UI changes)
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
    # Store latent product features for use in DCM
    if 'latent_product_features' not in st.session_state:
        st.session_state.latent_product_features = None
    if 'latent_feature_source' not in st.session_state:
        st.session_state.latent_feature_source = None
    
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
            
            # Latent product features options
            st.markdown("---")
            st.subheader("🔮 Latent Product Features")
            
            dcm_latent_mode = st.radio(
                "How to incorporate latent product structure?",
                options=[
                    "None (standard DCM)",
                    "Use pre-computed latent features (two-stage)",
                    "Joint estimation (estimate latent factors in DCM)"
                ],
                help="""
                **None**: Standard discrete choice model with product intercepts only.
                **Two-stage**: Use product embeddings from a previously fitted factor model (FA, MCA, NMF).
                **Joint**: Simultaneously estimate latent product factors and choice parameters.
                """
            )
            
            use_latent_features = False
            use_joint_estimation = False
            include_interactions = False
            n_latent_factors = 2
            
            if dcm_latent_mode == "Use pre-computed latent features (two-stage)":
                use_latent_features = True
                if st.session_state.latent_product_features is not None:
                    st.success(f"✅ Latent features available from: {st.session_state.latent_feature_source}")
                    st.write(f"   Shape: {st.session_state.latent_product_features.shape[1]} dimensions")
                    
                    include_interactions = st.checkbox(
                        "Include household × latent dimension interactions",
                        value=False,
                        help="Allow household features to interact with latent product dimensions"
                    )
                else:
                    st.warning("⚠️ No latent features available. First run a factor model (FA, MCA, or NMF) to generate product embeddings.")
                    use_latent_features = False
            
            elif dcm_latent_mode == "Joint estimation (estimate latent factors in DCM)":
                use_joint_estimation = True
                n_latent_factors = st.slider(
                    "Number of latent factors",
                    min_value=1,
                    max_value=min(5, len(product_columns) - 1),
                    value=2,
                    help="Number of latent dimensions to estimate jointly with choice model"
                )
        
        # Hierarchical clustering option
        st.markdown("---")
        show_hierarchy = st.checkbox("Show hierarchical clustering of products", value=True)
        if show_hierarchy:
            linkage_method = st.selectbox(
                "Clustering linkage method",
                options=['average', 'complete', 'single']
            )
        
        # Build model parameters dict for cache key
        model_params = {}
        if model_type == "Latent Class Analysis (LCA)":
            model_params = {'n_classes': n_classes, 'max_iter': max_iter, 'n_init': n_init}
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
                'latent_mode': dcm_latent_mode,
                'n_latent_factors': n_latent_factors if use_joint_estimation else 0,
                'interactions': include_interactions if use_latent_features else False
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
                
                # Compute embeddings for biplot
                household_coords, product_coords = compute_lca_coordinates(
                    result['class_probs'], result['item_probs'], result['responsibilities']
                )
                
                # Compute similarity matrix
                residual_corr = compute_residual_correlations(X, result['responsibilities'], result['item_probs'])
                
                # Store in session state
                st.session_state.model_result = result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = residual_corr
                st.session_state.product_embeddings = product_coords
                st.session_state.household_embeddings = household_coords
                st.session_state.var_explained_cached = None  # LCA doesn't have variance explained
                st.session_state.cluster_result = None
                
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
                
                # Compute factor scores for biplot
                factor_scores = compute_factor_scores_regression(X, fa_result['loadings'])
                
                # Store in session state
                st.session_state.model_result = fa_result
                st.session_state.model_result['tetra_corr'] = tetra_corr
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = tetra_corr
                st.session_state.product_embeddings = fa_result['loadings']
                st.session_state.household_embeddings = factor_scores
                st.session_state.var_explained_cached = fa_result['var_explained_pct']
                st.session_state.cluster_result = None
                
                # Save latent features for potential use in DCM
                st.session_state.latent_product_features = fa_result['loadings']
                st.session_state.latent_feature_source = "Factor Analysis (Tetrachoric)"
                
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
            
            # =========== BAYESIAN FA (VI) ===========
            elif model_type == "Bayesian Factor Model (VI)":
                st.header("📊 Bayesian Factor Model (Variational Inference)")
                
                with st.spinner("Fitting Bayesian factor model..."):
                    bfa_result = fit_bayesian_factor_model_vi(X, n_factors, max_iter=max_iter)
                
                # Compute implied correlations
                loadings_norm = bfa_result['loadings'] / (np.linalg.norm(bfa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                # Store in session state
                st.session_state.model_result = bfa_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = implied_corr
                st.session_state.product_embeddings = bfa_result['loadings']
                st.session_state.household_embeddings = bfa_result['scores']
                st.session_state.var_explained_cached = bfa_result['var_explained_pct']
                st.session_state.cluster_result = None
                
                # Save latent features for potential use in DCM
                st.session_state.latent_product_features = bfa_result['loadings']
                st.session_state.latent_feature_source = "Bayesian Factor Model (VI)"
                
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
            
            # =========== BAYESIAN FA (PyMC) ===========
            elif model_type == "Bayesian Factor Model (PyMC)":
                st.header("📊 Bayesian Factor Model (PyMC MCMC)")
                
                with st.spinner("Running MCMC sampling... This may take a few minutes."):
                    bfa_result = fit_bayesian_factor_model_pymc(X, n_factors, n_samples=n_samples, n_tune=n_tune)
                
                # Compute implied correlations
                loadings_norm = bfa_result['loadings'] / (np.linalg.norm(bfa_result['loadings'], axis=0, keepdims=True) + 1e-10)
                implied_corr = loadings_norm @ loadings_norm.T
                
                # Compute factor scores for biplot
                factor_scores = compute_factor_scores_regression(X, bfa_result['loadings'])
                
                # Store in session state
                st.session_state.model_result = bfa_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = implied_corr
                st.session_state.product_embeddings = bfa_result['loadings']
                st.session_state.household_embeddings = factor_scores
                st.session_state.var_explained_cached = bfa_result['var_explained_pct']
                st.session_state.cluster_result = None
                
                # Save latent features for potential use in DCM
                st.session_state.latent_product_features = bfa_result['loadings']
                st.session_state.latent_feature_source = "Bayesian Factor Model (PyMC)"
                
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
            
            # =========== NMF ===========
            elif model_type == "Non-negative Matrix Factorization (NMF)":
                st.header("📊 Non-negative Matrix Factorization")
                
                with st.spinner("Fitting NMF model..."):
                    nmf_result = fit_nmf(X, n_components, max_iter=max_iter)
                
                # Compute similarity
                loadings_norm = nmf_result['loadings'] / (np.linalg.norm(nmf_result['loadings'], axis=1, keepdims=True) + 1e-10)
                cosine_sim = loadings_norm @ loadings_norm.T
                
                # Store in session state
                st.session_state.model_result = nmf_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = cosine_sim
                st.session_state.product_embeddings = nmf_result['loadings']
                st.session_state.household_embeddings = nmf_result['scores']
                st.session_state.var_explained_cached = nmf_result['var_explained_pct']
                st.session_state.cluster_result = None
                
                # Save latent features for potential use in DCM
                st.session_state.latent_product_features = nmf_result['loadings']
                st.session_state.latent_feature_source = "Non-negative Matrix Factorization (NMF)"
                
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
                st.session_state.model_result = mca_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = mca_result['similarity_matrix']
                st.session_state.product_embeddings = mca_result['column_coordinates']
                st.session_state.household_embeddings = mca_result['row_coordinates']
                st.session_state.var_explained_cached = mca_result['var_explained_pct']
                st.session_state.cluster_result = None
                
                # Save latent features for potential use in DCM
                st.session_state.latent_product_features = mca_result['column_coordinates']
                st.session_state.latent_feature_source = "Multiple Correspondence Analysis (MCA)"
                
                st.success("MCA complete!")
            
            # =========== DISCRETE CHOICE MODEL (PyMC) ===========
            elif model_type == "Discrete Choice Model (PyMC)":
                st.header("📊 Discrete Choice Model (PyMC)")
                
                # Prepare household features
                hh_features = None
                if household_feature_columns:
                    hh_features = df.loc[data_subset.index, household_feature_columns].values
                    hh_features = np.nan_to_num(hh_features, nan=0)
                
                # Prepare product features (from uploaded file)
                prod_features = None
                if product_feature_df is not None and use_product_features:
                    prod_feature_cols = [c for c in product_feature_df.columns if c != 'product']
                    prod_features = product_feature_df[prod_feature_cols].values.astype(float)
                
                # ============ JOINT LATENT FACTOR DCM ============
                if use_joint_estimation:
                    st.info(f"""
                    **Joint Latent Factor DCM** estimates:
                    - **Λ (Lambda)**: {n_latent_factors}-dimensional latent product embeddings
                    - **θ (theta)**: Household preferences over latent dimensions
                    - **α (alpha)**: Product intercepts
                    - Utility = α + θ @ Λ' + household effects
                    """)
                    
                    with st.spinner(f"Running joint latent factor DCM with {n_latent_factors} factors..."):
                        dcm_result = fit_latent_factor_dcm_pymc(
                            X,
                            n_factors=n_latent_factors,
                            household_features=hh_features,
                            include_random_effects=include_random_effects,
                            n_samples=n_samples,
                            n_tune=n_tune
                        )
                    
                    # Product embeddings are the estimated Lambda
                    product_embeds = dcm_result['Lambda']
                    household_embeds = dcm_result['theta']
                    var_explained = dcm_result['var_explained_pct']
                    
                    st.success("Joint estimation complete!")
                    
                    # Store latent features for future use
                    st.session_state.latent_product_features = dcm_result['Lambda']
                    st.session_state.latent_feature_source = f"Joint DCM ({n_latent_factors} factors)"
                
                # ============ TWO-STAGE WITH LATENT FEATURES ============
                elif use_latent_features and st.session_state.latent_product_features is not None:
                    latent_features = st.session_state.latent_product_features
                    
                    st.info(f"""
                    **Two-Stage DCM** using latent features from: {st.session_state.latent_feature_source}
                    - **γ (gamma)**: Effect of each latent dimension on utility
                    - **α (alpha)**: Product intercepts
                    {"- **δ (delta)**: Household × latent dimension interactions" if include_interactions else ""}
                    """)
                    
                    with st.spinner("Running DCM with latent product features..."):
                        dcm_result = fit_dcm_with_latent_features(
                            X,
                            latent_product_features=latent_features,
                            household_features=hh_features,
                            include_random_effects=include_random_effects,
                            include_interactions=include_interactions,
                            n_samples=n_samples,
                            n_tune=n_tune
                        )
                    
                    product_embeds = latent_features
                    household_embeds = None
                    var_explained = None
                    
                    st.success("Two-stage DCM complete!")
                
                # ============ STANDARD DCM ============
                else:
                    st.info("""
                    **Standard DCM** with product intercepts and optional features.
                    Features are standardized automatically.
                    """)
                    
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
                    
                    # For standard DCM, create simple embeddings from coefficients
                    product_embeds = dcm_result['alpha'].reshape(-1, 1)
                    if 'beta' in dcm_result:
                        product_embeds = np.hstack([product_embeds, dcm_result['beta']])
                    household_embeds = None
                    var_explained = None
                    
                    st.success("MCMC sampling complete!")
                
                # Compute similarity from raw data correlation
                similarity_matrix = np.corrcoef(X.T)
                
                # Store in session state
                st.session_state.model_result = dcm_result
                st.session_state.model_cache_key = current_cache_key
                st.session_state.model_type_cached = model_type
                st.session_state.product_columns_cached = product_columns
                st.session_state.similarity_matrix_cached = similarity_matrix
                st.session_state.product_embeddings = product_embeds
                st.session_state.household_embeddings = household_embeds
                st.session_state.var_explained_cached = var_explained
                st.session_state.cluster_result = None
                
                # ============ DISPLAY RESULTS ============
                
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
                
                # ============ JOINT MODEL SPECIFIC OUTPUTS ============
                if use_joint_estimation:
                    st.subheader("Estimated Latent Product Factors (Λ)")
                    st.caption("Product positions in the estimated latent space")
                    
                    factor_labels = [f"Factor {k+1}" for k in range(n_latent_factors)]
                    fig = plot_loadings_heatmap(
                        dcm_result['Lambda'], 
                        product_columns,
                        factor_labels,
                        "Product Loadings on Latent Factors"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Variance explained by latent factors
                    if 'var_explained_pct' in dcm_result:
                        fig = plot_variance_explained(dcm_result['var_explained_pct'], "Joint DCM Latent Factors")
                        st.plotly_chart(fig, use_container_width=True)
                
                # ============ TWO-STAGE SPECIFIC OUTPUTS ============
                elif use_latent_features:
                    st.subheader("Latent Dimension Effects (γ)")
                    st.caption(f"How each latent dimension from {st.session_state.latent_feature_source} affects utility")
                    
                    n_latent = dcm_result['n_latent']
                    gamma_df = pd.DataFrame({
                        'Latent Dimension': [f"Dim {i+1}" for i in range(n_latent)],
                        'Coefficient': dcm_result['gamma'],
                        'Std': dcm_result['gamma_std']
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=gamma_df['Latent Dimension'],
                        y=gamma_df['Coefficient'],
                        error_y=dict(type='data', array=gamma_df['Std'], visible=True),
                        marker_color='purple'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title='Effect of Latent Dimensions on Purchase Utility',
                        xaxis_title='Latent Dimension',
                        yaxis_title='Coefficient (γ)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interactions if estimated
                    if include_interactions and 'delta' in dcm_result:
                        st.subheader("Household × Latent Dimension Interactions (δ)")
                        st.caption("How household features moderate preferences for latent dimensions")
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=dcm_result['delta'],
                            x=household_feature_columns,
                            y=[f"Dim {i+1}" for i in range(n_latent)],
                            colorscale='RdBu_r',
                            zmid=0,
                            text=np.round(dcm_result['delta'], 2),
                            texttemplate='%{text}',
                            hovertemplate='%{y} × %{x}<br>δ: %{z:.3f}<extra></extra>'
                        ))
                        fig.update_layout(
                            title='Interaction Effects: Latent Dimensions × Household Features',
                            xaxis_title='Household Feature',
                            yaxis_title='Latent Dimension',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # ============ COMMON OUTPUTS ============
                
                # Household feature effects
                if 'beta' in dcm_result and household_feature_columns:
                    st.subheader("Household Feature Effects (β)")
                    fig = plot_beta_coefficients(dcm_result['beta'], dcm_result['beta_std'],
                                                 product_columns, household_feature_columns)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Product feature effects (only for standard DCM with uploaded features)
                if 'gamma' in dcm_result and not use_latent_features and not use_joint_estimation:
                    if product_feature_df is not None:
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
            
            # =========== MODEL-SPECIFIC ADDITIONAL OUTPUTS ===========
            if model_type == "Latent Class Analysis (LCA)":
                st.subheader("Residual Correlations (Substitution Patterns)")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Residual Correlations (negative = substitution)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Bayesian Factor Model (VI)":
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Implied Correlations from Factor Model")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Bayesian Factor Model (PyMC)":
                st.subheader("Implied Product Correlations")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Implied Correlations from Bayesian FA")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Non-negative Matrix Factorization (NMF)":
                st.subheader("Product Similarity (from NMF)")
                fig = plot_correlation_matrix(similarity_matrix, product_columns_cached, 
                    "Cosine Similarity (NMF)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Multiple Correspondence Analysis (MCA)":
                # Variance explained
                st.subheader("Explained Inertia (Variance)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Inertia", f"{model_result['total_inertia']:.4f}")
                with col2:
                    total_explained = sum(model_result['var_explained_pct'][:model_result['n_components']])
                    st.metric("Total Explained", f"{total_explained:.1f}%")
                
                fig = plot_variance_explained(model_result['var_explained_pct'], "MCA")
                st.plotly_chart(fig, use_container_width=True)
                
                # Product contributions
                st.subheader("Product Contributions to Dimensions")
                st.caption("Higher contribution = product is more important in defining that dimension")
                
                fig = plot_mca_contributions(
                    model_result['contributions'],
                    model_result['product_labels'],
                    n_dims=min(3, model_result['n_components'])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Product coordinates (loadings equivalent)
                st.subheader("Product Coordinates (Dimension Loadings)")
                dim_labels = [f"Dim {k+1}" for k in range(model_result['n_components'])]
                fig = plot_loadings_heatmap(
                    model_result['column_coordinates'],
                    model_result['product_labels'],
                    dim_labels,
                    "Product Coordinates in MCA Space"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Product similarity
                st.subheader("Product Similarity (from MCA)")
                fig = plot_correlation_matrix(
                    model_result['similarity_matrix'],
                    model_result['product_labels'],
                    "Product Similarity (cosine in MCA space)"
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
                    model_name=model_type.split(" (")[0],  # Clean model name
                    cluster_labels=cluster_labels,
                    show_households=show_households
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # =========== PRODUCT CLUSTERING SECTION ===========
            if product_embeddings is not None:
                st.markdown("---")
                st.subheader("🔮 Product Clustering")
                st.caption("Cluster products based on their positions in the latent space.")
                
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
                                # Find optimal K
                                opt_result = find_optimal_clusters(product_embeddings, max_k)
                                n_clusters = opt_result['optimal_k']
                                st.info(f"Optimal number of clusters: {n_clusters} (based on silhouette score)")
                                
                                # Show silhouette scores
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
                            
                            # Run clustering
                            if cluster_method == "K-Means":
                                cluster_result = cluster_products_kmeans(product_embeddings, n_clusters)
                            else:
                                cluster_result = cluster_products_hierarchical(
                                    product_embeddings, n_clusters, method='ward'
                                )
                            
                            st.session_state.cluster_result = cluster_result
                    
                    # Display clustering results if available
                    if st.session_state.cluster_result is not None:
                        cluster_result = st.session_state.cluster_result
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Number of Clusters", cluster_result['n_clusters'])
                        with col2:
                            if cluster_result.get('silhouette') is not None:
                                st.metric("Silhouette Score", f"{cluster_result['silhouette']:.3f}")
                            else:
                                st.metric("Silhouette Score", "N/A")
                        
                        # Show cluster membership
                        st.subheader("Cluster Membership")
                        if model_type == "Multiple Correspondence Analysis (MCA)":
                            prod_labels = model_result['product_labels']
                        else:
                            prod_labels = product_columns_cached
                        
                        cluster_df = get_cluster_members(cluster_result['labels'], prod_labels)
                        
                        # Display as columns for each cluster
                        n_clusters_actual = cluster_result['n_clusters']
                        cols = st.columns(min(n_clusters_actual, 4))
                        for i in range(n_clusters_actual):
                            with cols[i % len(cols)]:
                                cluster_products = cluster_df[cluster_df['Cluster'] == i + 1]['Product'].tolist()
                                st.markdown(f"**Cluster {i + 1}** ({len(cluster_products)} products)")
                                for prod in cluster_products:
                                    st.write(f"• {prod}")
                        
                        # Cluster summary visualization
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
            
            The DCM now supports three modes for incorporating latent product structure:
            
            **1. Standard DCM (None)**
            - Product intercepts (α): baseline purchase probability for each product
            - Household features (β): how demographics affect purchase of each product
            - No latent structure assumed
            
            **2. Two-Stage Approach**
            - First fit a factor model (FA, MCA, NMF) to discover product embeddings
            - Then use those embeddings as features in DCM
            - **γ (gamma)**: Effect of each latent dimension on utility
            - **δ (delta)**: Optional interactions between household features and latent dimensions
            - Advantage: Interpretable latent features from dedicated model
            - Limitation: Uncertainty from first stage not propagated
            
            **3. Joint Estimation**
            - Simultaneously estimate latent factors and choice parameters
            - **Λ (Lambda)**: Product positions in latent space (like factor loadings)
            - **θ (theta)**: Household preferences over latent dimensions
            - Advantage: Fully Bayesian, accounts for all uncertainty
            - Limitation: More complex, slower to fit
            
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
            
            ### Workflow Recommendation
            
            1. **Start with exploratory models**: Run MCA or FA to understand latent structure
            2. **Cluster products**: Use the embedding space to find natural groupings
            3. **Build DCM**: Use two-stage or joint estimation to model purchase drivers
            4. **Iterate**: Compare models using WAIC to find the best specification
            
            ### General Tips
            
            - **Hierarchical clustering**: Use to identify natural product groupings
            - **Negative residual correlations** in LCA suggest substitution patterns
            - **High loadings** on multiple factors suggest products that bridge categories
            - **Cluster products** to create actionable segments for marketing/assortment
            """)


if __name__ == "__main__":
    main()