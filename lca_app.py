"""
Streamlit App for Latent Structure Analysis on Binary Purchase Data
Supports: LCA, Factor Analysis (Tetrachoric), Bayesian Factor Models (VI & PyMC), 
          NMF, Discrete Choice Model (PyMC)
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
    n00 = np.sum((x == 0) & (y == 0))
    n01 = np.sum((x == 0) & (y == 1))
    n10 = np.sum((x == 1) & (y == 0))
    n11 = np.sum((x == 1) & (y == 1))
    n = n00 + n01 + n10 + n11
    
    if n == 0:
        return 0.0
    
    p_x1 = (n10 + n11) / n
    p_y1 = (n01 + n11) / n
    
    if p_x1 <= 0.001 or p_x1 >= 0.999 or p_y1 <= 0.001 or p_y1 >= 0.999:
        return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0.0
    
    tau_x = norm.ppf(1 - p_x1)
    tau_y = norm.ppf(1 - p_y1)
    
    if n00 == 0 or n01 == 0 or n10 == 0 or n11 == 0:
        r_pearson = np.corrcoef(x, y)[0, 1]
        return r_pearson if not np.isnan(r_pearson) else 0.0
    
    ad = n00 * n11
    bc = n01 * n10
    
    if ad + bc > 0:
        r_approx = np.cos(np.pi * bc / (ad + bc))
    else:
        r_approx = 0.0
    
    def neg_log_lik(rho):
        rho = np.clip(rho, -0.99, 0.99)
        sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)
        
        z_x = -tau_x
        z_y = -tau_y
        
        p11 = norm.cdf(z_x) * norm.cdf((z_y - rho * z_x) / sqrt_1_minus_rho2)
        p10 = norm.cdf(z_x) * (1 - norm.cdf((z_y - rho * z_x) / sqrt_1_minus_rho2))
        p01 = (1 - norm.cdf(z_x)) * norm.cdf((z_y - rho * (-tau_x)) / sqrt_1_minus_rho2)
        p00 = 1 - p11 - p10 - p01
        
        eps = 1e-10
        p00 = np.clip(p00, eps, 1)
        p01 = np.clip(p01, eps, 1)
        p10 = np.clip(p10, eps, 1)
        p11 = np.clip(p11, eps, 1)
        
        ll = n00 * np.log(p00) + n01 * np.log(p01) + n10 * np.log(p10) + n11 * np.log(p11)
        return -ll
    
    result = minimize_scalar(neg_log_lik, bounds=(-0.99, 0.99), method='bounded')
    return result.x


def compute_tetrachoric_matrix(data: np.ndarray, progress_callback=None) -> np.ndarray:
    """Compute full tetrachoric correlation matrix."""
    n_items = data.shape[1]
    corr_matrix = np.eye(n_items)
    
    total_pairs = n_items * (n_items - 1) // 2
    current_pair = 0
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rho = compute_tetrachoric_single(data[:, i], data[:, j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
            
            current_pair += 1
            if progress_callback:
                progress_callback(current_pair / total_pairs)
    
    return corr_matrix


def factor_analysis_principal_axis(corr_matrix: np.ndarray, n_factors: int, 
                                    max_iter: int = 100, tol: float = 1e-4) -> dict:
    """Principal Axis Factor Analysis on a correlation matrix."""
    n_items = corr_matrix.shape[0]
    
    try:
        R_inv = np.linalg.pinv(corr_matrix)
        communalities = 1 - 1 / np.diag(R_inv)
    except:
        communalities = np.ones(n_items) * 0.5
    
    communalities = np.clip(communalities, 0.1, 0.99)
    
    for iteration in range(max_iter):
        old_communalities = communalities.copy()
        
        R_reduced = corr_matrix.copy()
        np.fill_diagonal(R_reduced, communalities)
        
        eigenvalues, eigenvectors = np.linalg.eigh(R_reduced)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        pos_eigenvalues = np.maximum(eigenvalues[:n_factors], 0)
        loadings = eigenvectors[:, :n_factors] * np.sqrt(pos_eigenvalues)
        
        communalities = np.sum(loadings ** 2, axis=1)
        communalities = np.clip(communalities, 0.01, 0.99)
        
        if np.max(np.abs(communalities - old_communalities)) < tol:
            break
    
    loadings_rotated = varimax_rotation(loadings)
    
    # Variance explained - relative importance of each factor
    total_var = np.sum(loadings_rotated ** 2)
    var_explained = np.sum(loadings_rotated ** 2, axis=0)
    var_explained_pct = (var_explained / total_var * 100) if total_var > 0 else np.zeros(n_factors)
    
    return {
        'loadings': loadings_rotated,
        'communalities': communalities,
        'eigenvalues': eigenvalues[:n_factors],
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'n_factors': n_factors,
        'n_iter': iteration + 1
    }


def varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Apply varimax rotation to factor loadings."""
    n_items, n_factors = loadings.shape
    
    if n_factors < 2:
        return loadings
    
    rotated = loadings.copy()
    
    for _ in range(max_iter):
        old_rotated = rotated.copy()
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                x = rotated[:, i]
                y = rotated[:, j]
                
                u = x**2 - y**2
                v = 2 * x * y
                
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)
                
                num = D - 2 * A * B / n_items
                denom = C - (A**2 - B**2) / n_items
                
                if abs(denom) < 1e-10:
                    continue
                    
                phi = 0.25 * np.arctan2(num, denom)
                
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                rotated[:, i] = cos_phi * x + sin_phi * y
                rotated[:, j] = -sin_phi * x + cos_phi * y
        
        if np.max(np.abs(rotated - old_rotated)) < tol:
            break
    
    return rotated


# =============================================================================
# BAYESIAN FACTOR MODEL (Variational Inference - Original)
# =============================================================================

def fit_bayesian_factor_model_vi(data: np.ndarray, n_factors: int, max_iter: int = 200, 
                                  tol: float = 1e-4, seed: int = 42) -> dict:
    """
    Bayesian Factor Model for binary data using variational inference.
    Model: P(x_ij = 1) = Phi(sum_k lambda_jk * z_ik)
    """
    np.random.seed(seed)
    n_obs, n_items = data.shape
    
    mu_z = np.random.randn(n_obs, n_factors) * 0.1
    sigma_z = np.ones((n_obs, n_factors))
    
    mu_lambda = np.random.randn(n_items, n_factors) * 0.5
    sigma_lambda = np.ones((n_items, n_factors))
    
    elbo_history = []
    learning_rate = 0.01
    
    for iteration in range(max_iter):
        for i in range(n_obs):
            linear_pred = mu_z[i] @ mu_lambda.T
            prob = norm.cdf(linear_pred)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            
            for k in range(n_factors):
                grad = np.sum((data[i] - prob) * mu_lambda[:, k] * norm.pdf(linear_pred) / (prob * (1 - prob) + 1e-10))
                grad -= mu_z[i, k]
                mu_z[i, k] += learning_rate * grad
        
        for j in range(n_items):
            linear_pred = mu_z @ mu_lambda[j]
            prob = norm.cdf(linear_pred)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            
            for k in range(n_factors):
                grad = np.sum((data[:, j] - prob) * mu_z[:, k] * norm.pdf(linear_pred) / (prob * (1 - prob) + 1e-10))
                grad -= mu_lambda[j, k]
                mu_lambda[j, k] += learning_rate * grad
        
        linear_pred_all = mu_z @ mu_lambda.T
        prob_all = norm.cdf(linear_pred_all)
        prob_all = np.clip(prob_all, 1e-6, 1 - 1e-6)
        
        log_lik = np.sum(data * np.log(prob_all) + (1 - data) * np.log(1 - prob_all))
        kl_z = 0.5 * np.sum(mu_z**2 + sigma_z - np.log(sigma_z) - 1)
        kl_lambda = 0.5 * np.sum(mu_lambda**2 + sigma_lambda - np.log(sigma_lambda) - 1)
        
        elbo = log_lik - kl_z - kl_lambda
        elbo_history.append(elbo)
        
        if iteration > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            break
    
    loadings_rotated = varimax_rotation(mu_lambda)
    
    # Variance explained - relative importance of each factor
    total_var = np.sum(loadings_rotated ** 2)
    var_explained = np.sum(loadings_rotated ** 2, axis=0)
    var_explained_pct = (var_explained / total_var * 100) if total_var > 0 else np.zeros(n_factors)
    
    return {
        'loadings': loadings_rotated,
        'factor_scores': mu_z,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'elbo_history': elbo_history,
        'n_factors': n_factors,
        'n_iter': iteration + 1
    }


# =============================================================================
# BAYESIAN FACTOR MODEL (PyMC) - PROPERLY IDENTIFIED
# =============================================================================

def fit_bayesian_factor_model_pymc(data: np.ndarray, n_factors: int, 
                                    n_samples: int = 1000, n_tune: int = 1000,
                                    seed: int = 42) -> dict:
    """
    Bayesian Factor Model for binary data using PyMC.
    
    Uses a properly identified parameterization to avoid divergences:
    1. Lower triangular loadings matrix with positive diagonal (fixes rotation)
    2. Non-centered parameterization for latent factors
    3. Regularizing priors on loadings
    
    Model:
        z_i ~ N(0, I)                           # Latent factors for household i
        Lambda = LowerTriangular(positive_diag) # Identified loadings matrix  
        P(x_ij = 1) = Phi(alpha_j + [z_i @ Lambda.T]_j)
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not installed. Install with: pip install pymc arviz")
    
    n_obs, n_items = data.shape
    
    # Ensure we don't have more factors than can be identified
    max_factors = n_items
    if n_factors > max_factors:
        n_factors = max_factors
        st.warning(f"Reduced factors to {max_factors} for identification")
    
    with pm.Model() as factor_model:
        # === IDENTIFIED LOADINGS PARAMETERIZATION ===
        # Use lower triangular structure with positive diagonal
        # This removes rotation invariance and sign ambiguity
        
        # Number of free parameters in lower triangular loadings
        # First factor: item 0 only (positive)
        # Second factor: items 0-1 (item 1 positive on diagonal)
        # etc.
        
        # Diagonal elements (positive, one per factor)
        # These are the "anchor" loadings that identify each factor
        diag_loadings = pm.HalfNormal('diag_loadings', sigma=1.0, shape=n_factors)
        
        # Off-diagonal elements in the lower triangular part
        # For factor k, items k+1 to n_items-1 can load on it
        n_offdiag = sum(n_items - k - 1 for k in range(n_factors))
        
        if n_offdiag > 0:
            offdiag_loadings = pm.Normal('offdiag_loadings', mu=0, sigma=0.5, 
                                         shape=n_offdiag)
        
        # Build the loadings matrix
        # Lambda[i,k] is the loading of item i on factor k
        # Constraint: Lambda[i,k] = 0 for i < k (upper triangular zeros)
        # Lambda[k,k] > 0 (positive diagonal)
        
        def build_loadings_matrix(diag, offdiag, n_items, n_factors):
            """Build lower triangular loadings matrix with positive diagonal."""
            Lambda = np.zeros((n_items, n_factors))
            offdiag_idx = 0
            
            for k in range(n_factors):
                # Diagonal element (positive)
                Lambda[k, k] = diag[k]
                
                # Off-diagonal elements (items below the diagonal)
                for i in range(k + 1, n_items):
                    if n_offdiag > 0:
                        Lambda[i, k] = offdiag[offdiag_idx]
                        offdiag_idx += 1
            
            return Lambda
        
        # Build loadings using pytensor ops
        import pytensor.tensor as pt
        
        Lambda = pt.zeros((n_items, n_factors))
        offdiag_idx = 0
        
        for k in range(n_factors):
            # Set diagonal
            Lambda = pt.set_subtensor(Lambda[k, k], diag_loadings[k])
            
            # Set off-diagonal
            for i in range(k + 1, n_items):
                if n_offdiag > 0:
                    Lambda = pt.set_subtensor(Lambda[i, k], offdiag_loadings[offdiag_idx])
                    offdiag_idx += 1
        
        # === LATENT FACTORS (Non-centered parameterization) ===
        # z_raw ~ N(0, 1), then z = z_raw (already standard normal)
        z = pm.Normal('z', mu=0, sigma=1, shape=(n_obs, n_factors))
        
        # === INTERCEPTS ===
        # Item intercepts (baseline log-odds of purchase)
        # Slightly informative prior centered on typical purchase rates
        alpha = pm.Normal('alpha', mu=0, sigma=1.5, shape=n_items)
        
        # === LINEAR PREDICTOR ===
        # eta_ij = alpha_j + sum_k(z_ik * Lambda_jk)
        eta = alpha + pm.math.dot(z, Lambda.T)
        
        # === LIKELIHOOD ===
        # Probit link: P(x=1) = Phi(eta)
        p = pm.math.invprobit(eta)
        
        # Observed data
        y_obs = pm.Bernoulli('y_obs', p=p, observed=data)
        
        # === SAMPLING ===
        # Use nutpie if available, otherwise standard NUTS
        try:
            import nutpie
            trace = pm.sample(
                n_samples, 
                tune=n_tune, 
                random_seed=seed,
                return_inferencedata=True, 
                progressbar=True,
                nuts_sampler="nutpie",
                target_accept=0.9
            )
        except (ImportError, Exception):
            # Fallback to standard PyMC sampler
            trace = pm.sample(
                n_samples, 
                tune=n_tune, 
                random_seed=seed,
                return_inferencedata=True, 
                progressbar=True,
                cores=1,
                target_accept=0.9,
                compile_kwargs={"mode": "FAST_COMPILE"}
            )
    
    # === EXTRACT RESULTS ===
    # Reconstruct full loadings matrix from posterior
    diag_mean = trace.posterior['diag_loadings'].mean(dim=['chain', 'draw']).values
    
    if n_offdiag > 0:
        offdiag_mean = trace.posterior['offdiag_loadings'].mean(dim=['chain', 'draw']).values
        offdiag_std = trace.posterior['offdiag_loadings'].std(dim=['chain', 'draw']).values
    else:
        offdiag_mean = np.array([])
        offdiag_std = np.array([])
    
    diag_std = trace.posterior['diag_loadings'].std(dim=['chain', 'draw']).values
    
    # Build posterior mean loadings matrix
    loadings_mean = np.zeros((n_items, n_factors))
    loadings_std = np.zeros((n_items, n_factors))
    offdiag_idx = 0
    
    for k in range(n_factors):
        loadings_mean[k, k] = diag_mean[k]
        loadings_std[k, k] = diag_std[k]
        
        for i in range(k + 1, n_items):
            if n_offdiag > 0:
                loadings_mean[i, k] = offdiag_mean[offdiag_idx]
                loadings_std[i, k] = offdiag_std[offdiag_idx]
                offdiag_idx += 1
    
    alpha_mean = trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
    alpha_std = trace.posterior['alpha'].std(dim=['chain', 'draw']).values
    z_mean = trace.posterior['z'].mean(dim=['chain', 'draw']).values
    
    # Variance explained - compute as proportion of total loading variance
    # This gives relative importance of each factor (sums to 100%)
    total_loading_var = np.sum(loadings_mean ** 2)
    var_explained = np.sum(loadings_mean ** 2, axis=0)
    if total_loading_var > 0:
        var_explained_pct = var_explained / total_loading_var * 100
    else:
        var_explained_pct = np.zeros(n_factors)
    
    # Model diagnostics
    try:
        waic = az.waic(trace)
    except Exception:
        waic = None
    
    # Sampling diagnostics
    try:
        summary = az.summary(trace, var_names=['diag_loadings', 'alpha'])
        n_divergences = trace.sample_stats.diverging.sum().values
    except Exception:
        summary = None
        n_divergences = 0
    
    return {
        'loadings': loadings_mean,
        'loadings_std': loadings_std,
        'alpha': alpha_mean,
        'alpha_std': alpha_std,
        'factor_scores': z_mean,
        'var_explained': var_explained,
        'var_explained_pct': var_explained_pct,
        'trace': trace,
        'waic': waic,
        'n_factors': n_factors,
        'n_divergences': int(n_divergences),
        'summary': summary
    }


# =============================================================================
# DISCRETE CHOICE MODEL (PyMC)
# =============================================================================

def fit_discrete_choice_model_pymc(purchase_data: np.ndarray, 
                                    household_features: np.ndarray = None,
                                    product_features: np.ndarray = None,
                                    product_names: list = None,
                                    include_random_effects: bool = True,
                                    n_samples: int = 1000, 
                                    n_tune: int = 500,
                                    seed: int = 42) -> dict:
    """
    Discrete Choice Model for binary purchase data using PyMC.
    
    Model:
        U_ij = alpha_j + X_i @ beta_j + Z_j @ gamma + (random effects) + epsilon_ij
        P(purchase_ij = 1) = logit^{-1}(U_ij)
    
    Where:
        - alpha_j: Product-specific intercept
        - X_i: Household features (n_households x n_hh_features)
        - beta_j: Product-specific coefficients for household features
        - Z_j: Product features (n_products x n_prod_features)
        - gamma: Coefficients for product features (shared across households)
        - Random effects: Optional household random effects
    
    Uses non-centered parameterization for hierarchical effects to reduce divergences.
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not installed. Install with: pip install pymc arviz")
    
    n_obs, n_items = purchase_data.shape
    
    # Handle features
    has_hh_features = household_features is not None and household_features.shape[1] > 0
    has_prod_features = product_features is not None and product_features.shape[1] > 0
    
    if has_hh_features:
        n_hh_features = household_features.shape[1]
        # Standardize household features
        hh_mean = household_features.mean(axis=0)
        hh_std = household_features.std(axis=0) + 1e-6
        X = (household_features - hh_mean) / hh_std
    else:
        n_hh_features = 0
        X = None
    
    if has_prod_features:
        n_prod_features = product_features.shape[1]
        # Standardize product features
        prod_mean = product_features.mean(axis=0)
        prod_std = product_features.std(axis=0) + 1e-6
        Z = (product_features - prod_mean) / prod_std
    else:
        n_prod_features = 0
        Z = None
    
    with pm.Model() as choice_model:
        # Product intercepts (baseline purchase probabilities)
        # Slightly tighter prior
        alpha = pm.Normal('alpha', mu=0, sigma=1.5, shape=n_items)
        
        # Initialize linear predictor with intercepts
        linear_pred = alpha.reshape((1, n_items))  # Broadcast-ready shape
        
        # Household feature effects (product-specific) - NON-CENTERED
        if has_hh_features:
            # Hierarchical prior with non-centered parameterization
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=0.5, shape=n_hh_features)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.5, shape=n_hh_features)
            
            # Non-centered: beta_raw ~ N(0,1), then beta = mu + sigma * beta_raw
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, 
                                shape=(n_items, n_hh_features))
            beta = pm.Deterministic('beta', mu_beta + sigma_beta * beta_raw)
            
            # Add to linear predictor
            linear_pred = linear_pred + pm.math.dot(X, beta.T)
        
        # Product feature effects (shared across households)
        if has_prod_features:
            gamma = pm.Normal('gamma', mu=0, sigma=0.5, shape=n_prod_features)
            prod_effect = pm.math.dot(Z, gamma)
            linear_pred = linear_pred + prod_effect
        
        # Optional: Household random effects - NON-CENTERED
        if include_random_effects:
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=0.5)
            hh_effect_raw = pm.Normal('hh_effect_raw', mu=0, sigma=1, shape=n_obs)
            hh_effect = pm.Deterministic('hh_effect', sigma_hh * hh_effect_raw)
            linear_pred = linear_pred + hh_effect[:, None]
        
        # Logistic link function
        p = pm.math.sigmoid(linear_pred)
        
        # Likelihood
        y_obs = pm.Bernoulli('y_obs', p=p, observed=purchase_data)
        
        # Sample - try nutpie first, fall back to standard
        try:
            import nutpie
            trace = pm.sample(
                n_samples, 
                tune=n_tune, 
                random_seed=seed,
                return_inferencedata=True, 
                progressbar=True,
                nuts_sampler="nutpie",
                target_accept=0.9
            )
        except (ImportError, Exception):
            trace = pm.sample(
                n_samples, 
                tune=n_tune, 
                random_seed=seed,
                return_inferencedata=True, 
                progressbar=True,
                cores=1,
                target_accept=0.9,
                compile_kwargs={"mode": "FAST_COMPILE"}
            )
    
    # Extract results
    results = {
        'alpha': trace.posterior['alpha'].mean(dim=['chain', 'draw']).values,
        'alpha_std': trace.posterior['alpha'].std(dim=['chain', 'draw']).values,
        'trace': trace,
        'n_items': n_items,
        'product_names': product_names
    }
    
    # Divergence count
    try:
        results['n_divergences'] = int(trace.sample_stats.diverging.sum().values)
    except Exception:
        results['n_divergences'] = 0
    
    if has_hh_features:
        results['beta'] = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        results['beta_std'] = trace.posterior['beta'].std(dim=['chain', 'draw']).values
        results['mu_beta'] = trace.posterior['mu_beta'].mean(dim=['chain', 'draw']).values
        results['hh_feature_scaling'] = {'mean': hh_mean, 'std': hh_std}
    
    if has_prod_features:
        results['gamma'] = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        results['gamma_std'] = trace.posterior['gamma'].std(dim=['chain', 'draw']).values
        results['prod_feature_scaling'] = {'mean': prod_mean, 'std': prod_std}
    
    if include_random_effects:
        results['hh_effect'] = trace.posterior['hh_effect'].mean(dim=['chain', 'draw']).values
        results['sigma_hh'] = float(trace.posterior['sigma_hh'].mean(dim=['chain', 'draw']).values)
    
    # Model diagnostics
    try:
        results['waic'] = az.waic(trace)
    except Exception:
        results['waic'] = None
    
    # Compute predicted probabilities
    pred_probs = 1 / (1 + np.exp(-results['alpha']))
    results['predicted_probs'] = pred_probs
    
    return results


# =============================================================================
# NON-NEGATIVE MATRIX FACTORIZATION (NMF)
# =============================================================================

def fit_nmf(data: np.ndarray, n_components: int, max_iter: int = 200, 
            tol: float = 1e-4, seed: int = 42) -> dict:
    """Non-negative Matrix Factorization using multiplicative update rules."""
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
    n_dims = min(n_dims, contributions.shape[0])
    
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
    n_factors = loadings.shape[1]
    factor_labels = [f"Factor {k+1}" for k in range(n_factors)]
    
    # Create text showing mean ± std
    text = np.array([[f"{loadings[i,j]:.2f}±{loadings_std[i,j]:.2f}" 
                      for j in range(n_factors)] for i in range(len(item_names))])
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=factor_labels,
        y=item_names,
        colorscale='RdBu_r',
        zmid=0,
        text=text,
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


def plot_variance_explained(var_explained_pct: np.ndarray, model_name: str) -> go.Figure:
    """Bar chart of variance explained by each factor/component."""
    n_components = len(var_explained_pct)
    labels = [f"Factor {k+1}" for k in range(n_components)]
    cumulative = np.cumsum(var_explained_pct)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=labels, y=var_explained_pct, name="Individual", marker_color='steelblue'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=labels, y=cumulative, name="Cumulative", 
                   mode='lines+markers', marker_color='coral'),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'Variance Explained - {model_name}',
        xaxis_title='Factor/Component',
        height=400
    )
    
    fig.update_yaxes(title_text="% Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    
    return fig


def plot_lca_profiles(item_probs: np.ndarray, item_names: list, class_probs: np.ndarray) -> go.Figure:
    """Create a heatmap of item probabilities by class."""
    n_classes = item_probs.shape[0]
    class_labels = [f"Class {k+1} ({class_probs[k]*100:.1f}%)" for k in range(n_classes)]
    
    fig = go.Figure(data=go.Heatmap(
        z=item_probs,
        x=item_names,
        y=class_labels,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=np.round(item_probs, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='Class: %{y}<br>Product: %{x}<br>P(Purchase): %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Purchase Probability by Latent Class',
        xaxis_title='Product',
        yaxis_title='Latent Class',
        height=max(300, 80 * n_classes),
        xaxis={'tickangle': 45}
    )
    
    return fig


def plot_correlation_matrix(corr_matrix: np.ndarray, item_names: list, title: str) -> go.Figure:
    """Heatmap of correlation matrix."""
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
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Latent Structure Analysis", layout="wide")
    
    st.title("🛒 Latent Structure Analysis for Purchase Data")
    st.markdown("""
    Discover latent customer segments and product relationships using multiple statistical methods.
    """)
    
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
                help="Numeric features describing households (e.g., income, household size)"
            )
            
            st.subheader("📦 Product Features (optional)")
            st.markdown("Enter product features as a table (one row per product)")
            
            use_product_features = st.checkbox("Add product features", value=False)
            
            if use_product_features:
                st.markdown("Edit the table below to add product features:")
                
                # Create editable dataframe for product features
                product_feature_template = pd.DataFrame({
                    'product': product_columns,
                    'price': [0.0] * len(product_columns),
                    'is_eco_friendly': [0] * len(product_columns),
                    'brand_tier': [1] * len(product_columns)
                })
                
                product_feature_df = st.data_editor(
                    product_feature_template,
                    num_rows="fixed",
                    use_container_width=True
                )
        
        if len(product_columns) < 2:
            st.warning("Please select at least 2 product columns for analysis")
            return
        
        # Validate binary columns
        data_subset = df[product_columns].copy()
        
        non_binary_cols = []
        for col in product_columns:
            unique_vals = data_subset[col].dropna().unique()
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                non_binary_cols.append(col)
        
        if non_binary_cols:
            st.error(f"These columns contain non-binary values: {non_binary_cols}")
            return
        
        # Handle missing values
        missing_counts = data_subset.isnull().sum()
        if missing_counts.sum() > 0:
            st.warning(f"Found {missing_counts.sum()} missing values. Rows with missing values will be excluded.")
            data_subset = data_subset.dropna()
        
        X = data_subset.values.astype(float)
        
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
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=comparison_df['Classes'], y=comparison_df['BIC'],
                                             mode='lines+markers', name='BIC', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=comparison_df['Classes'], y=comparison_df['AIC'],
                                             mode='lines+markers', name='AIC', line=dict(color='red')))
                    fig.update_layout(title='Model Selection: BIC and AIC by Number of Classes',
                                      xaxis_title='Number of Classes',
                                      yaxis_title='Information Criterion (lower is better)',
                                      height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    best_bic_k = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Classes']
                    st.info(f"💡 Based on BIC, the optimal number of classes is **{int(best_bic_k)}**")
                
                st.header(f"📊 {n_classes}-Class LCA Results")
                
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
                
                loadings = bfa_result['loadings']
                implied_corr = loadings @ loadings.T
                np.fill_diagonal(implied_corr, 1)
                
                similarity_matrix = implied_corr
            
            # =========== BAYESIAN FA (PyMC) ===========
            elif model_type == "Bayesian Factor Model (PyMC)":
                st.header("📊 Bayesian Factor Model (PyMC - Identified)")
                
                st.info("""
                **Model uses identified parameterization:**
                - Lower triangular loadings matrix with positive diagonal
                - This removes rotation invariance and sign ambiguity
                - Factor k is "anchored" by item k having positive loading
                """)
                
                with st.spinner("Running MCMC sampling (this may take a few minutes)..."):
                    bfa_result = fit_bayesian_factor_model_pymc(X, n_factors, 
                                                                n_samples=n_samples, 
                                                                n_tune=n_tune)
                
                st.success("MCMC sampling complete!")
                
                # Model diagnostics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if bfa_result['waic'] is not None:
                        st.metric("WAIC", f"{bfa_result['waic'].elpd_waic:.2f}")
                    else:
                        st.metric("WAIC", "N/A")
                with col2:
                    if bfa_result['waic'] is not None:
                        st.metric("Effective Parameters", f"{bfa_result['waic'].p_waic:.1f}")
                    else:
                        st.metric("Effective Parameters", "N/A")
                with col3:
                    n_div = bfa_result.get('n_divergences', 0)
                    if n_div > 0:
                        st.metric("⚠️ Divergences", n_div)
                    else:
                        st.metric("✅ Divergences", 0)
                
                if bfa_result.get('n_divergences', 0) > 0:
                    st.warning(f"Model had {bfa_result['n_divergences']} divergent transitions. "
                              "Results may be unreliable. Consider: fewer factors, more tuning samples, "
                              "or higher target_accept.")
                
                st.subheader("Factor Loadings (Posterior Mean ± SD)")
                st.caption("Lower triangular structure: Factor k is anchored by item k (diagonal)")
                fig = plot_loadings_with_uncertainty(bfa_result['loadings'], 
                                                     bfa_result['loadings_std'],
                                                     product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                fig = plot_variance_explained(bfa_result['var_explained_pct'], "Bayesian FA (PyMC)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Product Intercepts (Baseline Purchase Probability)")
                fig = plot_product_intercepts(bfa_result['alpha'], bfa_result['alpha_std'], product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                loadings = bfa_result['loadings']
                implied_corr = loadings @ loadings.T
                np.fill_diagonal(implied_corr, 1)
                
                similarity_matrix = implied_corr
            
            # =========== NMF ===========
            elif model_type == "Non-negative Matrix Factorization (NMF)":
                st.header("📊 Non-negative Matrix Factorization")
                
                with st.spinner("Fitting NMF model..."):
                    nmf_result = fit_nmf(X, n_components, max_iter=max_iter)
                
                st.success(f"Model converged in {nmf_result['n_iter']} iterations")
                
                st.metric("Final Reconstruction Error", f"{nmf_result['reconstruction_error']:.2f}")
                
                st.subheader("Convergence")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=nmf_result['reconstruction_errors'], mode='lines'))
                fig.update_layout(title='Reconstruction Error Over Iterations',
                                  xaxis_title='Iteration', yaxis_title='Error', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Component Loadings")
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
                
                st.success("MCA complete!")
                
                # Variance explained
                st.subheader("Explained Inertia (Variance)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Inertia", f"{mca_result['total_inertia']:.4f}")
                with col2:
                    total_explained = sum(mca_result['var_explained_pct'][:n_components])
                    st.metric("Total Explained", f"{total_explained:.1f}%")
                
                fig = plot_variance_explained(mca_result['var_explained_pct'], "MCA")
                st.plotly_chart(fig, use_container_width=True)
                
                # Biplot
                st.subheader("MCA Biplot")
                st.caption("Products close together have similar purchase patterns. Households near a product tend to buy it.")
                
                if n_components >= 2:
                    dim_options = list(range(n_components))
                    col1, col2 = st.columns(2)
                    with col1:
                        dim1 = st.selectbox("X-axis dimension", dim_options, index=0)
                    with col2:
                        dim2 = st.selectbox("Y-axis dimension", dim_options, index=1 if n_components > 1 else 0)
                    
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
                    n_dims=min(3, n_components)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Product coordinates (loadings equivalent)
                st.subheader("Product Coordinates (Dimension Loadings)")
                dim_labels = [f"Dim {k+1}" for k in range(n_components)]
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
                st.subheader("Baseline Purchase Probabilities")
                fig = plot_product_intercepts(dcm_result['alpha'], dcm_result['alpha_std'], product_columns)
                st.plotly_chart(fig, use_container_width=True)
                
                # Household feature effects
                if 'beta' in dcm_result:
                    st.subheader("Household Feature Effects by Product")
                    
                    fig = plot_beta_coefficients(
                        dcm_result['beta'],
                        dcm_result['beta_std'],
                        product_columns,
                        household_feature_columns
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary table
                    st.markdown("**Hierarchical Prior Means (μ_β):**")
                    mu_beta_df = pd.DataFrame({
                        'Feature': household_feature_columns,
                        'Mean Effect': dcm_result['mu_beta']
                    })
                    st.dataframe(mu_beta_df, use_container_width=True)
                
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
            
            # =========== HIERARCHICAL CLUSTERING ===========
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
                
                ### Discrete Choice Model Interpretation
                
                - **Product intercepts (α)**: Baseline purchase probability for each product
                - **Household feature effects (β)**: How household characteristics affect purchase of each product
                - **Product feature effects (γ)**: How product attributes affect purchase probability (shared across households)
                - **Household random effects**: Unobserved heterogeneity in purchase propensity
                
                ### Substitution Patterns
                - Products with opposite signs on household features → potential substitutes
                - Products with same signs → potential complements
                - Products on opposite sides of MCA origin → different customer bases
                """)
            
            # =========== DOWNLOADS ===========
            st.subheader("📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if similarity_matrix is not None:
                    sim_df = pd.DataFrame(similarity_matrix, index=product_columns, columns=product_columns)
                    csv_sim = sim_df.to_csv()
                    st.download_button(
                        "Download Similarity Matrix",
                        csv_sim,
                        "similarity_matrix.csv",
                        "text/csv"
                    )
            
            with col2:
                if model_type == "Latent Class Analysis (LCA)":
                    output_df = df.loc[data_subset.index].copy()
                    output_df['assigned_class'] = result['class_assignments'] + 1
                    for k in range(n_classes):
                        output_df[f'prob_class_{k+1}'] = result['responsibilities'][:, k]
                    csv = output_df.to_csv(index=False)
                    st.download_button("Download Class Assignments", csv, "lca_assignments.csv", "text/csv")
                
                elif model_type == "Factor Analysis (Tetrachoric)":
                    loadings_df = pd.DataFrame(fa_result['loadings'], index=product_columns,
                                               columns=[f'Factor_{k+1}' for k in range(n_factors)])
                    csv = loadings_df.to_csv()
                    st.download_button("Download Factor Loadings", csv, "factor_loadings.csv", "text/csv")
                
                elif model_type in ["Bayesian Factor Model (VI)", "Bayesian Factor Model (PyMC)"]:
                    loadings_df = pd.DataFrame(bfa_result['loadings'], index=product_columns,
                                               columns=[f'Factor_{k+1}' for k in range(n_factors)])
                    csv = loadings_df.to_csv()
                    st.download_button("Download Factor Loadings", csv, "bayesian_loadings.csv", "text/csv")
                
                elif model_type == "Non-negative Matrix Factorization (NMF)":
                    loadings_df = pd.DataFrame(nmf_result['loadings'], index=product_columns,
                                               columns=[f'Component_{k+1}' for k in range(n_components)])
                    csv = loadings_df.to_csv()
                    st.download_button("Download NMF Loadings", csv, "nmf_loadings.csv", "text/csv")
                
                elif model_type == "Multiple Correspondence Analysis (MCA)":
                    # Download product coordinates
                    coords_df = pd.DataFrame(
                        mca_result['column_coordinates'],
                        index=mca_result['product_labels'],
                        columns=[f'Dim_{k+1}' for k in range(n_components)]
                    )
                    csv = coords_df.to_csv()
                    st.download_button("Download MCA Coordinates", csv, "mca_coordinates.csv", "text/csv")
                
                elif model_type == "Discrete Choice Model (PyMC)":
                    # Download coefficient summary
                    coef_data = {'Product': product_columns, 'Intercept': dcm_result['alpha']}
                    if 'beta' in dcm_result:
                        for i, feat in enumerate(household_feature_columns):
                            coef_data[f'beta_{feat}'] = dcm_result['beta'][:, i]
                    coef_df = pd.DataFrame(coef_data)
                    csv = coef_df.to_csv(index=False)
                    st.download_button("Download Coefficients", csv, "dcm_coefficients.csv", "text/csv")


if __name__ == "__main__":
    main()