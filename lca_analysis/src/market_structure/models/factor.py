"""
Factor Analysis for Binary Data.

This module provides two factor analysis approaches suitable for binary data:

1. Tetrachoric Factor Analysis: Computes the tetrachoric correlation matrix
   (the correlation between latent continuous variables assumed to underlie
   each binary variable), then performs standard factor analysis on that matrix.

2. Bayesian Factor Model (Variational Inference): A fast approximate Bayesian
   approach using coordinate ascent variational inference. Provides uncertainty
   estimates without the computational cost of full MCMC.

Both approaches assume an underlying continuous latent structure that generates
the observed binary data through thresholding. This is more appropriate than
standard factor analysis on raw binary correlations.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from typing import Tuple, Dict, Optional


# =============================================================================
# TETRACHORIC CORRELATION
# =============================================================================

def compute_tetrachoric_single(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute tetrachoric correlation between two binary variables.
    
    The tetrachoric correlation estimates the correlation between two latent
    continuous variables assumed to generate the observed binary data through
    thresholding. It's computed by finding the bivariate normal correlation
    that produces the observed 2x2 contingency table.
    
    Args:
        x: Binary array (0/1 values)
        y: Binary array (0/1 values)
        
    Returns:
        Tetrachoric correlation coefficient in [-1, 1]
    """
    # Build 2x2 contingency table with small regularization
    # to avoid edge cases with zero cells
    n = len(x)
    a = np.sum((x == 1) & (y == 1)) + 0.5  # Both 1
    b = np.sum((x == 1) & (y == 0)) + 0.5  # x=1, y=0
    c = np.sum((x == 0) & (y == 1)) + 0.5  # x=0, y=1
    d = np.sum((x == 0) & (y == 0)) + 0.5  # Both 0
    
    total = a + b + c + d
    
    # Compute marginal probabilities
    p_x1 = (a + b) / total  # P(x = 1)
    p_y1 = (a + c) / total  # P(y = 1)
    
    # Find thresholds on standard normal that give these marginals
    h = norm.ppf(1 - p_x1)  # Threshold for x
    k = norm.ppf(1 - p_y1)  # Threshold for y
    
    # Observed joint probability
    p_joint = a / total  # P(x=1, y=1)
    
    def neg_log_likelihood(rho: float) -> float:
        """
        Negative log-likelihood of observed joint probability given correlation.
        
        For bivariate normal with correlation rho, compute P(X > h, Y > k)
        using Drezner-Wesolowsky approximation for bivariate normal CDF.
        """
        # Constrain rho to avoid numerical issues
        rho = np.clip(rho, -0.99, 0.99)
        
        # Compute P(X > h, Y > k) for bivariate normal
        # Using conditional normal: X > h and Y > k
        # P(X > h, Y > k) = P(X > h) * P(Y > k | X > h)
        
        # More accurate: direct bivariate normal integration
        # Using approximation for speed
        from scipy import stats
        
        # P(X > h, Y > k) for standard bivariate normal with correlation rho
        # = P(X < -h, Y < -k) by symmetry
        try:
            # Owen's T function approximation
            # For bivariate normal, P(X>h, Y>k) with correlation rho
            denom = np.sqrt(1 - rho**2)
            
            # Compute using conditional probability
            p_x_gt_h = 1 - norm.cdf(h)
            
            # Conditional distribution of Y given X > h
            # This is complex for truncated normal, use numerical integration
            # For speed, use a simpler approximation
            
            # Alternative: Gaussian quadrature approximation
            # Using Owen's approximation for bivariate normal tail probability
            q = rho
            L = -h
            
            # P(X < L, Y < k) approximation
            from scipy.stats import multivariate_normal
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            rv = multivariate_normal(mean, cov)
            
            # P(X > h, Y > k) = P(X < -h, Y < -k) by symmetry
            p_model = rv.cdf([-h, -k])
            
        except:
            # Fallback to simpler approximation
            p_model = (1 - norm.cdf(h)) * (1 - norm.cdf(k))
            if rho > 0:
                p_model += rho * 0.1 * norm.pdf(h) * norm.pdf(k)
            else:
                p_model -= abs(rho) * 0.1 * norm.pdf(h) * norm.pdf(k)
            p_model = np.clip(p_model, 1e-10, 1 - 1e-10)
        
        # Negative log-likelihood
        p_model = np.clip(p_model, 1e-10, 1 - 1e-10)
        return -np.log(p_model) * a - np.log(1 - p_model) * (total - a)
    
    # Optimize to find best rho
    result = minimize_scalar(neg_log_likelihood, bounds=(-0.99, 0.99), method='bounded')
    
    return result.x


def compute_tetrachoric_correlation(data: np.ndarray) -> np.ndarray:
    """
    Compute the full tetrachoric correlation matrix for binary data.
    
    This is the appropriate correlation matrix to use when performing factor
    analysis on binary data. Standard Pearson correlations underestimate the
    true correlation between the underlying continuous variables.
    
    Args:
        data: (n_observations, n_items) binary matrix (0/1 values)
        
    Returns:
        (n_items, n_items) tetrachoric correlation matrix
    """
    n_items = data.shape[1]
    corr_matrix = np.eye(n_items)
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rho = compute_tetrachoric_single(data[:, i], data[:, j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
    
    return corr_matrix


# =============================================================================
# CLASSICAL FACTOR ANALYSIS
# =============================================================================

def fit_factor_analysis_tetrachoric(data: np.ndarray, n_factors: int,
                                     max_iter: int = 100) -> Dict:
    """
    Fit factor analysis using tetrachoric correlations.
    
    This is a two-step procedure:
    1. Compute the tetrachoric correlation matrix (appropriate for binary data)
    2. Extract factors using eigenvalue decomposition
    
    The tetrachoric approach properly accounts for the binary nature of the data
    by estimating the correlation between underlying continuous latent variables.
    
    Args:
        data: (n_observations, n_items) binary matrix
        n_factors: Number of factors to extract
        max_iter: Not currently used, included for API consistency
        
    Returns:
        Dictionary with:
        - loadings: (n_items, n_factors) factor loading matrix
        - eigenvalues: Eigenvalues from decomposition
        - var_explained_pct: Variance explained by each factor (%)
        - tetra_corr: The tetrachoric correlation matrix
        - n_iter: Number of iterations (always 1 for this method)
    """
    n_items = data.shape[1]
    
    # Step 1: Compute tetrachoric correlations
    tetra_corr = compute_tetrachoric_correlation(data)
    
    # Step 2: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(tetra_corr)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Extract top n_factors
    loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))
    
    # Variance explained
    # For correlation matrix, total variance = n_items
    var_explained = np.maximum(eigenvalues[:n_factors], 0)
    var_explained_pct = var_explained / n_items * 100
    
    return {
        'loadings': loadings,
        'eigenvalues': eigenvalues,
        'var_explained_pct': var_explained_pct,
        'tetra_corr': tetra_corr,
        'n_iter': 1
    }


# =============================================================================
# BAYESIAN FACTOR ANALYSIS (VARIATIONAL INFERENCE)
# =============================================================================

def fit_bayesian_factor_vi(data: np.ndarray, n_factors: int,
                           max_iter: int = 100, tol: float = 1e-4) -> Dict:
    """
    Fit Bayesian factor model using coordinate ascent variational inference.
    
    This is a fast approximate Bayesian approach that provides uncertainty
    estimates for the loadings without the computational cost of full MCMC.
    Uses a mean-field variational family with Gaussian variational distributions.
    
    The model assumes:
    - Loadings ~ N(0, 1)
    - Factors ~ N(0, I)
    - Observations ~ N(Loadings @ Factors.T, sigma^2)
    
    For binary data, this is an approximation (treating binary as continuous),
    but it often works well in practice for exploratory analysis.
    
    Args:
        data: (n_observations, n_items) binary matrix
        n_factors: Number of latent factors
        max_iter: Maximum VI iterations
        tol: Convergence tolerance on ELBO improvement
        
    Returns:
        Dictionary with:
        - loadings: (n_items, n_factors) posterior mean loadings
        - loadings_var: (n_items, n_factors) posterior variances
        - scores: (n_observations, n_factors) posterior mean factor scores
        - var_explained_pct: Variance explained by each factor
        - elbo_history: ELBO values over iterations
        - n_iter: Number of iterations to convergence
    """
    n_obs, n_items = data.shape
    
    # Initialize variational parameters
    np.random.seed(42)
    
    # Loadings: q(Lambda) = N(m_Lambda, S_Lambda)
    m_Lambda = np.random.randn(n_items, n_factors) * 0.1
    S_Lambda = np.ones((n_items, n_factors)) * 0.5  # Diagonal variance per element
    
    # Factors: q(F) = N(m_F, S_F)
    m_F = np.zeros((n_obs, n_factors))
    S_F = np.ones(n_factors)  # Shared diagonal variance across observations
    
    # Noise precision
    tau = 1.0
    
    # Track ELBO for convergence
    elbo_history = []
    
    for iteration in range(max_iter):
        # Update factors q(F)
        # Posterior precision: I + tau * sum_j Lambda_j Lambda_j^T
        Lambda_cov = np.diag(S_Lambda.sum(axis=0))  # Approximate covariance contribution
        F_precision = np.eye(n_factors) + tau * (m_Lambda.T @ m_Lambda + Lambda_cov * n_items / n_factors)
        F_cov = np.linalg.inv(F_precision)
        S_F = np.diag(F_cov)  # Store diagonal
        
        # Posterior mean: F_cov @ tau @ (data @ m_Lambda)
        m_F = tau * (data @ m_Lambda) @ F_cov
        
        # Update loadings q(Lambda)
        # For each item j, posterior precision is: 1 + tau * sum_i f_i f_i^T
        F_second_moment = m_F.T @ m_F + n_obs * np.diag(S_F)
        Lambda_precision = 1.0 + tau * np.diag(F_second_moment / n_obs)
        
        for j in range(n_items):
            S_Lambda[j, :] = 1.0 / Lambda_precision
            m_Lambda[j, :] = tau * S_Lambda[j, :] * (data[:, j] @ m_F)
        
        # Update noise precision tau
        # Expected reconstruction error
        recon = m_F @ m_Lambda.T
        sq_error = np.sum((data - recon) ** 2)
        # Add variance terms
        sq_error += n_obs * np.sum(S_Lambda * np.diag(F_second_moment / n_obs))
        sq_error += np.sum(S_F * np.sum(m_Lambda ** 2, axis=0))
        
        tau = (n_obs * n_items) / sq_error
        tau = np.clip(tau, 0.1, 10.0)  # Prevent extreme values
        
        # Compute ELBO (simplified)
        elbo = -0.5 * tau * sq_error
        elbo += 0.5 * n_obs * n_items * np.log(tau / (2 * np.pi))
        elbo += 0.5 * np.sum(np.log(S_Lambda))  # Entropy of q(Lambda)
        elbo += 0.5 * n_obs * np.sum(np.log(S_F))  # Entropy of q(F)
        
        elbo_history.append(elbo)
        
        # Check convergence
        if len(elbo_history) > 1 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            break
    
    # Compute variance explained
    var_explained = np.sum(m_Lambda ** 2, axis=0)
    var_explained_pct = var_explained / n_items * 100
    
    # Sort by variance explained
    sort_idx = np.argsort(var_explained)[::-1]
    m_Lambda = m_Lambda[:, sort_idx]
    S_Lambda = S_Lambda[:, sort_idx]
    m_F = m_F[:, sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]
    
    return {
        'loadings': m_Lambda,
        'loadings_var': S_Lambda,
        'scores': m_F,
        'var_explained_pct': var_explained_pct,
        'elbo_history': elbo_history,
        'n_iter': iteration + 1
    }


# =============================================================================
# FACTOR SCORE COMPUTATION
# =============================================================================

def compute_factor_scores_regression(data: np.ndarray, 
                                      loadings: np.ndarray) -> np.ndarray:
    """
    Compute factor scores using the regression (Thomson) method.
    
    This method computes factor scores as the regression of observed data
    on the factor loadings. It's appropriate when loadings are available
    but factor scores weren't computed during model fitting (e.g., for
    PyMC models where we sample loadings but not per-observation factors).
    
    The formula is: F = X @ Lambda @ (Lambda.T @ Lambda)^-1
    
    Args:
        data: (n_observations, n_items) data matrix (can be binary)
        loadings: (n_items, n_factors) factor loading matrix
        
    Returns:
        (n_observations, n_factors) matrix of factor scores
    """
    # Compute (Lambda.T @ Lambda)^-1
    LtL_inv = np.linalg.inv(loadings.T @ loadings + 1e-6 * np.eye(loadings.shape[1]))
    
    # Factor scores via regression
    scores = data @ loadings @ LtL_inv
    
    return scores