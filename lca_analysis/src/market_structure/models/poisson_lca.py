"""
Poisson and Zero-Inflated Poisson Latent Class Analysis for Count Data.

This module extends standard LCA to handle count data (e.g., number of purchases)
rather than binary purchase indicators. It provides three variants:

1. Poisson-LCA: Each class has a rate parameter λ_jk for each product
   y_ij | class=k ~ Poisson(λ_jk)

2. Zero-Inflated Poisson LCA (ZIP-LCA): Adds a zero-inflation parameter
   y_ij | class=k ~ ZIP(λ_jk, ψ_jk)
   where ψ is the probability of "structural zero" (never-buyers)

3. Negative Binomial LCA (NB-LCA): Handles overdispersion
   y_ij | class=k ~ NegativeBinomial(μ_jk, α_k)

The EM algorithm is adapted for these distributions:
- E-step: Compute posterior class membership using count likelihoods
- M-step: Update rate parameters (and zero-inflation/dispersion if applicable)

Key outputs:
- class_probs: Prior probability of each class (segment sizes)
- item_rates: λ_jk expected counts per product and class
- zero_probs: ψ_jk zero-inflation probabilities (ZIP only)
- responsibilities: Posterior class membership probabilities
"""

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional, Literal


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_poisson_pmf(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """
    Compute log Poisson PMF in a numerically stable way.
    
    log P(Y=y | λ) = y*log(λ) - λ - log(y!)
    
    Args:
        y: Count data, shape (n_obs, n_items) or broadcastable
        lam: Rate parameters, shape (n_classes, n_items) or broadcastable
        
    Returns:
        Log probabilities, shape depends on broadcasting
    """
    # Use gammaln for log(y!) = gammaln(y+1)
    # Add small epsilon to lambda to avoid log(0)
    lam_safe = np.maximum(lam, 1e-10)
    return y * np.log(lam_safe) - lam_safe - gammaln(y + 1)


def log_zip_pmf(y: np.ndarray, lam: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Compute log Zero-Inflated Poisson PMF.
    
    P(Y=0) = ψ + (1-ψ)*exp(-λ)
    P(Y=y>0) = (1-ψ) * Poisson(y; λ)
    
    Args:
        y: Count data, shape (n_obs, n_items)
        lam: Rate parameters, shape (n_classes, n_items)
        psi: Zero-inflation probabilities, shape (n_classes, n_items)
        
    Returns:
        Log probabilities
    """
    psi_safe = np.clip(psi, 1e-10, 1 - 1e-10)
    lam_safe = np.maximum(lam, 1e-10)
    
    # For y=0: log(ψ + (1-ψ)*exp(-λ))
    log_prob_zero = np.log(psi_safe + (1 - psi_safe) * np.exp(-lam_safe))
    
    # For y>0: log(1-ψ) + log_poisson(y, λ)
    log_prob_positive = np.log(1 - psi_safe) + log_poisson_pmf(y, lam_safe)
    
    # Combine based on whether y is zero
    is_zero = (y == 0)
    return np.where(is_zero, log_prob_zero, log_prob_positive)


def log_negbin_pmf(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Compute log Negative Binomial PMF (mean-dispersion parameterization).
    
    NB(y; μ, α) where:
    - μ is the mean
    - α is the dispersion (variance = μ + α*μ²)
    - r = 1/α (number of failures in alternative parameterization)
    
    Args:
        y: Count data
        mu: Mean parameters, shape (n_classes, n_items)
        alpha: Dispersion parameters, shape (n_classes,) or (n_classes, n_items)
        
    Returns:
        Log probabilities
    """
    mu_safe = np.maximum(mu, 1e-10)
    alpha_safe = np.maximum(alpha, 1e-10)
    
    # r = 1/alpha (the "size" parameter)
    r = 1.0 / alpha_safe
    
    # p = r / (r + mu) (success probability in standard parameterization)
    p = r / (r + mu_safe)
    
    # Log PMF: log(Gamma(y+r)) - log(Gamma(r)) - log(y!) + r*log(p) + y*log(1-p)
    log_prob = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + 
                r * np.log(p) + y * np.log(1 - p + 1e-10))
    
    return log_prob


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_poisson_lca_parameters(
    n_classes: int, 
    n_items: int,
    data: np.ndarray,
    model_type: Literal['poisson', 'zip', 'negbin'] = 'poisson',
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Initialize Poisson/ZIP/NB-LCA parameters.
    
    Uses data-driven initialization:
    - Class probabilities from Dirichlet
    - Item rates initialized around observed means with class-specific variation
    - Zero-inflation (ZIP) initialized from observed zero proportions
    - Dispersion (NB) initialized from observed variance/mean ratios
    
    Args:
        n_classes: Number of latent classes
        n_items: Number of products
        data: (n_obs, n_items) count matrix for data-driven init
        model_type: 'poisson', 'zip', or 'negbin'
        seed: Random seed
        
    Returns:
        Dictionary with initialized parameters
    """
    np.random.seed(seed)
    
    # Class probabilities
    class_probs = np.random.dirichlet(np.ones(n_classes))
    
    # Data-driven rate initialization
    observed_means = data.mean(axis=0)  # (n_items,)
    observed_means = np.maximum(observed_means, 0.1)  # Avoid zero rates
    
    # Initialize rates with class-specific variation around observed means
    # Classes get different "profiles" - some higher, some lower
    rate_multipliers = np.exp(np.random.randn(n_classes, 1) * 0.5)  # Log-normal variation
    item_rates = rate_multipliers * observed_means[np.newaxis, :]
    item_rates = np.clip(item_rates, 0.01, None)  # Ensure positive
    
    params = {
        'class_probs': class_probs,
        'item_rates': item_rates,
    }
    
    if model_type == 'zip':
        # Initialize zero-inflation from observed zero proportions
        observed_zero_prop = (data == 0).mean(axis=0)  # (n_items,)
        # Add class variation
        zero_probs = np.random.beta(2, 2, size=(n_classes, n_items))
        # Center around observed proportions
        zero_probs = zero_probs * 0.5 + observed_zero_prop[np.newaxis, :] * 0.5
        zero_probs = np.clip(zero_probs, 0.01, 0.99)
        params['zero_probs'] = zero_probs
        
    elif model_type == 'negbin':
        # Initialize dispersion from observed overdispersion
        observed_var = data.var(axis=0)
        observed_mean = data.mean(axis=0) + 1e-10
        # Estimate alpha from var = mu + alpha*mu^2
        # alpha = (var - mu) / mu^2
        est_alpha = np.maximum((observed_var - observed_mean) / (observed_mean**2), 0.1)
        # Use mean estimate as starting point, with class variation
        dispersion = np.random.gamma(2, est_alpha.mean() / 2, size=n_classes)
        dispersion = np.clip(dispersion, 0.01, 10)
        params['dispersion'] = dispersion
    
    return params


# =============================================================================
# E-STEP IMPLEMENTATIONS
# =============================================================================

def poisson_lca_e_step(
    data: np.ndarray,
    class_probs: np.ndarray,
    item_rates: np.ndarray,
    return_log_likelihood: bool = False
) -> Tuple[np.ndarray, Optional[float]]:
    """
    E-step for Poisson LCA: compute posterior class memberships.
    
    Fully vectorized using matrix operations.
    
    Args:
        data: (n_obs, n_items) count matrix
        class_probs: (n_classes,) prior class probabilities
        item_rates: (n_classes, n_items) Poisson rate parameters
        return_log_likelihood: Whether to also return log-likelihood
        
    Returns:
        responsibilities: (n_obs, n_classes) posterior class probabilities
        log_likelihood: Total log-likelihood (if requested)
    """
    n_obs, n_items = data.shape
    n_classes = len(class_probs)
    
    # Compute log P(x_i | class c) for all observations and classes
    # log P(x_i | c) = sum_j log P(y_ij | λ_jc)
    
    # Shape: (n_obs, 1, n_items) @ (1, n_classes, n_items).T style computation
    # Using einsum for clarity: sum over items of log_poisson for each obs, class
    
    log_class_probs = np.log(class_probs + 1e-10)  # (n_classes,)
    
    # Compute log-likelihood for each observation under each class
    # log_lik[i, c] = sum_j log_poisson_pmf(y_ij, λ_jc)
    log_lik = np.zeros((n_obs, n_classes))
    for c in range(n_classes):
        # (n_obs, n_items) with rates broadcast from (n_items,)
        log_lik[:, c] = log_poisson_pmf(data, item_rates[c, :]).sum(axis=1)
    
    # log P(c) + log P(x_i | c)
    log_joint = log_class_probs + log_lik  # (n_obs, n_classes)
    
    # Responsibilities via log-sum-exp normalization
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    log_sum_exp = max_log_joint + np.log(np.exp(log_joint - max_log_joint).sum(axis=1, keepdims=True))
    log_responsibilities = log_joint - log_sum_exp
    responsibilities = np.exp(log_responsibilities)
    
    if return_log_likelihood:
        # Log-likelihood is sum of log marginal probabilities
        log_likelihood = log_sum_exp.sum()
        return responsibilities, log_likelihood
    
    return responsibilities, None


def zip_lca_e_step(
    data: np.ndarray,
    class_probs: np.ndarray,
    item_rates: np.ndarray,
    zero_probs: np.ndarray,
    return_log_likelihood: bool = False
) -> Tuple[np.ndarray, Optional[float]]:
    """
    E-step for Zero-Inflated Poisson LCA.
    
    Args:
        data: (n_obs, n_items) count matrix
        class_probs: (n_classes,) prior class probabilities
        item_rates: (n_classes, n_items) Poisson rate parameters
        zero_probs: (n_classes, n_items) zero-inflation probabilities
        return_log_likelihood: Whether to return log-likelihood
        
    Returns:
        responsibilities: (n_obs, n_classes) posterior class probabilities
        log_likelihood: Total log-likelihood (if requested)
    """
    n_obs = data.shape[0]
    n_classes = len(class_probs)
    
    log_class_probs = np.log(class_probs + 1e-10)
    
    # Compute log-likelihood under ZIP for each observation and class
    log_lik = np.zeros((n_obs, n_classes))
    for c in range(n_classes):
        log_lik[:, c] = log_zip_pmf(data, item_rates[c, :], zero_probs[c, :]).sum(axis=1)
    
    log_joint = log_class_probs + log_lik
    
    # Normalize
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    log_sum_exp = max_log_joint + np.log(np.exp(log_joint - max_log_joint).sum(axis=1, keepdims=True))
    responsibilities = np.exp(log_joint - log_sum_exp)
    
    if return_log_likelihood:
        return responsibilities, log_sum_exp.sum()
    
    return responsibilities, None


def negbin_lca_e_step(
    data: np.ndarray,
    class_probs: np.ndarray,
    item_rates: np.ndarray,
    dispersion: np.ndarray,
    return_log_likelihood: bool = False
) -> Tuple[np.ndarray, Optional[float]]:
    """
    E-step for Negative Binomial LCA.
    
    Args:
        data: (n_obs, n_items) count matrix
        class_probs: (n_classes,) prior class probabilities
        item_rates: (n_classes, n_items) mean parameters (μ)
        dispersion: (n_classes,) dispersion parameters (α)
        return_log_likelihood: Whether to return log-likelihood
        
    Returns:
        responsibilities, log_likelihood
    """
    n_obs = data.shape[0]
    n_classes = len(class_probs)
    
    log_class_probs = np.log(class_probs + 1e-10)
    
    log_lik = np.zeros((n_obs, n_classes))
    for c in range(n_classes):
        # Broadcast dispersion to all items
        alpha_c = dispersion[c] * np.ones(item_rates.shape[1])
        log_lik[:, c] = log_negbin_pmf(data, item_rates[c, :], alpha_c).sum(axis=1)
    
    log_joint = log_class_probs + log_lik
    
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    log_sum_exp = max_log_joint + np.log(np.exp(log_joint - max_log_joint).sum(axis=1, keepdims=True))
    responsibilities = np.exp(log_joint - log_sum_exp)
    
    if return_log_likelihood:
        return responsibilities, log_sum_exp.sum()
    
    return responsibilities, None


# =============================================================================
# M-STEP IMPLEMENTATIONS
# =============================================================================

def poisson_lca_m_step(
    data: np.ndarray,
    responsibilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M-step for Poisson LCA: update class probabilities and item rates.
    
    The MLE for Poisson rate given soft assignments is:
    λ_jc = (sum_i r_ic * y_ij) / (sum_i r_ic)
    
    Args:
        data: (n_obs, n_items) count matrix
        responsibilities: (n_obs, n_classes) posterior class probabilities
        
    Returns:
        class_probs: (n_classes,) updated class probabilities
        item_rates: (n_classes, n_items) updated rate parameters
    """
    n_obs = data.shape[0]
    
    # Class probabilities
    class_counts = responsibilities.sum(axis=0)  # (n_classes,)
    class_probs = class_counts / n_obs
    
    # Item rates: weighted mean of counts
    # Numerator: sum_i r_ic * y_ij for each class c and item j
    weighted_counts = responsibilities.T @ data  # (n_classes, n_items)
    item_rates = weighted_counts / (class_counts[:, np.newaxis] + 1e-10)
    
    # Clip to avoid numerical issues
    item_rates = np.clip(item_rates, 0.001, None)
    
    return class_probs, item_rates


def zip_lca_m_step(
    data: np.ndarray,
    responsibilities: np.ndarray,
    item_rates: np.ndarray,
    zero_probs: np.ndarray,
    n_inner_iter: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step for ZIP LCA: update class probs, rates, and zero-inflation.
    
    The ZIP M-step requires iterating between updating ψ and λ since they're
    coupled through the posterior probability that a zero came from the
    point mass vs the Poisson component.
    
    Args:
        data: (n_obs, n_items) count matrix
        responsibilities: (n_obs, n_classes) posterior class probabilities
        item_rates: Current rate estimates (for inner iteration)
        zero_probs: Current zero-inflation estimates
        n_inner_iter: Number of inner EM iterations for ZIP parameters
        
    Returns:
        class_probs, item_rates, zero_probs
    """
    n_obs, n_items = data.shape
    n_classes = responsibilities.shape[1]
    
    # Class probabilities (same as Poisson)
    class_counts = responsibilities.sum(axis=0)
    class_probs = class_counts / n_obs
    
    # Inner EM for ZIP parameters
    # We compute the posterior probability that each zero is structural:
    # P(structural | y=0, class=c) = ψ_jc / (ψ_jc + (1-ψ_jc)*exp(-λ_jc))
    
    new_rates = item_rates.copy()
    new_zero_probs = zero_probs.copy()
    
    for _ in range(n_inner_iter):
        for c in range(n_classes):
            r_c = responsibilities[:, c]  # (n_obs,)
            weighted_n = r_c.sum() + 1e-10
            
            for j in range(n_items):
                y_j = data[:, j]  # (n_obs,)
                lam = np.maximum(new_rates[c, j], 0.01)
                psi = new_zero_probs[c, j]
                
                # Posterior prob of structural zero for observations with y=0
                is_zero = (y_j == 0)
                
                # P(structural | y=0) = ψ / (ψ + (1-ψ)*exp(-λ))
                poisson_zero_prob = np.exp(-lam)
                denom = psi + (1 - psi) * poisson_zero_prob
                p_structural_given_zero = psi / (denom + 1e-10)
                
                # For y > 0, P(structural) = 0
                p_structural = np.where(is_zero, p_structural_given_zero, 0.0)
                
                # Update ψ: expected fraction of structural zeros
                # ψ_new = E[# structural zeros] / E[# observations]
                # = sum_i r_ic * p_structural_i / sum_i r_ic
                weighted_structural = (r_c * p_structural).sum()
                new_zero_probs[c, j] = np.clip(weighted_structural / weighted_n, 0.01, 0.95)
                
                # Update λ: expected rate from Poisson component
                # For zeros: contribute λ * P(from Poisson | y=0) = λ * (1-p_structural)
                # For non-zeros: contribute y * 1
                # But we're estimating the rate, so:
                # λ_new = sum_i r_ic * y_i / sum_i r_ic * (1 - p_structural_i)
                
                # Weight by probability of coming from Poisson component
                poisson_weights = r_c * (1 - p_structural)
                weighted_sum = (poisson_weights * y_j).sum()
                weight_total = poisson_weights.sum() + 1e-10
                new_rates[c, j] = weighted_sum / weight_total
        
        new_rates = np.clip(new_rates, 0.01, None)
    
    return class_probs, new_rates, new_zero_probs


def negbin_lca_m_step(
    data: np.ndarray,
    responsibilities: np.ndarray,
    dispersion: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step for Negative Binomial LCA.
    
    Updates class probs, item rates (μ), and dispersion (α).
    Dispersion is estimated via method of moments or Newton-Raphson.
    
    Args:
        data: (n_obs, n_items) count matrix
        responsibilities: (n_obs, n_classes) posterior class probabilities
        dispersion: Current dispersion estimates
        
    Returns:
        class_probs, item_rates, dispersion
    """
    n_obs, n_items = data.shape
    n_classes = responsibilities.shape[1]
    
    # Class probabilities
    class_counts = responsibilities.sum(axis=0)
    class_probs = class_counts / n_obs
    
    # Item rates (same as Poisson - MLE for mean)
    weighted_counts = responsibilities.T @ data
    item_rates = weighted_counts / (class_counts[:, np.newaxis] + 1e-10)
    item_rates = np.clip(item_rates, 0.001, None)
    
    # Update dispersion per class using method of moments
    new_dispersion = np.zeros(n_classes)
    for c in range(n_classes):
        r_c = responsibilities[:, c]
        weighted_n = r_c.sum()
        
        # Weighted mean and variance
        weighted_mean = (r_c[:, np.newaxis] * data).sum(axis=0) / (weighted_n + 1e-10)
        weighted_var = (r_c[:, np.newaxis] * (data - weighted_mean)**2).sum(axis=0) / (weighted_n + 1e-10)
        
        # Estimate alpha from var = mu + alpha*mu^2
        # alpha = (var - mu) / mu^2, averaged across items
        mu_safe = np.maximum(weighted_mean, 0.01)
        alpha_est = (weighted_var - mu_safe) / (mu_safe**2 + 1e-10)
        alpha_est = np.maximum(alpha_est, 0.01)
        
        new_dispersion[c] = np.median(alpha_est)  # Robust estimate
    
    new_dispersion = np.clip(new_dispersion, 0.01, 10)
    
    return class_probs, item_rates, new_dispersion


# =============================================================================
# MAIN FITTING FUNCTIONS
# =============================================================================

def fit_poisson_lca(
    data: np.ndarray,
    n_classes: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 10,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Fit Poisson LCA model using EM algorithm.
    
    Runs multiple random initializations and returns best solution.
    
    Args:
        data: (n_households, n_items) count matrix (non-negative integers)
        n_classes: Number of latent classes
        max_iter: Maximum EM iterations per initialization
        tol: Convergence tolerance for log-likelihood
        n_init: Number of random initializations
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with:
        - class_probs: (n_classes,) class prior probabilities
        - item_rates: (n_classes, n_items) Poisson rate parameters
        - responsibilities: (n_obs, n_classes) posterior memberships
        - log_likelihood: Final log-likelihood
        - n_iter: Number of iterations
        - bic, aic: Information criteria
    """
    n_obs, n_items = data.shape
    best_result = None
    best_ll = -np.inf
    
    for init in range(n_init):
        # Initialize
        params = initialize_poisson_lca_parameters(
            n_classes, n_items, data, model_type='poisson', seed=seed + init
        )
        class_probs = params['class_probs']
        item_rates = params['item_rates']
        
        prev_ll = -np.inf
        
        for n_iter in range(1, max_iter + 1):
            # E-step
            responsibilities, ll = poisson_lca_e_step(
                data, class_probs, item_rates, return_log_likelihood=True
            )
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            
            # M-step
            class_probs, item_rates = poisson_lca_m_step(data, responsibilities)
        
        if verbose:
            print(f"Init {init + 1}/{n_init}: LL = {ll:.2f}, iter = {n_iter}")
        
        if ll > best_ll:
            best_ll = ll
            best_result = {
                'class_probs': class_probs.copy(),
                'item_rates': item_rates.copy(),
                'responsibilities': responsibilities.copy(),
                'log_likelihood': ll,
                'n_iter': n_iter,
            }
    
    # Compute information criteria
    # Parameters: (K-1) class probs + K*J rate parameters
    n_params = (n_classes - 1) + n_classes * n_items
    best_result['bic'] = -2 * best_result['log_likelihood'] + n_params * np.log(n_obs)
    best_result['aic'] = -2 * best_result['log_likelihood'] + 2 * n_params
    best_result['n_classes'] = n_classes
    best_result['model_type'] = 'poisson'
    
    return best_result


def fit_zip_lca(
    data: np.ndarray,
    n_classes: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 10,
    n_inner_iter: int = 3,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Fit Zero-Inflated Poisson LCA model using EM algorithm.
    
    Appropriate when there are "structural zeros" - households that never
    buy certain products, beyond what a Poisson model would predict.
    
    Args:
        data: (n_households, n_items) count matrix
        n_classes: Number of latent classes
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        n_init: Number of random initializations
        n_inner_iter: Inner iterations for ZIP parameter updates
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with model results including zero_probs
    """
    n_obs, n_items = data.shape
    best_result = None
    best_ll = -np.inf
    
    for init in range(n_init):
        params = initialize_poisson_lca_parameters(
            n_classes, n_items, data, model_type='zip', seed=seed + init
        )
        class_probs = params['class_probs']
        item_rates = params['item_rates']
        zero_probs = params['zero_probs']
        
        prev_ll = -np.inf
        
        for n_iter in range(1, max_iter + 1):
            # E-step
            responsibilities, ll = zip_lca_e_step(
                data, class_probs, item_rates, zero_probs, return_log_likelihood=True
            )
            
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            
            # M-step
            class_probs, item_rates, zero_probs = zip_lca_m_step(
                data, responsibilities, item_rates, zero_probs, n_inner_iter
            )
        
        if verbose:
            print(f"Init {init + 1}/{n_init}: LL = {ll:.2f}, iter = {n_iter}")
        
        if ll > best_ll:
            best_ll = ll
            best_result = {
                'class_probs': class_probs.copy(),
                'item_rates': item_rates.copy(),
                'zero_probs': zero_probs.copy(),
                'responsibilities': responsibilities.copy(),
                'log_likelihood': ll,
                'n_iter': n_iter,
            }
    
    # Parameters: (K-1) class + K*J rates + K*J zero_probs
    n_params = (n_classes - 1) + 2 * n_classes * n_items
    best_result['bic'] = -2 * best_result['log_likelihood'] + n_params * np.log(n_obs)
    best_result['aic'] = -2 * best_result['log_likelihood'] + 2 * n_params
    best_result['n_classes'] = n_classes
    best_result['model_type'] = 'zip'
    
    return best_result


def fit_negbin_lca(
    data: np.ndarray,
    n_classes: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 10,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Fit Negative Binomial LCA model using EM algorithm.
    
    Appropriate when count data shows overdispersion (variance > mean).
    Each class has its own dispersion parameter.
    
    Args:
        data: (n_households, n_items) count matrix
        n_classes: Number of latent classes
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        n_init: Number of random initializations
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with model results including dispersion parameters
    """
    n_obs, n_items = data.shape
    best_result = None
    best_ll = -np.inf
    
    for init in range(n_init):
        params = initialize_poisson_lca_parameters(
            n_classes, n_items, data, model_type='negbin', seed=seed + init
        )
        class_probs = params['class_probs']
        item_rates = params['item_rates']
        dispersion = params['dispersion']
        
        prev_ll = -np.inf
        
        for n_iter in range(1, max_iter + 1):
            # E-step
            responsibilities, ll = negbin_lca_e_step(
                data, class_probs, item_rates, dispersion, return_log_likelihood=True
            )
            
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            
            # M-step
            class_probs, item_rates, dispersion = negbin_lca_m_step(
                data, responsibilities, dispersion
            )
        
        if verbose:
            print(f"Init {init + 1}/{n_init}: LL = {ll:.2f}, iter = {n_iter}")
        
        if ll > best_ll:
            best_ll = ll
            best_result = {
                'class_probs': class_probs.copy(),
                'item_rates': item_rates.copy(),
                'dispersion': dispersion.copy(),
                'responsibilities': responsibilities.copy(),
                'log_likelihood': ll,
                'n_iter': n_iter,
            }
    
    # Parameters: (K-1) class + K*J rates + K dispersion
    n_params = (n_classes - 1) + n_classes * n_items + n_classes
    best_result['bic'] = -2 * best_result['log_likelihood'] + n_params * np.log(n_obs)
    best_result['aic'] = -2 * best_result['log_likelihood'] + 2 * n_params
    best_result['n_classes'] = n_classes
    best_result['model_type'] = 'negbin'
    
    return best_result


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def fit_count_lca(
    data: np.ndarray,
    n_classes: int,
    model_type: Literal['poisson', 'zip', 'negbin'] = 'poisson',
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 10,
    seed: int = 42,
    verbose: bool = False,
    **kwargs
) -> Dict:
    """
    Unified interface for fitting LCA models to count data.
    
    Args:
        data: (n_households, n_items) count matrix
        n_classes: Number of latent classes
        model_type: 'poisson', 'zip', or 'negbin'
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        n_init: Number of initializations
        seed: Random seed
        verbose: Print progress
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model results dictionary
    """
    if model_type == 'poisson':
        return fit_poisson_lca(data, n_classes, max_iter, tol, n_init, seed, verbose)
    elif model_type == 'zip':
        n_inner_iter = kwargs.get('n_inner_iter', 3)
        return fit_zip_lca(data, n_classes, max_iter, tol, n_init, n_inner_iter, seed, verbose)
    elif model_type == 'negbin':
        return fit_negbin_lca(data, n_classes, max_iter, tol, n_init, seed, verbose)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'poisson', 'zip', or 'negbin'.")


# =============================================================================
# POST-PROCESSING UTILITIES
# =============================================================================

def compute_count_lca_coordinates(
    class_probs: np.ndarray,
    item_rates: np.ndarray,
    responsibilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coordinates for visualizing count LCA results.
    
    Maps households and products into latent class space for biplots.
    For count data, we use log(rates) for product coordinates to make
    the scale more interpretable.
    
    Args:
        class_probs: (n_classes,) prior class probabilities
        item_rates: (n_classes, n_items) rate parameters
        responsibilities: (n_households, n_classes) posterior memberships
        
    Returns:
        (household_coords, product_coords)
    """
    household_coords = responsibilities
    # Use log-rates for product coordinates (more interpretable scaling)
    product_coords = np.log(item_rates + 0.01).T  # (n_items, n_classes)
    
    return household_coords, product_coords


def compute_expected_counts(
    responsibilities: np.ndarray,
    item_rates: np.ndarray,
    zero_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute expected counts under the fitted model.
    
    E[Y_ij] = sum_c r_ic * E[Y_ij | class=c]
    
    For ZIP: E[Y | class=c] = (1-ψ) * λ
    For Poisson/NB: E[Y | class=c] = λ (or μ)
    
    Args:
        responsibilities: (n_obs, n_classes) posterior memberships
        item_rates: (n_classes, n_items) rate parameters
        zero_probs: (n_classes, n_items) zero-inflation (ZIP only)
        
    Returns:
        expected_counts: (n_obs, n_items) expected count matrix
    """
    if zero_probs is not None:
        # ZIP: E[Y] = (1 - ψ) * λ
        class_means = (1 - zero_probs) * item_rates  # (n_classes, n_items)
    else:
        class_means = item_rates
    
    # Weighted sum over classes
    expected_counts = responsibilities @ class_means  # (n_obs, n_items)
    
    return expected_counts


def compute_residual_matrix(
    data: np.ndarray,
    responsibilities: np.ndarray,
    item_rates: np.ndarray,
    zero_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute Pearson residuals for model diagnostics.
    
    Residual_ij = (observed - expected) / sqrt(expected)
    
    Large residuals indicate poor fit for specific household-product pairs.
    """
    expected = compute_expected_counts(responsibilities, item_rates, zero_probs)
    residuals = (data - expected) / np.sqrt(expected + 1e-10)
    return residuals


def select_n_classes_count_lca(
    data: np.ndarray,
    max_classes: int = 10,
    model_type: Literal['poisson', 'zip', 'negbin'] = 'poisson',
    criterion: str = 'bic',
    **fit_kwargs
) -> Dict:
    """
    Select optimal number of classes using information criteria.
    
    Fits models with 2 to max_classes classes and returns results
    for model selection.
    
    Args:
        data: Count matrix
        max_classes: Maximum number of classes to try
        model_type: Which count LCA variant to use
        criterion: 'bic' or 'aic'
        **fit_kwargs: Additional arguments passed to fit function
        
    Returns:
        Dictionary with results for each n_classes
    """
    results = {}
    
    for k in range(2, max_classes + 1):
        result = fit_count_lca(data, k, model_type=model_type, **fit_kwargs)
        results[k] = {
            'n_classes': k,
            'log_likelihood': result['log_likelihood'],
            'bic': result['bic'],
            'aic': result['aic'],
            'n_iter': result['n_iter'],
        }
    
    # Find optimal
    if criterion == 'bic':
        optimal_k = min(results.keys(), key=lambda k: results[k]['bic'])
    else:
        optimal_k = min(results.keys(), key=lambda k: results[k]['aic'])
    
    return {
        'results_by_k': results,
        'optimal_k': optimal_k,
        'criterion': criterion,
    }