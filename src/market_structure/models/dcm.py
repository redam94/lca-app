"""
Discrete Choice Model (DCM) for Binary Purchase Data using PyMC.

This module implements a Bayesian discrete choice model that can incorporate:
- Product-specific intercepts (baseline purchase probabilities)
- Household features (demographics, characteristics)
- Product features (if available)
- Household random effects (unobserved heterogeneity)
- Latent product-household interactions (learned latent features)

The model treats each product's purchase as an independent binary choice
(Bernoulli outcome) with utility:

    U_ij = α_j + X_i β_j + Z_j γ + u_i + (Θ_i · Λ_j)
    P(purchase_ij = 1) = sigmoid(U_ij)

Where:
- α_j: Product-specific intercept
- X_i β_j: Effect of household features on product j
- Z_j γ: Effect of product features (shared coefficients)
- u_i: Household random effect
- Θ_i · Λ_j: Latent interaction (household preferences × product attributes)

The latent features (Θ and Λ) are particularly useful when explicit product
features aren't available. They learn an implicit embedding space where
products with similar latent features are substitutes, and households with
similar preferences have correlated purchase patterns.

Requires: pymc, pytensor, arviz
"""

import numpy as np
from typing import Dict, Optional

from ..config import PYMC_AVAILABLE, pm, pt, az


def fit_discrete_choice_model(purchase_data: np.ndarray, 
                               household_features: Optional[np.ndarray] = None,
                               product_features: Optional[np.ndarray] = None,
                               n_samples: int = 1000, 
                               n_tune: int = 500,
                               include_random_effects: bool = False,
                               n_latent_features: int = 0,
                               latent_prior_scale: float = 1.0) -> Dict:
    """
    Fit Discrete Choice Model using PyMC with MCMC sampling.
    
    This is a flexible model that can range from simple logistic regression
    (with just intercepts) to a sophisticated model with latent product-household
    interactions. The latent feature model is especially useful for discovering
    implicit product similarities without requiring explicit product attributes.
    
    The utility for household i choosing product j is:
        U_ij = α_j + (household effects) + (product effects) + (random effects) + (latent)
    
    With probability: P(purchase_ij = 1) = sigmoid(U_ij)
    
    Args:
        purchase_data: (n_households, n_products) binary purchase matrix
        household_features: Optional (n_households, n_hh_features) matrix of
                           household characteristics (should be standardized)
        product_features: Optional (n_products, n_prod_features) matrix of
                         product characteristics
        n_samples: Number of MCMC samples per chain
        n_tune: Number of tuning (warmup) samples
        include_random_effects: Whether to include household random effects
        n_latent_features: Number of latent dimensions for product-household
                          interactions. Set to 0 to disable latent features.
        latent_prior_scale: Prior scale for latent feature distributions.
                           Smaller values = stronger regularization.
        
    Returns:
        Dictionary with posterior estimates including:
        - alpha: (n_products,) product intercepts (posterior mean)
        - alpha_std: Standard deviations of alpha
        - beta: Household feature effects (if household_features provided)
        - gamma: Product feature effects (if product_features provided)
        - product_latent: (n_products, n_latent) latent product features
        - household_latent: (n_households, n_latent) latent preferences
        - trace: Full ArviZ InferenceData for diagnostics
        - n_latent_features: Number of latent dimensions used
        - waic: WAIC model comparison statistic
        - n_divergences: Number of divergent transitions
        
    Raises:
        ImportError: If PyMC is not available
    """
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC is required for this model. "
            "Install with: pip install pymc arviz"
        )
    
    n_households, n_items = purchase_data.shape
    
    with pm.Model() as dcm:
        # Product intercepts (baseline purchase probabilities)
        # Using a weakly informative prior centered at 0
        alpha = pm.Normal('alpha', mu=0, sigma=2, shape=n_items)
        
        # Initialize utility with just intercepts
        utility = alpha
        
        # Household feature effects (if provided)
        # Each product gets its own coefficient for each household feature
        if household_features is not None:
            n_hh_features = household_features.shape[1]
            # beta[j, k] = effect of household feature k on product j
            beta = pm.Normal('beta', mu=0, sigma=1, shape=(n_items, n_hh_features))
            # Compute household effects: (n_households, n_items)
            hh_effect = pm.math.dot(household_features, beta.T)
            utility = utility + hh_effect
        
        # Product feature effects (if provided)
        # Shared coefficients across products (each feature has one global effect)
        if product_features is not None:
            n_prod_features = product_features.shape[1]
            gamma = pm.Normal('gamma', mu=0, sigma=1, shape=n_prod_features)
            # Compute product effects: broadcasts to (n_households, n_items)
            prod_effect = pm.math.dot(product_features, gamma)
            utility = utility + prod_effect
        
        # Household random effects (unobserved heterogeneity)
        # Captures the idea that some households are just more likely to buy
        if include_random_effects:
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=1)
            hh_random = pm.Normal('hh_random', mu=0, sigma=sigma_hh, shape=n_households)
            # Add to utility (broadcasts across products)
            utility = utility + hh_random[:, None]
        
        # Latent product-household interactions
        # This learns implicit features that explain co-purchase patterns
        if n_latent_features > 0:
            # Hierarchical priors on the scale of latent features
            # This provides automatic relevance determination
            lambda_sd = pm.HalfNormal('lambda_sd', sigma=latent_prior_scale)
            theta_sd = pm.HalfNormal('theta_sd', sigma=latent_prior_scale)
            
            # Product latent features: Λ[j, k] = how much product j has attribute k
            product_latent = pm.Normal(
                'product_latent', 
                mu=0, 
                sigma=lambda_sd, 
                shape=(n_items, n_latent_features)
            )
            
            # Household latent preferences: Θ[i, k] = how much household i values attribute k
            household_latent = pm.Normal(
                'household_latent', 
                mu=0, 
                sigma=theta_sd,
                shape=(n_households, n_latent_features)
            )
            
            # Latent utility contribution: Θ @ Λ.T
            # Shape: (n_households, n_items)
            latent_utility = pm.math.dot(household_latent, product_latent.T)
            utility = utility + latent_utility
        
        # Likelihood: Bernoulli with logistic link
        p = pm.math.sigmoid(utility)
        likelihood = pm.Bernoulli('obs', p=p, observed=purchase_data)
        
        # Sample using nutpie
        trace = pm.sample(
            n_samples, 
            tune=n_tune, 
            nuts_sampler='nutpie',
            progressbar=True,
            return_inferencedata=True,
            target_accept=0.9,
            random_seed=42
        )
        
        # Compute log-likelihood for model comparison
        trace = pm.compute_log_likelihood(trace)
    
    # Extract posterior summaries
    result = {
        'alpha': trace.posterior['alpha'].mean(dim=['chain', 'draw']).values,
        'alpha_std': trace.posterior['alpha'].std(dim=['chain', 'draw']).values,
        'trace': trace,
        'n_latent_features': n_latent_features
    }
    
    # Add household feature effects if present
    if household_features is not None:
        result['beta'] = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        result['beta_std'] = trace.posterior['beta'].std(dim=['chain', 'draw']).values
    
    # Add product feature effects if present
    if product_features is not None:
        result['gamma'] = trace.posterior['gamma'].mean(dim=['chain', 'draw']).values
        result['gamma_std'] = trace.posterior['gamma'].std(dim=['chain', 'draw']).values
    
    # Add random effects if present
    if include_random_effects:
        result['sigma_hh'] = float(trace.posterior['sigma_hh'].mean().values)
        result['hh_random'] = trace.posterior['hh_random'].mean(dim=['chain', 'draw']).values
    
    # Add latent features if present
    if n_latent_features > 0:
        result['product_latent'] = trace.posterior['product_latent'].mean(dim=['chain', 'draw']).values
        result['household_latent'] = trace.posterior['household_latent'].mean(dim=['chain', 'draw']).values
        result['lambda_sd'] = float(trace.posterior['lambda_sd'].mean().values)
        result['theta_sd'] = float(trace.posterior['theta_sd'].mean().values)
    
    # Compute WAIC for model comparison
    try:
        result['waic'] = az.waic(trace)
    except Exception:
        result['waic'] = None
    
    # Count divergences
    result['n_divergences'] = int(trace.sample_stats['diverging'].sum().values)
    
    return result


def compute_dcm_product_similarity(product_latent: np.ndarray) -> np.ndarray:
    """
    Compute product similarity from DCM latent features.
    
    Products with similar latent feature vectors have similar relationships
    with households, indicating they may be substitutes or belong to the
    same category.
    
    Args:
        product_latent: (n_products, n_latent) matrix of latent features
        
    Returns:
        (n_products, n_products) cosine similarity matrix
    """
    # Normalize each product's latent vector
    norms = np.linalg.norm(product_latent, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = product_latent / norms
    
    # Cosine similarity
    similarity = normalized @ normalized.T
    
    return similarity


def interpret_dcm_latent_dimensions(product_latent: np.ndarray,
                                     product_names: list,
                                     top_n: int = 5) -> Dict:
    """
    Interpret latent dimensions by finding products that define each.
    
    For each latent dimension, identifies the products with highest and
    lowest loadings, which helps understand what the dimension captures.
    
    Args:
        product_latent: (n_products, n_latent) latent feature matrix
        product_names: List of product names
        top_n: Number of top/bottom products to return per dimension
        
    Returns:
        Dictionary with interpretation for each dimension
    """
    n_dims = product_latent.shape[1]
    interpretation = {}
    
    for d in range(n_dims):
        loadings = product_latent[:, d]
        sorted_idx = np.argsort(loadings)
        
        interpretation[f'dimension_{d+1}'] = {
            'high_loading_products': [product_names[i] for i in sorted_idx[-top_n:][::-1]],
            'high_loading_values': loadings[sorted_idx[-top_n:][::-1]].tolist(),
            'low_loading_products': [product_names[i] for i in sorted_idx[:top_n]],
            'low_loading_values': loadings[sorted_idx[:top_n]].tolist()
        }
    
    return interpretation