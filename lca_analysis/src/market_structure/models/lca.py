"""
Latent Class Analysis (LCA) for Binary Purchase Data.

LCA is a clustering method that identifies discrete "latent classes" or segments
of households based on their purchase patterns. Unlike factor analysis which
assumes continuous latent variables, LCA posits that each household belongs
to one of K discrete classes, each with its own probability profile across products.

The model is fit using the Expectation-Maximization (EM) algorithm:
- E-step: Compute posterior probability of class membership for each household
- M-step: Update class probabilities and item probabilities given assignments

Key outputs:
- class_probs: Prior probability of each class (segment sizes)
- item_probs: P(purchase | class) for each product and class (class profiles)
- responsibilities: Posterior class membership probabilities
"""

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_lca_parameters(n_classes: int, n_items: int, 
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize LCA parameters randomly.
    
    Uses Dirichlet prior for class probabilities and Beta prior for item
    probabilities to ensure valid probability values.
    
    Args:
        n_classes: Number of latent classes to fit
        n_items: Number of products/items
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (class_probs, item_probs) where:
        - class_probs: (n_classes,) array of class prior probabilities
        - item_probs: (n_classes, n_items) array of item probabilities per class
    """
    np.random.seed(seed)
    # Dirichlet with uniform concentration gives a valid probability simplex
    class_probs = np.random.dirichlet(np.ones(n_classes))
    # Beta(2,2) is a reasonable prior centered at 0.5 with some spread
    item_probs = np.random.beta(2, 2, size=(n_classes, n_items))
    return class_probs, item_probs


# =============================================================================
# EM ALGORITHM STEPS
# =============================================================================

def lca_e_step(data: np.ndarray, class_probs: np.ndarray, 
               item_probs: np.ndarray, 
               return_log_likelihood: bool = False) -> np.ndarray:
    """
    E-step: Compute posterior probability of class membership for each observation.
    
    Uses Bayes' theorem to compute P(class | data) for each household.
    Calculations are done in log-space for numerical stability.
    
    This implementation is fully vectorized over both observations AND classes,
    using matrix multiplication to compute log-likelihoods efficiently.
    
    Optionally returns the log-likelihood as a byproduct, since it shares
    computation with responsibility calculation (avoids redundant work in EM loop).
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        class_probs: (n_classes,) prior class probabilities
        item_probs: (n_classes, n_items) P(purchase | class)
        return_log_likelihood: If True, also return total log-likelihood
        
    Returns:
        responsibilities: (n_households, n_classes) posterior probabilities
        If return_log_likelihood=True, returns (responsibilities, log_likelihood)
    """
    # Precompute log probabilities (add small constant for numerical stability)
    log_item_probs = np.log(item_probs + 1e-10)          # (n_classes, n_items)
    log_1minus_item = np.log(1 - item_probs + 1e-10)     # (n_classes, n_items)
    log_class_probs = np.log(class_probs + 1e-10)        # (n_classes,)
    
    # Vectorized log-likelihood computation using matrix multiplication:
    # For each household i and class c, we need:
    #   sum_j [ x_ij * log(p_cj) + (1-x_ij) * log(1-p_cj) ]
    # 
    # This equals: data @ log(item_probs).T + (1-data) @ log(1-item_probs).T
    log_lik_purchases = data @ log_item_probs.T          # (n_obs, n_classes)
    log_lik_non_purchases = (1 - data) @ log_1minus_item.T  # (n_obs, n_classes)
    
    # log P(c) + log P(data | c) = log P(c, data)
    log_joint = log_class_probs + log_lik_purchases + log_lik_non_purchases
    
    # Log-sum-exp trick for numerical stability
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    log_joint_shifted = log_joint - max_log_joint
    exp_log_joint = np.exp(log_joint_shifted)
    sum_exp = exp_log_joint.sum(axis=1, keepdims=True)
    
    # Responsibilities: P(c | data) = P(c, data) / P(data)
    responsibilities = exp_log_joint / sum_exp
    
    if return_log_likelihood:
        # log P(data) = logsumexp(log P(c, data))
        # = max + log(sum(exp(shifted)))
        log_marginal = max_log_joint.squeeze() + np.log(sum_exp.squeeze())
        log_likelihood = log_marginal.sum()
        return responsibilities, log_likelihood
    
    return responsibilities


def lca_m_step(data: np.ndarray, responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    M-step: Update parameters given current responsibilities.
    
    Computes weighted sufficient statistics using the soft class assignments
    from the E-step.
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        responsibilities: (n_households, n_classes) posterior probabilities
        
    Returns:
        Tuple of (class_probs, item_probs) with updated parameters
    """
    n_obs = data.shape[0]
    
    # Class probabilities = average responsibility per class
    class_counts = responsibilities.sum(axis=0)
    class_probs = class_counts / n_obs
    
    # Item probabilities = weighted purchase rate per class
    # Numerator: sum of (responsibility * purchase) for each class
    # Denominator: sum of responsibilities (expected count in class)
    item_probs = (responsibilities.T @ data) / (class_counts[:, np.newaxis] + 1e-10)
    # Clip to avoid extreme probabilities that cause numerical issues
    item_probs = np.clip(item_probs, 0.01, 0.99)
    
    return class_probs, item_probs


def compute_lca_log_likelihood(data: np.ndarray, class_probs: np.ndarray, 
                                item_probs: np.ndarray) -> float:
    """
    Compute the observed data log-likelihood under the LCA model.
    
    This is the marginal likelihood, summing over all possible class assignments.
    Used for convergence checking and model comparison (BIC/AIC).
    
    This implementation is fully vectorized using the log-sum-exp trick:
    log P(x_i) = log sum_c [ P(c) * P(x_i|c) ]
               = logsumexp_c [ log P(c) + log P(x_i|c) ]
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        class_probs: (n_classes,) prior class probabilities
        item_probs: (n_classes, n_items) P(purchase | class)
        
    Returns:
        Total log-likelihood across all observations
    """
    # Precompute log probabilities
    log_item_probs = np.log(item_probs + 1e-10)          # (n_classes, n_items)
    log_1minus_item = np.log(1 - item_probs + 1e-10)     # (n_classes, n_items)
    log_class_probs = np.log(class_probs + 1e-10)        # (n_classes,)
    
    # Compute log P(x_i | class c) for all observations and classes
    # Using matrix multiplication for efficiency
    log_lik_purchases = data @ log_item_probs.T          # (n_obs, n_classes)
    log_lik_non_purchases = (1 - data) @ log_1minus_item.T  # (n_obs, n_classes)
    
    # log P(c) + log P(x_i | c) for all i, c
    log_joint = log_class_probs + log_lik_purchases + log_lik_non_purchases  # (n_obs, n_classes)
    
    # Marginal log-likelihood per observation using log-sum-exp
    # log P(x_i) = logsumexp_c [ log P(c, x_i) ]
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    log_marginal = max_log_joint.squeeze() + np.log(np.exp(log_joint - max_log_joint).sum(axis=1))
    
    # Total log-likelihood
    return log_marginal.sum()


# =============================================================================
# MAIN FITTING FUNCTION
# =============================================================================

def fit_lca(data: np.ndarray, n_classes: int, max_iter: int = 100, 
            tol: float = 1e-4, n_init: int = 10, seed: int = 42) -> Dict:
    """
    Fit Latent Class Analysis model using the EM algorithm.
    
    Runs multiple random initializations and returns the solution with
    highest log-likelihood to avoid local optima.
    
    The implementation uses fully vectorized E-step and M-step operations,
    computing log-likelihoods as a byproduct of the E-step to avoid
    redundant matrix operations.
    
    Args:
        data: (n_households, n_items) binary purchase matrix (0/1 values)
        n_classes: Number of latent classes to fit
        max_iter: Maximum EM iterations per initialization
        tol: Convergence tolerance on log-likelihood improvement
        n_init: Number of random initializations to try
        seed: Base random seed (incremented for each initialization)
        
    Returns:
        Dictionary with:
        - class_probs: (n_classes,) prior class probabilities
        - item_probs: (n_classes, n_items) purchase probabilities per class
        - responsibilities: (n_households, n_classes) posterior memberships
        - log_likelihood: Final log-likelihood
        - bic: Bayesian Information Criterion
        - aic: Akaike Information Criterion
        - n_iter: Number of iterations to convergence
        - n_classes: Number of classes (for reference)
    """
    n_obs, n_items = data.shape
    
    # Track best solution across initializations
    best_ll = -np.inf
    best_result = None
    
    for init in range(n_init):
        # Initialize with different random seed
        class_probs, item_probs = initialize_lca_parameters(
            n_classes, n_items, seed=seed + init
        )
        
        prev_ll = -np.inf
        n_iter = 0
        
        # EM iterations
        for iteration in range(max_iter):
            # E-step: compute responsibilities AND log-likelihood in one pass
            # This avoids redundant computation of log joint probabilities
            responsibilities, ll = lca_e_step(
                data, class_probs, item_probs, return_log_likelihood=True
            )
            
            # M-step: update parameters (already vectorized)
            class_probs, item_probs = lca_m_step(data, responsibilities)
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                n_iter = iteration + 1
                break
            
            prev_ll = ll
        else:
            n_iter = max_iter
        
        # Keep best solution
        if ll > best_ll:
            best_ll = ll
            best_result = {
                'class_probs': class_probs.copy(),
                'item_probs': item_probs.copy(),
                'responsibilities': responsibilities.copy(),
                'log_likelihood': ll,
                'n_iter': n_iter
            }
    
    # Compute information criteria for model selection
    # Number of parameters: (K-1) class probs + K*J item probs
    n_params = (n_classes - 1) + n_classes * n_items
    best_result['bic'] = -2 * best_result['log_likelihood'] + n_params * np.log(n_obs)
    best_result['aic'] = -2 * best_result['log_likelihood'] + 2 * n_params
    best_result['n_classes'] = n_classes
    
    return best_result


# =============================================================================
# POST-PROCESSING UTILITIES
# =============================================================================

def compute_lca_coordinates(class_probs: np.ndarray, item_probs: np.ndarray,
                            responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coordinates for visualizing LCA results in latent space.
    
    Maps both households and products into a common space defined by the
    latent classes. This enables biplot-style visualizations.
    
    Args:
        class_probs: (n_classes,) prior class probabilities
        item_probs: (n_classes, n_items) purchase probabilities per class
        responsibilities: (n_households, n_classes) posterior memberships
        
    Returns:
        Tuple of (household_coords, product_coords) where:
        - household_coords: (n_households, n_classes) responsibility vectors
        - product_coords: (n_items, n_classes) item probability vectors
    """
    # Households are represented by their responsibility vectors
    # (their "soft" position in class-space)
    household_coords = responsibilities
    
    # Products are represented by their probability profiles across classes
    # Transpose so shape is (n_items, n_classes)
    product_coords = item_probs.T
    
    return household_coords, product_coords


def compute_residual_correlations(data: np.ndarray, responsibilities: np.ndarray,
                                   item_probs: np.ndarray) -> np.ndarray:
    """
    Compute residual correlations between products after accounting for LCA classes.
    
    Residual correlations capture product relationships that aren't explained by
    class membership. Negative residuals often indicate substitution patterns
    (products that compete within segments). Positive residuals suggest
    complementary relationships beyond what class membership explains.
    
    The residual is computed as: observed_ij - E[observed_ij | class assignments]
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        responsibilities: (n_households, n_classes) posterior memberships
        item_probs: (n_classes, n_items) purchase probabilities per class
        
    Returns:
        (n_items, n_items) correlation matrix of residuals
    """
    n_obs, n_items = data.shape
    
    # Compute expected purchase probability for each household-product pair
    # by summing over class assignments weighted by responsibilities
    expected = responsibilities @ item_probs  # (n_households, n_items)
    
    # Residuals: difference between observed and expected
    residuals = data - expected
    
    # Compute correlation matrix of residuals
    # Center residuals (they should already be mean-zero approximately)
    residuals_centered = residuals - residuals.mean(axis=0)
    
    # Covariance and correlation
    cov = (residuals_centered.T @ residuals_centered) / (n_obs - 1)
    std = np.sqrt(np.diag(cov) + 1e-10)
    corr = cov / np.outer(std, std)
    
    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr, 1.0)
    
    return corr


# =============================================================================
# LCA WITH HOUSEHOLD COVARIATES
# =============================================================================
# 
# This extension allows household features (demographics, geography, etc.) to
# influence class membership probabilities. Instead of assuming all households
# have the same prior probability of belonging to each class, we model:
#
#   P(class = c | Z_i) = softmax(Z_i @ beta_c)
#
# where Z_i are household covariates and beta_c are class-specific coefficients.
# This is equivalent to a multinomial logistic regression predicting class
# membership from household features.
#
# Benefits:
# - Explains WHY certain households belong to certain segments
# - Improves classification by leveraging additional information
# - Enables demographic profiling of segments
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities along the last axis.
    
    Uses the log-sum-exp trick for numerical stability: subtracting the max
    before exponentiating prevents overflow while producing the same result.
    
    Args:
        x: Array of log-odds or scores. Shape can be (n_classes,) for a single
           observation or (n_obs, n_classes) for a batch.
           
    Returns:
        Array of same shape with probabilities summing to 1 along last axis.
    """
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def compute_class_probs_from_covariates(covariates: np.ndarray, 
                                         beta: np.ndarray) -> np.ndarray:
    """
    Compute household-specific class probabilities from covariates.
    
    Uses multinomial logistic regression (softmax) to map household features
    to class membership probabilities. Each household gets its own probability
    distribution over classes based on its covariate values.
    
    Args:
        covariates: (n_households, n_features) matrix of household features.
                   Should include a column of 1s for the intercept.
        beta: (n_features, n_classes) regression coefficients.
              The last class is typically the reference (beta[:, -1] = 0).
              
    Returns:
        (n_households, n_classes) matrix of class probabilities per household.
        Each row sums to 1.
    """
    # Linear predictor: Z @ beta gives log-odds relative to reference class
    # Shape: (n_households, n_classes)
    logits = covariates @ beta
    
    # Convert to probabilities via softmax
    return softmax(logits)


def initialize_lca_with_covariates(n_classes: int, n_items: int, n_features: int,
                                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize parameters for LCA with covariates.
    
    The regression coefficients (beta) start near zero, implying roughly equal
    class probabilities initially. Item probabilities are initialized with a
    Beta(2,2) prior as in standard LCA.
    
    Args:
        n_classes: Number of latent classes
        n_items: Number of products/items
        n_features: Number of covariates (including intercept)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (beta, item_probs) where:
        - beta: (n_features, n_classes) regression coefficients
        - item_probs: (n_classes, n_items) purchase probabilities per class
    """
    np.random.seed(seed)
    
    # Initialize regression coefficients with small random values
    # This gives roughly equal class probabilities at the start
    # We use the last class as reference (set to 0) for identifiability
    beta = np.random.randn(n_features, n_classes) * 0.1
    beta[:, -1] = 0  # Reference class constraint
    
    # Item probabilities: Beta(2,2) prior centered at 0.5
    item_probs = np.random.beta(2, 2, size=(n_classes, n_items))
    
    return beta, item_probs


def lca_e_step_with_covariates(data: np.ndarray, 
                                covariates: np.ndarray,
                                beta: np.ndarray,
                                item_probs: np.ndarray,
                                return_log_likelihood: bool = False):
    """
    E-step for LCA with covariates: compute responsibilities using 
    household-specific class priors.
    
    Instead of using a fixed class prior π_c for all households, each household
    has its own prior based on its covariates: P(class | Z_i) = softmax(Z_i @ β).
    
    The posterior is then:
        P(class | X_i, Z_i) ∝ P(class | Z_i) × P(X_i | class)
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        covariates: (n_households, n_features) household features (with intercept)
        beta: (n_features, n_classes) regression coefficients
        item_probs: (n_classes, n_items) P(purchase | class)
        return_log_likelihood: If True, also return the log-likelihood
        
    Returns:
        responsibilities: (n_households, n_classes) posterior probabilities
        If return_log_likelihood=True, returns (responsibilities, log_likelihood)
    """
    # Compute household-specific class priors from covariates
    # Shape: (n_households, n_classes)
    class_probs_per_hh = compute_class_probs_from_covariates(covariates, beta)
    
    # Compute log-likelihood of purchases under each class
    # This is the same vectorized computation as standard LCA
    log_item_probs = np.log(item_probs + 1e-10)
    log_1minus_item = np.log(1 - item_probs + 1e-10)
    
    log_lik_purchases = data @ log_item_probs.T          # (n_obs, n_classes)
    log_lik_non_purchases = (1 - data) @ log_1minus_item.T
    
    # Log of household-specific priors
    log_class_probs = np.log(class_probs_per_hh + 1e-10)  # (n_obs, n_classes)
    
    # Log joint: log P(class | Z_i) + log P(X_i | class)
    log_joint = log_class_probs + log_lik_purchases + log_lik_non_purchases
    
    # Normalize to get posteriors using log-sum-exp trick
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    log_joint_shifted = log_joint - max_log_joint
    exp_log_joint = np.exp(log_joint_shifted)
    sum_exp = exp_log_joint.sum(axis=1, keepdims=True)
    
    responsibilities = exp_log_joint / sum_exp
    
    if return_log_likelihood:
        log_marginal = max_log_joint.squeeze() + np.log(sum_exp.squeeze())
        log_likelihood = log_marginal.sum()
        return responsibilities, log_likelihood
    
    return responsibilities


def lca_m_step_beta(covariates: np.ndarray, responsibilities: np.ndarray,
                    beta: np.ndarray, learning_rate: float = 0.1,
                    n_steps: int = 10, l2_penalty: float = 0.1) -> np.ndarray:
    """
    M-step for regression coefficients: regularized multinomial logistic regression.
    
    Given soft class assignments (responsibilities), update the regression
    coefficients to maximize the expected complete-data log-likelihood with
    L2 regularization for stability.
    
    The update uses gradient ascent on the penalized objective:
        Q(β) = Σᵢ Σc rᵢc log P(c|Zᵢ,β) - (λ/2) ||β||²
    
    We use a few steps of gradient ascent rather than full optimization,
    since we'll iterate in the EM loop anyway. This is sometimes called
    "generalized EM" or partial M-step.
    
    Args:
        covariates: (n_households, n_features) household features
        responsibilities: (n_households, n_classes) current soft assignments
        beta: (n_features, n_classes) current regression coefficients
        learning_rate: Step size for gradient updates (default reduced for stability)
        n_steps: Number of gradient steps per M-step
        l2_penalty: L2 regularization strength (prevents coefficient explosion)
        
    Returns:
        Updated beta coefficients (n_features, n_classes)
    """
    n_obs = covariates.shape[0]
    n_classes = responsibilities.shape[1]
    beta = beta.copy()
    
    for _ in range(n_steps):
        # Current predicted class probabilities
        pred_probs = compute_class_probs_from_covariates(covariates, beta)
        
        # Gradient of log-likelihood: Z^T @ (responsibilities - predicted)
        # Normalized by n_obs for stability across different sample sizes
        gradient = covariates.T @ (responsibilities - pred_probs) / n_obs
        
        # L2 regularization gradient (don't penalize intercept, first row)
        reg_gradient = l2_penalty * beta.copy()
        reg_gradient[0, :] = 0  # Don't regularize intercept
        
        # Update (keeping last class as reference)
        beta[:, :-1] += learning_rate * (gradient[:, :-1] - reg_gradient[:, :-1])
    
    return beta


def lca_m_step_with_covariates(data: np.ndarray, covariates: np.ndarray,
                                responsibilities: np.ndarray,
                                beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full M-step for LCA with covariates: update both beta and item_probs.
    
    Args:
        data: (n_households, n_items) binary purchase matrix
        covariates: (n_households, n_features) household features
        responsibilities: (n_households, n_classes) current soft assignments
        beta: (n_features, n_classes) current regression coefficients
        
    Returns:
        Tuple of (new_beta, new_item_probs)
    """
    # Update regression coefficients
    new_beta = lca_m_step_beta(covariates, responsibilities, beta)
    
    # Update item probabilities (same as standard LCA M-step)
    class_counts = responsibilities.sum(axis=0)
    new_item_probs = (responsibilities.T @ data) / (class_counts[:, np.newaxis] + 1e-10)
    new_item_probs = np.clip(new_item_probs, 0.01, 0.99)
    
    return new_beta, new_item_probs


def fit_lca_with_covariates(data: np.ndarray, covariates: np.ndarray,
                             n_classes: int, max_iter: int = 100,
                             tol: float = 1e-4, n_init: int = 10,
                             seed: int = 42) -> Dict:
    """
    Fit Latent Class Analysis with household covariates.
    
    This extends standard LCA by allowing household features to influence
    class membership probabilities. Instead of fixed class priors, each
    household has its own prior distribution over classes:
    
        P(class = c | Z_i) = softmax(Z_i @ beta_c)
    
    This enables the model to:
    - Explain why certain households belong to certain segments
    - Improve classification accuracy using auxiliary information
    - Profile segments by their demographic characteristics
    
    Args:
        data: (n_households, n_items) binary purchase matrix (0/1 values)
        covariates: (n_households, n_features) household features.
                   Should be standardized for numeric features.
                   An intercept column is added automatically.
        n_classes: Number of latent classes to fit
        max_iter: Maximum EM iterations per initialization
        tol: Convergence tolerance on log-likelihood improvement
        n_init: Number of random initializations to try
        seed: Base random seed
        
    Returns:
        Dictionary with:
        - beta: (n_features+1, n_classes) regression coefficients (with intercept)
        - item_probs: (n_classes, n_items) purchase probabilities per class
        - responsibilities: (n_households, n_classes) posterior memberships
        - class_probs_per_hh: (n_households, n_classes) prior probs per household
        - log_likelihood: Final log-likelihood
        - bic: Bayesian Information Criterion
        - aic: Akaike Information Criterion
        - n_iter: Number of iterations to convergence
        - n_classes: Number of classes
        - feature_names: Will be None; caller should track these
    """
    n_obs, n_items = data.shape
    
    # Add intercept column to covariates
    covariates_with_intercept = np.column_stack([np.ones(n_obs), covariates])
    n_features = covariates_with_intercept.shape[1]
    
    best_ll = -np.inf
    best_result = None
    
    for init in range(n_init):
        # Initialize parameters
        beta, item_probs = initialize_lca_with_covariates(
            n_classes, n_items, n_features, seed=seed + init
        )
        
        prev_ll = -np.inf
        n_iter = 0
        
        # EM iterations
        for iteration in range(max_iter):
            # E-step with household-specific priors
            responsibilities, ll = lca_e_step_with_covariates(
                data, covariates_with_intercept, beta, item_probs,
                return_log_likelihood=True
            )
            
            # M-step: update both beta and item_probs
            beta, item_probs = lca_m_step_with_covariates(
                data, covariates_with_intercept, responsibilities, beta
            )
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                n_iter = iteration + 1
                break
            
            prev_ll = ll
        else:
            n_iter = max_iter
        
        if ll > best_ll:
            best_ll = ll
            # Compute final class probabilities per household
            final_class_probs = compute_class_probs_from_covariates(
                covariates_with_intercept, beta
            )
            best_result = {
                'beta': beta.copy(),
                'item_probs': item_probs.copy(),
                'responsibilities': responsibilities.copy(),
                'class_probs_per_hh': final_class_probs,
                'log_likelihood': ll,
                'n_iter': n_iter
            }
    
    # Number of parameters:
    # - Beta: (n_features) * (n_classes - 1) due to reference class
    # - Item probs: n_classes * n_items
    n_beta_params = n_features * (n_classes - 1)
    n_item_params = n_classes * n_items
    n_params = n_beta_params + n_item_params
    
    best_result['bic'] = -2 * best_result['log_likelihood'] + n_params * np.log(n_obs)
    best_result['aic'] = -2 * best_result['log_likelihood'] + 2 * n_params
    best_result['n_classes'] = n_classes
    best_result['n_features'] = n_features
    best_result['feature_names'] = None  # Caller should set this
    
    return best_result


def interpret_covariate_effects(beta: np.ndarray, 
                                 feature_names: list,
                                 class_names: list = None) -> Dict:
    """
    Interpret the regression coefficients from LCA with covariates.
    
    Converts raw coefficients to odds ratios and provides a summary
    that's easier to interpret. Positive coefficients mean that feature
    increases the odds of belonging to that class (vs. reference class).
    
    Args:
        beta: (n_features, n_classes) regression coefficients (with intercept)
        feature_names: List of feature names (including "Intercept")
        class_names: Optional list of class names. Defaults to "Class 1", etc.
        
    Returns:
        Dictionary with:
        - odds_ratios: DataFrame of exp(beta) for interpretation
        - coefficients: DataFrame of raw coefficients
        - summary: Text summary of key effects
    """
    import pandas as pd
    
    n_features, n_classes = beta.shape
    
    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(n_classes)]
    
    # Create DataFrames for coefficients and odds ratios
    coef_df = pd.DataFrame(
        beta,
        index=feature_names,
        columns=class_names
    )
    
    # Odds ratios (exponentiated coefficients)
    # For a 1-unit increase in the feature, odds of class c vs reference multiply by exp(beta)
    odds_ratio_df = pd.DataFrame(
        np.exp(beta),
        index=feature_names,
        columns=class_names
    )
    
    # Find the most influential features for each class
    summary_lines = []
    for c in range(n_classes - 1):  # Skip reference class
        class_name = class_names[c]
        # Get non-intercept coefficients for this class
        class_coefs = beta[1:, c]  # Skip intercept
        feature_subset = feature_names[1:]  # Skip "Intercept"
        
        # Find strongest positive and negative effects
        if len(class_coefs) > 0:
            max_idx = np.argmax(class_coefs)
            min_idx = np.argmin(class_coefs)
            
            if class_coefs[max_idx] > 0.1:
                summary_lines.append(
                    f"{class_name}: Higher '{feature_subset[max_idx]}' increases membership "
                    f"(OR={np.exp(class_coefs[max_idx]):.2f})"
                )
            if class_coefs[min_idx] < -0.1:
                summary_lines.append(
                    f"{class_name}: Higher '{feature_subset[min_idx]}' decreases membership "
                    f"(OR={np.exp(class_coefs[min_idx]):.2f})"
                )
    
    return {
        'coefficients': coef_df,
        'odds_ratios': odds_ratio_df,
        'summary': "\n".join(summary_lines) if summary_lines else "No strong covariate effects detected."
    }