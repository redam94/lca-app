"""
Latent Dirichlet Allocation (LDA) for Purchase Data.

LDA is a generative probabilistic model that discovers latent "topics"
in a collection of documents. When applied to binary purchase data:
- Each household is treated as a "document"
- Each product purchase is treated as a "word"
- Topics represent latent purchase patterns or "shopping themes"

The model decomposes the purchase matrix into:
- Topic-product distributions: What products define each topic
- Household-topic distributions: What topics characterize each household

This is useful for market structure analysis because:
1. Topics can reveal natural product groupings based on co-purchase behavior
2. Household topic assignments provide soft clustering/segmentation
3. Unlike hard clustering, households can belong to multiple topics
"""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from typing import Dict


def fit_lda(data: np.ndarray, n_topics: int, max_iter: int = 100,
            learning_method: str = 'online', random_state: int = 42) -> Dict:
    """
    Fit Latent Dirichlet Allocation model to binary purchase data.

    LDA treats purchase data as a bag-of-words model where:
    - Households are documents
    - Products are vocabulary terms
    - Purchase indicators are word counts (0 or 1)

    Args:
        data: (n_households, n_products) binary purchase matrix
        n_topics: Number of latent topics to discover
        max_iter: Maximum iterations for the optimization
        learning_method: 'online' for mini-batch updates (faster),
                        'batch' for full-batch updates (more stable)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
        - topic_product_dist: (n_topics, n_products) topic-product distributions
          Each row sums to 1, showing probability of each product given topic
        - household_topic_dist: (n_households, n_topics) document-topic distributions
          Each row sums to 1, showing probability of each topic for household
        - loadings: topic_product_dist.T for biplot compatibility
        - scores: household_topic_dist for household embeddings
        - var_explained_pct: Approximate variance explained by each topic
        - perplexity: Model perplexity (lower is better fit)
        - log_likelihood: Log-likelihood of the data
        - n_iter: Number of iterations run
        - n_topics: Number of topics fit
    """
    # Ensure data is non-negative (required for LDA)
    X = np.maximum(data, 0)

    # Initialize and fit LDA model
    model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method=learning_method,
        random_state=random_state,
        doc_topic_prior=None,  # Use default (1/n_topics)
        topic_word_prior=None,  # Use default (1/n_topics)
        learning_offset=10.0,  # Helps early iterations be more stable
        n_jobs=-1,  # Use all cores for E-step
    )

    # Fit and transform: get household-topic distributions
    household_topic_dist = model.fit_transform(X)

    # Get topic-product distributions (components_ is already normalized)
    topic_product_dist = model.components_

    # Normalize to proper probability distributions
    # topic_product_dist rows should sum to 1 (probability of product given topic)
    topic_product_dist_norm = topic_product_dist / topic_product_dist.sum(axis=1, keepdims=True)

    # household_topic_dist is already normalized from fit_transform

    # Compute approximate variance explained by each topic
    # We use the proportion of total "mass" captured by each topic
    topic_weights = household_topic_dist.sum(axis=0)
    var_explained_pct = (topic_weights / topic_weights.sum()) * 100

    # Sort topics by importance (variance explained)
    sort_idx = np.argsort(var_explained_pct)[::-1]
    topic_product_dist_norm = topic_product_dist_norm[sort_idx]
    household_topic_dist = household_topic_dist[:, sort_idx]
    var_explained_pct = var_explained_pct[sort_idx]

    # Compute perplexity (measure of how well model predicts held-out data)
    perplexity = model.perplexity(X)

    # Compute log-likelihood
    log_likelihood = model.score(X)

    return {
        'topic_product_dist': topic_product_dist_norm,  # (n_topics, n_products)
        'household_topic_dist': household_topic_dist,   # (n_households, n_topics)
        'loadings': topic_product_dist_norm.T,          # (n_products, n_topics) for biplot
        'scores': household_topic_dist,                  # (n_households, n_topics) for biplot
        'var_explained_pct': var_explained_pct,
        'perplexity': perplexity,
        'log_likelihood': log_likelihood,
        'n_iter': model.n_iter_,
        'n_topics': n_topics,
    }


def get_top_products_per_topic(topic_product_dist: np.ndarray,
                                product_names: list,
                                n_top: int = 10) -> Dict[int, list]:
    """
    Get the top products for each topic.

    Args:
        topic_product_dist: (n_topics, n_products) topic-product distribution matrix
        product_names: List of product names
        n_top: Number of top products to return per topic

    Returns:
        Dictionary mapping topic index to list of (product_name, probability) tuples
    """
    n_topics = topic_product_dist.shape[0]
    top_products = {}

    for topic_idx in range(n_topics):
        # Get indices of top products for this topic
        top_indices = np.argsort(topic_product_dist[topic_idx])[::-1][:n_top]

        # Get product names and probabilities
        top_products[topic_idx] = [
            (product_names[idx], float(topic_product_dist[topic_idx, idx]))
            for idx in top_indices
        ]

    return top_products


def compute_topic_similarity(topic_product_dist: np.ndarray) -> np.ndarray:
    """
    Compute similarity between topics based on their product distributions.

    Uses Jensen-Shannon divergence (symmetric, bounded) converted to similarity.

    Args:
        topic_product_dist: (n_topics, n_products) topic-product distribution matrix

    Returns:
        (n_topics, n_topics) similarity matrix with values in [0, 1]
    """
    from scipy.spatial.distance import jensenshannon

    n_topics = topic_product_dist.shape[0]
    similarity = np.zeros((n_topics, n_topics))

    for i in range(n_topics):
        for j in range(n_topics):
            if i == j:
                similarity[i, j] = 1.0
            else:
                # JS divergence is in [0, 1] for probability distributions
                js_div = jensenshannon(topic_product_dist[i], topic_product_dist[j])
                # Convert to similarity (1 - divergence)
                similarity[i, j] = 1.0 - js_div

    return similarity
