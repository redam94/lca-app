"""
Network Analysis for Purchase Data.

This module builds a product co-purchase network from binary purchase data
and performs network analysis to discover market structure. The approach:

1. Construct a weighted graph where:
   - Nodes are products
   - Edge weights represent co-purchase strength (correlation, lift, etc.)

2. Analyze the network using:
   - Community detection to find natural product groupings
   - Centrality measures to identify key products
   - Network metrics to characterize market structure

Network analysis complements latent variable models by providing:
- Interpretable visualization of product relationships
- No assumption about number of dimensions/factors
- Focus on pairwise relationships rather than latent factors
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy import sparse


def _compute_copurchase_matrix(data: np.ndarray, method: str = 'lift') -> np.ndarray:
    """
    Compute co-purchase strength between all product pairs.

    Args:
        data: (n_households, n_products) binary purchase matrix
        method: How to measure co-purchase strength:
            - 'correlation': Pearson correlation
            - 'cosine': Cosine similarity
            - 'lift': P(A,B) / (P(A) * P(B)) - 1
            - 'jaccard': |A ∩ B| / |A ∪ B|

    Returns:
        (n_products, n_products) symmetric matrix of co-purchase strengths
    """
    n_households, n_products = data.shape

    if method == 'correlation':
        # Pearson correlation
        copurchase = np.corrcoef(data.T)
        # Replace NaN (from zero-variance columns) with 0
        copurchase = np.nan_to_num(copurchase, nan=0.0)

    elif method == 'cosine':
        # Cosine similarity
        norms = np.linalg.norm(data, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        data_normalized = data / norms
        copurchase = data_normalized.T @ data_normalized

    elif method == 'lift':
        # Lift: P(A,B) / (P(A) * P(B)) - 1
        # Values > 0 indicate positive association
        probs = data.mean(axis=0)  # P(product)
        joint_probs = (data.T @ data) / n_households  # P(A and B)

        # Expected under independence
        expected = np.outer(probs, probs)
        expected = np.maximum(expected, 1e-10)

        # Lift - 1 (so 0 = independent, >0 = positive association)
        copurchase = (joint_probs / expected) - 1

        # Set diagonal to 0 (self-lift is not meaningful)
        np.fill_diagonal(copurchase, 0)

    elif method == 'jaccard':
        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        # Intersection: both bought
        intersection = data.T @ data

        # Union: at least one bought
        # |A| + |B| - |A ∩ B|
        sums = data.sum(axis=0)
        union = np.add.outer(sums, sums) - intersection
        union = np.maximum(union, 1e-10)

        copurchase = intersection / union
        np.fill_diagonal(copurchase, 0)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'correlation', 'cosine', 'lift', or 'jaccard'")

    return copurchase


def fit_network_analysis(data: np.ndarray, threshold: float = 0.1,
                         community_method: str = 'louvain',
                         edge_method: str = 'lift',
                         random_state: int = 42) -> Dict:
    """
    Build and analyze a product co-purchase network.

    Constructs a graph where products are nodes and edges represent
    co-purchase relationships. Performs community detection and
    computes centrality measures.

    Args:
        data: (n_households, n_products) binary purchase matrix
        threshold: Minimum edge weight to include (filters weak connections)
        community_method: Community detection algorithm:
            - 'louvain': Louvain method (modularity optimization)
            - 'label_propagation': Label propagation
            - 'greedy_modularity': Greedy modularity maximization
        edge_method: How to compute edge weights:
            - 'lift': Association strength (recommended)
            - 'correlation': Pearson correlation
            - 'cosine': Cosine similarity
            - 'jaccard': Jaccard similarity
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
        - adjacency_matrix: (n_products, n_products) weighted adjacency matrix
        - communities: List of community assignments for each product
        - n_communities: Number of detected communities
        - centrality_scores: Eigenvector centrality for each product
        - degree_centrality: Degree centrality for each product
        - betweenness_centrality: Betweenness centrality for each product
        - loadings: Community membership one-hot encoding for biplot
        - scores: Household profiles based on community purchases
        - graph_metrics: Network statistics (density, modularity, etc.)
        - edge_list: List of (product_i, product_j, weight) tuples
    """
    try:
        import networkx as nx
        from networkx.algorithms import community as nx_community
    except ImportError:
        raise ImportError(
            "networkx is required for network analysis. "
            "Install it with: pip install networkx"
        )

    n_households, n_products = data.shape

    # Compute co-purchase matrix
    copurchase = _compute_copurchase_matrix(data, method=edge_method)

    # Apply threshold to create adjacency matrix
    adjacency = copurchase.copy()
    adjacency[adjacency < threshold] = 0

    # Build NetworkX graph
    G = nx.from_numpy_array(adjacency)

    # Remove self-loops and isolated nodes for analysis
    G.remove_edges_from(nx.selfloop_edges(G))

    # Community detection
    if community_method == 'louvain':
        communities_gen = nx_community.louvain_communities(
            G, weight='weight', seed=random_state
        )
        communities_sets = list(communities_gen)
    elif community_method == 'label_propagation':
        communities_gen = nx_community.label_propagation_communities(G)
        communities_sets = list(communities_gen)
    elif community_method == 'greedy_modularity':
        communities_gen = nx_community.greedy_modularity_communities(
            G, weight='weight'
        )
        communities_sets = list(communities_gen)
    else:
        raise ValueError(
            f"Unknown community method: {community_method}. "
            "Use 'louvain', 'label_propagation', or 'greedy_modularity'"
        )

    # Convert community sets to per-node assignments
    community_labels = np.zeros(n_products, dtype=int)
    for comm_idx, comm_set in enumerate(communities_sets):
        for node in comm_set:
            community_labels[node] = comm_idx

    n_communities = len(communities_sets)

    # Compute centrality measures
    if G.number_of_edges() > 0:
        # eigenvector_centrality_numpy doesn't give consistent results for
        # disconnected graphs, so compute per connected component and combine
        if nx.is_connected(G):
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
        else:
            eigenvector_centrality = {i: 0.0 for i in range(n_products)}
            for component in nx.connected_components(G):
                if len(component) < 2:
                    continue
                subgraph = G.subgraph(component)
                try:
                    sub_centrality = nx.eigenvector_centrality_numpy(subgraph, weight='weight')
                    # Scale by component size relative to graph size so larger
                    # components contribute more
                    scale = len(component) / n_products
                    for node, score in sub_centrality.items():
                        eigenvector_centrality[node] = score * scale
                except Exception:
                    # Fallback: use degree centrality for this component
                    sub_degree = nx.degree_centrality(subgraph)
                    scale = len(component) / n_products
                    for node, score in sub_degree.items():
                        eigenvector_centrality[node] = score * scale
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    else:
        # Empty graph - all centralities are 0
        eigenvector_centrality = {i: 0.0 for i in range(n_products)}
        degree_centrality = {i: 0.0 for i in range(n_products)}
        betweenness_centrality = {i: 0.0 for i in range(n_products)}

    # Convert centralities to arrays
    centrality_scores = np.array([eigenvector_centrality.get(i, 0.0) for i in range(n_products)])
    degree_scores = np.array([degree_centrality.get(i, 0.0) for i in range(n_products)])
    betweenness_scores = np.array([betweenness_centrality.get(i, 0.0) for i in range(n_products)])

    # Create community membership matrix (one-hot) for biplot compatibility
    loadings = np.zeros((n_products, n_communities))
    for i, comm in enumerate(community_labels):
        loadings[i, comm] = 1.0

    # Weight by centrality for more informative loadings
    loadings = loadings * centrality_scores[:, np.newaxis]

    # Compute household scores based on community purchases
    # Score = weighted average of purchases in each community
    scores = np.zeros((n_households, n_communities))
    for comm_idx in range(n_communities):
        comm_mask = community_labels == comm_idx
        if comm_mask.sum() > 0:
            # Average purchases in this community, weighted by centrality
            weights = centrality_scores[comm_mask]
            weights = weights / (weights.sum() + 1e-10)
            scores[:, comm_idx] = data[:, comm_mask] @ weights

    # Compute graph metrics
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        density = nx.density(G)
        try:
            modularity = nx_community.modularity(G, communities_sets, weight='weight')
        except:
            modularity = 0.0
        avg_clustering = nx.average_clustering(G, weight='weight')

        # Check if graph is connected
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
    else:
        density = 0.0
        modularity = 0.0
        avg_clustering = 0.0
        avg_path_length = 0.0

    graph_metrics = {
        'n_nodes': n_products,
        'n_edges': G.number_of_edges(),
        'density': density,
        'modularity': modularity,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length,
        'n_communities': n_communities,
    }

    # Create edge list for visualization
    edge_list = [
        (int(u), int(v), float(d['weight']))
        for u, v, d in G.edges(data=True)
        if d['weight'] > 0
    ]

    # Compute variance explained approximation
    # Use proportion of total edge weight in each community
    community_weights = np.zeros(n_communities)
    for comm_idx in range(n_communities):
        comm_mask = community_labels == comm_idx
        # Sum of edge weights within community
        comm_adj = adjacency[np.ix_(comm_mask, comm_mask)]
        community_weights[comm_idx] = comm_adj.sum()

    total_weight = adjacency.sum()
    if total_weight > 0:
        var_explained_pct = (community_weights / total_weight) * 100
    else:
        var_explained_pct = np.ones(n_communities) * (100 / n_communities)

    return {
        'adjacency_matrix': adjacency,
        'communities': community_labels.tolist(),
        'n_communities': n_communities,
        'centrality_scores': centrality_scores,
        'degree_centrality': degree_scores,
        'betweenness_centrality': betweenness_scores,
        'loadings': loadings,
        'scores': scores,
        'var_explained_pct': var_explained_pct,
        'graph_metrics': graph_metrics,
        'edge_list': edge_list,
        'threshold': threshold,
        'edge_method': edge_method,
        'community_method': community_method,
    }


def get_community_members(communities: List[int],
                          product_names: List[str]) -> Dict[int, List[str]]:
    """
    Get product names for each community.

    Args:
        communities: List of community assignments for each product
        product_names: List of product names

    Returns:
        Dictionary mapping community index to list of product names
    """
    community_members = {}
    for prod_idx, comm_idx in enumerate(communities):
        if comm_idx not in community_members:
            community_members[comm_idx] = []
        community_members[comm_idx].append(product_names[prod_idx])

    return community_members


def get_top_central_products(centrality_scores: np.ndarray,
                              product_names: List[str],
                              n_top: int = 10) -> List[Tuple[str, float]]:
    """
    Get the most central (important) products in the network.

    Args:
        centrality_scores: Centrality score for each product
        product_names: List of product names
        n_top: Number of top products to return

    Returns:
        List of (product_name, centrality_score) tuples, sorted by centrality
    """
    top_indices = np.argsort(centrality_scores)[::-1][:n_top]
    return [
        (product_names[idx], float(centrality_scores[idx]))
        for idx in top_indices
    ]
