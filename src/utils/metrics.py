"""
Evaluation metrics for the Movie Recommendation System.
"""
# @author 成员 B — 基础推荐算法 & 工具库

import numpy as np
from typing import List, Set, Union, Optional


def calculate_precision(
    recommended: List,
    relevant: Set,
    k: Optional[int] = None
) -> float:
    """
    Calculate Precision@K.

    Parameters
    ----------
    recommended : List
        List of recommended items
    relevant : Set
        Set of relevant (ground truth) items
    k : int, optional
        Number of top recommendations to consider

    Returns
    -------
    float
        Precision score (0 to 1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    relevant_set = set(relevant)
    hits = sum(1 for item in recommended if item in relevant_set)

    return hits / len(recommended)


def calculate_recall(
    recommended: List,
    relevant: Set,
    k: Optional[int] = None
) -> float:
    """
    Calculate Recall@K.

    Parameters
    ----------
    recommended : List
        List of recommended items
    relevant : Set
        Set of relevant (ground truth) items
    k : int, optional
        Number of top recommendations to consider

    Returns
    -------
    float
        Recall score (0 to 1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(relevant) == 0:
        return 0.0

    relevant_set = set(relevant)
    hits = sum(1 for item in recommended if item in relevant_set)

    return hits / len(relevant_set)


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.

    Parameters
    ----------
    precision : float
        Precision score
    recall : float
        Recall score

    Returns
    -------
    float
        F1 score (0 to 1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_ndcg(
    recommended: List,
    relevant: Set,
    relevance_scores: Optional[dict] = None,
    k: Optional[int] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Parameters
    ----------
    recommended : List
        List of recommended items (in order)
    relevant : Set
        Set of relevant items
    relevance_scores : dict, optional
        Dictionary of {item: relevance_score}. If None, binary relevance is used.
    k : int, optional
        Number of top recommendations to consider

    Returns
    -------
    float
        NDCG score (0 to 1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    relevant_set = set(relevant)

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant_set:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            # Position is 1-indexed for log calculation
            dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed

    # Calculate ideal DCG
    if relevance_scores:
        # Sort relevant items by relevance score
        sorted_relevant = sorted(
            [(item, relevance_scores.get(item, 1.0)) for item in relevant_set],
            key=lambda x: x[1],
            reverse=True
        )
        ideal_scores = [score for _, score in sorted_relevant]
    else:
        ideal_scores = [1.0] * len(relevant_set)

    # Limit to k items
    if k is not None:
        ideal_scores = ideal_scores[:k]

    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_map(
    recommended: List,
    relevant: Set,
    k: Optional[int] = None
) -> float:
    """
    Calculate Mean Average Precision (MAP@K).

    Parameters
    ----------
    recommended : List
        List of recommended items
    relevant : Set
        Set of relevant items
    k : int, optional
        Number of top recommendations to consider

    Returns
    -------
    float
        MAP score (0 to 1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(relevant) == 0:
        return 0.0

    relevant_set = set(relevant)
    precision_sum = 0.0
    hits = 0

    for i, item in enumerate(recommended):
        if item in relevant_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            precision_sum += precision_at_i

    if hits == 0:
        return 0.0

    return precision_sum / min(len(relevant_set), len(recommended))


def calculate_coverage(
    all_recommendations: List[List],
    catalog_size: int
) -> float:
    """
    Calculate catalog coverage.

    Parameters
    ----------
    all_recommendations : List[List]
        List of recommendation lists for all users
    catalog_size : int
        Total number of items in the catalog

    Returns
    -------
    float
        Coverage ratio (0 to 1)
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0


def calculate_novelty(
    recommendations: List,
    item_popularity: dict,
    total_interactions: int
) -> float:
    """
    Calculate novelty (inverse popularity) of recommendations.

    Parameters
    ----------
    recommendations : List
        List of recommended items
    item_popularity : dict
        Dictionary of {item: number_of_interactions}
    total_interactions : int
        Total number of interactions in the system

    Returns
    -------
    float
        Average novelty score (higher = more novel)
    """
    if len(recommendations) == 0:
        return 0.0

    novelty_sum = 0.0
    for item in recommendations:
        pop = item_popularity.get(item, 1)
        # Self-information: -log2(p(item))
        probability = pop / total_interactions if total_interactions > 0 else 0.001
        novelty_sum += -np.log2(max(probability, 1e-10))

    return novelty_sum / len(recommendations)


def calculate_diversity(
    recommendations: List,
    similarity_func,
) -> float:
    """
    Calculate intra-list diversity (1 - average similarity).

    Parameters
    ----------
    recommendations : List
        List of recommended items
    similarity_func : callable
        Function that takes two items and returns similarity

    Returns
    -------
    float
        Diversity score (0 to 1, higher = more diverse)
    """
    n = len(recommendations)
    if n < 2:
        return 1.0

    total_sim = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_sim += similarity_func(recommendations[i], recommendations[j])
            count += 1

    avg_sim = total_sim / count if count > 0 else 0.0
    return 1.0 - avg_sim
