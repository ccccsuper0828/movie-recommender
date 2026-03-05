"""
Similarity calculation utilities for the Movie Recommendation System.
"""
# @author 成员 B — 基础推荐算法 & 工具库

import numpy as np
from typing import List, Set, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from scipy.sparse import issparse


def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        First vector
    vec2 : np.ndarray
        Second vector

    Returns
    -------
    float
        Cosine similarity score (0 to 1)
    """
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def jaccard_sim(set1: Union[Set, List], set2: Union[Set, List]) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Parameters
    ----------
    set1 : Set or List
        First set of items
    set2 : Set or List
        Second set of items

    Returns
    -------
    float
        Jaccard similarity score (0 to 1)
    """
    set1 = set(set1) if not isinstance(set1, set) else set1
    set2 = set(set2) if not isinstance(set2, set) else set2

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def pearson_correlation(ratings1: np.ndarray, ratings2: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two rating arrays.

    Parameters
    ----------
    ratings1 : np.ndarray
        First rating array
    ratings2 : np.ndarray
        Second rating array

    Returns
    -------
    float
        Pearson correlation coefficient (-1 to 1)
    """
    ratings1 = np.asarray(ratings1)
    ratings2 = np.asarray(ratings2)

    # Find common non-zero ratings
    mask = (ratings1 != 0) & (ratings2 != 0)

    if mask.sum() < 2:
        return 0.0

    r1 = ratings1[mask]
    r2 = ratings2[mask]

    mean1 = np.mean(r1)
    mean2 = np.mean(r2)

    numerator = np.sum((r1 - mean1) * (r2 - mean2))
    denominator = np.sqrt(np.sum((r1 - mean1) ** 2) * np.sum((r2 - mean2) ** 2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def compute_similarity_matrix(
    matrix: np.ndarray,
    method: str = 'cosine'
) -> np.ndarray:
    """
    Compute pairwise similarity matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix where rows are items
    method : str
        Similarity method ('cosine', 'pearson')

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix
    """
    n = matrix.shape[0]

    if method == 'cosine':
        if issparse(matrix):
            return sklearn_cosine(matrix)
        return sklearn_cosine(matrix)

    elif method == 'pearson':
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim = pearson_correlation(matrix[i], matrix[j])
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        return sim_matrix

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def get_top_similar(
    similarity_scores: np.ndarray,
    idx: int,
    top_n: int = 10,
    exclude_self: bool = True
) -> List[tuple]:
    """
    Get top N most similar items.

    Parameters
    ----------
    similarity_scores : np.ndarray
        Row of similarity scores (or full matrix)
    idx : int
        Index of the query item
    top_n : int
        Number of results to return
    exclude_self : bool
        Whether to exclude the query item from results

    Returns
    -------
    List[tuple]
        List of (index, score) tuples sorted by similarity
    """
    if similarity_scores.ndim == 2:
        scores = similarity_scores[idx]
    else:
        scores = similarity_scores

    # Create (index, score) pairs
    score_pairs = list(enumerate(scores))

    # Sort by score descending
    score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Filter and return
    results = []
    for item_idx, score in score_pairs:
        if exclude_self and item_idx == idx:
            continue
        results.append((item_idx, float(score)))
        if len(results) >= top_n:
            break

    return results


def weighted_average_similarity(
    scores_dict: dict,
    weights: dict
) -> float:
    """
    Calculate weighted average of multiple similarity scores.

    Parameters
    ----------
    scores_dict : dict
        Dictionary of {method_name: score}
    weights : dict
        Dictionary of {method_name: weight}

    Returns
    -------
    float
        Weighted average similarity
    """
    total_weight = 0
    weighted_sum = 0

    for method, score in scores_dict.items():
        weight = weights.get(method, 0)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight
