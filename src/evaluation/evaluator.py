"""
Recommendation system evaluator using real MovieLens ratings.

Workflow
--------
1.  Load MovieLens ratings and build a user×movie matrix aligned to the
    TMDB movie DataFrame.
2.  Split into train / test per user (leave-k-out).
3.  Fit each recommender on **train** ratings.
4.  For every test user, generate top-K recs and measure against held-out
    items using Precision@K, Recall@K, NDCG@K, MAP@K, Coverage, Novelty,
    and Diversity.
"""
# @author 成员 E — 可解释性 & 评估系统

import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from src.utils.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_ndcg,
    calculate_map,
    calculate_coverage,
    calculate_novelty,
    calculate_diversity,
)

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """Evaluate recommendation quality with real user ratings."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        rating_matrix: np.ndarray,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        movies_df : pd.DataFrame
            Preprocessed TMDB movie dataframe.
        rating_matrix : np.ndarray
            Full user × movie rating matrix (from MovieLensLoader).
        test_ratio : float
            Fraction of each user's ratings to hold out for testing.
        seed : int
            Random seed.
        """
        from src.data.movielens_loader import MovieLensLoader

        self.movies_df = movies_df
        self.full_matrix = rating_matrix
        self.n_users, self.n_movies = rating_matrix.shape

        # Split
        self.train_matrix, self.test_matrix = MovieLensLoader.train_test_split(
            rating_matrix, test_ratio=test_ratio, seed=seed
        )

        # Precompute item popularity on training data (for novelty)
        self._item_pop = (self.train_matrix > 0).sum(axis=0)  # shape (n_movies,)
        self._total_interactions = int(self._item_pop.sum())

    # ------------------------------------------------------------------
    # Per-user evaluation helpers
    # ------------------------------------------------------------------
    def _user_ground_truth(self, user_idx: int, min_rating: float = 4.0):
        """Return set of movie indices the user rated >= min_rating in test."""
        row = self.test_matrix[user_idx]
        return set(np.where(row >= min_rating)[0])

    def _user_top_k_from_similarity(
        self,
        similarity_matrix: np.ndarray,
        user_idx: int,
        k: int,
    ) -> List[int]:
        """
        For a user, aggregate similarity-based scores weighted by their
        *training* ratings, then return the top-k unseen items.
        """
        user_ratings = self.train_matrix[user_idx]
        rated_mask = user_ratings > 0

        # Weighted sum: for each candidate movie, sum similarity to
        # movies the user has rated, weighted by the rating.
        scores = similarity_matrix.dot(user_ratings * rated_mask)

        # Zero out already-rated movies
        scores[rated_mask] = -1

        top_k = np.argsort(scores)[::-1][:k]
        return top_k.tolist()

    # ------------------------------------------------------------------
    # Evaluate a single recommender (given its similarity matrix)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        similarity_matrix: np.ndarray,
        method_name: str = "unknown",
        k: int = 10,
        max_eval_users: int = 500,
        min_test_items: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate a recommender that exposes a movie × movie similarity
        matrix (content, metadata, cf, hybrid).

        Parameters
        ----------
        similarity_matrix : np.ndarray  (n_movies, n_movies)
        method_name : str
        k : int   Top-K for metrics.
        max_eval_users : int  Cap to speed up evaluation.
        min_test_items : int  Skip users with fewer test items.

        Returns
        -------
        dict with aggregated metrics.
        """
        logger.info(f"Evaluating '{method_name}' @ K={k} …")

        # Find users that have test items
        eval_users = []
        for u in range(self.n_users):
            gt = self._user_ground_truth(u)
            if len(gt) >= min_test_items:
                eval_users.append(u)
            if len(eval_users) >= max_eval_users:
                break

        if not eval_users:
            logger.warning("No users with enough test items.")
            return {"method": method_name, "error": "no eligible users"}

        prec_list, rec_list, ndcg_list, map_list = [], [], [], []
        all_recs: List[List[int]] = []
        item_pop_dict = {i: int(self._item_pop[i]) for i in range(self.n_movies)}

        for u in eval_users:
            gt = self._user_ground_truth(u)
            recs = self._user_top_k_from_similarity(similarity_matrix, u, k)

            prec_list.append(calculate_precision(recs, gt, k))
            rec_list.append(calculate_recall(recs, gt, k))
            ndcg_list.append(calculate_ndcg(recs, gt, k=k))
            map_list.append(calculate_map(recs, gt, k=k))
            all_recs.append(recs)

        # Aggregated beyond-accuracy metrics
        coverage = calculate_coverage(all_recs, self.n_movies)
        novelty_scores = []
        for recs in all_recs:
            novelty_scores.append(
                calculate_novelty(recs, item_pop_dict, self._total_interactions)
            )

        # Diversity (sample – expensive for all users)
        def _sim(a, b):
            return float(similarity_matrix[a, b])

        diversity_scores = []
        for recs in all_recs[:100]:
            diversity_scores.append(calculate_diversity(recs, _sim))

        results = {
            "method": method_name,
            "K": k,
            "n_eval_users": len(eval_users),
            "precision@k": float(np.mean(prec_list)),
            "recall@k": float(np.mean(rec_list)),
            "ndcg@k": float(np.mean(ndcg_list)),
            "map@k": float(np.mean(map_list)),
            "coverage": float(coverage),
            "novelty": float(np.mean(novelty_scores)),
            "diversity": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        }

        logger.info(
            f"  {method_name}: P@{k}={results['precision@k']:.4f}  "
            f"R@{k}={results['recall@k']:.4f}  NDCG@{k}={results['ndcg@k']:.4f}  "
            f"Coverage={results['coverage']:.2%}"
        )
        return results

    # ------------------------------------------------------------------
    # Convenience: evaluate all methods of a HybridRecommender
    # ------------------------------------------------------------------
    def evaluate_all_methods(
        self,
        hybrid_recommender,
        k: int = 10,
        max_eval_users: int = 500,
    ) -> pd.DataFrame:
        """
        Evaluate content, metadata, CF, and hybrid recommenders.

        Parameters
        ----------
        hybrid_recommender : HybridRecommender
            Fitted hybrid recommender (exposes sub-recommenders).
        k : int
        max_eval_users : int

        Returns
        -------
        pd.DataFrame  one row per method.
        """
        methods = {
            "Content-Based": hybrid_recommender.content_recommender.similarity_matrix,
            "Metadata-Based": hybrid_recommender.metadata_recommender.similarity_matrix,
            "Collaborative": hybrid_recommender.cf_recommender.similarity_matrix,
            "Hybrid": hybrid_recommender.similarity_matrix,
        }

        rows = []
        for name, sim in methods.items():
            if sim is not None:
                rows.append(self.evaluate(sim, name, k=k, max_eval_users=max_eval_users))
            else:
                rows.append({"method": name, "error": "similarity matrix is None"})

        return pd.DataFrame(rows)
