"""
KNN + SVD Ensemble Recommender.

Reference: Movie-Analysis project §5.3 — "用户 KNN + 奇异值分解"
  1. User-KNN: find k most similar users via cosine on rating vectors.
  2. Candidate pool: movies those neighbours rated but the target user hasn't.
  3. SVD re-ranking: predict the target user's rating for every candidate
     using truncated SVD on the training rating matrix, pick top-n.
"""
# @author 成员 C — 进阶推荐算法 & 推荐页面

import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class KNNSVDEnsembleRecommender(BaseRecommender):
    """
    Personalised recommender that combines User-KNN candidate generation
    with SVD score prediction.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        n_factors: int = 50,
        random_seed: int = 42,
    ):
        super().__init__(name="KNN_SVD_Ensemble")
        self.n_neighbors = n_neighbors
        self.n_factors = n_factors
        self.random_seed = random_seed

        self._rating_matrix: Optional[np.ndarray] = None
        self._user_knn: Optional[NearestNeighbors] = None
        self._user_sim: Optional[np.ndarray] = None

        # SVD components
        self._svd_predictions: Optional[np.ndarray] = None
        self._user_ratings_mean: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        movies_df: pd.DataFrame,
        rating_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "KNNSVDEnsembleRecommender":
        """
        Parameters
        ----------
        movies_df : pd.DataFrame
        rating_matrix : np.ndarray  shape (n_users, n_movies)
            If not supplied, attempts to load from MovieLens.
        """
        logger.info("Fitting KNN_SVD_Ensemble …")
        self._create_index_mappings(movies_df)

        if rating_matrix is not None:
            self._rating_matrix = rating_matrix
        else:
            self._rating_matrix = self._load_ratings(movies_df)

        # --- User-KNN ---
        logger.info("  Building User-KNN …")
        sparse = csr_matrix(self._rating_matrix)
        self._user_knn = NearestNeighbors(
            metric="cosine", algorithm="brute",
            n_neighbors=min(self.n_neighbors + 1, self._rating_matrix.shape[0]),
        )
        self._user_knn.fit(sparse)
        self._user_sim = cosine_similarity(self._rating_matrix)

        # --- SVD ---
        logger.info("  Building SVD predictions …")
        self._user_ratings_mean = np.mean(self._rating_matrix, axis=1)
        centred = self._rating_matrix - self._user_ratings_mean.reshape(-1, 1)
        k = min(self.n_factors, min(self._rating_matrix.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(centred), k=k)
        self._svd_predictions = (
            np.dot(np.dot(U, np.diag(sigma)), Vt)
            + self._user_ratings_mean.reshape(-1, 1)
        )

        # Build an item-level similarity from SVD item factors (for base class)
        item_factors = Vt.T  # (n_movies, k)
        self._similarity_matrix = cosine_similarity(item_factors)

        self._is_fitted = True
        logger.info(
            f"KNN_SVD_Ensemble fitted: {self._rating_matrix.shape[0]} users, "
            f"{self._rating_matrix.shape[1]} movies, k_svd={k}"
        )
        return self

    # ------------------------------------------------------------------
    def _load_ratings(self, movies_df: pd.DataFrame) -> np.ndarray:
        try:
            from src.data.movielens_loader import MovieLensLoader
            ml = MovieLensLoader()
            if ml.ratings_path.exists():
                matrix, _, _ = ml.build_rating_matrix(movies_df, min_ratings_per_user=5)
                return matrix
        except Exception as e:
            logger.warning(f"MovieLens load failed: {e}")
        # Fallback: empty matrix (will degrade to SVD-only)
        return np.zeros((100, len(movies_df)), dtype=np.float32)

    # ------------------------------------------------------------------
    # Core: recommend for a specific USER
    # ------------------------------------------------------------------
    def recommend_for_user(
        self,
        user_idx: int,
        top_n: int = 10,
    ) -> Optional[pd.DataFrame]:
        """
        1. Find similar users via KNN.
        2. Collect their rated movies that the target user hasn't rated.
        3. Rank candidates by SVD-predicted rating.
        """
        if not self._validate_fitted():
            return None

        n_users, n_movies = self._rating_matrix.shape
        if user_idx >= n_users:
            return None

        # Step 1 — similar users
        dists, indices = self._user_knn.kneighbors(
            self._rating_matrix[user_idx].reshape(1, -1),
            n_neighbors=min(self.n_neighbors + 1, n_users),
        )
        neighbor_ids = indices.flatten()[1:]  # exclude self

        # Step 2 — candidate movies
        watched = set(np.where(self._rating_matrix[user_idx] > 0)[0])
        candidates = set()
        for n_id in neighbor_ids:
            candidates.update(np.where(self._rating_matrix[n_id] > 0)[0])
        candidates -= watched

        if not candidates:
            return None

        # Step 3 — SVD predicted scores
        candidates = list(candidates)
        scores = self._svd_predictions[user_idx, candidates]
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        selected = [candidates[i] for i in ranked_idx]
        selected_scores = scores[ranked_idx]

        # Format output
        cols = ["title", "genres_list", "vote_average", "popularity"]
        recs = self._format_recommendations(selected, selected_scores.tolist(), cols)
        recs.rename(columns={"similarity_score": "predicted_rating"}, inplace=True)
        return recs

    # ------------------------------------------------------------------
    # API-compatible: recommend by movie title (uses SVD item similarity)
    # ------------------------------------------------------------------
    def recommend(self, title: str, top_n: int = 10, **kwargs) -> Optional[pd.DataFrame]:
        """
        Title-based fallback: use SVD-derived item similarity to find
        movies most similar in the latent space.
        """
        if not self._validate_fitted():
            return None

        idx = self.get_movie_index(title)
        if idx is None:
            logger.warning(f"Movie not found: {title}")
            return None

        sim_scores = list(enumerate(self._similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: top_n + 1]

        movie_indices = [s[0] for s in sim_scores]
        scores = [s[1] for s in sim_scores]

        cols = ["title", "genres_list", "vote_average", "popularity"]
        return self._format_recommendations(movie_indices, scores, cols)
