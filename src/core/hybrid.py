"""
Hybrid recommendation combining multiple methods.
"""
# @author 成员 C — 进阶推荐算法 & 推荐页面

from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation import RecommenderEvaluator
import pandas as pd
import numpy as np
import logging

from .base_recommender import BaseRecommender
from .content_based import ContentBasedRecommender
from .metadata_based import MetadataBasedRecommender
from .collaborative import CollaborativeFilteringRecommender
from config.settings import get_settings

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender that combines content-based, metadata-based,
    and collaborative filtering methods with configurable weights.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.1, 0.1, 0.8),  # Favor CF based on performance
        content_params: Optional[Dict] = None,
        metadata_params: Optional[Dict] = None,
        cf_params: Optional[Dict] = None
    ):
        """
        Initialize the hybrid recommender.

        Parameters
        ----------
        weights : tuple
            Weights for (content, metadata, cf) methods
        content_params : dict, optional
            Parameters for content-based recommender
        metadata_params : dict, optional
            Parameters for metadata-based recommender
        cf_params : dict, optional
            Parameters for collaborative filtering recommender
        """
        super().__init__(name="HybridRecommender")

        settings = get_settings()
        self.weights = weights or settings.hybrid_weights

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = tuple(w / total_weight for w in self.weights)

        # Initialize component recommenders
        self._content_recommender = ContentBasedRecommender(**(content_params or {}))
        self._metadata_recommender = MetadataBasedRecommender(**(metadata_params or {}))
        self._cf_recommender = CollaborativeFilteringRecommender(**(cf_params or {}))

    def fit(self, movies_df: pd.DataFrame, **kwargs) -> 'HybridRecommender':
        """
        Fit all component recommenders.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe
        **kwargs
            Additional parameters

        Returns
        -------
        HybridRecommender
            Self for method chaining
        """
        logger.info("Fitting HybridRecommender...")
        logger.info(f"Weights: Content={self.weights[0]:.2f}, "
                    f"Metadata={self.weights[1]:.2f}, CF={self.weights[2]:.2f}")

        # Create index mappings
        self._create_index_mappings(movies_df)

        # Fit component recommenders
        logger.info("Fitting content-based recommender...")
        self._content_recommender.fit(movies_df)

        logger.info("Fitting metadata-based recommender...")
        self._metadata_recommender.fit(movies_df)

        logger.info("Fitting collaborative filtering recommender...")
        self._cf_recommender.fit(movies_df)

        # Compute hybrid similarity matrix
        self._compute_hybrid_similarity()

        self._is_fitted = True
        logger.info("HybridRecommender fitting complete.")

        return self

    def _compute_hybrid_similarity(self):
        """Compute the weighted hybrid similarity matrix with normalization."""
        logger.info("Computing hybrid similarity matrix...")

        n_movies = len(self._movies_df)
        self._similarity_matrix = np.zeros((n_movies, n_movies))

        def _normalize_similarity(sim_matrix: np.ndarray) -> np.ndarray:
            """Normalize similarity matrix to [0, 1] range."""
            sim_min, sim_max = sim_matrix.min(), sim_matrix.max()
            if sim_max > sim_min:
                return (sim_matrix - sim_min) / (sim_max - sim_min)
            else:
                return sim_matrix

        # Add weighted components with normalization
        if self._content_recommender.similarity_matrix is not None:
            content_sim = self._content_recommender.similarity_matrix
            content_sim_norm = _normalize_similarity(content_sim)
            self._similarity_matrix += self.weights[0] * content_sim_norm

        if self._metadata_recommender.similarity_matrix is not None:
            metadata_sim = self._metadata_recommender.similarity_matrix
            metadata_sim_norm = _normalize_similarity(metadata_sim)
            self._similarity_matrix += self.weights[1] * metadata_sim_norm

        if self._cf_recommender.similarity_matrix is not None:
            cf_sim = self._cf_recommender.similarity_matrix
            cf_sim_norm = _normalize_similarity(cf_sim)
            self._similarity_matrix += self.weights[2] * cf_sim_norm

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        return_component_scores: bool = False,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get hybrid recommendations.

        Parameters
        ----------
        title : str
            Movie title
        top_n : int
            Number of recommendations
        return_component_scores : bool
            Include individual method scores in output
        **kwargs
            Additional parameters

        Returns
        -------
        pd.DataFrame or None
            Recommendations or None if movie not found
        """
        if not self._validate_fitted():
            return None

        idx = self.get_movie_index(title)
        if idx is None:
            logger.warning(f"Movie not found: {title}")
            matches = self.find_similar_titles(title)
            if matches:
                logger.info(f"Similar titles: {matches}")
            return None

        # Get similarity scores
        hybrid_scores = self._similarity_matrix[idx]

        sim_scores = list(enumerate(hybrid_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        # Format recommendations
        columns = ['title', 'genres_list', 'director', 'vote_average', 'popularity']
        recommendations = self._format_recommendations(movie_indices, scores, columns)
        recommendations.rename(columns={'similarity_score': 'hybrid_score'}, inplace=True)

        # Add component scores if requested
        if return_component_scores:
            content_scores = []
            metadata_scores = []
            cf_scores = []

            for movie_idx in movie_indices:
                content_scores.append(
                    self._content_recommender.similarity_matrix[idx, movie_idx]
                )
                metadata_scores.append(
                    self._metadata_recommender.similarity_matrix[idx, movie_idx]
                )
                cf_scores.append(
                    self._cf_recommender.similarity_matrix[idx, movie_idx]
                )

            recommendations['content_score'] = content_scores
            recommendations['metadata_score'] = metadata_scores
            recommendations['cf_score'] = cf_scores

        return recommendations

    def recommend_with_method(
        self,
        title: str,
        method: str,
        top_n: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Get recommendations using a specific method.

        Parameters
        ----------
        title : str
            Movie title
        method : str
            Method to use ('content', 'metadata', 'cf', 'hybrid')
        top_n : int
            Number of recommendations

        Returns
        -------
        pd.DataFrame or None
            Recommendations
        """
        if method == 'content':
            return self._content_recommender.recommend(title, top_n)
        elif method == 'metadata':
            return self._metadata_recommender.recommend(title, top_n)
        elif method == 'cf':
            return self._cf_recommender.recommend(title, top_n)
        else:
            return self.recommend(title, top_n)

    def compare_methods(
        self,
        title: str,
        top_n: int = 5
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Compare recommendations from all methods.

        Parameters
        ----------
        title : str
            Movie title
        top_n : int
            Number of recommendations per method

        Returns
        -------
        dict or None
            Dictionary of method name to recommendations
        """
        if not self._validate_fitted():
            return None

        idx = self.get_movie_index(title)
        if idx is None:
            return None

        results = {
            'content_based': self._content_recommender.recommend(title, top_n),
            'metadata_based': self._metadata_recommender.recommend(title, top_n),
            'collaborative_filtering': self._cf_recommender.recommend(title, top_n),
            'hybrid': self.recommend(title, top_n)
        }

        return results

    def get_method_scores(self, title1: str, title2: str) -> Optional[Dict[str, float]]:
        """
        Get similarity scores from each method for a movie pair.

        Parameters
        ----------
        title1 : str
            First movie title
        title2 : str
            Second movie title

        Returns
        -------
        dict or None
            Dictionary of method scores
        """
        idx1 = self.get_movie_index(title1)
        idx2 = self.get_movie_index(title2)

        if idx1 is None or idx2 is None:
            return None

        return {
            'content': float(self._content_recommender.similarity_matrix[idx1, idx2]),
            'metadata': float(self._metadata_recommender.similarity_matrix[idx1, idx2]),
            'cf': float(self._cf_recommender.similarity_matrix[idx1, idx2]),
            'hybrid': float(self._similarity_matrix[idx1, idx2])
        }

    def set_weights(
        self,
        content_weight: float,
        metadata_weight: float,
        cf_weight: float
    ):
        """
        Update method weights and recompute hybrid similarity.

        Parameters
        ----------
        content_weight : float
            Weight for content-based method
        metadata_weight : float
            Weight for metadata-based method
        cf_weight : float
            Weight for collaborative filtering method
        """
        total = content_weight + metadata_weight + cf_weight
        self.weights = (
            content_weight / total,
            metadata_weight / total,
            cf_weight / total
        )

        if self._is_fitted:
            self._compute_hybrid_similarity()

        logger.info(f"Updated weights: Content={self.weights[0]:.2f}, "
                    f"Metadata={self.weights[1]:.2f}, CF={self.weights[2]:.2f}")

    def optimize_weights_from_evaluation(
        self,
        evaluator,
        k: int = 10
    ) -> Tuple[float, float, float]:
        """
        Optimize weights based on evaluation results.

        Parameters
        ----------
        evaluator : RecommenderEvaluator
            Evaluator instance
        k : int
            Top-K for evaluation

        Returns
        -------
        tuple
            Optimized weights (content, metadata, cf)
        """
        logger.info("Optimizing weights based on evaluation results...")

        # Evaluate each method
        content_results = evaluator.evaluate(
            self._content_recommender.similarity_matrix,
            "Content-Based",
            k=k
        )
        metadata_results = evaluator.evaluate(
            self._metadata_recommender.similarity_matrix,
            "Metadata-Based",
            k=k
        )
        cf_results = evaluator.evaluate(
            self._cf_recommender.similarity_matrix,
            "Collaborative",
            k=k
        )

        # Use NDCG as the performance metric
        content_ndcg = content_results.get('ndcg@k', 0.0)
        metadata_ndcg = metadata_results.get('ndcg@k', 0.0)
        cf_ndcg = cf_results.get('ndcg@k', 0.0)

        logger.info(f"Method performance - Content: {content_ndcg:.4f}, "
                    f"Metadata: {metadata_ndcg:.4f}, CF: {cf_ndcg:.4f}")

        # Softmax-like weighting (avoid zero weights)
        scores = np.array([content_ndcg, metadata_ndcg, cf_ndcg])
        scores = np.maximum(scores, 0.01)  # Minimum weight to avoid zero
        weights = scores / scores.sum()

        self.set_weights(weights[0], weights[1], weights[2])
        logger.info(f"Optimized weights: Content={weights[0]:.3f}, "
                    f"Metadata={weights[1]:.3f}, CF={weights[2]:.3f}")

        return tuple(weights)

    @property
    def content_recommender(self) -> ContentBasedRecommender:
        """Get the content-based recommender."""
        return self._content_recommender

    @property
    def metadata_recommender(self) -> MetadataBasedRecommender:
        """Get the metadata-based recommender."""
        return self._metadata_recommender

    @property
    def cf_recommender(self) -> CollaborativeFilteringRecommender:
        """Get the collaborative filtering recommender."""
        return self._cf_recommender

    @property
    def current_weights(self) -> Dict[str, float]:
        """Get the current weights."""
        return {
            'content': self.weights[0],
            'metadata': self.weights[1],
            'cf': self.weights[2]
        }
