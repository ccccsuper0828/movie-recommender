"""
Recommendation service - main business logic for recommendations.
"""

from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import logging

from src.core import (
    ContentBasedRecommender,
    MetadataBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender
)
from src.data import DataLoader, DataPreprocessor, CacheManager
from src.explainability import RuleBasedExplainer
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service layer for movie recommendations.

    Handles initialization of recommenders, caching, and provides
    a unified interface for all recommendation operations.
    """

    def __init__(self, auto_init: bool = True):
        """
        Initialize the recommendation service.

        Parameters
        ----------
        auto_init : bool
            Whether to automatically initialize recommenders
        """
        self.settings = get_settings()
        self.cache = CacheManager()

        # Data
        self._movies_df: Optional[pd.DataFrame] = None
        self._preprocessor: Optional[DataPreprocessor] = None

        # Recommenders
        self._content_recommender: Optional[ContentBasedRecommender] = None
        self._metadata_recommender: Optional[MetadataBasedRecommender] = None
        self._cf_recommender: Optional[CollaborativeFilteringRecommender] = None
        self._hybrid_recommender: Optional[HybridRecommender] = None

        # Explainer
        self._explainer: Optional[RuleBasedExplainer] = None

        self._initialized = False

        if auto_init:
            self.initialize()

    def initialize(self) -> 'RecommendationService':
        """
        Initialize data and recommenders.

        Returns
        -------
        RecommendationService
            Self for method chaining
        """
        if self._initialized:
            return self

        logger.info("Initializing RecommendationService...")

        # Load and preprocess data
        loader = DataLoader()
        merged_df = loader.get_merged_data()

        self._preprocessor = DataPreprocessor(merged_df)
        self._movies_df = self._preprocessor.preprocess()

        logger.info(f"Loaded {len(self._movies_df)} movies")

        # Initialize recommenders
        self._init_recommenders()

        # Initialize explainer
        self._explainer = RuleBasedExplainer(
            self._movies_df,
            self._content_recommender.tfidf_vectorizer if self._content_recommender else None,
            self._content_recommender.tfidf_matrix if self._content_recommender else None
        )

        self._initialized = True
        logger.info("RecommendationService initialization complete")

        return self

    def _init_recommenders(self):
        """Initialize all recommender models."""
        logger.info("Initializing recommenders...")

        # Content-based
        self._content_recommender = ContentBasedRecommender()
        self._content_recommender.fit(self._movies_df)

        # Metadata-based
        self._metadata_recommender = MetadataBasedRecommender()
        self._metadata_recommender.fit(self._movies_df)

        # Collaborative filtering
        self._cf_recommender = CollaborativeFilteringRecommender()
        self._cf_recommender.fit(self._movies_df)

        # Hybrid
        self._hybrid_recommender = HybridRecommender(
            weights=self.settings.hybrid_weights
        )
        self._hybrid_recommender.fit(self._movies_df)

    def get_recommendations(
        self,
        title: str,
        top_n: int = 10,
        method: str = "hybrid",
        weights: Optional[Tuple[float, float, float]] = None,
        include_explanation: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get recommendations for a movie.

        Parameters
        ----------
        title : str
            Movie title
        top_n : int
            Number of recommendations
        method : str
            Method: 'content', 'metadata', 'cf', or 'hybrid'
        weights : tuple, optional
            Custom weights for hybrid method
        include_explanation : bool
            Whether to include explanations

        Returns
        -------
        dict or None
            Recommendation results
        """
        if not self._initialized:
            self.initialize()

        # Get recommendations based on method
        if method == "content":
            recs_df = self._content_recommender.recommend(title, top_n)
        elif method == "metadata":
            recs_df = self._metadata_recommender.recommend(title, top_n)
        elif method == "cf":
            recs_df = self._cf_recommender.recommend(title, top_n)
        else:
            if weights:
                self._hybrid_recommender.set_weights(*weights)
            recs_df = self._hybrid_recommender.recommend(title, top_n, return_component_scores=True)

        if recs_df is None:
            # Try to find similar titles
            matches = self._preprocessor.search_titles(title, max_results=5)
            return {
                'error': f"Movie not found: {title}",
                'suggestions': matches
            }

        # Format response
        recommendations = []
        for rank, (_, row) in enumerate(recs_df.iterrows(), 1):
            rec = {
                'rank': rank,
                'title': row['title'],
                'genres': row.get('genres_list', []),
                'vote_average': row.get('vote_average', 0),
                'similarity_score': row.get('hybrid_score', row.get('similarity_score', 0))
            }

            # Add method-specific scores if available
            if 'content_score' in row:
                rec['method_scores'] = {
                    'content': row['content_score'],
                    'metadata': row['metadata_score'],
                    'cf': row['cf_score']
                }

            # Add explanation
            if include_explanation:
                explanation = self._explainer.explain_metadata_based(title, row['title'])
                rec['explanation'] = explanation['summary']
                rec['reasons'] = explanation['reasons']

            recommendations.append(rec)

        return {
            'source_movie': title,
            'method': method,
            'recommendations': recommendations,
            'total_count': len(recommendations)
        }

    def compare_methods(
        self,
        title: str,
        top_n: int = 5
    ) -> Optional[Dict[str, Any]]:
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
            Comparison results
        """
        if not self._initialized:
            self.initialize()

        results = self._hybrid_recommender.compare_methods(title, top_n)

        if results is None:
            return None

        formatted = {}
        for method, recs_df in results.items():
            if recs_df is not None:
                formatted[method] = [
                    {
                        'title': row['title'],
                        'similarity_score': row.get('similarity_score', row.get('hybrid_score', 0))
                    }
                    for _, row in recs_df.iterrows()
                ]

        return formatted

    def get_similarity_scores(
        self,
        title1: str,
        title2: str
    ) -> Optional[Dict[str, float]]:
        """
        Get similarity scores between two movies.

        Parameters
        ----------
        title1 : str
            First movie title
        title2 : str
            Second movie title

        Returns
        -------
        dict or None
            Similarity scores from each method
        """
        if not self._initialized:
            self.initialize()

        return self._hybrid_recommender.get_method_scores(title1, title2)

    def explain_recommendation(
        self,
        source_title: str,
        target_title: str,
        method: str = "metadata"
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed explanation for a recommendation.

        Parameters
        ----------
        source_title : str
            Source movie
        target_title : str
            Recommended movie
        method : str
            Explanation method

        Returns
        -------
        dict or None
            Explanation details
        """
        if not self._initialized:
            self.initialize()

        if method == "content":
            return self._explainer.explain_content_based(source_title, target_title)
        elif method == "metadata":
            return self._explainer.explain_metadata_based(source_title, target_title)
        elif method == "cf":
            return self._explainer.explain_collaborative_filtering(
                source_title, target_title,
                user_movie_matrix=self._cf_recommender.user_movie_matrix
            )
        elif method == "hybrid":
            scores = self.get_similarity_scores(source_title, target_title)
            if scores:
                return self._explainer.explain_hybrid(
                    source_title, target_title,
                    content_score=scores.get('content'),
                    metadata_score=scores.get('metadata'),
                    cf_score=scores.get('cf')
                )
        return None

    def get_popular_movies(self, top_n: int = 20) -> pd.DataFrame:
        """Get the most popular movies."""
        if not self._initialized:
            self.initialize()

        return self._movies_df.nlargest(top_n, 'popularity')[
            ['title', 'genres_list', 'vote_average', 'popularity']
        ]

    def search_movies(self, query: str, max_results: int = 10) -> List[str]:
        """Search for movies by title."""
        if not self._initialized:
            self.initialize()

        return self._preprocessor.search_titles(query, max_results)

    def get_movie_info(self, title: str) -> Optional[Dict[str, Any]]:
        """Get movie information."""
        if not self._initialized:
            self.initialize()

        movie = self._preprocessor.get_movie_by_title(title)
        if movie is None:
            return None

        return {
            'title': movie.get('title'),
            'overview': movie.get('overview', ''),
            'genres': movie.get('genres_list', []),
            'director': movie.get('director', ''),
            'cast': movie.get('cast_list', [])[:5],
            'vote_average': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0),
            'release_date': movie.get('release_date', ''),
            'runtime': movie.get('runtime', 0)
        }

    @property
    def movie_count(self) -> int:
        """Get total number of movies."""
        return len(self._movies_df) if self._movies_df is not None else 0

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
