"""
Content-based recommendation using TF-IDF and cosine similarity.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import logging

from .base_recommender import BaseRecommender
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommender using TF-IDF vectorization of movie overviews.

    This recommender analyzes movie descriptions/overviews and finds similar
    movies based on textual content using TF-IDF and cosine similarity.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        stop_words: str = 'english'
    ):
        """
        Initialize the content-based recommender.

        Parameters
        ----------
        max_features : int
            Maximum number of features for TF-IDF
        ngram_range : tuple
            Range of n-grams to consider
        stop_words : str
            Stop words to remove
        """
        super().__init__(name="ContentBasedRecommender")

        settings = get_settings()
        self.max_features = max_features or settings.tfidf_max_features
        self.ngram_range = ngram_range or settings.tfidf_ngram_range
        self.stop_words = stop_words

        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None

    def fit(self, movies_df: pd.DataFrame, **kwargs) -> 'ContentBasedRecommender':
        """
        Fit the content-based model.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe with 'overview' column
        **kwargs
            Additional parameters (unused)

        Returns
        -------
        ContentBasedRecommender
            Self for method chaining
        """
        logger.info("Fitting ContentBasedRecommender...")

        # Create index mappings
        self._create_index_mappings(movies_df)

        # Ensure overview column exists and handle missing values
        if 'overview' not in self._movies_df.columns:
            raise ValueError("DataFrame must contain 'overview' column")

        overviews = self._movies_df['overview'].fillna('')

        # Create TF-IDF vectorizer
        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )

        # Fit and transform
        logger.info("Building TF-IDF matrix...")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(overviews)

        logger.info(f"TF-IDF matrix shape: {self._tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self._tfidf_vectorizer.vocabulary_)}")

        # Compute similarity matrix
        logger.info("Computing cosine similarity matrix...")
        self._similarity_matrix = linear_kernel(self._tfidf_matrix, self._tfidf_matrix)

        self._is_fitted = True
        logger.info("ContentBasedRecommender fitting complete.")

        return self

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get content-based recommendations for a movie.

        Parameters
        ----------
        title : str
            Movie title
        top_n : int
            Number of recommendations
        **kwargs
            Additional parameters (unused)

        Returns
        -------
        pd.DataFrame or None
            Recommendations or None if movie not found
        """
        if not self._validate_fitted():
            return None

        # Get movie index
        idx = self.get_movie_index(title)

        if idx is None:
            logger.warning(f"Movie not found: {title}")
            matches = self.find_similar_titles(title)
            if matches:
                logger.info(f"Similar titles: {matches}")
            return None

        # Get similarity scores
        sim_scores = list(enumerate(self._similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the movie itself and get top N
        sim_scores = sim_scores[1:top_n + 1]

        # Extract indices and scores
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        # Format and return
        columns = ['title', 'genres_list', 'vote_average', 'overview']
        return self._format_recommendations(movie_indices, scores, columns)

    def get_top_keywords(self, title: str, top_n: int = 10) -> Optional[List[Tuple[str, float]]]:
        """
        Get the top TF-IDF keywords for a movie.

        Parameters
        ----------
        title : str
            Movie title
        top_n : int
            Number of keywords to return

        Returns
        -------
        List[Tuple[str, float]] or None
            List of (keyword, score) tuples or None
        """
        if not self._validate_fitted():
            return None

        idx = self.get_movie_index(title)
        if idx is None:
            return None

        # Get TF-IDF vector for this movie
        tfidf_vector = self._tfidf_matrix[idx].toarray().flatten()

        # Get feature names
        feature_names = self._tfidf_vectorizer.get_feature_names_out()

        # Get top N keywords
        top_indices = np.argsort(tfidf_vector)[::-1][:top_n]

        keywords = [
            (feature_names[i], float(tfidf_vector[i]))
            for i in top_indices
            if tfidf_vector[i] > 0
        ]

        return keywords

    def get_common_keywords(
        self,
        title1: str,
        title2: str,
        top_n: int = 5
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Get common important keywords between two movies.

        Parameters
        ----------
        title1 : str
            First movie title
        title2 : str
            Second movie title
        top_n : int
            Number of keywords to return

        Returns
        -------
        List[Tuple[str, float]] or None
            List of (keyword, combined_score) tuples or None
        """
        if not self._validate_fitted():
            return None

        idx1 = self.get_movie_index(title1)
        idx2 = self.get_movie_index(title2)

        if idx1 is None or idx2 is None:
            return None

        # Get TF-IDF vectors
        vec1 = self._tfidf_matrix[idx1].toarray().flatten()
        vec2 = self._tfidf_matrix[idx2].toarray().flatten()

        # Compute geometric mean (emphasizes keywords important in both)
        combined = np.sqrt(vec1 * vec2)

        # Get feature names
        feature_names = self._tfidf_vectorizer.get_feature_names_out()

        # Get top N
        top_indices = np.argsort(combined)[::-1][:top_n]

        keywords = [
            (feature_names[i], float(combined[i]))
            for i in top_indices
            if combined[i] > 0
        ]

        return keywords

    @property
    def tfidf_vectorizer(self) -> Optional[TfidfVectorizer]:
        """Get the TF-IDF vectorizer."""
        return self._tfidf_vectorizer

    @property
    def tfidf_matrix(self):
        """Get the TF-IDF matrix."""
        return self._tfidf_matrix

    @property
    def vocabulary_size(self) -> int:
        """Get the vocabulary size."""
        if self._tfidf_vectorizer is None:
            return 0
        return len(self._tfidf_vectorizer.vocabulary_)
