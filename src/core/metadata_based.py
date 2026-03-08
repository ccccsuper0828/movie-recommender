"""
Metadata-based recommendation using genres, directors, cast, and keywords.
"""
# @author 成员 B — 基础推荐算法 & 工具库

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .base_recommender import BaseRecommender
from config.settings import get_settings

logger = logging.getLogger(__name__)


class MetadataBasedRecommender(BaseRecommender):
    """
    Metadata-based recommender using movie attributes.

    This recommender combines multiple metadata features (genres, director,
    cast, keywords) with configurable weights to find similar movies.
    """

    def __init__(
        self,
        max_features: int = 10000,
        genre_weight: int = 5,  # Increased from 3
        director_weight: int = 4,  # Increased from 3
        cast_weight: int = 3,  # Increased from 2
        keyword_weight: int = 2  # Increased from 1
    ):
        """
        Initialize the metadata-based recommender.

        Parameters
        ----------
        max_features : int
            Maximum features for CountVectorizer
        genre_weight : int
            Weight multiplier for genres
        director_weight : int
            Weight multiplier for director
        cast_weight : int
            Weight multiplier for cast
        keyword_weight : int
            Weight multiplier for keywords
        """
        super().__init__(name="MetadataBasedRecommender")

        settings = get_settings()
        self.max_features = max_features or settings.count_vectorizer_max_features

        # Feature weights
        self.genre_weight = genre_weight
        self.director_weight = director_weight
        self.cast_weight = cast_weight
        self.keyword_weight = keyword_weight

        self._count_vectorizer: Optional[CountVectorizer] = None
        self._count_matrix = None

    def _create_soup(self, row: pd.Series) -> str:
        """
        Create a 'soup' of metadata features for a movie.

        Parameters
        ----------
        row : pd.Series
            Movie row

        Returns
        -------
        str
            Combined feature string
        """
        features = []

        # Add genres with weight
        genres = row.get('genres_clean', [])
        if isinstance(genres, list):
            features.extend(genres * self.genre_weight)

        # Add director with weight
        director = row.get('director_clean', '')
        if director:
            features.extend([director] * self.director_weight)

        # Add cast with weight
        cast = row.get('cast_clean', [])
        if isinstance(cast, list):
            features.extend(cast * self.cast_weight)

        # Add keywords with weight
        keywords = row.get('keywords_clean', [])
        if isinstance(keywords, list):
            features.extend(keywords * self.keyword_weight)

        return ' '.join(features)

    def fit(self, movies_df: pd.DataFrame, **kwargs) -> 'MetadataBasedRecommender':
        """
        Fit the metadata-based model.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe with metadata columns
        **kwargs
            Additional parameters (unused)

        Returns
        -------
        MetadataBasedRecommender
            Self for method chaining
        """
        logger.info("Fitting MetadataBasedRecommender...")

        # Create index mappings
        self._create_index_mappings(movies_df)

        # Create soup feature
        logger.info("Creating metadata soup...")
        soup = self._movies_df.apply(self._create_soup, axis=1)

        # Create CountVectorizer
        self._count_vectorizer = CountVectorizer(
            stop_words='english',
            max_features=self.max_features
        )

        # Fit and transform
        logger.info("Building count matrix...")
        self._count_matrix = self._count_vectorizer.fit_transform(soup)

        logger.info(f"Count matrix shape: {self._count_matrix.shape}")

        # Compute similarity matrix
        logger.info("Computing cosine similarity matrix...")
        self._similarity_matrix = cosine_similarity(self._count_matrix, self._count_matrix)

        self._is_fitted = True
        logger.info("MetadataBasedRecommender fitting complete.")

        return self

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get metadata-based recommendations for a movie.

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
        columns = ['title', 'genres_list', 'director', 'cast_list', 'vote_average']
        return self._format_recommendations(movie_indices, scores, columns)

    def get_matching_features(
        self,
        title1: str,
        title2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get matching features between two movies.

        Parameters
        ----------
        title1 : str
            First movie title
        title2 : str
            Second movie title

        Returns
        -------
        dict or None
            Dictionary of matching features
        """
        idx1 = self.get_movie_index(title1)
        idx2 = self.get_movie_index(title2)

        if idx1 is None or idx2 is None:
            return None

        movie1 = self._movies_df.iloc[idx1]
        movie2 = self._movies_df.iloc[idx2]

        matches = {
            'same_director': False,
            'director': None,
            'common_genres': [],
            'common_cast': [],
            'common_keywords': []
        }

        # Check director
        dir1 = movie1.get('director', '')
        dir2 = movie2.get('director', '')
        if dir1 and dir2 and dir1 == dir2:
            matches['same_director'] = True
            matches['director'] = dir1

        # Check genres
        genres1 = set(movie1.get('genres_list', []) or [])
        genres2 = set(movie2.get('genres_list', []) or [])
        matches['common_genres'] = list(genres1 & genres2)

        # Check cast
        cast1 = set(movie1.get('cast_list', []) or [])
        cast2 = set(movie2.get('cast_list', []) or [])
        matches['common_cast'] = list(cast1 & cast2)

        # Check keywords
        kw1 = set(movie1.get('keywords_list', []) or [])
        kw2 = set(movie2.get('keywords_list', []) or [])
        matches['common_keywords'] = list(kw1 & kw2)

        return matches

    def get_feature_contributions(
        self,
        title1: str,
        title2: str
    ) -> Optional[Dict[str, float]]:
        """
        Calculate the contribution of each feature type to similarity.

        Parameters
        ----------
        title1 : str
            First movie title
        title2 : str
            Second movie title

        Returns
        -------
        dict or None
            Dictionary of feature contributions
        """
        matches = self.get_matching_features(title1, title2)
        if matches is None:
            return None

        contributions = {}
        total_weight = (
            self.genre_weight +
            self.director_weight +
            self.cast_weight +
            self.keyword_weight
        )

        # Director contribution
        director_contrib = self.director_weight / total_weight if matches['same_director'] else 0
        contributions['director'] = director_contrib

        # Genre contribution (based on overlap)
        movie1 = self._movies_df.iloc[self.get_movie_index(title1)]
        n_genres = len(movie1.get('genres_list', []) or [])
        n_common_genres = len(matches['common_genres'])
        if n_genres > 0:
            genre_overlap = n_common_genres / n_genres
            contributions['genres'] = (self.genre_weight / total_weight) * genre_overlap
        else:
            contributions['genres'] = 0

        # Cast contribution
        n_cast = len(movie1.get('cast_list', []) or [])
        n_common_cast = len(matches['common_cast'])
        if n_cast > 0:
            cast_overlap = n_common_cast / n_cast
            contributions['cast'] = (self.cast_weight / total_weight) * cast_overlap
        else:
            contributions['cast'] = 0

        # Keywords contribution
        n_keywords = len(movie1.get('keywords_list', []) or [])
        n_common_keywords = len(matches['common_keywords'])
        if n_keywords > 0:
            keyword_overlap = n_common_keywords / n_keywords
            contributions['keywords'] = (self.keyword_weight / total_weight) * keyword_overlap
        else:
            contributions['keywords'] = 0

        contributions['total'] = sum(contributions.values())

        return contributions

    @property
    def weights(self) -> Dict[str, int]:
        """Get the feature weights."""
        return {
            'genre': self.genre_weight,
            'director': self.director_weight,
            'cast': self.cast_weight,
            'keyword': self.keyword_weight
        }
