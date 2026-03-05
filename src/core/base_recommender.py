"""
Base recommender abstract class defining the interface for all recommenders.
"""
# @author 成员 B — 基础推荐算法 & 工具库

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation algorithms.

    All recommender implementations should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize the base recommender.

        Parameters
        ----------
        name : str
            Name of the recommender for logging/identification
        """
        self.name = name
        self._is_fitted = False
        self._movies_df: Optional[pd.DataFrame] = None
        self._similarity_matrix: Optional[np.ndarray] = None
        self._title_to_idx: Dict[str, int] = {}
        self._idx_to_title: Dict[int, str] = {}

    @abstractmethod
    def fit(self, movies_df: pd.DataFrame, **kwargs) -> 'BaseRecommender':
        """
        Fit the recommender model on the movie data.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Preprocessed movie dataframe
        **kwargs
            Additional parameters for fitting

        Returns
        -------
        BaseRecommender
            Self for method chaining
        """
        pass

    @abstractmethod
    def recommend(
        self,
        title: str,
        top_n: int = 10,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get recommendations for a given movie.

        Parameters
        ----------
        title : str
            Movie title to get recommendations for
        top_n : int
            Number of recommendations to return
        **kwargs
            Additional parameters

        Returns
        -------
        pd.DataFrame or None
            DataFrame with recommendations or None if movie not found
        """
        pass

    def _validate_fitted(self) -> bool:
        """Check if the recommender has been fitted."""
        if not self._is_fitted:
            logger.warning(f"{self.name} has not been fitted. Call fit() first.")
            return False
        return True

    def _create_index_mappings(self, movies_df: pd.DataFrame):
        """Create title-to-index and index-to-title mappings."""
        self._movies_df = movies_df.reset_index(drop=True)

        if 'title' in movies_df.columns:
            self._title_to_idx = pd.Series(
                self._movies_df.index,
                index=self._movies_df['title']
            ).to_dict()
            self._idx_to_title = pd.Series(
                self._movies_df['title'],
                index=self._movies_df.index
            ).to_dict()

    def get_movie_index(self, title: str) -> Optional[int]:
        """
        Get the index of a movie by title.

        Parameters
        ----------
        title : str
            Movie title

        Returns
        -------
        int or None
            Movie index or None if not found
        """
        return self._title_to_idx.get(title)

    def get_movie_title(self, idx: int) -> Optional[str]:
        """
        Get the title of a movie by index.

        Parameters
        ----------
        idx : int
            Movie index

        Returns
        -------
        str or None
            Movie title or None if not found
        """
        return self._idx_to_title.get(idx)

    def find_similar_titles(self, title: str, max_results: int = 5) -> List[str]:
        """
        Find movies with similar titles (fuzzy match).

        Parameters
        ----------
        title : str
            Partial title to search for
        max_results : int
            Maximum results to return

        Returns
        -------
        List[str]
            List of matching titles
        """
        if self._movies_df is None:
            return []

        matches = [
            t for t in self._title_to_idx.keys()
            if title.lower() in t.lower()
        ]
        return matches[:max_results]

    def get_similarity_score(self, title1: str, title2: str) -> Optional[float]:
        """
        Get the similarity score between two movies.

        Parameters
        ----------
        title1 : str
            First movie title
        title2 : str
            Second movie title

        Returns
        -------
        float or None
            Similarity score or None if movies not found
        """
        if self._similarity_matrix is None:
            return None

        idx1 = self.get_movie_index(title1)
        idx2 = self.get_movie_index(title2)

        if idx1 is None or idx2 is None:
            return None

        return float(self._similarity_matrix[idx1, idx2])

    def _format_recommendations(
        self,
        indices: List[int],
        scores: List[float],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Format recommendation results as a DataFrame.

        Parameters
        ----------
        indices : List[int]
            Movie indices
        scores : List[float]
            Similarity scores
        columns : List[str], optional
            Columns to include in output

        Returns
        -------
        pd.DataFrame
            Formatted recommendations
        """
        if columns is None:
            columns = ['title', 'genres_list', 'vote_average', 'popularity']

        # Filter to existing columns
        available_columns = [c for c in columns if c in self._movies_df.columns]

        recommendations = self._movies_df.iloc[indices][available_columns].copy()
        recommendations['similarity_score'] = scores

        return recommendations

    @property
    def is_fitted(self) -> bool:
        """Check if the recommender has been fitted."""
        return self._is_fitted

    @property
    def similarity_matrix(self) -> Optional[np.ndarray]:
        """Get the similarity matrix."""
        return self._similarity_matrix

    @property
    def movie_count(self) -> int:
        """Get the number of movies."""
        return len(self._movies_df) if self._movies_df is not None else 0

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.name}({status}, {self.movie_count} movies)"
