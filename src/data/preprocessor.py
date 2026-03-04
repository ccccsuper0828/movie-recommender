"""
Data preprocessing module for the Movie Recommendation System.
"""

import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Any, Optional, Callable
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles preprocessing of movie data including JSON parsing,
    feature extraction, and data cleaning.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the preprocessor.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Dataframe to preprocess
        """
        self.df = df
        self.title_to_idx: Dict[str, int] = {}
        self.idx_to_title: Dict[int, str] = {}
        self._processed = False

    def set_data(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Set the dataframe to preprocess.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        DataPreprocessor
            Self for method chaining
        """
        self.df = df.copy()
        self._processed = False
        return self

    @staticmethod
    def safe_literal_eval(x: Any) -> Any:
        """
        Safely evaluate a string as a Python literal.

        Parameters
        ----------
        x : Any
            Value to evaluate

        Returns
        -------
        Any
            Evaluated value or empty list if evaluation fails
        """
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(str(x))
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def get_names(x: List[Dict], max_items: int = 5) -> List[str]:
        """
        Extract 'name' field from a list of dictionaries.

        Parameters
        ----------
        x : List[Dict]
            List of dictionaries
        max_items : int
            Maximum number of items to return

        Returns
        -------
        List[str]
            List of names
        """
        if not isinstance(x, list):
            return []
        return [item.get('name', '') for item in x if isinstance(item, dict)][:max_items]

    @staticmethod
    def get_director(crew: List[Dict]) -> str:
        """
        Extract director name from crew list.

        Parameters
        ----------
        crew : List[Dict]
            List of crew members

        Returns
        -------
        str
            Director name or empty string
        """
        if not isinstance(crew, list):
            return ''
        for member in crew:
            if isinstance(member, dict) and member.get('job') == 'Director':
                return member.get('name', '')
        return ''

    @staticmethod
    def clean_text_list(items: List[str], lowercase: bool = True, remove_spaces: bool = True) -> List[str]:
        """
        Clean a list of text items.

        Parameters
        ----------
        items : List[str]
            List of text items
        lowercase : bool
            Convert to lowercase
        remove_spaces : bool
            Remove spaces from items

        Returns
        -------
        List[str]
            Cleaned list
        """
        if not isinstance(items, list):
            return []

        cleaned = []
        for item in items:
            if not item:
                continue
            text = str(item)
            if lowercase:
                text = text.lower()
            if remove_spaces:
                text = text.replace(' ', '')
            cleaned.append(text)
        return cleaned

    def preprocess(self) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe
        """
        if self.df is None:
            raise ValueError("No dataframe set. Use set_data() first.")

        logger.info("Starting preprocessing...")

        # Parse JSON columns
        self._parse_json_columns()

        # Extract features
        self._extract_features()

        # Clean text features
        self._clean_features()

        # Handle missing values
        self._handle_missing_values()

        # Create index mappings
        self._create_index_mappings()

        self._processed = True
        logger.info(f"Preprocessing complete. {len(self.df)} movies processed.")

        return self.df

    def _parse_json_columns(self):
        """Parse JSON string columns into Python objects."""
        json_columns = ['genres', 'keywords', 'cast', 'crew',
                        'production_companies', 'production_countries',
                        'spoken_languages']

        for col in json_columns:
            if col in self.df.columns:
                logger.debug(f"Parsing JSON column: {col}")
                self.df[col] = self.df[col].apply(self.safe_literal_eval)

    def _extract_features(self):
        """Extract features from parsed columns."""
        # Genres
        if 'genres' in self.df.columns:
            self.df['genres_list'] = self.df['genres'].apply(self.get_names)

        # Keywords
        if 'keywords' in self.df.columns:
            self.df['keywords_list'] = self.df['keywords'].apply(self.get_names)

        # Cast
        if 'cast' in self.df.columns:
            self.df['cast_list'] = self.df['cast'].apply(lambda x: self.get_names(x, max_items=5))

        # Director
        if 'crew' in self.df.columns:
            self.df['director'] = self.df['crew'].apply(self.get_director)

        # Production companies
        if 'production_companies' in self.df.columns:
            self.df['production_companies_list'] = self.df['production_companies'].apply(self.get_names)

        logger.debug("Feature extraction complete")

    def _clean_features(self):
        """Clean text features for similarity computation."""
        # Clean genres
        if 'genres_list' in self.df.columns:
            self.df['genres_clean'] = self.df['genres_list'].apply(self.clean_text_list)

        # Clean keywords
        if 'keywords_list' in self.df.columns:
            self.df['keywords_clean'] = self.df['keywords_list'].apply(self.clean_text_list)

        # Clean cast
        if 'cast_list' in self.df.columns:
            self.df['cast_clean'] = self.df['cast_list'].apply(self.clean_text_list)

        # Clean director
        if 'director' in self.df.columns:
            self.df['director_clean'] = self.df['director'].apply(
                lambda x: str(x).lower().replace(' ', '') if x else ''
            )

        logger.debug("Feature cleaning complete")

    def _handle_missing_values(self):
        """Handle missing values in key columns."""
        # Overview
        if 'overview' in self.df.columns:
            self.df['overview'] = self.df['overview'].fillna('')

        # Release date
        if 'release_date' in self.df.columns:
            self.df['release_date'] = self.df['release_date'].fillna('')

        # Runtime
        if 'runtime' in self.df.columns:
            self.df['runtime'] = self.df['runtime'].fillna(0)

        # Vote average
        if 'vote_average' in self.df.columns:
            self.df['vote_average'] = self.df['vote_average'].fillna(0)

        # Popularity
        if 'popularity' in self.df.columns:
            self.df['popularity'] = self.df['popularity'].fillna(0)

        # Budget and revenue
        if 'budget' in self.df.columns:
            self.df['budget'] = self.df['budget'].fillna(0)
        if 'revenue' in self.df.columns:
            self.df['revenue'] = self.df['revenue'].fillna(0)

        # Initialize empty lists for list columns
        list_columns = ['genres_list', 'keywords_list', 'cast_list',
                        'genres_clean', 'keywords_clean', 'cast_clean']
        for col in list_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: x if isinstance(x, list) else [])

        logger.debug("Missing value handling complete")

    def _create_index_mappings(self):
        """Create title-to-index and index-to-title mappings."""
        self.df = self.df.reset_index(drop=True)

        if 'title' in self.df.columns:
            self.title_to_idx = pd.Series(self.df.index, index=self.df['title']).to_dict()
            self.idx_to_title = pd.Series(self.df['title'], index=self.df.index).to_dict()

    def create_soup(
        self,
        genre_weight: int = 3,
        director_weight: int = 3,
        cast_weight: int = 2,
        keyword_weight: int = 1
    ) -> pd.Series:
        """
        Create 'soup' feature for metadata-based similarity.

        Parameters
        ----------
        genre_weight : int
            Weight for genre features
        director_weight : int
            Weight for director feature
        cast_weight : int
            Weight for cast features
        keyword_weight : int
            Weight for keyword features

        Returns
        -------
        pd.Series
            Series of soup strings
        """
        def _create_soup(row):
            features = []

            # Genres
            if 'genres_clean' in row and isinstance(row['genres_clean'], list):
                features.extend(row['genres_clean'] * genre_weight)

            # Director
            if 'director_clean' in row and row['director_clean']:
                features.extend([row['director_clean']] * director_weight)

            # Cast
            if 'cast_clean' in row and isinstance(row['cast_clean'], list):
                features.extend(row['cast_clean'] * cast_weight)

            # Keywords
            if 'keywords_clean' in row and isinstance(row['keywords_clean'], list):
                features.extend(row['keywords_clean'] * keyword_weight)

            return ' '.join(features)

        self.df['soup'] = self.df.apply(_create_soup, axis=1)
        return self.df['soup']

    def get_movie_by_title(self, title: str) -> Optional[pd.Series]:
        """
        Get movie data by title.

        Parameters
        ----------
        title : str
            Movie title

        Returns
        -------
        pd.Series or None
            Movie data or None if not found
        """
        idx = self.title_to_idx.get(title)
        if idx is not None:
            return self.df.iloc[idx]
        return None

    def get_movie_by_idx(self, idx: int) -> Optional[pd.Series]:
        """
        Get movie data by index.

        Parameters
        ----------
        idx : int
            Movie index

        Returns
        -------
        pd.Series or None
            Movie data or None if not found
        """
        if 0 <= idx < len(self.df):
            return self.df.iloc[idx]
        return None

    def search_titles(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search for movie titles containing the query.

        Parameters
        ----------
        query : str
            Search query
        max_results : int
            Maximum number of results

        Returns
        -------
        List[str]
            List of matching titles
        """
        if 'title' not in self.df.columns:
            return []

        mask = self.df['title'].str.contains(query, case=False, na=False)
        return self.df.loc[mask, 'title'].head(max_results).tolist()

    @property
    def is_processed(self) -> bool:
        """Check if data has been processed."""
        return self._processed
