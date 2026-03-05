"""
Data loading module for the Movie Recommendation System.
"""
# @author 成员 A — 数据工程 & 预处理

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading of movie and credits data from CSV files.
    """

    def __init__(
        self,
        movies_path: Optional[Path] = None,
        credits_path: Optional[Path] = None
    ):
        """
        Initialize the data loader.

        Parameters
        ----------
        movies_path : Path, optional
            Path to movies CSV file
        credits_path : Path, optional
            Path to credits CSV file
        """
        settings = get_settings()
        self.movies_path = movies_path or settings.movies_path
        self.credits_path = credits_path or settings.credits_path

        self._movies_df: Optional[pd.DataFrame] = None
        self._credits_df: Optional[pd.DataFrame] = None

    def load_movies(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load movies data from CSV.

        Parameters
        ----------
        force_reload : bool
            Force reload even if data is cached

        Returns
        -------
        pd.DataFrame
            Movies dataframe
        """
        if self._movies_df is not None and not force_reload:
            return self._movies_df

        logger.info(f"Loading movies from {self.movies_path}")

        if not self.movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {self.movies_path}")

        self._movies_df = pd.read_csv(self.movies_path)
        logger.info(f"Loaded {len(self._movies_df)} movies")

        return self._movies_df

    def load_credits(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load credits data from CSV.

        Parameters
        ----------
        force_reload : bool
            Force reload even if data is cached

        Returns
        -------
        pd.DataFrame
            Credits dataframe
        """
        if self._credits_df is not None and not force_reload:
            return self._credits_df

        logger.info(f"Loading credits from {self.credits_path}")

        if not self.credits_path.exists():
            raise FileNotFoundError(f"Credits file not found: {self.credits_path}")

        self._credits_df = pd.read_csv(self.credits_path)
        logger.info(f"Loaded {len(self._credits_df)} credit records")

        return self._credits_df

    def load_all(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both movies and credits data.

        Parameters
        ----------
        force_reload : bool
            Force reload even if data is cached

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of (movies_df, credits_df)
        """
        movies = self.load_movies(force_reload)
        credits = self.load_credits(force_reload)
        return movies, credits

    def get_merged_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Get merged movies and credits data.

        Parameters
        ----------
        force_reload : bool
            Force reload even if data is cached

        Returns
        -------
        pd.DataFrame
            Merged dataframe
        """
        movies, credits = self.load_all(force_reload)

        # Standardize column names
        if 'movie_id' in credits.columns:
            credits = credits.rename(columns={'movie_id': 'id'})

        # Select columns to merge
        credits_cols = ['id']
        if 'cast' in credits.columns:
            credits_cols.append('cast')
        if 'crew' in credits.columns:
            credits_cols.append('crew')

        # Merge on id
        merged = movies.merge(credits[credits_cols], on='id', how='left')
        logger.info(f"Merged dataset has {len(merged)} records")

        return merged

    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that dataframe has required columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate
        required_columns : list
            List of required column names

        Returns
        -------
        bool
            True if all required columns are present
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True

    def get_movie_count(self) -> int:
        """Get the number of movies in the dataset."""
        if self._movies_df is None:
            self.load_movies()
        return len(self._movies_df)

    def clear_cache(self):
        """Clear cached dataframes."""
        self._movies_df = None
        self._credits_df = None
        logger.info("Data cache cleared")
