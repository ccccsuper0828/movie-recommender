"""
MovieLens data loader and TMDB linker.

Loads the MovieLens 1M dataset and links it to the existing TMDB 5000
movie catalog via fuzzy title matching, producing a real user-movie
rating matrix that replaces the previously simulated one.
"""
# @author 成员 A — 数据工程 & 预处理

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

from config.settings import get_settings

logger = logging.getLogger(__name__)


class MovieLensLoader:
    """Load MovieLens 1M data and link it to the TMDB movie catalog."""

    def __init__(
        self,
        ratings_path: Optional[Path] = None,
        ml_movies_path: Optional[Path] = None,
        ml_users_path: Optional[Path] = None,
    ):
        settings = get_settings()
        raw_dir = settings.data_dir / "raw"

        self.ratings_path = ratings_path or raw_dir / "ml_ratings.csv"
        self.ml_movies_path = ml_movies_path or raw_dir / "ml_movies.csv"
        self.ml_users_path = ml_users_path or raw_dir / "ml_users.csv"

        # Caches
        self._ratings: Optional[pd.DataFrame] = None
        self._ml_movies: Optional[pd.DataFrame] = None
        self._ml_users: Optional[pd.DataFrame] = None
        self._link_map: Optional[Dict[int, int]] = None  # ml_movieId -> tmdb_idx

    # ------------------------------------------------------------------
    # Raw loaders
    # ------------------------------------------------------------------
    def load_ratings(self) -> pd.DataFrame:
        if self._ratings is None:
            logger.info(f"Loading MovieLens ratings from {self.ratings_path}")
            self._ratings = pd.read_csv(self.ratings_path)
            logger.info(f"  {len(self._ratings):,} ratings loaded")
        return self._ratings

    def load_ml_movies(self) -> pd.DataFrame:
        if self._ml_movies is None:
            self._ml_movies = pd.read_csv(self.ml_movies_path)
        return self._ml_movies

    def load_ml_users(self) -> pd.DataFrame:
        if self._ml_users is None:
            self._ml_users = pd.read_csv(self.ml_users_path)
        return self._ml_users

    # ------------------------------------------------------------------
    # Title normalisation helpers
    # ------------------------------------------------------------------
    _YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")

    @classmethod
    def _normalise(cls, title: str) -> str:
        """Lower-case, strip year suffix, collapse whitespace."""
        t = cls._YEAR_RE.sub("", str(title))
        t = re.sub(r"[^\w\s]", " ", t.lower())
        return " ".join(t.split())

    # ------------------------------------------------------------------
    # Linking MovieLens movieId → TMDB DataFrame index
    # ------------------------------------------------------------------
    def build_link_map(
        self,
        tmdb_df: pd.DataFrame,
        fuzzy_threshold: float = 0.85,
    ) -> Dict[int, int]:
        """
        Match MovieLens movies to TMDB movies by title.

        Returns a dict  {ml_movieId: tmdb_row_index}.
        """
        if self._link_map is not None:
            return self._link_map

        ml_movies = self.load_ml_movies()

        # Build normalised lookup for TMDB
        tmdb_lookup: Dict[str, int] = {}
        for idx, title in enumerate(tmdb_df["title"]):
            tmdb_lookup[self._normalise(title)] = idx

        link: Dict[int, int] = {}
        unmatched = 0

        for _, row in ml_movies.iterrows():
            ml_id = int(row["movieId"])
            norm = self._normalise(row["title"])

            # 1) exact match
            if norm in tmdb_lookup:
                link[ml_id] = tmdb_lookup[norm]
                continue

            # 2) fuzzy match (only against titles starting with same letter)
            best_score, best_idx = 0.0, -1
            first_char = norm[0] if norm else ""
            for t_norm, t_idx in tmdb_lookup.items():
                if t_norm and t_norm[0] == first_char:
                    score = SequenceMatcher(None, norm, t_norm).ratio()
                    if score > best_score:
                        best_score, best_idx = score, t_idx

            if best_score >= fuzzy_threshold:
                link[ml_id] = best_idx
            else:
                unmatched += 1

        logger.info(
            f"MovieLens→TMDB link: {len(link)} matched, {unmatched} unmatched "
            f"(of {len(ml_movies)} ML movies)"
        )
        self._link_map = link
        return link

    # ------------------------------------------------------------------
    # Build the real user-movie rating matrix
    # ------------------------------------------------------------------
    def build_rating_matrix(
        self,
        tmdb_df: pd.DataFrame,
        min_ratings_per_user: int = 5,
        max_users: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a dense user × movie rating matrix aligned to *tmdb_df* rows.

        Parameters
        ----------
        tmdb_df : pd.DataFrame
            The preprocessed TMDB movie dataframe (index = movie position).
        min_ratings_per_user : int
            Drop users with fewer linked ratings.
        max_users : int | None
            Cap number of users (for speed during dev). None = all.

        Returns
        -------
        rating_matrix : np.ndarray  shape (n_users, n_movies)
            0 means "not rated".
        user_ids : np.ndarray       original MovieLens userIds
        movie_indices : np.ndarray  tmdb_df row indices that have ratings
        """
        ratings = self.load_ratings()
        link = self.build_link_map(tmdb_df)

        n_movies = len(tmdb_df)

        # Map ml movieIds → tmdb indices inside the ratings table
        ratings = ratings.copy()
        ratings["tmdb_idx"] = ratings["movieId"].map(link)
        ratings = ratings.dropna(subset=["tmdb_idx"])
        ratings["tmdb_idx"] = ratings["tmdb_idx"].astype(int)

        # Filter users with enough linked ratings
        user_counts = ratings.groupby("userId").size()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        ratings = ratings[ratings["userId"].isin(valid_users)]

        # Optional user cap
        unique_users = ratings["userId"].unique()
        if max_users is not None and len(unique_users) > max_users:
            rng = np.random.RandomState(42)
            unique_users = rng.choice(unique_users, max_users, replace=False)
            ratings = ratings[ratings["userId"].isin(unique_users)]

        # Re-index users 0..N-1
        user_ids = np.sort(ratings["userId"].unique())
        uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
        n_users = len(user_ids)

        matrix = np.zeros((n_users, n_movies), dtype=np.float32)
        for _, r in ratings.iterrows():
            u = uid_to_row[r["userId"]]
            m = int(r["tmdb_idx"])
            matrix[u, m] = float(r["rating"])

        n_ratings = np.count_nonzero(matrix)
        sparsity = 1 - n_ratings / (n_users * n_movies)

        logger.info(
            f"Real rating matrix: {n_users} users × {n_movies} movies, "
            f"{n_ratings:,} ratings, sparsity {sparsity:.2%}"
        )

        movie_indices = np.where(matrix.sum(axis=0) > 0)[0]

        return matrix, user_ids, movie_indices

    # ------------------------------------------------------------------
    # Train / test split (per-user leave-k-out)
    # ------------------------------------------------------------------
    @staticmethod
    def train_test_split(
        matrix: np.ndarray,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-user random split: for each user, hold out *test_ratio* of
        their ratings into a test matrix.

        Returns (train_matrix, test_matrix) with the same shape.
        """
        rng = np.random.RandomState(seed)
        train = matrix.copy()
        test = np.zeros_like(matrix)

        for u in range(matrix.shape[0]):
            rated = np.where(matrix[u] > 0)[0]
            if len(rated) < 2:
                continue
            n_test = max(1, int(len(rated) * test_ratio))
            test_items = rng.choice(rated, n_test, replace=False)
            test[u, test_items] = matrix[u, test_items]
            train[u, test_items] = 0.0

        logger.info(
            f"Train/test split: train {np.count_nonzero(train):,}, "
            f"test {np.count_nonzero(test):,} ratings"
        )
        return train, test
