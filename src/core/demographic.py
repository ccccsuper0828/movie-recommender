"""
Demographic recommender using IMDB weighted rating formula.

Reference: Movie-Analysis project §4.1 — "基于人口统计学的推荐"
  score = (v / (v + m)) * r + (m / (m + v)) * c
where v = vote_count, m = min_votes (90th percentile), r = vote_average, c = global mean.
"""
# @author 成员 B — 基础推荐算法 & 工具库

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class DemographicRecommender(BaseRecommender):
    """
    Non-personalised recommender that ranks movies by the IMDB weighted
    rating formula.  Useful as a baseline / cold-start fallback.
    """

    def __init__(self, quantile: float = 0.90):
        super().__init__(name="DemographicRecommender")
        self.quantile = quantile
        self._min_votes: float = 0
        self._global_mean: float = 0
        self._scored_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fit(self, movies_df: pd.DataFrame, **kwargs) -> "DemographicRecommender":
        logger.info("Fitting DemographicRecommender …")
        self._create_index_mappings(movies_df)

        self._global_mean = float(self._movies_df["vote_average"].mean())
        self._min_votes = float(
            self._movies_df["vote_count"].quantile(self.quantile)
        )

        # Compute weighted score for every movie
        c = self._global_mean
        m = self._min_votes
        v = self._movies_df["vote_count"].values.astype(float)
        r = self._movies_df["vote_average"].values.astype(float)

        scores = (v / (v + m)) * r + (m / (v + m)) * c
        self._movies_df = self._movies_df.copy()
        self._movies_df["demo_score"] = scores

        # Build a trivial "similarity" matrix: sim(i,j) = score_j
        # so that recommend() can reuse the base class pattern.
        n = len(self._movies_df)
        self._similarity_matrix = np.tile(scores, (n, 1))

        self._is_fitted = True
        logger.info(
            f"DemographicRecommender: min_votes={m:.0f}, global_mean={c:.2f}"
        )
        return self

    # ------------------------------------------------------------------
    def recommend(self, title: str = "", top_n: int = 10, **kwargs) -> Optional[pd.DataFrame]:
        """
        Return the top-n globally best-scored movies.
        *title* is accepted for API compatibility but ignored
        (this method is non-personalised).
        """
        if not self._validate_fitted():
            return None

        ranked = self._movies_df.nlargest(top_n, "demo_score")
        out = ranked[["title", "genres_list", "vote_average",
                       "vote_count", "popularity", "demo_score"]].copy()
        out = out.rename(columns={"demo_score": "similarity_score"})
        return out

    # ------------------------------------------------------------------
    def get_top_movies(
        self,
        genre: Optional[str] = None,
        min_votes: Optional[int] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Flexible query: top movies optionally filtered by genre and/or
        minimum vote count.
        """
        if not self._validate_fitted():
            return pd.DataFrame()

        df = self._movies_df.copy()
        if min_votes is not None:
            df = df[df["vote_count"] >= min_votes]
        if genre is not None:
            df = df[df["genres_list"].apply(
                lambda x: genre in x if isinstance(x, list) else False
            )]
        return df.nlargest(top_n, "demo_score")[
            ["title", "genres_list", "vote_average", "vote_count", "demo_score"]
        ]
