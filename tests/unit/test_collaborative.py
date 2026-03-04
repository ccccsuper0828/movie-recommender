"""
Unit tests for CollaborativeFilteringRecommender using current public APIs.
"""

import pandas as pd


class TestCollaborativeFilteringRecommender:
    def test_fit_creates_item_similarity(self, collaborative_recommender):
        assert collaborative_recommender.is_fitted is True
        assert collaborative_recommender.item_similarity is not None

    def test_fit_stores_user_movie_matrix(self, collaborative_recommender):
        matrix = collaborative_recommender.user_movie_matrix
        assert matrix is not None
        assert matrix.ndim == 2

    def test_recommend_item_based(self, collaborative_recommender):
        result = collaborative_recommender.recommend("The Matrix", top_n=5, method="item_based")
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_recommend_user_based(self, collaborative_recommender):
        result = collaborative_recommender.recommend("The Matrix", top_n=5, method="user_based")
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_recommend_svd(self, collaborative_recommender):
        result = collaborative_recommender.recommend("The Matrix", top_n=5, method="svd")
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_invalid_movie_returns_none(self, collaborative_recommender):
        result = collaborative_recommender.recommend("NonexistentMovie123", top_n=5)
        assert result is None
