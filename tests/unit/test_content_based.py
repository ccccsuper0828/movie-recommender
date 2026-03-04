"""
Unit tests for ContentBasedRecommender using current public APIs.
"""

import pandas as pd


class TestContentBasedRecommender:
    def test_fit_creates_similarity_matrix(self, content_recommender):
        assert content_recommender.is_fitted is True
        assert content_recommender.similarity_matrix is not None
        assert content_recommender.similarity_matrix.shape[0] == content_recommender.movie_count

    def test_recommend_returns_dataframe(self, content_recommender):
        result = content_recommender.recommend("The Matrix", top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5
        assert "title" in result.columns
        assert "similarity_score" in result.columns

    def test_recommend_excludes_source_movie(self, content_recommender):
        result = content_recommender.recommend("The Matrix", top_n=5)
        assert "The Matrix" not in result["title"].tolist()

    def test_recommend_invalid_movie_returns_none(self, content_recommender):
        result = content_recommender.recommend("NonexistentMovie123", top_n=5)
        assert result is None

    def test_get_similarity_score(self, content_recommender):
        score = content_recommender.get_similarity_score("The Matrix", "The Matrix Reloaded")
        assert isinstance(score, float)
        assert 0 <= score <= 1
