"""
Unit tests for MetadataBasedRecommender using current public APIs.
"""

import pandas as pd


class TestMetadataBasedRecommender:
    def test_fit_creates_similarity_matrix(self, metadata_recommender):
        assert metadata_recommender.is_fitted is True
        assert metadata_recommender.similarity_matrix is not None
        assert metadata_recommender.similarity_matrix.shape[0] == metadata_recommender.movie_count

    def test_recommend_returns_dataframe(self, metadata_recommender):
        result = metadata_recommender.recommend("The Matrix", top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5
        assert "title" in result.columns

    def test_recommend_excludes_source_movie(self, metadata_recommender):
        result = metadata_recommender.recommend("The Matrix", top_n=5)
        assert "The Matrix" not in result["title"].tolist()

    def test_get_matching_features(self, metadata_recommender):
        matches = metadata_recommender.get_matching_features("The Matrix", "The Matrix Reloaded")
        assert isinstance(matches, dict)
        assert "common_genres" in matches

    def test_recommend_invalid_movie_returns_none(self, metadata_recommender):
        result = metadata_recommender.recommend("NonexistentMovie123", top_n=5)
        assert result is None
