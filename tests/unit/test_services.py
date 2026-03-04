"""
Unit tests for service layer using current APIs.
"""

from pathlib import Path

import pandas as pd


class TestSearchService:
    def test_search_returns_paginated_payload(self, sample_movies_df):
        from src.services import SearchService

        service = SearchService(sample_movies_df)
        result = service.search("Matrix", page=1, page_size=10)

        assert isinstance(result, dict)
        assert "results" in result
        assert result["total_results"] >= 1

    def test_get_suggestions(self, sample_movies_df):
        from src.services import SearchService

        service = SearchService(sample_movies_df)
        suggestions = service.get_suggestions("Mat", max_results=5)
        assert isinstance(suggestions, list)


class TestUserService:
    def test_create_get_and_rate_user(self, temp_data_dir):
        from src.services import UserService

        storage_path = Path(temp_data_dir) / "users.json"
        service = UserService(storage_path=storage_path)
        user = service.create_user("testuser", "test@example.com")

        assert user.id > 0
        assert service.get_user(user.id) is not None

        updated = service.add_rating(user.id, movie_id=1, rating=4.5)
        assert updated is not None
        assert len(service.get_user_ratings(user.id)) == 1


class TestAnalyticsService:
    def test_get_overview(self, sample_movies_df):
        from src.services import AnalyticsService

        service = AnalyticsService(sample_movies_df)
        metrics = service.get_overview()

        assert isinstance(metrics, dict)
        assert metrics["total_movies"] == len(sample_movies_df)
        assert "rating_distribution" in metrics


class TestRecommendationService:
    def test_initialize_and_get_recommendations(self):
        from src.services import RecommendationService

        service = RecommendationService(auto_init=True)
        result = service.get_recommendations("The Matrix", top_n=5, method="hybrid")

        assert isinstance(result, dict)
        assert "recommendations" in result
        assert result["source_movie"] == "The Matrix"
