"""
Dependency injection for FastAPI routes.
"""

from functools import lru_cache
from typing import Generator
import logging

from src.services.recommendation_service import RecommendationService
from src.services.user_service import UserService
from src.services.search_service import SearchService
from src.services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


# Singleton instances
_recommendation_service: RecommendationService = None
_user_service: UserService = None


@lru_cache()
def get_recommendation_service() -> RecommendationService:
    """Get the recommendation service (singleton)."""
    global _recommendation_service

    if _recommendation_service is None:
        logger.info("Initializing RecommendationService...")
        _recommendation_service = RecommendationService(auto_init=True)

    return _recommendation_service


@lru_cache()
def get_user_service() -> UserService:
    """Get the user service (singleton)."""
    global _user_service

    if _user_service is None:
        _user_service = UserService()

    return _user_service


def get_search_service() -> SearchService:
    """Get the search service."""
    rec_service = get_recommendation_service()
    return SearchService(rec_service._movies_df)


def get_analytics_service() -> AnalyticsService:
    """Get the analytics service."""
    rec_service = get_recommendation_service()
    return AnalyticsService(rec_service._movies_df)
