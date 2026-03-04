"""
Central configuration settings for the Movie Recommendation System.
Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Movie Recommender"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Optional[Path] = Field(default=None)
    cache_dir: Optional[Path] = Field(default=None)

    # Data files
    movies_file: str = "tmdb_5000_movies.csv"
    credits_file: str = "tmdb_5000_credits.csv"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    # Model parameters
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    count_vectorizer_max_features: int = 10000

    # Collaborative filtering
    cf_n_users: int = 1000
    cf_sparsity: float = 0.02
    svd_n_factors: int = 50
    knn_n_neighbors: int = 20

    # Hybrid weights (content, metadata, cf)
    hybrid_weights: tuple[float, float, float] = (0.3, 0.4, 0.3)

    # Recommendations
    default_top_n: int = 10
    max_top_n: int = 50

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Frontend
    streamlit_port: int = 8501
    theme_primary_color: str = "#e50914"
    theme_secondary_color: str = "#f5c518"
    theme_background_color: str = "#0a0a0f"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def model_post_init(self, __context):
        """Initialize derived paths after model creation."""
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.cache_dir is None:
            self.cache_dir = self.base_dir / "data" / "cache"

    @property
    def movies_path(self) -> Path:
        """Full path to movies CSV file."""
        return self.data_dir / "raw" / self.movies_file

    @property
    def credits_path(self) -> Path:
        """Full path to credits CSV file."""
        return self.data_dir / "raw" / self.credits_file

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed data."""
        return self.data_dir / "processed"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access to settings
settings = get_settings()
