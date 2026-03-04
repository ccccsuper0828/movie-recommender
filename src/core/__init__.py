"""Core recommendation algorithms package."""

from .base_recommender import BaseRecommender
from .content_based import ContentBasedRecommender
from .metadata_based import MetadataBasedRecommender
from .collaborative import CollaborativeFilteringRecommender
from .hybrid import HybridRecommender
from .demographic import DemographicRecommender
from .knn_svd_ensemble import KNNSVDEnsembleRecommender

__all__ = [
    "BaseRecommender",
    "ContentBasedRecommender",
    "MetadataBasedRecommender",
    "CollaborativeFilteringRecommender",
    "HybridRecommender",
    "DemographicRecommender",
    "KNNSVDEnsembleRecommender",
]
