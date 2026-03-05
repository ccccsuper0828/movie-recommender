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

# ── Register all recommenders in the global registry ──
from src.registry import RECOMMENDER_REGISTRY

RECOMMENDER_REGISTRY.register("content", ContentBasedRecommender)
RECOMMENDER_REGISTRY.register("metadata", MetadataBasedRecommender)
RECOMMENDER_REGISTRY.register("collaborative", CollaborativeFilteringRecommender)
RECOMMENDER_REGISTRY.register("hybrid", HybridRecommender)
RECOMMENDER_REGISTRY.register("demographic", DemographicRecommender)
RECOMMENDER_REGISTRY.register("knn_svd", KNNSVDEnsembleRecommender)
