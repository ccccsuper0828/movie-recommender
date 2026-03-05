"""
Model Registry — plug-and-play module system for recommenders and predictors.

Usage
-----
# Register a new recommender:
    from src.registry import RECOMMENDER_REGISTRY
    RECOMMENDER_REGISTRY.register("my_algo", MyAlgoRecommender)

# Get a recommender by name:
    cls = RECOMMENDER_REGISTRY.get("my_algo")
    model = cls()
    model.fit(movies_df)

# List available models:
    print(RECOMMENDER_REGISTRY.list())
"""

from typing import Dict, Type, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Generic registry for pluggable model classes."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str, cls: Type) -> None:
        """Register a model class under the given key."""
        self._registry[key] = cls
        logger.debug(f"[{self.name}] Registered: {key} → {cls.__name__}")

    def get(self, key: str) -> Type:
        """Get a registered model class by key."""
        if key not in self._registry:
            raise KeyError(
                f"[{self.name}] '{key}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def list(self) -> Dict[str, str]:
        """List all registered models as {key: class_name}."""
        return {k: v.__name__ for k, v in self._registry.items()}

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# ── Global registries ──

RECOMMENDER_REGISTRY = ModelRegistry("Recommender")
PREDICTOR_REGISTRY = ModelRegistry("Predictor")
