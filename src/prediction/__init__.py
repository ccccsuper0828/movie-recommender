"""Box office prediction module."""

from .base_predictor import BasePredictor
from .box_office_predictor import BoxOfficePredictor

__all__ = ["BasePredictor", "BoxOfficePredictor"]

# ── Register all predictors in the global registry ──
from src.registry import PREDICTOR_REGISTRY

PREDICTOR_REGISTRY.register("lgb_xgb_cat", BoxOfficePredictor)
