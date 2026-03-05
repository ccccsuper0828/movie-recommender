"""
Abstract base class for all box office / revenue predictors.

To add a new predictor:
  1. Create a new file in src/prediction/
  2. Subclass BasePredictor and implement fit() / predict()
  3. Register it in src/prediction/__init__.py:
       PREDICTOR_REGISTRY.register("my_model", MyPredictor)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract interface that every revenue predictor must implement."""

    def __init__(self, name: str = "BasePredictor"):
        self.name = name
        self._is_fitted = False
        self.cv_results: Optional[Dict[str, Any]] = None
        self.fold_results: List[Dict] = []

    @abstractmethod
    def fit(self, df: Optional[pd.DataFrame] = None) -> "BasePredictor":
        """Train the predictor. Returns self."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict revenue (dollars) for each row. Returns np.ndarray."""
        ...

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame (optional)."""
        return pd.DataFrame(columns=["feature", "importance"])

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Optional[Path] = None) -> "BasePredictor":
        """Load a saved model from disk."""
        ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
