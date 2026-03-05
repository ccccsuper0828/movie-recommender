"""Explainability modules for recommendation explanations."""

from .rule_based import RuleBasedExplainer

# SHAP/visualization may be unavailable in some environments. Keep imports optional
# so core services can still run and tests can exercise non-SHAP functionality.
try:
from .shap_explainer import SHAPExplainer
except Exception:  # pragma: no cover - environment-dependent optional dependency
    SHAPExplainer = None

try:
from .visualization import ExplanationVisualizer
except Exception:  # pragma: no cover - environment-dependent optional dependency
    ExplanationVisualizer = None

__all__ = [
    "RuleBasedExplainer",
    "SHAPExplainer",
    "ExplanationVisualizer",
]
