"""Explainability modules for recommendation explanations."""

from .rule_based import RuleBasedExplainer

# SHAP/visualization may be unavailable in some environments.
try:
    from .shap_explainer import SHAPExplainer
except Exception:
    SHAPExplainer = None

try:
    from .visualization import ExplanationVisualizer
except Exception:
    ExplanationVisualizer = None

__all__ = [
    "RuleBasedExplainer",
    "SHAPExplainer",
    "ExplanationVisualizer",
]
