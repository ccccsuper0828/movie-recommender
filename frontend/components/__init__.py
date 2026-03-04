"""UI components for the Streamlit frontend."""

from .movie_card import MovieCard
from .recommendation_list import RecommendationList
from .search_bar import SearchBar
from .sidebar import Sidebar
from .explanation_panel import ExplanationPanel
from .charts import Charts

__all__ = [
    "MovieCard",
    "RecommendationList",
    "SearchBar",
    "Sidebar",
    "ExplanationPanel",
    "Charts",
]
