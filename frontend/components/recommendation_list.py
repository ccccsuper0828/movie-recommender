"""
Recommendation list component.
"""
# @author 成员 F — 前端框架 & API & 测试

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional

from .movie_card import MovieCard


class RecommendationList:
    """
    Component for displaying a list of movie recommendations.
    """

    @staticmethod
    def render(
        recommendations: pd.DataFrame,
        source_movie: str,
        method: str = "hybrid",
        show_explanation: bool = True
    ):
        """
        Render a list of recommendations.

        Parameters
        ----------
        recommendations : pd.DataFrame
            DataFrame with recommendations
        source_movie : str
            Source movie title
        method : str
            Recommendation method used
        show_explanation : bool
            Whether to show explanations
        """
        if recommendations is None or len(recommendations) == 0:
            st.warning("No recommendations found.")
            return

        # Header
        method_labels = {
            'content': 'Content-Based (TF-IDF)',
            'metadata': 'Metadata-Based',
            'cf': 'Collaborative Filtering',
            'hybrid': 'Hybrid'
        }

        st.markdown(
            f'<div style="background:#161b22;padding:16px 20px;border-radius:12px;margin-bottom:20px;border:1px solid #30363d;">'
            f'<p style="color:#a3a3a3;margin:0 0 4px 0;font-size:0.9rem;">Recommendations based on</p>'
            f'<h3 style="color:#ffffff;margin:0;">{source_movie}</h3>'
            f'<p style="color:#e5383b;margin:8px 0 0 0;font-size:0.85rem;">Method: {method_labels.get(method, method)}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Render each recommendation
        for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
            MovieCard.render(
                title=row['title'],
                genres=row.get('genres_list', []),
                rating=row.get('vote_average', 0),
                similarity_score=row.get('similarity_score', row.get('hybrid_score', None)),
                director=row.get('director'),
                year=str(row.get('release_date', ''))[:4] if row.get('release_date') else None,
                overview=row.get('overview'),
                explanation=row.get('explanation'),
                rank=rank,
                show_explanation=show_explanation
            )

    @staticmethod
    def render_comparison(
        comparisons: Dict[str, pd.DataFrame],
        source_movie: str
    ):
        """
        Render a side-by-side comparison of methods.

        Parameters
        ----------
        comparisons : dict
            Dictionary of method name to recommendations DataFrame
        source_movie : str
            Source movie title
        """
        st.markdown(
            f'<h2 style="color:#ffffff;margin-bottom:20px;">Method Comparison for &quot;{source_movie}&quot;</h2>',
            unsafe_allow_html=True
        )

        # Create columns for each method
        methods = list(comparisons.keys())
        cols = st.columns(len(methods))

        method_labels = {
            'content_based': 'Content-Based',
            'metadata_based': 'Metadata-Based',
            'collaborative_filtering': 'Collaborative Filtering',
            'hybrid': 'Hybrid'
        }

        method_colors = {
            'content_based': '#3498db',
            'metadata_based': '#2ecc71',
            'collaborative_filtering': '#9b59b6',
            'hybrid': '#e5383b'
        }

        for col, method in zip(cols, methods):
            with col:
                color = method_colors.get(method, '#e5383b')
                label = method_labels.get(method, method)

                st.markdown(
                    f'<div style="background:#1c2333;padding:16px;border-radius:12px;margin-bottom:16px;border-top:3px solid {color};">'
                    f'<h4 style="color:{color};margin:0;">{label}</h4></div>',
                    unsafe_allow_html=True
                )

                recs = comparisons[method]
                if recs is not None and len(recs) > 0:
                    for _, row in recs.head(5).iterrows():
                        MovieCard.render_compact(
                            title=row['title'],
                            similarity_score=row.get('similarity_score', row.get('hybrid_score', 0)),
                            rating=row.get('vote_average', 0)
                        )
                else:
                    st.info("No recommendations")

    @staticmethod
    def render_grid(
        movies: List[Dict[str, Any]],
        columns: int = 3
    ):
        """
        Render movies in a grid layout.

        Parameters
        ----------
        movies : list
            List of movie dictionaries
        columns : int
            Number of columns
        """
        cols = st.columns(columns)

        for i, movie in enumerate(movies):
            with cols[i % columns]:
                MovieCard.render(
                    title=movie.get('title', ''),
                    genres=movie.get('genres', []),
                    rating=movie.get('vote_average', 0),
                    similarity_score=movie.get('similarity_score'),
                    show_explanation=False
                )
