"""
Sidebar component for settings and navigation.
"""
# @author 成员 F — 前端框架 & API & 测试

import streamlit as st
from typing import Dict, Tuple, Optional


class Sidebar:
    """
    Custom sidebar component for the movie recommender.
    """

    @staticmethod
    def render_settings(
        default_top_n: int = 10,
        default_weights: Tuple[float, float, float] = (0.3, 0.4, 0.3)
    ) -> Dict:
        """
        Render the settings sidebar.

        Parameters
        ----------
        default_top_n : int
            Default number of recommendations
        default_weights : tuple
            Default weights for (content, metadata, cf)

        Returns
        -------
        dict
            Settings values
        """
        with st.sidebar:
            # Logo/Brand
            st.markdown(
                '<div style="text-align:center;padding:20px 0 30px 0;border-bottom:1px solid #30363d;margin-bottom:20px;">'
                '<h1 style="color:#e5383b;font-size:1.8rem;margin:0;font-weight:700;">🎬 CineMatch</h1>'
                '<p style="color:#666;font-size:0.85rem;margin-top:5px;">Intelligent Movie Recommendations</p>'
                '</div>',
                unsafe_allow_html=True
            )

            # Settings section
            st.markdown("### ⚙️ Settings")

            # Number of recommendations
            top_n = st.slider(
                "Number of recommendations",
                min_value=5,
                max_value=20,
                value=default_top_n,
                key="settings_top_n"
            )

            st.markdown("---")

            # Method weights
            st.markdown("### ⚖️ Hybrid Weights")

            content_weight = st.slider(
                "Content-Based",
                min_value=0.0,
                max_value=1.0,
                value=default_weights[0],
                step=0.05,
                key="settings_content_weight"
            )

            metadata_weight = st.slider(
                "Metadata-Based",
                min_value=0.0,
                max_value=1.0,
                value=default_weights[1],
                step=0.05,
                key="settings_metadata_weight"
            )

            cf_weight = st.slider(
                "Collaborative Filtering",
                min_value=0.0,
                max_value=1.0,
                value=default_weights[2],
                step=0.05,
                key="settings_cf_weight"
            )

            # Normalize weights
            total = content_weight + metadata_weight + cf_weight
            if total > 0:
                weights = (
                    content_weight / total,
                    metadata_weight / total,
                    cf_weight / total
                )
            else:
                weights = (0.33, 0.34, 0.33)

            # Display normalized weights
            st.markdown(
                f'<div style="background:#1c2333;padding:12px;border-radius:8px;margin-top:10px;">'
                f'<p style="color:#666;font-size:0.8rem;margin:0;">Normalized:</p>'
                f'<p style="color:#fff;font-size:0.9rem;margin:5px 0 0 0;">'
                f'{weights[0]:.0%} | {weights[1]:.0%} | {weights[2]:.0%}</p></div>',
                unsafe_allow_html=True
            )

            st.markdown("---")

            # Statistics
            st.markdown("### 📊 Statistics")

            return {
                'top_n': top_n,
                'weights': weights,
                'content_weight': content_weight,
                'metadata_weight': metadata_weight,
                'cf_weight': cf_weight
            }

    @staticmethod
    def render_stats(
        movie_count: int,
        user_count: int = 500
    ):
        """
        Render statistics in the sidebar.

        Parameters
        ----------
        movie_count : int
            Total number of movies
        user_count : int
            Number of simulated users
        """
        with st.sidebar:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Movies", f"{movie_count:,}")

            with col2:
                st.metric("Users", f"{user_count:,}")

    @staticmethod
    def render_navigation() -> str:
        """
        Render navigation links.

        Returns
        -------
        str
            Selected page
        """
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🧭 Navigation")

            pages = {
                "🏠 Home": "home",
                "🎯 Get Recommendations": "recommendations",
                "🔍 Explore": "explore",
                "📊 Analytics": "analytics",
                "💰 Box Office Prediction": "box_office",
                "⚖️ Compare Methods": "compare",
                "ℹ️ About": "about"
            }

            selected = st.radio(
                "Go to",
                options=list(pages.keys()),
                key="nav_radio",
                label_visibility="collapsed"
            )

            return pages[selected]

    @staticmethod
    def render_about():
        """Render about information in sidebar."""
        with st.sidebar:
            st.markdown("---")
            st.markdown(
                '<div style="padding:15px;background:#1c2333;border-radius:8px;margin-top:20px;">'
                '<h4 style="color:#fff;margin:0 0 10px 0;">About CineMatch</h4>'
                '<p style="color:#a3a3a3;font-size:0.85rem;margin:0;line-height:1.5;">'
                'Powered by 3 ML methods:<br>'
                '• TF-IDF Content Analysis<br>'
                '• Metadata Similarity<br>'
                '• Collaborative Filtering</p>'
                '<p style="color:#666;font-size:0.75rem;margin-top:10px;">Dataset: TMDB 5000 Movies</p>'
                '</div>',
                unsafe_allow_html=True
            )
