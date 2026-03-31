"""
Chart components for data visualization.
Uses Plotly's built-in 'plotly_dark' template for consistent dark theme rendering.
"""
# @author 成员 F — 前端框架 & API & 测试

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


class Charts:
    """Custom chart components with dark theme styling."""

    @staticmethod
    def similarity_radar(scores: Dict[str, float], title: str = "Method Comparison"):
        categories = list(scores.keys())
        values = list(scores.values())
        categories += [categories[0]]
        values += [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself',
            fillcolor='rgba(229,56,59,0.3)',
            line=dict(color='#e5383b', width=2), name='Similarity',
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template=TEMPLATE, title=title, showlegend=False,
        )
        st.plotly_chart(fig, width="stretch", theme=None)

    @staticmethod
    def genre_distribution(genre_counts: Dict[str, int], title: str = "Genre Distribution"):
        sorted_g = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        genres = [g[0] for g in sorted_g]
        counts = [g[1] for g in sorted_g]

        fig = go.Figure(go.Bar(
            y=genres, x=counts, orientation='h',
            marker=dict(color=counts, colorscale='Reds'),
        ))
        fig.update_layout(
            title=title, template=TEMPLATE,
            yaxis=dict(autorange='reversed'), height=400,
        )
        st.plotly_chart(fig, width="stretch", theme=None)

    @staticmethod
    def rating_histogram(ratings: List[float], title: str = "Rating Distribution"):
        fig = go.Figure(go.Histogram(
            x=ratings, nbinsx=20,
            marker=dict(color='#e5383b', line=dict(color='#ba181b', width=1)),
        ))
        fig.update_layout(
            title=title, template=TEMPLATE,
            xaxis=dict(title='Rating', range=[0, 10]),
            yaxis=dict(title='Count'), bargap=0.1,
        )
        st.plotly_chart(fig, width="stretch", theme=None)

    @staticmethod
    def movies_timeline(year_counts: Dict[int, int], title: str = "Movies Per Year"):
        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]

        fig = go.Figure(go.Scatter(
            x=years, y=counts, mode='lines+markers',
            line=dict(color='#f5c518', width=2),
            marker=dict(color='#f5c518', size=5),
            fill='tozeroy', fillcolor='rgba(245,197,24,0.15)',
        ))
        fig.update_layout(
            title=title, template=TEMPLATE,
            xaxis_title='Year', yaxis_title='Number of Movies',
        )
        st.plotly_chart(fig, width="stretch", theme=None)

    @staticmethod
    def feature_importance(features: List[str], importances: List[float],
                           title: str = "Feature Importance"):
        pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:15]
        names = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]

        fig = go.Figure(go.Bar(
            y=names, x=vals, orientation='h',
            marker=dict(color=vals, colorscale='Reds'),
        ))
        fig.update_layout(
            title=title, template=TEMPLATE,
            yaxis=dict(autorange='reversed'), height=400,
        )
        st.plotly_chart(fig, width="stretch", theme=None)

    @staticmethod
    def similarity_heatmap(similarity_matrix: np.ndarray, labels: List[str],
                           title: str = "Similarity Matrix"):
        fig = go.Figure(go.Heatmap(
            z=similarity_matrix, x=labels, y=labels,
            colorscale='RdYlBu_r', showscale=True,
        ))
        fig.update_layout(
            title=title, template=TEMPLATE,
            xaxis=dict(tickangle=45), height=500,
        )
        st.plotly_chart(fig, width="stretch", theme=None)

    @staticmethod
    def comparison_bar(data: Dict[str, Dict[str, float]],
                       title: str = "Method Comparison"):
        movies = list(data.keys())
        methods = list(data[movies[0]].keys()) if movies else []
        colors = ['#e5383b', '#f5c518', '#58a6ff', '#3fb950']

        fig = go.Figure()
        for i, method in enumerate(methods):
            values = [data[m].get(method, 0) for m in movies]
            fig.add_trace(go.Bar(name=method, x=movies, y=values,
                                 marker=dict(color=colors[i % len(colors)])))

        fig.update_layout(
            title=title, template=TEMPLATE,
            barmode='group', yaxis_title='Similarity Score',
        )
        st.plotly_chart(fig, width="stretch", theme=None)
