"""
Chart components for data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go


class Charts:
    """
    Custom chart components with dark theme styling.
    """

    # Dark theme template for Plotly
    DARK_TEMPLATE = {
        'layout': {
            'paper_bgcolor': '#0a0a0f',
            'plot_bgcolor': '#0a0a0f',
            'font': {'color': '#a3a3a3'},
            'title': {'font': {'color': '#ffffff'}},
            'xaxis': {
                'gridcolor': '#2a2a3a',
                'linecolor': '#2a2a3a'
            },
            'yaxis': {
                'gridcolor': '#2a2a3a',
                'linecolor': '#2a2a3a'
            }
        }
    }

    @staticmethod
    def similarity_radar(
        scores: Dict[str, float],
        title: str = "Method Comparison"
    ):
        """
        Create a radar chart for method scores.

        Parameters
        ----------
        scores : dict
            Dictionary of method names to scores
        title : str
            Chart title
        """
        categories = list(scores.keys())
        values = list(scores.values())

        # Close the radar
        categories = categories + [categories[0]]
        values = values + [values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(229, 9, 20, 0.3)',
            line=dict(color='#e50914', width=2),
            name='Similarity'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='#2a2a3a'
                ),
                angularaxis=dict(
                    gridcolor='#2a2a3a'
                ),
                bgcolor='#0a0a0f'
            ),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            title=dict(text=title, font=dict(color='#ffffff')),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def genre_distribution(
        genre_counts: Dict[str, int],
        title: str = "Genre Distribution"
    ):
        """
        Create a horizontal bar chart for genre distribution.

        Parameters
        ----------
        genre_counts : dict
            Dictionary of genre names to counts
        title : str
            Chart title
        """
        # Sort by count
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        genres = [g[0] for g in sorted_genres]
        counts = [g[1] for g in sorted_genres]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=genres,
            x=counts,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale=[[0, '#2a2a3a'], [0.5, '#e50914'], [1, '#f5c518']]
            )
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#ffffff')),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            xaxis=dict(gridcolor='#2a2a3a'),
            yaxis=dict(gridcolor='#2a2a3a', autorange='reversed'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def rating_histogram(
        ratings: List[float],
        title: str = "Rating Distribution"
    ):
        """
        Create a histogram for rating distribution.

        Parameters
        ----------
        ratings : list
            List of ratings
        title : str
            Chart title
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=ratings,
            nbinsx=20,
            marker=dict(
                color='#e50914',
                line=dict(color='#b20710', width=1)
            )
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#ffffff')),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            xaxis=dict(
                title='Rating',
                gridcolor='#2a2a3a',
                range=[0, 10]
            ),
            yaxis=dict(
                title='Count',
                gridcolor='#2a2a3a'
            ),
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def movies_timeline(
        year_counts: Dict[int, int],
        title: str = "Movies Per Year"
    ):
        """
        Create a line chart for movies over time.

        Parameters
        ----------
        year_counts : dict
            Dictionary of year to movie count
        title : str
            Chart title
        """
        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            mode='lines+markers',
            line=dict(color='#f5c518', width=2),
            marker=dict(color='#f5c518', size=6),
            fill='tozeroy',
            fillcolor='rgba(245, 197, 24, 0.2)'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#ffffff')),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            xaxis=dict(
                title='Year',
                gridcolor='#2a2a3a'
            ),
            yaxis=dict(
                title='Number of Movies',
                gridcolor='#2a2a3a'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def feature_importance(
        features: List[str],
        importances: List[float],
        title: str = "Feature Importance"
    ):
        """
        Create a horizontal bar chart for feature importance.

        Parameters
        ----------
        features : list
            Feature names
        importances : list
            Importance values
        title : str
            Chart title
        """
        # Sort by importance
        sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:15]
        features = [p[0] for p in sorted_pairs]
        importances = [p[1] for p in sorted_pairs]

        fig = go.Figure()

        colors = ['#e50914' if i > 0 else '#3498db' for i in importances]

        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(color=colors)
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#ffffff')),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            xaxis=dict(
                title='Importance',
                gridcolor='#2a2a3a'
            ),
            yaxis=dict(
                gridcolor='#2a2a3a',
                autorange='reversed'
            ),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def similarity_heatmap(
        similarity_matrix: np.ndarray,
        labels: List[str],
        title: str = "Similarity Matrix"
    ):
        """
        Create a heatmap for similarity matrix.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Similarity matrix
        labels : list
            Labels for rows/columns
        title : str
            Chart title
        """
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale=[[0, '#0a0a0f'], [0.5, '#e50914'], [1, '#f5c518']],
            showscale=True
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#ffffff')),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            xaxis=dict(tickangle=45),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def comparison_bar(
        data: Dict[str, Dict[str, float]],
        title: str = "Method Comparison"
    ):
        """
        Create a grouped bar chart for method comparison.

        Parameters
        ----------
        data : dict
            Dictionary of movie -> {method: score}
        title : str
            Chart title
        """
        movies = list(data.keys())
        methods = list(data[movies[0]].keys()) if movies else []

        fig = go.Figure()

        colors = ['#e50914', '#f5c518', '#3498db', '#2ecc71']

        for i, method in enumerate(methods):
            values = [data[movie].get(method, 0) for movie in movies]
            fig.add_trace(go.Bar(
                name=method,
                x=movies,
                y=values,
                marker=dict(color=colors[i % len(colors)])
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#ffffff')),
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#a3a3a3'),
            barmode='group',
            xaxis=dict(gridcolor='#2a2a3a'),
            yaxis=dict(
                title='Similarity Score',
                gridcolor='#2a2a3a'
            ),
            legend=dict(
                bgcolor='rgba(26, 26, 36, 0.8)',
                bordercolor='#2a2a3a'
            )
        )

        st.plotly_chart(fig, use_container_width=True)
