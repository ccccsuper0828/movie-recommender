"""
Analytics visualizations for the movie recommendation system.
"""
# @author 成员 D — 票房预测 & 数据可视化

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AnalyticsVisualizations:
    """
    Visualization components for analytics dashboard.

    Uses Plotly for interactive charts with a consistent dark theme.
    """

    # Dark theme configuration
    THEME = {
        'paper_bgcolor': '#0a0a0f',
        'plot_bgcolor': '#0a0a0f',
        'font_color': '#a3a3a3',
        'title_color': '#ffffff',
        'grid_color': '#2a2a3a',
        'primary_color': '#e50914',
        'secondary_color': '#f5c518',
        'accent_colors': ['#e50914', '#f5c518', '#3498db', '#2ecc71', '#9b59b6']
    }

    @classmethod
    def _apply_theme(cls, fig: go.Figure) -> go.Figure:
        """Apply dark theme to a figure."""
        fig.update_layout(
            paper_bgcolor=cls.THEME['paper_bgcolor'],
            plot_bgcolor=cls.THEME['plot_bgcolor'],
            font=dict(color=cls.THEME['font_color']),
            title=dict(font=dict(color=cls.THEME['title_color'])),
            xaxis=dict(gridcolor=cls.THEME['grid_color']),
            yaxis=dict(gridcolor=cls.THEME['grid_color']),
            legend=dict(
                bgcolor='rgba(26, 26, 36, 0.8)',
                bordercolor=cls.THEME['grid_color']
            )
        )
        return fig

    @classmethod
    def genre_distribution_chart(
        cls,
        genre_counts: Dict[str, int],
        title: str = "Genre Distribution"
    ) -> go.Figure:
        """
        Create a horizontal bar chart for genre distribution.

        Parameters
        ----------
        genre_counts : dict
            Genre name to count mapping
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        sorted_genres = sorted(
            genre_counts.items(), key=lambda x: x[1], reverse=True
        )[:15]

        genres = [g[0] for g in sorted_genres]
        counts = [g[1] for g in sorted_genres]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=genres,
            x=counts,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale=[
                    [0, '#2a2a3a'],
                    [0.5, cls.THEME['primary_color']],
                    [1, cls.THEME['secondary_color']]
                ]
            )
        ))

        fig.update_layout(
            title=title,
            yaxis=dict(autorange='reversed'),
            height=450
        )

        return cls._apply_theme(fig)

    @classmethod
    def rating_distribution_chart(
        cls,
        ratings: List[float],
        title: str = "Rating Distribution"
    ) -> go.Figure:
        """
        Create a histogram for rating distribution.

        Parameters
        ----------
        ratings : list
            List of ratings
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=ratings,
            nbinsx=20,
            marker=dict(
                color=cls.THEME['primary_color'],
                line=dict(color='#b20710', width=1)
            )
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Rating', range=[0, 10]),
            yaxis=dict(title='Count'),
            bargap=0.1,
            height=400
        )

        return cls._apply_theme(fig)

    @classmethod
    def movies_timeline_chart(
        cls,
        year_counts: Dict[int, int],
        title: str = "Movies Per Year"
    ) -> go.Figure:
        """
        Create a line chart for movies over time.

        Parameters
        ----------
        year_counts : dict
            Year to count mapping
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            mode='lines+markers',
            line=dict(color=cls.THEME['secondary_color'], width=2),
            marker=dict(color=cls.THEME['secondary_color'], size=6),
            fill='tozeroy',
            fillcolor='rgba(245, 197, 24, 0.2)'
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Year'),
            yaxis=dict(title='Number of Movies'),
            height=400
        )

        return cls._apply_theme(fig)

    @classmethod
    def method_comparison_chart(
        cls,
        method_stats: Dict[str, Dict[str, Any]],
        title: str = "Recommendation Method Comparison"
    ) -> go.Figure:
        """
        Create a grouped bar chart for method comparison.

        Parameters
        ----------
        method_stats : dict
            Method statistics
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        methods = list(method_stats.keys())

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Request Count', 'Avg Response Time (ms)')
        )

        # Request counts
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[method_stats[m]['total_requests'] for m in methods],
                marker=dict(color=cls.THEME['accent_colors'][:len(methods)]),
                name='Requests'
            ),
            row=1, col=1
        )

        # Response times
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[method_stats[m]['avg_response_time_ms'] for m in methods],
                marker=dict(color=cls.THEME['accent_colors'][:len(methods)]),
                name='Response Time',
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=title,
            showlegend=False,
            height=400
        )

        return cls._apply_theme(fig)

    @classmethod
    def method_usage_pie_chart(
        cls,
        method_stats: Dict[str, Dict[str, Any]],
        title: str = "Method Usage Distribution"
    ) -> go.Figure:
        """
        Create a pie chart for method usage distribution.

        Parameters
        ----------
        method_stats : dict
            Method statistics
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        methods = list(method_stats.keys())
        values = [method_stats[m]['total_requests'] for m in methods]

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=methods,
            values=values,
            marker=dict(colors=cls.THEME['accent_colors'][:len(methods)]),
            hole=0.4,
            textinfo='label+percent'
        ))

        fig.update_layout(
            title=title,
            height=400
        )

        return cls._apply_theme(fig)

    @classmethod
    def response_time_trend_chart(
        cls,
        daily_stats: Dict[str, Dict],
        title: str = "Response Time Trend"
    ) -> go.Figure:
        """
        Create a line chart for response time trends.

        Parameters
        ----------
        daily_stats : dict
            Daily statistics
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        dates = sorted(daily_stats.keys())
        response_times = [
            daily_stats[d].get('avg_response_time_ms', 0) for d in dates
        ]
        requests = [
            daily_stats[d].get('total_requests', 0) for d in dates
        ]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Response time line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=response_times,
                mode='lines+markers',
                name='Avg Response Time (ms)',
                line=dict(color=cls.THEME['primary_color'], width=2)
            ),
            secondary_y=False
        )

        # Request count bars
        fig.add_trace(
            go.Bar(
                x=dates,
                y=requests,
                name='Requests',
                marker=dict(color=cls.THEME['secondary_color'], opacity=0.5)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=title,
            xaxis=dict(title='Date'),
            height=400
        )

        fig.update_yaxes(title_text="Response Time (ms)", secondary_y=False)
        fig.update_yaxes(title_text="Request Count", secondary_y=True)

        return cls._apply_theme(fig)

    @classmethod
    def director_performance_chart(
        cls,
        directors: List[tuple],
        ratings: Dict[str, float],
        title: str = "Top Directors Analysis"
    ) -> go.Figure:
        """
        Create a scatter chart for director performance.

        Parameters
        ----------
        directors : list
            List of (director, movie_count) tuples
        ratings : dict
            Director to average rating mapping
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        director_names = [d[0] for d in directors]
        movie_counts = [d[1] for d in directors]
        avg_ratings = [ratings.get(d[0], 0) for d in directors]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=movie_counts,
            y=avg_ratings,
            mode='markers+text',
            text=[d[:15] + '...' if len(d) > 15 else d for d in director_names],
            textposition='top center',
            marker=dict(
                size=12,
                color=avg_ratings,
                colorscale=[
                    [0, '#2a2a3a'],
                    [0.5, cls.THEME['primary_color']],
                    [1, cls.THEME['secondary_color']]
                ],
                showscale=True,
                colorbar=dict(title='Avg Rating')
            )
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Number of Movies'),
            yaxis=dict(title='Average Rating'),
            height=500
        )

        return cls._apply_theme(fig)

    @classmethod
    def genre_rating_heatmap(
        cls,
        genre_data: Dict[str, Dict[str, float]],
        title: str = "Genre Performance Matrix"
    ) -> go.Figure:
        """
        Create a heatmap for genre performance.

        Parameters
        ----------
        genre_data : dict
            Genre statistics
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        genres = list(genre_data.keys())[:15]
        metrics = ['average_rating', 'movie_count', 'avg_popularity']

        # Normalize values for heatmap
        z_values = []
        for metric in metrics:
            row = []
            for genre in genres:
                val = genre_data.get(genre, {}).get(metric, 0)
                row.append(val)
            # Normalize row
            max_val = max(row) if row else 1
            row = [v / max_val if max_val > 0 else 0 for v in row]
            z_values.append(row)

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=z_values,
            x=genres,
            y=['Avg Rating', 'Movie Count', 'Popularity'],
            colorscale=[
                [0, '#0a0a0f'],
                [0.5, cls.THEME['primary_color']],
                [1, cls.THEME['secondary_color']]
            ],
            showscale=True
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=45),
            height=350
        )

        return cls._apply_theme(fig)

    @classmethod
    def user_engagement_chart(
        cls,
        engagement_data: Dict[str, Any],
        title: str = "User Engagement Metrics"
    ) -> go.Figure:
        """
        Create a chart for user engagement metrics.

        Parameters
        ----------
        engagement_data : dict
            User engagement data
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        interaction_types = engagement_data.get('interaction_types', {})

        if not interaction_types:
            fig = go.Figure()
            fig.add_annotation(
                text="No engagement data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=cls.THEME['font_color'])
            )
            fig.update_layout(height=300)
            return cls._apply_theme(fig)

        types = list(interaction_types.keys())
        counts = list(interaction_types.values())

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=types,
            y=counts,
            marker=dict(color=cls.THEME['accent_colors'][:len(types)])
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Interaction Type'),
            yaxis=dict(title='Count'),
            height=400
        )

        return cls._apply_theme(fig)

    @classmethod
    def peak_hours_chart(
        cls,
        peak_hours: Dict[int, int],
        title: str = "Peak Usage Hours"
    ) -> go.Figure:
        """
        Create a chart for peak usage hours.

        Parameters
        ----------
        peak_hours : dict
            Hour to count mapping
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        # Fill in all 24 hours
        all_hours = {h: peak_hours.get(h, 0) for h in range(24)}
        hours = list(all_hours.keys())
        counts = list(all_hours.values())

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=counts,
            marker=dict(
                color=counts,
                colorscale=[
                    [0, '#2a2a3a'],
                    [0.5, cls.THEME['primary_color']],
                    [1, cls.THEME['secondary_color']]
                ]
            )
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Hour of Day'),
            yaxis=dict(title='Requests'),
            height=350
        )

        return cls._apply_theme(fig)

    @classmethod
    def system_health_gauge(
        cls,
        health_data: Dict[str, Any],
        title: str = "System Health"
    ) -> go.Figure:
        """
        Create a gauge chart for system health.

        Parameters
        ----------
        health_data : dict
            System health data
        title : str
            Chart title

        Returns
        -------
        go.Figure
            Plotly figure
        """
        avg_response = health_data.get('avg_response_time_ms', 0)

        # Map response time to health score (0-100)
        if avg_response < 100:
            health_score = 100 - (avg_response / 2)
        elif avg_response < 500:
            health_score = 50 - ((avg_response - 100) / 20)
        else:
            health_score = max(0, 30 - ((avg_response - 500) / 50))

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            title=dict(text="Health Score", font=dict(color=cls.THEME['title_color'])),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=cls.THEME['font_color']
                ),
                bar=dict(color=cls.THEME['primary_color']),
                bgcolor='#1a1a24',
                bordercolor=cls.THEME['grid_color'],
                steps=[
                    dict(range=[0, 30], color='#e74c3c'),
                    dict(range=[30, 70], color='#f39c12'),
                    dict(range=[70, 100], color='#2ecc71')
                ],
                threshold=dict(
                    line=dict(color=cls.THEME['secondary_color'], width=4),
                    thickness=0.75,
                    value=health_score
                )
            )
        ))

        fig.update_layout(
            height=300
        )

        return cls._apply_theme(fig)

    @classmethod
    def create_dashboard_layout(
        cls,
        movies_df: pd.DataFrame,
        analytics_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """
        Create all charts for the analytics dashboard.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe
        analytics_data : dict
            Analytics data from AnalyticsDashboard

        Returns
        -------
        dict
            Dictionary of chart names to figures
        """
        charts = {}

        # Genre distribution
        genre_counts = analytics_data.get('genre_analytics', {}).get('counts', {})
        if genre_counts:
            charts['genre_distribution'] = cls.genre_distribution_chart(genre_counts)

        # Rating distribution
        ratings = movies_df['vote_average'].dropna().tolist()
        if ratings:
            charts['rating_distribution'] = cls.rating_distribution_chart(ratings)

        # Movies timeline
        temporal = analytics_data.get('temporal_analytics', {})
        year_counts = temporal.get('movies_per_year', {})
        if year_counts:
            charts['movies_timeline'] = cls.movies_timeline_chart(year_counts)

        # Method comparison
        method_stats = analytics_data.get('method_comparison', {})
        if method_stats:
            charts['method_comparison'] = cls.method_comparison_chart(method_stats)
            charts['method_usage'] = cls.method_usage_pie_chart(method_stats)

        # System health
        health_data = analytics_data.get('overview', {}).get('system', {})
        if health_data:
            charts['system_health'] = cls.system_health_gauge(health_data)

        # User engagement
        engagement = analytics_data.get('user_engagement', {})
        if engagement:
            charts['user_engagement'] = cls.user_engagement_chart(engagement)

        # Peak hours
        peak_hours = analytics_data.get('recommendation_insights', {}).get('peak_hours', {})
        if peak_hours:
            charts['peak_hours'] = cls.peak_hours_chart(peak_hours)

        return charts
