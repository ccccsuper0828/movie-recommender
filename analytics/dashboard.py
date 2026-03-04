"""
Analytics dashboard for the movie recommendation system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from analytics.metrics_tracker import MetricsTracker


class AnalyticsDashboard:
    """
    Analytics dashboard providing insights into recommendation system performance.

    Provides:
    - Usage statistics
    - Model performance comparisons
    - User engagement metrics
    - System health monitoring
    """

    def __init__(self, metrics_tracker: Optional[MetricsTracker] = None):
        """
        Initialize analytics dashboard.

        Parameters
        ----------
        metrics_tracker : MetricsTracker, optional
            Metrics tracker instance
        """
        self.metrics_tracker = metrics_tracker or MetricsTracker()

    def get_overview_stats(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get overview statistics for the dashboard.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe

        Returns
        -------
        dict
            Overview statistics
        """
        # Dataset stats
        total_movies = len(movies_df)
        avg_rating = movies_df['vote_average'].mean()

        # Genre stats
        all_genres = set()
        for genres in movies_df.get('genres_list', []):
            if isinstance(genres, list):
                all_genres.update(genres)

        # Director stats
        unique_directors = movies_df['director'].dropna().nunique()

        # System stats
        system_health = self.metrics_tracker.get_system_health()

        return {
            'dataset': {
                'total_movies': total_movies,
                'average_rating': round(avg_rating, 2),
                'total_genres': len(all_genres),
                'unique_directors': unique_directors
            },
            'system': system_health,
            'usage': {
                'total_requests': len(self.metrics_tracker.recommendation_events),
                'unique_users': len(set(
                    e.user_id for e in self.metrics_tracker.recommendation_events
                    if e.user_id
                ))
            }
        }

    def get_method_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison of recommendation methods.

        Returns
        -------
        dict
            Method comparison statistics
        """
        methods = ['content', 'metadata', 'cf', 'hybrid']
        comparison = {}

        for method in methods:
            stats = self.metrics_tracker.get_method_stats(method)
            comparison[method] = {
                'total_requests': stats['total_requests'],
                'avg_response_time_ms': round(stats['avg_response_time_ms'], 2),
                'usage_percentage': 0  # Calculated below
            }

        # Calculate usage percentages
        total_requests = sum(c['total_requests'] for c in comparison.values())
        if total_requests > 0:
            for method in methods:
                comparison[method]['usage_percentage'] = round(
                    comparison[method]['total_requests'] / total_requests * 100, 1
                )

        return comparison

    def get_genre_analytics(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get genre-related analytics.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe

        Returns
        -------
        dict
            Genre analytics
        """
        genre_counts = {}
        genre_ratings = {}
        genre_popularity = {}

        for _, row in movies_df.iterrows():
            genres = row.get('genres_list', [])
            if not isinstance(genres, list):
                continue

            rating = row.get('vote_average', 0)
            popularity = row.get('popularity', 0)

            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

                if genre not in genre_ratings:
                    genre_ratings[genre] = []
                genre_ratings[genre].append(rating)

                if genre not in genre_popularity:
                    genre_popularity[genre] = []
                genre_popularity[genre].append(popularity)

        # Calculate averages
        genre_avg_ratings = {
            genre: round(np.mean(ratings), 2)
            for genre, ratings in genre_ratings.items()
        }

        genre_avg_popularity = {
            genre: round(np.mean(pop), 2)
            for genre, pop in genre_popularity.items()
        }

        return {
            'counts': genre_counts,
            'average_ratings': genre_avg_ratings,
            'average_popularity': genre_avg_popularity,
            'top_by_count': sorted(
                genre_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            'top_by_rating': sorted(
                genre_avg_ratings.items(), key=lambda x: x[1], reverse=True
            )[:10]
        }

    def get_temporal_analytics(
        self,
        movies_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get temporal analytics (movies over time).

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe

        Returns
        -------
        dict
            Temporal analytics
        """
        year_counts = {}
        year_ratings = {}
        decade_counts = {}

        for _, row in movies_df.iterrows():
            release_date = row.get('release_date', '')
            if not release_date or not isinstance(release_date, str):
                continue

            try:
                year = int(release_date[:4])
                if year < 1900 or year > 2025:
                    continue
            except (ValueError, TypeError):
                continue

            rating = row.get('vote_average', 0)
            decade = (year // 10) * 10

            year_counts[year] = year_counts.get(year, 0) + 1

            if year not in year_ratings:
                year_ratings[year] = []
            year_ratings[year].append(rating)

            decade_counts[decade] = decade_counts.get(decade, 0) + 1

        # Calculate year averages
        year_avg_ratings = {
            year: round(np.mean(ratings), 2)
            for year, ratings in year_ratings.items()
        }

        return {
            'movies_per_year': dict(sorted(year_counts.items())),
            'average_rating_per_year': dict(sorted(year_avg_ratings.items())),
            'movies_per_decade': dict(sorted(decade_counts.items())),
            'peak_year': max(year_counts.items(), key=lambda x: x[1]) if year_counts else None,
            'best_rated_year': max(
                year_avg_ratings.items(), key=lambda x: x[1]
            ) if year_avg_ratings else None
        }

    def get_rating_distribution(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get rating distribution analytics.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe

        Returns
        -------
        dict
            Rating distribution
        """
        ratings = movies_df['vote_average'].dropna().tolist()

        if not ratings:
            return {
                'histogram': {},
                'statistics': {}
            }

        # Create histogram bins
        bins = np.arange(0, 11, 0.5)
        hist, bin_edges = np.histogram(ratings, bins=bins)

        histogram = {
            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": int(hist[i])
            for i in range(len(hist))
        }

        return {
            'histogram': histogram,
            'statistics': {
                'mean': round(np.mean(ratings), 2),
                'median': round(np.median(ratings), 2),
                'std': round(np.std(ratings), 2),
                'min': round(min(ratings), 2),
                'max': round(max(ratings), 2),
                'percentile_25': round(np.percentile(ratings, 25), 2),
                'percentile_75': round(np.percentile(ratings, 75), 2)
            }
        }

    def get_director_analytics(
        self,
        movies_df: pd.DataFrame,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Get director-related analytics.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe
        top_n : int
            Number of top directors to return

        Returns
        -------
        dict
            Director analytics
        """
        director_counts = {}
        director_ratings = {}
        director_revenue = {}

        for _, row in movies_df.iterrows():
            director = row.get('director')
            if not director or pd.isna(director):
                continue

            director_counts[director] = director_counts.get(director, 0) + 1

            rating = row.get('vote_average', 0)
            if director not in director_ratings:
                director_ratings[director] = []
            director_ratings[director].append(rating)

            revenue = row.get('revenue', 0)
            if revenue and revenue > 0:
                if director not in director_revenue:
                    director_revenue[director] = []
                director_revenue[director].append(revenue)

        # Calculate averages
        director_avg_ratings = {
            director: round(np.mean(ratings), 2)
            for director, ratings in director_ratings.items()
            if len(ratings) >= 3  # At least 3 movies
        }

        director_total_revenue = {
            director: sum(revenues)
            for director, revenues in director_revenue.items()
        }

        return {
            'most_prolific': sorted(
                director_counts.items(), key=lambda x: x[1], reverse=True
            )[:top_n],
            'highest_rated': sorted(
                director_avg_ratings.items(), key=lambda x: x[1], reverse=True
            )[:top_n],
            'highest_grossing': sorted(
                director_total_revenue.items(), key=lambda x: x[1], reverse=True
            )[:top_n],
            'total_directors': len(director_counts)
        }

    def get_recommendation_insights(self) -> Dict[str, Any]:
        """
        Get insights about recommendation patterns.

        Returns
        -------
        dict
            Recommendation insights
        """
        events = self.metrics_tracker.recommendation_events

        if not events:
            return {
                'popular_sources': [],
                'method_trends': {},
                'peak_hours': {}
            }

        # Popular source movies
        popular_sources = self.metrics_tracker.get_popular_source_movies(10)

        # Method trends over time
        method_by_day = {}
        hour_counts = {}

        for event in events:
            try:
                dt = datetime.fromisoformat(event.timestamp)
                day = dt.strftime('%Y-%m-%d')
                hour = dt.hour

                if day not in method_by_day:
                    method_by_day[day] = {}
                method_by_day[day][event.method] = \
                    method_by_day[day].get(event.method, 0) + 1

                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except:
                continue

        # Find peak hours
        peak_hours = sorted(
            hour_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            'popular_sources': popular_sources,
            'method_trends': method_by_day,
            'peak_hours': dict(peak_hours),
            'total_recommendations': len(events)
        }

    def get_user_engagement(self) -> Dict[str, Any]:
        """
        Get user engagement metrics.

        Returns
        -------
        dict
            User engagement metrics
        """
        interactions = self.metrics_tracker.user_interactions

        if not interactions:
            return {
                'total_interactions': 0,
                'interaction_types': {},
                'average_rating': 0,
                'active_users': 0
            }

        interaction_types = {}
        ratings = []
        users = set()

        for interaction in interactions:
            users.add(interaction.user_id)

            itype = interaction.interaction_type
            interaction_types[itype] = interaction_types.get(itype, 0) + 1

            if itype == 'rating' and interaction.value:
                ratings.append(interaction.value)

        return {
            'total_interactions': len(interactions),
            'interaction_types': interaction_types,
            'average_rating': round(np.mean(ratings), 2) if ratings else 0,
            'active_users': len(users),
            'interactions_per_user': round(len(interactions) / len(users), 2) if users else 0
        }

    def generate_report(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe

        Returns
        -------
        dict
            Complete analytics report
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'overview': self.get_overview_stats(movies_df),
            'method_comparison': self.get_method_comparison(),
            'genre_analytics': self.get_genre_analytics(movies_df),
            'temporal_analytics': self.get_temporal_analytics(movies_df),
            'rating_distribution': self.get_rating_distribution(movies_df),
            'director_analytics': self.get_director_analytics(movies_df),
            'recommendation_insights': self.get_recommendation_insights(),
            'user_engagement': self.get_user_engagement()
        }

    def export_report(
        self,
        movies_df: pd.DataFrame,
        filepath: Path
    ) -> None:
        """
        Export analytics report to JSON file.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movies dataframe
        filepath : Path
            Output file path
        """
        import json

        report = self.generate_report(movies_df)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
