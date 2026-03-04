"""
Analytics service for tracking and reporting.
"""

from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for analytics and reporting on movies and recommendations.
    """

    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize the analytics service.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe
        """
        self.movies = movies_df

    def get_overview(self) -> Dict[str, Any]:
        """
        Get general statistics overview.

        Returns
        -------
        dict
            Overview statistics
        """
        return {
            'total_movies': len(self.movies),
            'avg_rating': round(self.movies['vote_average'].mean(), 2),
            'avg_popularity': round(self.movies['popularity'].mean(), 2),
            'total_genres': len(self._get_all_genres()),
            'year_range': self._get_year_range(),
            'rating_distribution': self._get_rating_distribution()
        }

    def _get_all_genres(self) -> List[str]:
        """Get all unique genres."""
        genres = set()
        for genre_list in self.movies['genres_list']:
            if isinstance(genre_list, list):
                genres.update(genre_list)
        return list(genres)

    def _get_year_range(self) -> Dict[str, int]:
        """Get year range."""
        years = []
        for date_str in self.movies['release_date']:
            if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                try:
                    years.append(int(date_str[:4]))
                except ValueError:
                    pass

        if years:
            return {'min': min(years), 'max': max(years)}
        return {'min': None, 'max': None}

    def _get_rating_distribution(self) -> Dict[str, int]:
        """Get rating distribution."""
        bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        labels = ['0-2', '2-4', '4-6', '6-8', '8-10']

        distribution = {}
        for (low, high), label in zip(bins, labels):
            count = len(self.movies[
                (self.movies['vote_average'] >= low) &
                (self.movies['vote_average'] < high)
            ])
            distribution[label] = count

        return distribution

    def get_genre_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each genre.

        Returns
        -------
        list
            Genre statistics
        """
        genre_stats = {}

        for _, row in self.movies.iterrows():
            genres = row.get('genres_list', [])
            if not isinstance(genres, list):
                continue

            for genre in genres:
                if genre not in genre_stats:
                    genre_stats[genre] = {
                        'count': 0,
                        'total_rating': 0,
                        'total_popularity': 0
                    }

                genre_stats[genre]['count'] += 1
                genre_stats[genre]['total_rating'] += row.get('vote_average', 0)
                genre_stats[genre]['total_popularity'] += row.get('popularity', 0)

        # Calculate averages
        results = []
        for genre, stats in genre_stats.items():
            results.append({
                'genre': genre,
                'count': stats['count'],
                'avg_rating': round(stats['total_rating'] / stats['count'], 2),
                'avg_popularity': round(stats['total_popularity'] / stats['count'], 2)
            })

        # Sort by count
        results.sort(key=lambda x: x['count'], reverse=True)
        return results

    def get_top_rated_movies(self, top_n: int = 20, min_votes: int = 100) -> List[Dict[str, Any]]:
        """
        Get top rated movies.

        Parameters
        ----------
        top_n : int
            Number of movies
        min_votes : int
            Minimum vote count

        Returns
        -------
        list
            Top rated movies
        """
        filtered = self.movies[self.movies['vote_count'] >= min_votes]
        top = filtered.nlargest(top_n, 'vote_average')

        return [
            {
                'title': row['title'],
                'vote_average': row['vote_average'],
                'vote_count': row.get('vote_count', 0),
                'genres': row.get('genres_list', [])
            }
            for _, row in top.iterrows()
        ]

    def get_most_popular_movies(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get most popular movies.

        Parameters
        ----------
        top_n : int
            Number of movies

        Returns
        -------
        list
            Most popular movies
        """
        top = self.movies.nlargest(top_n, 'popularity')

        return [
            {
                'title': row['title'],
                'popularity': row['popularity'],
                'vote_average': row.get('vote_average', 0),
                'genres': row.get('genres_list', [])
            }
            for _, row in top.iterrows()
        ]

    def get_movies_per_year(self) -> Dict[int, int]:
        """
        Get movie count per year.

        Returns
        -------
        dict
            Year to count mapping
        """
        year_counts = {}

        for date_str in self.movies['release_date']:
            if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    year_counts[year] = year_counts.get(year, 0) + 1
                except ValueError:
                    pass

        return dict(sorted(year_counts.items()))

    def get_director_statistics(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get statistics for top directors.

        Parameters
        ----------
        top_n : int
            Number of directors

        Returns
        -------
        list
            Director statistics
        """
        director_stats = {}

        for _, row in self.movies.iterrows():
            director = row.get('director', '')
            if not director:
                continue

            if director not in director_stats:
                director_stats[director] = {
                    'count': 0,
                    'total_rating': 0,
                    'movies': []
                }

            director_stats[director]['count'] += 1
            director_stats[director]['total_rating'] += row.get('vote_average', 0)
            director_stats[director]['movies'].append(row['title'])

        # Calculate averages and sort
        results = []
        for director, stats in director_stats.items():
            if stats['count'] >= 2:  # At least 2 movies
                results.append({
                    'director': director,
                    'movie_count': stats['count'],
                    'avg_rating': round(stats['total_rating'] / stats['count'], 2),
                    'top_movies': stats['movies'][:3]
                })

        results.sort(key=lambda x: x['movie_count'], reverse=True)
        return results[:top_n]

    def get_budget_revenue_analysis(self) -> Dict[str, Any]:
        """
        Analyze budget and revenue relationships.

        Returns
        -------
        dict
            Budget/revenue analysis
        """
        # Filter movies with budget and revenue data
        valid = self.movies[
            (self.movies['budget'] > 0) &
            (self.movies['revenue'] > 0)
        ].copy()

        if len(valid) == 0:
            return {'error': 'No budget/revenue data available'}

        valid['profit'] = valid['revenue'] - valid['budget']
        valid['roi'] = (valid['profit'] / valid['budget'] * 100).replace(
            [np.inf, -np.inf], np.nan
        )

        return {
            'total_movies_with_data': len(valid),
            'avg_budget': round(valid['budget'].mean() / 1e6, 2),  # Millions
            'avg_revenue': round(valid['revenue'].mean() / 1e6, 2),
            'avg_profit': round(valid['profit'].mean() / 1e6, 2),
            'avg_roi': round(valid['roi'].mean(), 2),
            'top_profitable': [
                {
                    'title': row['title'],
                    'profit': round(row['profit'] / 1e9, 2),  # Billions
                    'roi': round(row['roi'], 0)
                }
                for _, row in valid.nlargest(5, 'profit').iterrows()
            ],
            'top_roi': [
                {
                    'title': row['title'],
                    'roi': round(row['roi'], 0),
                    'budget': round(row['budget'] / 1e6, 1)
                }
                for _, row in valid[valid['budget'] >= 1e6].nlargest(5, 'roi').iterrows()
            ]
        }

    def get_correlation_analysis(self) -> Dict[str, float]:
        """
        Analyze correlations between numeric features.

        Returns
        -------
        dict
            Correlation coefficients
        """
        numeric_cols = ['budget', 'revenue', 'popularity', 'vote_average', 'vote_count', 'runtime']
        available_cols = [c for c in numeric_cols if c in self.movies.columns]

        correlations = {}
        for i, col1 in enumerate(available_cols):
            for col2 in available_cols[i+1:]:
                corr = self.movies[[col1, col2]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[f'{col1}_vs_{col2}'] = round(corr, 3)

        return correlations
