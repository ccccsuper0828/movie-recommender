"""
Search service for movie discovery.
"""
# @author 成员 F — 前端框架 & API & 测试

from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for searching and filtering movies.
    """

    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize the search service.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe
        """
        self.movies = movies_df.reset_index(drop=True)
        self._title_index = pd.Series(
            self.movies.index,
            index=self.movies['title'].str.lower()
        ).to_dict()

        # Extract all unique values for filters
        self._all_genres = self._extract_all_genres()
        self._all_years = self._extract_all_years()

    def _extract_all_genres(self) -> List[str]:
        """Extract all unique genres."""
        genres = set()
        for genre_list in self.movies['genres_list']:
            if isinstance(genre_list, list):
                genres.update(genre_list)
        return sorted(genres)

    def _extract_all_years(self) -> List[int]:
        """Extract all unique years."""
        years = set()
        for date_str in self.movies['release_date']:
            if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    years.add(year)
                except ValueError:
                    pass
        return sorted(years, reverse=True)

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search movies with optional filters.

        Parameters
        ----------
        query : str
            Search query (matches against title)
        filters : dict, optional
            Filters to apply:
            - genres: List[str] - filter by genre(s)
            - year_min: int - minimum release year
            - year_max: int - maximum release year
            - rating_min: float - minimum rating
            - rating_max: float - maximum rating
            - director: str - filter by director
        page : int
            Page number (1-indexed)
        page_size : int
            Results per page

        Returns
        -------
        dict
            Search results with pagination
        """
        # Start with all movies
        results = self.movies.copy()

        # Text search
        if query:
            mask = results['title'].str.contains(query, case=False, na=False)
            results = results[mask]

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Calculate pagination
        total_results = len(results)
        total_pages = (total_results + page_size - 1) // page_size

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        results_page = results.iloc[start_idx:end_idx]

        # Format results
        movies = []
        for _, row in results_page.iterrows():
            movies.append({
                'id': int(row.get('id', 0)),
                'title': row.get('title', ''),
                'genres': row.get('genres_list', []),
                'vote_average': row.get('vote_average', 0),
                'popularity': row.get('popularity', 0),
                'release_date': row.get('release_date', ''),
                'overview': row.get('overview', '')[:200]
            })

        return {
            'query': query,
            'total_results': total_results,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'results': movies
        }

    def _apply_filters(
        self,
        df: pd.DataFrame,
        filters: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply filters to dataframe."""
        # Genre filter
        if filters.get('genres'):
            genres = filters['genres']
            if isinstance(genres, str):
                genres = [genres]
            mask = df['genres_list'].apply(
                lambda x: any(g in x for g in genres) if isinstance(x, list) else False
            )
            df = df[mask]

        # Year filters
        if filters.get('year_min') or filters.get('year_max'):
            def get_year(date_str):
                if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                    try:
                        return int(date_str[:4])
                    except ValueError:
                        return None
                return None

            df['_year'] = df['release_date'].apply(get_year)

            if filters.get('year_min'):
                df = df[df['_year'] >= filters['year_min']]
            if filters.get('year_max'):
                df = df[df['_year'] <= filters['year_max']]

            df = df.drop(columns=['_year'])

        # Rating filters
        if filters.get('rating_min'):
            df = df[df['vote_average'] >= filters['rating_min']]
        if filters.get('rating_max'):
            df = df[df['vote_average'] <= filters['rating_max']]

        # Director filter
        if filters.get('director'):
            director_query = filters['director'].lower()
            mask = df['director'].str.lower().str.contains(director_query, na=False)
            df = df[mask]

        return df

    def fuzzy_search(
        self,
        query: str,
        max_results: int = 10,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy search for movie titles.

        Parameters
        ----------
        query : str
            Search query
        max_results : int
            Maximum results to return
        threshold : float
            Minimum similarity threshold (0 to 1)

        Returns
        -------
        list
            List of matching movies with similarity scores
        """
        query_lower = query.lower()
        matches = []

        for title in self.movies['title']:
            similarity = SequenceMatcher(None, query_lower, title.lower()).ratio()
            if similarity >= threshold:
                matches.append((title, similarity))

        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for title, similarity in matches[:max_results]:
            idx = self._title_index.get(title.lower())
            if idx is not None:
                row = self.movies.iloc[idx]
                results.append({
                    'title': title,
                    'similarity': round(similarity, 3),
                    'genres': row.get('genres_list', []),
                    'vote_average': row.get('vote_average', 0)
                })

        return results

    def get_suggestions(self, partial_query: str, max_results: int = 5) -> List[str]:
        """
        Get autocomplete suggestions for partial query.

        Parameters
        ----------
        partial_query : str
            Partial search query
        max_results : int
            Maximum suggestions

        Returns
        -------
        list
            List of suggested titles
        """
        if not partial_query:
            return []

        query_lower = partial_query.lower()
        suggestions = []

        # Prioritize titles starting with query
        for title in self.movies['title']:
            if title.lower().startswith(query_lower):
                suggestions.append(title)
                if len(suggestions) >= max_results:
                    break

        # If not enough, include titles containing query
        if len(suggestions) < max_results:
            for title in self.movies['title']:
                if query_lower in title.lower() and title not in suggestions:
                    suggestions.append(title)
                    if len(suggestions) >= max_results:
                        break

        return suggestions

    def get_movies_by_genre(
        self,
        genre: str,
        sort_by: str = 'popularity',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get movies by genre.

        Parameters
        ----------
        genre : str
            Genre to filter by
        sort_by : str
            Column to sort by
        top_n : int
            Number of results

        Returns
        -------
        pd.DataFrame
            Filtered and sorted movies
        """
        mask = self.movies['genres_list'].apply(
            lambda x: genre in x if isinstance(x, list) else False
        )
        filtered = self.movies[mask]

        if sort_by in filtered.columns:
            filtered = filtered.sort_values(sort_by, ascending=False)

        return filtered.head(top_n)[
            ['title', 'genres_list', 'vote_average', 'popularity', 'release_date']
        ]

    def get_movies_by_director(
        self,
        director: str,
        sort_by: str = 'vote_average'
    ) -> pd.DataFrame:
        """
        Get movies by director.

        Parameters
        ----------
        director : str
            Director name (partial match)
        sort_by : str
            Column to sort by

        Returns
        -------
        pd.DataFrame
            Movies by the director
        """
        mask = self.movies['director'].str.contains(director, case=False, na=False)
        filtered = self.movies[mask]

        if sort_by in filtered.columns:
            filtered = filtered.sort_values(sort_by, ascending=False)

        return filtered[
            ['title', 'director', 'genres_list', 'vote_average', 'release_date']
        ]

    @property
    def available_genres(self) -> List[str]:
        """Get all available genres."""
        return self._all_genres

    @property
    def available_years(self) -> List[int]:
        """Get all available years."""
        return self._all_years
