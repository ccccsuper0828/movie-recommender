"""
Search bar component with autocomplete.
"""
# @author 成员 F — 前端框架 & API & 测试

import streamlit as st
from typing import List, Optional, Callable


class SearchBar:
    """
    Custom search bar component with suggestions.
    """

    @staticmethod
    def render(
        label: str = "Search movies",
        placeholder: str = "Enter movie title...",
        suggestions: Optional[List[str]] = None,
        on_select: Optional[Callable] = None,
        key: str = "search_bar"
    ) -> str:
        """
        Render a search bar with optional suggestions.

        Parameters
        ----------
        label : str
            Label for the search input
        placeholder : str
            Placeholder text
        suggestions : list, optional
            List of suggestion options
        on_select : callable, optional
            Callback when a suggestion is selected
        key : str
            Unique key for the component

        Returns
        -------
        str
            The current search value
        """
        # Search container styling
        st.markdown("""
        <style>
        .search-container {
            position: relative;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Text input
        search_query = st.text_input(
            label,
            placeholder=placeholder,
            key=f"{key}_input",
            label_visibility="collapsed"
        )

        # Show suggestions if available and query is not empty
        if suggestions and search_query:
            # Filter suggestions based on query
            filtered = [
                s for s in suggestions
                if search_query.lower() in s.lower()
            ][:10]

            if filtered:
                selected = st.selectbox(
                    "Suggestions",
                    options=[""] + filtered,
                    key=f"{key}_suggestions",
                    label_visibility="collapsed"
                )

                if selected:
                    search_query = selected
                    if on_select:
                        on_select(selected)

        return search_query

    @staticmethod
    def render_with_dropdown(
        movies: List[str],
        label: str = "Select a movie",
        default_index: int = 0,
        key: str = "movie_select"
    ) -> Optional[str]:
        """
        Render a search bar with dropdown selection.

        Parameters
        ----------
        movies : list
            List of movie titles
        label : str
            Label for the dropdown
        default_index : int
            Default selection index
        key : str
            Unique key

        Returns
        -------
        str or None
            Selected movie title
        """
        # Search filter
        search_query = st.text_input(
            "Filter movies",
            placeholder="Type to filter...",
            key=f"{key}_filter"
        )

        # Filter movies
        if search_query:
            filtered_movies = [
                m for m in movies
                if search_query.lower() in m.lower()
            ]
        else:
            filtered_movies = movies[:100]  # Limit for performance

        if not filtered_movies:
            st.warning("No movies found matching your search.")
            return None

        # Movie selection
        selected = st.selectbox(
            label,
            options=filtered_movies,
            index=min(default_index, len(filtered_movies) - 1),
            key=f"{key}_select"
        )

        return selected

    @staticmethod
    def render_hero_search(
        movies: List[str],
        popular_movies: List[str],
        key: str = "hero_search"
    ) -> Optional[str]:
        """
        Render a hero-style search bar.

        Parameters
        ----------
        movies : list
            All movie titles
        popular_movies : list
            Popular movie titles for quick selection
        key : str
            Unique key

        Returns
        -------
        str or None
            Selected movie title
        """
        st.markdown(
            '<div style="text-align:center;padding:40px 0;">'
            '<h1 style="font-size:2.5rem;font-weight:700;'
            'background:linear-gradient(135deg,#ffffff 0%,#f5c518 100%);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px;">'
            'Find Your Next Favorite Movie</h1>'
            '<p style="color:#a3a3a3;font-size:1.1rem;">'
            'Get personalized recommendations based on movies you love</p></div>',
            unsafe_allow_html=True
        )

        # Search input
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            selected = SearchBar.render_with_dropdown(
                movies=movies,
                label="Select a movie",
                key=key
            )

        # Quick picks
        if popular_movies:
            st.markdown(
                '<p style="color:#666;text-align:center;margin-top:20px;">Quick picks:</p>',
                unsafe_allow_html=True
            )

            # Show popular movies as chips
            cols = st.columns(min(len(popular_movies[:6]), 6))
            for i, movie in enumerate(popular_movies[:6]):
                with cols[i]:
                    if st.button(
                        movie[:15] + "..." if len(movie) > 15 else movie,
                        key=f"{key}_quick_{i}",
                        width="stretch"
                    ):
                        return movie

        return selected
