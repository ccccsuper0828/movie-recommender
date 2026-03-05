"""
Movie API endpoints.
"""
# @author 成员 F — 前端框架 & API & 测试

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_recommendation_service, get_search_service, get_analytics_service
from src.services.recommendation_service import RecommendationService
from src.services.search_service import SearchService
from src.services.analytics_service import AnalyticsService
from src.models.schemas import (
    MovieResponse,
    MovieDetail,
    SearchRequest,
    SearchResponse
)

router = APIRouter()


@router.get("/", response_model=SearchResponse)
async def list_movies(
    query: str = Query(default="", description="Search query"),
    genres: Optional[str] = Query(default=None, description="Comma-separated genres"),
    year_min: Optional[int] = Query(default=None, description="Minimum year"),
    year_max: Optional[int] = Query(default=None, description="Maximum year"),
    rating_min: Optional[float] = Query(default=None, ge=0, le=10),
    rating_max: Optional[float] = Query(default=None, ge=0, le=10),
    director: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Search and list movies with filters.
    """
    filters = {}

    if genres:
        filters['genres'] = [g.strip() for g in genres.split(',')]
    if year_min:
        filters['year_min'] = year_min
    if year_max:
        filters['year_max'] = year_max
    if rating_min:
        filters['rating_min'] = rating_min
    if rating_max:
        filters['rating_max'] = rating_max
    if director:
        filters['director'] = director

    result = search_service.search(
        query=query,
        filters=filters if filters else None,
        page=page,
        page_size=page_size
    )

    return SearchResponse(
        query=result['query'],
        total_results=result['total_results'],
        page=result['page'],
        page_size=result['page_size'],
        results=[MovieResponse(**m) for m in result['results']]
    )


@router.get("/popular")
async def get_popular_movies(
    top_n: int = Query(default=20, ge=1, le=100),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get most popular movies.
    """
    popular = service.get_popular_movies(top_n)

    return {
        "count": len(popular),
        "movies": [
            {
                "title": row['title'],
                "genres": row.get('genres_list', []),
                "vote_average": row.get('vote_average', 0),
                "popularity": row.get('popularity', 0)
            }
            for _, row in popular.iterrows()
        ]
    }


@router.get("/top-rated")
async def get_top_rated(
    top_n: int = Query(default=20, ge=1, le=100),
    min_votes: int = Query(default=100, ge=0),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get top rated movies.
    """
    return analytics.get_top_rated_movies(top_n, min_votes)


@router.get("/suggestions")
async def get_suggestions(
    q: str = Query(..., min_length=1, description="Partial query"),
    limit: int = Query(default=5, ge=1, le=20),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get autocomplete suggestions for movie search.
    """
    suggestions = search_service.get_suggestions(q, limit)
    return {"query": q, "suggestions": suggestions}


@router.get("/genres")
async def get_genres(
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get all available genres.
    """
    return {"genres": search_service.available_genres}


@router.get("/years")
async def get_years(
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get all available release years.
    """
    return {"years": search_service.available_years}


@router.get("/by-genre/{genre}")
async def get_movies_by_genre(
    genre: str,
    sort_by: str = Query(default="popularity", regex="^(popularity|vote_average)$"),
    top_n: int = Query(default=20, ge=1, le=100),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get movies by genre.
    """
    movies = search_service.get_movies_by_genre(genre, sort_by, top_n)

    return {
        "genre": genre,
        "count": len(movies),
        "movies": [
            {
                "title": row['title'],
                "genres": row.get('genres_list', []),
                "vote_average": row.get('vote_average', 0),
                "popularity": row.get('popularity', 0)
            }
            for _, row in movies.iterrows()
        ]
    }


@router.get("/by-director/{director}")
async def get_movies_by_director(
    director: str,
    sort_by: str = Query(default="vote_average", regex="^(vote_average|release_date)$"),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get movies by director.
    """
    movies = search_service.get_movies_by_director(director, sort_by)

    return {
        "director": director,
        "count": len(movies),
        "movies": [
            {
                "title": row['title'],
                "director": row.get('director', ''),
                "genres": row.get('genres_list', []),
                "vote_average": row.get('vote_average', 0)
            }
            for _, row in movies.iterrows()
        ]
    }


@router.get("/{movie_title}")
async def get_movie(
    movie_title: str,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get movie details by title.
    """
    movie = service.get_movie_info(movie_title)

    if movie is None:
        # Try fuzzy search
        suggestions = service.search_movies(movie_title, max_results=5)
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Movie not found: {movie_title}",
                "suggestions": suggestions
            }
        )

    return movie


@router.get("/analytics/overview")
async def get_analytics_overview(
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get analytics overview.
    """
    return analytics.get_overview()


@router.get("/analytics/genres")
async def get_genre_statistics(
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get genre statistics.
    """
    return analytics.get_genre_statistics()


@router.get("/analytics/directors")
async def get_director_statistics(
    top_n: int = Query(default=20, ge=1, le=100),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get director statistics.
    """
    return analytics.get_director_statistics(top_n)
