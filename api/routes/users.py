"""
User API endpoints.
"""
# @author 成员 F — 前端框架 & API & 测试

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_user_service, get_recommendation_service
from src.services.user_service import UserService
from src.services.recommendation_service import RecommendationService
from src.models.schemas import (
    UserCreate,
    UserResponse,
    UserProfile,
    RatingCreate,
    RatingResponse
)

router = APIRouter()


@router.post("/", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    """
    Create a new user.
    """
    # Check if username exists
    existing = user_service.get_user_by_username(user.username)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Username already exists: {user.username}"
        )

    new_user = user_service.create_user(
        username=user.username,
        email=user.email,
        favorite_genres=user.favorite_genres,
        favorite_directors=user.favorite_directors
    )

    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        created_at=new_user.created_at,
        favorite_genres=new_user.favorite_genres,
        num_ratings=new_user.num_ratings
    )


@router.get("/{user_id}", response_model=UserProfile)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    """
    Get user profile.
    """
    profile = user_service.get_user_profile(user_id)

    if profile is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    return UserProfile(**profile)


@router.get("/")
async def list_users(
    user_service: UserService = Depends(get_user_service)
):
    """
    List all users.
    """
    return {
        "count": user_service.user_count,
        "users": user_service.get_all_users()
    }


@router.put("/{user_id}/preferences")
async def update_preferences(
    user_id: int,
    favorite_genres: Optional[List[str]] = None,
    favorite_directors: Optional[List[str]] = None,
    disliked_genres: Optional[List[str]] = None,
    user_service: UserService = Depends(get_user_service)
):
    """
    Update user preferences.
    """
    user = user_service.update_preferences(
        user_id=user_id,
        favorite_genres=favorite_genres,
        favorite_directors=favorite_directors,
        disliked_genres=disliked_genres
    )

    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    return {
        "message": "Preferences updated",
        "favorite_genres": user.favorite_genres,
        "favorite_directors": user.favorite_directors,
        "disliked_genres": user.disliked_genres
    }


@router.post("/{user_id}/ratings")
async def add_rating(
    user_id: int,
    rating: RatingCreate,
    user_service: UserService = Depends(get_user_service)
):
    """
    Add a movie rating.
    """
    user = user_service.add_rating(
        user_id=user_id,
        movie_id=rating.movie_id,
        rating=rating.rating
    )

    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    return {
        "message": "Rating added",
        "user_id": user_id,
        "movie_id": rating.movie_id,
        "rating": rating.rating
    }


@router.get("/{user_id}/ratings")
async def get_ratings(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    """
    Get user's ratings.
    """
    user = user_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    ratings = user_service.get_user_ratings(user_id)

    return {
        "user_id": user_id,
        "count": len(ratings),
        "avg_rating": user.avg_rating,
        "ratings": ratings
    }


@router.post("/{user_id}/watchlist/{movie_id}")
async def add_to_watchlist(
    user_id: int,
    movie_id: int,
    user_service: UserService = Depends(get_user_service)
):
    """
    Add movie to user's watchlist.
    """
    success = user_service.add_to_watchlist(user_id, movie_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    return {"message": "Added to watchlist", "movie_id": movie_id}


@router.delete("/{user_id}/watchlist/{movie_id}")
async def remove_from_watchlist(
    user_id: int,
    movie_id: int,
    user_service: UserService = Depends(get_user_service)
):
    """
    Remove movie from user's watchlist.
    """
    success = user_service.remove_from_watchlist(user_id, movie_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    return {"message": "Removed from watchlist", "movie_id": movie_id}


@router.get("/{user_id}/watchlist")
async def get_watchlist(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get user's watchlist with movie details.
    """
    user = user_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    # Get movie details for watchlist
    watchlist_movies = []
    for movie_id in user.watchlist:
        # In a real app, we'd look up by ID; here we'd need to add that capability
        watchlist_movies.append({"movie_id": movie_id})

    return {
        "user_id": user_id,
        "count": len(user.watchlist),
        "watchlist": watchlist_movies
    }


@router.get("/{user_id}/recommendations")
async def get_personalized_recommendations(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50),
    user_service: UserService = Depends(get_user_service),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get personalized recommendations based on user's ratings and preferences.
    """
    user = user_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    # Get recommendations based on liked movies
    if user.liked_movies:
        # Use the most recently liked movie
        recommendations = []
        seen_titles = set()

        for movie_id in user.liked_movies[:3]:  # Use top 3 liked movies
            # In a real app, we'd get the title from movie_id
            # For now, this is a placeholder
            pass

    # Fallback to popular movies filtered by preferred genres
    popular = rec_service.get_popular_movies(top_n * 2)

    # Filter by user preferences if available
    results = []
    for _, row in popular.iterrows():
        if len(results) >= top_n:
            break

        movie_genres = row.get('genres_list', [])

        # Skip disliked genres
        if any(g in user.disliked_genres for g in movie_genres):
            continue

        # Boost preferred genres
        score_boost = 0
        if any(g in user.favorite_genres for g in movie_genres):
            score_boost = 0.1

        results.append({
            "title": row['title'],
            "genres": movie_genres,
            "vote_average": row.get('vote_average', 0),
            "personalization_score": row.get('popularity', 0) / 1000 + score_boost
        })

    return {
        "user_id": user_id,
        "recommendations": sorted(
            results,
            key=lambda x: x['personalization_score'],
            reverse=True
        )
    }


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    """
    Delete a user.
    """
    success = user_service.delete_user(user_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    return {"message": "User deleted", "user_id": user_id}
