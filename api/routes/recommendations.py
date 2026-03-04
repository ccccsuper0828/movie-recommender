"""
Recommendation API endpoints.
"""

from typing import Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_recommendation_service
from src.services.recommendation_service import RecommendationService
from src.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    ExplanationRequest,
    ExplanationResponse
)

router = APIRouter()


@router.get("/{movie_title}", response_model=RecommendationResponse)
async def get_recommendations(
    movie_title: str,
    top_n: int = Query(default=10, ge=1, le=50),
    method: str = Query(default="hybrid", regex="^(content|metadata|cf|hybrid)$"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get movie recommendations.

    Parameters
    ----------
    movie_title : str
        Movie title to get recommendations for
    top_n : int
        Number of recommendations (1-50)
    method : str
        Recommendation method: content, metadata, cf, or hybrid
    """
    result = service.get_recommendations(
        title=movie_title,
        top_n=top_n,
        method=method,
        include_explanation=True
    )

    if result is None:
        raise HTTPException(status_code=404, detail=f"Movie not found: {movie_title}")

    if 'error' in result:
        raise HTTPException(
            status_code=404,
            detail={
                "message": result['error'],
                "suggestions": result.get('suggestions', [])
            }
        )

    # Convert to response model
    recommendations = [
        RecommendationItem(
            rank=rec['rank'],
            title=rec['title'],
            genres=rec['genres'],
            vote_average=rec['vote_average'],
            similarity_score=rec['similarity_score'],
            explanation=rec.get('explanation'),
            method_scores=rec.get('method_scores')
        )
        for rec in result['recommendations']
    ]

    return RecommendationResponse(
        source_movie=result['source_movie'],
        method=result['method'],
        recommendations=recommendations,
        total_count=result['total_count']
    )


@router.post("/", response_model=RecommendationResponse)
async def post_recommendations(
    request: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get movie recommendations (POST version for complex requests).
    """
    weights = None
    if request.weights:
        weights = (
            request.weights.get('content', 0.3),
            request.weights.get('metadata', 0.4),
            request.weights.get('cf', 0.3)
        )

    result = service.get_recommendations(
        title=request.title,
        top_n=request.top_n,
        method=request.method,
        weights=weights,
        include_explanation=True
    )

    if result is None or 'error' in result:
        raise HTTPException(
            status_code=404,
            detail=result.get('error', 'Movie not found') if result else 'Movie not found'
        )

    recommendations = [
        RecommendationItem(
            rank=rec['rank'],
            title=rec['title'],
            genres=rec['genres'],
            vote_average=rec['vote_average'],
            similarity_score=rec['similarity_score'],
            explanation=rec.get('explanation'),
            method_scores=rec.get('method_scores')
        )
        for rec in result['recommendations']
    ]

    return RecommendationResponse(
        source_movie=result['source_movie'],
        method=result['method'],
        recommendations=recommendations,
        total_count=result['total_count']
    )


@router.post("/batch", response_model=BatchRecommendationResponse)
async def batch_recommendations(
    request: BatchRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get recommendations for multiple movies.
    """
    results = {}

    for title in request.titles:
        result = service.get_recommendations(
            title=title,
            top_n=request.top_n,
            method=request.method,
            include_explanation=False
        )

        if result and 'error' not in result:
            recommendations = [
                RecommendationItem(
                    rank=rec['rank'],
                    title=rec['title'],
                    genres=rec['genres'],
                    vote_average=rec['vote_average'],
                    similarity_score=rec['similarity_score']
                )
                for rec in result['recommendations']
            ]

            results[title] = RecommendationResponse(
                source_movie=title,
                method=request.method,
                recommendations=recommendations,
                total_count=len(recommendations)
            )

    return BatchRecommendationResponse(results=results)


@router.get("/compare/{movie_title}")
async def compare_methods(
    movie_title: str,
    top_n: int = Query(default=5, ge=1, le=20),
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Compare recommendations from all methods.
    """
    result = service.compare_methods(movie_title, top_n)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Movie not found: {movie_title}")

    return {
        "source_movie": movie_title,
        "comparisons": result
    }


@router.post("/explain", response_model=ExplanationResponse)
async def explain_recommendation(
    request: ExplanationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get detailed explanation for a recommendation.
    """
    explanation = service.explain_recommendation(
        source_title=request.source_title,
        target_title=request.target_title,
        method=request.method
    )

    if explanation is None:
        raise HTTPException(
            status_code=404,
            detail="Could not generate explanation"
        )

    return ExplanationResponse(
        source_movie=explanation['source_movie'],
        recommended_movie=explanation['recommended_movie'],
        method=explanation['method'],
        reasons=explanation['reasons'],
        details=explanation['details'],
        summary=explanation['summary']
    )


@router.get("/similarity/{movie1}/{movie2}")
async def get_similarity(
    movie1: str,
    movie2: str,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get similarity scores between two movies.
    """
    scores = service.get_similarity_scores(movie1, movie2)

    if scores is None:
        raise HTTPException(
            status_code=404,
            detail="One or both movies not found"
        )

    return {
        "movie1": movie1,
        "movie2": movie2,
        "similarity_scores": scores
    }
