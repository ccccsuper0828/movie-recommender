"""
Pydantic schemas for API request/response validation.
"""
# @author 成员 F — 前端框架 & API & 测试

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field, EmailStr


# ==================== Movie Schemas ====================

class MovieBase(BaseModel):
    """Base movie schema."""
    title: str
    overview: Optional[str] = ""
    genres: List[str] = []
    vote_average: float = 0.0
    popularity: float = 0.0


class MovieCreate(MovieBase):
    """Schema for creating a movie."""
    director: Optional[str] = ""
    cast: List[str] = []
    keywords: List[str] = []
    release_date: Optional[date] = None
    runtime: int = 0
    budget: int = 0
    revenue: int = 0


class MovieResponse(MovieBase):
    """Schema for movie response."""
    id: int
    director: str = ""
    cast: List[str] = []
    release_date: Optional[str] = None
    runtime: int = 0

    class Config:
        from_attributes = True


class MovieDetail(MovieResponse):
    """Detailed movie response."""
    keywords: List[str] = []
    budget: int = 0
    revenue: int = 0
    vote_count: int = 0
    original_language: str = "en"


# ==================== Recommendation Schemas ====================

class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    title: str = Field(..., description="Movie title to get recommendations for")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    method: str = Field(
        default="hybrid",
        description="Recommendation method: content, metadata, cf, or hybrid"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom weights for hybrid method"
    )


class RecommendationItem(BaseModel):
    """A single recommendation item."""
    rank: int
    title: str
    genres: List[str]
    vote_average: float
    similarity_score: float
    explanation: Optional[str] = None
    method_scores: Optional[Dict[str, float]] = None


class RecommendationResponse(BaseModel):
    """Response containing recommendations."""
    source_movie: str
    method: str
    recommendations: List[RecommendationItem]
    total_count: int


class BatchRecommendationRequest(BaseModel):
    """Request for batch recommendations."""
    titles: List[str] = Field(..., min_length=1, max_length=10)
    top_n: int = Field(default=5, ge=1, le=20)
    method: str = Field(default="hybrid")


class BatchRecommendationResponse(BaseModel):
    """Response for batch recommendations."""
    results: Dict[str, RecommendationResponse]


# ==================== User Schemas ====================

class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a user."""
    favorite_genres: List[str] = []
    favorite_directors: List[str] = []


class UserResponse(UserBase):
    """Schema for user response."""
    id: int
    created_at: datetime
    favorite_genres: List[str] = []
    num_ratings: int = 0

    class Config:
        from_attributes = True


class UserProfile(UserResponse):
    """Detailed user profile."""
    favorite_directors: List[str] = []
    disliked_genres: List[str] = []
    watch_history_count: int = 0
    watchlist_count: int = 0
    avg_rating: Optional[float] = None


# ==================== Rating Schemas ====================

class RatingCreate(BaseModel):
    """Schema for creating a rating."""
    movie_id: int
    rating: float = Field(..., ge=1.0, le=5.0)


class RatingResponse(BaseModel):
    """Schema for rating response."""
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime


# ==================== Search Schemas ====================

class SearchRequest(BaseModel):
    """Request for movie search."""
    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class SearchResponse(BaseModel):
    """Response for movie search."""
    query: str
    total_results: int
    page: int
    page_size: int
    results: List[MovieResponse]


# ==================== Explanation Schemas ====================

class ExplanationRequest(BaseModel):
    """Request for recommendation explanation."""
    source_title: str
    target_title: str
    method: str = Field(default="metadata")


class ExplanationResponse(BaseModel):
    """Response containing explanation."""
    source_movie: str
    recommended_movie: str
    method: str
    reasons: List[str]
    details: Dict[str, Any]
    summary: str


# ==================== Analytics Schemas ====================

class AnalyticsOverview(BaseModel):
    """Analytics overview response."""
    total_movies: int
    total_users: int
    total_ratings: int
    avg_movie_rating: float
    popular_genres: List[Dict[str, Any]]
    top_rated_movies: List[MovieResponse]


class MethodComparisonResult(BaseModel):
    """Method comparison result."""
    movie: str
    scores: Dict[str, float]
    overlap_matrix: Dict[str, Dict[str, float]]


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]
