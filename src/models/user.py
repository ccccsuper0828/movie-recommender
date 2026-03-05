"""
User data model.
"""
# @author 成员 F — 前端框架 & API & 测试

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class UserRating:
    """A user's rating for a movie."""
    movie_id: int
    rating: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class User:
    """
    User data class representing a user entity.
    """
    id: int
    username: str
    email: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Preferences
    favorite_genres: List[str] = field(default_factory=list)
    favorite_directors: List[str] = field(default_factory=list)
    disliked_genres: List[str] = field(default_factory=list)

    # History
    ratings: List[UserRating] = field(default_factory=list)
    watch_history: List[int] = field(default_factory=list)  # Movie IDs
    watchlist: List[int] = field(default_factory=list)  # Movie IDs to watch

    def add_rating(self, movie_id: int, rating: float):
        """Add a rating for a movie."""
        # Remove existing rating for this movie if any
        self.ratings = [r for r in self.ratings if r.movie_id != movie_id]
        self.ratings.append(UserRating(movie_id=movie_id, rating=rating))

        # Add to watch history if not already there
        if movie_id not in self.watch_history:
            self.watch_history.append(movie_id)

    def get_rating(self, movie_id: int) -> Optional[float]:
        """Get the user's rating for a movie."""
        for rating in self.ratings:
            if rating.movie_id == movie_id:
                return rating.rating
        return None

    def add_to_watchlist(self, movie_id: int):
        """Add a movie to the watchlist."""
        if movie_id not in self.watchlist:
            self.watchlist.append(movie_id)

    def remove_from_watchlist(self, movie_id: int):
        """Remove a movie from the watchlist."""
        if movie_id in self.watchlist:
            self.watchlist.remove(movie_id)

    def has_watched(self, movie_id: int) -> bool:
        """Check if user has watched a movie."""
        return movie_id in self.watch_history

    @property
    def num_ratings(self) -> int:
        """Get the number of ratings."""
        return len(self.ratings)

    @property
    def avg_rating(self) -> Optional[float]:
        """Get the average rating."""
        if not self.ratings:
            return None
        return sum(r.rating for r in self.ratings) / len(self.ratings)

    @property
    def liked_movies(self) -> List[int]:
        """Get movies rated 4 or above."""
        return [r.movie_id for r in self.ratings if r.rating >= 4.0]

    @property
    def disliked_movies(self) -> List[int]:
        """Get movies rated below 3."""
        return [r.movie_id for r in self.ratings if r.rating < 3.0]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'favorite_genres': self.favorite_genres,
            'favorite_directors': self.favorite_directors,
            'disliked_genres': self.disliked_genres,
            'ratings': [
                {'movie_id': r.movie_id, 'rating': r.rating, 'timestamp': r.timestamp.isoformat()}
                for r in self.ratings
            ],
            'watch_history': self.watch_history,
            'watchlist': self.watchlist
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create a User from a dictionary."""
        user = cls(
            id=data.get('id', 0),
            username=data.get('username', ''),
            email=data.get('email', ''),
            favorite_genres=data.get('favorite_genres', []),
            favorite_directors=data.get('favorite_directors', []),
            disliked_genres=data.get('disliked_genres', []),
            watch_history=data.get('watch_history', []),
            watchlist=data.get('watchlist', [])
        )

        if data.get('created_at'):
            try:
                user.created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                pass

        for rating_data in data.get('ratings', []):
            user.ratings.append(UserRating(
                movie_id=rating_data['movie_id'],
                rating=rating_data['rating'],
                timestamp=datetime.fromisoformat(rating_data.get('timestamp', datetime.now().isoformat()))
            ))

        return user

    def __str__(self) -> str:
        return f"{self.username} (ID: {self.id})"

    def __repr__(self) -> str:
        return f"User(id={self.id}, username='{self.username}')"
