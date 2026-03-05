"""
Movie data model.
"""
# @author 成员 F — 前端框架 & API & 测试

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date


@dataclass
class Movie:
    """
    Movie data class representing a movie entity.
    """
    id: int
    title: str
    overview: str = ""
    genres: List[str] = field(default_factory=list)
    director: str = ""
    cast: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    vote_average: float = 0.0
    vote_count: int = 0
    popularity: float = 0.0
    release_date: Optional[date] = None
    runtime: int = 0
    budget: int = 0
    revenue: int = 0
    original_language: str = "en"

    @property
    def year(self) -> Optional[int]:
        """Get the release year."""
        return self.release_date.year if self.release_date else None

    @property
    def profit(self) -> int:
        """Calculate profit."""
        return self.revenue - self.budget

    @property
    def roi(self) -> Optional[float]:
        """Calculate return on investment."""
        if self.budget > 0:
            return (self.revenue - self.budget) / self.budget * 100
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'overview': self.overview,
            'genres': self.genres,
            'director': self.director,
            'cast': self.cast,
            'keywords': self.keywords,
            'vote_average': self.vote_average,
            'vote_count': self.vote_count,
            'popularity': self.popularity,
            'release_date': self.release_date.isoformat() if self.release_date else None,
            'runtime': self.runtime,
            'budget': self.budget,
            'revenue': self.revenue,
            'original_language': self.original_language
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Movie':
        """Create a Movie from a dictionary."""
        release_date = None
        if data.get('release_date'):
            try:
                release_date = date.fromisoformat(data['release_date'])
            except (ValueError, TypeError):
                pass

        return cls(
            id=data.get('id', 0),
            title=data.get('title', ''),
            overview=data.get('overview', ''),
            genres=data.get('genres', []),
            director=data.get('director', ''),
            cast=data.get('cast', []),
            keywords=data.get('keywords', []),
            vote_average=data.get('vote_average', 0.0),
            vote_count=data.get('vote_count', 0),
            popularity=data.get('popularity', 0.0),
            release_date=release_date,
            runtime=data.get('runtime', 0),
            budget=data.get('budget', 0),
            revenue=data.get('revenue', 0),
            original_language=data.get('original_language', 'en')
        )

    def __str__(self) -> str:
        return f"{self.title} ({self.year or 'N/A'})"

    def __repr__(self) -> str:
        return f"Movie(id={self.id}, title='{self.title}')"
