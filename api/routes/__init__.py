"""API route modules."""

from .recommendations import router as recommendations_router
from .movies import router as movies_router
from .users import router as users_router
from .health import router as health_router

__all__ = [
    "recommendations_router",
    "movies_router",
    "users_router",
    "health_router",
]
