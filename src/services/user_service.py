"""
User service for user management and personalization.
"""
# @author 成员 F — 前端框架 & API & 测试

from typing import Optional, Dict, List, Any
from datetime import datetime
import json
from pathlib import Path
import logging

from src.models.user import User, UserRating

logger = logging.getLogger(__name__)


class UserService:
    """
    Service for user management, preferences, and personalization.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the user service.

        Parameters
        ----------
        storage_path : Path, optional
            Path for user data storage (JSON file)
        """
        self.storage_path = storage_path or Path('data/users.json')
        self._users: Dict[int, User] = {}
        self._next_id = 1

        # Load existing users
        self._load_users()

    def _load_users(self):
        """Load users from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get('users', []):
                        user = User.from_dict(user_data)
                        self._users[user.id] = user
                        self._next_id = max(self._next_id, user.id + 1)
                logger.info(f"Loaded {len(self._users)} users")
            except Exception as e:
                logger.warning(f"Error loading users: {e}")

    def _save_users(self):
        """Save users to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'users': [user.to_dict() for user in self._users.values()]
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Users saved to storage")
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    def create_user(
        self,
        username: str,
        email: str,
        favorite_genres: Optional[List[str]] = None,
        favorite_directors: Optional[List[str]] = None
    ) -> User:
        """
        Create a new user.

        Parameters
        ----------
        username : str
            Username
        email : str
            Email address
        favorite_genres : list, optional
            List of favorite genres
        favorite_directors : list, optional
            List of favorite directors

        Returns
        -------
        User
            Created user
        """
        user = User(
            id=self._next_id,
            username=username,
            email=email,
            favorite_genres=favorite_genres or [],
            favorite_directors=favorite_directors or []
        )

        self._users[user.id] = user
        self._next_id += 1
        self._save_users()

        logger.info(f"Created user: {username} (ID: {user.id})")
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        """
        Get a user by ID.

        Parameters
        ----------
        user_id : int
            User ID

        Returns
        -------
        User or None
            User if found
        """
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by username.

        Parameters
        ----------
        username : str
            Username

        Returns
        -------
        User or None
            User if found
        """
        for user in self._users.values():
            if user.username.lower() == username.lower():
                return user
        return None

    def update_user(
        self,
        user_id: int,
        **kwargs
    ) -> Optional[User]:
        """
        Update user attributes.

        Parameters
        ----------
        user_id : int
            User ID
        **kwargs
            Attributes to update

        Returns
        -------
        User or None
            Updated user if found
        """
        user = self._users.get(user_id)
        if user is None:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        self._save_users()
        return user

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user.

        Parameters
        ----------
        user_id : int
            User ID

        Returns
        -------
        bool
            True if deleted
        """
        if user_id in self._users:
            del self._users[user_id]
            self._save_users()
            return True
        return False

    def add_rating(
        self,
        user_id: int,
        movie_id: int,
        rating: float
    ) -> Optional[User]:
        """
        Add a movie rating for a user.

        Parameters
        ----------
        user_id : int
            User ID
        movie_id : int
            Movie ID
        rating : float
            Rating (1-5)

        Returns
        -------
        User or None
            Updated user
        """
        user = self._users.get(user_id)
        if user is None:
            return None

        rating = max(1.0, min(5.0, rating))  # Clamp to 1-5
        user.add_rating(movie_id, rating)
        self._save_users()

        return user

    def get_user_ratings(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all ratings for a user.

        Parameters
        ----------
        user_id : int
            User ID

        Returns
        -------
        list
            List of ratings
        """
        user = self._users.get(user_id)
        if user is None:
            return []

        return [
            {
                'movie_id': r.movie_id,
                'rating': r.rating,
                'timestamp': r.timestamp.isoformat()
            }
            for r in user.ratings
        ]

    def add_to_watchlist(self, user_id: int, movie_id: int) -> bool:
        """
        Add a movie to user's watchlist.

        Parameters
        ----------
        user_id : int
            User ID
        movie_id : int
            Movie ID

        Returns
        -------
        bool
            True if added
        """
        user = self._users.get(user_id)
        if user is None:
            return False

        user.add_to_watchlist(movie_id)
        self._save_users()
        return True

    def remove_from_watchlist(self, user_id: int, movie_id: int) -> bool:
        """
        Remove a movie from user's watchlist.

        Parameters
        ----------
        user_id : int
            User ID
        movie_id : int
            Movie ID

        Returns
        -------
        bool
            True if removed
        """
        user = self._users.get(user_id)
        if user is None:
            return False

        user.remove_from_watchlist(movie_id)
        self._save_users()
        return True

    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user profile summary.

        Parameters
        ----------
        user_id : int
            User ID

        Returns
        -------
        dict or None
            User profile
        """
        user = self._users.get(user_id)
        if user is None:
            return None

        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.isoformat(),
            'favorite_genres': user.favorite_genres,
            'favorite_directors': user.favorite_directors,
            'num_ratings': user.num_ratings,
            'avg_rating': user.avg_rating,
            'watch_history_count': len(user.watch_history),
            'watchlist_count': len(user.watchlist),
            'liked_movies': user.liked_movies[:10],
            'disliked_movies': user.disliked_movies[:10]
        }

    def update_preferences(
        self,
        user_id: int,
        favorite_genres: Optional[List[str]] = None,
        favorite_directors: Optional[List[str]] = None,
        disliked_genres: Optional[List[str]] = None
    ) -> Optional[User]:
        """
        Update user preferences.

        Parameters
        ----------
        user_id : int
            User ID
        favorite_genres : list, optional
            Favorite genres
        favorite_directors : list, optional
            Favorite directors
        disliked_genres : list, optional
            Disliked genres

        Returns
        -------
        User or None
            Updated user
        """
        user = self._users.get(user_id)
        if user is None:
            return None

        if favorite_genres is not None:
            user.favorite_genres = favorite_genres
        if favorite_directors is not None:
            user.favorite_directors = favorite_directors
        if disliked_genres is not None:
            user.disliked_genres = disliked_genres

        self._save_users()
        return user

    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users (summary).

        Returns
        -------
        list
            List of user summaries
        """
        return [
            {
                'id': user.id,
                'username': user.username,
                'num_ratings': user.num_ratings
            }
            for user in self._users.values()
        ]

    @property
    def user_count(self) -> int:
        """Get total number of users."""
        return len(self._users)
