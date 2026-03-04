"""
Cache manager for similarity matrices and computed data.
"""

import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of computed data like similarity matrices.
    """

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: int = 3600):
        """
        Initialize the cache manager.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory for cache files
        ttl_seconds : int
            Time-to-live for cache entries in seconds
        """
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.ttl_seconds = ttl_seconds
        self.enabled = settings.enable_caching

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for small objects
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Cache manager initialized. Directory: {self.cache_dir}")

    def _get_cache_key(self, name: str, params: Optional[Dict] = None) -> str:
        """
        Generate a cache key from name and parameters.

        Parameters
        ----------
        name : str
            Base name for the cache entry
        params : dict, optional
            Parameters to include in the key

        Returns
        -------
        str
            Cache key hash
        """
        key_data = {'name': name}
        if params:
            key_data['params'] = params

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get the metadata file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.meta.json"

    def _is_expired(self, cache_key: str) -> bool:
        """Check if a cache entry is expired."""
        metadata_path = self._get_metadata_path(cache_key)

        if not metadata_path.exists():
            return True

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            created_at = datetime.fromisoformat(metadata['created_at'])
            ttl = metadata.get('ttl_seconds', self.ttl_seconds)

            return datetime.now() > created_at + timedelta(seconds=ttl)
        except Exception as e:
            logger.warning(f"Error reading cache metadata: {e}")
            return True

    def get(
        self,
        name: str,
        params: Optional[Dict] = None,
        default: Any = None
    ) -> Any:
        """
        Get a cached value.

        Parameters
        ----------
        name : str
            Cache entry name
        params : dict, optional
            Parameters that were used to create the cached value
        default : Any
            Default value if cache miss

        Returns
        -------
        Any
            Cached value or default
        """
        if not self.enabled:
            return default

        cache_key = self._get_cache_key(name, params)

        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if datetime.now() < entry['expires_at']:
                logger.debug(f"Memory cache hit: {name}")
                return entry['value']
            else:
                del self._memory_cache[cache_key]

        # Check file cache
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {name}")
            return default

        if self._is_expired(cache_key):
            logger.debug(f"Cache expired: {name}")
            self._delete_cache_entry(cache_key)
            return default

        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            logger.info(f"Cache hit: {name}")
            return value
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return default

    def set(
        self,
        name: str,
        value: Any,
        params: Optional[Dict] = None,
        ttl_seconds: Optional[int] = None,
        memory_only: bool = False
    ) -> bool:
        """
        Set a cached value.

        Parameters
        ----------
        name : str
            Cache entry name
        value : Any
            Value to cache
        params : dict, optional
            Parameters used to create the value
        ttl_seconds : int, optional
            Custom TTL for this entry
        memory_only : bool
            Only cache in memory (for small objects)

        Returns
        -------
        bool
            True if caching succeeded
        """
        if not self.enabled:
            return False

        cache_key = self._get_cache_key(name, params)
        ttl = ttl_seconds or self.ttl_seconds

        # Memory cache
        self._memory_cache[cache_key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }

        if memory_only:
            return True

        # File cache
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)

            # Save value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            # Save metadata
            metadata = {
                'name': name,
                'params': params,
                'created_at': datetime.now().isoformat(),
                'ttl_seconds': ttl,
                'size_bytes': cache_path.stat().st_size
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            logger.info(f"Cached: {name} ({cache_path.stat().st_size / 1024:.1f} KB)")
            return True

        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            return False

    def delete(self, name: str, params: Optional[Dict] = None) -> bool:
        """
        Delete a cached value.

        Parameters
        ----------
        name : str
            Cache entry name
        params : dict, optional
            Parameters used to create the value

        Returns
        -------
        bool
            True if deletion succeeded
        """
        cache_key = self._get_cache_key(name, params)
        return self._delete_cache_entry(cache_key)

    def _delete_cache_entry(self, cache_key: str) -> bool:
        """Delete a cache entry by key."""
        try:
            # Memory cache
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]

            # File cache
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)

            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

            return True
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns
        -------
        int
            Number of entries cleared
        """
        count = 0

        # Clear memory cache
        count += len(self._memory_cache)
        self._memory_cache.clear()

        # Clear file cache
        for path in self.cache_dir.glob("*.pkl"):
            try:
                path.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error deleting {path}: {e}")

        for path in self.cache_dir.glob("*.meta.json"):
            try:
                path.unlink()
            except Exception:
                pass

        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Cache statistics
        """
        total_size = 0
        entry_count = 0
        expired_count = 0

        for path in self.cache_dir.glob("*.pkl"):
            entry_count += 1
            total_size += path.stat().st_size

            cache_key = path.stem
            if self._is_expired(cache_key):
                expired_count += 1

        return {
            'enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'entry_count': entry_count,
            'total_size_mb': total_size / (1024 * 1024),
            'memory_entries': len(self._memory_cache),
            'expired_entries': expired_count
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns
        -------
        int
            Number of entries removed
        """
        count = 0

        for path in self.cache_dir.glob("*.pkl"):
            cache_key = path.stem
            if self._is_expired(cache_key):
                if self._delete_cache_entry(cache_key):
                    count += 1

        logger.info(f"Cleaned up {count} expired cache entries")
        return count


# Convenience function for caching similarity matrices
def cache_similarity_matrix(
    name: str,
    compute_func,
    params: Optional[Dict] = None,
    cache_manager: Optional[CacheManager] = None
) -> np.ndarray:
    """
    Get or compute a similarity matrix with caching.

    Parameters
    ----------
    name : str
        Matrix name
    compute_func : callable
        Function to compute the matrix if not cached
    params : dict, optional
        Parameters for cache key
    cache_manager : CacheManager, optional
        Cache manager instance

    Returns
    -------
    np.ndarray
        Similarity matrix
    """
    if cache_manager is None:
        cache_manager = CacheManager()

    cached = cache_manager.get(name, params)
    if cached is not None:
        return cached

    logger.info(f"Computing {name}...")
    matrix = compute_func()

    cache_manager.set(name, matrix, params)
    return matrix
