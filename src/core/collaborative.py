"""
Collaborative filtering recommendation using user-item interactions.
"""

from typing import Optional, List, Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import logging

from .base_recommender import BaseRecommender
from config.settings import get_settings

logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender(BaseRecommender):
    """
    Collaborative filtering recommender using simulated user-item interactions.

    Supports three methods:
    - Item-based: Find similar items based on user co-ratings
    - User-based: Find similar users and recommend their favorites
    - SVD: Matrix factorization for latent factor modeling
    """

    def __init__(
        self,
        n_users: int = 1000,
        sparsity: float = 0.02,
        n_factors: int = 50,
        n_neighbors: int = 20,
        random_seed: int = 42
    ):
        """
        Initialize the collaborative filtering recommender.

        Parameters
        ----------
        n_users : int
            Number of simulated users
        sparsity : float
            Sparsity of the user-movie matrix
        n_factors : int
            Number of latent factors for SVD
        n_neighbors : int
            Number of neighbors for KNN
        random_seed : int
            Random seed for reproducibility
        """
        super().__init__(name="CollaborativeFilteringRecommender")

        settings = get_settings()
        self.n_users = n_users or settings.cf_n_users
        self.sparsity = sparsity or settings.cf_sparsity
        self.n_factors = n_factors or settings.svd_n_factors
        self.n_neighbors = n_neighbors or settings.knn_n_neighbors
        self.random_seed = random_seed

        # Matrices and models
        self._user_movie_matrix: Optional[np.ndarray] = None
        self._item_similarity_cf: Optional[np.ndarray] = None
        self._user_similarity_cf: Optional[np.ndarray] = None
        self._item_knn: Optional[NearestNeighbors] = None
        self._user_knn: Optional[NearestNeighbors] = None

        # SVD components
        self._svd_user_factors: Optional[np.ndarray] = None
        self._svd_sigma: Optional[np.ndarray] = None
        self._svd_item_factors: Optional[np.ndarray] = None
        self._svd_predictions: Optional[np.ndarray] = None
        self._user_ratings_mean: Optional[np.ndarray] = None

    def _try_load_movielens(self, movies_df: pd.DataFrame) -> np.ndarray:
        """Try to load real MovieLens ratings; fall back to simulated."""
        try:
            from src.data.movielens_loader import MovieLensLoader
            ml = MovieLensLoader()
            if ml.ratings_path.exists():
                logger.info("MovieLens data found – building real rating matrix…")
                matrix, _, _ = ml.build_rating_matrix(
                    movies_df, min_ratings_per_user=5, max_users=self.n_users
                )
                return matrix
        except Exception as e:
            logger.warning(f"MovieLens loading failed ({e}), using simulated ratings.")
        return self._generate_user_movie_matrix(movies_df)

    def _generate_user_movie_matrix(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Generate a simulated user-movie rating matrix.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe

        Returns
        -------
        np.ndarray
            User-movie rating matrix
        """
        logger.info("Generating simulated user-movie rating matrix...")

        n_movies = len(movies_df)
        ratings = np.zeros((self.n_users, n_movies))

        # Get all genres
        all_genres = set()
        for genres in movies_df['genres_list']:
            if isinstance(genres, list):
                all_genres.update(genres)
        genre_list = sorted(list(all_genres))

        np.random.seed(self.random_seed)

        for user_id in range(self.n_users):
            # User preferences (1-3 random genres)
            n_preferred = np.random.randint(1, 4)
            preferred_genres = np.random.choice(
                genre_list,
                size=min(n_preferred, len(genre_list)),
                replace=False
            )

            # Number of movies to watch
            n_watched = int(n_movies * self.sparsity * np.random.uniform(0.5, 2.0))
            n_watched = max(10, min(n_watched, n_movies // 10))

            # Calculate movie selection probabilities
            movie_probs = np.ones(n_movies) * 0.1

            for movie_idx in range(n_movies):
                movie_genres = movies_df.iloc[movie_idx].get('genres_list', [])
                if isinstance(movie_genres, list):
                    # Higher probability for preferred genres
                    if any(g in preferred_genres for g in movie_genres):
                        movie_probs[movie_idx] = 0.8

                # Higher probability for highly rated movies
                vote_avg = movies_df.iloc[movie_idx].get('vote_average', 5)
                movie_probs[movie_idx] *= (vote_avg / 10.0 + 0.5)

            # Normalize probabilities
            movie_probs = movie_probs / movie_probs.sum()

            # Select movies to watch
            watched_movies = np.random.choice(
                n_movies,
                size=n_watched,
                replace=False,
                p=movie_probs
            )

            # Generate ratings
            for movie_idx in watched_movies:
                movie_genres = movies_df.iloc[movie_idx].get('genres_list', [])
                vote_avg = movies_df.iloc[movie_idx].get('vote_average', 5)

                # Base rating
                base_rating = 2.5 + (vote_avg - 5) / 2

                # Adjust for user preferences
                if isinstance(movie_genres, list) and any(g in preferred_genres for g in movie_genres):
                    base_rating += np.random.uniform(0.5, 1.5)
                else:
                    base_rating += np.random.uniform(-1.0, 0.5)

                # Add noise and clip
                rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)
                ratings[user_id, movie_idx] = round(rating, 1)

        # Statistics
        n_ratings = np.count_nonzero(ratings)
        actual_sparsity = 1 - (n_ratings / (self.n_users * n_movies))

        logger.info(f"User-movie matrix shape: {ratings.shape}")
        logger.info(f"Number of ratings: {n_ratings}")
        logger.info(f"Actual sparsity: {actual_sparsity:.2%}")

        return ratings

    def fit(self, movies_df: pd.DataFrame, **kwargs) -> 'CollaborativeFilteringRecommender':
        """
        Fit the collaborative filtering model.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe
        **kwargs
            method : str - 'item_based', 'user_based', 'svd', or 'all'
            rating_matrix : np.ndarray - pre-built rating matrix (optional)

        Returns
        -------
        CollaborativeFilteringRecommender
            Self for method chaining
        """
        logger.info("Fitting CollaborativeFilteringRecommender...")

        # Create index mappings
        self._create_index_mappings(movies_df)

        # Use provided rating matrix, try MovieLens, or fall back to simulated
        external_matrix = kwargs.get('rating_matrix', None)
        if external_matrix is not None:
            logger.info("Using externally provided rating matrix.")
            self._user_movie_matrix = external_matrix
        else:
            self._user_movie_matrix = self._try_load_movielens(self._movies_df)

        # Get method to build
        method = kwargs.get('method', 'all')

        if method == 'item_based' or method == 'all':
            self._build_item_based()

        if method == 'user_based' or method == 'all':
            self._build_user_based()

        if method == 'svd' or method == 'all':
            self._build_svd()

        # Use item similarity as default
        self._similarity_matrix = self._item_similarity_cf

        self._is_fitted = True
        logger.info("CollaborativeFilteringRecommender fitting complete.")

        return self

    def _build_item_based(self):
        """Build item-based collaborative filtering model."""
        logger.info("Building item-based CF model...")

        # Transpose: movies in rows
        movie_user_matrix = self._user_movie_matrix.T

        # Build KNN model
        sparse_matrix = csr_matrix(movie_user_matrix)
        self._item_knn = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=self.n_neighbors
        )
        self._item_knn.fit(sparse_matrix)

        # Compute full similarity matrix
        self._item_similarity_cf = cosine_similarity(movie_user_matrix)

        logger.info(f"Item similarity matrix shape: {self._item_similarity_cf.shape}")

    def _build_user_based(self):
        """Build user-based collaborative filtering model."""
        logger.info("Building user-based CF model...")

        # Build KNN model
        sparse_matrix = csr_matrix(self._user_movie_matrix)
        self._user_knn = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=self.n_neighbors
        )
        self._user_knn.fit(sparse_matrix)

        # Compute user similarity matrix
        self._user_similarity_cf = cosine_similarity(self._user_movie_matrix)

        logger.info(f"User similarity matrix shape: {self._user_similarity_cf.shape}")

    def _build_svd(self):
        """Build SVD-based collaborative filtering model."""
        logger.info("Building SVD-based CF model...")

        # Center ratings
        self._user_ratings_mean = np.mean(self._user_movie_matrix, axis=1)
        ratings_centered = self._user_movie_matrix - self._user_ratings_mean.reshape(-1, 1)

        # SVD decomposition
        n_factors = min(self.n_factors, min(self._user_movie_matrix.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(ratings_centered), k=n_factors)

        # Store components
        self._svd_user_factors = U
        self._svd_sigma = np.diag(sigma)
        self._svd_item_factors = Vt

        # Predict all ratings
        self._svd_predictions = (
            np.dot(np.dot(U, self._svd_sigma), Vt) +
            self._user_ratings_mean.reshape(-1, 1)
        )

        logger.info(f"SVD factors - U: {U.shape}, V: {Vt.shape}")
        logger.info(f"Number of latent factors: {n_factors}")

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        method: Literal['item_based', 'user_based', 'svd'] = 'item_based',
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get collaborative filtering recommendations.

        Parameters
        ----------
        title : str
            Movie title
        top_n : int
            Number of recommendations
        method : str
            CF method to use
        **kwargs
            Additional parameters (unused)

        Returns
        -------
        pd.DataFrame or None
            Recommendations or None if movie not found
        """
        if not self._validate_fitted():
            return None

        idx = self.get_movie_index(title)
        if idx is None:
            logger.warning(f"Movie not found: {title}")
            matches = self.find_similar_titles(title)
            if matches:
                logger.info(f"Similar titles: {matches}")
            return None

        if method == 'item_based':
            return self._recommend_item_based(idx, top_n)
        elif method == 'user_based':
            return self._recommend_user_based(idx, top_n)
        elif method == 'svd':
            return self._recommend_svd(idx, top_n)
        else:
            return self._recommend_item_based(idx, top_n)

    def _recommend_item_based(self, movie_idx: int, top_n: int) -> pd.DataFrame:
        """Get item-based recommendations."""
        if self._item_similarity_cf is None:
            self._build_item_based()

        sim_scores = list(enumerate(self._item_similarity_cf[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        columns = ['title', 'genres_list', 'vote_average', 'popularity']
        recommendations = self._format_recommendations(movie_indices, scores, columns)
        recommendations.rename(columns={'similarity_score': 'cf_similarity'}, inplace=True)

        return recommendations

    def _recommend_user_based(self, movie_idx: int, top_n: int) -> pd.DataFrame:
        """Get user-based recommendations."""
        if self._user_similarity_cf is None:
            self._build_user_based()

        # Find users who liked this movie
        movie_ratings = self._user_movie_matrix[:, movie_idx]
        liked_users = np.where(movie_ratings >= 4.0)[0]

        if len(liked_users) == 0:
            liked_users = np.where(movie_ratings > 0)[0][:10]

        if len(liked_users) == 0:
            return self._recommend_item_based(movie_idx, top_n)

        # Aggregate preferences
        user_preferences = np.zeros(len(self._movies_df))

        for user_id in liked_users:
            similar_users = np.argsort(self._user_similarity_cf[user_id])[::-1][1:11]
            for sim_user in similar_users:
                sim_score = self._user_similarity_cf[user_id, sim_user]
                user_preferences += sim_score * self._user_movie_matrix[sim_user]

        # Exclude query movie
        user_preferences[movie_idx] = -1

        # Get top recommendations
        top_indices = np.argsort(user_preferences)[::-1][:top_n]
        scores = user_preferences[top_indices]

        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()

        columns = ['title', 'genres_list', 'vote_average', 'popularity']
        recommendations = self._format_recommendations(top_indices.tolist(), scores.tolist(), columns)
        recommendations.rename(columns={'similarity_score': 'cf_score'}, inplace=True)

        return recommendations

    def _recommend_svd(self, movie_idx: int, top_n: int) -> pd.DataFrame:
        """Get SVD-based recommendations."""
        if self._svd_item_factors is None:
            self._build_svd()

        # Use item factors to compute similarity
        item_factors = self._svd_item_factors.T
        target_factor = item_factors[movie_idx].reshape(1, -1)
        similarities = cosine_similarity(target_factor, item_factors)[0]

        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        columns = ['title', 'genres_list', 'vote_average', 'popularity']
        recommendations = self._format_recommendations(movie_indices, scores, columns)
        recommendations.rename(columns={'similarity_score': 'svd_similarity'}, inplace=True)

        return recommendations

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        method: Literal['svd', 'user_based'] = 'svd'
    ) -> Optional[pd.DataFrame]:
        """
        Get recommendations for a specific user.

        Parameters
        ----------
        user_id : int
            User ID
        top_n : int
            Number of recommendations
        method : str
            Method to use ('svd' or 'user_based')

        Returns
        -------
        pd.DataFrame or None
            Personalized recommendations
        """
        if not self._validate_fitted():
            return None

        if user_id >= self.n_users:
            logger.warning(f"User ID out of range (max: {self.n_users - 1})")
            return None

        # Get movies user has already watched
        watched = np.where(self._user_movie_matrix[user_id] > 0)[0]

        if method == 'svd':
            if self._svd_predictions is None:
                self._build_svd()

            predictions = self._svd_predictions[user_id].copy()
            predictions[watched] = -1

            top_indices = np.argsort(predictions)[::-1][:top_n]
            scores = predictions[top_indices]
        else:
            if self._user_similarity_cf is None:
                self._build_user_based()

            similar_users = np.argsort(self._user_similarity_cf[user_id])[::-1][1:21]

            scores_agg = np.zeros(len(self._movies_df))
            for sim_user in similar_users:
                sim_score = self._user_similarity_cf[user_id, sim_user]
                scores_agg += sim_score * self._user_movie_matrix[sim_user]

            scores_agg[watched] = -1
            top_indices = np.argsort(scores_agg)[::-1][:top_n]
            scores = scores_agg[top_indices]

        columns = ['title', 'genres_list', 'vote_average', 'popularity']
        recommendations = self._format_recommendations(top_indices.tolist(), scores.tolist(), columns)
        recommendations.rename(columns={'similarity_score': 'predicted_score'}, inplace=True)

        return recommendations

    @property
    def user_movie_matrix(self) -> Optional[np.ndarray]:
        """Get the user-movie rating matrix."""
        return self._user_movie_matrix

    @property
    def item_similarity(self) -> Optional[np.ndarray]:
        """Get the item similarity matrix."""
        return self._item_similarity_cf

    @property
    def user_similarity(self) -> Optional[np.ndarray]:
        """Get the user similarity matrix."""
        return self._user_similarity_cf
