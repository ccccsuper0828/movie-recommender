"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def sample_movies_data():
    """Create sample movie data for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'The Matrix',
            'The Matrix Reloaded',
            'Inception',
            'Interstellar',
            'The Dark Knight',
            'Pulp Fiction',
            'Fight Club',
            'The Godfather',
            'The Shawshank Redemption',
            'Forrest Gump'
        ],
        'overview': [
            'A computer hacker learns about the true nature of reality.',
            'Neo and the rebels fight the machines in the Matrix.',
            'A thief enters dreams to steal secrets from the subconscious.',
            'Explorers travel through a wormhole to save humanity.',
            'Batman faces the Joker in Gotham City.',
            'The lives of two mob hitmen intertwine in unexpected ways.',
            'An insomniac office worker forms an underground fight club.',
            'The aging patriarch of a crime dynasty transfers control.',
            'Two imprisoned men bond over years finding redemption.',
            'A man with a low IQ accomplishes great things.'
        ],
        'genres_list': [
            ['Action', 'Science Fiction'],
            ['Action', 'Science Fiction'],
            ['Action', 'Science Fiction', 'Thriller'],
            ['Adventure', 'Drama', 'Science Fiction'],
            ['Action', 'Crime', 'Drama'],
            ['Crime', 'Drama'],
            ['Drama', 'Thriller'],
            ['Crime', 'Drama'],
            ['Drama'],
            ['Comedy', 'Drama', 'Romance']
        ],
        'vote_average': [8.7, 7.2, 8.4, 8.6, 9.0, 8.9, 8.4, 9.2, 9.3, 8.8],
        'vote_count': [24000, 11000, 33000, 30000, 28000, 25000, 26000, 18000, 24000, 20000],
        'popularity': [83.0, 51.0, 84.0, 72.0, 94.0, 68.0, 73.0, 52.0, 77.0, 65.0],
        'release_date': [
            '1999-03-30', '2003-05-15', '2010-07-15', '2014-11-05',
            '2008-07-16', '1994-09-10', '1999-10-15', '1972-03-14',
            '1994-09-23', '1994-06-23'
        ],
        'director': [
            'Lana Wachowski', 'Lana Wachowski', 'Christopher Nolan',
            'Christopher Nolan', 'Christopher Nolan', 'Quentin Tarantino',
            'David Fincher', 'Francis Ford Coppola', 'Frank Darabont',
            'Robert Zemeckis'
        ],
        'cast_list': [
            ['Keanu Reeves', 'Laurence Fishburne', 'Carrie-Anne Moss'],
            ['Keanu Reeves', 'Laurence Fishburne', 'Carrie-Anne Moss'],
            ['Leonardo DiCaprio', 'Joseph Gordon-Levitt', 'Tom Hardy'],
            ['Matthew McConaughey', 'Anne Hathaway', 'Jessica Chastain'],
            ['Christian Bale', 'Heath Ledger', 'Aaron Eckhart'],
            ['John Travolta', 'Uma Thurman', 'Samuel L. Jackson'],
            ['Brad Pitt', 'Edward Norton', 'Helena Bonham Carter'],
            ['Marlon Brando', 'Al Pacino', 'James Caan'],
            ['Tim Robbins', 'Morgan Freeman', 'Bob Gunton'],
            ['Tom Hanks', 'Robin Wright', 'Gary Sinise']
        ],
        'keywords_list': [
            ['artificial intelligence', 'dystopia', 'virtual reality'],
            ['artificial intelligence', 'sequel', 'fight'],
            ['dream', 'heist', 'subconscious'],
            ['space', 'wormhole', 'survival'],
            ['vigilante', 'villain', 'chaos'],
            ['nonlinear', 'crime', 'dialogue'],
            ['identity', 'anti-hero', 'violence'],
            ['mafia', 'family', 'power'],
            ['prison', 'hope', 'friendship'],
            ['history', 'love', 'journey']
        ],
        'revenue': [
            463517383, 742128461, 836836967, 677471339, 1004558444,
            213928762, 100853753, 245066411, 58300000, 677387716
        ],
        'budget': [
            63000000, 150000000, 160000000, 165000000, 185000000,
            8000000, 63000000, 6000000, 25000000, 55000000
        ],
        'runtime': [136, 138, 148, 169, 152, 154, 139, 175, 142, 142]
    })


@pytest.fixture(scope="session")
def sample_movies_df(sample_movies_data):
    """Return sample movies dataframe with necessary preprocessing."""
    df = sample_movies_data.copy()

    # Add combined features for content-based filtering
    df['combined_features'] = df.apply(
        lambda x: ' '.join([
            str(x['overview'] or ''),
            ' '.join(x.get('genres_list', [])),
            str(x.get('director', '') or ''),
            ' '.join(x.get('cast_list', [])[:3]),
            ' '.join(x.get('keywords_list', []))
        ]),
        axis=1
    )

    # Fields expected by metadata recommender implementation
    df["genres_clean"] = df["genres_list"].apply(
        lambda xs: [str(x).lower().replace(" ", "") for x in (xs or [])]
    )
    df["cast_clean"] = df["cast_list"].apply(
        lambda xs: [str(x).lower().replace(" ", "") for x in (xs or [])]
    )
    df["keywords_clean"] = df["keywords_list"].apply(
        lambda xs: [str(x).lower().replace(" ", "") for x in (xs or [])]
    )
    df["director_clean"] = df["director"].apply(
        lambda x: str(x).lower().replace(" ", "") if x else ""
    )

    return df


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_user_ratings():
    """Create sample user ratings for collaborative filtering testing."""
    # Create a user-movie ratings matrix
    np.random.seed(42)
    n_users = 50
    n_movies = 10

    ratings = np.zeros((n_users, n_movies))

    # Add some ratings (not all users rate all movies)
    for user_id in range(n_users):
        # Each user rates 3-7 movies
        n_ratings = np.random.randint(3, 8)
        movie_ids = np.random.choice(n_movies, n_ratings, replace=False)
        for movie_id in movie_ids:
            ratings[user_id, movie_id] = np.random.randint(1, 6)

    return ratings


@pytest.fixture
def content_recommender(sample_movies_df):
    """Create and fit a content-based recommender."""
    from src.core import ContentBasedRecommender

    recommender = ContentBasedRecommender()
    recommender.fit(sample_movies_df)
    return recommender


@pytest.fixture
def metadata_recommender(sample_movies_df):
    """Create and fit a metadata-based recommender."""
    from src.core import MetadataBasedRecommender

    recommender = MetadataBasedRecommender()
    recommender.fit(sample_movies_df)
    return recommender


@pytest.fixture
def collaborative_recommender(sample_movies_df, sample_user_ratings):
    """Create and fit a collaborative filtering recommender."""
    from src.core import CollaborativeFilteringRecommender

    recommender = CollaborativeFilteringRecommender(
        n_users=80,
        random_seed=42
    )
    recommender.fit(sample_movies_df)
    return recommender


@pytest.fixture
def hybrid_recommender(sample_movies_df):
    """Create and fit a hybrid recommender."""
    from src.core import HybridRecommender

    recommender = HybridRecommender()
    recommender.fit(sample_movies_df)
    return recommender


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


@pytest.fixture
def mock_recommendation_service(sample_movies_df):
    """Create a mock recommendation service for testing."""
    from unittest.mock import Mock

    service = Mock()
    service.movies_df = sample_movies_df
    service.get_recommendations.return_value = sample_movies_df.head(5)

    return service
