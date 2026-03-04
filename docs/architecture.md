# CineMatch Architecture Documentation

## Overview

CineMatch is a modular movie recommendation system built with a clean, layered architecture designed for team collaboration and maintainability.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                  │
│                    (Streamlit Application)                        │
│  ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ Pages    │  │ Components  │  │ Styles   │  │ Theme Config  │ │
│  └──────────┘  └─────────────┘  └──────────┘  └───────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API Layer                                   │
│                     (FastAPI REST API)                            │
│  ┌──────────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Recommendations  │  │ Movies      │  │ Users               │ │
│  │ Routes           │  │ Routes      │  │ Routes              │ │
│  └──────────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Service Layer                                 │
│  ┌───────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Recommendation    │  │ Search          │  │ User           │ │
│  │ Service           │  │ Service         │  │ Service        │ │
│  └───────────────────┘  └─────────────────┘  └────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Analytics Service                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                                 │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────────┐ │
│  │ Content-Based │  │ Metadata-Based │  │ Collaborative       │ │
│  │ Recommender   │  │ Recommender    │  │ Filtering           │ │
│  └───────────────┘  └────────────────┘  └─────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Hybrid Recommender                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Explainability Module                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                 │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │ DataLoader  │  │ Preprocessor   │  │ CacheManager          │ │
│  └─────────────┘  └────────────────┘  └───────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Storage                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ Raw CSV Files   │  │ Processed Data   │  │ Cached Matrices │ │
│  │ (TMDB Dataset)  │  │                  │  │                 │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
MOVIE/
├── config/                 # Configuration management
│   ├── settings.py         # Environment-based settings
│   └── logging_config.py   # Logging configuration
│
├── src/                    # Core source code
│   ├── core/               # Recommendation algorithms
│   ├── data/               # Data handling
│   ├── explainability/     # Explanation modules
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   └── utils/              # Utilities
│
├── api/                    # REST API
│   ├── main.py             # FastAPI application
│   ├── dependencies.py     # Dependency injection
│   └── routes/             # API endpoints
│
├── frontend/               # Web interface
│   ├── app.py              # Main Streamlit app
│   ├── components/         # UI components
│   ├── pages/              # Multi-page routing
│   └── styles/             # Theming
│
├── analytics/              # Analytics module
├── tests/                  # Test suite
├── docs/                   # Documentation
└── data/                   # Data files
```

## Component Details

### Core Recommendation Algorithms

#### 1. Content-Based Recommender
- **Algorithm**: TF-IDF vectorization + Cosine similarity
- **Features**: Movie overviews, genres, keywords
- **Strength**: Finds movies with similar themes and storylines

```python
from src.core import ContentBasedRecommender

recommender = ContentBasedRecommender()
recommender.fit(movies_df)
recommendations = recommender.recommend("The Matrix", n_recommendations=10)
```

#### 2. Metadata-Based Recommender
- **Algorithm**: Weighted feature matching
- **Features**: Genres, directors, cast, keywords
- **Strength**: Finds movies with similar production elements

```python
from src.core import MetadataBasedRecommender

recommender = MetadataBasedRecommender(
    genre_weight=2.0,
    director_weight=1.5,
    cast_weight=1.0
)
recommender.fit(movies_df)
```

#### 3. Collaborative Filtering Recommender
- **Algorithms**: Item-based CF, User-based CF, SVD
- **Data**: User-movie rating matrix
- **Strength**: Discovers hidden patterns based on user behavior

#### 4. Hybrid Recommender
- **Strategy**: Weighted combination of all methods
- **Customization**: Adjustable weights per method

```python
from src.core import HybridRecommender

recommender = HybridRecommender()
recommender.set_weights(
    content_weight=0.3,
    metadata_weight=0.4,
    cf_weight=0.3
)
```

### Explainability Module

#### Rule-Based Explainer
Provides human-readable explanations based on feature matching:
- Same director
- Common genres
- Shared cast members
- Similar keywords

#### SHAP Explainer
Uses SHAP (SHapley Additive exPlanations) for ML-based explanations:
- Feature importance values
- Positive/negative contributors
- Visual explanations

### Service Layer

Services provide a clean interface between the API and core algorithms:

| Service | Responsibility |
|---------|---------------|
| RecommendationService | Orchestrates recommendation generation |
| SearchService | Movie search with fuzzy matching |
| UserService | User management and preferences |
| AnalyticsService | System metrics and statistics |

### API Layer

RESTful API built with FastAPI:

| Endpoint | Method | Description |
|----------|--------|-------------|
| /recommendations | GET | Get movie recommendations |
| /recommendations/batch | POST | Batch recommendations |
| /movies | GET | List/search movies |
| /movies/{id} | GET | Movie details |
| /users | POST | Create user |
| /users/{id}/recommendations | GET | Personalized recommendations |
| /health | GET | Health check |

## Data Flow

### Recommendation Request Flow

```
1. User selects movie in frontend
          │
          ▼
2. Frontend calls API endpoint
          │
          ▼
3. RecommendationService receives request
          │
          ▼
4. Service calls appropriate recommender(s)
          │
          ▼
5. Recommenders compute similarity scores
          │
          ▼
6. Results aggregated and ranked
          │
          ▼
7. ExplainabilityModule generates explanations
          │
          ▼
8. Response returned to frontend
          │
          ▼
9. UI displays recommendations with explanations
```

### Data Processing Flow

```
Raw CSV Files (TMDB)
        │
        ▼
DataLoader (parse JSON, load data)
        │
        ▼
DataPreprocessor (clean, extract features)
        │
        ▼
CacheManager (cache similarity matrices)
        │
        ▼
Recommenders (fit on processed data)
```

## Design Patterns

### 1. Abstract Base Class
All recommenders extend `BaseRecommender`:

```python
class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def recommend(self, title: str, n: int) -> pd.DataFrame:
        pass
```

### 2. Dependency Injection
FastAPI dependencies for service injection:

```python
def get_recommendation_service():
    return RecommendationService(get_movies_df())
```

### 3. Strategy Pattern
Hybrid recommender uses strategy pattern for method selection:

```python
recommender.recommend(movie, method="content")
recommender.recommend(movie, method="metadata")
recommender.recommend(movie, method="hybrid")
```

### 4. Caching
Similarity matrices are cached for performance:

```python
cache = CacheManager()
cache.save("content_similarity", similarity_matrix)
loaded = cache.load("content_similarity")
```

## Scalability Considerations

### Current Limitations
- In-memory similarity matrices
- Single-machine architecture
- JSON file storage for users

### Future Improvements
- Redis caching for distributed cache
- PostgreSQL for persistent storage
- Kubernetes for horizontal scaling
- Async processing for batch jobs

## Security

### Current Measures
- CORS configuration
- Input validation with Pydantic
- No sensitive data in logs

### Recommendations
- Add authentication (JWT)
- Rate limiting
- API key management
- HTTPS enforcement

## Testing Strategy

### Unit Tests
- Test individual recommenders
- Test service methods
- Test data processing

### Integration Tests
- Test API endpoints
- Test end-to-end flows
- Test error handling

### Running Tests
```bash
pytest tests/ -v
pytest tests/unit/ -v --cov=src
pytest tests/integration/ -v
```
