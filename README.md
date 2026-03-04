# 🎬 CineMatch - Movie Recommendation System

A professional, modular movie recommendation system featuring multiple ML algorithms, a REST API, and an interactive web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### Recommendation Algorithms

1. **Content-Based Filtering** - TF-IDF vectorization of movie descriptions with cosine similarity
2. **Metadata-Based Filtering** - Weighted combination of genres, directors, cast, and keywords
3. **Collaborative Filtering** - Item-based CF, User-based CF, and SVD matrix factorization
4. **Hybrid Recommender** - Intelligent combination of all methods with customizable weights

### Explainability

- **Rule-Based Explanations** - Human-readable reasons for recommendations
- **SHAP Integration** - Feature importance analysis for ML transparency

### Technical Features

- 🚀 **FastAPI REST API** - Production-ready backend with OpenAPI documentation
- 🎨 **Modern Web UI** - Cinema-inspired Streamlit frontend
- 📊 **Analytics Dashboard** - System metrics and performance tracking
- 🧪 **Comprehensive Tests** - Unit and integration test suite
- 🐳 **Docker Support** - Containerized deployment ready

## 📊 Dataset

Uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata):
- ~4,800 movies with rich metadata
- Genres, keywords, cast, and crew information
- Plot descriptions and user ratings

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cinematch.git
cd cinematch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data files to data/raw/
# - tmdb_5000_movies.csv
# - tmdb_5000_credits.csv
```

### Run the Application

```bash
# Start the Streamlit frontend
streamlit run frontend/app.py

# Start the FastAPI backend (in another terminal)
uvicorn api.main:app --reload
```

Access:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📁 Project Structure

```
MOVIE/
├── config/                 # Configuration management
│   ├── settings.py         # Environment-based settings
│   └── logging_config.py   # Logging configuration
│
├── src/                    # Core source code
│   ├── core/               # Recommendation algorithms
│   │   ├── base_recommender.py
│   │   ├── content_based.py
│   │   ├── metadata_based.py
│   │   ├── collaborative.py
│   │   └── hybrid.py
│   ├── data/               # Data loading and preprocessing
│   ├── explainability/     # Explanation generation
│   ├── models/             # Data models and schemas
│   ├── services/           # Business logic layer
│   └── utils/              # Utility functions
│
├── api/                    # FastAPI REST API
│   ├── main.py
│   ├── dependencies.py
│   └── routes/
│
├── frontend/               # Streamlit web interface
│   ├── app.py
│   ├── components/
│   └── styles/
│
├── analytics/              # Analytics and metrics
├── tests/                  # Test suite
├── docs/                   # Documentation
└── data/                   # Data files
```

## 💻 Usage

### Python API

```python
from src.core import HybridRecommender
from src.data import DataLoader, DataPreprocessor

# Load and preprocess data
loader = DataLoader()
merged_df = loader.get_merged_data()
preprocessor = DataPreprocessor(merged_df)
movies_df = preprocessor.preprocess()

# Initialize recommender
recommender = HybridRecommender()
recommender.fit(movies_df)

# Get recommendations
recommendations = recommender.recommend("The Matrix", n_recommendations=10)
print(recommendations[['title', 'hybrid_score']])

# Adjust method weights
recommender.set_weights(
    content_weight=0.4,
    metadata_weight=0.3,
    cf_weight=0.3
)
```

### REST API

```bash
# Get recommendations
curl "http://localhost:8000/api/v1/recommendations/The%20Matrix?top_n=5&method=hybrid"

# Search/list movies
curl "http://localhost:8000/api/v1/movies/?query=inception&page=1&page_size=10"

# Get movie details by title
curl "http://localhost:8000/api/v1/movies/The%20Matrix"
```

### Using Services

```python
from src.services import RecommendationService, SearchService

# Initialize services
rec_service = RecommendationService(movies_df)
search_service = SearchService(movies_df)

# Search and recommend
results = search_service.search("Matrix", limit=5)
recommendations = rec_service.get_recommendations("The Matrix", n=10, method="hybrid")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## 📈 Methods Comparison

| Method | Technique | Strengths | Limitations |
|--------|-----------|-----------|-------------|
| Content-Based | TF-IDF + Cosine Similarity | No user data needed, interpretable | Limited to similar content |
| Metadata-Based | Weighted feature matching | Multi-dimensional, accurate | Requires rich metadata |
| Collaborative | User/Item similarity, SVD | Discovers hidden patterns | Cold start problem |
| Hybrid | Weighted combination | Best of all methods | Requires weight tuning |

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Documentation](docs/api_docs.md)
- [Contributing Guide](docs/contributing.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) for the movie dataset
- [Streamlit](https://streamlit.io/) for the web framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [SHAP](https://github.com/slundberg/shap) for explainability
