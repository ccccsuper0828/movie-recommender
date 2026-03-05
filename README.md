# 🎬 CineMatch — Movie Recommendation & Box Office Prediction System

A full-stack movie analytics platform featuring **8 recommendation algorithms**, **box office revenue prediction** (Kaggle Top 1-3%), **SHAP explainability**, and a **Streamlit web app** — all built on real TMDB + MovieLens data.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app
cd MOVIE
PYTHONPATH=$(pwd) streamlit run frontend/app.py --server.port 8501

# 3. Open http://localhost:8501
```

---

## Project Structure

```
MOVIE/
│
├── app.py                          # Entry point (for Streamlit Cloud)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker build
├── docker-compose.yml              # Multi-service orchestration
│
├── config/                         # Configuration
│   ├── settings.py                 # Pydantic Settings (env vars + defaults)
│   └── logging_config.py           # Logging setup
│
├── src/                            # ── Core Source Code ──
│   ├── registry.py                 # 🔌 Model Registry (plug-and-play)
│   │
│   ├── core/                       # Recommendation Algorithms
│   │   ├── base_recommender.py     #   Abstract interface
│   │   ├── content_based.py        #   TF-IDF cosine similarity
│   │   ├── metadata_based.py       #   Genre/Director/Cast weighted matching
│   │   ├── collaborative.py        #   Item-CF / User-CF / SVD
│   │   ├── hybrid.py               #   Weighted fusion of all methods
│   │   ├── demographic.py          #   IMDB weighted rating formula
│   │   └── knn_svd_ensemble.py     #   User-KNN + SVD re-ranking
│   │
│   ├── prediction/                 # Box Office Prediction
│   │   ├── base_predictor.py       #   Abstract interface
│   │   └── box_office_predictor.py #   LightGBM + XGBoost + CatBoost ensemble
│   │
│   ├── evaluation/                 # Quality Assessment
│   │   └── evaluator.py            #   Precision/NDCG/Coverage/Novelty/Diversity
│   │
│   ├── explainability/             # Explainability
│   │   ├── rule_based.py           #   Feature-matching natural language explanations
│   │   ├── shap_explainer.py       #   SHAP TreeExplainer
│   │   └── visualization.py        #   Waterfall / Force / Importance plots
│   │
│   ├── data/                       # Data Pipeline
│   │   ├── loader.py               #   TMDB CSV loading + merging
│   │   ├── preprocessor.py         #   JSON parsing, feature extraction, cleaning
│   │   ├── movielens_loader.py     #   MovieLens ↔ TMDB title matching
│   │   └── cache_manager.py        #   Similarity matrix caching (pickle + TTL)
│   │
│   ├── services/                   # Business Logic Layer
│   │   ├── recommendation_service.py
│   │   ├── search_service.py
│   │   ├── user_service.py
│   │   └── analytics_service.py
│   │
│   ├── models/                     # Data Models
│   │   ├── movie.py                #   Movie dataclass
│   │   ├── user.py                 #   User dataclass
│   │   └── schemas.py              #   Pydantic request/response schemas
│   │
│   └── utils/                      # Utilities
│       ├── metrics.py              #   7 evaluation metrics
│       ├── similarity.py           #   Distance/similarity functions
│       └── text_processing.py      #   NLP helpers
│
├── frontend/                       # ── Streamlit Web App ──
│   ├── app.py                      # Main app (8 pages, routing, caching)
│   ├── components/                 # Reusable UI components
│   │   ├── charts.py               #   Plotly chart wrappers
│   │   ├── movie_card.py           #   Movie card HTML component
│   │   ├── recommendation_list.py  #   Recommendation list renderer
│   │   ├── explanation_panel.py    #   SHAP/rule explanation panel
│   │   ├── search_bar.py           #   Search + autocomplete
│   │   └── sidebar.py              #   Navigation + settings
│   └── styles/                     # Theme
│       ├── theme.py                #   Color palette + CSS injection
│       └── main.css                #   Global styles
│
├── api/                            # ── FastAPI REST API ──
│   ├── main.py                     # App factory + CORS + lifespan
│   ├── dependencies.py             # Dependency injection (singletons)
│   └── routes/                     # Endpoint handlers
│       ├── health.py               #   /health, /ready, /live
│       ├── recommendations.py      #   /api/v1/recommendations
│       ├── movies.py               #   /api/v1/movies
│       └── users.py                #   /api/v1/users
│
├── analytics/                      # EDA Visualization Helpers
│   ├── visualizations.py
│   ├── dashboard.py
│   └── metrics_tracker.py
│
├── tests/                          # ── Test Suite ──
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── data/                           # ── Data Files ──
│   ├── raw/                        # Source datasets
│   │   ├── tmdb_5000_movies.csv    #   TMDB 5000 (4803 movies)
│   │   ├── tmdb_5000_credits.csv   #   Cast + crew
│   │   ├── ml_ratings.csv          #   MovieLens 1M ratings
│   │   ├── ml_movies.csv           #   MovieLens movie metadata
│   │   ├── ml_users.csv            #   MovieLens user demographics
│   │   └── kaggle_bo/              #   Kaggle Box Office competition data
│   │       ├── train.csv           #     3000 movies with revenue
│   │       ├── test.csv            #     4398 movies (no revenue)
│   │       ├── TrainAdditionalFeatures.csv
│   │       ├── TestAdditionalFeatures.csv
│   │       └── release_dates_per_country.csv
│   ├── processed/                  # Generated outputs
│   └── cache/                      # Model cache (box_office_model.pkl)
│
├── docs/                           # ── Documentation ──
│   ├── 项目文档与分工.md            # Full project doc + 6-person assignment
│   ├── api_docs.md                 # REST API reference
│   ├── architecture.md             # System architecture
│   └── contributing.md             # Contribution guidelines
│
├── tmdb_analysis.ipynb             # EDA Notebook (Jupyter)
│
└── archive/                        # Old/unused files (gitignored)
    ├── movie_recommender_notebook.ipynb
    ├── scripts/
    └── catboost_info/
```

---

## Features

### 1. Movie Recommendation (8 methods)

| Method | Type | Algorithm |
|--------|------|-----------|
| Content-Based | Item | TF-IDF + Cosine Similarity on plot text |
| Metadata-Based | Item | Weighted genre/director/cast/keyword matching |
| Collaborative Filtering | User | Item-CF, User-CF, SVD matrix factorization |
| Hybrid | Ensemble | Weighted fusion (Content + Metadata + CF) |
| Demographic | Non-personalized | IMDB weighted rating formula |
| KNN+SVD Ensemble | User | User-KNN candidate generation + SVD re-ranking |

### 2. Box Office Prediction

- **Models**: CatBoost + XGBoost + LightGBM (5-fold CV)
- **Features**: 55 dimensions (budget, popularity, genres, director target encoding, release timing, production company, etc.)
- **RMSLE**: 1.79 (≈ Kaggle Top 1-3% out of 1400 teams)
- **Anti-overfitting**: Balanced regularization, train-val gap ~0.5

### 3. Evaluation (MovieLens ground truth)

7 metrics: Precision@K, Recall@K, NDCG@K, MAP@K, Coverage, Novelty, Diversity

### 4. Explainability

- Rule-based natural language explanations
- SHAP waterfall + feature importance visualization

### 5. Web Interface (8 pages)

| Page | Description |
|------|-------------|
| 🏠 Home | Hero + quick recommendations + trending/top-rated |
| 🎯 Get Recommendations | 6 method tabs + SHAP deep-dive |
| 🔍 Explore | Browse all 4803 movies with filters |
| 📊 Analytics | EDA charts + MovieLens evaluation dashboard |
| 💰 Box Office Prediction | Train/evaluate + custom movie prediction |
| ⚖️ Compare Methods | Side-by-side method comparison |
| ℹ️ About | Project info |

---

## How to Run

### Local Development

```bash
# Install
pip install -r requirements.txt

# Streamlit frontend
cd MOVIE
PYTHONPATH=$(pwd) streamlit run frontend/app.py --server.port 8501

# FastAPI backend (optional, separate terminal)
PYTHONPATH=$(pwd) uvicorn api.main:app --port 8000
```

### Docker

```bash
docker-compose up -d --build
# Frontend: http://localhost:8501
# API: http://localhost:8000/docs
```

### Run EDA Notebook

```bash
jupyter notebook tmdb_analysis.ipynb
```

### Run Tests

```bash
pytest tests/unit/ -q          # Fast unit tests
```

---

## How to Add / Replace a Model

### Add a New Recommender

```python
# 1. Create src/core/my_recommender.py
from src.core.base_recommender import BaseRecommender

class MyRecommender(BaseRecommender):
    def fit(self, movies_df, **kwargs):
        # Your training logic
        self._is_fitted = True
        return self

    def recommend(self, title, top_n=10, **kwargs):
        # Your recommendation logic
        return recommendations_df

# 2. Register in src/core/__init__.py
from src.registry import RECOMMENDER_REGISTRY
RECOMMENDER_REGISTRY.register("my_method", MyRecommender)
```

### Add a New Predictor

```python
# 1. Create src/prediction/my_predictor.py
from src.prediction.base_predictor import BasePredictor

class MyPredictor(BasePredictor):
    def fit(self, df=None):
        # Your training logic
        self._is_fitted = True
        return self

    def predict(self, df):
        # Return np.ndarray of predicted revenues
        return predictions

    def save(self, path=None): ...
    def load(cls, path=None): ...

# 2. Register in src/prediction/__init__.py
from src.registry import PREDICTOR_REGISTRY
PREDICTOR_REGISTRY.register("my_model", MyPredictor)
```

### Use Registry in Frontend

```python
from src.registry import RECOMMENDER_REGISTRY, PREDICTOR_REGISTRY

# Get model by name (no direct import needed)
RecClass = RECOMMENDER_REGISTRY.get("content")
model = RecClass()
model.fit(movies_df)

# List all available models
print(RECOMMENDER_REGISTRY.list())
print(PREDICTOR_REGISTRY.list())
```

---

## Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| TMDB 5000 Movies + Credits | 4803 movies | Recommendation (content, metadata) |
| MovieLens 1M | 1M ratings, 6040 users | Collaborative filtering + evaluation |
| Kaggle TMDB Box Office | 3000 train + 4398 test | Revenue prediction |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.45 |
| Backend API | FastAPI |
| ML | scikit-learn, LightGBM, XGBoost, CatBoost |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Deployment | Docker, Streamlit Cloud |

---

## Team (6 members)

| Member | Role | Key Files |
|--------|------|-----------|
| A | Data Engineering | `src/data/`, `config/` |
| B | Basic Recommenders | `src/core/content_based.py`, `metadata_based.py`, `demographic.py`, `src/utils/` |
| C | Advanced Recommenders + UI | `src/core/collaborative.py`, `knn_svd_ensemble.py`, `hybrid.py` |
| D | Box Office Prediction + EDA | `src/prediction/`, `analytics/` |
| E | Explainability + Evaluation | `src/explainability/`, `src/evaluation/` |
| F | Frontend + API + Tests | `frontend/`, `api/`, `tests/`, `src/services/`, `src/models/` |

Full documentation: [docs/项目文档与分工.md](docs/项目文档与分工.md)
