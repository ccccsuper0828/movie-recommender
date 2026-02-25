# Movie Recommender System

A movie recommendation system built with three different machine learning approaches.

## Features

This project implements three recommendation algorithms:

1. **Content-Based Filtering** - Uses TF-IDF to analyze movie descriptions and cosine similarity to find similar movies
2. **Metadata-Based Filtering** - Combines genres, directors, actors, and keywords with weighted features
3. **Collaborative Filtering** - Implements User-Based CF, Item-Based CF, and SVD matrix factorization

Additionally, a **Hybrid Recommender** combines all three methods with adjustable weights.

## Dataset

Uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) containing:
- 4,803 movies
- Movie metadata (genres, keywords, cast, crew)
- Movie descriptions and ratings

## Installation

```bash
# Clone the repository
git clone https://github.com/ccccsuper0828/movie-recommender.git
cd movie-recommender

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- streamlit
- matplotlib
- seaborn

## Usage

### Run the Streamlit Web Demo

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Use as Python Module

```python
from movie_recommender import MovieRecommenderSystem

# Initialize
recommender = MovieRecommenderSystem(
    'tmdb_5000_movies.csv',
    'tmdb_5000_credits.csv'
)

# Build models
recommender.build_content_based_model()
recommender.build_metadata_based_model()
recommender.build_collaborative_filtering_model()

# Get recommendations
recommender.content_based_recommend('Avatar', top_n=10)
recommender.metadata_based_recommend('Avatar', top_n=10)
recommender.collaborative_filtering_recommend('Avatar', method='item_based')
recommender.collaborative_filtering_recommend('Avatar', method='svd')

# Hybrid recommendation
recommender.hybrid_recommend('Avatar', weights=(0.3, 0.4, 0.3))

# Compare all methods
recommender.compare_recommendations('Avatar', top_n=5)

# Recommend for a user
recommender.recommend_for_user(user_id=0, top_n=10)
```

### Jupyter Notebook

```bash
jupyter notebook movie_recommender_notebook.ipynb
```

## Project Structure

```
movie-recommender/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit web demo
├── movie_recommender.py            # Main recommender class
├── movie_recommender_notebook.ipynb # Interactive notebook
├── tmdb_5000_movies.csv            # Movie dataset
├── tmdb_5000_credits.csv           # Credits dataset
└── *.png                           # Visualization outputs
```

## Methods Comparison

| Method | Technique | Pros | Cons |
|--------|-----------|------|------|
| Content-Based | TF-IDF + Cosine Similarity | No user data needed, interpretable | Only recommends similar content |
| Metadata-Based | Feature combination with weights | Multi-dimensional, accurate | Requires rich metadata |
| Collaborative Filtering | User/Item similarity, SVD | Discovers hidden patterns | Cold start problem, needs user data |
| Hybrid | Weighted combination | Best of all methods | Requires tuning |

## License

MIT License
