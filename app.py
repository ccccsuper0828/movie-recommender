"""
Movie Recommendation System - Streamlit Web Demo
=================================================
Interactive demo for three ML-based recommendation methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    .method-header {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Data Loading and Preprocessing ====================

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data"""
    movies_path = os.path.join(SCRIPT_DIR, 'tmdb_5000_movies.csv')
    credits_path = os.path.join(SCRIPT_DIR, 'tmdb_5000_credits.csv')

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    def get_list(x):
        if isinstance(x, list):
            return [i['name'] for i in x][:5]
        return []

    def get_director(x):
        if isinstance(x, list):
            for crew_member in x:
                if crew_member.get('job') == 'Director':
                    return crew_member.get('name', '')
        return ''

    def get_top_cast(x, n=5):
        if isinstance(x, list):
            return [i['name'] for i in x][:n]
        return []

    def clean_text(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        elif isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ''

    # Merge datasets
    credits.columns = ['id', 'title', 'cast', 'crew']
    movies = movies.merge(credits[['id', 'cast', 'crew']], on='id')

    # Parse JSON fields
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(safe_literal_eval)

    # Extract features
    movies['genres_list'] = movies['genres'].apply(get_list)
    movies['keywords_list'] = movies['keywords'].apply(get_list)
    movies['cast_list'] = movies['cast'].apply(get_top_cast)
    movies['director'] = movies['crew'].apply(get_director)

    # Clean text
    movies['genres_clean'] = movies['genres_list'].apply(clean_text)
    movies['keywords_clean'] = movies['keywords_list'].apply(clean_text)
    movies['cast_clean'] = movies['cast_list'].apply(clean_text)
    movies['director_clean'] = movies['director'].apply(clean_text)

    # Handle missing values
    movies['overview'] = movies['overview'].fillna('')
    movies['release_date'] = movies['release_date'].fillna('')
    movies['runtime'] = movies['runtime'].fillna(0)

    # Reset index
    movies = movies.reset_index(drop=True)

    return movies


@st.cache_data
def build_content_similarity(_movies):
    """Build content-based similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(_movies['overview'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)


@st.cache_data
def build_metadata_similarity(_movies):
    """Build metadata-based similarity matrix"""
    def create_soup(row):
        features = []
        features.extend(row['genres_clean'] * 3)
        if row['director_clean']:
            features.extend([row['director_clean']] * 3)
        features.extend(row['cast_clean'] * 2)
        features.extend(row['keywords_clean'])
        return ' '.join(features)

    soup = _movies.apply(create_soup, axis=1)
    count_vectorizer = CountVectorizer(stop_words='english', max_features=10000)
    count_matrix = count_vectorizer.fit_transform(soup)
    return cosine_similarity(count_matrix, count_matrix)


@st.cache_data
def build_cf_similarity(_movies, n_users=500):
    """Build collaborative filtering similarity matrix"""
    n_movies = len(_movies)
    ratings = np.zeros((n_users, n_movies))

    all_genres = set()
    for genres in _movies['genres_list']:
        all_genres.update(genres)
    genre_list = sorted(list(all_genres))

    np.random.seed(42)

    for user_id in range(n_users):
        n_preferred = np.random.randint(1, 4)
        preferred_genres = np.random.choice(genre_list, size=min(n_preferred, len(genre_list)), replace=False)

        n_watched = int(n_movies * 0.02 * np.random.uniform(0.5, 2.0))
        n_watched = max(10, min(n_watched, n_movies // 10))

        movie_probs = np.ones(n_movies) * 0.1
        for movie_idx in range(n_movies):
            movie_genres = _movies.iloc[movie_idx]['genres_list']
            if any(g in preferred_genres for g in movie_genres):
                movie_probs[movie_idx] = 0.8
            vote_avg = _movies.iloc[movie_idx]['vote_average']
            movie_probs[movie_idx] *= (vote_avg / 10.0 + 0.5)

        movie_probs = movie_probs / movie_probs.sum()
        watched_movies = np.random.choice(n_movies, size=n_watched, replace=False, p=movie_probs)

        for movie_idx in watched_movies:
            movie_genres = _movies.iloc[movie_idx]['genres_list']
            vote_avg = _movies.iloc[movie_idx]['vote_average']
            base_rating = 2.5 + (vote_avg - 5) / 2

            if any(g in preferred_genres for g in movie_genres):
                base_rating += np.random.uniform(0.5, 1.5)
            else:
                base_rating += np.random.uniform(-1.0, 0.5)

            rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)
            ratings[user_id, movie_idx] = round(rating, 1)

    movie_user_matrix = ratings.T
    return cosine_similarity(movie_user_matrix), ratings


# ==================== Recommendation Functions ====================

def get_recommendations(title, movies, similarity_matrix, top_n=10):
    """Get recommendations"""
    title_to_idx = pd.Series(movies.index, index=movies['title']).to_dict()

    if title not in title_to_idx:
        return None

    idx = title_to_idx[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    recommendations = movies.iloc[movie_indices].copy()
    recommendations['similarity_score'] = scores

    return recommendations


def get_hybrid_recommendations(title, movies, content_sim, metadata_sim, cf_sim, weights, top_n=10):
    """Get hybrid recommendations"""
    title_to_idx = pd.Series(movies.index, index=movies['title']).to_dict()

    if title not in title_to_idx:
        return None

    idx = title_to_idx[title]
    n_movies = len(movies)

    hybrid_scores = np.zeros(n_movies)
    hybrid_scores += weights[0] * content_sim[idx]
    hybrid_scores += weights[1] * metadata_sim[idx]
    hybrid_scores += weights[2] * cf_sim[idx]

    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    recommendations = movies.iloc[movie_indices].copy()
    recommendations['similarity_score'] = scores

    return recommendations


# ==================== SHAP Explainability Functions ====================

@st.cache_resource
def build_shap_explainer(_movies, _similarity_matrix):
    """Build SHAP explainer (cached)"""
    if not SHAP_AVAILABLE:
        return None, None, None, None

    # Prepare feature matrix
    genre_encoder = MultiLabelBinarizer()
    genres = _movies['genres_list'].apply(lambda x: x if isinstance(x, list) else [])
    genre_encoded = genre_encoder.fit_transform(genres)
    genre_names = [f"Genre_{g}" for g in genre_encoder.classes_]

    vote_avg = _movies['vote_average'].fillna(0).values.reshape(-1, 1)
    popularity = _movies['popularity'].fillna(0).values.reshape(-1, 1)
    cast_count = _movies['cast_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).values.reshape(-1, 1)
    keyword_count = _movies['keywords_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).values.reshape(-1, 1)

    feature_matrix = np.hstack([genre_encoded, vote_avg, popularity, cast_count, keyword_count])
    feature_names = genre_names + ['Rating', 'Popularity', 'CastCount', 'KeywordCount']

    # Prepare training data
    n_movies = len(_movies)
    n_samples = 20000
    np.random.seed(42)
    idx1 = np.random.randint(0, n_movies, n_samples)
    idx2 = np.random.randint(0, n_movies, n_samples)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    X, y = [], []
    for i1, i2 in zip(idx1, idx2):
        f1, f2 = feature_matrix[i1], feature_matrix[i2]
        X.append(np.concatenate([np.abs(f1-f2), f1*f2]))
        y.append(_similarity_matrix[i1, i2])

    X, y = np.array(X), np.array(y)
    pair_names = [f"{n}_Diff" for n in feature_names] + [f"{n}_Match" for n in feature_names]

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    return explainer, model, feature_matrix, pair_names


def create_shap_waterfall_fig(shap_values, base_value, feature_names, title="SHAP Waterfall"):
    """Create SHAP waterfall plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get top features
    indices = np.argsort(np.abs(shap_values))[::-1][:15]
    values = shap_values[indices]
    names = [feature_names[i][:20] for i in indices]

    colors = ['#ff0051' if v > 0 else '#008bfb' for v in values]

    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP Value')
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff0051', label='Increases Similarity'),
        Patch(facecolor='#008bfb', label='Decreases Similarity')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return fig


def create_shap_summary_fig(shap_values, X, feature_names, title="SHAP Summary"):
    """Create SHAP summary plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate feature importance (mean absolute SHAP value)
    importance = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(importance)[::-1][:20]

    values = importance[indices]
    names = [feature_names[i][:20] for i in indices]

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(values)))

    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def create_shap_force_fig(shap_values, base_value, feature_names, predicted_value, title=""):
    """Create SHAP force plot"""
    fig, ax = plt.subplots(figsize=(12, 3))

    indices = np.argsort(np.abs(shap_values))[::-1][:10]

    # Draw bars
    left_pos = base_value
    left_neg = base_value

    for idx in indices:
        val = shap_values[idx]
        if val > 0:
            ax.barh(0, val, left=left_pos, color='#ff0051', height=0.5, alpha=0.8)
            left_pos += val
        else:
            ax.barh(0, val, left=left_neg, color='#008bfb', height=0.5, alpha=0.8)
            left_neg += val

    ax.axvline(x=base_value, color='gray', linestyle='--', label=f'Base: {base_value:.3f}')
    ax.axvline(x=predicted_value, color='black', linestyle='-', linewidth=2, label=f'Pred: {predicted_value:.3f}')

    ax.set_yticks([])
    ax.set_xlabel('Similarity Score')
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def get_shap_explanation(source_title, target_title, movies, similarity_matrix,
                          explainer, model, feature_matrix, pair_names):
    """Get SHAP explanation"""
    if explainer is None:
        return None

    title_to_idx = pd.Series(movies.index, index=movies['title']).to_dict()
    source_idx = title_to_idx.get(source_title)
    target_idx = title_to_idx.get(target_title)

    if source_idx is None or target_idx is None:
        return None

    # Create features
    f1, f2 = feature_matrix[source_idx], feature_matrix[target_idx]
    X = np.concatenate([np.abs(f1-f2), f1*f2]).reshape(1, -1)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)[0]
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])

    predicted = model.predict(X)[0]
    actual = similarity_matrix[source_idx, target_idx]

    # Analyze contributions
    shap_df = pd.DataFrame({
        'feature': pair_names,
        'shap_value': shap_values
    })
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=False)

    positive = shap_df[shap_df['shap_value'] > 0.001].head(5)
    negative = shap_df[shap_df['shap_value'] < -0.001].head(5)

    return {
        'shap_values': shap_values,
        'base_value': base_value,
        'predicted': predicted,
        'actual': actual,
        'positive': positive,
        'negative': negative,
        'X': X
    }


# ==================== Explainability Functions ====================

def get_explanation(source_movie, target_movie, movies):
    """Generate recommendation explanation"""
    title_to_idx = pd.Series(movies.index, index=movies['title']).to_dict()

    source_idx = title_to_idx.get(source_movie)
    target_idx = title_to_idx.get(target_movie)

    if source_idx is None or target_idx is None:
        return {'reasons': [], 'summary': 'Cannot generate explanation'}

    source = movies.iloc[source_idx]
    target = movies.iloc[target_idx]

    reasons = []
    details = []

    # 1. Check director
    if source['director'] and target['director'] and source['director'] == target['director']:
        reasons.append(f"🎬 Same Director: {source['director']}")
        details.append(('Director', source['director'], 'High'))

    # 2. Check genres
    source_genres = set(source.get('genres_list', []))
    target_genres = set(target.get('genres_list', []))
    common_genres = source_genres & target_genres
    if common_genres:
        reasons.append(f"🎭 Same Genres: {', '.join(common_genres)}")
        details.append(('Genres', list(common_genres), 'High'))

    # 3. Check cast
    source_cast = set(source.get('cast_list', []))
    target_cast = set(target.get('cast_list', []))
    common_cast = source_cast & target_cast
    if common_cast:
        reasons.append(f"👥 Common Cast: {', '.join(list(common_cast)[:2])}")
        details.append(('Cast', list(common_cast), 'Medium'))

    # 4. Check keywords
    source_keywords = set(source.get('keywords_list', []))
    target_keywords = set(target.get('keywords_list', []))
    common_keywords = source_keywords & target_keywords
    if common_keywords:
        reasons.append(f"🏷️ Similar Themes: {', '.join(list(common_keywords)[:2])}")
        details.append(('Keywords', list(common_keywords), 'Standard'))

    # Generate summary
    if reasons:
        summary = reasons[0].split(': ')[1] if ': ' in reasons[0] else reasons[0]
    else:
        summary = "Overall feature similarity"

    return {
        'reasons': reasons,
        'details': details,
        'summary': summary
    }


# ==================== Display Functions ====================

def display_movie_info(movie_row):
    """Display movie details"""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### 🎬 {movie_row['title']}")
        st.markdown(f"**Rating:** ⭐ {movie_row['vote_average']}/10")
        st.markdown(f"**Year:** {movie_row['release_date'][:4] if movie_row['release_date'] else 'N/A'}")
        st.markdown(f"**Runtime:** {int(movie_row['runtime'])} min")

    with col2:
        st.markdown(f"**Genres:** {', '.join(movie_row['genres_list'])}")
        st.markdown(f"**Director:** {movie_row['director']}")
        st.markdown(f"**Cast:** {', '.join(movie_row['cast_list'][:3])}")

    st.markdown(f"**Overview:** {movie_row['overview'][:300]}..." if len(movie_row['overview']) > 300 else f"**Overview:** {movie_row['overview']}")


def display_recommendations(recommendations, method_name, source_movie=None, movies=None, show_explanation=True):
    """Display recommendations with explainability"""
    if recommendations is None or len(recommendations) == 0:
        st.warning("No recommendations found")
        return

    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        with st.container():
            col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1])

            with col1:
                st.markdown(f"### {idx}")

            with col2:
                st.markdown(f"**{row['title']}**")
                genres = ', '.join(row['genres_list'][:3]) if row['genres_list'] else 'N/A'
                st.caption(f"🎭 {genres}")

            with col3:
                st.markdown(f"⭐ {row['vote_average']}/10")
                if row['director']:
                    st.caption(f"🎬 {row['director']}")

            with col4:
                score = row['similarity_score']
                st.metric("Similarity", f"{score:.1%}")

            # Show recommendation explanation
            if show_explanation and source_movie and movies is not None:
                explanation = get_explanation(source_movie, row['title'], movies)
                if explanation['reasons']:
                    with st.expander("💡 Why this recommendation?", expanded=False):
                        for reason in explanation['reasons']:
                            st.markdown(f"  {reason}")
                else:
                    st.caption("💡 Overall feature similarity")

            st.divider()


# ==================== Main Application ====================

def main():
    # Title
    st.markdown('<div class="main-header">🎬 Movie Recommender System</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Intelligent recommendations using three ML methods</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading data...'):
        movies = load_and_preprocess_data()

    # Build similarity matrices
    with st.spinner('Building recommendation models...'):
        content_similarity = build_content_similarity(movies)
        metadata_similarity = build_metadata_similarity(movies)
        cf_similarity, user_movie_matrix = build_cf_similarity(movies)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Movie search
        st.markdown("### 🔍 Select Movie")

        # Search box
        search_query = st.text_input("Search movies", placeholder="Enter movie title...")

        if search_query:
            filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
            movie_options = filtered_movies['title'].tolist()[:20]
        else:
            # Default: show popular movies
            movie_options = movies.nlargest(50, 'popularity')['title'].tolist()

        selected_movie = st.selectbox(
            "Select a movie",
            options=movie_options,
            index=0 if movie_options else None
        )

        st.markdown("---")

        # Number of recommendations
        top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

        st.markdown("---")

        # Hybrid weights
        st.markdown("### ⚖️ Hybrid Weights")
        w1 = st.slider("Content weight", 0.0, 1.0, 0.3, 0.1)
        w2 = st.slider("Metadata weight", 0.0, 1.0, 0.4, 0.1)
        w3 = st.slider("Collaborative filtering weight", 0.0, 1.0, 0.3, 0.1)

        # Normalize weights
        total = w1 + w2 + w3
        if total > 0:
            weights = (w1/total, w2/total, w3/total)
        else:
            weights = (0.33, 0.34, 0.33)

        st.caption(f"Normalized: {weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}")

        st.markdown("---")
        st.markdown("### 📊 Data Statistics")
        st.metric("Total Movies", f"{len(movies):,}")
        st.metric("Simulated Users", "500")

    # Main content area
    if selected_movie:
        # Display selected movie info
        st.markdown("## 📽️ Selected Movie")
        movie_idx = movies[movies['title'] == selected_movie].index[0]
        movie_info = movies.iloc[movie_idx]

        with st.container():
            display_movie_info(movie_info)

        st.markdown("---")

        # Recommendation tabs
        st.markdown("## 🎯 Recommendations")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Content-Based",
            "🏷️ Metadata-Based",
            "👥 Collaborative Filtering",
            "🔀 Hybrid",
            "🔬 SHAP Explanation"
        ])

        with tab1:
            st.markdown("""
            <div class="method-header">
            <b>Method:</b> Uses TF-IDF to analyze movie descriptions and finds similar content using cosine similarity
            </div>
            """, unsafe_allow_html=True)

            content_recs = get_recommendations(selected_movie, movies, content_similarity, top_n)
            display_recommendations(content_recs, "Content-Based", selected_movie, movies)

        with tab2:
            st.markdown("""
            <div class="method-header">
            <b>Method:</b> Analyzes genres, directors, cast, and keywords to find similar movies
            </div>
            """, unsafe_allow_html=True)

            metadata_recs = get_recommendations(selected_movie, movies, metadata_similarity, top_n)
            display_recommendations(metadata_recs, "Metadata-Based", selected_movie, movies)

        with tab3:
            st.markdown("""
            <div class="method-header">
            <b>Method:</b> Uses simulated user behavior data with collaborative filtering algorithms
            </div>
            """, unsafe_allow_html=True)

            cf_recs = get_recommendations(selected_movie, movies, cf_similarity, top_n)
            display_recommendations(cf_recs, "Collaborative Filtering", selected_movie, movies)

        with tab4:
            st.markdown(f"""
            <div class="method-header">
            <b>Method:</b> Weighted combination of all methods (Content:{weights[0]:.0%}, Metadata:{weights[1]:.0%}, CF:{weights[2]:.0%})
            </div>
            """, unsafe_allow_html=True)

            hybrid_recs = get_hybrid_recommendations(
                selected_movie, movies,
                content_similarity, metadata_similarity, cf_similarity,
                weights, top_n
            )
            display_recommendations(hybrid_recs, "Hybrid", selected_movie, movies)

        with tab5:
            st.markdown("""
            <div class="method-header">
            <b>SHAP Explainability:</b> Uses SHAP (SHapley Additive exPlanations) to explain recommendations
            based on game theory's Shapley values for fair feature contribution allocation
            </div>
            """, unsafe_allow_html=True)

            if not SHAP_AVAILABLE:
                st.error("SHAP library not installed. Please run: pip install shap")
            else:
                # Build SHAP explainer
                with st.spinner("Building SHAP explainer..."):
                    shap_explainer, shap_model, feature_matrix, pair_names = build_shap_explainer(
                        movies, metadata_similarity
                    )

                if shap_explainer is not None:
                    # Select movie to explain
                    st.markdown("### Select Recommendation to Explain")

                    if metadata_recs is not None and len(metadata_recs) > 0:
                        target_options = metadata_recs['title'].tolist()[:10]
                        selected_target = st.selectbox(
                            "Select a recommended movie for SHAP analysis",
                            options=target_options,
                            key="shap_target"
                        )

                        if selected_target:
                            # Get SHAP explanation
                            shap_result = get_shap_explanation(
                                selected_movie, selected_target, movies, metadata_similarity,
                                shap_explainer, shap_model, feature_matrix, pair_names
                            )

                            if shap_result:
                                # Display basic info
                                st.markdown("### 📊 Similarity Analysis")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Actual Similarity", f"{shap_result['actual']:.2%}")
                                with col2:
                                    st.metric("Model Prediction", f"{shap_result['predicted']:.2%}")
                                with col3:
                                    st.metric("Base Value", f"{shap_result['base_value']:.4f}")

                                # Display contributions
                                st.markdown("### 🔍 Feature Contribution Analysis")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**✅ Positive Contributions (Increases Similarity)**")
                                    if len(shap_result['positive']) > 0:
                                        for _, row in shap_result['positive'].iterrows():
                                            st.success(f"{row['feature']}: +{row['shap_value']:.4f}")
                                    else:
                                        st.info("No significant positive contributions")

                                with col2:
                                    st.markdown("**❌ Negative Contributions (Decreases Similarity)**")
                                    if len(shap_result['negative']) > 0:
                                        for _, row in shap_result['negative'].iterrows():
                                            st.error(f"{row['feature']}: {row['shap_value']:.4f}")
                                    else:
                                        st.info("No significant negative contributions")

                                # SHAP visualizations
                                st.markdown("### 📈 SHAP Visualizations")

                                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                                    "Waterfall Plot",
                                    "Force Plot",
                                    "Feature Importance"
                                ])

                                with viz_tab1:
                                    st.markdown("**Waterfall Plot**: Shows how each feature pushes the prediction from base value")
                                    fig_waterfall = create_shap_waterfall_fig(
                                        shap_result['shap_values'],
                                        shap_result['base_value'],
                                        pair_names,
                                        f"SHAP Waterfall: {selected_movie} → {selected_target}"
                                    )
                                    st.pyplot(fig_waterfall)
                                    plt.close(fig_waterfall)

                                with viz_tab2:
                                    st.markdown("**Force Plot**: Shows the tug-of-war between positive and negative contributions")
                                    fig_force = create_shap_force_fig(
                                        shap_result['shap_values'],
                                        shap_result['base_value'],
                                        pair_names,
                                        shap_result['predicted'],
                                        f"SHAP Force: {selected_movie} → {selected_target}"
                                    )
                                    st.pyplot(fig_force)
                                    plt.close(fig_force)

                                with viz_tab3:
                                    st.markdown("**Global Feature Importance**: Mean SHAP contribution across all samples")
                                    with st.spinner("Calculating global feature importance..."):
                                        # Sample calculation
                                        n_samples = min(200, len(movies))
                                        sample_indices = np.random.choice(len(movies), n_samples, replace=False)
                                        source_idx = movies[movies['title'] == selected_movie].index[0]

                                        X_samples = []
                                        for idx in sample_indices:
                                            if idx != source_idx:
                                                f1 = feature_matrix[source_idx]
                                                f2 = feature_matrix[idx]
                                                X_samples.append(np.concatenate([np.abs(f1-f2), f1*f2]))

                                        if X_samples:
                                            X_samples = np.array(X_samples)
                                            shap_values_all = shap_explainer.shap_values(X_samples)

                                            fig_summary = create_shap_summary_fig(
                                                shap_values_all,
                                                X_samples,
                                                pair_names,
                                                f"SHAP Feature Importance for {selected_movie}"
                                            )
                                            st.pyplot(fig_summary)
                                            plt.close(fig_summary)

                                # Explanation guide
                                with st.expander("💡 How to interpret SHAP plots?"):
                                    st.markdown("""
                                    **SHAP Value Interpretation:**
                                    - **Positive values (Red)**: Feature increases movie similarity
                                    - **Negative values (Blue)**: Feature decreases movie similarity
                                    - **Absolute magnitude**: Indicates strength of the feature's impact

                                    **Feature Naming Convention:**
                                    - `XXX_Diff`: Difference between two movies on that feature
                                    - `XXX_Match`: Match degree (both movies have the feature)

                                    **Examples:**
                                    - `Genre_Action_Match` is positive → Both are action movies, increases similarity
                                    - `Rating_Diff` is negative → Large rating difference, decreases similarity
                                    """)
                            else:
                                st.warning("Cannot generate SHAP explanation")
                    else:
                        st.info("Please get recommendations first")
                else:
                    st.error("SHAP explainer initialization failed")

        # Method comparison
        st.markdown("---")
        st.markdown("## 📊 Method Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📝 Content-Based Top 5")
            if content_recs is not None:
                for _, row in content_recs.head(5).iterrows():
                    st.markdown(f"- **{row['title']}** ({row['similarity_score']:.1%})")

        with col2:
            st.markdown("### 🏷️ Metadata-Based Top 5")
            if metadata_recs is not None:
                for _, row in metadata_recs.head(5).iterrows():
                    st.markdown(f"- **{row['title']}** ({row['similarity_score']:.1%})")

        with col3:
            st.markdown("### 👥 Collaborative Filtering Top 5")
            if cf_recs is not None:
                for _, row in cf_recs.head(5).iterrows():
                    st.markdown(f"- **{row['title']}** ({row['similarity_score']:.1%})")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🎬 Movie Recommender Demo | TMDB 5000 Dataset</p>
        <p>Three recommendation methods: Content-Based (TF-IDF) | Metadata-Based | Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
