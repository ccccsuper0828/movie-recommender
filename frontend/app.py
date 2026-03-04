"""
CineMatch - Movie Recommendation System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.styles.theme import inject_custom_css, get_page_config
from frontend.components.sidebar import Sidebar
from frontend.components.search_bar import SearchBar
from frontend.components.movie_card import MovieCard
from frontend.components.recommendation_list import RecommendationList
from frontend.components.explanation_panel import ExplanationPanel
from frontend.components.charts import Charts

# Page configuration
st.set_page_config(**get_page_config())

# Inject custom CSS
st.markdown(inject_custom_css(), unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and preprocess movie data."""
    from src.data import DataLoader, DataPreprocessor

    loader = DataLoader()
    merged_df = loader.get_merged_data()

    preprocessor = DataPreprocessor(merged_df)
    movies_df = preprocessor.preprocess()

    return movies_df, preprocessor


@st.cache_resource
def load_recommenders(_movies_df):
    """Load recommendation models."""
    from src.core import HybridRecommender

    recommender = HybridRecommender()
    recommender.fit(_movies_df)

    return recommender


@st.cache_resource
def load_shap_explainer(_movies_df, _similarity_matrix):
    """Load and train the SHAP explainer (cached)."""
    from src.explainability import SHAPExplainer
    if SHAPExplainer is None:
        return None
    try:
        explainer = SHAPExplainer(_movies_df, _similarity_matrix)
        explainer.train(n_samples=10000)
        return explainer
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"SHAP explainer init failed: {e}")
        return None


def main():
    """Main application."""
    # Load data
    with st.spinner("Loading movie database..."):
        try:
            movies_df, preprocessor = load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Make sure the data files are in data/raw/")
            return

    # Load recommenders
    with st.spinner("Initializing recommendation models..."):
        try:
            recommender = load_recommenders(movies_df)
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return

    # Sidebar settings
    settings = Sidebar.render_settings()
    Sidebar.render_stats(len(movies_df))
    page = Sidebar.render_navigation()
    Sidebar.render_about()

    # Page routing
    if page == "home":
        render_home_page(movies_df, recommender, settings)
    elif page == "recommendations":
        render_recommendations_page(movies_df, recommender, preprocessor, settings)
    elif page == "explore":
        render_explore_page(movies_df, preprocessor)
    elif page == "analytics":
        render_analytics_page(movies_df, recommender)
    elif page == "box_office":
        render_box_office_page(movies_df)
    elif page == "compare":
        render_compare_page(movies_df, recommender, preprocessor, settings)
    elif page == "about":
        render_about_page()


# ── Cached loaders for the two new recommenders ──

@st.cache_resource
def _fit_demographic(_movies_df):
    from src.core import DemographicRecommender
    rec = DemographicRecommender()
    rec.fit(_movies_df)
    return rec


@st.cache_resource
def _fit_knn_svd(_movies_df):
    from src.core import KNNSVDEnsembleRecommender
    rec = KNNSVDEnsembleRecommender(n_neighbors=20, n_factors=50)
    rec.fit(_movies_df)
    return rec


def _get_demographic_recs(movies_df, top_n):
    rec = _fit_demographic(movies_df)
    return rec.recommend(top_n=top_n)


def _get_knn_svd_recs(movies_df, title, top_n):
    rec = _fit_knn_svd(movies_df)
    return rec.recommend(title, top_n=top_n)


@st.cache_resource
def _fit_box_office():
    """Train the box office predictor on Kaggle competition data (cached)."""
    from src.prediction import BoxOfficePredictor
    bp = BoxOfficePredictor(n_folds=5, seed=2019)
    bp.fit()
    return bp


def render_home_page(movies_df, recommender, settings):
    """Render the home page."""
    # Hero section
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1c2333 0%,#0d1117 100%);border-radius:20px;'
        'padding:60px 40px;margin-bottom:40px;text-align:center;position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;right:0;width:60%;height:100%;'
        'background:radial-gradient(circle at top right,rgba(229,9,20,0.15),transparent 50%);"></div>'
        '<h1 style="font-size:3rem;font-weight:800;'
        'background:linear-gradient(135deg,#ffffff 0%,#f5c518 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px;">'
        '🎬 CineMatch</h1>'
        '<p style="color:#a3a3a3;font-size:1.2rem;max-width:600px;margin:0 auto 30px auto;">'
        'Discover your next favorite movie with our intelligent recommendation engine '
        'powered by three ML algorithms</p></div>',
        unsafe_allow_html=True
    )

    # Quick search
    st.markdown("### 🎯 Quick Recommendations")

    popular_movies = movies_df.nlargest(50, 'popularity')['title'].tolist()

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_movie = st.selectbox(
            "Select a movie you like",
            options=popular_movies,
            key="home_movie_select"
        )

    with col2:
        st.write("")
        st.write("")
        get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)

    if get_recs and selected_movie:
        with st.spinner("Finding similar movies..."):
            recs = recommender.recommend(selected_movie, settings['top_n'])

            if recs is not None:
                st.markdown("### 🎬 Recommended for You")
                RecommendationList.render(recs, selected_movie, "hybrid")

    # Featured sections
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Trending Now")
        trending = movies_df.nlargest(5, 'popularity')
        for _, row in trending.iterrows():
            MovieCard.render_compact(
                title=row['title'],
                similarity_score=row['popularity'] / movies_df['popularity'].max(),
                rating=row['vote_average']
            )

    with col2:
        st.markdown("### ⭐ Top Rated")
        top_rated = movies_df[movies_df['vote_count'] >= 100].nlargest(5, 'vote_average')
        for _, row in top_rated.iterrows():
            MovieCard.render_compact(
                title=row['title'],
                similarity_score=row['vote_average'] / 10,
                rating=row['vote_average']
            )


def render_recommendations_page(movies_df, recommender, preprocessor, settings):
    """Render the recommendations page."""
    st.markdown("## 🎯 Get Recommendations")

    # Movie selection
    all_movies = movies_df['title'].tolist()
    popular = movies_df.nlargest(100, 'popularity')['title'].tolist()

    selected_movie = SearchBar.render_hero_search(
        movies=all_movies,
        popular_movies=popular[:6],
        key="rec_search"
    )

    if selected_movie:
        # Method selection
        st.markdown("### Select Method")

        method_tabs = st.tabs([
            "🔀 Hybrid (Recommended)",
            "📝 Content-Based",
            "🏷️ Metadata-Based",
            "👥 Collaborative Filtering",
            "🌟 Demographic",
            "🔗 KNN+SVD Ensemble",
        ])

        methods = ["hybrid", "content", "metadata", "cf", "demographic", "knn_svd"]

        for tab, method in zip(method_tabs, methods):
            with tab:
                # Get recommendations
                if method == "hybrid":
                    recommender.set_weights(*settings['weights'])
                    recs = recommender.recommend(
                        selected_movie,
                        settings['top_n'],
                        return_component_scores=True
                    )
                elif method == "demographic":
                    recs = _get_demographic_recs(movies_df, settings['top_n'])
                elif method == "knn_svd":
                    recs = _get_knn_svd_recs(movies_df, selected_movie, settings['top_n'])
                else:
                    recs = recommender.recommend_with_method(
                        selected_movie,
                        method,
                        settings['top_n']
                    )

                if recs is not None and len(recs) > 0:
                    # Show source movie info
                    source_movie = preprocessor.get_movie_by_title(selected_movie)
                    if source_movie is not None:
                        genres_str = ', '.join(source_movie.get('genres_list', [])[:3])
                        rating_val = source_movie.get('vote_average', 0)
                        director_str = f" • 🎬 {source_movie.get('director', '')}" if source_movie.get('director') else ""
                        st.markdown(
                            f'<div style="background:#1c2333;padding:20px;border-radius:12px;'
                            f'margin-bottom:20px;border-left:4px solid #e5383b;">'
                            f'<h3 style="color:#fff;margin:0 0 10px 0;">{selected_movie}</h3>'
                            f'<p style="color:#a3a3a3;margin:0;">'
                            f'{genres_str} • ★ {rating_val:.1f}{director_str}</p></div>',
                            unsafe_allow_html=True
                        )

                    # Display recommendations
                    for rank, (_, row) in enumerate(recs.iterrows(), 1):
                        MovieCard.render(
                            title=row['title'],
                            genres=row.get('genres_list', []),
                            rating=row.get('vote_average', 0),
                            similarity_score=row.get('hybrid_score', row.get('similarity_score', 0)),
                            director=row.get('director'),
                            overview=row.get('overview'),
                            rank=rank,
                            show_explanation=True
                        )

                        # Show explanation for hybrid
                        if method == "hybrid" and 'content_score' in row:
                            ExplanationPanel.render(
                                source_movie=selected_movie,
                                target_movie=row['title'],
                                explanation={'summary': "Similar based on plot, genre, and user preferences"},
                                method_scores={
                                    'content': row.get('content_score', 0),
                                    'metadata': row.get('metadata_score', 0),
                                    'cf': row.get('cf_score', 0)
                                }
                            )

                    # ── SHAP Deep-Dive Section (after recommendation list) ──
                    if method == "hybrid":
                        st.markdown("---")
                        st.markdown("### 🔬 SHAP Deep-Dive Analysis")
                        st.caption(
                            "Train a surrogate model on the hybrid similarity matrix, "
                            "then use SHAP to explain *why* two movies are similar."
                        )
                        shap_col1, shap_col2 = st.columns([2, 1])
                        with shap_col1:
                            shap_target = st.selectbox(
                                "Pick a recommended movie to analyze",
                                options=recs['title'].tolist(),
                                key="shap_target_select"
                            )
                        with shap_col2:
                            st.write("")
                            st.write("")
                            run_shap = st.button("Run SHAP", type="primary", use_container_width=True)

                        if run_shap and shap_target:
                            with st.spinner("Training SHAP surrogate model & computing values…"):
                                shap_explainer = load_shap_explainer(
                                    movies_df,
                                    recommender.similarity_matrix
                                )
                            if shap_explainer is None:
                                st.error(
                                    "SHAP is not available in this environment. "
                                    "Install it with `pip install shap` and restart."
                                )
                            else:
                                result = shap_explainer.explain_pair(selected_movie, shap_target)
                                if result is None:
                                    st.warning("Could not generate SHAP explanation for this pair.")
                                else:
                                    # ── Scores row ──
                                    sc1, sc2, sc3 = st.columns(3)
                                    sc1.metric("Actual Similarity", f"{result['actual_similarity']:.1%}")
                                    sc2.metric("Predicted (Surrogate)", f"{result['predicted_similarity']:.1%}")
                                    sc3.metric("SHAP Base Value", f"{result['base_value']:.4f}")

                                    # ── Waterfall chart via ExplanationVisualizer ──
                                    from src.explainability import ExplanationVisualizer
                                    if ExplanationVisualizer is not None:
                                        viz = ExplanationVisualizer()
                                        fig = viz.plot_shap_waterfall(
                                            result['shap_values'],
                                            result['feature_names'],
                                            result['base_value'],
                                            title=f"SHAP: {selected_movie} → {shap_target}",
                                            max_features=15
                                        )
                                        st.pyplot(fig)
                                        viz.close_all()

                                    # ── Positive / Negative contributors panel ──
                                    ExplanationPanel.render_shap_explanation(
                                        result, selected_movie, shap_target
                                    )

                                    # ── Global feature importance ──
                                    with st.expander("📊 Global Feature Importance (across all pairs)", expanded=False):
                                        fi = shap_explainer.get_feature_importance(n_samples=200)
                                        if len(fi) > 0:
                                            import plotly.express as px
                                            top_fi = fi.head(20)
                                            fig_fi = px.bar(
                                                top_fi, x='importance', y='feature',
                                                orientation='h',
                                                title='Top 20 Features by Mean |SHAP Value|',
                                                color='importance',
                                                color_continuous_scale='Reds',
                                                template='plotly_dark',
                                            )
                                            fig_fi.update_layout(
                                                yaxis=dict(autorange="reversed"),
                                                height=500,
                                            )
                                            st.plotly_chart(fig_fi, use_container_width=True)

                else:
                    st.warning("No recommendations found.")


def render_explore_page(movies_df, preprocessor):
    """Render the explore page."""
    st.markdown("## 🔍 Explore Movies")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        # Genre filter
        all_genres = set()
        for genres in movies_df['genres_list']:
            if isinstance(genres, list):
                all_genres.update(genres)
        selected_genre = st.selectbox(
            "Genre",
            options=["All"] + sorted(all_genres),
            key="explore_genre"
        )

    with col2:
        # Rating filter
        min_rating = st.slider(
            "Minimum Rating",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            key="explore_rating"
        )

    with col3:
        # Sort by
        sort_by = st.selectbox(
            "Sort by",
            options=["Popularity", "Rating", "Title"],
            key="explore_sort"
        )

    # Apply filters
    filtered = movies_df.copy()

    if selected_genre != "All":
        filtered = filtered[filtered['genres_list'].apply(
            lambda x: selected_genre in x if isinstance(x, list) else False
        )]

    filtered = filtered[filtered['vote_average'] >= min_rating]

    # Sort
    sort_map = {
        "Popularity": ("popularity", False),
        "Rating": ("vote_average", False),
        "Title": ("title", True)
    }
    sort_col, ascending = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    # Display count
    st.markdown(f"**{len(filtered):,}** movies found")

    # Pagination
    page_size = 20
    total_pages = (len(filtered) + page_size - 1) // page_size

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(total_pages, 1),
        value=1,
        key="explore_page"
    )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Display movies
    for _, row in filtered.iloc[start_idx:end_idx].iterrows():
        MovieCard.render(
            title=row['title'],
            genres=row.get('genres_list', []),
            rating=row.get('vote_average', 0),
            director=row.get('director'),
            year=str(row.get('release_date', ''))[:4] if row.get('release_date') else None,
            overview=row.get('overview'),
            show_explanation=False
        )

    st.markdown(f"Page {page} of {total_pages}")


def render_analytics_page(movies_df, recommender=None):
    """Render the analytics page."""
    st.markdown("## 📊 Analytics Dashboard")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Movies", f"{len(movies_df):,}")
    with col2:
        st.metric("Avg Rating", f"{movies_df['vote_average'].mean():.1f}")
    with col3:
        all_genres = set()
        for g in movies_df['genres_list']:
            if isinstance(g, list):
                all_genres.update(g)
        st.metric("Genres", len(all_genres))
    with col4:
        directors = movies_df['director'].dropna().nunique()
        st.metric("Directors", f"{directors:,}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Genre Distribution")
        genre_counts = {}
        for genres in movies_df['genres_list']:
            if isinstance(genres, list):
                for g in genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        Charts.genre_distribution(genre_counts)

    with col2:
        st.markdown("### Rating Distribution")
        Charts.rating_histogram(movies_df['vote_average'].tolist())

    # Movies over time
    st.markdown("### Movies Per Year")
    year_counts = {}
    for date_str in movies_df['release_date']:
        if date_str and isinstance(date_str, str) and len(date_str) >= 4:
            try:
                year = int(date_str[:4])
                if 1900 < year < 2025:
                    year_counts[year] = year_counts.get(year, 0) + 1
            except:
                pass
    Charts.movies_timeline(year_counts)

    # ── MovieLens Evaluation Section ──
    st.markdown("---")
    st.markdown("### 🧪 Recommendation Quality Evaluation (MovieLens)")
    st.caption(
        "Uses real user ratings from MovieLens 1M to measure how well each "
        "recommendation method predicts what users actually watch and enjoy."
    )

    eval_col1, eval_col2 = st.columns([1, 1])
    with eval_col1:
        eval_k = st.selectbox("Top-K", options=[5, 10, 20], index=1, key="eval_k")
    with eval_col2:
        eval_users = st.selectbox(
            "Max eval users", options=[100, 300, 500, 1000], index=1, key="eval_users"
        )

    if st.button("▶ Run Evaluation", type="primary"):
        if recommender is not None:
            with st.spinner("Loading MovieLens ratings & running evaluation… (may take 1-2 min)"):
                try:
                    from src.data.movielens_loader import MovieLensLoader
                    from src.evaluation import RecommenderEvaluator

                    ml = MovieLensLoader()
                    matrix, _, _ = ml.build_rating_matrix(movies_df, min_ratings_per_user=5)

                    evaluator = RecommenderEvaluator(
                        movies_df, matrix, test_ratio=0.2, seed=42
                    )
                    results_df = evaluator.evaluate_all_methods(
                        recommender, k=eval_k, max_eval_users=eval_users
                    )
                    st.session_state["eval_results"] = results_df
                    st.session_state["eval_k"] = eval_k
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    # Show results if available (persists across reruns)
    if "eval_results" in st.session_state:
        results_df = st.session_state["eval_results"]
        eval_k_display = st.session_state.get("eval_k", 10)
        import plotly.express as px

        if "error" in results_df.columns:
            results_df = results_df[results_df["error"].isna()].drop(columns=["error"], errors="ignore")

        st.markdown("#### 📈 Results")

        # Metric cards
        metric_cols = st.columns(len(results_df))
        for i, (_, row) in enumerate(results_df.iterrows()):
            with metric_cols[i]:
                st.markdown(
                    f'<div style="background:#1c2333;border-radius:12px;padding:16px;text-align:center;'
                    f'border:1px solid #30363d;">'
                    f'<h4 style="color:#f5c518;margin:0 0 8px 0;">{row["method"]}</h4>'
                    f'<p style="color:#fff;font-size:1.3rem;margin:0;">NDCG: {row["ndcg@k"]:.4f}</p>'
                    f'<p style="color:#8b949e;font-size:0.85rem;margin:4px 0 0 0;">'
                    f'P@{eval_k_display}: {row["precision@k"]:.4f}</p></div>',
                    unsafe_allow_html=True
                )

        # Full table
        display_cols = [
            "method", "precision@k", "recall@k", "ndcg@k",
            "map@k", "coverage", "novelty", "diversity"
        ]
        display_df = results_df[[c for c in display_cols if c in results_df.columns]].copy()
        for c in display_df.columns:
            if c != "method":
                display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        chart_metrics = ["precision@k", "recall@k", "ndcg@k", "map@k"]
        available = [m for m in chart_metrics if m in results_df.columns]
        if available:
            melted = results_df.melt(
                id_vars=["method"], value_vars=available,
                var_name="Metric", value_name="Score"
            )
            fig = px.bar(
                melted, x="Metric", y="Score", color="method",
                barmode="group", title="Accuracy Metrics by Method",
                color_discrete_sequence=["#58a6ff", "#3fb950", "#bc8cff", "#e5383b"]
            )
            fig.update_layout(
                height=400, legend_title_text="Method", template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Beyond-accuracy chart
        ba_metrics = ["coverage", "novelty", "diversity"]
        available_ba = [m for m in ba_metrics if m in results_df.columns]
        if available_ba:
            melted_ba = results_df.melt(
                id_vars=["method"], value_vars=available_ba,
                var_name="Metric", value_name="Score"
            )
            fig_ba = px.bar(
                melted_ba, x="Metric", y="Score", color="method",
                barmode="group", title="Beyond-Accuracy Metrics",
                color_discrete_sequence=["#58a6ff", "#3fb950", "#bc8cff", "#e5383b"],
                template="plotly_dark",
            )
            fig_ba.update_layout(
                height=400, legend_title_text="Method",
            )
            st.plotly_chart(fig_ba, use_container_width=True)

        st.success(f"✅ Evaluation complete — {results_df['n_eval_users'].iloc[0]} users, K={eval_k_display}")


def render_box_office_page(movies_df):
    """Render the standalone box office prediction page."""
    st.markdown("## 💰 Box Office Revenue Prediction")
    st.markdown(
        '<p style="color:#a3a3a3;font-size:1rem;margin-bottom:24px;">'
        'LightGBM + XGBoost ensemble trained on TMDB features '
        '(budget, popularity, vote, genres, runtime, release date…).<br>'
        'Reference: Movie-Analysis project — Kaggle top-7 %.</p>',
        unsafe_allow_html=True,
    )

    # ── Train / Evaluate ──
    if st.button("▶ Train & Evaluate (5-fold CV)", type="primary", key="run_bo_page"):
        with st.spinner("Training box office model …"):
            try:
                bp = _fit_box_office()
                st.session_state["bo_trained"] = True
            except ImportError:
                st.warning("Install `lightgbm`, `xgboost`, `catboost`: `pip install lightgbm xgboost catboost`")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Show results if model is trained (persists across reruns)
    if st.session_state.get("bo_trained"):
        try:
            import plotly.express as px
            bp = _fit_box_office()
            cv = bp.cv_results

            # Metric cards
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("RMSLE", f"{cv['rmsle']:.4f}")
            bc2.metric("MAE", f"${cv['mae']/1e6:.1f} M")
            bc3.metric("Training Movies", f"{cv['n_movies']:,}")
            rmsle_val = cv['rmsle']
            if rmsle_val < 1.85:
                rank_est = "~Top 3%"
            elif rmsle_val < 2.0:
                rank_est = "~Top 5-10%"
            elif rmsle_val < 2.2:
                rank_est = "~Top 10-15%"
            else:
                rank_est = "~Top 20%+"
            bc4.metric("Kaggle Est. Rank", rank_est)

            st.markdown(f"**Ensemble**: {cv.get('models', '')}")

            # Feature importance
            fi = bp.feature_importance()
            if len(fi) > 0:
                fig_fi = px.bar(
                    fi.head(15), x="importance", y="feature", orientation="h",
                    title="Top-15 Feature Importance (LightGBM gain)",
                    color="importance", color_continuous_scale="Reds",
                    template="plotly_dark",
                )
                fig_fi.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=450,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # Actual vs Predicted scatter
            kaggle_train, _ = bp.load_data()
            sample = kaggle_train.sample(min(500, len(kaggle_train)), random_state=42).copy()
            sample["predicted_revenue"] = bp.predict(sample)

            fig_sc = px.scatter(
                sample, x="revenue", y="predicted_revenue",
                hover_name="title", opacity=0.6,
                labels={"revenue": "Actual Revenue ($)", "predicted_revenue": "Predicted ($)"},
                title="Actual vs Predicted Revenue (Kaggle train sample)",
                template="plotly_dark",
            )
            max_val = max(sample["revenue"].max(), sample["predicted_revenue"].max())
            fig_sc.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="#e5383b", dash="dash"),
            )
            fig_sc.update_layout(height=500)
            st.plotly_chart(fig_sc, use_container_width=True)

        except Exception as e:
            st.error(f"Chart rendering failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.markdown("---")

    # ── Predict for a single movie ──
    st.markdown("### 🔮 Predict Revenue for a Movie")
    st.caption("Select a movie from the Kaggle competition training set.")

    try:
        bp_loaded = _fit_box_office()
        kaggle_train, _ = bp_loaded.load_data()
        valid_movies = kaggle_train[kaggle_train["budget"] > 0].nlargest(200, "popularity")
        selected = st.selectbox(
            "Select movie",
            options=valid_movies["title"].tolist(),
            key="bo_single_movie",
        )

        if selected and st.button("Predict", key="bo_predict_single"):
            row = kaggle_train[kaggle_train["title"] == selected].head(1)
            pred = bp_loaded.predict(row)[0]
            actual = float(row["revenue"].iloc[0])
            budget = float(row["budget"].iloc[0])

            c1, c2, c3 = st.columns(3)
            c1.metric("Budget", f"${budget/1e6:,.1f} M")
            c2.metric("Predicted Revenue", f"${pred/1e6:,.1f} M")
            if actual > 0:
                c3.metric("Actual Revenue", f"${actual/1e6:,.1f} M")
            else:
                c3.metric("Actual Revenue", "N/A")

            if actual > 0:
                error_pct = abs(pred - actual) / actual * 100
                roi_actual = (actual - budget) / budget * 100 if budget > 0 else 0
                roi_pred = (pred - budget) / budget * 100 if budget > 0 else 0
                st.markdown(
                    f'<div style="background:#1c2333;border-radius:12px;padding:16px;margin-top:12px;">'
                    f'<p style="color:#a3a3a3;margin:0;">Prediction error: <b style="color:#f5c518;">{error_pct:.1f}%</b>'
                    f' &nbsp;|&nbsp; Actual ROI: <b style="color:#2ecc71;">{roi_actual:.0f}%</b>'
                    f' &nbsp;|&nbsp; Predicted ROI: <b style="color:#3498db;">{roi_pred:.0f}%</b></p></div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.info("Train the model first by clicking the button above.")


def render_compare_page(movies_df, recommender, preprocessor, settings):
    """Render the compare methods page."""
    st.markdown("## ⚖️ Compare Methods")

    popular = movies_df.nlargest(100, 'popularity')['title'].tolist()

    selected_movie = st.selectbox(
        "Select a movie to compare",
        options=popular,
        key="compare_movie"
    )

    if selected_movie:
        with st.spinner("Comparing recommendation methods..."):
            comparisons = recommender.compare_methods(selected_movie, 5)

            if comparisons:
                RecommendationList.render_comparison(comparisons, selected_movie)

                # Similarity scores visualization
                if len(comparisons.get('hybrid', [])) > 0:
                    st.markdown("### Method Score Comparison")

                    first_rec = comparisons['hybrid'].iloc[0]['title']
                    scores = recommender.get_method_scores(selected_movie, first_rec)

                    if scores:
                        Charts.similarity_radar(scores)


def render_about_page():
    """Render the about page."""
    st.markdown("""
    ## ℹ️ About CineMatch

    CineMatch is an intelligent movie recommendation system that uses three different
    machine learning approaches to suggest movies you might enjoy.

    ### 🔬 Methods

    **1. Content-Based Filtering (TF-IDF)**
    - Analyzes movie plot descriptions
    - Uses Term Frequency-Inverse Document Frequency
    - Finds movies with similar themes and storylines

    **2. Metadata-Based Filtering**
    - Considers genres, directors, cast, and keywords
    - Weighted feature matching
    - Great for finding movies with similar production elements

    **3. Collaborative Filtering**
    - Based on user behavior patterns
    - Uses simulated user-movie ratings
    - Discovers hidden connections between movies

    **4. Hybrid Approach**
    - Combines all three methods
    - Customizable weights
    - Provides the most comprehensive recommendations

    ### 📊 Dataset

    - **Source:** TMDB 5000 Movies Dataset
    - **Movies:** ~4,800 titles
    - **Features:** Plot, genres, cast, crew, ratings, and more

    ### 🛠️ Technology Stack

    - **Frontend:** Streamlit
    - **Backend:** FastAPI
    - **ML:** scikit-learn, SHAP
    - **Visualization:** Plotly

    ---

    Built with ❤️ for movie lovers everywhere.
    """)


if __name__ == "__main__":
    main()
