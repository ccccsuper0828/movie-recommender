"""
电影推荐系统 - Streamlit Web Demo
===================================
三种机器学习推荐方法的交互式演示界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 页面配置
st.set_page_config(
    page_title="电影推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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


# ==================== 数据加载和预处理 ====================

@st.cache_data
def load_and_preprocess_data():
    """加载和预处理数据"""
    # 加载数据 - 使用绝对路径确保在任何环境下都能找到文件
    movies_path = os.path.join(SCRIPT_DIR, 'tmdb_5000_movies.csv')
    credits_path = os.path.join(SCRIPT_DIR, 'tmdb_5000_credits.csv')

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # 辅助函数
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

    # 合并数据集
    credits.columns = ['id', 'title', 'cast', 'crew']
    movies = movies.merge(credits[['id', 'cast', 'crew']], on='id')

    # 解析JSON字段
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(safe_literal_eval)

    # 提取特征
    movies['genres_list'] = movies['genres'].apply(get_list)
    movies['keywords_list'] = movies['keywords'].apply(get_list)
    movies['cast_list'] = movies['cast'].apply(get_top_cast)
    movies['director'] = movies['crew'].apply(get_director)

    # 清理文本
    movies['genres_clean'] = movies['genres_list'].apply(clean_text)
    movies['keywords_clean'] = movies['keywords_list'].apply(clean_text)
    movies['cast_clean'] = movies['cast_list'].apply(clean_text)
    movies['director_clean'] = movies['director'].apply(clean_text)

    # 处理缺失值
    movies['overview'] = movies['overview'].fillna('')
    movies['release_date'] = movies['release_date'].fillna('')
    movies['runtime'] = movies['runtime'].fillna(0)

    # 重置索引
    movies = movies.reset_index(drop=True)

    return movies


@st.cache_data
def build_content_similarity(_movies):
    """构建基于内容的相似度矩阵"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(_movies['overview'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)


@st.cache_data
def build_metadata_similarity(_movies):
    """构建基于元数据的相似度矩阵"""
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
    """构建协同过滤相似度矩阵"""
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


# ==================== 推荐函数 ====================

def get_recommendations(title, movies, similarity_matrix, top_n=10):
    """获取推荐结果"""
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
    """获取混合推荐结果"""
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


# ==================== 显示函数 ====================

def display_movie_info(movie_row):
    """显示电影详细信息"""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### 🎬 {movie_row['title']}")
        st.markdown(f"**评分:** ⭐ {movie_row['vote_average']}/10")
        st.markdown(f"**年份:** {movie_row['release_date'][:4] if movie_row['release_date'] else 'N/A'}")
        st.markdown(f"**时长:** {int(movie_row['runtime'])} 分钟")

    with col2:
        st.markdown(f"**类型:** {', '.join(movie_row['genres_list'])}")
        st.markdown(f"**导演:** {movie_row['director']}")
        st.markdown(f"**主演:** {', '.join(movie_row['cast_list'][:3])}")

    st.markdown(f"**简介:** {movie_row['overview'][:300]}..." if len(movie_row['overview']) > 300 else f"**简介:** {movie_row['overview']}")


def display_recommendations(recommendations, method_name):
    """显示推荐结果"""
    if recommendations is None or len(recommendations) == 0:
        st.warning("未找到推荐结果")
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
                st.metric("相似度", f"{score:.1%}")

            st.divider()


# ==================== 主应用 ====================

def main():
    # 标题
    st.markdown('<div class="main-header">🎬 电影推荐系统</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">基于三种机器学习方法的智能推荐</p>', unsafe_allow_html=True)

    # 加载数据
    with st.spinner('正在加载数据...'):
        movies = load_and_preprocess_data()

    # 构建相似度矩阵
    with st.spinner('正在构建推荐模型...'):
        content_similarity = build_content_similarity(movies)
        metadata_similarity = build_metadata_similarity(movies)
        cf_similarity, user_movie_matrix = build_cf_similarity(movies)

    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 设置")

        # 电影搜索
        st.markdown("### 🔍 选择电影")

        # 搜索框
        search_query = st.text_input("搜索电影", placeholder="输入电影名称...")

        if search_query:
            filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
            movie_options = filtered_movies['title'].tolist()[:20]
        else:
            # 默认显示热门电影
            movie_options = movies.nlargest(50, 'popularity')['title'].tolist()

        selected_movie = st.selectbox(
            "选择一部电影",
            options=movie_options,
            index=0 if movie_options else None
        )

        st.markdown("---")

        # 推荐数量
        top_n = st.slider("推荐数量", min_value=5, max_value=20, value=10)

        st.markdown("---")

        # 混合推荐权重
        st.markdown("### ⚖️ 混合推荐权重")
        w1 = st.slider("内容相似度权重", 0.0, 1.0, 0.3, 0.1)
        w2 = st.slider("元数据相似度权重", 0.0, 1.0, 0.4, 0.1)
        w3 = st.slider("协同过滤权重", 0.0, 1.0, 0.3, 0.1)

        # 归一化权重
        total = w1 + w2 + w3
        if total > 0:
            weights = (w1/total, w2/total, w3/total)
        else:
            weights = (0.33, 0.34, 0.33)

        st.caption(f"归一化: {weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}")

        st.markdown("---")
        st.markdown("### 📊 数据统计")
        st.metric("电影总数", f"{len(movies):,}")
        st.metric("模拟用户数", "500")

    # 主内容区
    if selected_movie:
        # 显示选中的电影信息
        st.markdown("## 📽️ 选中的电影")
        movie_idx = movies[movies['title'] == selected_movie].index[0]
        movie_info = movies.iloc[movie_idx]

        with st.container():
            display_movie_info(movie_info)

        st.markdown("---")

        # 推荐结果标签页
        st.markdown("## 🎯 推荐结果")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 基于内容",
            "🏷️ 基于元数据",
            "👥 协同过滤",
            "🔀 混合推荐"
        ])

        with tab1:
            st.markdown("""
            <div class="method-header">
            <b>方法说明:</b> 使用TF-IDF分析电影简介文本，通过余弦相似度找到内容相似的电影
            </div>
            """, unsafe_allow_html=True)

            content_recs = get_recommendations(selected_movie, movies, content_similarity, top_n)
            display_recommendations(content_recs, "基于内容")

        with tab2:
            st.markdown("""
            <div class="method-header">
            <b>方法说明:</b> 综合分析电影类型、导演、演员、关键词等元数据进行推荐
            </div>
            """, unsafe_allow_html=True)

            metadata_recs = get_recommendations(selected_movie, movies, metadata_similarity, top_n)
            display_recommendations(metadata_recs, "基于元数据")

        with tab3:
            st.markdown("""
            <div class="method-header">
            <b>方法说明:</b> 基于模拟用户行为数据，使用协同过滤算法找到用户可能喜欢的电影
            </div>
            """, unsafe_allow_html=True)

            cf_recs = get_recommendations(selected_movie, movies, cf_similarity, top_n)
            display_recommendations(cf_recs, "协同过滤")

        with tab4:
            st.markdown(f"""
            <div class="method-header">
            <b>方法说明:</b> 综合三种方法的加权结果 (内容:{weights[0]:.0%}, 元数据:{weights[1]:.0%}, 协同过滤:{weights[2]:.0%})
            </div>
            """, unsafe_allow_html=True)

            hybrid_recs = get_hybrid_recommendations(
                selected_movie, movies,
                content_similarity, metadata_similarity, cf_similarity,
                weights, top_n
            )
            display_recommendations(hybrid_recs, "混合推荐")

        # 三种方法对比
        st.markdown("---")
        st.markdown("## 📊 三种方法对比")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📝 基于内容 Top 5")
            if content_recs is not None:
                for _, row in content_recs.head(5).iterrows():
                    st.markdown(f"- **{row['title']}** ({row['similarity_score']:.1%})")

        with col2:
            st.markdown("### 🏷️ 基于元数据 Top 5")
            if metadata_recs is not None:
                for _, row in metadata_recs.head(5).iterrows():
                    st.markdown(f"- **{row['title']}** ({row['similarity_score']:.1%})")

        with col3:
            st.markdown("### 👥 协同过滤 Top 5")
            if cf_recs is not None:
                for _, row in cf_recs.head(5).iterrows():
                    st.markdown(f"- **{row['title']}** ({row['similarity_score']:.1%})")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🎬 电影推荐系统 Demo | 基于 TMDB 5000 数据集</p>
        <p>三种推荐方法: 基于内容 (TF-IDF) | 基于元数据 | 协同过滤</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
