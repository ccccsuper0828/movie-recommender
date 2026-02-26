"""
电影推荐系统 - 使用三种机器学习方法
=============================================
1. 基于内容的推荐 (Content-Based Filtering) - 使用TF-IDF和余弦相似度
2. 基于元数据的推荐 (Metadata-Based) - 使用类型、关键词、演员、导演
3. 协同过滤推荐 (Collaborative Filtering) - 基于用户-电影交互矩阵

作者: Movie Recommendation System
数据集: TMDB 5000 Movies Dataset
"""

import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

# 机器学习和NLP库
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns


class MovieRecommenderSystem:
    """
    电影推荐系统类
    包含三种不同的推荐算法
    """

    def __init__(self, movies_path, credits_path):
        """
        初始化推荐系统

        Parameters:
        -----------
        movies_path : str
            电影数据文件路径
        credits_path : str
            演职人员数据文件路径
        """
        print("=" * 60)
        print("电影推荐系统初始化中...")
        print("=" * 60)

        # 加载数据
        self.movies = pd.read_csv(movies_path)
        self.credits = pd.read_csv(credits_path)

        print(f"加载了 {len(self.movies)} 部电影")

        # 预处理数据
        self._preprocess_data()

        # 初始化各种相似度矩阵
        self.content_similarity = None
        self.metadata_similarity = None

        # 保存向量化器和矩阵（用于可解释性）
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.explainer = None

        # 协同过滤相关
        self.user_movie_matrix = None
        self.item_similarity_cf = None
        self.user_similarity_cf = None
        self.svd_user_factors = None
        self.svd_item_factors = None
        self.svd_predictions = None
        self.user_ratings_mean = None
        self.item_knn = None
        self.user_knn = None
        self.svd_sigma = None
        self.n_users = 1000  # 模拟用户数量

    def _safe_literal_eval(self, x):
        """安全地解析JSON字符串"""
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    def _get_list(self, x):
        """从字典列表中提取名称"""
        if isinstance(x, list):
            return [i['name'] for i in x][:5]  # 最多取5个
        return []

    def _get_director(self, x):
        """从crew中提取导演"""
        if isinstance(x, list):
            for crew_member in x:
                if crew_member.get('job') == 'Director':
                    return crew_member.get('name', '')
        return ''

    def _get_top_cast(self, x, n=5):
        """获取前n个演员"""
        if isinstance(x, list):
            return [i['name'] for i in x][:n]
        return []

    def _clean_text(self, x):
        """清理文本，移除空格并转为小写"""
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        elif isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ''

    def _preprocess_data(self):
        """数据预处理"""
        print("\n数据预处理中...")

        # 合并数据集
        self.credits.columns = ['id', 'title', 'cast', 'crew']
        self.movies = self.movies.merge(self.credits[['id', 'cast', 'crew']], on='id')

        # 解析JSON字段
        json_columns = ['genres', 'keywords', 'cast', 'crew']
        for col in json_columns:
            if col in self.movies.columns:
                self.movies[col] = self.movies[col].apply(self._safe_literal_eval)

        # 提取特征
        self.movies['genres_list'] = self.movies['genres'].apply(self._get_list)
        self.movies['keywords_list'] = self.movies['keywords'].apply(self._get_list)
        self.movies['cast_list'] = self.movies['cast'].apply(self._get_top_cast)
        self.movies['director'] = self.movies['crew'].apply(self._get_director)

        # 清理文本
        self.movies['genres_clean'] = self.movies['genres_list'].apply(self._clean_text)
        self.movies['keywords_clean'] = self.movies['keywords_list'].apply(self._clean_text)
        self.movies['cast_clean'] = self.movies['cast_list'].apply(self._clean_text)
        self.movies['director_clean'] = self.movies['director'].apply(self._clean_text)

        # 处理overview中的缺失值
        self.movies['overview'] = self.movies['overview'].fillna('')

        # 创建索引映射
        self.movies = self.movies.reset_index(drop=True)
        self.title_to_idx = pd.Series(self.movies.index, index=self.movies['title']).to_dict()
        self.idx_to_title = pd.Series(self.movies['title'], index=self.movies.index).to_dict()

        print(f"预处理完成！共处理 {len(self.movies)} 部电影")

    # =============================================
    # 方法1: 基于内容的推荐 (Content-Based Filtering)
    # =============================================

    def build_content_based_model(self):
        """
        构建基于内容的推荐模型
        使用TF-IDF向量化电影简介，然后计算余弦相似度
        """
        print("\n" + "=" * 60)
        print("方法1: 构建基于内容的推荐模型 (TF-IDF + 余弦相似度)")
        print("=" * 60)

        # TF-IDF向量化
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )

        # 对电影简介进行向量化
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['overview'])

        print(f"TF-IDF矩阵形状: {self.tfidf_matrix.shape}")
        print(f"特征数量: {len(self.tfidf_vectorizer.get_feature_names_out())}")

        # 计算余弦相似度
        self.content_similarity = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        print("基于内容的推荐模型构建完成！")

        return self.content_similarity

    def content_based_recommend(self, title, top_n=10):
        """
        基于内容推荐电影

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            推荐数量

        Returns:
        --------
        DataFrame : 推荐的电影列表
        """
        if self.content_similarity is None:
            self.build_content_based_model()

        # 获取电影索引
        if title not in self.title_to_idx:
            print(f"找不到电影: {title}")
            # 尝试模糊匹配
            matches = [t for t in self.title_to_idx.keys() if title.lower() in t.lower()]
            if matches:
                print(f"您是否在找: {matches[:5]}")
            return None

        idx = self.title_to_idx[title]

        # 获取相似度分数
        sim_scores = list(enumerate(self.content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # 排除自己

        # 获取电影索引
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        # 返回推荐结果
        recommendations = self.movies.iloc[movie_indices][['title', 'genres_list', 'vote_average', 'overview']].copy()
        recommendations['similarity_score'] = scores

        return recommendations

    # =============================================
    # 方法2: 基于元数据的推荐 (Metadata-Based)
    # =============================================

    def build_metadata_based_model(self):
        """
        构建基于元数据的推荐模型
        结合类型、关键词、演员、导演信息
        """
        print("\n" + "=" * 60)
        print("方法2: 构建基于元数据的推荐模型 (类型+关键词+演员+导演)")
        print("=" * 60)

        # 创建综合特征"汤"
        def create_soup(row):
            """创建综合特征字符串"""
            features = []

            # 类型（权重较高，重复3次）
            features.extend(row['genres_clean'] * 3)

            # 导演（权重较高，重复3次）
            if row['director_clean']:
                features.extend([row['director_clean']] * 3)

            # 演员（权重中等，重复2次）
            features.extend(row['cast_clean'] * 2)

            # 关键词
            features.extend(row['keywords_clean'])

            return ' '.join(features)

        self.movies['soup'] = self.movies.apply(create_soup, axis=1)

        # 使用CountVectorizer（因为特征已经是处理过的词）
        count_vectorizer = CountVectorizer(
            stop_words='english',
            max_features=10000
        )

        count_matrix = count_vectorizer.fit_transform(self.movies['soup'])

        print(f"特征矩阵形状: {count_matrix.shape}")

        # 计算余弦相似度
        self.metadata_similarity = cosine_similarity(count_matrix, count_matrix)

        print("基于元数据的推荐模型构建完成！")

        return self.metadata_similarity

    def metadata_based_recommend(self, title, top_n=10):
        """
        基于元数据推荐电影

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            推荐数量

        Returns:
        --------
        DataFrame : 推荐的电影列表
        """
        if self.metadata_similarity is None:
            self.build_metadata_based_model()

        # 获取电影索引
        if title not in self.title_to_idx:
            print(f"找不到电影: {title}")
            matches = [t for t in self.title_to_idx.keys() if title.lower() in t.lower()]
            if matches:
                print(f"您是否在找: {matches[:5]}")
            return None

        idx = self.title_to_idx[title]

        # 获取相似度分数
        sim_scores = list(enumerate(self.metadata_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        # 返回推荐结果
        recommendations = self.movies.iloc[movie_indices][
            ['title', 'genres_list', 'director', 'cast_list', 'vote_average']
        ].copy()
        recommendations['similarity_score'] = scores

        return recommendations

    # =============================================
    # 方法3: 协同过滤推荐 (Collaborative Filtering)
    # =============================================

    def _generate_user_movie_matrix(self, n_users=1000, sparsity=0.02):
        """
        生成模拟的用户-电影评分矩阵

        由于TMDB数据集没有用户评分数据，我们基于电影特征模拟用户偏好

        Parameters:
        -----------
        n_users : int
            模拟用户数量
        sparsity : float
            矩阵稀疏度（用户观看电影的比例）
        """
        print("生成模拟用户-电影评分矩阵...")

        n_movies = len(self.movies)
        self.n_users = n_users

        # 初始化评分矩阵
        ratings = np.zeros((n_users, n_movies))

        # 获取所有类型
        all_genres = set()
        for genres in self.movies['genres_list']:
            all_genres.update(genres)
        genre_list = sorted(list(all_genres))

        # 为每个用户分配偏好类型
        np.random.seed(42)

        for user_id in range(n_users):
            # 用户偏好的类型（随机选择1-3个）
            n_preferred = np.random.randint(1, 4)
            preferred_genres = np.random.choice(genre_list, size=min(n_preferred, len(genre_list)), replace=False)

            # 用户观看的电影数量
            n_watched = int(n_movies * sparsity * np.random.uniform(0.5, 2.0))
            n_watched = max(10, min(n_watched, n_movies // 10))

            # 选择电影（偏好类型的电影更容易被选中）
            movie_probs = np.ones(n_movies) * 0.1

            for movie_idx in range(n_movies):
                movie_genres = self.movies.iloc[movie_idx]['genres_list']
                # 如果电影包含用户偏好的类型，增加选择概率
                if any(g in preferred_genres for g in movie_genres):
                    movie_probs[movie_idx] = 0.8
                # 高评分电影也更容易被选中
                vote_avg = self.movies.iloc[movie_idx]['vote_average']
                movie_probs[movie_idx] *= (vote_avg / 10.0 + 0.5)

            # 归一化概率
            movie_probs = movie_probs / movie_probs.sum()

            # 选择电影
            watched_movies = np.random.choice(n_movies, size=n_watched, replace=False, p=movie_probs)

            # 生成评分（1-5分）
            for movie_idx in watched_movies:
                movie_genres = self.movies.iloc[movie_idx]['genres_list']
                vote_avg = self.movies.iloc[movie_idx]['vote_average']

                # 基础评分
                base_rating = 2.5 + (vote_avg - 5) / 2  # 映射到2.5附近

                # 如果是用户偏好类型，评分更高
                if any(g in preferred_genres for g in movie_genres):
                    base_rating += np.random.uniform(0.5, 1.5)
                else:
                    base_rating += np.random.uniform(-1.0, 0.5)

                # 添加噪声
                rating = base_rating + np.random.normal(0, 0.5)

                # 限制在1-5之间
                rating = np.clip(rating, 1, 5)
                ratings[user_id, movie_idx] = round(rating, 1)

        self.user_movie_matrix = ratings

        # 计算统计信息
        n_ratings = np.count_nonzero(ratings)
        actual_sparsity = 1 - (n_ratings / (n_users * n_movies))

        print(f"用户-电影矩阵形状: {ratings.shape}")
        print(f"评分数量: {n_ratings}")
        print(f"矩阵稀疏度: {actual_sparsity:.2%}")

        return self.user_movie_matrix

    def build_collaborative_filtering_model(self, method='item_based'):
        """
        构建协同过滤推荐模型

        Parameters:
        -----------
        method : str
            'item_based' - 基于物品的协同过滤
            'user_based' - 基于用户的协同过滤
            'svd' - 基于矩阵分解的协同过滤
        """
        print("\n" + "=" * 60)
        print("方法3: 构建协同过滤推荐模型")
        print("=" * 60)

        # 如果还没有用户-电影矩阵，先生成
        if self.user_movie_matrix is None:
            self._generate_user_movie_matrix()

        if method == 'item_based':
            self._build_item_based_cf()
        elif method == 'user_based':
            self._build_user_based_cf()
        elif method == 'svd':
            self._build_svd_cf()
        else:
            # 默认构建所有方法
            self._build_item_based_cf()
            self._build_user_based_cf()
            self._build_svd_cf()

        print("协同过滤模型构建完成！")

    def _build_item_based_cf(self):
        """构建基于物品的协同过滤"""
        print("\n构建基于物品的协同过滤 (Item-Based CF)...")

        # 转置矩阵，使电影在行上
        movie_user_matrix = self.user_movie_matrix.T

        # 创建稀疏矩阵
        sparse_matrix = csr_matrix(movie_user_matrix)

        # 使用KNN找到相似物品
        self.item_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        self.item_knn.fit(sparse_matrix)

        # 计算物品相似度矩阵（用于快速查找）
        self.item_similarity_cf = cosine_similarity(movie_user_matrix)

        print(f"物品相似度矩阵形状: {self.item_similarity_cf.shape}")

    def _build_user_based_cf(self):
        """构建基于用户的协同过滤"""
        print("\n构建基于用户的协同过滤 (User-Based CF)...")

        # 创建稀疏矩阵
        sparse_matrix = csr_matrix(self.user_movie_matrix)

        # 使用KNN找到相似用户
        self.user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        self.user_knn.fit(sparse_matrix)

        # 计算用户相似度矩阵
        self.user_similarity_cf = cosine_similarity(self.user_movie_matrix)

        print(f"用户相似度矩阵形状: {self.user_similarity_cf.shape}")

    def _build_svd_cf(self, n_factors=50):
        """构建基于SVD矩阵分解的协同过滤"""
        print("\n构建基于SVD的协同过滤 (Matrix Factorization)...")

        # 中心化评分矩阵
        user_ratings_mean = np.mean(self.user_movie_matrix, axis=1)
        ratings_centered = self.user_movie_matrix - user_ratings_mean.reshape(-1, 1)

        # SVD分解
        n_factors = min(n_factors, min(self.user_movie_matrix.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(ratings_centered), k=n_factors)

        # 转换为对角矩阵
        sigma = np.diag(sigma)

        # 保存分解结果
        self.svd_user_factors = U
        self.svd_sigma = sigma
        self.svd_item_factors = Vt
        self.user_ratings_mean = user_ratings_mean

        # 预测评分矩阵
        self.svd_predictions = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        print(f"SVD分解 - 用户因子: {U.shape}, 物品因子: {Vt.shape}")
        print(f"潜在因子数量: {n_factors}")

    def collaborative_filtering_recommend(self, title, top_n=10, method='item_based'):
        """
        基于协同过滤推荐电影

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            推荐数量
        method : str
            'item_based' - 基于物品的协同过滤
            'user_based' - 基于用户的协同过滤（为随机用户推荐）
            'svd' - 基于SVD的协同过滤

        Returns:
        --------
        DataFrame : 推荐的电影列表
        """
        # 确保模型已构建
        if self.user_movie_matrix is None:
            self.build_collaborative_filtering_model()

        # 获取电影索引
        if title not in self.title_to_idx:
            print(f"找不到电影: {title}")
            matches = [t for t in self.title_to_idx.keys() if title.lower() in t.lower()]
            if matches:
                print(f"您是否在找: {matches[:5]}")
            return None

        movie_idx = self.title_to_idx[title]

        if method == 'item_based':
            return self._item_based_recommend(movie_idx, top_n)
        elif method == 'user_based':
            return self._user_based_recommend(movie_idx, top_n)
        elif method == 'svd':
            return self._svd_recommend(movie_idx, top_n)
        else:
            return self._item_based_recommend(movie_idx, top_n)

    def _item_based_recommend(self, movie_idx, top_n=10):
        """基于物品的协同过滤推荐"""
        if self.item_similarity_cf is None:
            self._build_item_based_cf()

        # 获取与目标电影最相似的电影
        sim_scores = list(enumerate(self.item_similarity_cf[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # 排除自己

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        recommendations = self.movies.iloc[movie_indices][
            ['title', 'genres_list', 'vote_average', 'popularity']
        ].copy()
        recommendations['cf_similarity'] = scores

        return recommendations

    def _user_based_recommend(self, movie_idx, top_n=10):
        """基于用户的协同过滤推荐"""
        if self.user_similarity_cf is None:
            self._build_user_based_cf()

        # 找到喜欢这部电影的用户
        movie_ratings = self.user_movie_matrix[:, movie_idx]
        liked_users = np.where(movie_ratings >= 4.0)[0]  # 评分>=4的用户

        if len(liked_users) == 0:
            liked_users = np.where(movie_ratings > 0)[0][:10]

        if len(liked_users) == 0:
            return self._item_based_recommend(movie_idx, top_n)

        # 聚合这些用户的偏好
        user_preferences = np.zeros(len(self.movies))

        for user_id in liked_users:
            user_ratings = self.user_movie_matrix[user_id]
            # 找到相似用户
            similar_users = np.argsort(self.user_similarity_cf[user_id])[::-1][1:11]

            for sim_user in similar_users:
                sim_score = self.user_similarity_cf[user_id, sim_user]
                user_preferences += sim_score * self.user_movie_matrix[sim_user]

        # 排除已经是目标电影的
        user_preferences[movie_idx] = -1

        # 获取推荐
        top_indices = np.argsort(user_preferences)[::-1][:top_n]
        scores = user_preferences[top_indices]

        # 归一化分数
        if scores.max() > 0:
            scores = scores / scores.max()

        recommendations = self.movies.iloc[top_indices][
            ['title', 'genres_list', 'vote_average', 'popularity']
        ].copy()
        recommendations['cf_score'] = scores

        return recommendations

    def _svd_recommend(self, movie_idx, top_n=10):
        """基于SVD的协同过滤推荐"""
        if self.svd_item_factors is None:
            self._build_svd_cf()

        # 使用物品因子计算相似度
        item_factors = self.svd_item_factors.T  # (n_movies, n_factors)

        target_factor = item_factors[movie_idx].reshape(1, -1)
        similarities = cosine_similarity(target_factor, item_factors)[0]

        # 获取最相似的电影
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        recommendations = self.movies.iloc[movie_indices][
            ['title', 'genres_list', 'vote_average', 'popularity']
        ].copy()
        recommendations['svd_similarity'] = scores

        return recommendations

    def recommend_for_user(self, user_id, top_n=10, method='svd'):
        """
        为特定用户推荐电影

        Parameters:
        -----------
        user_id : int
            用户ID
        top_n : int
            推荐数量
        method : str
            推荐方法
        """
        if self.user_movie_matrix is None:
            self.build_collaborative_filtering_model()

        if user_id >= self.n_users:
            print(f"用户ID超出范围（最大: {self.n_users - 1}）")
            return None

        # 获取用户已看过的电影
        watched = np.where(self.user_movie_matrix[user_id] > 0)[0]

        if method == 'svd':
            # 使用SVD预测评分
            if self.svd_predictions is None:
                self._build_svd_cf()

            predictions = self.svd_predictions[user_id].copy()
            predictions[watched] = -1  # 排除已看过的

            top_indices = np.argsort(predictions)[::-1][:top_n]
            scores = predictions[top_indices]
        else:
            # 使用用户相似度
            if self.user_similarity_cf is None:
                self._build_user_based_cf()

            similar_users = np.argsort(self.user_similarity_cf[user_id])[::-1][1:21]

            scores_agg = np.zeros(len(self.movies))
            for sim_user in similar_users:
                sim_score = self.user_similarity_cf[user_id, sim_user]
                scores_agg += sim_score * self.user_movie_matrix[sim_user]

            scores_agg[watched] = -1
            top_indices = np.argsort(scores_agg)[::-1][:top_n]
            scores = scores_agg[top_indices]

        recommendations = self.movies.iloc[top_indices][
            ['title', 'genres_list', 'vote_average', 'popularity']
        ].copy()
        recommendations['predicted_score'] = scores

        return recommendations

    # =============================================
    # 综合推荐和评估
    # =============================================

    def hybrid_recommend(self, title, top_n=10, weights=(0.3, 0.4, 0.3)):
        """
        混合推荐：结合三种方法

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            推荐数量
        weights : tuple
            三种方法的权重 (内容, 元数据, 协同过滤)

        Returns:
        --------
        DataFrame : 综合推荐的电影列表
        """
        print("\n" + "=" * 60)
        print("混合推荐：结合三种方法")
        print(f"权重分配 - 内容: {weights[0]}, 元数据: {weights[1]}, 协同过滤: {weights[2]}")
        print("=" * 60)

        # 确保所有模型都已构建
        if self.content_similarity is None:
            self.build_content_based_model()
        if self.metadata_similarity is None:
            self.build_metadata_based_model()
        if self.item_similarity_cf is None:
            self.build_collaborative_filtering_model()

        if title not in self.title_to_idx:
            print(f"找不到电影: {title}")
            return None

        idx = self.title_to_idx[title]
        n_movies = len(self.movies)

        # 计算综合分数
        hybrid_scores = np.zeros(n_movies)

        # 内容相似度
        hybrid_scores += weights[0] * self.content_similarity[idx]

        # 元数据相似度
        hybrid_scores += weights[1] * self.metadata_similarity[idx]

        # 协同过滤相似度
        hybrid_scores += weights[2] * self.item_similarity_cf[idx]

        # 排序
        sim_scores = list(enumerate(hybrid_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        recommendations = self.movies.iloc[movie_indices][
            ['title', 'genres_list', 'director', 'vote_average', 'popularity']
        ].copy()
        recommendations['hybrid_score'] = scores

        return recommendations

    def compare_recommendations(self, title, top_n=5):
        """
        比较三种方法的推荐结果

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            每种方法的推荐数量
        """
        print("\n" + "=" * 60)
        print(f"比较三种推荐方法的结果")
        print(f"目标电影: {title}")
        print("=" * 60)

        # 获取目标电影信息
        if title in self.title_to_idx:
            idx = self.title_to_idx[title]
            movie_info = self.movies.iloc[idx]
            print(f"\n电影信息:")
            print(f"  类型: {movie_info['genres_list']}")
            print(f"  导演: {movie_info['director']}")
            print(f"  评分: {movie_info['vote_average']}")
            print(f"  简介: {movie_info['overview'][:200]}...")

        # 方法1推荐
        print("\n" + "-" * 40)
        print("方法1: 基于内容的推荐 (TF-IDF)")
        print("-" * 40)
        content_rec = self.content_based_recommend(title, top_n)
        if content_rec is not None:
            for i, row in content_rec.iterrows():
                print(f"  {row['title']} (相似度: {row['similarity_score']:.3f}, 评分: {row['vote_average']})")

        # 方法2推荐
        print("\n" + "-" * 40)
        print("方法2: 基于元数据的推荐")
        print("-" * 40)
        metadata_rec = self.metadata_based_recommend(title, top_n)
        if metadata_rec is not None:
            for i, row in metadata_rec.iterrows():
                print(f"  {row['title']} (相似度: {row['similarity_score']:.3f}, 导演: {row['director']})")

        # 方法3推荐 - 协同过滤
        print("\n" + "-" * 40)
        print("方法3: 协同过滤推荐 (Item-Based CF)")
        print("-" * 40)
        cf_rec = self.collaborative_filtering_recommend(title, top_n, method='item_based')
        if cf_rec is not None:
            for i, row in cf_rec.iterrows():
                print(f"  {row['title']} (CF相似度: {row['cf_similarity']:.3f}, 评分: {row['vote_average']})")

        # 混合推荐
        print("\n" + "-" * 40)
        print("混合推荐 (综合三种方法)")
        print("-" * 40)
        hybrid_rec = self.hybrid_recommend(title, top_n)
        if hybrid_rec is not None:
            for i, row in hybrid_rec.iterrows():
                print(f"  {row['title']} (综合分数: {row['hybrid_score']:.3f}, 评分: {row['vote_average']})")

        return {
            'content_based': content_rec,
            'metadata_based': metadata_rec,
            'collaborative_filtering': cf_rec,
            'hybrid': hybrid_rec
        }

    def visualize_user_movie_matrix(self, n_users=50, n_movies=100):
        """
        可视化用户-电影评分矩阵
        """
        print("\n生成用户-电影矩阵可视化...")

        if self.user_movie_matrix is None:
            self._generate_user_movie_matrix()

        # 取子集
        subset = self.user_movie_matrix[:n_users, :n_movies]

        plt.figure(figsize=(14, 8))
        sns.heatmap(subset, cmap='YlOrRd', xticklabels=False, yticklabels=False)
        plt.title(f'User-Movie Rating Matrix (Top {n_users} users x {n_movies} movies)', fontsize=14)
        plt.xlabel('Movies')
        plt.ylabel('Users')
        plt.tight_layout()
        plt.savefig('/Users/chaowang/MECHINE_LEARNING_BUSINESS/MOVIE/user_movie_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("用户-电影矩阵可视化已保存: user_movie_matrix.png")

    def visualize_cf_similarity(self, n_movies=50):
        """
        可视化协同过滤物品相似度矩阵
        """
        print("\n生成协同过滤相似度可视化...")

        if self.item_similarity_cf is None:
            self.build_collaborative_filtering_model(method='item_based')

        # 取前n_movies部电影
        subset = self.item_similarity_cf[:n_movies, :n_movies]

        # 获取电影标题
        titles = [self.movies.iloc[i]['title'][:15] + '...' if len(self.movies.iloc[i]['title']) > 15
                  else self.movies.iloc[i]['title'] for i in range(n_movies)]

        plt.figure(figsize=(14, 12))
        sns.heatmap(subset, cmap='coolwarm', xticklabels=titles, yticklabels=titles)
        plt.title('Item-Based Collaborative Filtering Similarity Matrix', fontsize=14)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        plt.savefig('/Users/chaowang/MECHINE_LEARNING_BUSINESS/MOVIE/cf_similarity_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("协同过滤相似度可视化已保存: cf_similarity_matrix.png")

    def get_popular_movies(self, top_n=20):
        """获取热门电影列表"""
        return self.movies.nlargest(top_n, 'popularity')[['title', 'genres_list', 'vote_average', 'popularity']]

    def search_movies(self, query):
        """搜索电影"""
        matches = self.movies[self.movies['title'].str.contains(query, case=False, na=False)]
        return matches[['title', 'genres_list', 'vote_average', 'release_date']].head(10)

    # =============================================
    # 可解释性功能 (Explainability)
    # =============================================

    def _init_explainer(self):
        """初始化解释器"""
        from explainability import RecommendationExplainer
        self.explainer = RecommendationExplainer(
            self.movies,
            self.tfidf_vectorizer,
            self.tfidf_matrix
        )

    def explain_recommendation(self, source_title, target_title, method='metadata'):
        """
        解释为什么推荐某部电影

        Parameters:
        -----------
        source_title : str
            源电影（用户喜欢的电影）
        target_title : str
            推荐的电影
        method : str
            'content' - 基于内容的解释
            'metadata' - 基于元数据的解释
            'cf' - 协同过滤的解释
            'hybrid' - 混合推荐的解释

        Returns:
        --------
        dict : 解释信息
        """
        if self.explainer is None:
            self._init_explainer()

        if method == 'content':
            return self.explainer.explain_content_based(source_title, target_title)
        elif method == 'metadata':
            return self.explainer.explain_metadata_based(source_title, target_title)
        elif method == 'cf':
            return self.explainer.explain_collaborative_filtering(
                source_title, target_title,
                user_movie_matrix=self.user_movie_matrix
            )
        elif method == 'hybrid':
            source_idx = self.title_to_idx.get(source_title)
            target_idx = self.title_to_idx.get(target_title)
            if source_idx is not None and target_idx is not None:
                content_score = self.content_similarity[source_idx, target_idx] if self.content_similarity is not None else None
                metadata_score = self.metadata_similarity[source_idx, target_idx] if self.metadata_similarity is not None else None
                cf_score = self.item_similarity_cf[source_idx, target_idx] if self.item_similarity_cf is not None else None
                return self.explainer.explain_hybrid(
                    source_title, target_title,
                    content_score, metadata_score, cf_score
                )
        return None

    def recommend_with_explanation(self, title, top_n=10, method='hybrid', weights=(0.3, 0.4, 0.3)):
        """
        生成带解释的推荐

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            推荐数量
        method : str
            推荐方法 ('content', 'metadata', 'cf', 'hybrid')
        weights : tuple
            混合推荐权重

        Returns:
        --------
        list : 包含推荐和解释的列表
        """
        if self.explainer is None:
            self._init_explainer()

        # 确保模型已构建
        if self.content_similarity is None:
            self.build_content_based_model()
        if self.metadata_similarity is None:
            self.build_metadata_based_model()
        if self.item_similarity_cf is None:
            self.build_collaborative_filtering_model()

        # 获取推荐结果
        if method == 'content':
            recommendations = self.content_based_recommend(title, top_n)
        elif method == 'metadata':
            recommendations = self.metadata_based_recommend(title, top_n)
        elif method == 'cf':
            recommendations = self.collaborative_filtering_recommend(title, top_n)
        else:  # hybrid
            recommendations = self.hybrid_recommend(title, top_n, weights)

        if recommendations is None:
            return None

        # 为每个推荐生成解释
        results = []
        for _, row in recommendations.iterrows():
            target_title = row['title']

            # 获取元数据解释（最易理解）
            explanation = self.explainer.explain_metadata_based(title, target_title)

            # 获取各方法分数
            source_idx = self.title_to_idx[title]
            target_idx = self.title_to_idx[target_title]

            scores = {
                'content': round(self.content_similarity[source_idx, target_idx], 3),
                'metadata': round(self.metadata_similarity[source_idx, target_idx], 3),
                'cf': round(self.item_similarity_cf[source_idx, target_idx], 3)
            }

            result = {
                'rank': len(results) + 1,
                'title': target_title,
                'genres': row.get('genres_list', []),
                'vote_average': row.get('vote_average', 0),
                'similarity_score': row.get('similarity_score', row.get('hybrid_score', 0)),
                'explanation': explanation,
                'method_scores': scores,
                'why_recommended': self._generate_simple_explanation(explanation)
            }
            results.append(result)

        return results

    def _generate_simple_explanation(self, explanation):
        """生成简单的一句话解释"""
        reasons = explanation.get('reasons', [])
        if reasons:
            return reasons[0]
        return "综合特征相似"

    def print_explained_recommendations(self, title, top_n=5, method='hybrid'):
        """
        打印带解释的推荐结果

        Parameters:
        -----------
        title : str
            电影标题
        top_n : int
            推荐数量
        method : str
            推荐方法
        """
        results = self.recommend_with_explanation(title, top_n, method)

        if results is None:
            print(f"找不到电影: {title}")
            return

        print("\n" + "=" * 70)
        print(f"为《{title}》生成的可解释推荐 (方法: {method})")
        print("=" * 70)

        for result in results:
            print(f"\n#{result['rank']} 《{result['title']}》")
            print(f"    评分: {result['vote_average']}/10")
            print(f"    类型: {', '.join(result['genres'][:3])}")
            print(f"    相似度: {result['similarity_score']:.1%}")
            print(f"    ✨ 推荐理由: {result['why_recommended']}")

            # 显示详细理由
            if len(result['explanation']['reasons']) > 1:
                print(f"    📋 详细原因:")
                for reason in result['explanation']['reasons'][1:]:
                    print(f"       - {reason}")

            # 显示各方法分数
            scores = result['method_scores']
            print(f"    📊 方法分数: 内容={scores['content']:.1%}, 元数据={scores['metadata']:.1%}, CF={scores['cf']:.1%}")

        print("\n" + "=" * 70)

    def get_explanation_report(self, source_title, target_title):
        """
        获取详细的解释报告

        Parameters:
        -----------
        source_title : str
            源电影
        target_title : str
            目标电影

        Returns:
        --------
        str : 格式化的解释报告
        """
        if self.explainer is None:
            self._init_explainer()

        # 确保模型已构建
        if self.content_similarity is None:
            self.build_content_based_model()
        if self.metadata_similarity is None:
            self.build_metadata_based_model()
        if self.item_similarity_cf is None:
            self.build_collaborative_filtering_model()

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            return "电影未找到"

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"推荐解释报告")
        report_lines.append("=" * 70)
        report_lines.append(f"\n源电影: 《{source_title}》")
        report_lines.append(f"推荐电影: 《{target_title}》")

        # 源电影信息
        source_movie = self.movies.iloc[source_idx]
        target_movie = self.movies.iloc[target_idx]

        report_lines.append(f"\n【电影信息对比】")
        report_lines.append(f"{'属性':<15} {'源电影':<25} {'推荐电影':<25}")
        report_lines.append("-" * 65)
        report_lines.append(f"{'类型':<15} {str(source_movie['genres_list'][:3]):<25} {str(target_movie['genres_list'][:3]):<25}")
        report_lines.append(f"{'导演':<15} {str(source_movie['director'])[:23]:<25} {str(target_movie['director'])[:23]:<25}")
        report_lines.append(f"{'评分':<15} {source_movie['vote_average']:<25} {target_movie['vote_average']:<25}")

        # 各方法相似度
        report_lines.append(f"\n【相似度分析】")
        report_lines.append(f"  内容相似度 (TF-IDF): {self.content_similarity[source_idx, target_idx]:.1%}")
        report_lines.append(f"  元数据相似度: {self.metadata_similarity[source_idx, target_idx]:.1%}")
        report_lines.append(f"  协同过滤相似度: {self.item_similarity_cf[source_idx, target_idx]:.1%}")

        # 元数据详细匹配
        report_lines.append(f"\n【特征匹配详情】")

        # 导演
        if source_movie['director'] == target_movie['director'] and source_movie['director']:
            report_lines.append(f"  ✓ 导演匹配: {source_movie['director']}")
        else:
            report_lines.append(f"  ✗ 导演不同")

        # 类型
        common_genres = set(source_movie['genres_list']) & set(target_movie['genres_list'])
        if common_genres:
            report_lines.append(f"  ✓ 共同类型: {', '.join(common_genres)}")
        else:
            report_lines.append(f"  ✗ 无共同类型")

        # 演员
        common_cast = set(source_movie['cast_list']) & set(target_movie['cast_list'])
        if common_cast:
            report_lines.append(f"  ✓ 共同演员: {', '.join(list(common_cast)[:3])}")
        else:
            report_lines.append(f"  ✗ 无共同演员")

        # 关键词
        common_keywords = set(source_movie['keywords_list']) & set(target_movie['keywords_list'])
        if common_keywords:
            report_lines.append(f"  ✓ 共同关键词: {', '.join(list(common_keywords)[:3])}")
        else:
            report_lines.append(f"  ✗ 无共同关键词")

        # 综合结论
        report_lines.append(f"\n【推荐结论】")
        metadata_exp = self.explainer.explain_metadata_based(source_title, target_title)
        if metadata_exp['reasons']:
            report_lines.append(f"  主要推荐原因: {metadata_exp['reasons'][0]}")
        else:
            report_lines.append(f"  主要推荐原因: 综合特征相似")

        report_lines.append("=" * 70)

        return '\n'.join(report_lines)


def main():
    """主函数：演示推荐系统"""

    # 数据路径
    movies_path = '/Users/chaowang/MECHINE_LEARNING_BUSINESS/MOVIE/tmdb_5000_movies.csv'
    credits_path = '/Users/chaowang/MECHINE_LEARNING_BUSINESS/MOVIE/tmdb_5000_credits.csv'

    # 创建推荐系统
    recommender = MovieRecommenderSystem(movies_path, credits_path)

    # 构建所有模型
    print("\n" + "=" * 60)
    print("构建所有推荐模型")
    print("=" * 60)

    recommender.build_content_based_model()
    recommender.build_metadata_based_model()
    recommender.build_collaborative_filtering_model()

    # 测试电影列表
    test_movies = [
        'Avatar',
        'The Dark Knight',
        'Inception',
        'Interstellar',
        'The Avengers'
    ]

    print("\n" + "=" * 60)
    print("推荐系统演示")
    print("=" * 60)

    for movie in test_movies:
        if movie in recommender.title_to_idx:
            print(f"\n{'='*60}")
            print(f"为《{movie}》生成推荐...")
            print("="*60)

            # 比较三种方法
            recommender.compare_recommendations(movie, top_n=5)
            break  # 只演示第一部电影

    # 生成可视化
    recommender.visualize_user_movie_matrix()
    recommender.visualize_cf_similarity()

    # 演示为用户推荐
    print("\n" + "=" * 60)
    print("为用户推荐电影 (User ID: 0)")
    print("=" * 60)
    user_rec = recommender.recommend_for_user(user_id=0, top_n=10)
    if user_rec is not None:
        print(user_rec)

    # 交互式推荐
    print("\n" + "=" * 60)
    print("可用的热门电影:")
    print("=" * 60)
    popular = recommender.get_popular_movies(10)
    for i, row in popular.iterrows():
        print(f"  - {row['title']} (评分: {row['vote_average']}, 类型: {row['genres_list'][:3]})")

    print("\n推荐系统已准备就绪！")
    print("使用方法:")
    print("  recommender.content_based_recommend('电影名', top_n=10)")
    print("  recommender.metadata_based_recommend('电影名', top_n=10)")
    print("  recommender.collaborative_filtering_recommend('电影名', top_n=10, method='item_based')")
    print("  recommender.collaborative_filtering_recommend('电影名', top_n=10, method='user_based')")
    print("  recommender.collaborative_filtering_recommend('电影名', top_n=10, method='svd')")
    print("  recommender.recommend_for_user(user_id=0, top_n=10)")
    print("  recommender.hybrid_recommend('电影名', top_n=10)")
    print("  recommender.compare_recommendations('电影名', top_n=5)")

    return recommender


if __name__ == "__main__":
    recommender = main()
