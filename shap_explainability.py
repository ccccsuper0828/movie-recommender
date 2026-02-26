"""
基于SHAP的推荐系统可解释性框架
================================
使用SHAP (SHapley Additive exPlanations) 解释推荐结果

SHAP基于博弈论中的Shapley值，能够公平地分配每个特征对预测的贡献

作者: Movie Recommendation System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap 包未安装，请运行 'pip install shap' 安装")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split


class SHAPRecommenderExplainer:
    """
    基于SHAP的推荐系统解释器

    使用机器学习模型模拟推荐系统的相似度计算，
    然后使用SHAP解释每个特征对推荐分数的贡献
    """

    def __init__(self, movies_df, similarity_matrix=None, matrix_type='metadata'):
        """
        初始化SHAP解释器

        Parameters:
        -----------
        movies_df : DataFrame
            电影数据
        similarity_matrix : ndarray
            相似度矩阵（可选）
        matrix_type : str
            相似度矩阵类型 ('content', 'metadata', 'cf')
        """
        self.movies = movies_df.copy()
        self.similarity_matrix = similarity_matrix
        self.matrix_type = matrix_type
        self.title_to_idx = pd.Series(movies_df.index, index=movies_df['title']).to_dict()

        # 模型相关
        self.feature_matrix = None
        self.feature_names = None
        self.model = None
        self.explainer = None

        # 编码器
        self.genre_mlb = MultiLabelBinarizer()
        self.director_encoder = LabelEncoder()

    def _prepare_features(self):
        """
        准备特征矩阵用于SHAP分析

        将电影元数据转换为数值特征
        """
        print("准备特征矩阵...")

        # 1. 处理类型（多标签编码）
        genres_encoded = self.genre_mlb.fit_transform(self.movies['genres_list'])
        genre_feature_names = [f"genre_{g}" for g in self.genre_mlb.classes_]

        # 2. 处理导演（Label编码）
        directors = self.movies['director'].fillna('Unknown')
        self.director_encoder.fit(directors)
        director_encoded = self.director_encoder.transform(directors).reshape(-1, 1)

        # 3. 处理数值特征
        numerical_features = self.movies[['vote_average', 'popularity']].fillna(0).values

        # 4. 处理演员数量
        cast_count = self.movies['cast_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).values.reshape(-1, 1)

        # 5. 处理关键词数量
        keyword_count = self.movies['keywords_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).values.reshape(-1, 1)

        # 组合所有特征
        self.feature_matrix = np.hstack([
            genres_encoded,
            director_encoded,
            numerical_features,
            cast_count,
            keyword_count
        ])

        self.feature_names = (
            genre_feature_names +
            ['director_id'] +
            ['vote_average', 'popularity'] +
            ['cast_count', 'keyword_count']
        )

        print(f"特征矩阵形状: {self.feature_matrix.shape}")
        print(f"特征数量: {len(self.feature_names)}")

        return self.feature_matrix

    def _create_training_data(self, n_samples=50000):
        """
        创建训练数据用于模拟相似度计算

        生成电影对及其相似度分数
        """
        print("生成训练数据...")

        if self.similarity_matrix is None:
            raise ValueError("需要提供相似度矩阵")

        n_movies = len(self.movies)

        # 随机采样电影对
        np.random.seed(42)
        indices1 = np.random.randint(0, n_movies, n_samples)
        indices2 = np.random.randint(0, n_movies, n_samples)

        # 获取相似度分数
        similarities = self.similarity_matrix[indices1, indices2]

        # 创建特征差异矩阵
        # 使用绝对差值和乘积来表示两部电影的关系
        feature_diffs = []
        feature_products = []

        for i1, i2 in zip(indices1, indices2):
            f1 = self.feature_matrix[i1]
            f2 = self.feature_matrix[i2]
            feature_diffs.append(np.abs(f1 - f2))
            feature_products.append(f1 * f2)

        X_diff = np.array(feature_diffs)
        X_prod = np.array(feature_products)

        # 组合特征
        X = np.hstack([X_diff, X_prod])
        y = similarities

        # 特征名称
        diff_names = [f"{n}_diff" for n in self.feature_names]
        prod_names = [f"{n}_match" for n in self.feature_names]
        combined_feature_names = diff_names + prod_names

        print(f"训练数据形状: X={X.shape}, y={y.shape}")

        return X, y, combined_feature_names

    def train_explainer_model(self, n_samples=50000):
        """
        训练用于解释的替代模型

        使用梯度提升回归模拟相似度计算
        """
        if self.feature_matrix is None:
            self._prepare_features()

        X, y, feature_names = self._create_training_data(n_samples)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("\n训练解释器模型...")

        # 使用梯度提升回归（比随机森林更适合SHAP）
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )

        self.model.fit(X_train, y_train)

        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"训练集 R²: {train_score:.4f}")
        print(f"测试集 R²: {test_score:.4f}")

        # 创建SHAP解释器
        if SHAP_AVAILABLE:
            print("\n创建SHAP解释器...")
            # 使用采样的背景数据
            background = shap.sample(X_train, 100)
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_feature_names = feature_names

        return self.model

    def explain_pair(self, source_title, target_title, plot=True):
        """
        解释两部电影之间的相似度

        Parameters:
        -----------
        source_title : str
            源电影
        target_title : str
            目标电影
        plot : bool
            是否绘制SHAP图

        Returns:
        --------
        dict : SHAP解释结果
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP包未安装"}

        if self.model is None:
            self.train_explainer_model()

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            return {"error": "电影未找到"}

        # 获取特征
        f1 = self.feature_matrix[source_idx]
        f2 = self.feature_matrix[target_idx]

        # 计算特征差异和乘积
        feature_diff = np.abs(f1 - f2)
        feature_prod = f1 * f2

        # 组合特征
        X = np.hstack([feature_diff, feature_prod]).reshape(1, -1)

        # 计算SHAP值
        shap_values = self.explainer.shap_values(X)

        # 获取基准值
        base_value = self.explainer.expected_value

        # 预测值
        predicted_similarity = self.model.predict(X)[0]

        # 实际相似度
        actual_similarity = self.similarity_matrix[source_idx, target_idx]

        # 解析SHAP值
        shap_df = pd.DataFrame({
            'feature': self.shap_feature_names,
            'shap_value': shap_values[0],
            'feature_value': X[0]
        })

        # 按绝对SHAP值排序
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)

        # 提取最重要的特征
        top_positive = shap_df[shap_df['shap_value'] > 0].head(5)
        top_negative = shap_df[shap_df['shap_value'] < 0].head(5)

        result = {
            'source_movie': source_title,
            'target_movie': target_title,
            'predicted_similarity': round(predicted_similarity, 4),
            'actual_similarity': round(actual_similarity, 4),
            'base_value': round(base_value, 4),
            'top_positive_features': top_positive[['feature', 'shap_value']].to_dict('records'),
            'top_negative_features': top_negative[['feature', 'shap_value']].to_dict('records'),
            'all_shap_values': shap_df.to_dict('records')
        }

        # 绘制SHAP图
        if plot:
            self._plot_shap_explanation(X, shap_values, source_title, target_title)

        return result

    def _plot_shap_explanation(self, X, shap_values, source_title, target_title):
        """绘制SHAP解释图"""
        if not SHAP_AVAILABLE:
            return

        plt.figure(figsize=(12, 8))

        # 创建解释对象
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value,
            data=X[0],
            feature_names=self.shap_feature_names
        )

        # 绘制waterfall图
        shap.waterfall_plot(explanation, max_display=15, show=False)

        plt.title(f'SHAP解释: {source_title} → {target_title}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_explanation.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("SHAP解释图已保存: shap_explanation.png")

    def explain_recommendations(self, source_title, recommendations_df, top_n=5):
        """
        解释一组推荐结果

        Parameters:
        -----------
        source_title : str
            源电影
        recommendations_df : DataFrame
            推荐结果
        top_n : int
            解释前n个推荐

        Returns:
        --------
        list : 解释列表
        """
        explanations = []

        for i, (_, row) in enumerate(recommendations_df.head(top_n).iterrows()):
            target_title = row['title']
            explanation = self.explain_pair(source_title, target_title, plot=False)

            if 'error' not in explanation:
                # 简化解释
                simplified = {
                    'rank': i + 1,
                    'title': target_title,
                    'similarity': explanation['actual_similarity'],
                    'positive_factors': [],
                    'negative_factors': []
                }

                # 提取人类可读的解释
                for feat in explanation['top_positive_features'][:3]:
                    feature_name = feat['feature']
                    shap_val = feat['shap_value']
                    simplified['positive_factors'].append(
                        self._interpret_feature(feature_name, shap_val, 'positive')
                    )

                for feat in explanation['top_negative_features'][:2]:
                    feature_name = feat['feature']
                    shap_val = feat['shap_value']
                    simplified['negative_factors'].append(
                        self._interpret_feature(feature_name, shap_val, 'negative')
                    )

                explanations.append(simplified)

        return explanations

    def _interpret_feature(self, feature_name, shap_value, direction):
        """将SHAP特征解释转换为人类可读的文本"""
        abs_val = abs(shap_value)

        if '_match' in feature_name:
            # 匹配特征
            base_name = feature_name.replace('_match', '')

            if 'genre_' in base_name:
                genre = base_name.replace('genre_', '')
                if direction == 'positive':
                    return f"两部电影都属于 {genre} 类型 (+{abs_val:.3f})"
                else:
                    return f"类型 {genre} 不匹配 (-{abs_val:.3f})"

            elif base_name == 'director_id':
                if direction == 'positive':
                    return f"同一导演的作品 (+{abs_val:.3f})"
                else:
                    return f"不同导演 (-{abs_val:.3f})"

            elif base_name == 'vote_average':
                if direction == 'positive':
                    return f"评分水平相近 (+{abs_val:.3f})"
                else:
                    return f"评分差异较大 (-{abs_val:.3f})"

            elif base_name == 'popularity':
                if direction == 'positive':
                    return f"热度水平相近 (+{abs_val:.3f})"
                else:
                    return f"热度差异较大 (-{abs_val:.3f})"

        elif '_diff' in feature_name:
            # 差异特征
            base_name = feature_name.replace('_diff', '')

            if 'genre_' in base_name:
                genre = base_name.replace('genre_', '')
                if direction == 'positive':
                    return f"类型 {genre} 差异小 (+{abs_val:.3f})"
                else:
                    return f"类型 {genre} 差异大 (-{abs_val:.3f})"

        return f"{feature_name}: {'正向' if direction == 'positive' else '负向'}影响 ({shap_value:.3f})"

    def plot_feature_importance(self, save_path='shap_feature_importance.png'):
        """
        绘制全局特征重要性图

        Parameters:
        -----------
        save_path : str
            保存路径
        """
        if not SHAP_AVAILABLE or self.model is None:
            print("需要先训练模型")
            return

        print("计算全局SHAP特征重要性...")

        # 采样数据
        X, _, _ = self._create_training_data(n_samples=1000)
        X_sample = shap.sample(X, 100)

        # 计算SHAP值
        shap_values = self.explainer.shap_values(X_sample)

        # 绘制汇总图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.shap_feature_names,
            show=False,
            max_display=20
        )
        plt.title('SHAP特征重要性汇总', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"特征重要性图已保存: {save_path}")

    def generate_explanation_report(self, source_title, target_title):
        """
        生成详细的SHAP解释报告

        Parameters:
        -----------
        source_title : str
            源电影
        target_title : str
            目标电影

        Returns:
        --------
        str : 格式化的报告
        """
        if not SHAP_AVAILABLE:
            return "SHAP包未安装，请运行 'pip install shap' 安装"

        if self.model is None:
            self.train_explainer_model()

        explanation = self.explain_pair(source_title, target_title, plot=True)

        if 'error' in explanation:
            return f"错误: {explanation['error']}"

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("SHAP 可解释性分析报告")
        report_lines.append("=" * 70)

        report_lines.append(f"\n源电影: 《{source_title}》")
        report_lines.append(f"推荐电影: 《{target_title}》")

        report_lines.append(f"\n【相似度分析】")
        report_lines.append(f"  实际相似度: {explanation['actual_similarity']:.1%}")
        report_lines.append(f"  模型预测相似度: {explanation['predicted_similarity']:.1%}")
        report_lines.append(f"  SHAP基准值: {explanation['base_value']:.4f}")

        report_lines.append(f"\n【正向影响因素】(提高相似度)")
        report_lines.append(f"{'特征':<40} {'SHAP贡献':<15}")
        report_lines.append("-" * 55)

        for feat in explanation['top_positive_features']:
            readable = self._interpret_feature(feat['feature'], feat['shap_value'], 'positive')
            report_lines.append(f"  {readable}")

        report_lines.append(f"\n【负向影响因素】(降低相似度)")
        report_lines.append(f"{'特征':<40} {'SHAP贡献':<15}")
        report_lines.append("-" * 55)

        for feat in explanation['top_negative_features']:
            readable = self._interpret_feature(feat['feature'], feat['shap_value'], 'negative')
            report_lines.append(f"  {readable}")

        report_lines.append(f"\n【解释说明】")
        report_lines.append(f"  SHAP值表示每个特征对相似度预测的贡献")
        report_lines.append(f"  正值表示该特征增加相似度，负值表示降低相似度")
        report_lines.append(f"  特征名称中 '_match' 表示两部电影在该特征上的匹配程度")
        report_lines.append(f"  特征名称中 '_diff' 表示两部电影在该特征上的差异程度")

        report_lines.append("\n" + "=" * 70)

        return '\n'.join(report_lines)


class FeatureContributionAnalyzer:
    """
    特征贡献分析器

    不依赖SHAP，使用直接的特征分解来解释推荐
    """

    def __init__(self, movies_df):
        """
        初始化分析器

        Parameters:
        -----------
        movies_df : DataFrame
            电影数据
        """
        self.movies = movies_df
        self.title_to_idx = pd.Series(movies_df.index, index=movies_df['title']).to_dict()

        # 特征权重（基于元数据推荐的设计）
        self.feature_weights = {
            'genres': 0.30,      # 类型权重30%
            'director': 0.30,   # 导演权重30%
            'cast': 0.20,       # 演员权重20%
            'keywords': 0.20    # 关键词权重20%
        }

    def analyze_contribution(self, source_title, target_title):
        """
        分析特征对推荐分数的贡献

        Parameters:
        -----------
        source_title : str
            源电影
        target_title : str
            目标电影

        Returns:
        --------
        dict : 特征贡献分析结果
        """
        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            return {"error": "电影未找到"}

        source = self.movies.iloc[source_idx]
        target = self.movies.iloc[target_idx]

        contributions = {}
        total_score = 0

        # 1. 类型贡献
        source_genres = set(source.get('genres_list', []))
        target_genres = set(target.get('genres_list', []))

        if source_genres and target_genres:
            genre_overlap = len(source_genres & target_genres) / len(source_genres | target_genres)
            genre_contribution = genre_overlap * self.feature_weights['genres']
        else:
            genre_contribution = 0

        contributions['genres'] = {
            'weight': self.feature_weights['genres'],
            'overlap': genre_overlap if source_genres and target_genres else 0,
            'contribution': round(genre_contribution, 4),
            'common': list(source_genres & target_genres),
            'source_only': list(source_genres - target_genres),
            'target_only': list(target_genres - source_genres)
        }
        total_score += genre_contribution

        # 2. 导演贡献
        source_director = source.get('director', '')
        target_director = target.get('director', '')

        if source_director and target_director:
            director_match = 1.0 if source_director == target_director else 0.0
            director_contribution = director_match * self.feature_weights['director']
        else:
            director_contribution = 0

        contributions['director'] = {
            'weight': self.feature_weights['director'],
            'match': director_match if source_director and target_director else 0,
            'contribution': round(director_contribution, 4),
            'source_director': source_director,
            'target_director': target_director,
            'is_same': source_director == target_director and source_director != ''
        }
        total_score += director_contribution

        # 3. 演员贡献
        source_cast = set(source.get('cast_list', []))
        target_cast = set(target.get('cast_list', []))

        if source_cast and target_cast:
            cast_overlap = len(source_cast & target_cast) / max(len(source_cast), len(target_cast))
            cast_contribution = cast_overlap * self.feature_weights['cast']
        else:
            cast_contribution = 0

        contributions['cast'] = {
            'weight': self.feature_weights['cast'],
            'overlap': cast_overlap if source_cast and target_cast else 0,
            'contribution': round(cast_contribution, 4),
            'common': list(source_cast & target_cast)
        }
        total_score += cast_contribution

        # 4. 关键词贡献
        source_keywords = set(source.get('keywords_list', []))
        target_keywords = set(target.get('keywords_list', []))

        if source_keywords and target_keywords:
            keyword_overlap = len(source_keywords & target_keywords) / max(len(source_keywords), len(target_keywords))
            keyword_contribution = keyword_overlap * self.feature_weights['keywords']
        else:
            keyword_contribution = 0

        contributions['keywords'] = {
            'weight': self.feature_weights['keywords'],
            'overlap': keyword_overlap if source_keywords and target_keywords else 0,
            'contribution': round(keyword_contribution, 4),
            'common': list(source_keywords & target_keywords)
        }
        total_score += keyword_contribution

        # 汇总
        result = {
            'source_movie': source_title,
            'target_movie': target_title,
            'total_similarity_score': round(total_score, 4),
            'feature_contributions': contributions,
            'contribution_breakdown': {
                '类型': round(genre_contribution / max(total_score, 0.001) * 100, 1),
                '导演': round(director_contribution / max(total_score, 0.001) * 100, 1),
                '演员': round(cast_contribution / max(total_score, 0.001) * 100, 1),
                '关键词': round(keyword_contribution / max(total_score, 0.001) * 100, 1)
            }
        }

        return result

    def generate_contribution_report(self, source_title, target_title):
        """
        生成特征贡献报告

        Parameters:
        -----------
        source_title : str
            源电影
        target_title : str
            目标电影

        Returns:
        --------
        str : 格式化的报告
        """
        analysis = self.analyze_contribution(source_title, target_title)

        if 'error' in analysis:
            return f"错误: {analysis['error']}"

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("特征贡献分析报告")
        report_lines.append("=" * 70)

        report_lines.append(f"\n源电影: 《{source_title}》")
        report_lines.append(f"推荐电影: 《{target_title}》")
        report_lines.append(f"综合相似度得分: {analysis['total_similarity_score']:.1%}")

        report_lines.append(f"\n【特征贡献分解】")
        report_lines.append(f"{'特征':<15} {'权重':<10} {'匹配度':<10} {'贡献':<10} {'占比':<10}")
        report_lines.append("-" * 55)

        for feature, data in analysis['feature_contributions'].items():
            feature_name = {'genres': '类型', 'director': '导演', 'cast': '演员', 'keywords': '关键词'}[feature]
            match_rate = data.get('overlap', data.get('match', 0))
            contribution_pct = analysis['contribution_breakdown'][feature_name]
            report_lines.append(
                f"{feature_name:<15} {data['weight']:.0%}{'':<6} {match_rate:.1%}{'':<6} "
                f"{data['contribution']:.4f}{'':<4} {contribution_pct:.1f}%"
            )

        report_lines.append(f"\n【详细匹配信息】")

        # 类型
        genres_data = analysis['feature_contributions']['genres']
        if genres_data['common']:
            report_lines.append(f"  共同类型: {', '.join(genres_data['common'])}")
        if genres_data['source_only']:
            report_lines.append(f"  源电影独有: {', '.join(genres_data['source_only'])}")

        # 导演
        director_data = analysis['feature_contributions']['director']
        if director_data['is_same']:
            report_lines.append(f"  同一导演: {director_data['source_director']}")
        else:
            report_lines.append(f"  不同导演: {director_data['source_director']} vs {director_data['target_director']}")

        # 演员
        cast_data = analysis['feature_contributions']['cast']
        if cast_data['common']:
            report_lines.append(f"  共同演员: {', '.join(cast_data['common'][:3])}")

        # 关键词
        keyword_data = analysis['feature_contributions']['keywords']
        if keyword_data['common']:
            report_lines.append(f"  共同主题: {', '.join(keyword_data['common'][:3])}")

        # 可视化贡献
        report_lines.append(f"\n【贡献可视化】")
        for feature_name, pct in analysis['contribution_breakdown'].items():
            bar_length = int(pct / 5)  # 每5%一个方块
            bar = "█" * bar_length + "░" * (20 - bar_length)
            report_lines.append(f"  {feature_name:<6} [{bar}] {pct:.1f}%")

        report_lines.append("\n" + "=" * 70)

        return '\n'.join(report_lines)

    def plot_contribution(self, source_title, target_title, save_path='feature_contribution.png'):
        """
        绘制特征贡献图

        Parameters:
        -----------
        source_title : str
            源电影
        target_title : str
            目标电影
        save_path : str
            保存路径
        """
        analysis = self.analyze_contribution(source_title, target_title)

        if 'error' in analysis:
            print(f"错误: {analysis['error']}")
            return

        # 准备数据
        features = ['类型', '导演', '演员', '关键词']
        contributions = [
            analysis['feature_contributions']['genres']['contribution'],
            analysis['feature_contributions']['director']['contribution'],
            analysis['feature_contributions']['cast']['contribution'],
            analysis['feature_contributions']['keywords']['contribution']
        ]
        weights = [0.30, 0.30, 0.20, 0.20]

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1: 贡献条形图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = axes[0].barh(features, contributions, color=colors)
        axes[0].set_xlabel('贡献值')
        axes[0].set_title(f'特征贡献分析\n{source_title} → {target_title}', fontweight='bold')
        axes[0].set_xlim(0, max(contributions) * 1.2 if contributions else 0.5)

        # 添加数值标签
        for bar, contrib in zip(bars, contributions):
            axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{contrib:.3f}', va='center')

        # 图2: 贡献占比饼图
        if sum(contributions) > 0:
            axes[1].pie(contributions, labels=features, autopct='%1.1f%%',
                       colors=colors, startangle=90)
            axes[1].set_title('贡献占比', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, '无匹配特征', ha='center', va='center', fontsize=14)
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"特征贡献图已保存: {save_path}")


def demo():
    """演示SHAP可解释性功能"""
    print("=" * 70)
    print("SHAP可解释性框架演示")
    print("=" * 70)

    if not SHAP_AVAILABLE:
        print("\n注意: SHAP包未安装")
        print("请运行: pip install shap")
        print("\n将使用不依赖SHAP的特征贡献分析器作为替代")

    print("\n使用示例:")
    print("""
    from movie_recommender import MovieRecommenderSystem
    from shap_explainability import SHAPRecommenderExplainer, FeatureContributionAnalyzer

    # 初始化推荐系统
    recommender = MovieRecommenderSystem('tmdb_5000_movies.csv', 'tmdb_5000_credits.csv')
    recommender.build_metadata_based_model()

    # 方法1: 使用SHAP解释器 (需要安装shap包)
    shap_explainer = SHAPRecommenderExplainer(
        recommender.movies,
        recommender.metadata_similarity,
        'metadata'
    )
    shap_explainer.train_explainer_model()
    report = shap_explainer.generate_explanation_report('Avatar', 'Titanic')
    print(report)

    # 方法2: 使用特征贡献分析器 (无需额外依赖)
    analyzer = FeatureContributionAnalyzer(recommender.movies)
    report = analyzer.generate_contribution_report('Avatar', 'Titanic')
    print(report)
    analyzer.plot_contribution('Avatar', 'Titanic')
    """)


if __name__ == "__main__":
    demo()
