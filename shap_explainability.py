"""
基于SHAP的推荐系统可解释性框架
================================
使用SHAP (SHapley Additive exPlanations) 解释推荐结果

参考: https://shap.readthedocs.io/en/latest/
SHAP基于博弈论中的Shapley值，能够公平地分配每个特征对预测的贡献

主要可视化方法:
- summary_plot (beeswarm): 全局特征重要性
- force_plot: 单个预测的特征贡献力场图
- waterfall_plot: 单个预测的瀑布图
- dependence_plot: 特征依赖关系图

作者: Movie Recommendation System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import warnings
warnings.filterwarnings('ignore')

# 检查SHAP是否可用
try:
    import shap
    shap.initjs()  # 初始化JavaScript可视化
    SHAP_AVAILABLE = True
    print(f"SHAP版本: {shap.__version__}")
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap 包未安装，请运行 'pip install shap' 安装")
except Exception as e:
    SHAP_AVAILABLE = True  # shap已安装但initjs可能失败（非Jupyter环境）

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MovieSHAPExplainer:
    """
    电影推荐系统SHAP解释器

    使用SHAP标准API解释推荐结果，提供多种可视化方法
    """

    def __init__(self, movies_df, similarity_matrix=None):
        """
        初始化SHAP解释器

        Parameters:
        -----------
        movies_df : DataFrame
            电影数据，需要包含以下列:
            - title: 电影标题
            - genres_list: 类型列表
            - director: 导演
            - cast_list: 演员列表
            - keywords_list: 关键词列表
            - vote_average: 评分
            - popularity: 热度
        similarity_matrix : ndarray, optional
            预计算的相似度矩阵
        """
        if not SHAP_AVAILABLE:
            raise ImportError("请先安装SHAP: pip install shap")

        self.movies = movies_df.reset_index(drop=True)
        self.similarity_matrix = similarity_matrix
        self.title_to_idx = pd.Series(self.movies.index, index=self.movies['title']).to_dict()

        # 模型和解释器
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_train = None
        self.feature_names = None
        self.feature_matrix = None

        # 编码器
        self.genre_encoder = MultiLabelBinarizer()
        self.scaler = StandardScaler()

        # 输出目录
        self.output_dir = os.path.dirname(os.path.abspath(__file__))

    def prepare_feature_matrix(self):
        """
        准备特征矩阵

        将电影的各种属性转换为数值特征矩阵
        """
        print("=" * 60)
        print("准备特征矩阵...")
        print("=" * 60)

        features = []
        feature_names = []

        # 1. 类型特征 (One-Hot编码)
        print("处理类型特征...")
        genres = self.movies['genres_list'].apply(lambda x: x if isinstance(x, list) else [])
        genre_encoded = self.genre_encoder.fit_transform(genres)
        features.append(genre_encoded)
        feature_names.extend([f"类型_{g}" for g in self.genre_encoder.classes_])
        print(f"  类型特征数: {len(self.genre_encoder.classes_)}")

        # 2. 数值特征
        print("处理数值特征...")
        vote_avg = self.movies['vote_average'].fillna(0).values.reshape(-1, 1)
        popularity = self.movies['popularity'].fillna(0).values.reshape(-1, 1)
        features.append(vote_avg)
        features.append(popularity)
        feature_names.extend(['评分', '热度'])

        # 3. 演员数量
        cast_count = self.movies['cast_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).values.reshape(-1, 1)
        features.append(cast_count)
        feature_names.append('演员数量')

        # 4. 关键词数量
        keyword_count = self.movies['keywords_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).values.reshape(-1, 1)
        features.append(keyword_count)
        feature_names.append('关键词数量')

        # 5. 是否有导演信息
        has_director = self.movies['director'].apply(
            lambda x: 1 if x and str(x) != '' else 0
        ).values.reshape(-1, 1)
        features.append(has_director)
        feature_names.append('有导演信息')

        # 合并所有特征
        self.feature_matrix = np.hstack(features)
        self.feature_names = feature_names

        print(f"\n特征矩阵形状: {self.feature_matrix.shape}")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"样本数量: {len(self.movies)}")

        return self.feature_matrix, self.feature_names

    def create_pair_features(self, idx1, idx2):
        """
        创建电影对的特征

        Parameters:
        -----------
        idx1, idx2 : int
            两部电影的索引

        Returns:
        --------
        ndarray : 电影对的组合特征
        """
        f1 = self.feature_matrix[idx1]
        f2 = self.feature_matrix[idx2]

        # 特征组合方式:
        # 1. 绝对差值 (差异程度)
        diff = np.abs(f1 - f2)
        # 2. 元素乘积 (匹配程度，对于one-hot特征)
        product = f1 * f2
        # 3. 最小值 (共同拥有)
        minimum = np.minimum(f1, f2)

        return np.concatenate([diff, product, minimum])

    def prepare_training_data(self, n_samples=30000):
        """
        准备训练数据

        生成电影对及其相似度分数用于训练替代模型

        Parameters:
        -----------
        n_samples : int
            采样的电影对数量
        """
        print("\n" + "=" * 60)
        print("准备训练数据...")
        print("=" * 60)

        if self.feature_matrix is None:
            self.prepare_feature_matrix()

        if self.similarity_matrix is None:
            raise ValueError("需要提供相似度矩阵")

        n_movies = len(self.movies)

        # 随机采样电影对
        np.random.seed(42)
        idx1 = np.random.randint(0, n_movies, n_samples)
        idx2 = np.random.randint(0, n_movies, n_samples)

        # 确保不是同一部电影
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]

        print(f"采样电影对数量: {len(idx1)}")

        # 创建特征和标签
        X = []
        y = []

        for i1, i2 in zip(idx1, idx2):
            pair_features = self.create_pair_features(i1, i2)
            X.append(pair_features)
            y.append(self.similarity_matrix[i1, i2])

        X = np.array(X)
        y = np.array(y)

        # 创建特征名称
        pair_feature_names = (
            [f"{n}_差异" for n in self.feature_names] +
            [f"{n}_匹配" for n in self.feature_names] +
            [f"{n}_共有" for n in self.feature_names]
        )

        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        print(f"相似度范围: [{y.min():.4f}, {y.max():.4f}]")

        return X, y, pair_feature_names

    def train_model(self, n_samples=30000):
        """
        训练替代模型

        使用梯度提升树模拟相似度计算，便于SHAP解释

        Parameters:
        -----------
        n_samples : int
            训练样本数量
        """
        print("\n" + "=" * 60)
        print("训练SHAP解释模型...")
        print("=" * 60)

        X, y, feature_names = self.prepare_training_data(n_samples)
        self.pair_feature_names = feature_names

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.X_train = X_train
        self.X_test = X_test

        # 训练梯度提升模型
        print("\n训练梯度提升回归模型...")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        self.model.fit(X_train, y_train)

        # 评估
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"\n模型性能:")
        print(f"  训练集 R²: {train_score:.4f}")
        print(f"  测试集 R²: {test_score:.4f}")

        # 创建SHAP解释器
        print("\n创建SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

        print("模型训练完成!")

        return self.model

    def compute_shap_values(self, X=None, max_samples=500):
        """
        计算SHAP值

        Parameters:
        -----------
        X : ndarray, optional
            要解释的样本，默认使用测试集
        max_samples : int
            最大样本数量
        """
        if self.explainer is None:
            self.train_model()

        if X is None:
            X = self.X_test[:max_samples]

        print(f"\n计算SHAP值 (样本数: {len(X)})...")
        self.shap_values = self.explainer.shap_values(X)
        self.shap_X = X

        return self.shap_values

    def explain_pair(self, source_title, target_title, save_plots=True):
        """
        解释两部电影的相似度

        Parameters:
        -----------
        source_title : str
            源电影标题
        target_title : str
            目标电影标题
        save_plots : bool
            是否保存可视化图

        Returns:
        --------
        dict : 解释结果
        """
        print("\n" + "=" * 60)
        print(f"SHAP解释: {source_title} → {target_title}")
        print("=" * 60)

        if self.model is None:
            self.train_model()

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None:
            print(f"错误: 找不到电影 '{source_title}'")
            return None
        if target_idx is None:
            print(f"错误: 找不到电影 '{target_title}'")
            return None

        # 创建电影对特征
        pair_features = self.create_pair_features(source_idx, target_idx)
        X = pair_features.reshape(1, -1)

        # 预测相似度
        predicted_sim = self.model.predict(X)[0]
        actual_sim = self.similarity_matrix[source_idx, target_idx]

        # 计算SHAP值
        shap_values = self.explainer.shap_values(X)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = float(expected_value[0]) if len(expected_value) > 0 else float(expected_value)
        else:
            expected_value = float(expected_value)

        print(f"\n实际相似度: {actual_sim:.4f}")
        print(f"预测相似度: {predicted_sim:.4f}")
        print(f"基准值 (平均预测): {expected_value:.4f}")

        # 分析SHAP值
        shap_df = pd.DataFrame({
            'feature': self.pair_feature_names,
            'shap_value': shap_values[0],
            'feature_value': X[0]
        })
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)

        # 正向和负向贡献
        positive_contrib = shap_df[shap_df['shap_value'] > 0.001].head(10)
        negative_contrib = shap_df[shap_df['shap_value'] < -0.001].head(10)

        print(f"\n【正向贡献因素】(增加相似度)")
        for _, row in positive_contrib.iterrows():
            print(f"  {row['feature']}: +{row['shap_value']:.4f}")

        print(f"\n【负向贡献因素】(降低相似度)")
        for _, row in negative_contrib.iterrows():
            print(f"  {row['feature']}: {row['shap_value']:.4f}")

        # 保存可视化
        if save_plots:
            self._plot_waterfall(X, shap_values, source_title, target_title)
            self._plot_force(X, shap_values, expected_value, source_title, target_title)

        return {
            'source_movie': source_title,
            'target_movie': target_title,
            'actual_similarity': float(actual_sim),
            'predicted_similarity': float(predicted_sim),
            'base_value': float(expected_value),
            'shap_values': shap_df.to_dict('records'),
            'top_positive': positive_contrib[['feature', 'shap_value']].to_dict('records'),
            'top_negative': negative_contrib[['feature', 'shap_value']].to_dict('records')
        }

    def _plot_waterfall(self, X, shap_values, source_title, target_title):
        """绘制瀑布图"""
        print("\n生成瀑布图...")

        # 获取base_values
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0]) if len(base_value) > 0 else float(base_value)

        # 创建SHAP Explanation对象
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X[0],
            feature_names=self.pair_feature_names
        )

        plt.figure(figsize=(12, 10))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f'SHAP Waterfall Plot\n{source_title} → {target_title}', fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'shap_waterfall.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"瀑布图已保存: {save_path}")

    def _plot_force(self, X, shap_values, expected_value, source_title, target_title):
        """绘制力场图"""
        print("生成力场图...")

        # 使用matplotlib绘制简化版力场图
        plt.figure(figsize=(14, 4))

        # 获取top特征
        indices = np.argsort(np.abs(shap_values[0]))[::-1][:10]
        top_shap = shap_values[0][indices]
        top_names = [self.pair_feature_names[i][:15] for i in indices]

        colors = ['#ff0051' if v > 0 else '#008bfb' for v in top_shap]

        plt.barh(range(len(top_shap)), top_shap, color=colors)
        plt.yticks(range(len(top_shap)), top_names)
        plt.xlabel('SHAP Value (对相似度的影响)')
        plt.title(f'SHAP Force Plot: {source_title} → {target_title}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff0051', label='增加相似度'),
            Patch(facecolor='#008bfb', label='降低相似度')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'shap_force.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"力场图已保存: {save_path}")

    def plot_summary(self, max_samples=500, max_display=20):
        """
        绘制SHAP Summary Plot (Beeswarm图)

        显示所有特征的全局重要性和分布

        Parameters:
        -----------
        max_samples : int
            用于计算的最大样本数
        max_display : int
            显示的最大特征数
        """
        print("\n" + "=" * 60)
        print("生成SHAP Summary Plot...")
        print("=" * 60)

        if self.shap_values is None:
            self.compute_shap_values(max_samples=max_samples)

        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            self.shap_X,
            feature_names=self.pair_feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot - Feature Importance', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'shap_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Summary Plot已保存: {save_path}")

        # Bar plot (特征重要性条形图)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.shap_X,
            feature_names=self.pair_feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance (Bar Plot)', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'shap_importance_bar.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"特征重要性条形图已保存: {save_path}")

    def plot_dependence(self, feature_name, interaction_feature=None):
        """
        绘制SHAP Dependence Plot

        显示单个特征与SHAP值的关系

        Parameters:
        -----------
        feature_name : str
            要分析的特征名称
        interaction_feature : str, optional
            交互特征名称
        """
        if self.shap_values is None:
            self.compute_shap_values()

        if feature_name not in self.pair_feature_names:
            print(f"错误: 特征 '{feature_name}' 不存在")
            print(f"可用特征: {self.pair_feature_names[:10]}...")
            return

        feature_idx = self.pair_feature_names.index(feature_name)

        plt.figure(figsize=(10, 6))

        if interaction_feature and interaction_feature in self.pair_feature_names:
            interaction_idx = self.pair_feature_names.index(interaction_feature)
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.shap_X,
                feature_names=self.pair_feature_names,
                interaction_index=interaction_idx,
                show=False
            )
        else:
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.shap_X,
                feature_names=self.pair_feature_names,
                show=False
            )

        plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f'shap_dependence_{feature_name[:20]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Dependence Plot已保存: {save_path}")

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
            解释前n个

        Returns:
        --------
        list : 解释列表
        """
        print("\n" + "=" * 60)
        print(f"解释推荐结果: {source_title}")
        print("=" * 60)

        explanations = []

        for i, (_, row) in enumerate(recommendations_df.head(top_n).iterrows()):
            target_title = row['title']
            print(f"\n[{i+1}/{top_n}] 解释: {target_title}")

            explanation = self.explain_pair(source_title, target_title, save_plots=False)
            if explanation:
                explanations.append(explanation)

        return explanations

    def generate_full_report(self, source_title, target_title):
        """
        生成完整的SHAP分析报告

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
        explanation = self.explain_pair(source_title, target_title, save_plots=True)

        if explanation is None:
            return "无法生成报告: 电影未找到"

        # 生成summary plot
        self.plot_summary()

        report = []
        report.append("=" * 70)
        report.append("SHAP 可解释性分析报告")
        report.append("=" * 70)
        report.append("")
        report.append(f"源电影: 《{source_title}》")
        report.append(f"推荐电影: 《{target_title}》")
        report.append("")
        report.append("【相似度分析】")
        report.append(f"  实际相似度: {explanation['actual_similarity']:.2%}")
        report.append(f"  模型预测: {explanation['predicted_similarity']:.2%}")
        report.append(f"  基准值: {explanation['base_value']:.4f}")
        report.append("")
        report.append("【正向影响因素】(增加相似度)")
        for item in explanation['top_positive'][:5]:
            report.append(f"  + {item['feature']}: +{item['shap_value']:.4f}")
        report.append("")
        report.append("【负向影响因素】(降低相似度)")
        for item in explanation['top_negative'][:5]:
            report.append(f"  - {item['feature']}: {item['shap_value']:.4f}")
        report.append("")
        report.append("【生成的可视化文件】")
        report.append(f"  - shap_waterfall.png (瀑布图)")
        report.append(f"  - shap_force.png (力场图)")
        report.append(f"  - shap_summary.png (Summary Plot)")
        report.append(f"  - shap_importance_bar.png (特征重要性)")
        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


def demo():
    """演示SHAP可解释性功能"""
    print("=" * 70)
    print("SHAP可解释性框架演示")
    print("=" * 70)

    if not SHAP_AVAILABLE:
        print("\n错误: SHAP包未安装")
        print("请运行: pip install shap")
        return

    print("\n使用方法:")
    print("-" * 70)
    print("""
# 1. 导入必要的模块
from movie_recommender import MovieRecommenderSystem
from shap_explainability import MovieSHAPExplainer

# 2. 初始化推荐系统并构建模型
recommender = MovieRecommenderSystem(
    'tmdb_5000_movies.csv',
    'tmdb_5000_credits.csv'
)
recommender.build_metadata_based_model()

# 3. 创建SHAP解释器
explainer = MovieSHAPExplainer(
    recommender.movies,
    recommender.metadata_similarity
)

# 4. 训练解释模型
explainer.train_model()

# 5. 解释单对电影
result = explainer.explain_pair('Avatar', 'Titanic')

# 6. 生成完整报告
report = explainer.generate_full_report('Avatar', 'Titanic')
print(report)

# 7. 绘制全局特征重要性
explainer.plot_summary()

# 8. 绘制特征依赖图
explainer.plot_dependence('类型_Action_匹配')
""")
    print("-" * 70)

    # 尝试运行实际演示
    try:
        print("\n尝试运行实际演示...")
        from movie_recommender import MovieRecommenderSystem

        recommender = MovieRecommenderSystem(
            'tmdb_5000_movies.csv',
            'tmdb_5000_credits.csv'
        )
        recommender.build_metadata_based_model()

        explainer = MovieSHAPExplainer(
            recommender.movies,
            recommender.metadata_similarity
        )

        explainer.train_model()

        report = explainer.generate_full_report('Avatar', 'Titanic')
        print(report)

    except Exception as e:
        print(f"\n演示运行出错: {e}")
        print("请确保在正确的目录下运行，并且数据文件存在")


if __name__ == "__main__":
    demo()
