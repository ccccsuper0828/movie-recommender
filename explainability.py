"""
推荐系统可解释性模块
====================
为每种推荐方法生成人类可读的解释

作者: Movie Recommendation System
"""

import pandas as pd
import numpy as np
from collections import Counter


class RecommendationExplainer:
    """
    推荐解释器类
    为推荐结果生成可解释的理由
    """

    def __init__(self, movies_df, tfidf_vectorizer=None, tfidf_matrix=None):
        """
        初始化解释器

        Parameters:
        -----------
        movies_df : DataFrame
            电影数据
        tfidf_vectorizer : TfidfVectorizer, optional
            TF-IDF向量化器（用于内容解释）
        tfidf_matrix : sparse matrix, optional
            TF-IDF矩阵
        """
        self.movies = movies_df
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.title_to_idx = pd.Series(movies_df.index, index=movies_df['title']).to_dict()

    def explain_content_based(self, source_title, target_title, top_keywords=5):
        """
        解释基于内容的推荐原因

        Parameters:
        -----------
        source_title : str
            源电影标题
        target_title : str
            推荐电影标题
        top_keywords : int
            返回的关键词数量

        Returns:
        --------
        dict : 包含解释信息的字典
        """
        explanation = {
            'method': '基于内容 (Content-Based)',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {},
            'summary': ''
        }

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            explanation['summary'] = '无法生成解释：电影未找到'
            return explanation

        source_movie = self.movies.iloc[source_idx]
        target_movie = self.movies.iloc[target_idx]

        # 1. 分析共同关键词（如果有TF-IDF向量化器）
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()

            source_vector = self.tfidf_matrix[source_idx].toarray().flatten()
            target_vector = self.tfidf_matrix[target_idx].toarray().flatten()

            # 找到两部电影共同的高权重词
            # 使用几何平均来找到双方都重要的词
            combined_importance = np.sqrt(source_vector * target_vector)
            top_indices = np.argsort(combined_importance)[::-1][:top_keywords]

            common_keywords = []
            for idx in top_indices:
                if combined_importance[idx] > 0:
                    common_keywords.append({
                        'keyword': feature_names[idx],
                        'source_weight': round(source_vector[idx], 3),
                        'target_weight': round(target_vector[idx], 3),
                        'combined_score': round(combined_importance[idx], 3)
                    })

            if common_keywords:
                keywords_str = ', '.join([kw['keyword'] for kw in common_keywords])
                explanation['reasons'].append(f"共同主题词: {keywords_str}")
                explanation['details']['common_keywords'] = common_keywords

        # 2. 分析简介相似性
        source_overview = str(source_movie.get('overview', ''))
        target_overview = str(target_movie.get('overview', ''))

        if source_overview and target_overview:
            # 简单的词重叠分析
            source_words = set(source_overview.lower().split())
            target_words = set(target_overview.lower().split())
            # 移除常见停用词
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                          'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                          'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                          'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                          'as', 'into', 'through', 'during', 'before', 'after', 'and',
                          'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                          'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
                          'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she',
                          'his', 'her', 'they', 'their', 'them', 'who', 'which', 'what',
                          'when', 'where', 'why', 'how', 'all', 'each', 'every', 'any'}
            source_words = source_words - stop_words
            target_words = target_words - stop_words
            common_words = source_words & target_words

            if len(common_words) > 3:
                # 只取最长的几个有意义的词
                meaningful_common = sorted([w for w in common_words if len(w) > 4],
                                           key=len, reverse=True)[:5]
                if meaningful_common:
                    explanation['details']['common_plot_words'] = meaningful_common

        # 生成总结
        if explanation['reasons']:
            explanation['summary'] = f"推荐《{target_title}》因为它与《{source_title}》在剧情主题上相似"
        else:
            explanation['summary'] = f"《{target_title}》与《{source_title}》的剧情简介在语义上相似"

        return explanation

    def explain_metadata_based(self, source_title, target_title):
        """
        解释基于元数据的推荐原因

        Parameters:
        -----------
        source_title : str
            源电影标题
        target_title : str
            推荐电影标题

        Returns:
        --------
        dict : 包含解释信息的字典
        """
        explanation = {
            'method': '基于元数据 (Metadata-Based)',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {
                'matching_features': [],
                'feature_contributions': {}
            },
            'summary': ''
        }

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            explanation['summary'] = '无法生成解释：电影未找到'
            return explanation

        source = self.movies.iloc[source_idx]
        target = self.movies.iloc[target_idx]

        total_contribution = 0
        contributions = {}

        # 1. 检查导演匹配 (权重: 30%)
        source_director = source.get('director', '')
        target_director = target.get('director', '')
        if source_director and target_director and source_director == target_director:
            explanation['reasons'].append(f"同一导演: {source_director}")
            explanation['details']['matching_features'].append({
                'feature': '导演',
                'value': source_director,
                'weight': '高 (×3)'
            })
            contributions['导演'] = 30
            total_contribution += 30

        # 2. 检查类型重叠 (权重: 30%)
        source_genres = set(source.get('genres_list', []))
        target_genres = set(target.get('genres_list', []))
        common_genres = source_genres & target_genres

        if common_genres:
            genres_str = ', '.join(common_genres)
            explanation['reasons'].append(f"相同类型: {genres_str}")
            explanation['details']['matching_features'].append({
                'feature': '类型',
                'value': list(common_genres),
                'weight': '高 (×3)'
            })
            # 根据重叠比例计算贡献
            overlap_ratio = len(common_genres) / max(len(source_genres), 1)
            genre_contribution = int(30 * overlap_ratio)
            contributions['类型'] = genre_contribution
            total_contribution += genre_contribution

        # 3. 检查演员重叠 (权重: 20%)
        source_cast = set(source.get('cast_list', []))
        target_cast = set(target.get('cast_list', []))
        common_cast = source_cast & target_cast

        if common_cast:
            cast_str = ', '.join(list(common_cast)[:3])  # 最多显示3个
            explanation['reasons'].append(f"共同演员: {cast_str}")
            explanation['details']['matching_features'].append({
                'feature': '演员',
                'value': list(common_cast),
                'weight': '中 (×2)'
            })
            overlap_ratio = len(common_cast) / max(len(source_cast), 1)
            cast_contribution = int(20 * overlap_ratio)
            contributions['演员'] = cast_contribution
            total_contribution += cast_contribution

        # 4. 检查关键词重叠 (权重: 20%)
        source_keywords = set(source.get('keywords_list', []))
        target_keywords = set(target.get('keywords_list', []))
        common_keywords = source_keywords & target_keywords

        if common_keywords:
            keywords_str = ', '.join(list(common_keywords)[:3])
            explanation['reasons'].append(f"相似主题: {keywords_str}")
            explanation['details']['matching_features'].append({
                'feature': '关键词',
                'value': list(common_keywords),
                'weight': '标准 (×1)'
            })
            overlap_ratio = len(common_keywords) / max(len(source_keywords), 1)
            keyword_contribution = int(20 * overlap_ratio)
            contributions['关键词'] = keyword_contribution
            total_contribution += keyword_contribution

        explanation['details']['feature_contributions'] = contributions
        explanation['details']['total_contribution_score'] = total_contribution

        # 生成总结
        if explanation['reasons']:
            primary_reason = explanation['reasons'][0]
            explanation['summary'] = f"推荐《{target_title}》主要因为: {primary_reason}"
        else:
            explanation['summary'] = f"《{target_title}》与《{source_title}》在综合特征上相似"

        return explanation

    def explain_collaborative_filtering(self, source_title, target_title,
                                        user_movie_matrix=None, similarity_score=None):
        """
        解释协同过滤的推荐原因

        Parameters:
        -----------
        source_title : str
            源电影标题
        target_title : str
            推荐电影标题
        user_movie_matrix : ndarray, optional
            用户-电影评分矩阵
        similarity_score : float, optional
            相似度分数

        Returns:
        --------
        dict : 包含解释信息的字典
        """
        explanation = {
            'method': '协同过滤 (Collaborative Filtering)',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {},
            'summary': ''
        }

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            explanation['summary'] = '无法生成解释：电影未找到'
            return explanation

        if user_movie_matrix is not None:
            # 找到喜欢源电影的用户
            source_ratings = user_movie_matrix[:, source_idx]
            source_fans = np.where(source_ratings >= 4.0)[0]

            # 找到喜欢目标电影的用户
            target_ratings = user_movie_matrix[:, target_idx]
            target_fans = np.where(target_ratings >= 4.0)[0]

            # 计算重叠
            common_fans = set(source_fans) & set(target_fans)

            if len(source_fans) > 0:
                overlap_percentage = len(common_fans) / len(source_fans) * 100

                explanation['reasons'].append(
                    f"喜欢《{source_title}》的观众中，{overlap_percentage:.0f}% 也喜欢《{target_title}》"
                )

                explanation['details']['source_fans_count'] = len(source_fans)
                explanation['details']['target_fans_count'] = len(target_fans)
                explanation['details']['common_fans_count'] = len(common_fans)
                explanation['details']['overlap_percentage'] = round(overlap_percentage, 1)

                # 分析共同粉丝的评分
                if len(common_fans) > 0:
                    common_fans_list = list(common_fans)
                    avg_source_rating = np.mean(source_ratings[common_fans_list])
                    avg_target_rating = np.mean(target_ratings[common_fans_list])

                    explanation['details']['avg_source_rating'] = round(avg_source_rating, 2)
                    explanation['details']['avg_target_rating'] = round(avg_target_rating, 2)

        if similarity_score is not None:
            explanation['details']['similarity_score'] = round(similarity_score, 3)
            explanation['reasons'].append(f"用户行为相似度: {similarity_score:.1%}")

        # 生成总结
        if explanation['reasons']:
            explanation['summary'] = explanation['reasons'][0]
        else:
            explanation['summary'] = f"基于用户行为模式，喜欢《{source_title}》的用户也可能喜欢《{target_title}》"

        return explanation

    def explain_hybrid(self, source_title, target_title,
                       content_score=None, metadata_score=None, cf_score=None,
                       weights=(0.3, 0.4, 0.3)):
        """
        解释混合推荐的原因

        Parameters:
        -----------
        source_title : str
            源电影标题
        target_title : str
            推荐电影标题
        content_score : float
            内容相似度分数
        metadata_score : float
            元数据相似度分数
        cf_score : float
            协同过滤相似度分数
        weights : tuple
            各方法权重

        Returns:
        --------
        dict : 包含解释信息的字典
        """
        explanation = {
            'method': '混合推荐 (Hybrid)',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {
                'method_scores': {},
                'method_contributions': {},
                'weights': {
                    '内容': weights[0],
                    '元数据': weights[1],
                    '协同过滤': weights[2]
                }
            },
            'summary': ''
        }

        # 记录各方法分数
        scores = {}
        contributions = {}

        if content_score is not None:
            scores['内容'] = content_score
            contributions['内容'] = content_score * weights[0]

        if metadata_score is not None:
            scores['元数据'] = metadata_score
            contributions['元数据'] = metadata_score * weights[1]

        if cf_score is not None:
            scores['协同过滤'] = cf_score
            contributions['协同过滤'] = cf_score * weights[2]

        explanation['details']['method_scores'] = {k: round(v, 3) for k, v in scores.items()}
        explanation['details']['method_contributions'] = {k: round(v, 3) for k, v in contributions.items()}

        # 找到贡献最大的方法
        if contributions:
            primary_method = max(contributions, key=contributions.get)
            primary_contribution = contributions[primary_method] / sum(contributions.values()) * 100

            explanation['reasons'].append(
                f"主要由{primary_method}方法贡献 ({primary_contribution:.0f}%)"
            )

            # 添加各方法的描述
            method_descriptions = {
                '内容': '剧情主题相似',
                '元数据': '类型/导演/演员相似',
                '协同过滤': '用户行为模式相似'
            }

            for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:  # 只显示有显著贡献的方法
                    explanation['reasons'].append(
                        f"{method}: {method_descriptions.get(method, '')} (相似度: {score:.1%})"
                    )

        # 生成总结
        if contributions:
            total_score = sum(contributions.values())
            explanation['details']['total_hybrid_score'] = round(total_score, 3)
            explanation['summary'] = f"综合三种方法推荐《{target_title}》，总相似度: {total_score:.1%}"
        else:
            explanation['summary'] = f"《{target_title}》是《{source_title}》的综合推荐结果"

        return explanation

    def generate_user_friendly_explanation(self, explanation_dict):
        """
        生成用户友好的解释文本

        Parameters:
        -----------
        explanation_dict : dict
            explain_* 方法返回的解释字典

        Returns:
        --------
        str : 格式化的解释文本
        """
        lines = []
        lines.append(f"{'='*50}")
        lines.append(f"推荐解释")
        lines.append(f"{'='*50}")
        lines.append(f"")
        lines.append(f"源电影: 《{explanation_dict['source_movie']}》")
        lines.append(f"推荐电影: 《{explanation_dict['recommended_movie']}》")
        lines.append(f"推荐方法: {explanation_dict['method']}")
        lines.append(f"")
        lines.append(f"【推荐理由】")

        if explanation_dict['reasons']:
            for i, reason in enumerate(explanation_dict['reasons'], 1):
                lines.append(f"  {i}. {reason}")
        else:
            lines.append(f"  整体特征匹配")

        lines.append(f"")
        lines.append(f"【总结】")
        lines.append(f"  {explanation_dict['summary']}")

        if explanation_dict['details']:
            lines.append(f"")
            lines.append(f"【详细信息】")
            for key, value in explanation_dict['details'].items():
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    - {k}: {v}")
                elif isinstance(value, list):
                    lines.append(f"  {key}: {', '.join(map(str, value[:5]))}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append(f"{'='*50}")

        return '\n'.join(lines)

    def get_comparison_explanation(self, source_title, recommendations_df,
                                   content_sim=None, metadata_sim=None, cf_sim=None,
                                   user_movie_matrix=None):
        """
        为多个推荐结果生成对比解释

        Parameters:
        -----------
        source_title : str
            源电影
        recommendations_df : DataFrame
            推荐结果DataFrame
        其他参数用于生成详细解释

        Returns:
        --------
        list : 解释列表
        """
        explanations = []
        source_idx = self.title_to_idx.get(source_title)

        for _, row in recommendations_df.iterrows():
            target_title = row['title']
            target_idx = self.title_to_idx.get(target_title)

            if target_idx is None:
                continue

            # 获取各方法的分数
            content_score = content_sim[source_idx, target_idx] if content_sim is not None else None
            metadata_score = metadata_sim[source_idx, target_idx] if metadata_sim is not None else None
            cf_score_val = cf_sim[source_idx, target_idx] if cf_sim is not None else None

            # 生成元数据解释（最易理解）
            metadata_explanation = self.explain_metadata_based(source_title, target_title)

            explanations.append({
                'title': target_title,
                'similarity_score': row.get('similarity_score', row.get('hybrid_score', 0)),
                'explanation': metadata_explanation,
                'scores': {
                    'content': round(content_score, 3) if content_score else None,
                    'metadata': round(metadata_score, 3) if metadata_score else None,
                    'cf': round(cf_score_val, 3) if cf_score_val else None
                }
            })

        return explanations


def demo_explainability():
    """演示可解释性功能"""
    print("=" * 60)
    print("推荐系统可解释性演示")
    print("=" * 60)

    # 这里需要实际的数据来演示
    print("\n请通过 MovieRecommenderSystem 类使用可解释性功能")
    print("\n使用示例:")
    print("""
    from movie_recommender import MovieRecommenderSystem
    from explainability import RecommendationExplainer

    # 初始化推荐系统
    recommender = MovieRecommenderSystem('tmdb_5000_movies.csv', 'tmdb_5000_credits.csv')
    recommender.build_content_based_model()
    recommender.build_metadata_based_model()

    # 创建解释器
    explainer = RecommendationExplainer(
        recommender.movies,
        recommender.tfidf_vectorizer,  # 需要保存TF-IDF向量化器
        recommender.tfidf_matrix       # 需要保存TF-IDF矩阵
    )

    # 生成解释
    explanation = explainer.explain_metadata_based('Avatar', 'Titanic')
    print(explainer.generate_user_friendly_explanation(explanation))
    """)


if __name__ == "__main__":
    demo_explainability()
