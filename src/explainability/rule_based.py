"""
Rule-based explainability for movie recommendations.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RuleBasedExplainer:
    """
    Generates human-readable explanations for recommendations
    based on feature matching rules.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        tfidf_vectorizer=None,
        tfidf_matrix=None
    ):
        """
        Initialize the rule-based explainer.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe
        tfidf_vectorizer : TfidfVectorizer, optional
            TF-IDF vectorizer for content explanations
        tfidf_matrix : sparse matrix, optional
            TF-IDF matrix
        """
        self.movies = movies_df
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.title_to_idx = pd.Series(movies_df.index, index=movies_df['title']).to_dict()

    def explain_content_based(
        self,
        source_title: str,
        target_title: str,
        top_keywords: int = 5
    ) -> Dict[str, Any]:
        """
        Explain content-based recommendation.

        Parameters
        ----------
        source_title : str
            Source movie title
        target_title : str
            Recommended movie title
        top_keywords : int
            Number of keywords to include

        Returns
        -------
        dict
            Explanation dictionary
        """
        explanation = {
            'method': 'Content-Based (TF-IDF)',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {},
            'summary': ''
        }

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            explanation['summary'] = 'Unable to generate explanation: movie not found'
            return explanation

        # Analyze common keywords if TF-IDF available
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()

            source_vec = self.tfidf_matrix[source_idx].toarray().flatten()
            target_vec = self.tfidf_matrix[target_idx].toarray().flatten()

            # Find common important keywords
            combined = np.sqrt(source_vec * target_vec)
            top_indices = np.argsort(combined)[::-1][:top_keywords]

            common_keywords = []
            for idx in top_indices:
                if combined[idx] > 0:
                    common_keywords.append({
                        'keyword': feature_names[idx],
                        'combined_score': round(combined[idx], 3)
                    })

            if common_keywords:
                keywords_str = ', '.join([kw['keyword'] for kw in common_keywords])
                explanation['reasons'].append(f"Common themes: {keywords_str}")
                explanation['details']['common_keywords'] = common_keywords

        # Generate summary
        if explanation['reasons']:
            explanation['summary'] = (
                f"Recommended '{target_title}' because it shares thematic "
                f"similarities with '{source_title}'"
            )
        else:
            explanation['summary'] = (
                f"'{target_title}' has similar plot description to '{source_title}'"
            )

        return explanation

    def explain_metadata_based(
        self,
        source_title: str,
        target_title: str
    ) -> Dict[str, Any]:
        """
        Explain metadata-based recommendation.

        Parameters
        ----------
        source_title : str
            Source movie title
        target_title : str
            Recommended movie title

        Returns
        -------
        dict
            Explanation dictionary
        """
        explanation = {
            'method': 'Metadata-Based',
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
            explanation['summary'] = 'Unable to generate explanation: movie not found'
            return explanation

        source = self.movies.iloc[source_idx]
        target = self.movies.iloc[target_idx]

        contributions = {}

        # Check director match (30% weight)
        source_dir = source.get('director', '')
        target_dir = target.get('director', '')
        if source_dir and target_dir and source_dir == target_dir:
            explanation['reasons'].append(f"Same director: {source_dir}")
            explanation['details']['matching_features'].append({
                'feature': 'Director',
                'value': source_dir,
                'weight': 'High (×3)'
            })
            contributions['Director'] = 30

        # Check genre overlap (30% weight)
        source_genres = set(source.get('genres_list', []) or [])
        target_genres = set(target.get('genres_list', []) or [])
        common_genres = source_genres & target_genres

        if common_genres:
            genres_str = ', '.join(common_genres)
            explanation['reasons'].append(f"Same genres: {genres_str}")
            explanation['details']['matching_features'].append({
                'feature': 'Genres',
                'value': list(common_genres),
                'weight': 'High (×3)'
            })
            overlap = len(common_genres) / max(len(source_genres), 1)
            contributions['Genres'] = int(30 * overlap)

        # Check cast overlap (20% weight)
        source_cast = set(source.get('cast_list', []) or [])
        target_cast = set(target.get('cast_list', []) or [])
        common_cast = source_cast & target_cast

        if common_cast:
            cast_str = ', '.join(list(common_cast)[:3])
            explanation['reasons'].append(f"Common cast: {cast_str}")
            explanation['details']['matching_features'].append({
                'feature': 'Cast',
                'value': list(common_cast),
                'weight': 'Medium (×2)'
            })
            overlap = len(common_cast) / max(len(source_cast), 1)
            contributions['Cast'] = int(20 * overlap)

        # Check keyword overlap (20% weight)
        source_kw = set(source.get('keywords_list', []) or [])
        target_kw = set(target.get('keywords_list', []) or [])
        common_kw = source_kw & target_kw

        if common_kw:
            kw_str = ', '.join(list(common_kw)[:3])
            explanation['reasons'].append(f"Similar themes: {kw_str}")
            explanation['details']['matching_features'].append({
                'feature': 'Keywords',
                'value': list(common_kw),
                'weight': 'Standard (×1)'
            })
            overlap = len(common_kw) / max(len(source_kw), 1)
            contributions['Keywords'] = int(20 * overlap)

        explanation['details']['feature_contributions'] = contributions

        # Generate summary
        if explanation['reasons']:
            explanation['summary'] = (
                f"Recommended '{target_title}' primarily because: {explanation['reasons'][0]}"
            )
        else:
            explanation['summary'] = (
                f"'{target_title}' has similar overall features to '{source_title}'"
            )

        return explanation

    def explain_collaborative_filtering(
        self,
        source_title: str,
        target_title: str,
        user_movie_matrix: Optional[np.ndarray] = None,
        similarity_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Explain collaborative filtering recommendation.

        Parameters
        ----------
        source_title : str
            Source movie title
        target_title : str
            Recommended movie title
        user_movie_matrix : np.ndarray, optional
            User-movie rating matrix
        similarity_score : float, optional
            CF similarity score

        Returns
        -------
        dict
            Explanation dictionary
        """
        explanation = {
            'method': 'Collaborative Filtering',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {},
            'summary': ''
        }

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            explanation['summary'] = 'Unable to generate explanation: movie not found'
            return explanation

        if user_movie_matrix is not None:
            # Find users who liked source movie
            source_ratings = user_movie_matrix[:, source_idx]
            source_fans = np.where(source_ratings >= 4.0)[0]

            # Find users who liked target movie
            target_ratings = user_movie_matrix[:, target_idx]
            target_fans = np.where(target_ratings >= 4.0)[0]

            # Calculate overlap
            common_fans = set(source_fans) & set(target_fans)

            if len(source_fans) > 0:
                overlap_pct = len(common_fans) / len(source_fans) * 100

                explanation['reasons'].append(
                    f"{overlap_pct:.0f}% of users who liked '{source_title}' "
                    f"also liked '{target_title}'"
                )

                explanation['details']['source_fans'] = len(source_fans)
                explanation['details']['target_fans'] = len(target_fans)
                explanation['details']['common_fans'] = len(common_fans)
                explanation['details']['overlap_percentage'] = round(overlap_pct, 1)

        if similarity_score is not None:
            explanation['details']['similarity_score'] = round(similarity_score, 3)
            explanation['reasons'].append(f"User behavior similarity: {similarity_score:.1%}")

        # Generate summary
        if explanation['reasons']:
            explanation['summary'] = explanation['reasons'][0]
        else:
            explanation['summary'] = (
                f"Users who liked '{source_title}' tend to also like '{target_title}'"
            )

        return explanation

    def explain_hybrid(
        self,
        source_title: str,
        target_title: str,
        content_score: Optional[float] = None,
        metadata_score: Optional[float] = None,
        cf_score: Optional[float] = None,
        weights: tuple = (0.3, 0.4, 0.3)
    ) -> Dict[str, Any]:
        """
        Explain hybrid recommendation.

        Parameters
        ----------
        source_title : str
            Source movie title
        target_title : str
            Recommended movie title
        content_score : float, optional
            Content-based similarity score
        metadata_score : float, optional
            Metadata-based similarity score
        cf_score : float, optional
            Collaborative filtering similarity score
        weights : tuple
            Method weights

        Returns
        -------
        dict
            Explanation dictionary
        """
        explanation = {
            'method': 'Hybrid',
            'source_movie': source_title,
            'recommended_movie': target_title,
            'reasons': [],
            'details': {
                'method_scores': {},
                'method_contributions': {},
                'weights': {
                    'content': weights[0],
                    'metadata': weights[1],
                    'cf': weights[2]
                }
            },
            'summary': ''
        }

        scores = {}
        contributions = {}

        if content_score is not None:
            scores['content'] = content_score
            contributions['content'] = content_score * weights[0]

        if metadata_score is not None:
            scores['metadata'] = metadata_score
            contributions['metadata'] = metadata_score * weights[1]

        if cf_score is not None:
            scores['cf'] = cf_score
            contributions['cf'] = cf_score * weights[2]

        explanation['details']['method_scores'] = {k: round(v, 3) for k, v in scores.items()}
        explanation['details']['method_contributions'] = {k: round(v, 3) for k, v in contributions.items()}

        # Find primary contributor
        if contributions:
            primary = max(contributions, key=contributions.get)
            primary_pct = contributions[primary] / sum(contributions.values()) * 100

            method_names = {
                'content': 'plot similarities',
                'metadata': 'genre/director/cast matches',
                'cf': 'user behavior patterns'
            }

            explanation['reasons'].append(
                f"Primarily recommended due to {method_names.get(primary, primary)} "
                f"({primary_pct:.0f}% contribution)"
            )

            # Add details for significant methods
            for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:
                    explanation['reasons'].append(
                        f"{method.capitalize()}: {method_names.get(method, method)} "
                        f"(similarity: {score:.1%})"
                    )

            total_score = sum(contributions.values())
            explanation['details']['total_hybrid_score'] = round(total_score, 3)
            explanation['summary'] = (
                f"Recommended '{target_title}' with combined similarity of {total_score:.1%}"
            )
        else:
            explanation['summary'] = (
                f"'{target_title}' is a comprehensive match for '{source_title}'"
            )

        return explanation

    def format_explanation(self, explanation: Dict[str, Any]) -> str:
        """
        Format an explanation dictionary as human-readable text.

        Parameters
        ----------
        explanation : dict
            Explanation dictionary

        Returns
        -------
        str
            Formatted explanation text
        """
        lines = [
            "=" * 50,
            "Recommendation Explanation",
            "=" * 50,
            "",
            f"Source: {explanation['source_movie']}",
            f"Recommended: {explanation['recommended_movie']}",
            f"Method: {explanation['method']}",
            "",
            "Reasons:",
        ]

        if explanation['reasons']:
            for i, reason in enumerate(explanation['reasons'], 1):
                lines.append(f"  {i}. {reason}")
        else:
            lines.append("  Overall feature similarity")

        lines.extend([
            "",
            "Summary:",
            f"  {explanation['summary']}",
            "=" * 50
        ])

        return '\n'.join(lines)
