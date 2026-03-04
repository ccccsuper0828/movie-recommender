"""
SHAP-based explainability for movie recommendations.
"""

from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings('ignore')

# Check SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainer for understanding recommendation similarities.

    Uses a surrogate model trained to predict similarity scores,
    then applies SHAP to explain the predictions.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        similarity_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize the SHAP explainer.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Movie dataframe
        similarity_matrix : np.ndarray, optional
            Pre-computed similarity matrix
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Run: pip install shap")

        self.movies = movies_df.reset_index(drop=True)
        self.similarity_matrix = similarity_matrix
        self.title_to_idx = pd.Series(self.movies.index, index=self.movies['title']).to_dict()

        # Model components
        self.model: Optional[GradientBoostingRegressor] = None
        self.explainer = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.pair_feature_names: List[str] = []

        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None

        # Encoders
        self.genre_encoder = MultiLabelBinarizer()

    def prepare_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for all movies.

        Returns
        -------
        tuple
            (feature_matrix, feature_names)
        """
        logger.info("Preparing feature matrix...")

        features = []
        feature_names = []

        # Genre features (one-hot)
        genres = self.movies['genres_list'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        genre_encoded = self.genre_encoder.fit_transform(genres)
        features.append(genre_encoded)
        feature_names.extend([f"Genre_{g}" for g in self.genre_encoder.classes_])

        # Numeric features
        for col in ['vote_average', 'popularity']:
            if col in self.movies.columns:
                values = self.movies[col].fillna(0).values.reshape(-1, 1)
                features.append(values)
                feature_names.append(col.replace('_', ' ').title())

        # Cast and keyword counts
        for col, name in [('cast_list', 'CastCount'), ('keywords_list', 'KeywordCount')]:
            if col in self.movies.columns:
                counts = self.movies[col].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                ).values.reshape(-1, 1)
                features.append(counts)
                feature_names.append(name)

        # Has director
        if 'director' in self.movies.columns:
            has_dir = self.movies['director'].apply(
                lambda x: 1 if x and str(x) != '' else 0
            ).values.reshape(-1, 1)
            features.append(has_dir)
            feature_names.append('HasDirector')

        self.feature_matrix = np.hstack(features)
        self.feature_names = feature_names

        logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")

        return self.feature_matrix, self.feature_names

    def _create_pair_features(self, idx1: int, idx2: int) -> np.ndarray:
        """Create combined features for a movie pair."""
        f1 = self.feature_matrix[idx1]
        f2 = self.feature_matrix[idx2]

        # Feature combinations
        diff = np.abs(f1 - f2)  # Difference
        match = f1 * f2  # Match (product)

        return np.concatenate([diff, match])

    def train(self, n_samples: int = 20000) -> 'SHAPExplainer':
        """
        Train the surrogate model for SHAP explanations.

        Parameters
        ----------
        n_samples : int
            Number of movie pairs to sample for training

        Returns
        -------
        SHAPExplainer
            Self for method chaining
        """
        logger.info("Training SHAP surrogate model...")

        if self.feature_matrix is None:
            self.prepare_features()

        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix required for training")

        n_movies = len(self.movies)

        # Sample movie pairs
        np.random.seed(42)
        idx1 = np.random.randint(0, n_movies, n_samples)
        idx2 = np.random.randint(0, n_movies, n_samples)

        # Remove same-movie pairs
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        # Create training data
        X, y = [], []
        for i1, i2 in zip(idx1, idx2):
            X.append(self._create_pair_features(i1, i2))
            y.append(self.similarity_matrix[i1, i2])

        X, y = np.array(X), np.array(y)

        # Create pair feature names
        self.pair_feature_names = (
            [f"{n}_Diff" for n in self.feature_names] +
            [f"{n}_Match" for n in self.feature_names]
        )

        logger.info(f"Training data: {X.shape}")

        # Train-test split
        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
        self.model.fit(self.X_train, y_train)

        # Evaluate
        train_score = self.model.score(self.X_train, y_train)
        test_score = self.model.score(self.X_test, y_test)
        logger.info(f"Model R² - Train: {train_score:.4f}, Test: {test_score:.4f}")

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP explainer created")

        return self

    def explain_pair(
        self,
        source_title: str,
        target_title: str
    ) -> Optional[Dict[str, Any]]:
        """
        Explain similarity between two movies using SHAP.

        Parameters
        ----------
        source_title : str
            Source movie title
        target_title : str
            Target movie title

        Returns
        -------
        dict or None
            SHAP explanation results
        """
        if self.model is None:
            logger.warning("Model not trained. Call train() first.")
            return None

        source_idx = self.title_to_idx.get(source_title)
        target_idx = self.title_to_idx.get(target_title)

        if source_idx is None or target_idx is None:
            logger.warning("Movie not found")
            return None

        # Create pair features
        X = self._create_pair_features(source_idx, target_idx).reshape(1, -1)

        # Get predictions
        predicted = self.model.predict(X)[0]
        actual = self.similarity_matrix[source_idx, target_idx]

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)[0]
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])

        # Analyze contributions
        shap_df = pd.DataFrame({
            'feature': self.pair_feature_names,
            'shap_value': shap_values
        })
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)

        positive = shap_df[shap_df['shap_value'] > 0.001].head(5)
        negative = shap_df[shap_df['shap_value'] < -0.001].head(5)

        return {
            'source_movie': source_title,
            'target_movie': target_title,
            'actual_similarity': float(actual),
            'predicted_similarity': float(predicted),
            'base_value': float(base_value),
            'shap_values': shap_values,
            'feature_names': self.pair_feature_names,
            'positive_contributors': positive.to_dict('records'),
            'negative_contributors': negative.to_dict('records'),
            'X': X
        }

    def get_feature_importance(
        self,
        n_samples: int = 200
    ) -> pd.DataFrame:
        """
        Get global feature importance based on SHAP values.

        Parameters
        ----------
        n_samples : int
            Number of samples to use

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.X_test is None:
            logger.warning("Model not trained")
            return pd.DataFrame()

        X_sample = self.X_test[:n_samples]
        shap_values = self.explainer.shap_values(X_sample)

        importance = np.abs(shap_values).mean(axis=0)

        return pd.DataFrame({
            'feature': self.pair_feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def format_explanation(self, result: Dict[str, Any]) -> str:
        """
        Format SHAP explanation as human-readable text.

        Parameters
        ----------
        result : dict
            Result from explain_pair()

        Returns
        -------
        str
            Formatted explanation
        """
        lines = [
            "=" * 60,
            "SHAP Explanation Report",
            "=" * 60,
            "",
            f"Source: {result['source_movie']}",
            f"Target: {result['target_movie']}",
            "",
            "Similarity Analysis:",
            f"  Actual: {result['actual_similarity']:.2%}",
            f"  Predicted: {result['predicted_similarity']:.2%}",
            f"  Base value: {result['base_value']:.4f}",
            "",
            "Positive Contributors (increase similarity):"
        ]

        for item in result['positive_contributors']:
            lines.append(f"  + {item['feature']}: +{item['shap_value']:.4f}")

        lines.extend([
            "",
            "Negative Contributors (decrease similarity):"
        ])

        for item in result['negative_contributors']:
            lines.append(f"  - {item['feature']}: {item['shap_value']:.4f}")

        lines.append("=" * 60)

        return '\n'.join(lines)

    @property
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self.model is not None


def is_shap_available() -> bool:
    """Check if SHAP is available."""
    return SHAP_AVAILABLE
