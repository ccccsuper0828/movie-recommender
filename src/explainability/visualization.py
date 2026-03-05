"""
Visualization utilities for recommendation explanations.
"""
# @author 成员 E — 可解释性 & 评估系统

from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExplanationVisualizer:
    """
    Creates visualizations for recommendation explanations.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the visualizer.

        Parameters
        ----------
        output_dir : Path, optional
            Directory to save visualizations
        """
        self.output_dir = output_dir or Path('.')

        # Color scheme
        self.colors = {
            'positive': '#e50914',  # Netflix red
            'negative': '#008bfb',  # Blue
            'neutral': '#666666',
            'background': '#1a1a24',
            'text': '#ffffff',
            'gold': '#f5c518'  # IMDb gold
        }

        # Font settings
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False

    def plot_shap_waterfall(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        base_value: float,
        title: str = "SHAP Waterfall",
        max_features: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a waterfall plot showing feature contributions.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values for each feature
        feature_names : List[str]
            Feature names
        base_value : float
            Base (expected) value
        title : str
            Plot title
        max_features : int
            Maximum features to display
        save_path : str, optional
            Path to save the figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get top features by absolute value
        indices = np.argsort(np.abs(shap_values))[::-1][:max_features]
        values = shap_values[indices]
        names = [feature_names[i][:25] for i in indices]

        # Colors based on positive/negative
        colors = [
            self.colors['positive'] if v > 0 else self.colors['negative']
            for v in values
        ]

        # Create horizontal bar chart
        y_pos = np.arange(len(values))
        ax.barh(y_pos, values, color=colors, height=0.7, alpha=0.8)

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('SHAP Value', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['positive'], label='Increases similarity'),
            Patch(facecolor=self.colors['negative'], label='Decreases similarity')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()

        if save_path:
            path = self.output_dir / save_path
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {path}")

        return fig

    def plot_shap_force(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        base_value: float,
        predicted_value: float,
        title: str = "SHAP Force Plot",
        max_features: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a force plot showing push/pull of features.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values
        feature_names : List[str]
            Feature names
        base_value : float
            Base value
        predicted_value : float
            Predicted value
        title : str
            Plot title
        max_features : int
            Maximum features to show
        save_path : str, optional
            Save path

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 3))

        # Get top features
        indices = np.argsort(np.abs(shap_values))[::-1][:max_features]

        # Draw force arrows
        left_pos = base_value
        left_neg = base_value

        for idx in indices:
            val = shap_values[idx]
            if val > 0:
                ax.barh(0, val, left=left_pos, color=self.colors['positive'],
                        height=0.5, alpha=0.8)
                left_pos += val
            else:
                ax.barh(0, val, left=left_neg, color=self.colors['negative'],
                        height=0.5, alpha=0.8)
                left_neg += val

        # Reference lines
        ax.axvline(x=base_value, color='gray', linestyle='--',
                   linewidth=1.5, label=f'Base: {base_value:.3f}')
        ax.axvline(x=predicted_value, color='black', linestyle='-',
                   linewidth=2, label=f'Predicted: {predicted_value:.3f}')

        ax.set_yticks([])
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()

        if save_path:
            path = self.output_dir / save_path
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {path}")

        return fig

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        title: str = "Feature Importance",
        max_features: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a feature importance bar plot.

        Parameters
        ----------
        feature_names : List[str]
            Feature names
        importance_values : np.ndarray
            Importance values
        title : str
            Plot title
        max_features : int
            Maximum features to display
        save_path : str, optional
            Save path

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort by importance
        indices = np.argsort(importance_values)[::-1][:max_features]
        values = importance_values[indices]
        names = [feature_names[i][:25] for i in indices]

        # Color gradient
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(values)))

        y_pos = np.arange(len(values))
        ax.barh(y_pos, values, color=colors, height=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            path = self.output_dir / save_path
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {path}")

        return fig

    def plot_method_comparison(
        self,
        method_scores: Dict[str, float],
        title: str = "Method Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a bar chart comparing different methods.

        Parameters
        ----------
        method_scores : dict
            Dictionary of method names to scores
        title : str
            Plot title
        save_path : str, optional
            Save path

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(method_scores.keys())
        scores = list(method_scores.values())

        colors = [self.colors['positive'], self.colors['gold'],
                  self.colors['negative'], '#4CAF50'][:len(methods)]

        bars = ax.bar(methods, scores, color=colors, alpha=0.8)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{score:.1%}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Similarity Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(scores) * 1.15)

        plt.tight_layout()

        if save_path:
            path = self.output_dir / save_path
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {path}")

        return fig

    def plot_similarity_ring(
        self,
        score: float,
        label: str = "",
        size: Tuple[int, int] = (3, 3),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a circular progress indicator for similarity score.

        Parameters
        ----------
        score : float
            Similarity score (0 to 1)
        label : str
            Label text
        size : tuple
            Figure size
        save_path : str, optional
            Save path

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=size)

        # Create ring
        theta = np.linspace(0, 2 * np.pi * score, 100)
        r = 1

        # Background ring
        ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)),
                np.sin(np.linspace(0, 2 * np.pi, 100)),
                color='#e0e0e0', linewidth=10)

        # Score ring
        ax.plot(np.cos(theta), np.sin(theta),
                color=self.colors['positive'], linewidth=10)

        # Center text
        ax.text(0, 0, f'{score:.0%}', ha='center', va='center',
                fontsize=16, fontweight='bold')

        if label:
            ax.text(0, -1.5, label, ha='center', va='center', fontsize=10)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        if save_path:
            path = self.output_dir / save_path
            fig.savefig(path, dpi=150, bbox_inches='tight',
                        facecolor='white', transparent=True)
            logger.info(f"Saved: {path}")

        return fig

    def close_all(self):
        """Close all matplotlib figures."""
        plt.close('all')
