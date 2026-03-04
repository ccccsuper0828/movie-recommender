"""
Explanation panel component for recommendation explanations.
"""

import streamlit as st
from typing import Dict, List, Any, Optional


class ExplanationPanel:
    """
    Component for displaying detailed recommendation explanations.
    """

    @staticmethod
    def render(
        source_movie: str,
        target_movie: str,
        explanation: Dict[str, Any],
        method_scores: Optional[Dict[str, float]] = None
    ):
        """Render an explanation panel as a single self-contained HTML block."""
        summary = explanation.get('summary', '')
        reasons = explanation.get('reasons', [])

        parts = []
        parts.append(
            '<div style="background:linear-gradient(135deg,#1c2333 0%,rgba(245,197,24,0.08) 100%);'
            'border-radius:16px;padding:24px;margin:20px 0;border:1px solid #30363d;">'
        )
        parts.append(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
            '<span style="font-size:1.5rem;">💡</span>'
            f'<h3 style="color:#f5c518;margin:0;font-size:1.1rem;">Why &quot;{target_movie}&quot;?</h3>'
            '</div>'
        )

        if summary:
            parts.append(
                f'<p style="color:#ffffff;font-size:1rem;margin:0 0 16px 0;'
                f'padding-left:20px;border-left:3px solid #f5c518;">{summary}</p>'
            )

        if reasons:
            parts.append('<h4 style="color:#a3a3a3;font-size:0.9rem;margin:16px 0 12px 0;">Key Factors:</h4>')
            for reason in reasons[:5]:
                parts.append(
                    f'<div style="background:rgba(229,9,20,0.1);border-radius:8px;'
                    f'padding:10px 14px;margin:8px 0;border-left:3px solid #e5383b;">'
                    f'<p style="color:#ffffff;margin:0;font-size:0.9rem;">{reason}</p></div>'
                )

        if method_scores:
            parts.append('<h4 style="color:#a3a3a3;font-size:0.9rem;margin:20px 0 12px 0;">Method Breakdown:</h4>')
            parts.append('<div style="display:flex;gap:12px;flex-wrap:wrap;">')

            method_colors = {'content': '#3498db', 'metadata': '#2ecc71', 'cf': '#9b59b6', 'hybrid': '#e5383b'}
            method_labels = {'content': 'Content', 'metadata': 'Metadata', 'cf': 'CF', 'hybrid': 'Hybrid'}

            for method, score in method_scores.items():
                color = method_colors.get(method, '#666')
                label = method_labels.get(method, method)
                score_pct = score * 100
                parts.append(
                    f'<div style="background:#0d1117;border-radius:8px;padding:12px 16px;'
                    f'text-align:center;min-width:80px;border:1px solid {color}40;">'
                    f'<p style="color:{color};font-size:0.75rem;margin:0 0 4px 0;text-transform:uppercase;">{label}</p>'
                    f'<p style="color:#ffffff;font-size:1.1rem;font-weight:700;margin:0;">{score_pct:.0f}%</p>'
                    f'</div>'
                )
            parts.append('</div>')

        parts.append('</div>')
        st.markdown(''.join(parts), unsafe_allow_html=True)

    @staticmethod
    def render_shap_explanation(
        shap_result: Dict[str, Any],
        source_movie: str,
        target_movie: str
    ):
        """Render SHAP-based explanation."""
        st.markdown(
            '<div style="background:#1c2333;border-radius:16px;padding:24px;margin:20px 0;border:1px solid #30363d;">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
            '<span style="font-size:1.5rem;">🔬</span>'
            '<h3 style="color:#ffffff;margin:0;">SHAP Analysis</h3></div></div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Actual Similarity", f"{shap_result.get('actual_similarity', 0):.1%}")
        with col2:
            st.metric("Model Prediction", f"{shap_result.get('predicted_similarity', 0):.1%}")
        with col3:
            st.metric("Base Value", f"{shap_result.get('base_value', 0):.4f}")

        positive = shap_result.get('positive_contributors', [])
        if positive:
            parts = ['<div style="background:#1c2333;border-radius:12px;padding:16px;margin:12px 0;">']
            parts.append('<h4 style="color:#2ecc71;margin:0 0 12px 0;">✅ Positive Factors (Increase Similarity)</h4>')
            for item in positive[:5]:
                feature = item.get('feature', '')
                value = item.get('shap_value', 0)
                parts.append(
                    f'<div style="background:rgba(46,204,113,0.1);border-radius:6px;padding:8px 12px;'
                    f'margin:6px 0;display:flex;justify-content:space-between;">'
                    f'<span style="color:#a3a3a3;">{feature}</span>'
                    f'<span style="color:#2ecc71;font-weight:600;">+{value:.4f}</span></div>'
                )
            parts.append('</div>')
            st.markdown(''.join(parts), unsafe_allow_html=True)

        negative = shap_result.get('negative_contributors', [])
        if negative:
            parts = ['<div style="background:#1c2333;border-radius:12px;padding:16px;margin:12px 0;">']
            parts.append('<h4 style="color:#e74c3c;margin:0 0 12px 0;">❌ Negative Factors (Decrease Similarity)</h4>')
            for item in negative[:5]:
                feature = item.get('feature', '')
                value = item.get('shap_value', 0)
                parts.append(
                    f'<div style="background:rgba(231,76,60,0.1);border-radius:6px;padding:8px 12px;'
                    f'margin:6px 0;display:flex;justify-content:space-between;">'
                    f'<span style="color:#a3a3a3;">{feature}</span>'
                    f'<span style="color:#e74c3c;font-weight:600;">{value:.4f}</span></div>'
                )
            parts.append('</div>')
            st.markdown(''.join(parts), unsafe_allow_html=True)

    @staticmethod
    def render_matching_features(features: Dict[str, Any]):
        """Render matching features between two movies."""
        parts = []
        parts.append('<div style="background:#1c2333;border-radius:12px;padding:20px;margin:16px 0;">')
        parts.append('<h4 style="color:#f5c518;margin:0 0 16px 0;">🎯 Matching Features</h4>')

        if features.get('same_director'):
            parts.append(
                f'<div style="margin:8px 0;"><span style="color:#a3a3a3;">Director: </span>'
                f'<span style="color:#2ecc71;font-weight:500;">✓ {features.get("director", "Same")}</span></div>'
            )

        common_genres = features.get('common_genres', [])
        if common_genres:
            parts.append(
                f'<div style="margin:8px 0;"><span style="color:#a3a3a3;">Common Genres: </span>'
                f'<span style="color:#e5383b;font-weight:500;">{", ".join(common_genres)}</span></div>'
            )

        common_cast = features.get('common_cast', [])
        if common_cast:
            parts.append(
                f'<div style="margin:8px 0;"><span style="color:#a3a3a3;">Common Cast: </span>'
                f'<span style="color:#3498db;font-weight:500;">{", ".join(common_cast[:3])}</span></div>'
            )

        common_keywords = features.get('common_keywords', [])
        if common_keywords:
            parts.append(
                f'<div style="margin:8px 0;"><span style="color:#a3a3a3;">Common Themes: </span>'
                f'<span style="color:#9b59b6;font-weight:500;">{", ".join(common_keywords[:3])}</span></div>'
            )

        parts.append('</div>')
        st.markdown(''.join(parts), unsafe_allow_html=True)
