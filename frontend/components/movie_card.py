"""
Movie card component for displaying movie information.
"""

import streamlit as st
from typing import Dict, List, Any, Optional


class MovieCard:
    """
    A custom movie card component with cinema-inspired styling.
    """

    @staticmethod
    def render(
        title: str,
        genres: List[str],
        rating: float,
        similarity_score: Optional[float] = None,
        director: Optional[str] = None,
        year: Optional[str] = None,
        overview: Optional[str] = None,
        explanation: Optional[str] = None,
        rank: Optional[int] = None,
        show_explanation: bool = True
    ):
        """Render a movie card as a single self-contained HTML block."""
        with st.container():
            # Build entire card as one HTML string
            parts = []
            parts.append('<div style="background:#1c2333;border-radius:14px;padding:20px 24px;margin:12px 0;border:1px solid #30363d;position:relative;">')

            # Rank badge
            if rank is not None:
                parts.append(
                    f'<div style="position:absolute;top:-10px;left:20px;background:linear-gradient(135deg,#e5383b 0%,#ba181b 100%);'
                    f'color:white;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;'
                    f'justify-content:center;font-weight:bold;font-size:14px;box-shadow:0 4px 12px rgba(229,9,20,0.4);">#{rank}</div>'
                )

            # Title row start
            parts.append('<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">')
            parts.append('<div>')
            parts.append(f'<h3 style="color:#ffffff;margin:0 0 8px 0;font-size:1.2rem;font-weight:600;">{title}</h3>')

            # Year and director meta
            meta_parts = []
            if year:
                meta_parts.append(year[:4] if len(str(year)) >= 4 else str(year))
            if director:
                meta_parts.append(f'🎬 {director}')
            if meta_parts:
                parts.append(f'<p style="color:#a3a3a3;margin:0;font-size:0.9rem;">{" • ".join(meta_parts)}</p>')

            parts.append('</div>')

            # Rating and similarity badges
            parts.append('<div style="display:flex;gap:12px;align-items:center;">')
            parts.append(
                f'<div style="background:#f5c518;color:#000;padding:4px 10px;border-radius:4px;'
                f'font-weight:700;font-size:0.85rem;display:flex;align-items:center;gap:4px;">'
                f'<span>★</span> {rating:.1f}</div>'
            )

            if similarity_score is not None:
                score_pct = similarity_score * 100
                color = '#2ecc71' if score_pct >= 70 else '#f5c518' if score_pct >= 40 else '#e5383b'
                parts.append(
                    f'<div style="width:50px;height:50px;border-radius:50%;'
                    f'background:conic-gradient({color} {score_pct * 3.6}deg,#30363d 0deg);'
                    f'display:flex;align-items:center;justify-content:center;">'
                    f'<div style="width:40px;height:40px;border-radius:50%;background:#1c2333;'
                    f'display:flex;align-items:center;justify-content:center;font-size:0.75rem;'
                    f'font-weight:700;color:#ffffff;">{score_pct:.0f}%</div></div>'
                )

            parts.append('</div></div>')

            # Genres
            if genres:
                parts.append('<div style="margin:12px 0;display:flex;flex-wrap:wrap;gap:6px;">')
                for genre in (genres if isinstance(genres, list) else [])[:4]:
                    parts.append(
                        f'<span style="background:rgba(229,9,20,0.15);color:#e5383b;padding:4px 10px;'
                        f'border-radius:4px;font-size:0.75rem;font-weight:500;text-transform:uppercase;'
                        f'letter-spacing:0.5px;border:1px solid rgba(229,9,20,0.3);">{genre}</span>'
                    )
                parts.append('</div>')

            # Overview
            if overview:
                truncated = (str(overview)[:150] + "...") if len(str(overview)) > 150 else str(overview)
                parts.append(f'<p style="color:#a3a3a3;font-size:0.9rem;line-height:1.5;margin:12px 0;">{truncated}</p>')

            parts.append('</div>')

            # Render as single block
            st.markdown(''.join(parts), unsafe_allow_html=True)

            # Explanation (using native Streamlit expander)
            if show_explanation and explanation:
                with st.expander("💡 Why this recommendation?", expanded=False):
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#1a1a24 0%,rgba(245,197,24,0.05) 100%);'
                        f'padding:12px;border-radius:8px;border-left:3px solid #f5c518;">'
                        f'<p style="color:#a3a3a3;margin:0;">{explanation}</p></div>',
                        unsafe_allow_html=True
                    )

    @staticmethod
    def render_compact(
        title: str,
        similarity_score: float,
        rating: float
    ):
        """Render a compact version of the movie card."""
        st.markdown(
            f'<div style="background:#1c2333;border-radius:8px;padding:12px 16px;margin:8px 0;'
            f'border:1px solid #30363d;display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:#ffffff;font-weight:500;">{title}</span>'
            f'<div style="display:flex;gap:12px;align-items:center;">'
            f'<span style="color:#f5c518;">★ {rating:.1f}</span>'
            f'<span style="background:#e5383b;color:white;padding:4px 8px;border-radius:4px;'
            f'font-size:0.8rem;">{similarity_score:.0%}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )
