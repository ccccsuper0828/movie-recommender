"""
Theme configuration for the Streamlit frontend.
Cinema-inspired dark theme with clean typography.
"""
# @author 成员 F — 前端框架 & API & 测试

from pathlib import Path

# Color palette — dark cinema theme
COLORS = {
    'bg_primary': '#0d1117',
    'bg_secondary': '#161b22',
    'bg_card': '#1c2333',
    'bg_card_hover': '#262f40',
    'accent_red': '#e5383b',
    'accent_red_dark': '#ba181b',
    'accent_gold': '#f5c518',
    'accent_gold_dark': '#d4a912',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_muted': '#484f58',
    'border': '#30363d',
    'success': '#3fb950',
    'warning': '#d29922',
    'info': '#58a6ff',
}


def get_css_path() -> Path:
    return Path(__file__).parent / "main.css"


def load_css() -> str:
    css_path = get_css_path()
    if css_path.exists():
        return css_path.read_text()
    return ""


def get_page_config():
    return {
        "page_title": "CineMatch — Movie Recommendations",
        "page_icon": "🎬",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "menu_items": {
            "About": "# CineMatch\nIntelligent movie recommendations powered by ML."
        }
    }


def inject_custom_css():
    """Return the full CSS to inject via st.markdown."""
    css = load_css()

    dynamic = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    {css}

/* ── Global reset ── */
html, body, .stApp {{
    background-color: {COLORS['bg_primary']} !important;
    color: {COLORS['text_primary']};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* ── Typography ── */
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    color: {COLORS['text_primary']} !important;
    letter-spacing: -0.02em;
}}
h1 {{ font-weight: 700 !important; }}
h2 {{ font-weight: 600 !important; font-size: 1.75rem !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.25rem !important; }}
p, li, span, label {{ color: {COLORS['text_secondary']}; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton {{ display: none !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {COLORS['bg_secondary']} !important;
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] * {{
    color: {COLORS['text_primary']} !important;
    }}
[data-testid="stSidebar"] .stSlider > div > div > div {{
    background-color: {COLORS['accent_red']} !important;
}}

/* ── Buttons ── */
    .stButton > button {{
    background: linear-gradient(135deg, {COLORS['accent_red']}, {COLORS['accent_red_dark']}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 8px rgba(229,56,59,0.25) !important;
    }}
    .stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(229,56,59,0.4) !important;
    }}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {{
    background-color: {COLORS['bg_card']} !important;
    color: {COLORS['text_primary']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    }}
    .stTextInput > div > div > input:focus {{
    border-color: {COLORS['accent_red']} !important;
    box-shadow: 0 0 0 3px rgba(229,56,59,0.15) !important;
    }}

/* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {COLORS['bg_secondary']};
    padding: 6px;
        border-radius: 12px;
    border: 1px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
    background: transparent;
        border-radius: 8px;
        color: {COLORS['text_secondary']};
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.85rem;
    border: none;
    }}
    .stTabs [aria-selected="true"] {{
    background: {COLORS['accent_red']} !important;
    color: #fff !important;
    font-weight: 600;
    }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {COLORS['bg_card']};
    padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid {COLORS['border']};
    }}
[data-testid="stMetricValue"] {{
    color: {COLORS['accent_gold']} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {COLORS['text_secondary']} !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    }}

/* ── Expander ── */
    .stExpander {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
}}
.stExpander summary {{
    color: {COLORS['text_primary']} !important;
    font-weight: 500 !important;
}}

/* ── DataFrame / Table ── */
.stDataFrame, [data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
        border: 1px solid {COLORS['border']};
}}

/* ── Radio buttons (sidebar nav) ── */
.stRadio > div {{
    gap: 4px !important;
    }}
.stRadio > div > label {{
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    transition: all 0.2s ease;
    font-size: 0.9rem !important;
}}
.stRadio > div > label:hover {{
    background: {COLORS['bg_card']} !important;
    border-color: {COLORS['border']} !important;
}}
.stRadio > div > label[data-checked="true"],
.stRadio > div > label:has(input:checked) {{
    background: rgba(229,56,59,0.12) !important;
    border-color: {COLORS['accent_red']} !important;
    color: {COLORS['accent_red']} !important;
}}

/* ── Slider ── */
.stSlider > div > div > div > div {{
    background-color: {COLORS['accent_red']} !important;
    }}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{
    color: {COLORS['text_muted']} !important;
    }}

/* ── Markdown text ── */
.stMarkdown, .stMarkdown p {{
        color: {COLORS['text_primary']};
    line-height: 1.65;
}}
.stCaption, .stMarkdown .caption {{
    color: {COLORS['text_muted']} !important;
    }}

/* ── Dividers ── */
    hr {{
    border-color: {COLORS['border']} !important;
    opacity: 0.5;
    }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg_primary']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['text_muted']}; }}

/* ── Plotly chart container ── */
.js-plotly-plot, .plotly {{
    border-radius: 12px !important;
}}
[data-testid="stPlotlyChart"] {{
    min-height: 400px !important;
}}
iframe[title="streamlit_plotly_events.streamlit_plotly_events"] {{
    min-height: 400px !important;
}}

/* ── Warning / info / error ── */
.stAlert {{
    border-radius: 10px !important;
    border: none !important;
}}
</style>"""

    return dynamic


COMPONENT_STYLES = {
    'movie_card': {
        'background': COLORS['bg_card'],
        'border_radius': '14px',
        'padding': '20px 24px',
        'border': f"1px solid {COLORS['border']}",
    },
    'rating_badge': {
        'background': COLORS['accent_gold'],
        'color': '#000',
        'padding': '4px 10px',
        'border_radius': '6px',
        'font_weight': '700',
    },
    'genre_tag': {
        'background': 'rgba(229,56,59,0.12)',
        'color': COLORS['accent_red'],
        'padding': '4px 10px',
        'border_radius': '6px',
        'font_size': '0.75rem',
    },
}
