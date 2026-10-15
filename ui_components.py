"""
GradeSense — Shared UI Components & Design System (Module B & C)

Provides reusable visual styling, layout helpers, and data/model loaders
for the GradeSense Streamlit application.
"""

from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student-mat.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifact.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


def _html(content: str):
    """Safely render raw HTML in Streamlit."""
    try:
        st.html(content)
    except AttributeError:
        st.markdown(content, unsafe_allow_html=True)


def inject_custom_styles():
    """Inject cohesive, clean light-theme CSS based on the GradeSense design system."""
    _html("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    /* ── Global Typography & Reset ── */
    html, body, [class*="css"], * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        box-sizing: border-box;
    }

    /* ── App Background ── */
    .stApp {
        background-color: #F8FAFC !important;
    }
    .main > .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem !important;
        max-width: 1200px !important;
    }

    /* ── Header Bar ── */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 1px 0 6px rgba(15, 23, 42, 0.03) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }

    /* ── Headings ── */
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.2rem !important;
    }
    h2 {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
        letter-spacing: -0.02em !important;
    }
    h3 {
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    p, span, label {
        color: #334155 !important;
    }

    /* ── Hero Banner ── */
    .gs-hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .gs-hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin: 0;
    }
    .gs-hero-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }
    .gs-badge-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #2563EB;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.65rem;
        border-radius: 99px;
        border: 1px solid #BFDBFE;
        letter-spacing: 0.02em;
    }

    /* ── Metric Cards ── */
    .gs-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 1.1rem 1.15rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        height: 100% !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .gs-card:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.08) !important;
    }
    .gs-card-label {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-bottom: 0.35rem !important;
    }
    .gs-card-value {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
        margin-bottom: 0.35rem !important;
    }
    .gs-card-sub {
        font-size: 0.78rem !important;
        color: #64748B !important;
        margin: 0 !important;
    }

    /* ── Section Dividers ── */
    .gs-section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1.5rem;
        margin-bottom: 0.9rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid #E2E8F0;
        letter-spacing: -0.01em;
    }

    /* ── Result Card ── */
    .gs-result-card {
        background: linear-gradient(145deg, #1E3A8A 0%, #1E40AF 100%) !important;
        border-radius: 14px !important;
        padding: 2rem 1.5rem !important;
        text-align: center !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.25) !important;
        margin-top: 1rem !important;
        margin-bottom: 1.25rem !important;
    }
    .gs-result-label {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #93C5FD !important;
        margin-bottom: 0.35rem !important;
    }
    .gs-result-grade {
        font-size: 3.8rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.03em !important;
        line-height: 1 !important;
        margin: 0 0 0.5rem 0 !important;
    }
    .gs-result-denom {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        opacity: 0.7 !important;
    }
    .gs-result-category {
        display: inline-block !important;
        background: rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        padding: 0.35rem 0.9rem !important;
        border-radius: 99px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        margin-top: 0.2rem !important;
    }

    /* ── Waterfall Explainer Table ── */
    .gs-table-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.8rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .gs-table-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.85rem;
    }
    .gs-table-row:last-child {
        border-bottom: none;
        font-weight: 700;
        padding-top: 0.6rem;
    }
    .gs-val-pos { color: #2563EB; font-weight: 600; }
    .gs-val-neg { color: #DC2626; font-weight: 600; }
    .gs-val-neutral { color: #64748B; font-weight: 600; }

    /* ── Buttons ── */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: #1D4ED8 !important;
    }
    </style>
    """)


@st.cache_resource
def load_artifact():
    """Load the serialized model artifact package."""
    if not ARTIFACT_PATH.exists():
        return None
    try:
        with open(ARTIFACT_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None


@st.cache_data
def load_dataset():
    """Load and return the ground-truth student dataset."""
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def render_metric_card(label: str, value: str, sub: str = None):
    """Return HTML for a clean GradeSense metric card."""
    sub_html = f'<div class="gs-card-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="gs-card">
        <div class="gs-card-label">{label}</div>
        <div class="gs-card-value">{value}</div>
        {sub_html}
    </div>
    """


def render_section_header(title: str, subtitle: str = None):
    """Render a clean section header with optional description."""
    sub_html = f'<p style="color:#64748B; font-size:0.85rem; margin-top:-0.5rem; margin-bottom:1rem;">{subtitle}</p>' if subtitle else ""
    _html(f'<div class="gs-section-header">{title}</div>{sub_html}')


def render_hero_banner(title: str, subtitle: str, badge: str = "v2.0 Verified Pipeline"):
    """Render the top hero banner."""
    _html(f"""
    <div class="gs-hero">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <h1 class="gs-hero-title">{title}</h1>
                <div class="gs-hero-subtitle">{subtitle}</div>
            </div>
            <span class="gs-badge-pill">{badge}</span>
        </div>
    </div>
    """)
