import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Helper: inject raw HTML (st.html is the correct API in Streamlit ≥ 1.32) ──
def _html(content: str):
    try:
        st.html(content)
    except AttributeError:
        st.markdown(content, unsafe_allow_html=True)


# ─── 1. PAGE CONFIGURATION ────────────────────────────────────────────────────
st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

# ─── 2. INJECT STYLES ─────────────────────────────────────────────────────────
_html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>

/* ── Global font ── */
html, body, [class*="css"], * {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

/* ── App / page background ── */
.stApp { background-color: #F4F5F7 !important; }
.main > .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 100% !important;
    background-color: #F4F5F7 !important;
}

/* ── Top header bar ── */
header[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #E5E7EB !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.05) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
    color: #111827 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: -0.02em !important;
}
section[data-testid="stSidebar"] p {
    color: #9CA3AF !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}
/* Nav item labels */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #4B5563 !important;
    cursor: pointer !important;
    margin-bottom: 2px !important;
    transition: background-color 0.15s ease !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #F3F0FD !important;
    color: #6C5CE7 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background-color: #EDE9FE !important;
    color: #6C5CE7 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* ── Headings ── */
h1 {
    color: #111827 !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    margin-bottom: 0.1rem !important;
}
h2, h3 { color: #1F2937 !important; font-weight: 700 !important; }
h4, h5 { color: #374151 !important; font-weight: 600 !important; }

/* ── Subtitle ── */
.subtitle {
    font-size: 0.93rem !important;
    color: #6B7280 !important;
    font-weight: 400 !important;
    margin-bottom: 1.75rem !important;
}

/* ── Section divider ── */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    margin-top: 1.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.55rem;
    border-bottom: 1.5px solid #E5E7EB;
    letter-spacing: -0.01em;
}

/* ── Metric cards ── */
.dash-card {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.3rem 1.1rem 1.3rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04) !important;
    position: relative !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
.dash-card:hover {
    box-shadow: 0 6px 24px rgba(108,92,231,0.15), 0 1px 6px rgba(0,0,0,0.05) !important;
    transform: translateY(-2px) !important;
}
.dash-card-label {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    color: #9CA3AF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 0.45rem !important;
}
.dash-card-value {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
    margin: 0 0 0.85rem 0 !important;
    white-space: nowrap !important;
    letter-spacing: -0.025em !important;
    line-height: 1 !important;
}
.dash-card-bar-track {
    height: 4px !important;
    background: #F3F4F6 !important;
    border-radius: 99px !important;
    overflow: hidden !important;
}
.dash-card-bar-fill {
    height: 100% !important;
    border-radius: 99px !important;
    animation: growBar 1s cubic-bezier(.4,0,.2,1) forwards !important;
}
@keyframes growBar {
    from { width: 0%; }
    to   { width: var(--bar-width, 60%); }
}
.bar-purple { background: linear-gradient(90deg, #6C5CE7, #A78BFA) !important; }
.bar-coral  { background: linear-gradient(90deg, #F97316, #FDBA74) !important; }
.bar-teal   { background: linear-gradient(90deg, #0D9488, #5EEAD4) !important; }
.bar-amber  { background: linear-gradient(90deg, #D97706, #FDE68A) !important; }

/* ── Predictor result card ── */
.result-card {
    background: linear-gradient(135deg, #6C5CE7 0%, #8B5CF6 100%) !important;
    border-radius: 18px !important;
    padding: 2.5rem 2rem !important;
    text-align: center !important;
    box-shadow: 0 12px 40px rgba(108,92,231,0.4) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    position: relative !important;
    overflow: hidden !important;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.result-card::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.result-label {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    color: rgba(255,255,255,0.6) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    margin-bottom: 0.5rem !important;
}
.result-grade {
    font-size: 4.5rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.04em !important;
    text-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
}
.result-sub {
    font-size: 1.8rem !important;
    opacity: 0.55 !important;
    color: #fff !important;
}
.result-category {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.88) !important;
    margin-top: 0.6rem !important;
}

/* ── Input summary ── */
.input-summary {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    font-size: 0.88rem !important;
    color: #4B5563 !important;
    line-height: 1.9 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6C5CE7, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(108,92,231,0.45) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6C5CE7, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(108,92,231,0.4) !important;
}

/* ── Inputs ── */
input, textarea {
    border-radius: 8px !important;
    border: 1.5px solid #E5E7EB !important;
    background: #FFFFFF !important;
    color: #111827 !important;
}
input:focus { border-color: #6C5CE7 !important; box-shadow: 0 0 0 3px rgba(108,92,231,0.15) !important; }
[data-baseweb="select"] { border-radius: 8px !important; }

/* ── Misc ── */
hr { border-color: #E5E7EB !important; margin: 1.5rem 0 !important; }
.stAlert { border-radius: 10px !important; }

</style>
""")

# ─── Card helpers ──────────────────────────────────────────────────────────────
_BAR_COLORS = ["bar-purple", "bar-coral", "bar-teal", "bar-amber"]
_BAR_WIDTHS = ["67%", "45%", "78%", "80%"]

def render_card(title, value, idx=0):
    bar_cls   = _BAR_COLORS[idx % 4]
    bar_width = _BAR_WIDTHS[idx % 4]
    return f"""
    <div class="dash-card">
        <div class="dash-card-label">{title}</div>
        <div class="dash-card-value">{value}</div>
        <div class="dash-card-bar-track">
            <div class="dash-card-bar-fill {bar_cls}" style="--bar-width:{bar_width};"></div>
        </div>
    </div>"""


# ─── 3. NAVIGATION ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

def go_to_predict():
    st.session_state.page = "Predictor"

st.sidebar.title("Navigation")
st.sidebar.markdown("Explore the dataset insights or jump to the ML prediction tool.")
page = st.sidebar.radio("Go to", ["Dashboard", "Analytics", "Predictor"], key="page")


# ─── 4. CACHED RESOURCES ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('student_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("student-mat.csv")
    except FileNotFoundError:
        return None

model = load_model()
df    = load_data()


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Dashboard":

    # Hero
    _html("<h1 style='text-align:center;'>Student Performance AI</h1>")
    _html('<p class="subtitle" style="text-align:center;">Predict and understand student academic performance using Machine Learning.</p>')

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        st.button("🔮 Predict Student Performance", type="primary",
                  use_container_width=True, on_click=go_to_predict)

    _html("<hr>")

    # Model overview cards
    _html('<div class="section-header">Model Overview</div>')
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(render_card("Dataset Size",   "395",               idx=0), unsafe_allow_html=True)
    c2.markdown(render_card("Model",          "Linear Regression", idx=1), unsafe_allow_html=True)
    c3.markdown(render_card("R² Score",       "0.78",              idx=2), unsafe_allow_html=True)
    c4.markdown(render_card("MAE",            "1.34",              idx=3), unsafe_allow_html=True)

    _html("<hr>")

    # Dataset insights
    _html('<div class="section-header">Dataset Insights</div>')

    if df is not None:
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.markdown(render_card("Total Students",      f"{len(df)}",                       idx=0), unsafe_allow_html=True)
        ic2.markdown(render_card("Avg Final Grade (G3)", f"{df['G3'].mean():.1f} / 20",     idx=1), unsafe_allow_html=True)
        ic3.markdown(render_card("Average Absences",    f"{df['absences'].mean():.1f}",     idx=2), unsafe_allow_html=True)
        ic4.markdown(render_card("Avg Study Time",      f"{df['studytime'].mean():.1f}",    idx=3), unsafe_allow_html=True)

        st.write("")

        _html('<div class="section-header">Visualizations</div>')
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("**Final Grade (G3) Distribution**")
            st.bar_chart(df['G3'].value_counts().sort_index())
        with vcol2:
            st.markdown("**First Period (G1) vs Final Grade (G3)**")
            st.scatter_chart(df, x="G1", y="G3")

        st.markdown("**Second Period (G2) vs Final Grade (G3)**")
        st.scatter_chart(df, x="G2", y="G3")
    else:
        st.warning("student-mat.csv not found for dataset insights.")

    _html("<hr>")
    st.markdown("### Project Summary")
    st.info("This application uses Linear Regression to estimate a student's final mathematics grade (G3) using previous period grades, absences, study time, and previous failures.")


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Analytics":

    if df is None:
        st.error("Error: Could not find student-mat.csv.")
        st.stop()

    _html("<h1>Student Analytics</h1>")
    _html('<p class="subtitle">Explore the factors associated with student mathematics performance.</p>')

    _html('<div class="section-header">Dataset Metrics</div>')
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.markdown(render_card("Total Students",       f"{len(df)}",                    idx=0), unsafe_allow_html=True)
    ic2.markdown(render_card("Avg Final Grade (G3)", f"{df['G3'].mean():.1f} / 20",   idx=1), unsafe_allow_html=True)
    ic3.markdown(render_card("Average Absences",     f"{df['absences'].mean():.1f}",  idx=2), unsafe_allow_html=True)
    ic4.markdown(render_card("Avg Study Time",       f"{df['studytime'].mean():.1f}", idx=3), unsafe_allow_html=True)

    _html('<div class="section-header">Grade Performance</div>')
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        st.markdown("**Final Grade (G3) Distribution**")
        st.bar_chart(df['G3'].value_counts().sort_index())
    with vcol2:
        st.markdown("**G1 vs Final Grade (G3)**")
        st.scatter_chart(df, x="G1", y="G3")
    with vcol3:
        st.markdown("**G2 vs Final Grade (G3)**")
        st.scatter_chart(df, x="G2", y="G3")

    _html('<div class="section-header">Factors Associated With Final Grade</div>')
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown("**Study Time vs Final Grade**")
        study_map = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}
        study_g3 = df.groupby('studytime')['G3'].mean().reset_index()
        study_g3['Study Time'] = study_g3['studytime'].map(study_map)
        st.bar_chart(study_g3.set_index('Study Time')['G3'])
    with fcol2:
        st.markdown("**Absences vs Final Grade**")
        st.scatter_chart(df, x="absences", y="G3")
    with fcol3:
        st.markdown("**Previous Failures vs Final Grade**")
        st.bar_chart(df.groupby('failures')['G3'].mean())

    _html('<div class="section-header">Correlation Analysis</div>')
    corr_cols   = ['G1', 'G2', 'G3', 'absences', 'studytime', 'failures']
    corr_matrix = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f",
                linewidths=.5, ax=ax, cbar=False)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
    with col_h2:
        st.pyplot(fig)

    _html('<div class="section-header">Key Insights</div>')
    corr_g1_g3       = corr_matrix.loc['G1', 'G3']
    corr_g2_g3       = corr_matrix.loc['G2', 'G3']
    corr_absences_g3 = corr_matrix.loc['absences', 'G3']
    corr_failures_g3 = corr_matrix.loc['failures', 'G3']
    insights = []
    if corr_g1_g3 > 0.6 and corr_g2_g3 > 0.6:
        insights.append("- **Academic History**: G1 and G2 show a strong positive relationship with G3.")
    elif corr_g1_g3 > 0.3 or corr_g2_g3 > 0.3:
        insights.append("- **Academic History**: Previous period grades (G1, G2) are positively associated with G3.")
    if corr_failures_g3 < -0.2:
        insights.append("- **Past Failures**: More past failures is associated with lower final grades.")
    if abs(corr_absences_g3) > 0.01:
        direction = "negative" if corr_absences_g3 < 0 else "positive"
        insights.append(f"- **Attendance**: Absences show a weak {direction} relationship with final performance.")
    if not insights:
        insights.append("- No strong linear correlations identified among primary input variables.")
    for insight in insights:
        st.markdown(insight)


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 3 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Predictor":

    if model is None:
        st.error("Error: Could not find student_model.pkl.")
        st.stop()

    _html("<h1>Predict Student Grade</h1>")
    _html('<p class="subtitle">Enter the academic details below to predict the final G3 grade.</p>')

    with st.form("prediction_form"):
        _html('<div class="section-header">Academic Performance</div>')
        col1, col2 = st.columns(2)
        with col1:
            g1 = st.number_input("First Period Grade (G1) [0-20]",  min_value=0, max_value=20, value=12)
        with col2:
            g2 = st.number_input("Second Period Grade (G2) [0-20]", min_value=0, max_value=20, value=13)

        _html('<div class="section-header">Attendance &amp; Study</div>')
        col3, col4, col5 = st.columns(3)
        with col3:
            absences = st.number_input("Absences", min_value=0, max_value=100, value=2)
        with col4:
            failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3, 4])
        with col5:
            study_options = {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}
            studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4],
                                     format_func=lambda x: study_options[x])
        st.write("")
        submit = st.form_submit_button("Predict Final Grade", type="primary", use_container_width=True)

    if submit:
        prediction  = model.predict(np.array([[g1, g2, absences, studytime, failures]]))[0]
        final_grade = max(0, min(20, prediction))

        if final_grade >= 16:
            category = "🌟 Excellent Performance"
        elif final_grade >= 14:
            category = "📈 Good Performance"
        elif final_grade >= 10:
            category = "✅ Satisfactory"
        else:
            category = "⚠️ Needs Improvement"

        _html(f"""
        <div class="result-card">
            <p class="result-label">Predicted Final Grade (G3)</p>
            <h2 class="result-grade">
                {final_grade:.1f}
                <span class="result-sub">/ 20</span>
            </h2>
            <p class="result-category">{category}</p>
        </div>""")

        st.markdown("##### Prediction Inputs")
        _html(f"""
        <div class="input-summary">
            <b>G1:</b> {g1} &nbsp;|&nbsp;
            <b>G2:</b> {g2} &nbsp;|&nbsp;
            <b>Absences:</b> {absences} &nbsp;|&nbsp;
            <b>Past Failures:</b> {failures} &nbsp;|&nbsp;
            <b>Study Time:</b> {study_options[studytime]}
        </div>""")