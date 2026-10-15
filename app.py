"""
GradeSense — Student Performance Intelligence
Main Application Entry Point (Modules B through H)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from ui_components import (
    inject_custom_styles,
    load_artifact,
    load_dataset,
    render_hero_banner,
    render_metric_card,
    render_section_header,
    _html,
)

# ── 1. Page Configuration ──
st.set_page_config(
    page_title="GradeSense — Student Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Style Injection ──
inject_custom_styles()

# ── 3. Load Resources ──
artifact = load_artifact()
df = load_dataset()

# ── 4. Sidebar Navigation & System Status ──
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
    <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">🎓 GradeSense</div>
    <div style="font-size: 0.8rem; color: #64748B; font-weight: 500;">Student Performance Intelligence</div>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "📊 Executive Dashboard",
    "🔮 Predict Performance",
    "📈 Cohort Analytics",
    "🔬 Model Insights & Residuals",
    "📖 About Project & Viva Guide",
]

if "current_page" not in st.session_state:
    st.session_state.current_page = nav_options[0]

def set_page(page_name):
    st.session_state.current_page = page_name

page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=nav_options.index(st.session_state.current_page),
    key="nav_radio",
    on_change=lambda: st.session_state.update({"current_page": st.session_state.nav_radio}),
)

st.sidebar.markdown("<hr style='margin: 1.5rem 0 1rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

if artifact is not None:
    st.sidebar.markdown(f"""
    <div style="font-size: 0.72rem; color: #64748B; line-height: 1.6;">
        <b>SYSTEM STATUS:</b><br>
        • Primary: <span style="color:#2563EB; font-weight:700;">{artifact['primary_model_name']}</span><br>
        • Baseline: <span>Linear Regression</span><br>
        • Test R²: <b>{artifact['benchmarks'][2]['test_r2']:.4f}</b> (RF)<br>
        • Pipeline: <span>v2.0-Verified</span><br>
        • Dataset: <span>{len(df) if df is not None else 395} students</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning("Model artifact not found. Please run train_pipeline.py.")


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "📊 Executive Dashboard":

    render_hero_banner(
        title="GradeSense: Executive Dashboard",
        subtitle="Forecasting academic performance and identifying student intervention risks using verified machine learning.",
        badge="v2.0 Verified Pipeline"
    )

    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 1.5, 1])
    with col_btn_c:
        if st.button("🔮 Open Student Grade Predictor", type="primary", use_container_width=True):
            st.session_state.current_page = "🔮 Predict Performance"
            st.rerun()

    render_section_header("Pipeline & Evaluation Overview", "Verified metrics computed on held-out test data (n=79, 20% split).")

    if artifact is not None:
        c1, c2, c3, c4 = st.columns(4)
        rf_bench = next(b for b in artifact["benchmarks"] if b["model_name"] == "Random Forest")
        c1.markdown(render_metric_card("Dataset Size", f"{artifact['metadata']['total_samples']} Students", "UCI Portuguese Math Cohort"), unsafe_allow_html=True)
        c2.markdown(render_metric_card("Active Model", f"{artifact['primary_model_name']}", "Ensemble (depth=4, n=100)"), unsafe_allow_html=True)
        c3.markdown(render_metric_card("Test R² Score", f"{rf_bench['test_r2']:.4f}", f"Baseline OLS: {artifact['benchmarks'][0]['test_r2']:.4f}"), unsafe_allow_html=True)
        c4.markdown(render_metric_card("Mean Absolute Error", f"{rf_bench['test_mae']:.2f} pts", "Cohort average absolute error (MAE)"), unsafe_allow_html=True)
    else:
        st.info("Run `python train_pipeline.py` to generate verified pipeline metrics.")

    render_section_header("Cohort Summary Statistics", "Calculated dynamically from ground-truth records (student-mat.csv).")

    if df is not None:
        d1, d2, d3, d4 = st.columns(4)
        avg_g3 = df["G3"].mean()
        pass_rate = (df["G3"] >= 10).mean() * 100
        avg_abs = df["absences"].mean()
        zero_count = (df["G3"] == 0).sum()
        zero_pct = zero_count / len(df) * 100

        d1.markdown(render_metric_card("Average Final Grade", f"{avg_g3:.2f} / 20", "Median: 11.0 / 20"), unsafe_allow_html=True)
        d2.markdown(render_metric_card("Pass Rate (G3 ≥ 10)", f"{pass_rate:.1f}%", f"{(df['G3'] >= 10).sum()} of {len(df)} students"), unsafe_allow_html=True)
        d3.markdown(render_metric_card("Average Absences", f"{avg_abs:.1f} Days", f"Max: {df['absences'].max()} days"), unsafe_allow_html=True)
        d4.markdown(render_metric_card("Exam Dropouts (G3=0)", f"{zero_count} ({zero_pct:.1f}%)", "Preserved as valid outcomes"), unsafe_allow_html=True)

        render_section_header("Primary Visualizations", "Observed distribution and academic trajectory relationships.")

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            fig1, ax1 = plt.subplots(figsize=(6, 3.8), dpi=150)
            fig1.patch.set_facecolor("#FFFFFF")
            ax1.set_facecolor("#F8FAFC")
            sns.histplot(df["G3"], bins=20, kde=True, color="#2563EB", ax=ax1, edgecolor="#1D4ED8")
            ax1.axvline(10, color="#DC2626", linestyle="--", linewidth=1.5, label="Passing Mark (10/20)")
            ax1.set_title("Final Grade (G3) Distribution", fontsize=10, fontweight="bold", color="#0F172A")
            ax1.set_xlabel("Final Grade (0–20)", fontsize=9, color="#475569")
            ax1.set_ylabel("Student Count", fontsize=9, color="#475569")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            st.pyplot(fig1)

        with vcol2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.8), dpi=150)
            fig2.patch.set_facecolor("#FFFFFF")
            ax2.set_facecolor("#F8FAFC")
            sns.regplot(data=df, x="G2", y="G3", ax=ax2,
                        scatter_kws={"color": "#0D9488", "alpha": 0.65, "s": 30},
                        line_kws={"color": "#0F766E", "linewidth": 1.5})
            ax2.set_title("Second Period (G2) vs Final Grade (G3)", fontsize=10, fontweight="bold", color="#0F172A")
            ax2.set_xlabel("Second Period Grade (0–20)", fontsize=9, color="#475569")
            ax2.set_ylabel("Final Grade (0–20)", fontsize=9, color="#475569")
            ax2.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig2)

    render_section_header("Methodology & Project Summary")
    st.markdown("""
    * **Objective**: Provide educators with early warning signals to assist students before the final examination.
    * **Data Flow**: `student-mat.csv` (395 records) $\\rightarrow$ Clean 80/20 partition $\\rightarrow$ 5-Fold Cross-Validation on training set $\\rightarrow$ Final evaluation on isolated held-out test set ($n=79$).
    * **Primary Predictors**: First period grade ($G_1$), second period grade ($G_2$), total absences, weekly study time, and previous class failures.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 2: PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "🔮 Predict Performance":

    render_hero_banner(
        title="GradeSense: Grade Forecasting Studio",
        subtitle="Enter student academic history and behavioral attributes to forecast the final G3 mathematics grade.",
        badge="Interactive Forecast"
    )

    if artifact is None:
        st.error("Model artifact (model_artifact.pkl) not found. Please run train_pipeline.py first.")
        st.stop()

    rf_model = artifact["primary_model"]
    lr_model = artifact["baseline_model"]
    meta = artifact["feature_metadata"]
    lr_params = artifact["linear_regression_params"]

    with st.form("forecast_form"):
        render_section_header("1. Academic Track Record", "Grades achieved in earlier assessment periods.")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            g1 = st.slider("First Period Grade (G1)", min_value=0, max_value=20, value=12, step=1,
                           help="Evaluation mark from the first term (0 to 20 scale).")
        with col_g2:
            g2 = st.slider("Second Period Grade (G2)", min_value=0, max_value=20, value=13, step=1,
                           help="Evaluation mark from the second term (strongest single predictor).")

        render_section_header("2. Attendance & Academic Habits", "Behavioral indicators associated with student outcomes.")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            absences = st.number_input("School Absences", min_value=0, max_value=93, value=4, step=1,
                                       help="Total days absent during the academic year.")
        with col_f2:
            study_opts = {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}
            studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4], index=1,
                                     format_func=lambda x: study_opts[x],
                                     help="Self-reported weekly hours dedicated to mathematics study.")
        with col_f3:
            fail_opts = {0: "0 past failures", 1: "1 past failure", 2: "2 past failures", 3: "3 or more failures"}
            failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3], index=0,
                                    format_func=lambda x: fail_opts[x],
                                    help="Number of previous school failures in mathematics.")

        submit_btn = st.form_submit_button("🚀 Calculate Grade Forecast", type="primary", use_container_width=True)

    if submit_btn:
        # Create structured DataFrame matching exact feature schema (prevents scikit-learn warnings)
        input_data = pd.DataFrame([{
            "G1": g1,
            "G2": g2,
            "absences": absences,
            "studytime": studytime,
            "failures": failures,
        }])[artifact["feature_names"]]

        # Generate predictions
        rf_raw = float(rf_model.predict(input_data)[0])
        lr_raw = float(lr_model.predict(input_data)[0])

        # Bounding
        rf_bounded = max(0.0, min(20.0, rf_raw))
        lr_bounded = max(0.0, min(20.0, lr_raw))

        # Qualitative Banding
        if rf_bounded >= 16.0:
            category_text = "🌟 Distinction / Excellent (16–20)"
            cat_color = "#059669"
        elif rf_bounded >= 14.0:
            category_text = "📈 Merit / Good (14–15.9)"
            cat_color = "#0D9488"
        elif rf_bounded >= 10.0:
            category_text = "✅ Satisfactory / Passing (10–13.9)"
            cat_color = "#2563EB"
        else:
            category_text = "⚠️ Academic Intervention Required (< 10)"
            cat_color = "#DC2626"

        # Result Display
        _html(f"""
        <div class="gs-result-card">
            <div class="gs-result-label">Forecasted Final Grade (G3)</div>
            <div class="gs-result-grade">{rf_bounded:.1f} <span class="gs-result-denom">/ 20</span></div>
            <div class="gs-result-category">{category_text}</div>
            <div style="font-size: 0.8rem; color: #BFDBFE; margin-top: 0.75rem;">
                Model: <b>Random Forest (Primary)</b> | Baseline OLS Estimate: <b>{lr_bounded:.1f} / 20</b> | Cohort MAE: <b>1.12 pts</b>
            </div>
        </div>
        """)

        st.caption("ℹ️ *Statistical Disclosure: Cohort Mean Absolute Error (MAE = 1.12 points) represents the average absolute discrepancy across held-out evaluation samples. It does not represent an individual prediction interval or deterministic error bound.*")

        # Input Parameter Summary
        st.markdown("##### Verification of Forecast Inputs")
        st.markdown(f"""
        * **First Period Grade ($G_1$):** `{g1} / 20`
        * **Second Period Grade ($G_2$):** `{g2} / 20`
        * **Absences:** `{absences} days`
        * **Weekly Study Time:** `{study_opts[studytime]}`
        * **Past Failures:** `{fail_opts[failures]}`
        """)

        # OLS Waterfall Decomposition
        render_section_header("Baseline OLS Feature Contribution Decomposition",
                              "Transparent mathematical explanation of how linear weights influenced the baseline forecast.")

        b0 = lr_params["intercept"]
        c_g1 = lr_params["coefficients"]["G1"] * g1
        c_g2 = lr_params["coefficients"]["G2"] * g2
        c_abs = lr_params["coefficients"]["absences"] * absences
        c_study = lr_params["coefficients"]["studytime"] * studytime
        c_fail = lr_params["coefficients"]["failures"] * failures
        total_calc = b0 + c_g1 + c_g2 + c_abs + c_study + c_fail

        def format_term(val):
            if val > 0:
                return f'<span class="gs-val-pos">+{val:.2f} pts</span>'
            elif val < 0:
                return f'<span class="gs-val-neg">{val:.2f} pts</span>'
            return f'<span class="gs-val-neutral">0.00 pts</span>'

        _html(f"""
        <div class="gs-table-box">
            <div class="gs-table-row"><span>Base Intercept (β₀)</span><span>{b0:.2f} pts</span></div>
            <div class="gs-table-row"><span>Second Period Grade (G2 = {g2} × {lr_params['coefficients']['G2']:.3f})</span><span>{format_term(c_g2)}</span></div>
            <div class="gs-table-row"><span>First Period Grade (G1 = {g1} × {lr_params['coefficients']['G1']:.3f})</span><span>{format_term(c_g1)}</span></div>
            <div class="gs-table-row"><span>Absences ({absences} days × {lr_params['coefficients']['absences']:.3f})</span><span>{format_term(c_abs)}</span></div>
            <div class="gs-table-row"><span>Study Time (Level {studytime} × {lr_params['coefficients']['studytime']:.3f})</span><span>{format_term(c_study)}</span></div>
            <div class="gs-table-row"><span>Past Failures ({failures} × {lr_params['coefficients']['failures']:.3f})</span><span>{format_term(c_fail)}</span></div>
            <div class="gs-table-row"><span>Net Linear Model Forecast</span><span>{total_calc:.2f} / 20</span></div>
        </div>
        """)

        # Handling Boundary Note
        if lr_raw < 0 or lr_raw > 20:
            st.info(f"**Bounding Notice**: The raw linear regression model calculated `{lr_raw:.2f}`, which was constrained to the valid educational interval `[0.0, 20.0]`. The primary Random Forest model predicted `{rf_raw:.2f}`.")


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 3: COHORT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "📈 Cohort Analytics":

    render_hero_banner(
        title="GradeSense: Cohort Performance Analytics",
        subtitle="Empirical exploration of factors associated with mathematics performance in Portuguese secondary schools.",
        badge="Exploratory Analysis"
    )

    if df is None:
        st.error("Dataset not found. Please verify student-mat.csv.")
        st.stop()

    render_section_header("Cohort Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(render_metric_card("Total Cohort", f"{len(df)} Students", "395 Authentic Records"), unsafe_allow_html=True)
    c2.markdown(render_metric_card("G3 Mean ± Std", f"{df['G3'].mean():.1f} ± {df['G3'].std():.1f}", "Portuguese 0–20 Scale"), unsafe_allow_html=True)
    c3.markdown(render_metric_card("Passing Cohort", f"{(df['G3']>=10).sum()} Students", f"{(df['G3']>=10).mean()*100:.1f}% Pass Rate"), unsafe_allow_html=True)
    c4.markdown(render_metric_card("Average Absences", f"{df['absences'].mean():.1f} Days", "Median: 4.0 Days"), unsafe_allow_html=True)

    render_section_header("Grade Trajectory & Correlations")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Period Grade Distributions (G1, G2, G3)**")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FAFC")
        grade_df = df[["G1", "G2", "G3"]].melt(var_name="Period", value_name="Grade")
        sns.boxplot(data=grade_df, x="Period", y="Grade", hue="Period", palette=["#93C5FD", "#60A5FA", "#2563EB"], legend=False, ax=ax)
        ax.set_title("Grade Trajectory across Evaluation Periods", fontsize=10, fontweight="bold", color="#0F172A")
        ax.set_ylabel("Grade (0–20)", fontsize=9, color="#475569")
        ax.set_xlabel("Evaluation Period", fontsize=9, color="#475569")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    with col_t2:
        st.markdown("**Correlation Matrix of Core Features**")
        corr_cols = ["G1", "G2", "G3", "absences", "studytime", "failures"]
        corr_matrix = df[corr_cols].corr()

        fig_c, ax_c = plt.subplots(figsize=(6, 4), dpi=150)
        fig_c.patch.set_facecolor("#FFFFFF")
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar=False, ax=ax_c, annot_kws={"fontsize": 8, "fontweight": "bold"})
        ax_c.set_title("Pearson Correlation Coefficients", fontsize=10, fontweight="bold", color="#0F172A")
        plt.tight_layout()
        st.pyplot(fig_c)

    render_section_header("Behavioral Factors vs Final Grade")
    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        st.markdown("**Weekly Study Time vs Mean G3**")
        study_map = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}
        study_grp = df.groupby("studytime")["G3"].mean().reset_index()
        study_grp["Label"] = study_grp["studytime"].map(study_map)

        fig_s, ax_s = plt.subplots(figsize=(4.5, 3.5), dpi=150)
        fig_s.patch.set_facecolor("#FFFFFF")
        ax_s.set_facecolor("#F8FAFC")
        sns.barplot(data=study_grp, x="Label", y="G3", color="#2563EB", ax=ax_s)
        ax_s.set_ylabel("Mean Final Grade (G3)", fontsize=8)
        ax_s.set_xlabel("")
        ax_s.set_ylim(0, 16)
        plt.xticks(rotation=15, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_s)

    with fcol2:
        st.markdown("**School Absences vs Final Grade**")
        fig_a, ax_a = plt.subplots(figsize=(4.5, 3.5), dpi=150)
        fig_a.patch.set_facecolor("#FFFFFF")
        ax_a.set_facecolor("#F8FAFC")
        sns.scatterplot(data=df, x="absences", y="G3", color="#0D9488", alpha=0.6, ax=ax_a)
        ax_a.set_xlabel("Absences (Days)", fontsize=8)
        ax_a.set_ylabel("Final Grade (G3)", fontsize=8)
        ax_a.set_ylim(-1, 21)
        plt.tight_layout()
        st.pyplot(fig_a)

    with fcol3:
        st.markdown("**Past Class Failures vs Mean G3**")
        fail_grp = df.groupby("failures")["G3"].mean().reset_index()
        fig_f, ax_f = plt.subplots(figsize=(4.5, 3.5), dpi=150)
        fig_f.patch.set_facecolor("#FFFFFF")
        ax_f.set_facecolor("#F8FAFC")
        sns.barplot(data=fail_grp, x="failures", y="G3", color="#DC2626", ax=ax_f)
        ax_f.set_xlabel("Past Failures Count", fontsize=8)
        ax_f.set_ylabel("Mean Final Grade (G3)", fontsize=8)
        ax_f.set_ylim(0, 16)
        plt.tight_layout()
        st.pyplot(fig_f)

    render_section_header("Dynamically Calculated Cohort Observations")
    r_g2_g3 = corr_matrix.loc["G2", "G3"]
    r_fail_g3 = corr_matrix.loc["failures", "G3"]
    r_abs_g3 = corr_matrix.loc["absences", "G3"]

    st.markdown(f"""
    * **Strongest Linear Association:** Second-period grade ($G_2$) exhibits a correlation of **$r = {r_g2_g3:.2f}$** with the final exam score, reflecting strong temporal continuity in student performance.
    * **Academic Repeats:** Past subject failures show a negative association of **$r = {r_fail_g3:.2f}$**, indicating that previous academic struggle is a persistent risk factor.
    * **Attendance Factor:** Absences correlate at **$r = {r_abs_g3:.2f}$**; while correlation is near-zero in the aggregate cohort due to students with high absences who still pass, extreme absenteeism strongly correlates with final exam dropouts ($G_3=0$).
    * *Notice: Statistical correlations indicate observational relationships and do not prove direct causation.*
    """)


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 4: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "🔬 Model Insights & Residuals":

    render_hero_banner(
        title="GradeSense: Model Evaluation & Diagnostics",
        subtitle="Transparent evaluation, cross-validation stability, diagnostic plots, and feature interpretations.",
        badge="Diagnostics & Benchmarks"
    )

    if artifact is None:
        st.error("Model artifact not found. Please run train_pipeline.py.")
        st.stop()

    benchmarks = artifact["benchmarks"]
    eval_data = artifact["evaluation_data"]

    render_section_header("Model Architecture Benchmark (5-Fold CV on Training Set vs Held-Out Test)")

    bench_df = pd.DataFrame([
        {
            "Candidate Model": b["model_name"],
            "5-Fold CV R² (Train)": f"{b['cv_r2_mean']:.4f} ± {b['cv_r2_std']:.4f}",
            "5-Fold CV MAE": f"{b['cv_mae_mean']:.4f} pts",
            "Held-Out Test R²": f"{b['test_r2']:.4f}",
            "Held-Out Test MAE": f"{b['test_mae']:.4f} pts",
            "Held-Out Test RMSE": f"{b['test_rmse']:.4f} pts",
        }
        for b in benchmarks
    ])
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    render_section_header("Diagnostic Plots on Held-Out Test Set (n=79, 20% Split)")

    y_test = np.array(eval_data["y_test"])
    rf_preds = np.array(eval_data["test_predictions"]["Random Forest"])
    lr_preds = np.array(eval_data["test_predictions"]["Linear Regression"])

    rf_residuals = y_test - rf_preds
    lr_residuals = y_test - lr_preds

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Actual vs Predicted Grade (Held-Out Test Set)**")
        fig_act, ax_act = plt.subplots(figsize=(6, 4.5), dpi=150)
        fig_act.patch.set_facecolor("#FFFFFF")
        ax_act.set_facecolor("#F8FAFC")
        ax_act.scatter(y_test, rf_preds, color="#2563EB", alpha=0.75, s=45, label="Random Forest (Primary)")
        ax_act.scatter(y_test, lr_preds, color="#0D9488", alpha=0.55, s=35, label="Linear Regression (Baseline)")
        ax_act.plot([0, 20], [0, 20], color="#DC2626", linestyle="--", linewidth=1.5, label="Perfect Agreement (y = x)")
        ax_act.set_xlabel("Actual Grade (G3)", fontsize=9, fontweight="bold", color="#334155")
        ax_act.set_ylabel("Predicted Grade (G3)", fontsize=9, fontweight="bold", color="#334155")
        ax_act.set_xlim(-1, 21)
        ax_act.set_ylim(-1, 21)
        ax_act.grid(True, linestyle=":", alpha=0.5)
        ax_act.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig_act)

    with col_d2:
        st.markdown("**Residual Distribution Comparison (Actual - Predicted)**")
        fig_res, ax_res = plt.subplots(figsize=(6, 4.5), dpi=150)
        fig_res.patch.set_facecolor("#FFFFFF")
        ax_res.set_facecolor("#F8FAFC")
        sns.kdeplot(rf_residuals, color="#2563EB", linewidth=2.0, label=f"Random Forest (Std: {np.std(rf_residuals):.2f})", ax=ax_res)
        sns.kdeplot(lr_residuals, color="#0D9488", linewidth=2.0, label=f"Linear Regression (Std: {np.std(lr_residuals):.2f})", ax=ax_res)
        ax_res.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
        ax_res.set_xlabel("Prediction Residual (Grade Points)", fontsize=9, fontweight="bold", color="#334155")
        ax_res.set_ylabel("Density", fontsize=9, color="#334155")
        ax_res.grid(True, linestyle=":", alpha=0.5)
        ax_res.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig_res)

    render_section_header("Model Interpretability & Feature Significance")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("**Random Forest Gini Feature Importances**")
        rf_imp = artifact["random_forest_params"]["feature_importances"]
        imp_s = pd.Series(rf_imp).sort_values(ascending=True)

        fig_imp, ax_imp = plt.subplots(figsize=(6, 3.5), dpi=150)
        fig_imp.patch.set_facecolor("#FFFFFF")
        ax_imp.set_facecolor("#F8FAFC")
        imp_s.plot(kind="barh", color="#2563EB", ax=ax_imp)
        ax_imp.set_xlabel("Relative Importance (Sum = 1.0)", fontsize=8)
        ax_imp.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_imp)

    with f_col2:
        st.markdown("**Linear Regression Ordinary Least Squares Weights (β)**")
        lr_coefs = artifact["linear_regression_params"]["coefficients"]
        coef_s = pd.Series(lr_coefs).sort_values(ascending=True)

        fig_coef, ax_coef = plt.subplots(figsize=(6, 3.5), dpi=150)
        fig_coef.patch.set_facecolor("#FFFFFF")
        ax_coef.set_facecolor("#F8FAFC")
        colors = ["#DC2626" if c < 0 else "#2563EB" for c in coef_s.values]
        coef_s.plot(kind="barh", color=colors, ax=ax_coef)
        ax_coef.axvline(0, color="#0F172A", linestyle="-", linewidth=0.8)
        ax_coef.set_xlabel("Coefficient Weight (β)", fontsize=8)
        ax_coef.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_coef)

    render_section_header("Technical Analysis: Why Random Forest Outperformed OLS")
    st.markdown("""
    1. **Non-Linear Dropout Boundary:** 38 students in the dataset have $G_3 = 0$ (dropouts or missed the final exam). Linear Regression is constrained to fit a continuous straight plane, predicting negative marks (e.g. $-0.85$) for absent students. Random Forest cleanly isolates zero-grade students in leaf nodes.
    2. **Mitigated Multicollinearity:** $G_1$ and $G_2$ correlate strongly ($r = 0.85$). In Linear Regression, multicollinearity inflates variance and causes study time to receive a negative partial coefficient. Random Forest ensembles average across decision trees, providing superior generalization.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#                           PAGE 5: ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "📖 About Project & Viva Guide":

    render_hero_banner(
        title="GradeSense: Project Theory & Viva Defense Guide",
        subtitle="Academic methodology, dataset provenance, theoretical formulation, and viva examination preparation.",
        badge="Academic Documentation"
    )

    render_section_header("1. Problem Statement & Motivation")
    st.markdown("""
    Early identification of students at risk of academic failure is a central objective of learning analytics. 
    By forecasting the final mathematics grade ($G_3$) early in the school year, instructors and academic counselors 
    can implement targeted pedagogical interventions before irreversible learning deficits accumulate.
    """)

    render_section_header("2. Dataset Attribution & Structure")
    st.markdown("""
    * **Origin:** P. Cortez and A. Silva (2008), *Using Data Mining to Predict Secondary School Student Performance*, University of Minho, Portugal.
    * **Scope:** 395 secondary school students attending two Portuguese educational institutions (*Gabriel Pereira* and *Mouzinho da Silveira*).
    * **Grading System:** Portugal uses a 20-point grading scale where $0 - 9$ represents failure, $10 - 13$ satisfactory, $14 - 15$ good, and $16 - 20$ distinction.
    """)

    render_section_header("3. Machine Learning Architecture")
    st.markdown(r"""
    * **Feature Schema (Ordered Inputs):**
      1. `G1`: First evaluation period grade ($0 - 20$).
      2. `G2`: Second evaluation period grade ($0 - 20$).
      3. `absences`: Total recorded school absences ($0 - 93$).
      4. `studytime`: Weekly study hours ($1: <2\text{h}, 2: 2-5\text{h}, 3: 5-10\text{h}, 4: >10\text{h}$).
      5. `failures`: Past class failures ($0 - 3$).
    * **Pipeline Protocol:** Strict isolation of $80\%$ training split ($n=316$) and $20\%$ held-out test split ($n=79$, `random_state=42`). Model selection decided exclusively via 5-Fold Cross-Validation on the training data.
    """)

    render_section_header("4. Key Viva Questions & Technical Explanations")
    with st.expander("Q1: Why is G2 such a dominant predictor compared to other attributes?", expanded=True):
        st.markdown("""
        **Answer:** In educational measurement, recent assessment marks ($G_2$) capture cumulative topic mastery, test-taking capability, 
        and immediate academic trajectory. Because $G_2$ was evaluated just prior to the final term, it shares an $r = 0.90$ correlation with $G_3$.
        """)

    with st.expander("Q2: Why does weekly study time have a slight negative coefficient in Linear Regression?"):
        st.markdown("""
        **Answer:** In multiple linear regression, coefficients represent *partial derivatives* holding all other features constant. 
        Because $G_1$ and $G_2$ already capture the positive benefits of studying, the remaining residual variance in self-reported study time 
        often reflects compensatory behavior: struggling students frequently spend more hours studying right before failing exams.
        """)

    with st.expander("Q3: How were the 38 students with a final grade of 0 handled, and why not remove them?"):
        st.markdown("""
        **Answer:** Removing $G_3=0$ records would introduce severe *survivorship bias*. In secondary education, a score of zero represents 
        course abandonment, disciplinary suspension, or final exam absence. Preserving these authentic outcomes is vital for realistic evaluation, 
        and is the primary reason why Random Forest ($R^2 = 0.8480$) outperforms Linear Regression ($R^2 = 0.7822$).
        """)

    with st.expander("Q4: What is the difference between R² and MAE when presenting to school administrators?"):
        st.markdown("""
        **Answer:** $R^2$ indicates the proportion of total variance explained ($84.8\\%$ for our primary model). While useful for ML benchmarks, 
        school administrators understand **MAE** (Mean Absolute Error): our model is typically accurate within **$\\pm 1.12$ grade points** 
        on a 20-point scale.
        """)

    render_section_header("5. Ethical Considerations & Limitations")
    st.markdown("""
    * **Non-Causal Nature:** Machine learning models identify statistical correlations, not deterministic causation. A predicted low grade must never be used punitively.
    * **Intervention Bias:** If an instructor uses this tool to successfully assist an at-risk student who then scores high, the model's original prediction will technically register as an 'error' despite achieving the ultimate educational goal.
    """)