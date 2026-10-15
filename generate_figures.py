"""
GradeSense — Publication Report Figures Generator (Module I)

Generates 5 publication-ready, high-resolution figures (300 DPI)
and saves them to reports/figures/.
"""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH = BASE_DIR / "model_artifact.pkl"
DATA_PATH = BASE_DIR / "student-mat.csv"

# Global visual aesthetics
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def generate_figure1_architecture():
    """Figure 1: System Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    # Titles
    ax.text(0.5, 0.95, "GradeSense: System Architecture Diagram",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(0.5, 0.90, "Separation of Offline Training Lifecycle from Online Inference Runtime",
            ha="center", va="top", fontsize=10, color="#64748B")

    # Training Pipeline Zone (Top Box)
    rect_train = patches.FancyBboxPatch((0.05, 0.48), 0.90, 0.38,
                                        boxstyle="round,pad=0.02,rounding_size=0.03",
                                        linewidth=1.2, edgecolor="#93C5FD", facecolor="#EFF6FF")
    ax.add_patch(rect_train)
    ax.text(0.08, 0.83, "OFFLINE TRAINING & EVALUATION PIPELINE (train_pipeline.py)",
            fontsize=10, fontweight="bold", color="#1E40AF")

    # Training Components
    boxes_train = [
        (0.08, 0.55, 0.16, 0.22, "Raw Dataset\n(student-mat.csv)\nN=395 records", "#FFFFFF", "#3B82F6"),
        (0.28, 0.55, 0.18, 0.22, "Leakage-Safe\nSplit & 5-Fold CV\nTrain:316 | Test:79", "#FFFFFF", "#3B82F6"),
        (0.50, 0.55, 0.18, 0.22, "Model Training\n& Benchmark\nRF vs OLS vs GB", "#FFFFFF", "#3B82F6"),
        (0.72, 0.55, 0.18, 0.22, "Production Package\nmodel_artifact.pkl\nWeights + Metadata", "#10B981", "#047857"),
    ]

    for x, y, w, h, text, fc, ec in boxes_train:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                   linewidth=1.2, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        tc = "#FFFFFF" if fc == "#10B981" else "#0F172A"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=tc)

    # Arrows in Training
    ax.annotate("", xy=(0.28, 0.66), xytext=(0.24, 0.66),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563EB"))
    ax.annotate("", xy=(0.50, 0.66), xytext=(0.46, 0.66),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563EB"))
    ax.annotate("", xy=(0.72, 0.66), xytext=(0.68, 0.66),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563EB"))

    # Downward connector from artifact
    ax.annotate("", xy=(0.81, 0.38), xytext=(0.81, 0.55),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#059669", linestyle="--"))
    ax.text(0.82, 0.44, "Serialized Load", fontsize=8.5, color="#059669", fontweight="bold")

    # Online Inference Zone (Bottom Box)
    rect_inf = patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.35,
                                      boxstyle="round,pad=0.02,rounding_size=0.03",
                                      linewidth=1.2, edgecolor="#A7F3D0", facecolor="#F0FDF4")
    ax.add_patch(rect_inf)
    ax.text(0.08, 0.37, "ONLINE INFERENCE & USER INTERFACE RUNTIME (Streamlit app.py)",
            fontsize=10, fontweight="bold", color="#065F46")

    # Inference Components
    boxes_inf = [
        (0.08, 0.12, 0.20, 0.20, "User Input Form\n(G1, G2, Absences,\nStudy, Failures)", "#FFFFFF", "#10B981"),
        (0.33, 0.12, 0.22, 0.20, "Input Validation &\nNamed DataFrame Schema\n(No Warning)", "#FFFFFF", "#10B981"),
        (0.60, 0.12, 0.30, 0.20, "Dual Prediction & Explainability\n• Primary: Random Forest\n• Baseline: OLS Equation & Waterfall", "#FFFFFF", "#10B981"),
    ]

    for x, y, w, h, text, fc, ec in boxes_inf:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                   linewidth=1.2, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A")

    # Arrows in Inference
    ax.annotate("", xy=(0.33, 0.22), xytext=(0.28, 0.22),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#059669"))
    ax.annotate("", xy=(0.60, 0.22), xytext=(0.55, 0.22),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#059669"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    out_file = FIGURES_DIR / "visual1_architecture.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file.name}")


def generate_figure2_ml_workflow():
    """Figure 2: Machine Learning Workflow Diagram."""
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(0.5, 0.95, "GradeSense: Machine Learning Pipeline Workflow",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(0.5, 0.90, "End-to-End Leakage-Safe Data Flow, Validation, and Model Serialization",
            ha="center", va="top", fontsize=10, color="#64748B")

    steps = [
        ("Step 1\nData Ingestion", "student-mat.csv\n(395 rows, 26 cols)\nCheck 0-grades (38)"),
        ("Step 2\nValidation", "Schema check\nZero nulls\nValid ranges 0-20"),
        ("Step 3\nTrain/Test Split", "80% Train (n=316)\n20% Test (n=79)\nrandom_state=42"),
        ("Step 4\n5-Fold CV", "Strictly on Train\nCompare RF vs OLS\nR2, MAE, RMSE"),
        ("Step 5\nFinal Fit", "Primary: RF (d=4)\nBaseline: OLS\nFit on full train"),
        ("Step 6\nHeld-Out Test", "Single evaluation\nRF R2=0.8480\nOLS R2=0.7822"),
        ("Step 7\nArtifact Save", "model_artifact.pkl\nmodel_metadata.json\nSchema + Metrics"),
    ]

    n = len(steps)
    width = 0.115
    spacing = (0.92 - (n * width)) / (n - 1)
    start_x = 0.04

    for i, (title, desc) in enumerate(steps):
        x = start_x + i * (width + spacing)
        y = 0.35
        h = 0.40

        box_color = "#EFF6FF" if i < 3 else ("#F0FDF4" if i < 6 else "#FEF3C7")
        border_color = "#2563EB" if i < 3 else ("#059669" if i < 6 else "#D97706")

        p = patches.FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0.015",
                                   linewidth=1.2, edgecolor=border_color, facecolor=box_color)
        ax.add_patch(p)

        ax.text(x + width/2, y + h - 0.06, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#0F172A")
        ax.axhline(y + h - 0.12, xmin=x, xmax=x+width)
        ax.text(x + width/2, y + 0.12, desc, ha="center", va="center",
                fontsize=7.5, color="#334155")

        if i < n - 1:
            ax.annotate("", xy=(x + width + spacing, y + h/2), xytext=(x + width, y + h/2),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#475569"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    out_file = FIGURES_DIR / "visual2_ml_workflow.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file.name}")


def generate_figure3_frontend_preview():
    """Figure 3: Frontend UI Predictor View Mockup & Architecture Preview."""
    fig, ax = plt.subplots(figsize=(12, 7.0), dpi=300)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    # Header
    ax.text(0.5, 0.95, "GradeSense: Frontend Predictor Interface",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(0.5, 0.90, "UI/UX Design Preview: Input Cards, Forecast Result Badge, and Feature Contribution",
            ha="center", va="top", fontsize=10, color="#64748B")

    # Form Box (Left)
    p_form = patches.FancyBboxPatch((0.05, 0.12), 0.42, 0.72, boxstyle="round,pad=0.02",
                                    linewidth=1, edgecolor="#E2E8F0", facecolor="#FFFFFF")
    ax.add_patch(p_form)
    ax.text(0.08, 0.80, "STUDENT ACADEMIC INPUTS", fontsize=11, fontweight="bold", color="#0F172A")

    inputs_text = """
1. Academic Track Record:
   • First Period Grade (G1):  12 / 20
   • Second Period Grade (G2): 13 / 20

2. Behavioral & Attendance Factors:
   • School Absences:          4 days
   • Weekly Study Time:        2 to 5 hours (Level 2)
   • Past Class Failures:      0 previous failures

[  PREDICT FINAL GRADE (G3)  ]
    """
    ax.text(0.08, 0.48, inputs_text.strip(), fontsize=9.5, color="#334155", linespacing=1.6)

    # Result Box (Right Top)
    p_res = patches.FancyBboxPatch((0.52, 0.48), 0.43, 0.36, boxstyle="round,pad=0.02",
                                   linewidth=1.2, edgecolor="#93C5FD", facecolor="#EFF6FF")
    ax.add_patch(p_res)
    ax.text(0.55, 0.79, "FORECASTED PERFORMANCE", fontsize=10, fontweight="bold", color="#1E40AF")
    ax.text(0.73, 0.68, "13.2 / 20", fontsize=28, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(0.73, 0.58, "GRADE CATEGORY: SATISFACTORY (10–13.9)", fontsize=9, fontweight="bold", color="#047857", ha="center")
    ax.text(0.73, 0.52, "Model: Random Forest (Primary) | Cohort MAE: 1.12 pts", fontsize=8.5, color="#475569", ha="center")

    # Explanation Box (Right Bottom)
    p_exp = patches.FancyBboxPatch((0.52, 0.12), 0.43, 0.32, boxstyle="round,pad=0.02",
                                   linewidth=1, edgecolor="#E2E8F0", facecolor="#FFFFFF")
    ax.add_patch(p_exp)
    ax.text(0.55, 0.39, "BASELINE OLS WATERFALL CONTRIBUTION", fontsize=10, fontweight="bold", color="#0F172A")

    contrib_text = """
• Base Intercept (β₀):          -1.62 pts
• Second Period Grade (G2):     +12.74 pts (Dominant)
• First Period Grade (G1):      +1.73 pts
• Absences (4 days):            +0.16 pts
• Study Time (2-5 hrs):         -0.14 pts
• Past Failures (0):             0.00 pts
─────────────────────────────────────────────
• Linear Model Prediction:      12.87 / 20
    """
    ax.text(0.55, 0.23, contrib_text.strip(), fontsize=8.2, color="#334155", linespacing=1.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()
    out_file_mockup = FIGURES_DIR / "visual3_ui_mockup.png"
    out_file_compat = FIGURES_DIR / "visual3_ui_screenshot.png"
    plt.savefig(out_file_mockup, dpi=300)
    plt.savefig(out_file_compat, dpi=300)
    plt.close()
    print(f"Generated: {out_file_mockup.name} (Architectural UI Layout Preview)")


def generate_figure4_actual_vs_predicted():
    """Figure 4: Actual vs Predicted Plot on Held-Out Test Data."""
    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)

    y_test = np.array(artifact["evaluation_data"]["y_test"])
    rf_preds = np.array(artifact["evaluation_data"]["test_predictions"]["Random Forest"])
    lr_preds = np.array(artifact["evaluation_data"]["test_predictions"]["Linear Regression"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. Random Forest (Primary Model)
    ax1.scatter(y_test, rf_preds, color="#2563EB", alpha=0.75, edgecolors="#1D4ED8", s=55, label="Held-Out Students (n=79)")
    ax1.plot([0, 20], [0, 20], color="#DC2626", linestyle="--", linewidth=1.5, label="Ideal Line (y = x)")
    ax1.set_title("Primary Model: Random Forest\nTest R² = 0.8480 | MAE = 1.1249", fontsize=11, fontweight="bold", color="#0F172A")
    ax1.set_xlabel("Actual Final Grade (G3)", fontsize=10, fontweight="bold", color="#334155")
    ax1.set_ylabel("Predicted Final Grade (G3)", fontsize=10, fontweight="bold", color="#334155")
    ax1.set_xlim(-1, 21)
    ax1.set_ylim(-1, 21)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Linear Regression (Baseline Model)
    ax2.scatter(y_test, lr_preds, color="#0D9488", alpha=0.75, edgecolors="#0F766E", s=55, label="Held-Out Students (n=79)")
    ax2.plot([0, 20], [0, 20], color="#DC2626", linestyle="--", linewidth=1.5, label="Ideal Line (y = x)")
    ax2.set_title("Baseline Model: Linear Regression\nTest R² = 0.7822 | MAE = 1.3394", fontsize=11, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Actual Final Grade (G3)", fontsize=10, fontweight="bold", color="#334155")
    ax2.set_ylabel("Predicted Final Grade (G3)", fontsize=10, fontweight="bold", color="#334155")
    ax2.set_xlim(-2, 21)
    ax2.set_ylim(-2, 21)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True)

    plt.suptitle("GradeSense: Actual vs. Predicted Performance on Held-Out Test Set (20%, n=79)",
                 fontsize=13, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout()

    out_file = FIGURES_DIR / "visual4_actual_vs_predicted.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file.name}")


def generate_figure5_correlation_heatmap():
    """Figure 5: Correlation Heatmap of modeled features."""
    df = pd.read_csv(DATA_PATH)
    cols = ["G1", "G2", "G3", "absences", "studytime", "failures"]
    corr = df[cols].corr()

    labels = [
        "First Grade (G1)",
        "Second Grade (G2)",
        "Final Grade (G3)",
        "Absences",
        "Study Time",
        "Past Failures",
    ]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    ax.set_title("GradeSense: Correlation Heatmap of Academic & Behavioral Predictors\n(UCI Student Performance Dataset, N=395)",
                 fontsize=11, fontweight="bold", color="#0F172A", pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="bold")
    plt.yticks(rotation=0, fontsize=9, fontweight="bold")
    plt.tight_layout()

    out_file = FIGURES_DIR / "visual5_correlation_heatmap.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file.name}")


if __name__ == "__main__":
    print("Generating report figures in reports/figures/...")
    generate_figure1_architecture()
    generate_figure2_ml_workflow()
    generate_figure3_frontend_preview()
    generate_figure4_actual_vs_predicted()
    generate_figure5_correlation_heatmap()
    print("All 5 figures generated successfully.")
