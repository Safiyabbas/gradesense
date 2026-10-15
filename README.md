# GradeSense — Student Performance Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**GradeSense** is a reproducible machine learning web application designed to forecast student academic performance in secondary school mathematics courses ($G_3 \in [0, 20]$). By synthesizing academic track records with study habits and attendance patterns, GradeSense enables educators and counselors to identify at-risk students before final examinations occur.

---

## 1. Project Overview & Key Features

* **Dual-Model Inference Engine**:
  * **Primary Model (Random Forest)**: Ensemble regressor ($n=100, \text{depth}=4$) capturing non-linear dropout patterns ($R^2 = 0.8480, \text{MAE} = 1.12$ pts).
  * **Baseline Model (Linear Regression)**: Transparent Ordinary Least Squares (OLS) equation providing exact $\beta$ feature weights for viva defense ($R^2 = 0.7822, \text{MAE} = 1.34$ pts).
* **Multi-Page Executive Interface**:
  * **Executive Dashboard**: Dynamic cohort benchmarks, pipeline verification status, and grade distribution charts.
  * **Forecasting Studio (Predictor)**: Grouped inputs with validation, real-time prediction, qualitative achievement banding, and an OLS waterfall feature contribution table.
  * **Cohort Analytics**: Deep-dive exploratory charts (distribution trajectories, study time, failures, and correlation heatmap).
  * **Model Insights & Diagnostics**: 5-Fold cross-validation benchmark table, actual-vs-predicted scatter with $y=x$ reference, residual distributions, and feature importances.
  * **About & Viva Guide**: Comprehensive theoretical formulations, dataset attribution, and model explanation notes for oral examination defense.
* **Leakage-Safe Architecture**: Strict train/test partitioning ($80/20$) with model selection conducted exclusively via 5-Fold Cross-Validation on the training data.
* **Complete Automated Test Suite**: 20 unit and integration tests covering dataset integrity, artifact loading, inference schema, boundary conditions, and page rendering.

---

## 2. System Architecture

```
[ Offline Training Lifecycle ]                     [ Online Inference Runtime ]
       student-mat.csv                                       Streamlit
              │                                                app.py
              ▼                                                  │
      train_pipeline.py                                          ├── Dashboard
   (80/20 Leakage-Safe Split)                                    ├── Predictor
              │                                                  ├── Analytics
              ├──> 5-Fold CV Model Selection                     ├── Model Insights
              │    (Random Forest vs OLS)                        └── About Project
              │                                                  │
              ▼                                                  ▼
      model_artifact.pkl ───────────────────────────────> Load Artifact
   (Fitted Models + Metadata)                              Named DataFrame
                                                           Prediction (0 Warnings)
```

---

## 3. Dataset Specifications

* **Source**: UCI Machine Learning Repository — *Student Performance Data Set* (Cortez and Silva, 2008).
* **Cohort Size**: 395 secondary school students from two Portuguese institutions (*Gabriel Pereira* and *Mouzinho da Silveira*).
* **Target Variable**: `G3` — Final mathematics grade on the Portuguese 20-point scale ($0 - 20$, passing threshold $\ge 10$).
* **Model Predictors (Exact Feature Sequence)**:
  1. `G1`: First evaluation period grade ($0 - 20$).
  2. `G2`: Second evaluation period grade ($0 - 20$).
  3. `absences`: Total recorded school absences ($0 - 93$).
  4. `studytime`: Weekly study hours ($1: <2\text{h}, 2: 2-5\text{h}, 3: 5-10\text{h}, 4: >10\text{h}$).
  5. `failures`: Number of past subject failures ($0 - 3$).
* **Zero-Grade Handling ($G_3=0$)**: Exactly 38 students ($9.6\%$) recorded a final grade of 0 due to final exam absenteeism or course withdrawal. These records are intentionally preserved to avoid survivorship bias.

---

## 4. Verified Evaluation Benchmark

Evaluated on an identical 80/20 split ($n_{\text{train}}=316, n_{\text{test}}=79, \text{random\_state}=42$):

| Architecture | 5-Fold CV $R^2$ (Train) | 5-Fold CV MAE | Test Set $R^2$ | Test Set MAE | Test Set RMSE | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** ($n=100, d=4$) | **$0.8587 \pm 0.0468$** | **0.9663** pts | **0.8480** | **1.1249** pts | **1.7656** pts | **Primary Model** |
| **Linear Regression** (OLS) | $0.8288 \pm 0.0426$ | 1.1609 pts | 0.7822 | 1.3394 pts | 2.1134 pts | **Baseline Model** |
| **Ridge Regression** ($\alpha=1.0$) | $0.8289 \pm 0.0427$ | 1.1607 pts | 0.7823 | 1.3392 pts | 2.1128 pts | Benchmark |
| **Gradient Boosting** ($n=100, d=2$) | $0.8714 \pm 0.0376$ | 0.9750 pts | 0.8462 | 1.1167 pts | 1.7758 pts | Benchmark |

### Mathematical Baseline Formulation (OLS Equation):
$$\hat{G}_3 = -1.6213 + 0.1445 \cdot G_1 + 0.9796 \cdot G_2 + 0.0392 \cdot \text{absences} - 0.0712 \cdot \text{studytime} - 0.4558 \cdot \text{failures}$$

---

## 5. Quick Start & Installation

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Git

### Step 1: Clone Repository & Create Environment
```bash
git clone https://github.com/your-username/gradesense.git
cd gradesense

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Activate on macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Pipeline (Optional — Artifact Pre-Packaged)
```bash
python train_pipeline.py
```

### Step 4: Run Automated Test Suite
```bash
python -m unittest discover -s tests -v
```

### Step 5: Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Project Structure

```
STUDENT-GRADE-PREDICTION-using-Linear-Regression-main/
│
├── app.py                      # Production Streamlit application (5 pages)
├── train_pipeline.py           # Reproducible training & evaluation pipeline
├── ui_components.py            # Design tokens, CSS styling, and HTML cards
├── generate_figures.py         # 300 DPI publication report figures generator
│
├── student-mat.csv             # Ground-truth student dataset (N=395)
├── model_artifact.pkl          # Serialized production models & evaluation data
├── model_metadata.json         # Complete metadata & benchmark documentation
├── metrics_summary.json        # Human-readable JSON metrics summary
│
├── tests/                      # Automated unit and integration test suite
│   ├── __init__.py
│   ├── test_pipeline.py        # Dataset integrity, artifact schema, boundary inputs
│   └── test_app.py             # UI components, loaders, AppTest page execution
│
├── reports/
│   └── figures/                # 5 High-resolution publication figures
│       ├── visual1_architecture.png
│       ├── visual2_ml_workflow.png
│       ├── visual3_ui_screenshot.png
│       ├── visual4_actual_vs_predicted.png
│       └── visual5_correlation_heatmap.png
│
├── archive/                    # Archived legacy scripts (fully integrated)
│   ├── README.md
│   ├── fix_notebook.py
│   ├── patch_notebook.py
│   └── update_notebook.py
│
├── requirements.txt            # Frozen dependency requirements
├── .gitignore                  # Git exclusion rules
└── README.md                   # Complete academic documentation
```

---

## 7. Viva Defense Checklist & Key Insights

1. **Why does Random Forest outperform Linear Regression on this dataset?**  
   The presence of 38 students with $G_3 = 0$ creates a non-linear step-function boundary. A straight regression plane projects negative grades (e.g. $-0.85$) for absent students, requiring artificial clipping. Random Forest naturally groups dropouts into dedicated zero-grade leaf nodes.
2. **Why does study time exhibit a negative partial slope ($-0.071$) in Linear Regression?**  
   In multiple regression, coefficients represent the partial effect holding all other variables constant. Because $G_1$ and $G_2$ already absorb the positive academic benefits of study habits, residual study time reflects compensatory behavior: struggling students often study more hours immediately before failing.
3. **What is the difference between $R^2$ and MAE?**  
   $R^2$ measures the proportion of variance explained ($84.8\%$ for Random Forest). MAE measures practical error magnitude: our forecasts are accurate within an average of $\pm 1.12$ grade points on a 20-point scale.
4. **Was there data leakage in earlier notebooks?**  
   Yes. Previous versions performed correlation-based feature selection across the entire dataset before `train_test_split`. In GradeSense, the dataset is split first, and all model selection is performed strictly on training folds.

---

## 8. License & Attribution

* **Dataset Citation**: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.
* **Code License**: MIT License. Reused notebook logic from historical repositories is acknowledged and documented.
