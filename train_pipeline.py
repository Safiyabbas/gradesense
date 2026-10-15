"""
GradeSense — Student Performance Intelligence
Training Pipeline & Model Artifact Serialization (Module A)

This module implements a reproducible, leakage-safe machine learning training
and evaluation pipeline. It compares Linear Regression (interpretable baseline)
with Random Forest (leading non-linear alternative) using identical 5-feature
definitions, 5-fold cross-validation on training data, and a single held-out
test evaluation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student-mat.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifact.pkl"
METRICS_PATH = BASE_DIR / "metrics_summary.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"

FEATURE_NAMES = ["G1", "G2", "absences", "studytime", "failures"]
TARGET_NAME = "G3"

FEATURE_METADATA = {
    "G1": {
        "label": "First Period Grade",
        "type": "integer",
        "min": 0,
        "max": 20,
        "default": 12,
        "unit": "marks (0–20)",
        "description": "Student grade achieved in the first evaluation period.",
    },
    "G2": {
        "label": "Second Period Grade",
        "type": "integer",
        "min": 0,
        "max": 20,
        "default": 13,
        "unit": "marks (0–20)",
        "description": "Student grade achieved in the second evaluation period.",
    },
    "absences": {
        "label": "School Absences",
        "type": "integer",
        "min": 0,
        "max": 93,
        "default": 4,
        "unit": "days",
        "description": "Total number of school absences throughout the academic year.",
    },
    "studytime": {
        "label": "Weekly Study Time",
        "type": "integer",
        "min": 1,
        "max": 4,
        "default": 2,
        "unit": "category (1–4)",
        "options": {
            1: "< 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "> 10 hours",
        },
        "description": "Weekly study time categorized into discrete intervals.",
    },
    "failures": {
        "label": "Past Class Failures",
        "type": "integer",
        "min": 0,
        "max": 3,
        "default": 0,
        "unit": "count (0–3)",
        "options": {
            0: "0 past failures",
            1: "1 past failure",
            2: "2 past failures",
            3: "3 or more past failures",
        },
        "description": "Number of previous class failures.",
    },
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the student dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = FEATURE_NAMES + [TARGET_NAME]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Validate nulls
    null_counts = df[required_cols].isnull().sum().to_dict()
    if sum(null_counts.values()) > 0:
        raise ValueError(f"Dataset contains null values: {null_counts}")

    return df


def build_candidate_models() -> dict:
    """Construct candidate regressors for comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.08,
            random_state=42,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict,
) -> tuple:
    """Perform leakage-safe cross-validation and test set evaluation.

    Returns:
        (benchmark_records, fitted_models, test_predictions)
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }

    benchmark_records = []
    fitted_models = {}
    test_predictions = {}

    for name, model in models.items():
        # 1. 5-Fold Cross-Validation strictly on Training Data
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring=scoring,
            return_train_score=True,
        )

        cv_r2_mean = float(np.mean(cv_results["test_r2"]))
        cv_r2_std = float(np.std(cv_results["test_r2"]))
        cv_mae_mean = float(-np.mean(cv_results["test_neg_mae"]))
        cv_mae_std = float(np.std(cv_results["test_neg_mae"]))
        cv_rmse_mean = float(-np.mean(cv_results["test_neg_rmse"]))
        cv_rmse_std = float(np.std(cv_results["test_neg_rmse"]))

        # 2. Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # 3. Train set performance
        train_preds = model.predict(X_train)
        train_r2 = float(r2_score(y_train, train_preds))
        train_mae = float(mean_absolute_error(y_train, train_preds))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))

        # 4. Held-out test set performance (evaluated ONCE)
        test_preds = model.predict(X_test)
        test_predictions[name] = test_preds.tolist()

        test_r2 = float(r2_score(y_test, test_preds))
        test_mae = float(mean_absolute_error(y_test, test_preds))
        test_mse = float(mean_squared_error(y_test, test_preds))
        test_rmse = float(np.sqrt(test_mse))

        benchmark_records.append(
            {
                "model_name": name,
                "cv_r2_mean": round(cv_r2_mean, 4),
                "cv_r2_std": round(cv_r2_std, 4),
                "cv_mae_mean": round(cv_mae_mean, 4),
                "cv_mae_std": round(cv_mae_std, 4),
                "cv_rmse_mean": round(cv_rmse_mean, 4),
                "cv_rmse_std": round(cv_rmse_std, 4),
                "train_r2": round(train_r2, 4),
                "train_mae": round(train_mae, 4),
                "train_rmse": round(train_rmse, 4),
                "test_r2": round(test_r2, 4),
                "test_mae": round(test_mae, 4),
                "test_mse": round(test_mse, 4),
                "test_rmse": round(test_rmse, 4),
            }
        )

    return benchmark_records, fitted_models, test_predictions


def run_pipeline() -> dict:
    """Execute complete training, model selection, and artifact generation."""
    print("=" * 60)
    print("GradeSense — Starting Leakage-Safe Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_dataset(DATA_PATH)
    zero_g3_count = int((df[TARGET_NAME] == 0).sum())
    print(f"Loaded dataset: {DATA_PATH.name} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Zero-grade records (G3=0): {zero_g3_count} ({zero_g3_count / len(df) * 100:.1f}%) — preserved as valid exam absentees/dropouts.")

    # 2. Extract features and target
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # 3. Leakage-safe Train/Test Split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Split complete: Train set = {len(X_train)} samples | Test set = {len(X_test)} samples")

    # 4. Candidate models
    models = build_candidate_models()

    # 5. Cross-validation & Evaluation
    benchmarks, fitted_models, test_predictions = evaluate_models(
        X_train, y_train, X_test, y_test, models
    )

    # 6. Display Benchmark Results
    print("\n" + "=" * 60)
    print("MODEL SELECTION BENCHMARK (5-Fold CV on Training Set)")
    print("=" * 60)
    header = f"{'Model':<20} | {'CV R² (Mean ± Std)':<20} | {'CV MAE':<10} | {'Test R²':<10} | {'Test MAE':<10}"
    print(header)
    print("-" * len(header))
    for b in benchmarks:
        cv_r2_str = f"{b['cv_r2_mean']:.4f} ± {b['cv_r2_std']:.4f}"
        print(f"{b['model_name']:<20} | {cv_r2_str:<20} | {b['cv_mae_mean']:<10.4f} | {b['test_r2']:<10.4f} | {b['test_mae']:<10.4f}")

    # 7. Model Selection Decision:
    rf_bench = next(b for b in benchmarks if b["model_name"] == "Random Forest")
    lr_bench = next(b for b in benchmarks if b["model_name"] == "Linear Regression")

    print("\n" + "=" * 60)
    print("MODEL SELECTION DECISION LOGIC")
    print("=" * 60)
    print(f"Linear Regression : CV R² = {lr_bench['cv_r2_mean']:.4f} | CV MAE = {lr_bench['cv_mae_mean']:.4f} | Test R² = {lr_bench['test_r2']:.4f}")
    print(f"Random Forest     : CV R² = {rf_bench['cv_r2_mean']:.4f} | CV MAE = {rf_bench['cv_mae_mean']:.4f} | Test R² = {rf_bench['test_r2']:.4f}")

    # Model selection decision based strictly on training-set cross-validation:
    if rf_bench["cv_r2_mean"] > lr_bench["cv_r2_mean"] and rf_bench["cv_mae_mean"] < lr_bench["cv_mae_mean"]:
        primary_model_name = "Random Forest"
        print(f"\n=> SELECTED PRIMARY MODEL: {primary_model_name}")
        print("   Reason: Outperformed Linear Regression in 5-Fold Cross-Validation on training data")
        print(f"   (CV R²: {rf_bench['cv_r2_mean']:.4f} vs {lr_bench['cv_r2_mean']:.4f}; CV MAE: {rf_bench['cv_mae_mean']:.4f} vs {lr_bench['cv_mae_mean']:.4f}).")
        print("=> RETAINED BASELINE MODEL: Linear Regression (for coefficient explainability & viva).")
    else:
        primary_model_name = "Linear Regression"
        print(f"\n=> SELECTED PRIMARY MODEL: {primary_model_name}")

    # 8. Extract Linear Regression parameters for equation transparency
    lr_model = fitted_models["Linear Regression"]
    lr_coefficients = {feat: float(coef) for feat, coef in zip(FEATURE_NAMES, lr_model.coef_)}
    lr_intercept = float(lr_model.intercept_)

    # 9. Extract Random Forest feature importances
    rf_model = fitted_models["Random Forest"]
    rf_importances = {feat: float(imp) for feat, imp in zip(FEATURE_NAMES, rf_model.feature_importances_)}

    # 10. Package Artifact Envelope
    artifact = {
        "primary_model_name": primary_model_name,
        "primary_model": fitted_models[primary_model_name],
        "baseline_model_name": "Linear Regression",
        "baseline_model": fitted_models["Linear Regression"],
        "all_models": fitted_models,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "feature_metadata": FEATURE_METADATA,
        "benchmarks": benchmarks,
        "linear_regression_params": {
            "intercept": lr_intercept,
            "coefficients": lr_coefficients,
        },
        "random_forest_params": {
            "feature_importances": rf_importances,
            "n_estimators": rf_model.n_estimators,
            "max_depth": rf_model.max_depth,
        },
        "evaluation_data": {
            "X_test": X_test,
            "y_test": y_test,
            "test_predictions": test_predictions,
        },
        "metadata": {
            "project_name": "GradeSense — Student Performance Intelligence",
            "pipeline_version": "2.0.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DATA_PATH.name),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "zero_g3_samples": zero_g3_count,
            "zero_g3_handling": "Preserved without deletion as valid observations representing final examination dropouts and chronic absenteeism.",
            "test_size": 0.20,
            "random_state": 42,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    }

    # 11. Save model artifact
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved enriched model artifact: {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size} bytes)")

    # 12. Save JSON metrics summary and model metadata
    metadata_export = {
        "project_name": "GradeSense — Student Performance Intelligence",
        "pipeline_version": "2.0.0",
        "created_at_utc": artifact["metadata"]["created_at_utc"],
        "primary_model": primary_model_name,
        "baseline_model": "Linear Regression",
        "feature_names": FEATURE_NAMES,
        "target_variable": TARGET_NAME,
        "dataset_info": {
            "name": str(DATA_PATH.name),
            "total_records": len(df),
            "training_records": len(X_train),
            "test_records": len(X_test),
            "zero_g3_records": zero_g3_count,
            "zero_g3_rationale": "Retained as authentic student outcomes indicating course dropouts or exam absentees.",
            "test_size_fraction": 0.20,
            "random_seed": 42,
        },
        "benchmarks": benchmarks,
        "primary_model_summary": {
            "model_type": "RandomForestRegressor",
            "hyperparameters": {
                "n_estimators": rf_model.n_estimators,
                "max_depth": rf_model.max_depth,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "feature_importances": rf_importances,
            "5_fold_cv_r2": rf_bench["cv_r2_mean"],
            "5_fold_cv_mae": rf_bench["cv_mae_mean"],
            "test_r2": rf_bench["test_r2"],
            "test_mae": rf_bench["test_mae"],
            "test_rmse": rf_bench["test_rmse"],
        },
        "baseline_model_summary": {
            "model_type": "LinearRegression",
            "intercept": lr_intercept,
            "coefficients": lr_coefficients,
            "5_fold_cv_r2": lr_bench["cv_r2_mean"],
            "5_fold_cv_mae": lr_bench["cv_mae_mean"],
            "test_r2": lr_bench["test_r2"],
            "test_mae": lr_bench["test_mae"],
            "test_rmse": lr_bench["test_rmse"],
        },
        "environment": {
            "python_scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_export, f, indent=2)
    print(f"Saved model metadata JSON   : {METADATA_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_export, f, indent=2)
    print(f"Saved metrics summary JSON  : {METRICS_PATH}")
    print("=" * 60)

    return artifact


if __name__ == "__main__":
    run_pipeline()
