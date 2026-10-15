"""
GradeSense — Unit Tests for Pipeline and Model Artifact (Stage 1)
Compatible with standard library unittest and pytest.
"""

from pathlib import Path
import pickle
import unittest
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "student-mat.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifact.pkl"
EXPECTED_FEATURES = ["G1", "G2", "absences", "studytime", "failures"]


class TestDatasetIntegrity(unittest.TestCase):
    """Test raw dataset structure and cleanliness."""

    def setUp(self):
        self.assertTrue(DATA_PATH.exists(), f"Dataset missing at {DATA_PATH}")
        self.df = pd.read_csv(DATA_PATH)

    def test_dataset_shape(self):
        """Verify row count and minimum column requirements."""
        self.assertEqual(len(self.df), 395, "Dataset must have exactly 395 student records")
        for feat in EXPECTED_FEATURES + ["G3"]:
            self.assertIn(feat, self.df.columns, f"Required column {feat} missing from CSV")

    def test_no_null_values(self):
        """Verify dataset has 0 missing values across all modeled columns."""
        null_count = self.df[EXPECTED_FEATURES + ["G3"]].isnull().sum().sum()
        self.assertEqual(null_count, 0, "Modeled columns must have 0 missing values")

    def test_feature_ranges(self):
        """Verify numeric bounds of dataset features."""
        self.assertTrue((self.df["G1"] >= 0).all() and (self.df["G1"] <= 20).all())
        self.assertTrue((self.df["G2"] >= 0).all() and (self.df["G2"] <= 20).all())
        self.assertTrue((self.df["G3"] >= 0).all() and (self.df["G3"] <= 20).all())
        self.assertTrue((self.df["studytime"] >= 1).all() and (self.df["studytime"] <= 4).all())
        self.assertTrue((self.df["failures"] >= 0).all() and (self.df["failures"] <= 3).all())


class TestArtifactIntegrity(unittest.TestCase):
    """Test the serialized model artifact package."""

    def setUp(self):
        self.assertTrue(ARTIFACT_PATH.exists(), f"Model artifact missing at {ARTIFACT_PATH}")
        with open(ARTIFACT_PATH, "rb") as f:
            self.artifact = pickle.load(f)

    def test_artifact_keys(self):
        """Verify all essential components exist in the artifact envelope."""
        required_keys = [
            "primary_model_name",
            "primary_model",
            "baseline_model_name",
            "baseline_model",
            "feature_names",
            "benchmarks",
            "metadata",
            "linear_regression_params",
            "evaluation_data",
        ]
        for k in required_keys:
            self.assertIn(k, self.artifact, f"Key '{k}' missing from artifact envelope")

    def test_feature_schema(self):
        """Verify feature names and sequence match expected order."""
        self.assertEqual(self.artifact["feature_names"], EXPECTED_FEATURES)

    def test_model_selection_validity(self):
        """Verify primary and baseline models are instantiated and fitted."""
        primary_name = self.artifact["primary_model_name"]
        self.assertIn(primary_name, ["Random Forest", "Linear Regression"])
        self.assertTrue(hasattr(self.artifact["primary_model"], "predict"))
        self.assertTrue(hasattr(self.artifact["baseline_model"], "predict"))

    def test_benchmark_metrics_populated(self):
        """Verify benchmark table contains valid numeric evaluation results."""
        benchmarks = self.artifact["benchmarks"]
        self.assertGreaterEqual(len(benchmarks), 2, "Benchmark must contain at least 2 models")
        for b in benchmarks:
            self.assertGreater(b["test_r2"], 0.70, f"Model {b['model_name']} test R2 surprisingly low")
            self.assertLess(b["test_mae"], 2.0, f"Model {b['model_name']} test MAE unexpectedly high")


class TestModelInference(unittest.TestCase):
    """Test inference reliability, schema validation, and boundary conditions."""

    def setUp(self):
        with open(ARTIFACT_PATH, "rb") as f:
            self.artifact = pickle.load(f)
        self.rf_model = self.artifact["primary_model"]
        self.lr_model = self.artifact["baseline_model"]

    def test_single_inference_dataframe(self):
        """Test inference with a structured pandas DataFrame."""
        sample = pd.DataFrame([{"G1": 12, "G2": 13, "absences": 4, "studytime": 2, "failures": 0}])
        pred_rf = self.rf_model.predict(sample)[0]
        pred_lr = self.lr_model.predict(sample)[0]

        self.assertIsInstance(float(pred_rf), float)
        self.assertIsInstance(float(pred_lr), float)
        self.assertTrue(0 <= pred_rf <= 20, f"RF prediction {pred_rf} out of bounds")
        self.assertTrue(0 <= pred_lr <= 20, f"LR prediction {pred_lr} out of bounds")

    def test_boundary_inputs(self):
        """Test edge cases: minimum inputs and maximum inputs."""
        min_sample = pd.DataFrame([{"G1": 0, "G2": 0, "absences": 0, "studytime": 1, "failures": 0}])
        max_sample = pd.DataFrame([{"G1": 20, "G2": 20, "absences": 50, "studytime": 4, "failures": 0}])

        # RF predictions should naturally stay within realistic bounds
        rf_min = self.rf_model.predict(min_sample)[0]
        rf_max = self.rf_model.predict(max_sample)[0]

        self.assertGreaterEqual(rf_min, 0.0)
        self.assertLessEqual(rf_max, 20.0)
        self.assertGreater(rf_max, rf_min, "Top student should score higher than failing student")

    def test_linear_regression_equation_reproducibility(self):
        """Verify that manual OLS equation calculation matches lr_model.predict exactly."""
        params = self.artifact["linear_regression_params"]
        intercept = params["intercept"]
        coefs = params["coefficients"]

        g1, g2, absences, studytime, failures = 14, 15, 2, 3, 0
        manual_calc = (
            intercept
            + coefs["G1"] * g1
            + coefs["G2"] * g2
            + coefs["absences"] * absences
            + coefs["studytime"] * studytime
            + coefs["failures"] * failures
        )

        sample = pd.DataFrame([{"G1": g1, "G2": g2, "absences": absences, "studytime": studytime, "failures": failures}])
        model_pred = self.lr_model.predict(sample)[0]

        self.assertAlmostEqual(manual_calc, model_pred, places=5, msg="Manual equation must match LR predict")


if __name__ == "__main__":
    unittest.main()
