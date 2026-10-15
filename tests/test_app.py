"""
GradeSense — Application & UI Component Automated Tests (Module J)
Compatible with standard library unittest and pytest.
"""

from pathlib import Path
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest

import ui_components

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = BASE_DIR / "app.py"


class TestUIComponents(unittest.TestCase):
    """Test shared UI components, loaders, and rendering helpers."""

    def test_load_dataset_success(self):
        """Verify load_dataset returns valid ground-truth DataFrame."""
        df = ui_components.load_dataset()
        self.assertIsNotNone(df, "load_dataset returned None")
        self.assertEqual(len(df), 395, "Dataset must contain exactly 395 rows")
        self.assertIn("G3", df.columns, "G3 target column must exist")

    def test_load_artifact_success(self):
        """Verify load_artifact returns valid model artifact envelope."""
        artifact = ui_components.load_artifact()
        self.assertIsNotNone(artifact, "load_artifact returned None")
        self.assertEqual(artifact["primary_model_name"], "Random Forest")
        self.assertEqual(artifact["baseline_model_name"], "Linear Regression")

    def test_render_metric_card_html(self):
        """Verify metric card HTML structure."""
        html = ui_components.render_metric_card("TEST LABEL", "100%", "Subtext info")
        self.assertIn("TEST LABEL", html)
        self.assertIn("100%", html)
        self.assertIn("Subtext info", html)
        self.assertIn("gs-card", html)

    def test_render_section_header(self):
        """Verify section header does not raise error."""
        # render_section_header calls _html which handles Streamlit runtime
        try:
            ui_components.render_section_header("Section Header", "Subtitle")
        except Exception as e:
            self.fail(f"render_section_header raised an unexpected exception: {e}")

    def test_missing_files_graceful_handling(self):
        """Verify loaders return None gracefully when files are missing."""
        orig_data_path = ui_components.DATA_PATH
        try:
            ui_components.DATA_PATH = Path("non_existent_data.csv")
            # Clear cache for testing
            ui_components.load_dataset.clear()
            res = ui_components.load_dataset()
            self.assertIsNone(res, "Missing dataset should return None gracefully")
        finally:
            ui_components.DATA_PATH = orig_data_path
            ui_components.load_dataset.clear()


class TestStreamlitAppPages(unittest.TestCase):
    """Integration tests verifying all 5 application pages render without exception."""

    def test_dashboard_page(self):
        """Test Executive Dashboard page rendering."""
        at = AppTest.from_file(str(APP_PATH))
        at.session_state["current_page"] = "📊 Executive Dashboard"
        at.session_state["nav_radio"] = "📊 Executive Dashboard"
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"Dashboard threw exceptions: {at.exception}")

    def test_predictor_page(self):
        """Test Predict Performance page rendering."""
        at = AppTest.from_file(str(APP_PATH))
        at.session_state["current_page"] = "🔮 Predict Performance"
        at.session_state["nav_radio"] = "🔮 Predict Performance"
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"Predictor threw exceptions: {at.exception}")

    def test_analytics_page(self):
        """Test Cohort Analytics page rendering."""
        at = AppTest.from_file(str(APP_PATH))
        at.session_state["current_page"] = "📈 Cohort Analytics"
        at.session_state["nav_radio"] = "📈 Cohort Analytics"
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"Analytics threw exceptions: {at.exception}")

    def test_model_insights_page(self):
        """Test Model Insights & Diagnostics page rendering."""
        at = AppTest.from_file(str(APP_PATH))
        at.session_state["current_page"] = "🔬 Model Insights & Residuals"
        at.session_state["nav_radio"] = "🔬 Model Insights & Residuals"
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"Model Insights threw exceptions: {at.exception}")

    def test_about_project_page(self):
        """Test About Project & Viva Guide page rendering."""
        at = AppTest.from_file(str(APP_PATH))
        at.session_state["current_page"] = "📖 About Project & Viva Guide"
        at.session_state["nav_radio"] = "📖 About Project & Viva Guide"
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"About Project threw exceptions: {at.exception}")


if __name__ == "__main__":
    unittest.main()
