
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


import os
import time
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import polars as pl
import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from analysis import (
    load_data,
    clean_data,
    engineer_features,
    train_model,
    clean_and_engineer_polars,
)


class TestUnit(unittest.TestCase):
    """Unit tests for functions"""

    def test_load_data_reads_csv(self):
        df = pd.DataFrame(
            {
                "Marital status": [1, 2],
                "Application mode": [1, 1],
                "Application order": [1, 2],
                "Course": [10, 10],
                "Daytime/evening attendance": [1, 1],
                "Previous qualification": [1, 1],
                "Nacionality": [1, 1],
                "Target": ["Dropout", "Graduate"],
            }
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            out = load_data(tmp.name)
        try:
            self.assertEqual(len(out), 2)
            self.assertIn("Target", out.columns)
        finally:
            os.remove(tmp.name)

    def test_clean_data_encodes_target_and_drops_dupes_outliers(self):
        df = pd.DataFrame(
            {
                "Target": ["Dropout", "Enrolled", "Graduate"],
                "Age at enrollment": [18, 19, 200],  # extreme outlier
                "Application order": [1, 1, 1],
            }
        )
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add duplicate
        cleaned = clean_data(df, target="Target", outlier_z=3.0)
        # target should be mapped to ints (0/1/2)
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned["Target"]))
        # duplicate removed; outlier likely removed
        self.assertLessEqual(len(cleaned), 3)

    def test_engineer_features_creates_columns(self):
        df = pd.DataFrame(
            {
                "Target": [0, 1, 2],
                "Curricular units 1st sem (approved)": [5, 3, 0],
                "Curricular units 1st sem (enrolled)": [5, 6, 0],
                "Curricular units 2nd sem (approved)": [4, 0, 2],
                "Curricular units 2nd sem (enrolled)": [4, 5, 2],
                "Curricular units 1st sem (grade)": [14.0, 12.0, 10.0],
                "Curricular units 2nd sem (grade)": [13.0, 11.0, 9.0],
            }
        )
        out = engineer_features(df)
        for col in [
            "first_sem_pass_rate",
            "second_sem_pass_rate",
            "total_approved",
            "avg_grade",
        ]:
            self.assertIn(col, out.columns)
        # pass_rate NaNs should be filled to 0.0
        self.assertFalse(out["first_sem_pass_rate"].isna().any())
        self.assertFalse(out["second_sem_pass_rate"].isna().any())


class TestIntegration(unittest.TestCase):
    """Integration tests for small pipelines."""

    def _make_small_df(self, as_strings=False):
        # minimal columns to run through cleaning + feature eng + ML
        target_vals = ["Dropout", "Enrolled", "Graduate"] if as_strings else [0, 1, 2]
        return pd.DataFrame(
            {
                "Target": target_vals * 10,
                "Application order": np.random.randint(1, 4, 30),
                "Age at enrollment": np.random.randint(17, 45, 30),
                "Curricular units 1st sem (approved)": np.random.randint(0, 6, 30),
                "Curricular units 1st sem (enrolled)": np.random.randint(1, 7, 30),
                "Curricular units 2nd sem (approved)": np.random.randint(0, 6, 30),
                "Curricular units 2nd sem (enrolled)": np.random.randint(1, 7, 30),
                "Curricular units 1st sem (grade)": np.random.uniform(8, 18, 30),
                "Curricular units 2nd sem (grade)": np.random.uniform(8, 18, 30),
            }
        )

    def test_pandas_pipeline_end_to_end(self):
        df = self._make_small_df(as_strings=True)
        cleaned = clean_data(df, target="Target", outlier_z=3.5)
        feat = engineer_features(cleaned)
        # one quick groupby to validate aggregations on engineered columns
        grp = feat.groupby("Target")[
            ["first_sem_pass_rate", "second_sem_pass_rate"]
        ].mean(numeric_only=True)
        self.assertGreaterEqual(len(grp), 1)
        # train model
        model = train_model(feat)
        self.assertIsNotNone(model)

    def test_polars_pipeline_consistency_with_pandas_counts(self):
        # compare row counts after very light polars cleaning/engineering
        df_pd = self._make_small_df(as_strings=True)
        df_pl = pl.DataFrame(df_pd)

        cleaned_pd = clean_data(df_pd, target="Target", outlier_z=3.5)
        eng_pd = engineer_features(cleaned_pd)

        df_pd = self._make_small_df(as_strings=True)
        df_pl = pl.DataFrame(df_pd)

        cleaned_pd = clean_data(df_pd, target="Target", outlier_z=3.5)
        eng_pd = engineer_features(cleaned_pd)

        cleaned_pl = clean_and_engineer_polars(df_pl)
        assert len(eng_pd) == cleaned_pl.height

        eng_pl = clean_and_engineer_polars(df_pl)

        # Both pipelines should produce engineered columns
        for col in ["first_sem_pass_rate", "second_sem_pass_rate", "avg_grade"]:
            self.assertIn(col, eng_pd.columns)
            self.assertIn(col, eng_pl.columns)

        self.assertGreater(len(eng_pd), 0)
        self.assertGreater(eng_pl.height, 0)


class TestMLUnit(unittest.TestCase):
    """Unit test focused on the ML training step."""

    def test_train_model_basic(self):
        rng = np.random.default_rng(0)
        n = 120
        df = pd.DataFrame(
            {
                "Target": ([0, 1, 2] * (n // 3))[:n],
                "Application order": rng.integers(1, 4, n),
                "Age at enrollment": rng.integers(17, 45, n),
                "Curricular units 1st sem (approved)": rng.integers(0, 6, n),
                "Curricular units 1st sem (enrolled)": rng.integers(1, 7, n),
                "Curricular units 2nd sem (approved)": rng.integers(0, 6, n),
                "Curricular units 2nd sem (enrolled)": rng.integers(1, 7, n),
                "Curricular units 1st sem (grade)": rng.uniform(8, 18, n),
                "Curricular units 2nd sem (grade)": rng.uniform(8, 18, n),
            }
        )

        feat = engineer_features(df)

        # Train the RF classifier; should succeed and return a fitted model
        model = train_model(feat)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "feature_importances_"))

        # The model should be able to predict on rows with the same feature columns
        X_some = feat.drop(columns=["Target"]).head(10)
        preds = model.predict(X_some)

        self.assertEqual(len(preds), 10)
        # Predictions should be among the valid classes {0,1,2}
        self.assertTrue(set(np.unique(preds)).issubset({0, 1, 2}))


class TestSystem(unittest.TestCase):
    """System test: write CSV, run full pipeline (no plots asserted)."""

    def test_system_run_from_csv(self):
        # create a realistic small CSV and run load->clean->engineer->train
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "Target": ["Dropout", "Enrolled", "Graduate"] * 20,
                "Application order": rng.integers(1, 4, 60),
                "Age at enrollment": rng.integers(17, 45, 60),
                "Curricular units 1st sem (approved)": rng.integers(0, 6, 60),
                "Curricular units 1st sem (enrolled)": rng.integers(1, 7, 60),
                "Curricular units 2nd sem (approved)": rng.integers(0, 6, 60),
                "Curricular units 2nd sem (enrolled)": rng.integers(1, 7, 60),
                "Curricular units 1st sem (grade)": rng.uniform(8, 18, 60),
                "Curricular units 2nd sem (grade)": rng.uniform(8, 18, 60),
            }
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            loaded = load_data(tmp.name)
        try:
            cleaned = clean_data(loaded, target="Target", outlier_z=3.5)
            feat = engineer_features(cleaned)
            model = train_model(feat)
            self.assertIsNotNone(model)
        finally:
            os.remove(tmp.name)


class TestPerformanceSmoke(unittest.TestCase):

    def test_small_data_finishes_quickly(self):
        df = pd.DataFrame(
            {
                "Target": [0, 1, 2] * 40,
                "Application order": np.random.randint(1, 4, 120),
                "Age at enrollment": np.random.randint(17, 45, 120),
                "Curricular units 1st sem (approved)": np.random.randint(0, 6, 120),
                "Curricular units 1st sem (enrolled)": np.random.randint(1, 7, 120),
                "Curricular units 2nd sem (approved)": np.random.randint(0, 6, 120),
                "Curricular units 2nd sem (enrolled)": np.random.randint(1, 7, 120),
                "Curricular units 1st sem (grade)": np.random.uniform(8, 18, 120),
                "Curricular units 2nd sem (grade)": np.random.uniform(8, 18, 120),
            }
        )
        t0 = time.time()
        cleaned = clean_data(df, target="Target", outlier_z=3.5)
        feat = engineer_features(cleaned)
        _ = train_model(feat)
        self.assertLess(time.time() - t0, 3.0)  # should be under 3s


if __name__ == "__main__":
    unittest.main(verbosity=2)
