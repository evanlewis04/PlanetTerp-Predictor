from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

import src.evaluator as evaluator_module
import src.model_trainer as trainer_module
from src.model_specs import ModelSpec
from src.model_trainer import ModelTrainer


def _fixture_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="Mean Baseline",
            estimator=DummyRegressor(strategy="mean"),
        ),
        ModelSpec(
            name="Linear Regression",
            estimator=LinearRegression(),
            needs_scaling=True,
            supports_native_importance=True,
        ),
    ]


class ModelTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_trainer_specs = trainer_module.get_model_specs
        self.original_evaluator_specs = evaluator_module.get_model_specs
        self.original_output_dir = evaluator_module.OUTPUT_DIR
        self.temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(self.temp_dir.name) / "outputs"
        output_dir.mkdir()
        trainer_module.get_model_specs = _fixture_model_specs
        evaluator_module.get_model_specs = _fixture_model_specs
        evaluator_module.OUTPUT_DIR = str(output_dir)

    def tearDown(self) -> None:
        trainer_module.get_model_specs = self.original_trainer_specs
        evaluator_module.get_model_specs = self.original_evaluator_specs
        evaluator_module.OUTPUT_DIR = self.original_output_dir
        self.temp_dir.cleanup()

    def test_trainer_fits_fixture_models_and_reports_holdout_metrics(self) -> None:
        features = pd.DataFrame(
            {
                "num_reviews": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "avg_sentiment_compound": [
                    -0.2,
                    -0.1,
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ],
                "avg_expected_grade": [
                    2.4,
                    2.5,
                    2.7,
                    2.8,
                    3.0,
                    3.1,
                    3.2,
                    3.4,
                    3.5,
                    3.6,
                    3.8,
                    3.9,
                ],
            }
        )
        target = pd.Series(
            [2.6, 2.7, 2.9, 3.0, 3.1, 3.3, 3.4, 3.6, 3.7, 3.8, 4.0, 4.1]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            best_model, feature_importance, results_table, best_model_name = (
                ModelTrainer().train_and_evaluate_models(features, target)
            )

        self.assertIsNotNone(best_model)
        self.assertIn(best_model_name, {"Mean Baseline", "Linear Regression"})
        self.assertEqual(set(results_table["Model"]), {"Mean Baseline", "Linear Regression"})
        self.assertTrue(
            {"RMSE", "MAE", "Median AE", "R2", "Adjusted R2"}.issubset(results_table.columns)
        )
        self.assertFalse(feature_importance.empty)
        self.assertEqual(
            set(feature_importance["Feature"]),
            {"num_reviews", "avg_sentiment_compound", "avg_expected_grade"},
        )

    def test_cross_validation_uses_same_fixture_registry(self) -> None:
        features = pd.DataFrame(
            {
                "num_reviews": [4, 5, 6, 7, 8, 9],
                "avg_sentiment_compound": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        target = pd.Series([2.5, 2.7, 3.0, 3.2, 3.5, 3.7])

        results = evaluator_module.ModelEvaluator().evaluate_models_with_cross_validation(
            features,
            target,
            cv=3,
        )

        self.assertEqual(set(results["Model"]), {"Mean Baseline", "Linear Regression"})
        self.assertTrue({"R2 Mean", "RMSE Mean", "MAE Mean"}.issubset(results.columns))


if __name__ == "__main__":
    unittest.main()
