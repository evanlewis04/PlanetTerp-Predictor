from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from api.repository import ExperimentRepository
from api.main import list_models


class ExperimentRepositoryTests(unittest.TestCase):
    def test_reads_run_artifacts_from_local_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir)
            run_dir = runs_dir / "20260504_120000_fixture"
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True)

            (run_dir / "metadata.json").write_text(
                json.dumps({
                    "created_at": "2026-05-04T12:00:00",
                    "experiment_name": "fixture",
                    "best_model_name": "Random Forest",
                    "feature_count": 3,
                    "git_commit": "abc123",
                }),
                encoding="utf-8",
            )
            pd.DataFrame([{"Model": "Random Forest", "R2": 0.4}]).to_csv(
                run_dir / "holdout_metrics.csv",
                index=False,
            )
            pd.DataFrame([{"Model": "Random Forest", "R2 Mean": 0.2}]).to_csv(
                run_dir / "cross_validation_metrics.csv",
                index=False,
            )
            pd.DataFrame([{"Feature": "avg_sentiment_compound", "Importance": 0.1}]).to_csv(
                run_dir / "best_feature_importance.csv",
                index=False,
            )
            (plots_dir / "model_comparison.png").write_bytes(b"fixture")

            repository = ExperimentRepository(runs_dir=runs_dir)

            self.assertEqual(repository.latest_run_id(), "20260504_120000_fixture")
            self.assertEqual(repository.list_runs()[0]["experiment_name"], "fixture")
            self.assertEqual(repository.read_metadata("20260504_120000_fixture")["feature_count"], 3)
            self.assertEqual(
                repository.read_metrics("20260504_120000_fixture")["holdout"][0]["Model"],
                "Random Forest",
            )
            self.assertEqual(
                repository.list_plots("20260504_120000_fixture"),
                [
                    {
                        "name": "model_comparison.png",
                        "url": "/artifacts/runs/20260504_120000_fixture/plots/model_comparison.png",
                    }
                ],
            )

    def test_model_endpoint_shape_exposes_registry_metadata(self) -> None:
        models = list_models()

        self.assertGreaterEqual(len(models), 8)
        first_model = models[0]
        self.assertIsInstance(first_model.name, str)
        self.assertIsInstance(first_model.needs_scaling, bool)
        self.assertIsInstance(first_model.tuned, bool)
        self.assertIsInstance(first_model.supports_native_importance, bool)


if __name__ == "__main__":
    unittest.main()
