from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
from api.repository import ExperimentRepository


class FixtureModel:
    def predict(self, rows: pd.DataFrame) -> list[float]:
        return [float(rows.iloc[0]["num_reviews"] + rows.iloc[0]["avg_sentiment_compound"])]


class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self.temp_dir.name)
        self.original_repository = api_main.repository
        api_main.repository = ExperimentRepository(runs_dir=self.runs_dir)
        self.client = TestClient(api_main.app)
        self.run_id = "20260504_130000_api-fixture"
        self._write_fixture_run(self.run_id)

    def tearDown(self) -> None:
        api_main.repository = self.original_repository
        self.temp_dir.cleanup()

    def test_health_models_and_run_listing_are_served_over_http(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        runs = self.client.get("/api/runs")
        self.assertEqual(runs.status_code, 200)
        self.assertEqual(runs.json()[0]["run_id"], self.run_id)
        self.assertEqual(runs.json()[0]["best_model_name"], "Fixture Model")

        models = self.client.get("/api/models")
        self.assertEqual(models.status_code, 200)
        self.assertGreaterEqual(len(models.json()), 8)
        self.assertIn("supports_native_importance", models.json()[0])

    def test_run_detail_metrics_and_plots_return_artifact_payloads(self) -> None:
        detail = self.client.get(f"/api/runs/{self.run_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["experiment_name"], "api-fixture")

        metrics = self.client.get(f"/api/runs/{self.run_id}/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json()["holdout"][0]["Model"], "Fixture Model")
        self.assertEqual(metrics.json()["best_feature_importance"][0]["Feature"], "num_reviews")

        plots = self.client.get(f"/api/runs/{self.run_id}/plots")
        self.assertEqual(plots.status_code, 200)
        self.assertEqual(
            plots.json(),
            [
                {
                    "name": "model_comparison.png",
                    "url": f"/artifacts/runs/{self.run_id}/plots/model_comparison.png",
                }
            ],
        )

    def test_missing_run_returns_not_found(self) -> None:
        response = self.client.get("/api/runs/missing-run")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Run not found", response.json()["detail"])

    def test_predict_fills_missing_features_and_reports_extra_features(self) -> None:
        response = self.client.post(
            "/api/predict",
            json={
                "run_id": self.run_id,
                "features": {
                    "num_reviews": 4,
                    "unused_feature": 99,
                },
                "fill_missing": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_name"], "Fixture Model")
        self.assertEqual(payload["prediction"], 4.0)
        self.assertEqual(payload["missing_features"], ["avg_sentiment_compound"])
        self.assertEqual(payload["extra_features"], ["unused_feature"])

    def test_predict_can_reject_incomplete_feature_vectors(self) -> None:
        response = self.client.post(
            "/api/predict",
            json={
                "run_id": self.run_id,
                "features": {
                    "num_reviews": 4,
                },
                "fill_missing": False,
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"]["missing_features"],
            ["avg_sentiment_compound"],
        )

    def test_train_accepts_live_api_request_and_reports_new_saved_run(self) -> None:
        new_run_id = "99999999_999999_dashboard-train"
        captured_kwargs = {}

        def fake_training(**kwargs):
            captured_kwargs.update(kwargs)
            run_dir = self.runs_dir / new_run_id
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": new_run_id,
                        "created_at": "2099-01-01T00:00:00",
                        "experiment_name": kwargs["experiment_name"],
                        "best_model_name": "Fixture Model",
                        "feature_count": 2,
                        "feature_columns": ["num_reviews", "avg_sentiment_compound"],
                        "git_commit": "fixture456",
                    },
                ),
                encoding="utf-8",
            )
            return object(), None, None, None

        with patch.object(api_main, "run_planetterp_analysis", side_effect=fake_training):
            response = self.client.post(
                "/api/train",
                json={
                    "snapshot": None,
                    "max_professors": 80,
                    "min_reviews": 1,
                    "experiment_name": "dashboard-train",
                    "save_experiment": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["latest_run_id"], new_run_id)
        self.assertEqual(captured_kwargs["num_professors"], 80)
        self.assertEqual(captured_kwargs["min_reviews"], 1)
        self.assertIsNone(captured_kwargs["professors"])
        self.assertIsNone(captured_kwargs["snapshot_metadata"])
        self.assertEqual(captured_kwargs["experiment_name"], "dashboard-train")
        self.assertTrue(captured_kwargs["save_experiment"])

    def _write_fixture_run(self, run_id: str) -> None:
        run_dir = self.runs_dir / run_id
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True)
        metadata = {
            "run_id": run_id,
            "created_at": "2026-05-04T13:00:00",
            "experiment_name": "api-fixture",
            "best_model_name": "Fixture Model",
            "feature_count": 2,
            "feature_columns": ["num_reviews", "avg_sentiment_compound"],
            "git_commit": "fixture123",
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        pd.DataFrame(
            [{"Model": "Fixture Model", "R2": 0.25, "RMSE": 0.5, "MAE": 0.4}]
        ).to_csv(run_dir / "holdout_metrics.csv", index=False)
        pd.DataFrame([{"Model": "Fixture Model", "R2 Mean": 0.1}]).to_csv(
            run_dir / "cross_validation_metrics.csv",
            index=False,
        )
        pd.DataFrame([{"Feature": "num_reviews", "Importance": 0.8}]).to_csv(
            run_dir / "best_feature_importance.csv",
            index=False,
        )
        (plots_dir / "model_comparison.png").write_bytes(b"fixture")
        joblib.dump(
            {
                "feature_columns": ["num_reviews", "avg_sentiment_compound"],
                "best_model_name": "Fixture Model",
                "model": FixtureModel(),
            },
            run_dir / "best_model.joblib",
        )


if __name__ == "__main__":
    unittest.main()
