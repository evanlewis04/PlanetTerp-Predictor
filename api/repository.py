"""Repository helpers for API access to local experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from planetterp_predictor.experiment_tracking import RUNS_DIR


class ExperimentRepository:
    """Read experiment artifacts written by the Phase 5 tracker."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = runs_dir

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        if not self.runs_dir.exists():
            return runs

        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = self.read_metadata(run_dir.name)
            runs.append({
                "run_id": run_dir.name,
                "created_at": metadata.get("created_at"),
                "experiment_name": metadata.get("experiment_name"),
                "best_model_name": metadata.get("best_model_name"),
                "feature_count": metadata.get("feature_count"),
                "git_commit": metadata.get("git_commit"),
            })
        return runs

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        return runs[0]["run_id"] if runs else None

    def run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return run_dir

    def read_metadata(self, run_id: str) -> dict[str, Any]:
        metadata_path = self.run_dir(run_id) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for run: {run_id}")
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def read_metrics(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
        run_dir = self.run_dir(run_id)
        return {
            "cross_validation": self._read_csv_records(run_dir / "cross_validation_metrics.csv"),
            "holdout": self._read_csv_records(run_dir / "holdout_metrics.csv"),
            "best_feature_importance": self._read_csv_records(
                run_dir / "best_feature_importance.csv"
            ),
        }

    def list_plots(self, run_id: str) -> list[dict[str, str]]:
        plots_dir = self.run_dir(run_id) / "plots"
        if not plots_dir.exists():
            return []
        return [
            {
                "name": plot_path.name,
                "url": f"/artifacts/runs/{run_id}/plots/{plot_path.name}",
            }
            for plot_path in sorted(plots_dir.glob("*.png"))
        ]

    def load_model_bundle(self, run_id: str) -> dict[str, Any]:
        model_path = self.run_dir(run_id) / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found for run: {run_id}")
        return joblib.load(model_path)

    def _read_csv_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pd.read_csv(path).where(pd.notnull, None).to_dict(orient="records")
