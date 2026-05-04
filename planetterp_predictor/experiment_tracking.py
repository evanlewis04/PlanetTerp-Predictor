"""Lightweight local experiment tracking."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.config import OUTPUT_DIR
from planetterp_predictor.settings import settings

EXPERIMENTS_DIR = Path("experiments")
RUNS_DIR = EXPERIMENTS_DIR / "runs"
FEATURE_PIPELINE_VERSION = "phase3_feature_extractor_v1"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, sort_keys=True)
        file.write("\n")


class ExperimentTracker:
    """Persist run metadata, metrics, plots, and the best model artifact."""

    def __init__(self, root_dir: Path = RUNS_DIR):
        self.root_dir = root_dir

    def create_run_dir(self, experiment_name: str | None = None) -> tuple[str, Path]:
        run_id = _timestamp()
        if experiment_name:
            run_id = f"{run_id}_{_slugify(experiment_name)}"
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def save_run(
        self,
        *,
        experiment_name: str | None,
        snapshot_metadata: dict[str, Any] | None,
        feature_columns: list[str],
        target_summary: dict[str, Any],
        imputation_strategy: str | None,
        cv_results: pd.DataFrame,
        holdout_results: pd.DataFrame,
        best_model_name: str,
        best_model: Any,
        best_feature_importance: pd.DataFrame | None,
    ) -> tuple[str, Path]:
        run_id, run_dir = self.create_run_dir(experiment_name)

        cv_path = run_dir / "cross_validation_metrics.csv"
        holdout_path = run_dir / "holdout_metrics.csv"
        importance_path = run_dir / "best_feature_importance.csv"
        model_path = run_dir / "best_model.joblib"

        cv_results.to_csv(cv_path, index=False)
        holdout_results.to_csv(holdout_path, index=False)
        if best_feature_importance is not None and not best_feature_importance.empty:
            best_feature_importance.to_csv(importance_path, index=False)

        model_bundle = {
            "model": best_model,
            "best_model_name": best_model_name,
            "feature_columns": feature_columns,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "imputation_strategy": imputation_strategy,
        }
        joblib.dump(model_bundle, model_path)

        plots_dir = self._copy_current_plots(run_dir)

        metadata = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit_hash(),
            "settings": settings.to_dict(),
            "snapshot": snapshot_metadata or {"source": "live_api"},
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "imputation_strategy": imputation_strategy,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "target_summary": target_summary,
            "best_model_name": best_model_name,
            "artifacts": {
                "cross_validation_metrics": str(cv_path),
                "holdout_metrics": str(holdout_path),
                "best_feature_importance": str(importance_path)
                if importance_path.exists()
                else None,
                "best_model": str(model_path),
                "plots_dir": str(plots_dir) if plots_dir else None,
            },
        }
        _write_json(run_dir / "metadata.json", metadata)

        return run_id, run_dir

    def _copy_current_plots(self, run_dir: Path) -> Path | None:
        output_dir = Path(OUTPUT_DIR)
        if not output_dir.exists():
            return None

        plot_paths = sorted(output_dir.glob("*.png"))
        if not plot_paths:
            return None

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            shutil.copy2(plot_path, plots_dir / plot_path.name)
        return plots_dir
