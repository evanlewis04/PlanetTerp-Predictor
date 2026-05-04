# Experiment Tracking

Phase 5 adds lightweight local experiment tracking without requiring MLflow.

Every normal training run writes a timestamped folder under:

```text
experiments/runs/<run_id>/
```

The run folder contains:

- `metadata.json`
- `cross_validation_metrics.csv`
- `holdout_metrics.csv`
- `best_feature_importance.csv` when available
- `best_model.joblib`
- `plots/` copied from the current `outputs/` folder

Generated experiment run folders are ignored by Git. The committed `experiments/.gitkeep` and `experiments/runs/.gitkeep` files preserve the intended directory structure.

## CLI Usage

Save a named experiment from the latest snapshot:

```powershell
.\.venv\Scripts\python.exe -m planetterp_predictor run --snapshot latest --min-reviews 1 --experiment-name phase5-smoke
```

Run without saving experiment artifacts:

```powershell
.\.venv\Scripts\python.exe -m planetterp_predictor run --snapshot latest --min-reviews 1 --no-save-experiment
```

## Model Bundle

The `best_model.joblib` artifact stores a dictionary containing:

- `model`: the fitted sklearn estimator or pipeline
- `best_model_name`
- `feature_columns`
- `feature_pipeline_version`
- `imputation_strategy`

Future phases should move more preprocessing into sklearn `Pipeline` or `ColumnTransformer` objects so every prediction-time transformation is captured directly in the saved model artifact.
