# Upgrade Plan: Professional Data Science Platform

## Goal

Transform the PlanetTerp Professor Rating Predictor from a college-style analysis script into a professional data science project with reliable data ingestion, reproducible experiments, stronger model performance, and a frontend application for exploring model results, statistics, and visualizations.

The upgraded project should answer three questions clearly:

1. What data was used, and how trustworthy is it?
2. Which model performs best, and why?
3. How can a user inspect predictions, metrics, plots, and feature drivers without reading Python logs?

## Current State

The current project is a Python script-driven machine learning workflow:

- Fetches professor and review data from the PlanetTerp API.
- Filters professors by minimum review count.
- Extracts review, course, grade, and sentiment features.
- Trains Linear Regression, Ridge Regression, and Random Forest models.
- Prints metrics to the terminal.
- Saves evaluation plots into `outputs/`.

This is a solid prototype, but it has several limitations:

- No persistent raw dataset or versioned processed dataset.
- No experiment tracking.
- No model artifacts saved for reuse.
- Limited feature engineering.
- Limited model selection and hyperparameter search.
- No frontend or API layer.
- Metrics and plots are only available as generated files.
- Reproducibility depends on live PlanetTerp API calls.

## Target Architecture

The professional version should be split into clear layers:

```text
planetterp_predictor/
|-- app/                         # React/Vite dashboard
|   |-- src/
|   `-- package.json
|-- api/                         # FastAPI backend for dashboard + predictions
|-- config/                      # Legacy-compatible constants
|-- data/
|   |-- raw/                     # Timestamped PlanetTerp API snapshots
|   `-- processed/               # Dataset summaries and feature CSVs
|-- docs/                        # Methodology, API, dashboard, deployment docs
|-- experiments/
|   `-- runs/                    # Saved metrics, plots, metadata, model bundles
|-- outputs/                     # Generated plots
|-- planetterp_predictor/        # CLI, settings, data artifacts, tracking
|-- src/                         # Data, features, models, evaluation
|-- tests/                       # Unit, API, and training tests
|-- pyproject.toml
`-- README.md
```

## Phase 1: Project Foundation

Status: implemented for the current platform scope.

Completed so far:

- Added `pyproject.toml` with package metadata, runtime dependencies, and dev-tool configuration.
- Added the `planetterp_predictor` package with module execution support.
- Added the `planetterp-predictor` console script entry point.
- Added structured settings with `PLANETTERP_` environment variable overrides.
- Added `.env.example`.
- Preserved compatibility with the original `main.py`, `src/`, `config/`, and `utils/` modules.
- Cleaned terminal metric labels from `R2` to `R2` for Windows console compatibility.
- Updated `README.md` with CLI and setup instructions.
- Added CI coverage for Python tests and the frontend production build.

### 1. Package And Environment Cleanup

- Replace loose script execution with an installable Python package.
- Add `pyproject.toml` for dependencies, formatting, linting, and test configuration.
- Keep `requirements.txt` only if needed for simple setup compatibility.
- Add `.env.example` for configurable values such as API limits, output paths, and frontend/backend ports.
- Add typed configuration using `pydantic-settings` or a similar structured config library.

Recommended tools:

- `ruff` for linting and formatting.
- `pytest` for testing.
- `mypy` or `pyright` for type checking.
- `pre-commit` for automated checks.

### 2. Reproducible Command Line Interface

Add a CLI so common workflows are repeatable:

```powershell
python -m planetterp_predictor data fetch --max-professors 1000
python -m planetterp_predictor data build-features
python -m planetterp_predictor train --experiment baseline
python -m planetterp_predictor evaluate --run-id latest
python -m planetterp_predictor serve-api
```

Each command should write clear artifacts and metadata.

## Phase 2: Data Engineering

Status: implemented for the current reproducible-data scope.

Completed so far:

- Added ignored `data/raw/` and `data/processed/` artifact directories.
- Added timestamped raw professor/review snapshots from PlanetTerp API results.
- Added dataset summary JSON artifacts with professor, review, rating, department, course, grade, and validation statistics.
- Added aggregate validation for missing names, malformed review fields, missing/invalid ratings, missing courses, missing expected grades, empty review text, and duplicate reviews.
- Added `data validate`, `data summary`, and `data build-features` CLI workflows.
- Added `run --snapshot` so model training can use a saved snapshot instead of live API data.

### 1. Persist API Snapshots

Instead of fetching live data every run, save raw API responses:

- `data/raw/professors_YYYYMMDD_HHMMSS.json`
- `data/raw/reviews_YYYYMMDD_HHMMSS.json`

This makes model results reproducible even if PlanetTerp data changes.

### 2. Add Data Validation

Add validation checks before training:

- Required fields exist: professor name, reviews, rating, course, expected grade.
- Ratings are in the expected range.
- Grades are mapped correctly.
- Duplicate reviews are handled.
- Empty or malformed reviews are tracked.
- Minimum sample sizes are enforced before model training.

Possible tools:

- `pandera` for dataframe validation.
- `pydantic` for API response schemas.

### 3. Dataset Summary Reports

Generate a dataset profile for each run:

- Number of professors fetched.
- Number of professors retained after filters.
- Review count distribution.
- Rating distribution.
- Department distribution.
- Missing grade/review/rating percentages.
- Top courses and departments represented.

The frontend should display this summary.

## Phase 3: Feature Engineering

Status: implemented for the current interpretable-feature scope.

Completed so far:

- Added richer non-rating-derived review length, readability, course mix, expected-grade, sentiment balance, and keyword-category features.
- Added `docs/feature_catalog.md` to document feature sources, transformations, and leakage risk.
- Explicitly deferred rating-derived features because they would leak the target into the predictors.

Future candidates include TF-IDF or embedding features, but the current dashboard and model artifacts intentionally favor interpretable, non-rating-derived predictors.

### 1. Improve Existing Features

Enhance the current feature set:

- Median rating, rating variance, and rating skew per professor.
- Review recency features if dates are available.
- Department-level aggregate statistics.
- Course-level aggregate statistics.
- Number of lower, mid, and upper-level courses.
- Expected grade distribution features beyond only A-ratio.
- Review length distribution, not just average length.

### 2. Better Text Features

The current project uses VADER sentiment. Keep it as a baseline, then add richer text features:

- TF-IDF features from review text.
- Topic modeling features.
- Review readability scores.
- Keyword/category features for difficulty, clarity, helpfulness, workload, exams, grading, lectures, and attendance.
- Sentence-transformer embeddings for semantic review representation.

Professional note: text embeddings may improve R2, but they also make interpretability harder. The dashboard should show both high-performing and interpretable model variants.

### 3. Leakage Review

Carefully check for target leakage. For example, if a feature directly or indirectly summarizes the same ratings being predicted, the model may look better than it really is.

Add a documented feature list with:

- Feature name.
- Source field.
- Transformation.
- Whether it is available before prediction time.
- Risk of leakage.

## Phase 4: Modeling Upgrade

Status: implemented for the current model-benchmarking scope.

Completed so far:

- Added a shared sklearn model registry in `src/model_specs.py`.
- Added mean and median baselines.
- Added Lasso, Elastic Net, Extra Trees, Gradient Boosting, Hist Gradient Boosting, KNN, and SVR model families.
- Added `GridSearchCV` tuning for configured regularized, tree, boosting, KNN, and SVR models.
- Added cross-validation RMSE and MAE alongside R2 and MSE.
- Added holdout RMSE, MAE, median absolute error, and adjusted R2.
- Added `docs/modeling.md` to document model families, metrics, tuning, and interpretation notes.

### 1. Establish Baselines

Keep simple baselines so advanced models can be compared honestly:

- Mean predictor.
- Department mean predictor.
- Linear Regression.
- Ridge Regression.
- Lasso Regression.
- Elastic Net.

### 2. Add More Model Families

Train and compare a broader set of models:

- Random Forest Regressor.
- Gradient Boosting Regressor.
- HistGradientBoostingRegressor.
- Extra Trees Regressor.
- Support Vector Regression.
- K-Nearest Neighbors Regressor.
- XGBoost, LightGBM, or CatBoost if external dependencies are acceptable.
- Stacked ensemble using the strongest base models.

### 3. Hyperparameter Tuning

Replace small hand-written tuning loops with structured search:

- `GridSearchCV` for smaller models.
- `RandomizedSearchCV` for larger search spaces.
- Optuna for more professional experiment optimization.

Tune at least:

- Ridge/Lasso/ElasticNet regularization.
- Random forest depth, leaf size, feature sampling, and tree count.
- Gradient boosting learning rate, depth, iterations, and regularization.
- Text feature dimensionality.
- Minimum review threshold.

### 4. Stronger Evaluation

Use multiple metrics:

- R2.
- Adjusted R2.
- RMSE.
- MAE.
- Median absolute error.
- Cross-validation mean and standard deviation.

Add visual diagnostics:

- Actual vs predicted scatter.
- Residual distribution.
- Residuals by predicted rating.
- Residuals by department.
- Calibration-style rating bucket comparison.
- Feature importance.
- SHAP values for tree-based models.

### 5. Improve R2 Responsibly

R2 improvement should come from better signal, not accidental leakage.

Recommended R2 improvement strategy:

1. Increase dataset size beyond the default 1000 professors if the API permits.
2. Tune `MIN_REVIEWS` and compare performance by review-count threshold.
3. Add TF-IDF or embedding features from review text.
4. Add department and course aggregate features.
5. Use gradient boosting models.
6. Tune hyperparameters with cross-validation.
7. Try target transformations only if rating distribution suggests it.
8. Evaluate on a held-out test set that is not used during tuning.

The final dashboard should display both the best R2 model and the best interpretable model.

## Phase 5: Experiment Tracking And Model Registry

Status: implemented for the current local experiment-tracking scope.

Completed so far:

- Added lightweight local experiment tracking under `experiments/runs/`.
- Added `metadata.json` for settings, Git commit, snapshot metadata, feature columns, target summary, and best model details.
- Added saved cross-validation and holdout metrics CSV artifacts.
- Added saved best-model feature importance CSV artifacts when native importances are available.
- Added `best_model.joblib` bundles containing the fitted model, feature columns, feature pipeline version, and imputation strategy.
- Added plot copying from `outputs/` into each experiment run folder.
- Added `docs/experiments.md` with artifact and CLI usage documentation.

### 1. Track Every Training Run

Each experiment should save:

- Git commit hash if available.
- Dataset snapshot ID.
- Feature pipeline version.
- Model type.
- Hyperparameters.
- Cross-validation metrics.
- Holdout metrics.
- Generated plots.
- Trained model artifact path.

Possible tools:

- MLflow for full experiment tracking.
- A lightweight local `experiments/` folder with JSON metadata if keeping the project simpler.

### 2. Save Model Artifacts

Save reusable artifacts:

- Trained model with `joblib`.
- Scaler/imputer/encoder pipeline.
- Feature column order.
- Model metadata.
- Evaluation report.

Prefer scikit-learn `Pipeline` and `ColumnTransformer` objects so preprocessing and prediction stay consistent.

## Phase 6: Backend API

Status: implemented for the current FastAPI/dashboard scope.

Completed so far:

- Added a FastAPI backend under `api/`.
- Added `/health`, `/api/runs`, `/api/runs/{run_id}`, `/api/runs/{run_id}/metrics`, `/api/runs/{run_id}/plots`, `/api/models`, `/api/predict`, and `/api/train`.
- Added static plot serving from experiment run folders.
- Added prediction from saved Phase 5 `best_model.joblib` bundles.
- Added synchronous training from the API for local/demo usage.
- Added `planetterp-predictor serve-api`.
- Added `docs/api.md`.
- Added HTTP-level FastAPI endpoint tests with temporary run artifacts and prediction fixtures.

Add a backend API with FastAPI.

Core endpoints:

```text
GET  /health
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/metrics
GET  /api/runs/{run_id}/plots
GET  /api/models
POST /api/predict
POST /api/train
```

The API should serve:

- Model comparison tables.
- Dataset summaries.
- Plot image URLs or base64 image references.
- Feature importance data.
- Prediction responses for sample professor/review inputs.

For early versions, training can remain a CLI task while the API only reads generated artifacts. Later, the frontend can trigger training jobs.

## Phase 7: Frontend Dashboard

Status: implemented in the first Phase 7 upgrade pass.

Completed so far:

- Added a React, TypeScript, and Vite dashboard under `app/`.
- Added dashboard views for overview, dataset metadata, model comparison, feature importance, plot gallery, prediction, and saved runs.
- Connected the frontend to the Phase 6 FastAPI endpoints for health, runs, metadata, metrics, plots, model registry, and predictions.
- Added a simple prediction form that submits feature payloads to `/api/predict`.
- Added frontend setup notes to `README.md` and `docs/api.md`.

Build a full frontend app, preferably with React, TypeScript, and Vite.

### 1. Dashboard Views

Create these main views:

- Overview.
- Dataset Explorer.
- Model Comparison.
- Model Detail.
- Feature Importance.
- Prediction Explorer.
- Training Runs.

### 2. Overview Page

Display:

- Latest run status.
- Best model by R2.
- Best model by MAE.
- Number of professors and reviews used.
- Rating distribution.
- Key warnings about missing data or small sample sizes.

### 3. Dataset Explorer

Display:

- Filters by department, course level, review count, and rating range.
- Dataset summary cards.
- Rating distribution chart.
- Review count histogram.
- Department sample-size table.

### 4. Model Comparison

Display a sortable table with:

- Model name.
- R2.
- RMSE.
- MAE.
- Cross-validation R2 mean.
- Cross-validation R2 standard deviation.
- Training time.
- Feature set used.

Also show:

- Bar charts comparing metrics.
- Actual vs predicted plots.
- Residual plots.
- Model ranking by chosen metric.

### 5. Model Detail

For each model:

- Hyperparameters.
- Feature pipeline.
- Top features.
- Error diagnostics.
- Prediction examples.
- Warnings about overfitting or unstable metrics.

### 6. Output Image Gallery

The frontend should display generated images from `outputs/`, including:

- `cross_validation_results.png`
- `model_comparison.png`
- `feature_importance_all_models.png`
- `linear_regression_residuals.png`
- `ridge_regression_residuals.png`
- `random_forest_residuals.png`
- `ridge_alpha_tuning.png`
- `prediction_scatter.png` if generated

Each image should be paired with a short explanation and the run ID that produced it.

### 7. Prediction Explorer

Allow a user to:

- Select a trained model.
- Enter hypothetical professor/review statistics.
- Enter sample review text.
- See predicted rating.
- See the most influential features for that prediction.

This turns the project from a static report into an interactive tool.

## Phase 8: Testing

Status: implemented in the first Phase 8 upgrade pass.

Completed so far:

- Added Python `unittest` coverage for dataset summary generation, feature extraction helpers, grade/course parsing, and local experiment repository reads.
- Added model registry endpoint-shape coverage through direct API function testing.
- Added HTTP-level FastAPI endpoint tests with temporary experiment artifacts.
- Added small fixture model-training tests for holdout metrics and cross-validation.
- Added a frontend smoke script that checks dashboard navigation, prediction, failed responses, console errors, and mobile overview rendering.
- Verified the new tests with `python -m unittest discover -s tests`, `npm run build`, and `npm run smoke`.

Add automated tests for:

- Grade mapping.
- Course level extraction.
- Sentiment feature extraction.
- API response parsing.
- Feature dataframe schema.
- Model training on a small fixture dataset.
- API endpoints.
- Frontend rendering smoke tests.

Recommended test structure:

```text
tests/
|-- test_api_endpoints.py
|-- test_api_repository.py
|-- test_data_artifacts.py
|-- test_feature_extractor.py
`-- test_model_training.py
```

Frontend tests:

- Component tests with Vitest.
- Basic end-to-end dashboard checks with Playwright.

## Phase 9: Documentation

Status: implemented in the first Phase 9 upgrade pass.

Completed so far:

- Rewrote `README.md` as a professional project overview with setup, workflows, testing, dashboard screenshots, and documentation links.
- Added `docs/data_dictionary.md` for raw snapshots, processed summaries, feature CSVs, and experiment run artifacts.
- Added `docs/methodology.md` for data, validation, target, feature, modeling, evaluation, interpretation, and prediction methodology.
- Added `docs/dashboard.md` with dashboard view descriptions and screenshots.
- Added `docs/limitations.md` with data, modeling, prediction, engineering, and responsible-use limitations.
- Added `docs/case_study.md` with the current smoke run metrics, feature signals, and next improvement path.

Upgrade documentation so the project is portfolio-ready:

- Rewrite `README.md` with a professional project overview.
- Add setup instructions for backend and frontend.
- Add a data dictionary.
- Add model methodology.
- Add interpretation notes.
- Add known limitations.
- Add screenshots of the dashboard.
- Add a short technical case study explaining how R2 was improved.

## Phase 10: Deployment Options

Status: implemented in the first Phase 10 deployment pass.

Completed so far:

- Added `Dockerfile.api` for the FastAPI/backend package.
- Added `app/Dockerfile` and `app/nginx.conf` for a production React build served by Nginx.
- Added `docker-compose.yml` with API and frontend services.
- Mounted `data/`, `experiments/`, and `outputs/` into the API container for persistent local artifacts.
- Added `.dockerignore` to keep caches and generated local artifacts out of Docker build context.
- Added `docs/deployment.md` and README deployment instructions.

Possible deployment paths:

- Local-only dashboard for portfolio demonstration.
- Docker Compose with frontend, API, and local artifact volume.
- Hosted frontend with a lightweight backend.
- Scheduled training job that refreshes data snapshots.

Recommended first deployment:

```text
Docker Compose
|-- frontend: React/Vite static app
|-- api: FastAPI backend
`-- volumes: data, outputs, experiments
```

## Post-Upgrade Hardening

The original 10-phase upgrade is now complete for a local portfolio deployment. Remaining work should be treated as polish and rigor rather than prerequisite platform construction:

1. Run a larger snapshot and training pass, then update the case study and screenshots with stronger evidence.
2. Add frontend component tests with Vitest and React Testing Library for empty states, metric tables, prediction errors, and API failure banners.
3. Add a richer model detail view with selected-model hyperparameters, warning flags, and associated diagnostics.
4. Consider TF-IDF, SHAP, Optuna, or external boosting libraries only after the current interpretable baseline is well documented on a larger dataset.

## Suggested Implementation Order

1. Restructure the Python project into package modules.
2. Add raw data snapshotting and processed dataset generation.
3. Add tests for existing feature extraction.
4. Convert modeling into reusable scikit-learn pipelines.
5. Add more models and structured hyperparameter tuning.
6. Save metrics, plots, and model artifacts per run.
7. Build FastAPI read endpoints for runs, metrics, plots, and models.
8. Build the React dashboard.
9. Add prediction explorer.
10. Add MLflow or local experiment tracking.
11. Polish documentation and screenshots.
12. Containerize the app.

## Success Criteria

The upgrade is complete when:

- A new user can run the project from documented commands.
- Data snapshots make experiments reproducible.
- At least eight model types are compared.
- Hyperparameter tuning is automated.
- Metrics are saved in structured files, not only printed.
- A frontend displays model statistics, plots, and comparisons.
- The project has tests for core data and modeling logic.
- The best model has a clearly documented R2 improvement attempt.
- The dashboard explains both performance and limitations.

## Brief Summary

This upgrade plan turns the project into a professional end-to-end data science application. The main work is to make the data pipeline reproducible, expand feature engineering, compare many more model types, tune models systematically to improve R2, save experiment artifacts, expose results through a FastAPI backend, and build a React frontend that displays metrics, plots, model comparisons, and interactive predictions.
