# Methodology

This project is a local, reproducible machine learning workflow for predicting professor average ratings from PlanetTerp review data.

## Data Collection

The data workflow fetches professor records and associated reviews from the PlanetTerp API through the existing Python package workflow. Instead of depending on live API state for every model run, the Phase 2 workflow saves timestamped raw snapshots under `data/raw/`.

Snapshotting matters because PlanetTerp data can change over time. A saved snapshot lets the same feature extraction and modeling code be rerun against the same input records.

## Data Validation

Validation summarizes common data quality issues:

- Missing or malformed professor names.
- Missing or malformed review lists.
- Missing, invalid, or out-of-range ratings.
- Missing courses.
- Missing or unmapped expected grades.
- Empty review text.
- Duplicate review-like records.

The dashboard and dataset summaries should be read alongside these validation outputs, especially for small snapshots.

## Target

The model target is `avg_rating`, the average numeric PlanetTerp review rating for each retained professor. The model trains one row per professor.

## Feature Engineering

The current feature pipeline is `phase3_feature_extractor_v1`. It intentionally avoids direct rating-derived predictors such as median rating, rating variance, or rating skew because those would summarize the target source itself.

Major feature families:

- Review volume and length.
- Course mix and course level.
- Expected-grade distribution.
- VADER sentiment aggregates.
- Interpretable keyword categories.
- Lightweight readability and punctuation.
- One-hot encoded department labels.

Expected-grade and review-text sentiment features are marked as medium leakage risk in the feature catalog. They do not use rating values directly, but they are close to student opinion and may not be available in all prediction scenarios.

## Modeling

The model registry currently compares:

- Mean Baseline
- Median Baseline
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest
- Extra Trees
- Gradient Boosting
- Hist Gradient Boosting
- KNN Regressor
- SVR

Models with configured grids use compact `GridSearchCV` tuning. The grids are deliberately small so local runs remain responsive.

## Evaluation

Cross-validation reports:

- R2 mean and standard deviation.
- RMSE mean and standard deviation.
- MAE mean and standard deviation.
- MSE mean and standard deviation.

Holdout evaluation reports:

- R2.
- Adjusted R2 when the holdout sample is large enough.
- RMSE.
- MAE.
- Median absolute error.
- MSE.

Adjusted R2 can be blank for small snapshots because the number of predictors can exceed the usable holdout sample size.

## Interpretation

Feature importance is saved when the best model exposes native importances or coefficients. For tree models, these importances identify features frequently useful for splitting, not causal drivers. A feature with high importance should be read as a model signal, not proof that changing that attribute would change a professor rating.

## Prediction

The API prediction endpoint loads `best_model.joblib` from a selected experiment run. The caller provides a feature dictionary keyed by saved feature column names. If `fill_missing` is true, omitted features are filled with `0.0` and reported back to the caller.

For production-quality predictions, callers should provide the full feature vector produced by the same feature pipeline used during training.
