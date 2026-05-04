# Known Limitations

This project is suitable as a local portfolio and experimentation platform. It is not yet a production rating system.

## Data Limitations

- PlanetTerp reviews are user-generated and may not represent all students.
- Review volume differs widely across professors.
- Small local smoke runs are useful for testing workflows but are not enough for stable model conclusions.
- Snapshot results can differ from live API results as public data changes.
- Expected grades and review text may be missing or inconsistently formatted.

## Modeling Limitations

- The target is average review rating, not teaching quality.
- Review sentiment and keyword features are close to student opinion and should be interpreted carefully.
- Current hyperparameter grids are compact for local responsiveness.
- External gradient boosting libraries such as XGBoost, LightGBM, and CatBoost are not yet included.
- Native feature importance is not available for every model family.
- SHAP and permutation importance are not implemented yet.

## Prediction Limitations

- `/api/predict` expects model-ready aggregate features, not raw professor names or raw review lists.
- The dashboard prediction form is a demonstration UI and fills omitted saved features with `0.0`.
- A production prediction flow should run the same feature engineering pipeline used during training.

## Engineering Limitations

- Training triggered through `/api/train` is synchronous.
- There is no background job queue yet.
- Experiment tracking is local-file based rather than MLflow or a database-backed registry.
- The frontend is configured for local development, not a hosted deployment.
- Docker Compose and deployment packaging are deferred to Phase 10.

## Responsible Use

The model should not be used to make high-stakes decisions about instructors, courses, hiring, promotion, or compensation. It is best understood as a demonstration of reproducible ML engineering, model comparison, and dashboarding around public review data.
