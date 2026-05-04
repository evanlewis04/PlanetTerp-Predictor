# Modeling Notes

Phase 4 expands the project from a three-model comparison into a broader sklearn model benchmark.

## Model Families

The shared model registry in `src/model_specs.py` currently includes:

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

External libraries such as XGBoost, LightGBM, and CatBoost are intentionally deferred so the project still runs in the existing environment.

## Metrics

Cross-validation reports:

- R2 mean and standard deviation
- RMSE mean and standard deviation
- MAE mean and standard deviation
- MSE mean and standard deviation

The broad comparison caps cross-validation at five folds by default so local runs stay responsive while comparing many model families.

Holdout evaluation reports:

- R2
- Adjusted R2 when the test set is large enough
- RMSE
- MAE
- Median absolute error
- MSE

Adjusted R2 is left blank when the holdout sample count is too small relative to the number of predictors. This is expected for tiny smoke-test datasets.

## Hyperparameter Tuning

`GridSearchCV` is used for models with configured search grids:

- Ridge, Lasso, and Elastic Net regularization
- Random Forest and Extra Trees depth, leaf size, and feature sampling
- Gradient Boosting learning rate, estimators, and depth
- Hist Gradient Boosting iterations, learning rate, and leaf count
- KNN neighbors and weighting
- SVR regularization, epsilon, and kernel

Linear Regression and the baseline models are fit directly.

Training-time grid search uses up to three folds by default. The grids are intentionally compact; later experiment tracking can introduce larger searches or Optuna runs.

## Interpretation

Native feature importance is reported for models that expose coefficients or tree importances. Models without reliable native importances are still evaluated, but they are omitted from the feature importance plot until a later permutation-importance or SHAP workflow is added.

The Phase 3 expanded feature set can overfit small datasets. Use snapshot sizes larger than the smoke-test default before treating R2 comparisons as meaningful.
