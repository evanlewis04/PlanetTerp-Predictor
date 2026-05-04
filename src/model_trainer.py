"""
Model training and evaluation functionality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from config.config import RANDOM_STATE, TEST_SIZE
from src.evaluator import ModelEvaluator
from src.model_specs import ModelSpec, get_model_specs, get_tuning_grid


class ModelTrainer:
    """Handles model training, tuning, and holdout evaluation."""

    def __init__(self):
        """Initialize model trainer."""
        self.evaluator = ModelEvaluator()

    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Train, tune, and evaluate all registered models on the dataset.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        results = []
        trained_models = {}
        feature_importances = {}
        predictions = {}

        for spec in get_model_specs():
            model, best_params = self._fit_model(spec, X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = self._calculate_holdout_metrics(
                y_test=y_test,
                y_pred=y_pred,
                n_features=X.shape[1],
            )
            row = {
                "Model": spec.name,
                **metrics,
                "Best Params": self._format_params(best_params),
            }
            results.append(row)
            trained_models[spec.name] = model
            predictions[spec.name] = y_pred
            feature_importances[spec.name] = self._extract_feature_importance(
                model,
                X.columns,
                spec,
            )

            print(
                f"{spec.name} - RMSE: {metrics['RMSE']:.4f}, "
                f"MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}"
            )
            if best_params:
                print(f"  Best params: {self._format_params(best_params)}")

        results_table = pd.DataFrame(results).sort_values("R2", ascending=False)
        self.evaluator.create_performance_comparison_plot(results_table)
        self.evaluator.create_feature_importance_plots(feature_importances)

        best_model, best_importance = self._find_best_model(
            results_table,
            trained_models,
            feature_importances,
        )
        best_model_name = results_table.iloc[0]["Model"]
        self.evaluator.create_residual_plot(predictions[best_model_name], y_test, best_model_name)
        self.evaluator.create_prediction_scatter_plot(y_test, predictions[best_model_name], best_model_name)
        self.last_results_table = results_table
        self.last_best_model_name = best_model_name
        self.last_feature_importances = feature_importances

        print("\nModel Performance Comparison:")
        display_columns = [
            "Model",
            "RMSE",
            "MAE",
            "Median AE",
            "MSE",
            "R2",
            "Adjusted R2",
            "Best Params",
        ]
        print(results_table[display_columns].to_string(index=False))

        return best_model, best_importance, results_table, best_model_name

    def _fit_model(self, spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, dict]:
        """Fit a model, using grid search when a tuning grid is configured."""
        print(f"\nTraining {spec.name}")
        estimator = spec.build_estimator()
        tuning_grid = get_tuning_grid(spec)

        if not tuning_grid:
            estimator.fit(X_train, y_train)
            return estimator, {}

        cv_splits = min(3, len(X_train))
        if cv_splits < 2:
            estimator.fit(X_train, y_train)
            return estimator, {}

        search = GridSearchCV(
            estimator=estimator,
            param_grid=tuning_grid,
            scoring="r2",
            cv=KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
            error_score=np.nan,
        )
        try:
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_
        except ValueError as exc:
            print(f"  Tuning skipped after grid-search failure: {exc}")
            estimator.fit(X_train, y_train)
            return estimator, {}

    def _calculate_holdout_metrics(
        self,
        *,
        y_test: pd.Series,
        y_pred: np.ndarray,
        n_features: int,
    ) -> dict[str, float]:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "Median AE": median_absolute_error(y_test, y_pred),
            "R2": r2,
            "Adjusted R2": self._adjusted_r2(r2, n_samples=len(y_test), n_features=n_features),
        }

    def _adjusted_r2(self, r2: float, *, n_samples: int, n_features: int) -> float:
        denominator = n_samples - n_features - 1
        if denominator <= 0:
            return np.nan
        return 1 - (1 - r2) * (n_samples - 1) / denominator

    def _extract_feature_importance(
        self,
        model: Any,
        feature_names: pd.Index,
        spec: ModelSpec,
    ) -> pd.DataFrame:
        if not spec.supports_native_importance:
            return pd.DataFrame(columns=["Feature", "Importance"])

        fitted_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(fitted_model, "coef_"):
            raw_importance = np.ravel(np.abs(fitted_model.coef_))
        elif hasattr(fitted_model, "feature_importances_"):
            raw_importance = np.ravel(fitted_model.feature_importances_)
        else:
            return pd.DataFrame(columns=["Feature", "Importance"])

        if len(raw_importance) != len(feature_names):
            return pd.DataFrame(columns=["Feature", "Importance"])

        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": raw_importance,
        }).sort_values("Importance", ascending=False)

    def _format_params(self, params: dict[str, Any]) -> str:
        if not params:
            return ""
        cleaned_params = {
            key.replace("model__", ""): value
            for key, value in params.items()
        }
        return ", ".join(f"{key}={value}" for key, value in sorted(cleaned_params.items()))

    def _find_best_model(
        self,
        results_table: pd.DataFrame,
        models_dict: dict[str, Any],
        feature_importances: dict[str, pd.DataFrame],
    ) -> tuple[Any, pd.DataFrame]:
        """Find and return the best performing model."""
        best_row = results_table.loc[results_table["R2"].idxmax()]
        best_model_name = best_row["Model"]
        best_model = models_dict[best_model_name]
        best_importance = feature_importances.get(best_model_name)

        print(f"\nBest holdout model: {best_model_name}")
        print("\nMost important feature for models with native importances:")
        for model_name, importance_df in feature_importances.items():
            if importance_df is None or importance_df.empty or importance_df["Importance"].sum() == 0:
                continue
            top_feature = importance_df.iloc[0]
            print(
                f"{model_name}: {top_feature['Feature']} "
                f"(Importance: {top_feature['Importance']:.4f})"
            )

        return best_model, best_importance
