"""
Model evaluation and cross-validation functionality.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from config.config import CV_FOLDS, OUTPUT_DIR, RANDOM_STATE
from src.model_specs import get_model_specs
from utils.helpers import create_output_directory


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class ModelEvaluator:
    """Handles model evaluation and cross-validation."""

    def __init__(self):
        """Initialize model evaluator."""
        create_output_directory(OUTPUT_DIR)

    def evaluate_models_with_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = CV_FOLDS,
    ) -> pd.DataFrame:
        """
        Perform cross-validation to evaluate registered regression models.
        """
        # Keep the broad model benchmark responsive for local iteration.
        n_splits = min(cv, len(X), 5)
        if n_splits < 2:
            raise ValueError("At least two samples are required for cross-validation.")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "r2": make_scorer(r2_score),
            "rmse": make_scorer(_rmse, greater_is_better=False),
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
        }

        rows = []
        for spec in get_model_specs():
            scores = cross_validate(
                spec.build_estimator(),
                X,
                y,
                cv=kf,
                scoring=scoring,
                error_score=np.nan,
            )
            rows.append({
                "Model": spec.name,
                "R2 Mean": np.nanmean(scores["test_r2"]),
                "R2 Std": np.nanstd(scores["test_r2"]),
                "RMSE Mean": -np.nanmean(scores["test_rmse"]),
                "RMSE Std": np.nanstd(-scores["test_rmse"]),
                "MAE Mean": -np.nanmean(scores["test_mae"]),
                "MAE Std": np.nanstd(-scores["test_mae"]),
                "MSE Mean": -np.nanmean(scores["test_mse"]),
                "MSE Std": np.nanstd(-scores["test_mse"]),
            })

        results_df = pd.DataFrame(rows).sort_values("R2 Mean", ascending=False)
        self._plot_cv_results(results_df)

        return results_df

    def _plot_cv_results(self, results_df: pd.DataFrame) -> None:
        """
        Plot cross-validation results.
        """
        plot_df = results_df.copy()
        models = plot_df["Model"]

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.bar(models, plot_df["R2 Mean"])
        plt.errorbar(models, plot_df["R2 Mean"], yerr=plot_df["R2 Std"], fmt="o", color="black")
        plt.title("R2 by Model (Cross-Validation)")
        r2_min = min(-1.0, float(plot_df["R2 Mean"].min()) - 0.1)
        r2_max = max(1.0, float(plot_df["R2 Mean"].max()) + 0.1)
        plt.ylim(r2_min, r2_max)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.bar(models, plot_df["RMSE Mean"])
        plt.errorbar(
            models,
            plot_df["RMSE Mean"],
            yerr=plot_df["RMSE Std"],
            fmt="o",
            color="black",
        )
        plt.title("RMSE by Model (Cross-Validation)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_results.png"))
        plt.close()

    def create_performance_comparison_plot(self, results_table: pd.DataFrame) -> None:
        """
        Create holdout model performance comparison plots.
        """
        results_table = results_table.sort_values("R2", ascending=False)
        models = results_table["Model"]

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.bar(models, results_table["RMSE"])
        plt.title("Holdout RMSE (lower is better)")
        plt.xticks(rotation=45, ha="right")

        plt.subplot(1, 3, 2)
        plt.bar(models, results_table["MAE"])
        plt.title("Holdout MAE (lower is better)")
        plt.xticks(rotation=45, ha="right")

        plt.subplot(1, 3, 3)
        plt.bar(models, results_table["R2"])
        plt.title("Holdout R2 (higher is better)")
        r2_min = min(-1.0, float(results_table["R2"].min()) - 0.1)
        r2_max = max(1.0, float(results_table["R2"].max()) + 0.1)
        plt.ylim(r2_min, r2_max)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
        plt.close()

    def create_feature_importance_plots(self, importances_dict: dict) -> None:
        """
        Create feature importance plots for models that expose useful importances.
        """
        non_empty_importances = {
            name: df
            for name, df in importances_dict.items()
            if df is not None and not df.empty and df["Importance"].sum() > 0
        }
        if not non_empty_importances:
            return

        num_models = min(len(non_empty_importances), 8)
        plt.figure(figsize=(16, 4 * num_models))

        for i, (model_name, importance_df) in enumerate(list(non_empty_importances.items())[:8], 1):
            plt.subplot(num_models, 1, i)
            top_features = importance_df.head(10)
            plt.barh(top_features["Feature"], top_features["Importance"])
            plt.xlabel("Importance")
            plt.title(f"Top 10 Important Features ({model_name})")
            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_all_models.png"))
        plt.close()

    def create_prediction_scatter_plot(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
    ) -> None:
        """
        Create scatter plot of predicted vs actual ratings.
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual Rating")
        plt.ylabel("Predicted Rating")
        plt.title(f"Actual vs Predicted Professor Ratings ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "prediction_scatter.png"))
        plt.close()

    def create_residual_plot(
        self,
        y_pred: np.ndarray,
        y_test: pd.Series,
        model_name: str,
    ) -> None:
        """
        Create residual plot for model evaluation.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, y_pred - y_test, alpha=0.5)
        plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), color="red", linestyle="--")
        plt.xlabel("Predicted Rating")
        plt.ylabel("Residuals")
        plt.title(f"{model_name} Residual Plot")
        plt.grid(True, linestyle="--", alpha=0.7)
        safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_residuals.png"))
        plt.close()
