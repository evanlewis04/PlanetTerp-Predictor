"""Shared model registry for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config.config import RANDOM_STATE, RIDGE_ALPHAS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    tune_grid: dict[str, list[Any]] = field(default_factory=dict)
    needs_scaling: bool = False
    supports_native_importance: bool = False

    def build_estimator(self) -> Any:
        estimator = clone(self.estimator)
        if self.needs_scaling:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])
        return estimator


def _pipeline_grid(param_grid: dict[str, list[Any]], needs_scaling: bool) -> dict[str, list[Any]]:
    if not needs_scaling:
        return param_grid
    return {f"model__{key}": value for key, value in param_grid.items()}


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="Mean Baseline",
        estimator=DummyRegressor(strategy="mean"),
    ),
    ModelSpec(
        name="Median Baseline",
        estimator=DummyRegressor(strategy="median"),
    ),
    ModelSpec(
        name="Linear Regression",
        estimator=LinearRegression(),
        needs_scaling=True,
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Ridge Regression",
        estimator=Ridge(random_state=RANDOM_STATE),
        tune_grid={"alpha": RIDGE_ALPHAS},
        needs_scaling=True,
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Lasso Regression",
        estimator=Lasso(random_state=RANDOM_STATE, max_iter=10000),
        tune_grid={"alpha": [0.01, 0.1, 1.0]},
        needs_scaling=True,
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Elastic Net",
        estimator=ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        tune_grid={
            "alpha": [0.01, 0.1],
            "l1_ratio": [0.2, 0.8],
        },
        needs_scaling=True,
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Random Forest",
        estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        tune_grid={
            "max_depth": [None, 6],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt"],
        },
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Extra Trees",
        estimator=ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
        tune_grid={
            "max_depth": [None, 6],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt"],
        },
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Gradient Boosting",
        estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
        tune_grid={
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2],
        },
        supports_native_importance=True,
    ),
    ModelSpec(
        name="Hist Gradient Boosting",
        estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        tune_grid={
            "max_iter": [50, 100],
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15],
        },
    ),
    ModelSpec(
        name="KNN Regressor",
        estimator=KNeighborsRegressor(),
        tune_grid={
            "n_neighbors": [3, 7],
            "weights": ["uniform", "distance"],
        },
        needs_scaling=True,
    ),
    ModelSpec(
        name="SVR",
        estimator=SVR(),
        tune_grid={
            "C": [1.0, 5.0],
            "epsilon": [0.1],
            "kernel": ["rbf"],
        },
        needs_scaling=True,
    ),
]


def get_model_specs() -> list[ModelSpec]:
    return MODEL_SPECS


def get_tuning_grid(spec: ModelSpec) -> dict[str, list[Any]]:
    return _pipeline_grid(spec.tune_grid, spec.needs_scaling)
