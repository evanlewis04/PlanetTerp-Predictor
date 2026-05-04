"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    run_id: str = Field(..., description="Experiment run id to load.")
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Feature values keyed by saved model feature column.",
    )
    fill_missing: bool = Field(
        default=True,
        description="Fill missing feature columns with 0.0 instead of rejecting the request.",
    )


class PredictionResponse(BaseModel):
    run_id: str
    model_name: str
    prediction: float
    missing_features: list[str]
    extra_features: list[str]


class TrainRequest(BaseModel):
    max_professors: int = 1000
    min_reviews: int = 10
    snapshot: str | None = Field(
        default="latest",
        description="Raw snapshot path, 'latest', or null to fetch live data.",
    )
    experiment_name: str | None = None
    save_experiment: bool = True


class TrainResponse(BaseModel):
    status: str
    latest_run_id: str | None = None
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str


class RunSummary(BaseModel):
    run_id: str
    created_at: str | None = None
    experiment_name: str | None = None
    best_model_name: str | None = None
    feature_count: int | None = None
    git_commit: str | None = None


class PlotInfo(BaseModel):
    name: str
    url: str


class ModelInfo(BaseModel):
    name: str
    needs_scaling: bool
    tuned: bool
    supports_native_importance: bool
    tune_grid: dict[str, list[Any]]
