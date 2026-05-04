"""FastAPI application for PlanetTerp Predictor."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

from api.repository import ExperimentRepository
from api.schemas import (
    HealthResponse,
    ModelInfo,
    PlotInfo,
    PredictionRequest,
    PredictionResponse,
    RunSummary,
    TrainRequest,
    TrainResponse,
)
from main import run_planetterp_analysis
from planetterp_predictor.data_artifacts import latest_raw_snapshot, load_raw_snapshot
from planetterp_predictor.experiment_tracking import RUNS_DIR
from src.model_specs import get_model_specs

app = FastAPI(
    title="PlanetTerp Predictor API",
    version="0.6.0",
    description="Backend API for browsing experiment artifacts and serving predictions.",
)
repository = ExperimentRepository()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts/runs", StaticFiles(directory=str(RUNS_DIR)), name="run-artifacts")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="planetterp-predictor-api")


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return [RunSummary(**run) for run in repository.list_runs()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return repository.read_metadata(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> dict:
    try:
        return repository.read_metrics(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/plots", response_model=list[PlotInfo])
def get_run_plots(run_id: str) -> list[PlotInfo]:
    try:
        return [PlotInfo(**plot) for plot in repository.list_plots(run_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            name=spec.name,
            needs_scaling=spec.needs_scaling,
            tuned=bool(spec.tune_grid),
            supports_native_importance=spec.supports_native_importance,
            tune_grid=spec.tune_grid,
        )
        for spec in get_model_specs()
    ]


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        bundle = repository.load_model_bundle(request.run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    feature_columns = bundle["feature_columns"]
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in request.features
    ]
    extra_features = [
        feature
        for feature in request.features
        if feature not in feature_columns
    ]

    if missing_features and not request.fill_missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Prediction request is missing required feature columns.",
                "missing_features": missing_features,
            },
        )

    row = {
        feature: float(request.features.get(feature, 0.0))
        for feature in feature_columns
    }
    prediction = float(bundle["model"].predict(pd.DataFrame([row], columns=feature_columns))[0])
    return PredictionResponse(
        run_id=request.run_id,
        model_name=bundle["best_model_name"],
        prediction=prediction,
        missing_features=missing_features,
        extra_features=extra_features,
    )


@app.post("/api/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    professors = None
    metadata = None

    if request.snapshot is not None:
        snapshot_path = _resolve_snapshot(request.snapshot)
        if snapshot_path is None:
            raise HTTPException(
                status_code=404,
                detail="No raw data snapshot found. Run data fetch first.",
            )
        professors, metadata = load_raw_snapshot(snapshot_path)

    before_latest = repository.latest_run_id()
    model, _, _, _ = run_planetterp_analysis(
        num_professors=request.max_professors,
        min_reviews=request.min_reviews,
        professors=professors,
        snapshot_metadata=metadata,
        experiment_name=request.experiment_name,
        save_experiment=request.save_experiment,
    )
    if model is None:
        raise HTTPException(status_code=500, detail="Training failed.")

    after_latest = repository.latest_run_id()
    return TrainResponse(
        status="completed",
        latest_run_id=after_latest if after_latest != before_latest else None,
        message="Training completed successfully.",
    )


def _resolve_snapshot(snapshot: str) -> Path | None:
    if snapshot == "latest":
        return latest_raw_snapshot()
    return Path(snapshot)
