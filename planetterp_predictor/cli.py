"""Command line interface for repeatable project workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from planetterp_predictor.settings import settings


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-professors",
        type=_positive_int,
        default=settings.max_professors,
        help=f"Maximum professors to fetch. Default: {settings.max_professors}.",
    )
    parser.add_argument(
        "--min-reviews",
        type=_non_negative_int,
        default=settings.min_reviews,
        help=f"Minimum reviews required per professor. Default: {settings.min_reviews}.",
    )


def _add_min_reviews_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-reviews",
        type=_non_negative_int,
        default=settings.min_reviews,
        help=f"Minimum reviews required per professor. Default: {settings.min_reviews}.",
    )


def _add_snapshot_arg(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--snapshot",
        default=None if required else "latest",
        required=required,
        help="Path to a raw professors snapshot JSON file, or 'latest'.",
    )


def _resolve_snapshot_path(snapshot: str | None) -> Path | None:
    from planetterp_predictor.data_artifacts import latest_raw_snapshot

    if snapshot is None:
        return None
    if snapshot == "latest":
        return latest_raw_snapshot()
    return Path(snapshot)


def run_analysis(args: argparse.Namespace) -> int:
    from main import run_planetterp_analysis
    from planetterp_predictor.data_artifacts import load_raw_snapshot

    professors = None
    metadata = None
    snapshot_path = _resolve_snapshot_path(args.snapshot)
    if args.snapshot and snapshot_path is None:
        print("No raw data snapshot found. Run `data fetch` first.")
        return 1
    if snapshot_path is not None:
        professors, metadata = load_raw_snapshot(snapshot_path)
        print(f"Using snapshot: {metadata.get('snapshot_path', snapshot_path)}")

    model, _, X, _ = run_planetterp_analysis(
        num_professors=args.max_professors,
        min_reviews=args.min_reviews,
        professors=professors,
        snapshot_metadata=metadata,
        experiment_name=args.experiment_name,
        save_experiment=not args.no_save_experiment,
    )
    if model is None or X is None:
        return 1
    return 0


def fetch_data(args: argparse.Namespace) -> int:
    from src.data_processor import PlanetTerpDataProcessor
    from planetterp_predictor.data_artifacts import (
        build_dataset_summary,
        save_dataset_summary,
        save_raw_snapshot,
    )

    processor = PlanetTerpDataProcessor()
    professors = processor.fetch_professor_data(max_professors=args.max_professors)
    snapshot_path = save_raw_snapshot(
        professors,
        max_professors=args.max_professors,
        min_reviews=args.min_reviews,
    )
    summary = build_dataset_summary(
        professors,
        min_reviews=args.min_reviews,
        metadata={"snapshot_path": str(snapshot_path)},
    )
    summary_path = save_dataset_summary(summary, label=snapshot_path.stem.removeprefix("professors_"))

    print(json.dumps({
        "fetched_professors": len(professors),
        "valid_professors": summary["retained_professor_count"],
        "min_reviews": args.min_reviews,
        "snapshot_path": str(snapshot_path),
        "summary_path": str(summary_path),
    }, indent=2))
    return 0


def validate_data(args: argparse.Namespace) -> int:
    from planetterp_predictor.data_artifacts import load_raw_snapshot
    from planetterp_predictor.data_validation import validate_professors

    snapshot_path = _resolve_snapshot_path(args.snapshot)
    if snapshot_path is None:
        print("No raw data snapshot found. Run `data fetch` first.")
        return 1
    professors, metadata = load_raw_snapshot(snapshot_path)
    report = validate_professors(professors)
    print(json.dumps({
        "snapshot": metadata,
        "validation": report.to_dict(),
    }, indent=2, sort_keys=True))
    return 0


def summarize_data(args: argparse.Namespace) -> int:
    from planetterp_predictor.data_artifacts import (
        build_dataset_summary,
        load_raw_snapshot,
        save_dataset_summary,
    )

    snapshot_path = _resolve_snapshot_path(args.snapshot)
    if snapshot_path is None:
        print("No raw data snapshot found. Run `data fetch` first.")
        return 1
    professors, metadata = load_raw_snapshot(snapshot_path)
    summary = build_dataset_summary(professors, min_reviews=args.min_reviews, metadata=metadata)
    summary_path = save_dataset_summary(summary, label=snapshot_path.stem.removeprefix("professors_"))
    print(json.dumps({
        "summary_path": str(summary_path),
        "professor_count": summary["professor_count"],
        "retained_professor_count": summary["retained_professor_count"],
        "total_review_count": summary["total_review_count"],
        "validation_warnings": summary["validation"]["warnings"],
    }, indent=2, sort_keys=True))
    return 0


def build_features(args: argparse.Namespace) -> int:
    from src.feature_extractor import FeatureExtractor
    from planetterp_predictor.data_artifacts import (
        build_dataset_summary,
        load_raw_snapshot,
        save_dataset_summary,
        save_features_dataset,
    )
    from utils.helpers import filter_valid_reviews

    snapshot_path = _resolve_snapshot_path(args.snapshot)
    if snapshot_path is None:
        print("No raw data snapshot found. Run `data fetch` first.")
        return 1

    professors, metadata = load_raw_snapshot(snapshot_path)
    valid_professors = filter_valid_reviews(professors, args.min_reviews)
    X, y = FeatureExtractor().prepare_data_for_modeling(valid_professors)
    if X is None or y is None:
        print("Failed to build features: no model-ready professor records found.")
        return 1

    features = X.assign(avg_rating=y.values)
    label = snapshot_path.stem.removeprefix("professors_")
    features_path = save_features_dataset(features, label=label)
    summary = build_dataset_summary(professors, min_reviews=args.min_reviews, metadata=metadata)
    summary_path = save_dataset_summary(summary, label=label)

    print(json.dumps({
        "features_path": str(features_path),
        "summary_path": str(summary_path),
        "rows": len(features),
        "columns": list(features.columns),
    }, indent=2, sort_keys=True))
    return 0


def print_config(_: argparse.Namespace) -> int:
    print(json.dumps(settings.to_dict(), indent=2, sort_keys=True))
    return 0


def serve_api(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="planetterp-predictor",
        description="Run PlanetTerp professor rating prediction workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full training and evaluation workflow.")
    _add_dataset_args(run_parser)
    run_parser.add_argument(
        "--snapshot",
        default=None,
        help="Optional raw professors snapshot JSON file, or 'latest', to train from saved data.",
    )
    run_parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional human-readable name to include in the saved experiment run id.",
    )
    run_parser.add_argument(
        "--no-save-experiment",
        action="store_true",
        help="Run training without writing experiment artifacts.",
    )
    run_parser.set_defaults(func=run_analysis)

    data_parser = subparsers.add_parser("data", help="Data workflow commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    fetch_parser = data_subparsers.add_parser(
        "fetch",
        help="Fetch professor and review data and report retained sample size.",
    )
    _add_dataset_args(fetch_parser)
    fetch_parser.set_defaults(func=fetch_data)

    validate_parser = data_subparsers.add_parser(
        "validate",
        help="Validate a raw professor snapshot.",
    )
    _add_snapshot_arg(validate_parser)
    validate_parser.set_defaults(func=validate_data)

    summary_parser = data_subparsers.add_parser(
        "summary",
        help="Create a dataset summary JSON artifact from a raw snapshot.",
    )
    _add_min_reviews_arg(summary_parser)
    _add_snapshot_arg(summary_parser)
    summary_parser.set_defaults(func=summarize_data)

    features_parser = data_subparsers.add_parser(
        "build-features",
        help="Create a processed feature CSV from a raw snapshot.",
    )
    _add_min_reviews_arg(features_parser)
    _add_snapshot_arg(features_parser)
    features_parser.set_defaults(func=build_features)

    config_parser = subparsers.add_parser("config", help="Print effective project settings.")
    config_parser.set_defaults(func=print_config)

    serve_parser = subparsers.add_parser("serve-api", help="Start the FastAPI backend.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind. Default: 8000.")
    serve_parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    serve_parser.set_defaults(func=serve_api)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
