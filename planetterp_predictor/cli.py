"""Command line interface for repeatable project workflows."""

from __future__ import annotations

import argparse
import json
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


def run_analysis(args: argparse.Namespace) -> int:
    from main import run_planetterp_analysis

    model, _, X, _ = run_planetterp_analysis(
        num_professors=args.max_professors,
        min_reviews=args.min_reviews,
    )
    if model is None or X is None:
        return 1
    return 0


def fetch_data(args: argparse.Namespace) -> int:
    from src.data_processor import PlanetTerpDataProcessor
    from utils.helpers import filter_valid_reviews

    processor = PlanetTerpDataProcessor()
    professors = processor.fetch_professor_data(max_professors=args.max_professors)
    valid_professors = filter_valid_reviews(professors, args.min_reviews)

    print(json.dumps({
        "fetched_professors": len(professors),
        "valid_professors": len(valid_professors),
        "min_reviews": args.min_reviews,
    }, indent=2))
    return 0


def print_config(_: argparse.Namespace) -> int:
    print(json.dumps(settings.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="planetterp-predictor",
        description="Run PlanetTerp professor rating prediction workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full training and evaluation workflow.")
    _add_dataset_args(run_parser)
    run_parser.set_defaults(func=run_analysis)

    data_parser = subparsers.add_parser("data", help="Data workflow commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    fetch_parser = data_subparsers.add_parser(
        "fetch",
        help="Fetch professor and review data and report retained sample size.",
    )
    _add_dataset_args(fetch_parser)
    fetch_parser.set_defaults(func=fetch_data)

    config_parser = subparsers.add_parser("config", help="Print effective project settings.")
    config_parser.set_defaults(func=print_config)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
