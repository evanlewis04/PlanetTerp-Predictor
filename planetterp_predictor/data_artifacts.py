"""Data snapshot and summary artifact helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from planetterp_predictor.data_validation import validate_professors
from planetterp_predictor.settings import settings

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def ensure_data_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def save_raw_snapshot(
    professors: list[dict[str, Any]],
    *,
    max_professors: int,
    min_reviews: int,
    raw_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Save a timestamped raw API snapshot and return its path."""
    ensure_data_directories()
    snapshot_id = _timestamp()
    path = raw_dir / f"professors_{snapshot_id}.json"
    payload = {
        "metadata": {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "PlanetTerp API via planetterp Python package",
            "max_professors": max_professors,
            "min_reviews": min_reviews,
            "professor_count": len(professors),
            "settings": settings.to_dict(),
        },
        "professors": professors,
    }
    _write_json(path, payload)
    return path


def load_raw_snapshot(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load professors from a snapshot file.

    Supports the Phase 2 snapshot format and plain list JSON files for easier
    local experimentation.
    """
    snapshot_path = Path(path)
    payload = _read_json(snapshot_path)
    if isinstance(payload, list):
        return payload, {"snapshot_path": str(snapshot_path)}
    if isinstance(payload, dict) and isinstance(payload.get("professors"), list):
        metadata = payload.get("metadata", {})
        metadata["snapshot_path"] = str(snapshot_path)
        return payload["professors"], metadata
    raise ValueError(f"Unsupported snapshot format: {snapshot_path}")


def latest_raw_snapshot(raw_dir: Path = RAW_DATA_DIR) -> Path | None:
    if not raw_dir.exists():
        return None
    snapshots = sorted(raw_dir.glob("professors_*.json"))
    return snapshots[-1] if snapshots else None


def _numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def build_dataset_summary(
    professors: list[dict[str, Any]],
    *,
    min_reviews: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary for a professor dataset."""
    valid_professors = [
        professor
        for professor in professors
        if isinstance(professor, dict)
        and isinstance(professor.get("reviews"), list)
        and len(professor["reviews"]) >= min_reviews
    ]
    review_counts = [
        len(professor.get("reviews", []))
        for professor in professors
        if isinstance(professor, dict) and isinstance(professor.get("reviews", []), list)
    ]
    retained_review_counts = [
        len(professor.get("reviews", []))
        for professor in valid_professors
        if isinstance(professor, dict) and isinstance(professor.get("reviews", []), list)
    ]

    departments = Counter(
        professor.get("department", "Unknown")
        for professor in professors
        if isinstance(professor, dict)
    )

    ratings: list[float] = []
    expected_grades = Counter()
    courses = Counter()
    review_text_count = 0
    for professor in professors:
        if not isinstance(professor, dict):
            continue
        for review in professor.get("reviews", []) or []:
            if not isinstance(review, dict):
                continue
            rating = review.get("rating")
            if isinstance(rating, int | float):
                ratings.append(float(rating))
            expected_grade = review.get("expected_grade")
            if expected_grade:
                expected_grades[str(expected_grade)] += 1
            course = review.get("course")
            if course:
                courses[str(course)] += 1
            if review.get("review"):
                review_text_count += 1

    validation = validate_professors(professors)

    return {
        "metadata": metadata or {},
        "min_reviews": min_reviews,
        "professor_count": len(professors),
        "retained_professor_count": len(valid_professors),
        "total_review_count": sum(review_counts),
        "retained_review_count": sum(retained_review_counts),
        "review_text_count": review_text_count,
        "review_count_distribution": _numeric_summary(review_counts),
        "retained_review_count_distribution": _numeric_summary(retained_review_counts),
        "rating_distribution": _numeric_summary(ratings),
        "department_distribution": dict(departments.most_common()),
        "top_courses": dict(courses.most_common(20)),
        "expected_grade_distribution": dict(expected_grades.most_common()),
        "validation": validation.to_dict(),
    }


def save_dataset_summary(
    summary: dict[str, Any],
    *,
    processed_dir: Path = PROCESSED_DATA_DIR,
    label: str | None = None,
) -> Path:
    ensure_data_directories()
    suffix = label or _timestamp()
    path = processed_dir / f"dataset_summary_{suffix}.json"
    _write_json(path, summary)
    return path


def save_features_dataset(
    features: pd.DataFrame,
    *,
    processed_dir: Path = PROCESSED_DATA_DIR,
    label: str | None = None,
) -> Path:
    ensure_data_directories()
    suffix = label or _timestamp()
    path = processed_dir / f"features_{suffix}.csv"
    features.to_csv(path, index=False)
    return path
