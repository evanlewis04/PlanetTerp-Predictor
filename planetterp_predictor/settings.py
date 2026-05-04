"""Structured application settings.

The project currently avoids mandatory runtime configuration dependencies so it can
still run in the original virtual environment. Values can be overridden with
environment variables prefixed by `PLANETTERP_`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_csv_floats(name: str, default: list[float]) -> list[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of floats") from exc


@dataclass(frozen=True)
class Settings:
    max_professors: int = 1000
    min_reviews: int = 10
    professors_per_batch: int = 100
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 10
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    output_dir: str = "outputs"
    figure_dpi: int = 300
    figure_format: str = "png"
    log_level: str = "INFO"
    progress_interval: int = 100

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_professors=_env_int("PLANETTERP_MAX_PROFESSORS", cls.max_professors),
            min_reviews=_env_int("PLANETTERP_MIN_REVIEWS", cls.min_reviews),
            professors_per_batch=_env_int(
                "PLANETTERP_PROFESSORS_PER_BATCH", cls.professors_per_batch
            ),
            random_state=_env_int("PLANETTERP_RANDOM_STATE", cls.random_state),
            test_size=_env_float("PLANETTERP_TEST_SIZE", cls.test_size),
            cv_folds=_env_int("PLANETTERP_CV_FOLDS", cls.cv_folds),
            ridge_alphas=tuple(_env_csv_floats("PLANETTERP_RIDGE_ALPHAS", list(cls.ridge_alphas))),
            output_dir=os.getenv("PLANETTERP_OUTPUT_DIR", cls.output_dir),
            figure_dpi=_env_int("PLANETTERP_FIGURE_DPI", cls.figure_dpi),
            figure_format=os.getenv("PLANETTERP_FIGURE_FORMAT", cls.figure_format),
            log_level=os.getenv("PLANETTERP_LOG_LEVEL", cls.log_level),
            progress_interval=_env_int("PLANETTERP_PROGRESS_INTERVAL", cls.progress_interval),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


settings = Settings.from_env()
