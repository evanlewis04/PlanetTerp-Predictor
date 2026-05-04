"""Validation helpers for PlanetTerp professor datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ValidationReport:
    total_professors: int = 0
    invalid_professor_records: int = 0
    missing_professor_names: int = 0
    missing_reviews_field: int = 0
    invalid_reviews_field: int = 0
    total_reviews: int = 0
    invalid_review_records: int = 0
    missing_review_text: int = 0
    missing_ratings: int = 0
    invalid_ratings: int = 0
    missing_courses: int = 0
    missing_expected_grades: int = 0
    duplicate_reviews: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_professors(professors: list[dict[str, Any]]) -> ValidationReport:
    """Validate a professor/review collection and return aggregate findings."""
    report = ValidationReport(total_professors=len(professors))
    seen_review_keys: set[tuple[Any, ...]] = set()

    for professor in professors:
        if not isinstance(professor, dict):
            report.invalid_professor_records += 1
            continue

        professor_name = professor.get("name")
        if not professor_name:
            report.missing_professor_names += 1

        if "reviews" not in professor:
            report.missing_reviews_field += 1
            continue

        reviews = professor.get("reviews")
        if not isinstance(reviews, list):
            report.invalid_reviews_field += 1
            continue

        for review in reviews:
            report.total_reviews += 1
            if not isinstance(review, dict):
                report.invalid_review_records += 1
                continue

            rating = review.get("rating")
            course = review.get("course")
            expected_grade = review.get("expected_grade")
            review_text = review.get("review")

            if not review_text:
                report.missing_review_text += 1
            if rating is None:
                report.missing_ratings += 1
            elif not isinstance(rating, int | float) or not 0 <= float(rating) <= 5:
                report.invalid_ratings += 1
            if not course:
                report.missing_courses += 1
            if not expected_grade:
                report.missing_expected_grades += 1

            review_key = (professor_name, course, rating, expected_grade, review_text)
            if review_key in seen_review_keys:
                report.duplicate_reviews += 1
            else:
                seen_review_keys.add(review_key)

    if report.total_professors == 0:
        report.warnings.append("Dataset contains no professor records.")
    if report.total_reviews == 0:
        report.warnings.append("Dataset contains no reviews.")
    if report.invalid_ratings:
        report.warnings.append("Some reviews have ratings outside the expected 0-5 range.")
    if report.missing_ratings:
        report.warnings.append("Some reviews are missing ratings.")

    return report
