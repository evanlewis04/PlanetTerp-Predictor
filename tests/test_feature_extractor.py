from __future__ import annotations

import unittest

from src.feature_extractor import FeatureExtractor
from utils.helpers import extract_course_level, filter_valid_grades
from config.config import GRADE_MAP


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = FeatureExtractor()
        self.reviews = [
            {
                "course": "CMSC131",
                "expected_grade": "A",
                "review": "Clear lecture and helpful office hours!",
                "rating": 5,
            },
            {
                "course": "CMSC330",
                "expected_grade": "B",
                "review": "Difficult exams but fair grading.",
                "rating": 4,
            },
            {
                "course": "CMSC graduate seminar",
                "expected_grade": "W",
                "review": "",
                "rating": 3,
            },
        ]

    def test_course_level_extraction_handles_course_codes(self) -> None:
        self.assertEqual(extract_course_level("CMSC131"), 1)
        self.assertEqual(extract_course_level("MATH410"), 4)
        self.assertEqual(extract_course_level("ENGL graduate seminar"), 0)

    def test_grade_mapping_filters_non_numeric_grades(self) -> None:
        self.assertEqual(filter_valid_grades(self.reviews, GRADE_MAP), [4.0, 3.0])

    def test_extract_course_and_grade_features(self) -> None:
        course_features = self.extractor.extract_course_features(self.reviews)
        grade_features = self.extractor.extract_grade_features(self.reviews)

        self.assertEqual(course_features["unique_courses"], 3)
        self.assertAlmostEqual(course_features["avg_course_level"], 2.0)
        self.assertAlmostEqual(course_features["lower_level_ratio"], 0.5)
        self.assertAlmostEqual(course_features["mid_level_ratio"], 0.5)
        self.assertEqual(grade_features["valid_expected_grade_count"], 2)
        self.assertAlmostEqual(grade_features["avg_expected_grade"], 3.5)
        self.assertAlmostEqual(grade_features["missing_expected_grade_ratio"], 0.0)

    def test_extract_text_features(self) -> None:
        review_stats = self.extractor.extract_review_stats(self.reviews)
        keyword_features = self.extractor.extract_keyword_features(self.reviews)
        readability_features = self.extractor.extract_readability_features(self.reviews)

        self.assertEqual(review_stats["num_reviews"], 3)
        self.assertAlmostEqual(review_stats["empty_review_ratio"], 1 / 3)
        self.assertGreater(keyword_features["clarity_keyword_count"], 0)
        self.assertGreater(keyword_features["difficulty_keyword_count"], 0)
        self.assertEqual(readability_features["avg_sentence_count"], 1)
        self.assertGreater(readability_features["exclamation_mark_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
