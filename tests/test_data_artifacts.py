from __future__ import annotations

import unittest

from planetterp_predictor.data_artifacts import build_dataset_summary


class DatasetSummaryTests(unittest.TestCase):
    def test_build_dataset_summary_counts_retained_records_and_distributions(self) -> None:
        professors = [
            {
                "name": "Ada Lovelace",
                "department": "CMSC",
                "reviews": [
                    {
                        "rating": 5,
                        "course": "CMSC131",
                        "expected_grade": "A",
                        "review": "Clear and helpful lectures.",
                    },
                    {
                        "rating": 4,
                        "course": "CMSC132",
                        "expected_grade": "B+",
                        "review": "Projects were fair.",
                    },
                ],
            },
            {
                "name": "Grace Hopper",
                "department": "MATH",
                "reviews": [
                    {
                        "rating": 3,
                        "course": "MATH140",
                        "expected_grade": "B",
                        "review": "",
                    }
                ],
            },
        ]

        summary = build_dataset_summary(
            professors,
            min_reviews=2,
            metadata={"snapshot_id": "fixture"},
        )

        self.assertEqual(summary["metadata"]["snapshot_id"], "fixture")
        self.assertEqual(summary["professor_count"], 2)
        self.assertEqual(summary["retained_professor_count"], 1)
        self.assertEqual(summary["total_review_count"], 3)
        self.assertEqual(summary["retained_review_count"], 2)
        self.assertEqual(summary["review_text_count"], 2)
        self.assertEqual(summary["department_distribution"], {"CMSC": 1, "MATH": 1})
        self.assertEqual(summary["top_courses"]["CMSC131"], 1)
        self.assertEqual(summary["expected_grade_distribution"]["A"], 1)
        self.assertEqual(summary["rating_distribution"]["count"], 3)
        self.assertAlmostEqual(summary["rating_distribution"]["mean"], 4.0)
        self.assertIn("warnings", summary["validation"])


if __name__ == "__main__":
    unittest.main()
