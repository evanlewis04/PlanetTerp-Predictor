# Data Dictionary

This document describes the major data objects used by the project. For the exact feature-by-feature model catalog, see [feature_catalog.md](feature_catalog.md).

## Raw Professor Snapshot

Raw snapshots are saved under:

```text
data/raw/professors_<snapshot_id>.json
```

Each snapshot contains:

| Field | Type | Description |
| --- | --- | --- |
| `metadata.snapshot_id` | string | Timestamp-style snapshot identifier. |
| `metadata.created_at` | string | Local creation timestamp. |
| `metadata.source` | string | Source description for the fetched data. |
| `metadata.max_professors` | integer | Fetch limit used for the snapshot. |
| `metadata.min_reviews` | integer | Review threshold recorded with the fetch. |
| `metadata.professor_count` | integer | Number of professor records saved. |
| `metadata.settings` | object | Effective project settings at snapshot time. |
| `professors` | array | Raw professor records returned by the PlanetTerp workflow. |

## Professor Record

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Professor display name. |
| `department` | string | Department label when available. |
| `reviews` | array | Review records attached to the professor. |

## Review Record

| Field | Type | Description |
| --- | --- | --- |
| `rating` | number | Student rating used to calculate the `avg_rating` target. |
| `course` | string | Course code or course label associated with the review. |
| `expected_grade` | string | Student expected grade, mapped to a GPA-like numeric value when possible. |
| `review` | string | Free-text review body used for length, sentiment, keyword, and readability features. |

## Processed Dataset Summary

Dataset summaries are saved under:

```text
data/processed/dataset_summary_<snapshot_id>.json
```

Important fields:

| Field | Description |
| --- | --- |
| `professor_count` | Raw professor records in the snapshot. |
| `retained_professor_count` | Professors meeting the selected minimum review threshold. |
| `total_review_count` | Total reviews across all raw professor records. |
| `retained_review_count` | Reviews attached to retained professors. |
| `review_text_count` | Reviews with non-empty text. |
| `review_count_distribution` | Count, min, max, mean, and median review counts. |
| `rating_distribution` | Count, min, max, mean, and median review ratings. |
| `department_distribution` | Professor counts by department. |
| `top_courses` | Most common course labels in the snapshot. |
| `expected_grade_distribution` | Expected grade frequencies. |
| `validation` | Aggregate validation findings and warnings. |

## Feature CSV

Feature datasets are saved under:

```text
data/processed/features_<snapshot_id>.csv
```

The target column is:

| Field | Description |
| --- | --- |
| `avg_rating` | Mean numeric review rating for a professor. |

Feature groups:

| Group | Examples | Purpose |
| --- | --- | --- |
| Review volume and length | `num_reviews`, `avg_review_length`, `empty_review_ratio` | Captures review quantity and text density. |
| Course mix | `unique_courses`, `avg_course_level`, `upper_level_ratio` | Captures course breadth and level mix. |
| Expected grade | `avg_expected_grade`, `grade_a_ratio`, `missing_expected_grade_ratio` | Captures grade expectations and missingness. |
| Sentiment | `avg_sentiment_compound`, `positive_review_ratio` | Captures VADER review tone. |
| Keyword categories | `clarity_keyword_count`, `workload_review_ratio` | Captures interpretable review themes. |
| Readability and punctuation | `avg_words_per_sentence`, `question_mark_ratio` | Captures lightweight writing style features. |
| Department | `department_*` | One-hot encoded department labels. |

## Experiment Run

Experiment runs are saved under:

```text
experiments/runs/<run_id>/
```

Key artifacts:

| Artifact | Description |
| --- | --- |
| `metadata.json` | Run settings, snapshot metadata, feature columns, target summary, Git commit, and best model name. |
| `cross_validation_metrics.csv` | Cross-validation R2, RMSE, MAE, and MSE summaries. |
| `holdout_metrics.csv` | Holdout R2, adjusted R2, RMSE, MAE, median AE, and MSE. |
| `best_feature_importance.csv` | Best-model native feature importances when available. |
| `best_model.joblib` | Saved model bundle used by `/api/predict`. |
| `plots/` | Copied PNG plots for the run. |
