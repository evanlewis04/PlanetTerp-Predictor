# Feature Catalog

This catalog documents the model features produced by `src/feature_extractor.py`.

The target variable is `avg_rating`, calculated from PlanetTerp review ratings for each professor. Features should avoid directly summarizing those same ratings because that would leak the answer into the model.

## Leakage Policy

The Phase 3 feature pass intentionally does not add rating-derived predictors such as median rating, rating variance, or rating skew. Those may be useful for descriptive analytics, but they are not valid predictors when the modeling target is average rating from the same reviews.

## Professor Identity

| Feature | Source | Transformation | Leakage Risk |
| --- | --- | --- | --- |
| `department_*` | Professor department | One-hot encoded category | Low |

## Review Volume And Length

| Feature | Source | Transformation | Leakage Risk |
| --- | --- | --- | --- |
| `num_reviews` | Reviews list | Count of reviews | Low |
| `has_reviews` | Reviews list | Binary indicator | Low |
| `avg_review_length` | Review text | Average character count | Low |
| `median_review_length` | Review text | Median character count | Low |
| `review_length_variance` | Review text | Variance of character count | Low |
| `avg_review_word_count` | Review text | Average word count | Low |
| `median_review_word_count` | Review text | Median word count | Low |
| `empty_review_ratio` | Review text | Share of reviews without text | Low |

## Course Mix

| Feature | Source | Transformation | Leakage Risk |
| --- | --- | --- | --- |
| `unique_courses` | Review course codes | Count of unique courses | Low |
| `avg_course_level` | Review course codes | Average numeric course level | Low |
| `course_level_variance` | Review course codes | Variance of numeric course level | Low |
| `lower_level_ratio` | Review course codes | Share of 100/200-level courses | Low |
| `mid_level_ratio` | Review course codes | Share of 300-level courses | Low |
| `upper_level_ratio` | Review course codes | Share of 300-level-or-above courses | Low |
| `graduate_level_ratio` | Review course codes | Share of 500-level-or-above courses | Low |
| `lower_level_course_count` | Review course codes | Count of 100/200-level course reviews | Low |
| `mid_level_course_count` | Review course codes | Count of 300-level course reviews | Low |
| `upper_level_course_count` | Review course codes | Count of 400-level course reviews | Low |
| `graduate_level_course_count` | Review course codes | Count of 500-level-or-above course reviews | Low |

## Expected Grade

| Feature | Source | Transformation | Leakage Risk |
| --- | --- | --- | --- |
| `avg_expected_grade` | Review expected grades | GPA-scale average | Medium |
| `grade_variance` | Review expected grades | GPA-scale variance | Medium |
| `grade_a_ratio` | Review expected grades | Share of A/A- expected grades | Medium |
| `grade_b_or_above_ratio` | Review expected grades | Share of B-or-better expected grades | Medium |
| `grade_c_or_below_ratio` | Review expected grades | Share of C-or-worse expected grades | Medium |
| `valid_expected_grade_count` | Review expected grades | Count of mappable expected grades | Low |
| `missing_expected_grade_ratio` | Review expected grades | Share of reviews missing expected grade | Low |

Expected grade features are marked medium risk because student-perceived grades may be correlated with professor ratings and may not be available in every real prediction scenario. They are acceptable for this exploratory model but should be toggled in future experiments.

## Sentiment

| Feature | Source | Transformation | Leakage Risk |
| --- | --- | --- | --- |
| `avg_sentiment_pos` | Review text | Mean VADER positive sentiment | Medium |
| `avg_sentiment_neg` | Review text | Mean VADER negative sentiment | Medium |
| `avg_sentiment_compound` | Review text | Mean VADER compound sentiment | Medium |
| `sentiment_variance` | Review text | Variance of VADER compound sentiment | Medium |
| `positive_review_ratio` | Review text | Share with compound sentiment >= 0.05 | Medium |
| `negative_review_ratio` | Review text | Share with compound sentiment <= -0.05 | Medium |
| `neutral_review_ratio` | Review text | Share with neutral compound sentiment | Medium |

Sentiment features are useful but should be interpreted carefully because review tone is naturally close to the rating being predicted.

## Keyword Categories

Each category produces:

- `{category}_keyword_count`
- `{category}_keyword_density`
- `{category}_review_ratio`

Categories:

- `difficulty`
- `clarity`
- `helpfulness`
- `workload`
- `exams`
- `grading`
- `lectures`
- `attendance`

These features are interpretable text aggregates. Leakage risk is medium because they summarize review content, which is closely related to the rating, but they do not directly use rating values.

## Readability And Punctuation

| Feature | Source | Transformation | Leakage Risk |
| --- | --- | --- | --- |
| `avg_sentence_count` | Review text | Average sentence count per review | Low |
| `avg_words_per_sentence` | Review text | Average words per sentence | Low |
| `question_mark_ratio` | Review text | Question marks per text review | Low |
| `exclamation_mark_ratio` | Review text | Exclamation marks per text review | Low |

