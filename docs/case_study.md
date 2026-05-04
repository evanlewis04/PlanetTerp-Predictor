# Technical Case Study

## Goal

The original project trained a small set of regressors and printed metrics to the terminal. The upgrade goal was to turn that prototype into a reproducible application that can answer:

- What data produced this model?
- Which model performed best?
- Which features mattered most?
- Can a user inspect and reuse the result without reading Python logs?

## Baseline Problem

The starting workflow depended on live API calls, had limited feature engineering, compared only a few models, and did not save reusable model artifacts. This made model results hard to reproduce and hard to explain.

## Upgrade Strategy

The project improved model development in five main steps:

1. Added timestamped raw snapshots so training can be repeated on the same input data.
2. Added validation and dataset summaries so small or messy samples are visible.
3. Expanded feature engineering from basic review/course/grade features into 61 saved feature columns.
4. Added a broader sklearn model registry with baselines, regularized linear models, tree ensembles, boosting, KNN, and SVR.
5. Saved every experiment run with metrics, plots, feature importances, metadata, and a `joblib` model bundle.

## Example Smoke Run

Run:

```text
20260504_102211_phase5-smoke
```

Snapshot:

```text
data/raw/professors_20260503_213043.json
```

Sample size:

| Quantity | Value |
| --- | ---: |
| Professors fetched | 80 |
| Model-ready professor rows | 41 |
| Feature columns | 61 |
| Target mean | 3.522 |
| Target range | 1.0 to 5.0 |

Top holdout results:

| Model | R2 | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Random Forest | 0.378 | 0.640 | 0.601 |
| Elastic Net | 0.327 | 0.666 | 0.563 |
| Lasso Regression | 0.326 | 0.666 | 0.580 |
| Extra Trees | 0.292 | 0.683 | 0.630 |
| Ridge Regression | 0.288 | 0.685 | 0.599 |

The best holdout R2 model in this smoke run was Random Forest. Cross-validation also ranked Random Forest first by R2 mean, but with a high standard deviation. That variance is expected for a small 41-row modeling dataset.

## Feature Signals

Top saved Random Forest feature importances:

| Feature | Importance |
| --- | ---: |
| `avg_sentiment_compound` | 0.083 |
| `positive_review_ratio` | 0.065 |
| `avg_words_per_sentence` | 0.061 |
| `negative_review_ratio` | 0.049 |
| `avg_sentiment_neg` | 0.047 |
| `workload_keyword_count` | 0.034 |
| `sentiment_variance` | 0.033 |
| `avg_sentiment_pos` | 0.032 |

The strongest signals are mostly review-text sentiment and text-shape features. That is plausible, but it also reinforces the need for careful leakage and availability notes.

## Product Outcome

The result is no longer just a model script. The local app can now:

- Show saved experiment runs.
- Compare holdout and cross-validation metrics.
- Display copied plot artifacts.
- Inspect model metadata and feature importances.
- Load a saved model bundle for predictions through the API.
- Verify dashboard behavior with a browser smoke test.

## Next Improvements

- Train on larger snapshots before treating performance as stable.
- Add an ablation experiment that compares feature groups with and without sentiment and expected-grade features.
- Add permutation importance or SHAP for model-agnostic interpretation.
- Move prediction-time feature construction into a shared pipeline.
- Containerize the API, dashboard, and artifact volumes in Phase 10.
