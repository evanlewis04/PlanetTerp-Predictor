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

## Latest Local Refresh Run

Run:

```text
20260507_185741_live-500-min5
```

Snapshot:

```text
data/raw/professors_20260507_185648.json
```

Sample size:

| Quantity | Value |
| --- | ---: |
| Professors fetched | 500 |
| Model-ready professor rows | 150 |
| Minimum reviews | 5 |
| Feature columns | 61 |
| Target mean | 3.542 |
| Target range | 1.44 to 5.0 |

Top holdout results:

| Model | R2 | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Extra Trees | 0.808 | 0.339 | 0.278 |
| Random Forest | 0.807 | 0.340 | 0.285 |
| Ridge Regression | 0.785 | 0.358 | 0.300 |
| Lasso Regression | 0.744 | 0.391 | 0.322 |
| Elastic Net | 0.737 | 0.397 | 0.336 |

The best holdout R2 model in this refresh run was Extra Trees, closely followed by Random Forest. Cross-validation also ranked Extra Trees first by R2 mean at 0.762 with a standard deviation of 0.062, which is much more stable than the earlier 41-row smoke run.

## Feature Signals

Top saved Extra Trees feature importances:

| Feature | Importance |
| --- | ---: |
| `avg_sentiment_compound` | 0.125 |
| `positive_review_ratio` | 0.111 |
| `avg_sentiment_pos` | 0.096 |
| `negative_review_ratio` | 0.075 |
| `avg_sentiment_neg` | 0.064 |
| `sentiment_variance` | 0.064 |
| `num_reviews` | 0.042 |
| `neutral_review_ratio` | 0.042 |

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

- Add an ablation experiment that compares feature groups with and without sentiment and expected-grade features.
- Add permutation importance or SHAP for model-agnostic interpretation.
- Move prediction-time feature construction into a shared pipeline.
- Add frontend component tests for the dashboard's empty, error, and metric-table states.
