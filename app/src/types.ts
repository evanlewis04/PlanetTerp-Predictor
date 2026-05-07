export type RunSummary = {
  run_id: string;
  created_at: string | null;
  experiment_name: string | null;
  best_model_name: string | null;
  feature_count: number | null;
  git_commit: string | null;
};

export type ModelInfo = {
  name: string;
  needs_scaling: boolean;
  tuned: boolean;
  supports_native_importance: boolean;
  tune_grid: Record<string, unknown[]>;
};

export type PlotInfo = {
  name: string;
  url: string;
};

export type MetricsResponse = {
  cross_validation: MetricRecord[];
  holdout: MetricRecord[];
  best_feature_importance: FeatureImportance[];
};

export type MetricRecord = {
  Model: string;
  [key: string]: string | number | null;
};

export type FeatureImportance = {
  Feature: string;
  Importance: number;
};

export type RunMetadata = {
  run_id: string;
  created_at: string;
  experiment_name: string | null;
  best_model_name: string | null;
  feature_columns: string[];
  feature_count: number;
  feature_pipeline_version: string | null;
  git_commit: string | null;
  settings: Record<string, unknown>;
  snapshot: Record<string, unknown> | null;
  target_summary: {
    count?: number;
    min?: number;
    max?: number;
    mean?: number;
  } | null;
};

export type PredictionResponse = {
  run_id: string;
  model_name: string;
  prediction: number;
  missing_features: string[];
  extra_features: string[];
};

export type TrainRequest = {
  max_professors: number;
  min_reviews: number;
  snapshot: "latest" | null;
  experiment_name: string | null;
  save_experiment: boolean;
};

export type TrainResponse = {
  status: string;
  latest_run_id: string | null;
  message: string;
};

export type HealthResponse = {
  status: string;
  service: string;
};
