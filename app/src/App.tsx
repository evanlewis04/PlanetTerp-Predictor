import {
  Activity,
  BarChart3,
  Brain,
  Database,
  FlaskConical,
  GalleryHorizontalEnd,
  Gauge,
  GitBranch,
  ListChecks,
  RefreshCw,
  Search,
  Sparkles,
  Table2,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  API_BASE_URL,
  artifactUrl,
  checkHealth,
  getRun,
  getRunMetrics,
  getRunPlots,
  listModels,
  listRuns,
  predict,
} from "./api";
import type {
  FeatureImportance,
  MetricRecord,
  MetricsResponse,
  ModelInfo,
  PlotInfo,
  PredictionResponse,
  RunMetadata,
  RunSummary,
} from "./types";

type ViewKey = "overview" | "dataset" | "comparison" | "features" | "plots" | "predict" | "runs";

type LoadState = "idle" | "loading" | "ready" | "error";

const navItems: Array<{ key: ViewKey; label: string; icon: typeof Gauge }> = [
  { key: "overview", label: "Overview", icon: Gauge },
  { key: "dataset", label: "Dataset", icon: Database },
  { key: "comparison", label: "Models", icon: BarChart3 },
  { key: "features", label: "Features", icon: Sparkles },
  { key: "plots", label: "Plots", icon: GalleryHorizontalEnd },
  { key: "predict", label: "Predict", icon: Brain },
  { key: "runs", label: "Runs", icon: ListChecks },
];

const predictionDefaults: Record<string, number> = {
  num_reviews: 10,
  avg_review_length: 360,
  avg_review_word_count: 62,
  unique_courses: 3,
  avg_course_level: 250,
  avg_expected_grade: 3.2,
  avg_sentiment_compound: 0.35,
  positive_review_ratio: 0.55,
  negative_review_ratio: 0.2,
  difficulty_keyword_count: 2,
  clarity_keyword_count: 3,
  helpfulness_keyword_count: 2,
  workload_keyword_count: 2,
  exams_keyword_count: 1,
};

const plotLabels: Record<string, string> = {
  "cross_validation_results.png": "Cross-validation performance",
  "model_comparison.png": "Holdout model comparison",
  "feature_importance_all_models.png": "Feature importance across models",
  "linear_regression_residuals.png": "Linear regression residuals",
  "ridge_regression_residuals.png": "Ridge regression residuals",
  "random_forest_residuals.png": "Random forest residuals",
  "ridge_alpha_tuning.png": "Ridge alpha tuning",
  "prediction_scatter.png": "Actual vs predicted ratings",
};

function App() {
  const [view, setView] = useState<ViewKey>("overview");
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<string>("unknown");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [plots, setPlots] = useState<PlotInfo[]>([]);

  async function loadDashboard(nextRunId?: string) {
    setLoadState("loading");
    setError(null);

    try {
      const [healthResponse, runResponse, modelResponse] = await Promise.all([
        checkHealth(),
        listRuns(),
        listModels(),
      ]);
      const chosenRunId = nextRunId || selectedRunId || runResponse[0]?.run_id || "";

      setHealth(healthResponse.status);
      setRuns(runResponse);
      setModels(modelResponse);
      setSelectedRunId(chosenRunId);

      if (chosenRunId) {
        const [runMetadata, runMetrics, runPlots] = await Promise.all([
          getRun(chosenRunId),
          getRunMetrics(chosenRunId),
          getRunPlots(chosenRunId),
        ]);
        setMetadata(runMetadata);
        setMetrics(runMetrics);
        setPlots(runPlots);
      } else {
        setMetadata(null);
        setMetrics(null);
        setPlots([]);
      }

      setLoadState("ready");
    } catch (caught) {
      setHealth("offline");
      setLoadState("error");
      setError(caught instanceof Error ? caught.message : "Dashboard data could not be loaded.");
    }
  }

  useEffect(() => {
    void loadDashboard();
  }, []);

  const bestHoldout = useMemo(() => bestBy(metrics?.holdout ?? [], "R2", "max"), [metrics]);
  const bestMae = useMemo(() => bestBy(metrics?.holdout ?? [], "MAE", "min"), [metrics]);
  const sortedHoldout = useMemo(
    () => [...(metrics?.holdout ?? [])].sort((a, b) => metricValue(b, "R2") - metricValue(a, "R2")),
    [metrics],
  );
  const sortedCv = useMemo(
    () =>
      [...(metrics?.cross_validation ?? [])].sort(
        (a, b) => metricValue(b, "R2 Mean") - metricValue(a, "R2 Mean"),
      ),
    [metrics],
  );
  const topFeatures = useMemo(
    () => [...(metrics?.best_feature_importance ?? [])].slice(0, 14),
    [metrics],
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">PT</div>
          <div>
            <p className="eyebrow">PlanetTerp</p>
            <h1>Predictor</h1>
          </div>
        </div>

        <nav className="nav-list" aria-label="Dashboard views">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                className={view === item.key ? "nav-item active" : "nav-item"}
                onClick={() => setView(item.key)}
                title={item.label}
                type="button"
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="api-panel">
          <div className={`status-dot ${health === "ok" ? "online" : ""}`} />
          <div>
            <span>API</span>
            <strong>{health}</strong>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Phase 7 Dashboard</p>
            <h2>{navItems.find((item) => item.key === view)?.label}</h2>
          </div>
          <div className="toolbar">
            <select
              aria-label="Experiment run"
              value={selectedRunId}
              onChange={(event) => void loadDashboard(event.target.value)}
            >
              {runs.length === 0 ? (
                <option value="">No saved runs</option>
              ) : (
                runs.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.experiment_name || run.run_id}
                  </option>
                ))
              )}
            </select>
            <button
              className="icon-button"
              onClick={() => void loadDashboard(selectedRunId)}
              title="Refresh dashboard"
              type="button"
            >
              <RefreshCw size={18} />
            </button>
          </div>
        </header>

        {loadState === "error" && <ErrorBanner error={error} />}
        {loadState === "loading" && <LoadingBand />}
        {loadState !== "loading" && runs.length === 0 && <EmptyRuns apiBaseUrl={API_BASE_URL} />}

        {runs.length > 0 && view === "overview" && (
          <Overview
            metadata={metadata}
            bestHoldout={bestHoldout}
            bestMae={bestMae}
            plots={plots}
            topFeatures={topFeatures}
          />
        )}
        {runs.length > 0 && view === "dataset" && <DatasetView metadata={metadata} />}
        {runs.length > 0 && view === "comparison" && (
          <ModelComparison holdout={sortedHoldout} cv={sortedCv} models={models} />
        )}
        {runs.length > 0 && view === "features" && (
          <FeatureImportance features={metrics?.best_feature_importance ?? []} metadata={metadata} />
        )}
        {runs.length > 0 && view === "plots" && <PlotGallery plots={plots} selectedRunId={selectedRunId} />}
        {runs.length > 0 && view === "predict" && (
          <PredictionExplorer runId={selectedRunId} metadata={metadata} />
        )}
        {runs.length > 0 && view === "runs" && (
          <RunsView runs={runs} selectedRunId={selectedRunId} models={models} />
        )}
      </main>
    </div>
  );
}

function Overview({
  metadata,
  bestHoldout,
  bestMae,
  plots,
  topFeatures,
}: {
  metadata: RunMetadata | null;
  bestHoldout: MetricRecord | null;
  bestMae: MetricRecord | null;
  plots: PlotInfo[];
  topFeatures: FeatureImportance[];
}) {
  const snapshot = metadata?.snapshot ?? {};
  const professorCount = numberFrom(snapshot.professor_count);
  const retainedCount = metadata?.target_summary?.count ?? null;
  const ratingMean = metadata?.target_summary?.mean ?? null;

  return (
    <section className="view-grid">
      <div className="kpi-grid">
        <Kpi icon={Brain} label="Best R2 Model" value={bestHoldout?.Model ?? "Unavailable"} detail={formatMetric(bestHoldout, "R2")} />
        <Kpi icon={Activity} label="Best MAE Model" value={bestMae?.Model ?? "Unavailable"} detail={formatMetric(bestMae, "MAE")} />
        <Kpi icon={Database} label="Model Rows" value={formatNumber(retainedCount)} detail={`${formatNumber(professorCount)} fetched`} />
        <Kpi icon={Table2} label="Feature Columns" value={formatNumber(metadata?.feature_count)} detail={metadata?.feature_pipeline_version ?? "No pipeline"} />
      </div>

      <div className="split-grid">
        <section className="panel">
          <PanelTitle icon={Gauge} title="Latest Run" />
          <div className="detail-list">
            <Detail label="Run ID" value={metadata?.run_id} />
            <Detail label="Experiment" value={metadata?.experiment_name} />
            <Detail label="Created" value={formatDate(metadata?.created_at)} />
            <Detail label="Git Commit" value={metadata?.git_commit?.slice(0, 12)} />
            <Detail label="Average Rating" value={formatNumber(ratingMean, 3)} />
          </div>
        </section>

        <section className="panel">
          <PanelTitle icon={Sparkles} title="Top Feature Signals" />
          <BarList rows={topFeatures.map((row) => ({ label: row.Feature, value: row.Importance }))} valueLabel="importance" />
        </section>
      </div>

      <section className="panel">
        <PanelTitle icon={GalleryHorizontalEnd} title="Run Plots" />
        <div className="plot-strip">
          {plots.slice(0, 4).map((plot) => (
            <figure key={plot.name}>
              <img src={artifactUrl(plot.url)} alt={plotLabels[plot.name] ?? plot.name} />
              <figcaption>{plotLabels[plot.name] ?? plot.name}</figcaption>
            </figure>
          ))}
        </div>
      </section>
    </section>
  );
}

function DatasetView({ metadata }: { metadata: RunMetadata | null }) {
  const snapshot = metadata?.snapshot ?? {};
  const settings = metadata?.settings ?? {};
  const target = metadata?.target_summary ?? {};

  return (
    <section className="view-grid">
      <div className="kpi-grid">
        <Kpi icon={Database} label="Fetched Professors" value={formatNumber(numberFrom(snapshot.professor_count))} detail={String(snapshot.snapshot_id ?? "snapshot")} />
        <Kpi icon={ListChecks} label="Model-Ready Rows" value={formatNumber(target.count)} detail={`min reviews ${String(snapshot.min_reviews ?? settings.min_reviews ?? "n/a")}`} />
        <Kpi icon={Gauge} label="Rating Range" value={`${formatNumber(target.min, 2)}-${formatNumber(target.max, 2)}`} detail={`mean ${formatNumber(target.mean, 3)}`} />
        <Kpi icon={GitBranch} label="Snapshot Source" value={String(snapshot.source ?? "Local artifacts")} detail={String(snapshot.created_at ?? "No timestamp")} />
      </div>

      <section className="panel">
        <PanelTitle icon={Table2} title="Dataset Metadata" />
        <div className="detail-grid">
          {Object.entries({
            "Snapshot Path": snapshot.snapshot_path,
            "Snapshot Created": snapshot.created_at,
            "Requested Professors": snapshot.max_professors,
            "Retained Rows": target.count,
            "Target Min": target.min,
            "Target Mean": target.mean,
            "Target Max": target.max,
            "CV Folds": settings.cv_folds,
            "Test Size": settings.test_size,
          }).map(([label, value]) => (
            <Detail key={label} label={label} value={formatLoose(value)} />
          ))}
        </div>
      </section>
    </section>
  );
}

function ModelComparison({
  holdout,
  cv,
  models,
}: {
  holdout: MetricRecord[];
  cv: MetricRecord[];
  models: ModelInfo[];
}) {
  return (
    <section className="view-grid">
      <section className="panel">
        <PanelTitle icon={BarChart3} title="Holdout Ranking" />
        <MetricBars rows={holdout} metric="R2" mode="max" />
        <MetricTable rows={holdout} columns={["Model", "R2", "RMSE", "MAE", "Median AE", "Best Params"]} />
      </section>

      <section className="panel">
        <PanelTitle icon={Activity} title="Cross-Validation Ranking" />
        <MetricTable rows={cv} columns={["Model", "R2 Mean", "R2 Std", "RMSE Mean", "MAE Mean"]} />
      </section>

      <section className="panel">
        <PanelTitle icon={FlaskConical} title="Model Registry" />
        <div className="registry-grid">
          {models.map((model) => (
            <div key={model.name} className="registry-row">
              <strong>{model.name}</strong>
              <span>{model.tuned ? "Tuned" : "Untuned"}</span>
              <span>{model.needs_scaling ? "Scaled" : "Native scale"}</span>
              <span>{model.supports_native_importance ? "Importance" : "No native importance"}</span>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
}

function FeatureImportance({
  features,
  metadata,
}: {
  features: FeatureImportance[];
  metadata: RunMetadata | null;
}) {
  const [query, setQuery] = useState("");
  const filtered = features.filter((row) => row.Feature.toLowerCase().includes(query.toLowerCase()));

  return (
    <section className="view-grid">
      <section className="panel">
        <div className="panel-heading with-actions">
          <PanelTitle icon={Sparkles} title="Best Model Feature Importance" />
          <label className="search-box">
            <Search size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter features"
            />
          </label>
        </div>
        <BarList rows={filtered.slice(0, 30).map((row) => ({ label: row.Feature, value: row.Importance }))} valueLabel="importance" />
      </section>

      <section className="panel">
        <PanelTitle icon={Table2} title="Feature Columns" />
        <div className="chip-grid">
          {(metadata?.feature_columns ?? []).map((feature) => (
            <span key={feature}>{feature}</span>
          ))}
        </div>
      </section>
    </section>
  );
}

function PlotGallery({ plots, selectedRunId }: { plots: PlotInfo[]; selectedRunId: string }) {
  return (
    <section className="plot-grid">
      {plots.map((plot) => (
        <figure key={plot.name} className="plot-card">
          <img src={artifactUrl(plot.url)} alt={plotLabels[plot.name] ?? plot.name} />
          <figcaption>
            <strong>{plotLabels[plot.name] ?? plot.name}</strong>
            <span>{selectedRunId}</span>
          </figcaption>
        </figure>
      ))}
    </section>
  );
}

function PredictionExplorer({ runId, metadata }: { runId: string; metadata: RunMetadata | null }) {
  const [features, setFeatures] = useState<Record<string, number>>(predictionDefaults);
  const [reviewText, setReviewText] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function updateFeature(name: string, value: number) {
    setFeatures((current) => ({ ...current, [name]: Number.isFinite(value) ? value : 0 }));
  }

  function deriveFromText(text: string) {
    setReviewText(text);
    const derived = deriveTextFeatures(text);
    setFeatures((current) => ({ ...current, ...derived }));
  }

  async function submitPrediction(event: FormEvent) {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      setPrediction(await predict(runId, features));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Prediction failed.");
      setPrediction(null);
    } finally {
      setIsSubmitting(false);
    }
  }

  const activeFeatures = Object.keys(predictionDefaults).filter(
    (name) => !metadata?.feature_columns || metadata.feature_columns.includes(name),
  );

  return (
    <section className="predict-layout">
      <form className="panel predict-form" onSubmit={(event) => void submitPrediction(event)}>
        <PanelTitle icon={Brain} title="Prediction Inputs" />
        <label className="field-block full">
          <span>Sample Review Text</span>
          <textarea
            value={reviewText}
            onChange={(event) => deriveFromText(event.target.value)}
            rows={5}
            placeholder="Clear lectures, fair exams, and helpful feedback..."
          />
        </label>

        <div className="input-grid">
          {activeFeatures.map((name) => (
            <label key={name} className="field-block">
              <span>{humanize(name)}</span>
              <input
                type="number"
                step="0.01"
                value={features[name] ?? 0}
                onChange={(event) => updateFeature(name, Number(event.target.value))}
              />
            </label>
          ))}
        </div>

        <button className="primary-button" type="submit" disabled={isSubmitting || !runId}>
          <Brain size={18} />
          {isSubmitting ? "Predicting" : "Predict Rating"}
        </button>
      </form>

      <section className="panel prediction-result">
        <PanelTitle icon={Gauge} title="Prediction Result" />
        {prediction ? (
          <>
            <div className="rating-meter">
              <span>{prediction.prediction.toFixed(2)}</span>
              <div>
                <div style={{ width: `${clamp((prediction.prediction / 5) * 100, 0, 100)}%` }} />
              </div>
            </div>
            <div className="detail-list">
              <Detail label="Run" value={prediction.run_id} />
              <Detail label="Model" value={prediction.model_name} />
              <Detail label="Missing Filled" value={prediction.missing_features.length} />
              <Detail label="Extra Ignored" value={prediction.extra_features.length} />
            </div>
          </>
        ) : (
          <div className="empty-state compact">
            <Brain size={28} />
            <p>No prediction yet.</p>
          </div>
        )}
        {error && <p className="form-error">{error}</p>}
      </section>
    </section>
  );
}

function RunsView({
  runs,
  selectedRunId,
  models,
}: {
  runs: RunSummary[];
  selectedRunId: string;
  models: ModelInfo[];
}) {
  return (
    <section className="view-grid">
      <section className="panel">
        <PanelTitle icon={ListChecks} title="Saved Experiment Runs" />
        <div className="run-list">
          {runs.map((run) => (
            <div key={run.run_id} className={run.run_id === selectedRunId ? "run-row selected" : "run-row"}>
              <strong>{run.experiment_name || run.run_id}</strong>
              <span>{run.best_model_name ?? "No best model"}</span>
              <span>{formatDate(run.created_at)}</span>
              <span>{run.git_commit?.slice(0, 12) ?? "No commit"}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <PanelTitle icon={FlaskConical} title="Available Model Families" />
        <div className="chip-grid">
          {models.map((model) => (
            <span key={model.name}>{model.name}</span>
          ))}
        </div>
      </section>
    </section>
  );
}

function Kpi({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <section className="kpi">
      <Icon size={20} />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </section>
  );
}

function PanelTitle({ icon: Icon, title }: { icon: typeof Gauge; title: string }) {
  return (
    <div className="panel-title">
      <Icon size={18} />
      <h3>{title}</h3>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{formatLoose(value)}</strong>
    </div>
  );
}

function MetricTable({ rows, columns }: { rows: MetricRecord[]; columns: string[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.Model}>
              {columns.map((column) => (
                <td key={column}>{formatLoose(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricBars({ rows, metric, mode }: { rows: MetricRecord[]; metric: string; mode: "min" | "max" }) {
  const values = rows.map((row) => metricValue(row, metric));
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);

  return (
    <div className="metric-bars">
      {rows.slice(0, 8).map((row) => {
        const value = metricValue(row, metric);
        const percent = mode === "max" ? ((value - min) / (max - min || 1)) * 100 : ((max - value) / (max - min || 1)) * 100;
        return (
          <div key={row.Model} className="metric-bar">
            <span>{row.Model}</span>
            <div>
              <div style={{ width: `${clamp(percent, 4, 100)}%` }} />
            </div>
            <strong>{formatNumber(value, 3)}</strong>
          </div>
        );
      })}
    </div>
  );
}

function BarList({ rows, valueLabel }: { rows: Array<{ label: string; value: number }>; valueLabel: string }) {
  const max = Math.max(...rows.map((row) => row.value), 0.01);

  return (
    <div className="bar-list">
      {rows.map((row) => (
        <div key={row.label} className="bar-row">
          <div>
            <span>{row.label}</span>
            <strong>{formatNumber(row.value, 4)}</strong>
          </div>
          <div aria-label={`${row.label} ${valueLabel}`}>
            <div style={{ width: `${clamp((row.value / max) * 100, 2, 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function ErrorBanner({ error }: { error: string | null }) {
  return (
    <section className="error-banner">
      <Activity size={18} />
      <span>{error ?? "Unable to load dashboard data."}</span>
    </section>
  );
}

function LoadingBand() {
  return (
    <section className="loading-band">
      <RefreshCw size={18} />
      <span>Loading dashboard data</span>
    </section>
  );
}

function EmptyRuns({ apiBaseUrl }: { apiBaseUrl: string }) {
  return (
    <section className="empty-state">
      <Database size={36} />
      <h3>No experiment runs found</h3>
      <p>API base: {apiBaseUrl}</p>
    </section>
  );
}

function bestBy(rows: MetricRecord[], key: string, mode: "min" | "max") {
  if (rows.length === 0) {
    return null;
  }
  return rows.reduce((best, row) => {
    const rowValue = metricValue(row, key);
    const bestValue = metricValue(best, key);
    return mode === "max" ? (rowValue > bestValue ? row : best) : rowValue < bestValue ? row : best;
  }, rows[0]);
}

function metricValue(row: MetricRecord, key: string) {
  const value = row[key];
  return typeof value === "number" ? value : Number(value ?? 0);
}

function formatMetric(row: MetricRecord | null, key: string) {
  if (!row) {
    return "No metric";
  }
  return `${key} ${formatNumber(metricValue(row, key), 3)}`;
}

function formatNumber(value: unknown, digits = 0) {
  const numberValue = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numberValue)) {
    return "n/a";
  }
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits > 0 ? Math.min(digits, 2) : 0,
  }).format(numberValue);
}

function formatLoose(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return formatNumber(value, Math.abs(value) < 10 ? 3 : 1);
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
}

function formatDate(value: string | null | undefined) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function numberFrom(value: unknown) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function humanize(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

function deriveTextFeatures(text: string) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const sentences = text.split(/[.!?]+/).filter((sentence) => sentence.trim().length > 0);
  const keywordGroups: Record<string, string[]> = {
    difficulty_keyword_count: ["hard", "difficult", "tough", "easy", "challenging"],
    clarity_keyword_count: ["clear", "confusing", "organized", "explain", "lecture"],
    helpfulness_keyword_count: ["helpful", "office", "feedback", "responsive", "support"],
    workload_keyword_count: ["workload", "homework", "project", "busy", "assignment"],
    exams_keyword_count: ["exam", "test", "quiz", "midterm", "final"],
  };
  const lower = text.toLowerCase();
  const positive = ["great", "clear", "helpful", "fair", "excellent", "kind"].reduce(
    (count, word) => count + occurrences(lower, word),
    0,
  );
  const negative = ["bad", "confusing", "hard", "unfair", "poor", "difficult"].reduce(
    (count, word) => count + occurrences(lower, word),
    0,
  );
  const sentiment = words.length === 0 ? 0 : clamp((positive - negative) / Math.max(words.length / 8, 1), -1, 1);
  const keywordValues = Object.fromEntries(
    Object.entries(keywordGroups).map(([name, group]) => [
      name,
      group.reduce((count, word) => count + occurrences(lower, word), 0),
    ]),
  );

  return {
    avg_review_length: text.length,
    avg_review_word_count: words.length,
    avg_sentence_count: sentences.length,
    avg_words_per_sentence: sentences.length > 0 ? words.length / sentences.length : 0,
    question_mark_ratio: text.length > 0 ? occurrences(text, "?") / text.length : 0,
    exclamation_mark_ratio: text.length > 0 ? occurrences(text, "!") / text.length : 0,
    avg_sentiment_compound: sentiment,
    positive_review_ratio: positive > negative ? 1 : 0,
    negative_review_ratio: negative > positive ? 1 : 0,
    neutral_review_ratio: positive === negative ? 1 : 0,
    ...keywordValues,
  };
}

function occurrences(text: string, token: string) {
  return text.split(token).length - 1;
}

export default App;
