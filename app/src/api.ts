import type {
  HealthResponse,
  MetricsResponse,
  ModelInfo,
  PlotInfo,
  PredictionResponse,
  RunMetadata,
  RunSummary,
} from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function checkHealth() {
  return requestJson<HealthResponse>("/health");
}

export function listRuns() {
  return requestJson<RunSummary[]>("/api/runs");
}

export function getRun(runId: string) {
  return requestJson<RunMetadata>(`/api/runs/${encodeURIComponent(runId)}`);
}

export function getRunMetrics(runId: string) {
  return requestJson<MetricsResponse>(`/api/runs/${encodeURIComponent(runId)}/metrics`);
}

export function getRunPlots(runId: string) {
  return requestJson<PlotInfo[]>(`/api/runs/${encodeURIComponent(runId)}/plots`);
}

export function listModels() {
  return requestJson<ModelInfo[]>("/api/models");
}

export function predict(runId: string, features: Record<string, number>) {
  return requestJson<PredictionResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify({
      run_id: runId,
      features,
      fill_missing: true,
    }),
  });
}

export function artifactUrl(url: string) {
  if (url.startsWith("http")) {
    return url;
  }
  return `${API_BASE_URL}${url}`;
}
