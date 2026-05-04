# Backend API

Phase 6 adds a FastAPI backend for reading experiment artifacts and serving predictions from saved model bundles.

## Start The API

```powershell
.\.venv\Scripts\python.exe -m planetterp_predictor serve-api --host 127.0.0.1 --port 8000
```

Interactive API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Start The Frontend Dashboard

```powershell
cd app
npm install
npm run dev
```

The dashboard is available at:

```text
http://127.0.0.1:5173
```

By default the frontend reads from `http://127.0.0.1:8000`. Override that with
`VITE_API_BASE_URL` when the backend is hosted elsewhere.

## Dashboard Smoke Test

With the API and dashboard running locally:

```powershell
cd app
$env:NODE_PATH="C:\Users\aruba\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
npm run smoke
```

The smoke test uses Playwright with the local Microsoft Edge executable. Override
`DASHBOARD_URL` or `EDGE_PATH` if either default is different on another machine.

## Endpoints

```text
GET  /health
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/metrics
GET  /api/runs/{run_id}/plots
GET  /api/models
POST /api/predict
POST /api/train
```

Plot artifacts are served from:

```text
/artifacts/runs/{run_id}/plots/{plot_name}
```

## Examples

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

List saved experiment runs:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/runs
```

List plots for a run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/runs/<run_id>/plots
```

Make a prediction from a saved run:

```powershell
$body = @{
  run_id = "<run_id>"
  fill_missing = $true
  features = @{
    num_reviews = 10
    avg_review_length = 400
    avg_sentiment_compound = 0.4
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://127.0.0.1:8000/api/predict -Method Post -Body $body -ContentType "application/json"
```

When `fill_missing` is `true`, omitted feature columns are filled with `0.0` and reported in the response. For production use, callers should send the full feature vector listed in the experiment run's `metadata.json`.

Trigger training from the latest snapshot:

```powershell
$body = @{
  snapshot = "latest"
  min_reviews = 1
  experiment_name = "api-smoke"
  save_experiment = $true
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/api/train -Method Post -Body $body -ContentType "application/json"
```

`POST /api/train` runs synchronously. For larger datasets, a later phase should move training into a background job queue.
