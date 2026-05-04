# Deployment Guide

Phase 10 adds a local Docker Compose deployment for demo and portfolio use. It is intentionally simple: one API container, one static frontend container, and host-mounted artifact directories.

## Architecture

```text
Docker Compose
|-- api
|   |-- FastAPI
|   |-- planetterp_predictor package
|   |-- mounted data/, experiments/, outputs/
|
`-- frontend
    |-- Vite production build
    `-- Nginx static server
```

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile.api` | Builds the Python/FastAPI image. |
| `app/Dockerfile` | Builds the React dashboard and serves it with Nginx. |
| `app/nginx.conf` | Static SPA fallback and asset caching. |
| `docker-compose.yml` | Runs API and frontend together. |
| `.dockerignore` | Keeps local caches and generated artifacts out of build context. |

## Run Locally

From the repository root:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:5173
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API docs are available at:

```text
http://127.0.0.1:8000/docs
```

Stop the stack:

```powershell
docker compose down
```

## Persistent Artifacts

The API service mounts these host folders into `/app`:

| Host path | Container path | Contents |
| --- | --- | --- |
| `./data` | `/app/data` | Raw snapshots and processed summaries/features. |
| `./experiments` | `/app/experiments` | Saved experiment runs and model bundles. |
| `./outputs` | `/app/outputs` | Generated plots. |

This means a run created locally is visible to the containerized API, and a run created in the container remains visible on the host.

## Frontend API URL

The frontend image is built with:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

That value is correct for local browser demos because the browser runs on the host and reaches the API through the published `8000` port. For a hosted deployment, rebuild the frontend with the public API URL:

```powershell
docker compose build --build-arg VITE_API_BASE_URL=https://your-api.example.com frontend
```

## Smoke Checks

After the stack is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:5173 -UseBasicParsing
```

The existing local browser smoke test can also target the Compose frontend:

```powershell
cd app
$env:NODE_PATH="C:\Users\aruba\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
$env:DASHBOARD_URL="http://127.0.0.1:5173"
npm run smoke
```

## Production Notes

This Compose setup is best for local demos. Before deploying publicly:

- Replace permissive API CORS settings with explicit frontend origins.
- Move synchronous training behind a background job queue.
- Add authentication if artifacts or predictions should not be public.
- Decide where model artifacts and snapshots live: mounted volume, object storage, or a registry.
- Add HTTPS termination through a reverse proxy or hosting platform.
