# MarinFold CoreWeave inference server

Minimal control-plane API for running MarinFold inference on CoreWeave. The
notebook/client remains the control plane: it writes run configs and targets to
GCS, calls this service to start/check runs, and reads chunked outputs from GCS.
The service owns the data-plane work inside the deployed pod.

This first increment only provides health/readiness endpoints plus bearer-token
auth plumbing for future run endpoints.

## Local development

```bash
cd experiments/server
uv sync --extra test
MARINFOLD_SERVER_TOKEN=test-token uv run uvicorn marinfold_server.app:app \
  --host 127.0.0.1 --port 8080
```

Smoke test:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/readyz
curl -H 'Authorization: Bearer test-token' http://127.0.0.1:8080/v1/auth-check
```

Run tests:

```bash
uv run pytest
```

## Planned API

```text
POST /v1/runs
GET  /v1/runs/{run_id}
POST /v1/runs/{run_id}/cancel
```

Run payloads should carry GCS paths, not protein data:

```json
{
  "run_id": "...",
  "config_gcs": "gs://.../config.json",
  "targets_gcs": "gs://.../targets.parquet",
  "output_gcs": "gs://.../runs/<run_id>/"
}
```
