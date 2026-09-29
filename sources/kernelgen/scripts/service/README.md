# KernelGen Service Management

Scripts for starting and stopping the KernelGen HTTP service with web UI.

## Quick Start

```bash
# Start service (localhost only, no auth)
./scripts/service/start.sh

# Open in browser
open http://localhost:8378/

# Stop service
./scripts/service/stop.sh
```

## The Service

KernelGen Service is a FastAPI application that exposes `kernelgen.cli.api` over HTTP:

- **HTTP API** at `/api/*` (submit, status, recent activity, cancel runs)
- **Web UI** at `/` (single-page console with polling refresh)
- **Authentication** via Bearer token or `X-KG-Token` header (optional)

Run Detail polls `/api/activity` alongside status and shows a bounded Recent
Activity feed from structured USER-visible run events. The endpoint returns only
an allowlisted event projection; it redacts URL and credential-shaped text and does
not expose raw runner/runtime logs, DEBUG or INTERNAL events, event data
payloads, prompts, tool payloads, or provider credentials.

## Usage

### Development Mode (Default)

```bash
./scripts/service/start.sh
```

- Binds to `127.0.0.1:8378` (localhost only)
- No authentication required
- Runs as background daemon
- Logs to `.service.log`

### Production Mode

```bash
./scripts/service/start.sh \
  --host 0.0.0.0 \
  --port 8378 \
  --token "your-secret-token"
```

- Accessible from other machines
- Requires token in requests:
  ```bash
  curl -H "X-KG-Token: your-secret-token" http://host:8378/api/runs
  ```
- UI prompts for token (stored in localStorage)

### Provider Environment

If an untracked `env.sh` exists at the repository root, `start.sh` sources it
before parsing command-line options. This lets a background service started from
a clean login shell inherit the same provider configuration as interactive runs:

```bash
# env.sh (do not commit credentials)
export ANTHROPIC_BASE_URL="https://provider.example"
export ANTHROPIC_AUTH_TOKEN="..."
export MODEL="provider-model-name"
```

Command-line `--host`, `--port`, and `--token` values still take precedence.
Keep `env.sh` untracked and readable only by the service account. The runtime
continues to pass the selected model through its isolated run configuration;
provider credentials are inherited through the process environment and are not
copied into run workspaces.

### Multi-Device Fleet Console

```bash
./scripts/service/start.sh \
  --host 0.0.0.0 \
  --port 8378 \
  --token "your-secret-token" \
  --with-fleet \
  --jump-user USER
```

This starts both the FastAPI/Web UI process and the fleet daemon. The daemon
scans actual KGS process ports, maintains one SSH proxy per endpoint, and writes
the filesystem registry consumed by the instance selector. Use
`./scripts/service/stop.sh --with-fleet` to stop both.

Plain `start.sh` intentionally does not start fleet discovery, so local service
development never initiates remote SSH unexpectedly.

### Foreground Mode (Debugging)

```bash
./scripts/service/start.sh --foreground
```

- Logs to stdout
- Ctrl+C to stop
- Use for debugging or when running under a supervisor

## Architecture

```
┌─────────────┐
│  Browser    │  http://localhost:8378/
│   (UI)      │  ← polls /api/runs, /api/status
└─────┬───────┘
      │
┌─────▼──────────────────────────────┐
│  KernelGen Service (FastAPI)       │
│  - GET  /api/runs                  │
│  - POST /api/runs                  │
│  - GET  /api/status?workspace=...  │
│  - GET  /api/activity?workspace=...│
│  - POST /api/cancel                │
│  - GET  /api/config                │
│  - GET  /         (UI)             │
└─────┬──────────────────────────────┘
      │
┌─────▼──────────────────────────────┐
│  kernelgen.cli.api                 │
│  - submit_run()                    │
│  - list_run_statuses()             │
│  - status()                        │
│  - run_activity()                  │
│  - cancel_run()                    │
└─────┬──────────────────────────────┘
      │
┌─────▼──────────────────────────────┐
│  kernelgen.framework.coordinator   │
│  (on-disk state, process mgmt)     │
└────────────────────────────────────┘
```

## Common Workflows

### Local Development

```bash
# Terminal 1: Start service
./scripts/service/start.sh

# Terminal 2: Submit a run
curl -X POST http://localhost:8378/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "definition": "gcd",
    "options": {
      "mode": "simple_opt",
      "target_hardware": "Ascend910B",
      "eval_server": "http://localhost:8000"
    }
  }'

# Or use the UI
open http://localhost:8378/
```

### Multi-Device Testing

```bash
# Start the Web/API service together with fleet discovery and managed proxies.
./scripts/service/start.sh --with-fleet --jump-user USER

# Stop both when finished.
./scripts/service/stop.sh --with-fleet
```

The UI discovers actual KGS instances from the fleet registry instead of relying
on the legacy fixed `180xx` mapping. Each selected instance keeps its own
`eval_server`. KernelGen worker-pool data remains available from the HTTP API,
but is not displayed in the current UI because it is not a KGS-internal queue.

### Remote Access

```bash
# On GPU machine (inside container)
./scripts/service/start.sh \
  --host 0.0.0.0 \
  --port 18308 \
  --token "team-shared-token"

# On dev machine: set up proxy
./scripts/proxy/start_all.sh --device nvidia

# Access from dev machine
curl -H "X-KG-Token: team-shared-token" \
  http://localhost:18008/api/runs
```

## Logs and PIDs

- **PID file**: `.service.pid`
- **Log file**: `.service.log`

```bash
# View live logs
tail -f .service.log

# Check if running
cat .service.pid && kill -0 $(cat .service.pid) && echo "running"
```

## Troubleshooting

### Port already in use

```bash
# Find what's using the port
lsof -i :8378

# Stop the service
./scripts/service/stop.sh

# Or use a different port
./scripts/service/start.sh --port 8379
```

### Service won't start

```bash
# Check logs
tail -50 .service.log

# Verify dependencies
pip list | grep -E "fastapi|uvicorn|httpx"

# Try foreground mode to see errors
./scripts/service/start.sh --foreground
```

### UI can't reach eval server

The service runs `kernelgen.cli.api` which expects:
- An eval server at the URL specified in submit requests
- For local testing: start `kg eval-server` in another terminal
- For remote testing: use proxy scripts to expose remote eval servers

## Related

- Service implementation: `service/` (FastAPI app + UI)
- Underlying API: `kernelgen/cli/api.py`
- Multi-device proxies: `scripts/proxy/`
- Tests: `tests/test_service.py`
