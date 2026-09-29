#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
repo_root=$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)
pid_file="$repo_root/.service.pid"
log_file="$repo_root/.service.log"
fleet_script="$repo_root/scripts/proxy/fleet.sh"
env_file="$repo_root/env.sh"

# Keep provider credentials and model selection outside tracked files while
# making daemon starts reproducible from a clean login shell.
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  start.sh [OPTIONS]

Start the KernelGen HTTP service with web UI.

Options:
  --host HOST       Bind address (default: 127.0.0.1)
  --port PORT       Listen port (default: 8378)
  --token TOKEN     Require authentication with this token
  --with-fleet      Also start KGS discovery and managed SSH proxies
  --jump-user USER  JumpServer user for --with-fleet (default: KG_JUMP_USER)
  --foreground      Run the HTTP service in foreground
  --help            Show this message

Examples:
  # Local development (no auth, localhost only)
  start.sh

  # Exposed service plus fleet discovery/proxies
  start.sh --host 0.0.0.0 --port 8378 --token "secret-token" \
    --with-fleet --jump-user USER

  # Foreground HTTP service (for debugging)
  start.sh --foreground

The service provides:
  - HTTP API at /api/* (submit/status/cancel runs)
  - Web UI at / (single-page console)

If <repo>/env.sh exists, it is sourced before options are parsed. This keeps
provider credentials and MODEL outside tracked files while making daemon starts
reproducible from a clean login shell. Explicit command-line options win.

Access the UI in your browser at http://HOST:PORT/
EOF
}

host=${KG_SERVICE_HOST:-"127.0.0.1"}
port=${KG_SERVICE_PORT:-"8378"}
token=${KG_SERVICE_TOKEN:-}
foreground=false
with_fleet=false
fleet_jump_user=${KG_JUMP_USER:-}
fleet_started_here=false

rollback_fleet() {
  if $fleet_started_here; then
    "$fleet_script" stop >/dev/null 2>&1 || true
  fi
}

while (($#)); do
  case "$1" in
    --host)
      host=$2
      shift 2
      ;;
    --port)
      port=$2
      shift 2
      ;;
    --token)
      token=$2
      shift 2
      ;;
    --with-fleet)
      with_fleet=true
      shift
      ;;
    --jump-user)
      fleet_jump_user=$2
      shift 2
      ;;
    --foreground)
      foreground=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$port" =~ ^[0-9]+$ ]] || {
  printf 'port must be a number: %s\n' "$port" >&2
  exit 2
}
if [[ -n "$fleet_jump_user" ]] && [[ ! "$fleet_jump_user" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'invalid jump user: %s\n' "$fleet_jump_user" >&2
  exit 2
fi

# Check if already running
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  printf 'Service already running (pid %s)\n' "$(cat "$pid_file")"
  printf 'Stop it first with: %s/stop.sh\n' "$script_dir"
  exit 1
fi

# Set environment variables for kg-service
export KG_SERVICE_HOST="$host"
export KG_SERVICE_PORT="$port"
if [[ -n "$token" ]]; then
  export KG_SERVICE_TOKEN="$token"
fi

# Security warning
if [[ "$host" != "127.0.0.1" && "$host" != "localhost" && -z "$token" ]]; then
  cat >&2 <<'EOF'
WARNING: Binding to non-loopback address without authentication.
The service will be accessible from other machines WITHOUT a token.
Press Ctrl+C to abort, or wait 5 seconds to continue...
EOF
  sleep 5
fi

if $with_fleet; then
  fleet_was_running=false
  if "$fleet_script" status >/dev/null 2>&1; then
    fleet_was_running=true
  fi
  printf 'Starting KGS fleet discovery and proxies...\n'
  if [[ -n "$fleet_jump_user" ]]; then
    KG_JUMP_USER="$fleet_jump_user" "$fleet_script" start
  else
    "$fleet_script" start
  fi
  if ! $fleet_was_running; then
    fleet_started_here=true
  fi
fi

printf 'Starting KernelGen service...\n'
printf '  Host: %s\n' "$host"
printf '  Port: %s\n' "$port"
printf '  Auth: %s\n' "${token:+enabled}"

if $foreground; then
  # A composed foreground service owns a newly started fleet for its lifetime.
  if $fleet_started_here; then
    trap rollback_fleet EXIT
    python3 -m kernelgen.service
  else
    exec python3 -m kernelgen.service
  fi
else
  # Run as background daemon
  python3 -m kernelgen.service > "$log_file" 2>&1 &
  service_pid=$!
  echo "$service_pid" > "$pid_file"

  # Wait a moment to check if it started successfully
  sleep 1
  if ! kill -0 "$service_pid" 2>/dev/null; then
    printf 'Service failed to start. Check logs:\n'
    printf '  tail -50 %s\n' "$log_file"
    rm -f "$pid_file"
    rollback_fleet
    exit 1
  fi

  printf '\nService started (pid %s)\n' "$service_pid"
  printf 'Logs: %s\n' "$log_file"
  printf '\nOpen in browser: http://%s:%s/\n' "$host" "$port"
  printf 'Stop with: %s/stop.sh\n' "$script_dir"
fi
