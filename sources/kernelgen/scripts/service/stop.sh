#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
repo_root=$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)
pid_file="$repo_root/.service.pid"
log_file="$repo_root/.service.log"
fleet_script="$repo_root/scripts/proxy/fleet.sh"

usage() {
  cat <<'EOF'
Usage:
  stop.sh [--force] [--with-fleet]

Stop the KernelGen HTTP service.

Options:
  --force       Use SIGKILL instead of SIGTERM
  --with-fleet  Also stop fleet discovery and managed SSH proxies
  --help        Show this message
EOF
}

signal="TERM"
with_fleet=false

while (($#)); do
  case "$1" in
    --force)
      signal="KILL"
      shift
      ;;
    --with-fleet)
      with_fleet=true
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

stop_fleet() {
  if $with_fleet; then
    "$fleet_script" stop
  fi
}

if [[ ! -f "$pid_file" ]]; then
  printf 'No PID file found. Service is not running.\n'
  stop_fleet
  exit 0
fi

pid=$(cat "$pid_file")

if ! kill -0 "$pid" 2>/dev/null; then
  printf 'Service (pid %s) is not running\n' "$pid"
  rm -f "$pid_file"
  stop_fleet
  exit 0
fi

printf 'Stopping service (pid %s) with SIG%s...\n' "$pid" "$signal"

kill -s "$signal" "$pid" 2>/dev/null || {
  printf 'Failed to signal process\n' >&2
  exit 1
}

# Wait for graceful shutdown
if [[ "$signal" == "TERM" ]]; then
  for i in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
fi

if kill -0 "$pid" 2>/dev/null; then
  printf 'Service still running (may need --force)\n'
  exit 1
fi

printf 'Service stopped\n'
rm -f "$pid_file"

# Show last few log lines
if [[ -f "$log_file" ]]; then
  printf '\nLast 10 log lines:\n'
  tail -10 "$log_file"
fi
stop_fleet
