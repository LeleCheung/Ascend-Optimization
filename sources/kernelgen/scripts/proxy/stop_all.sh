#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
repo_root=$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)
pid_dir="$repo_root/.proxy_pids"

usage() {
  cat <<'EOF'
Usage:
  stop_all.sh [--device DEVICE1,DEVICE2,...] [--force]

Stop HTTP proxies started by start_all.sh.

Options:
  --device DEVICES  Comma-separated device names (default: all)
  --force           Use SIGKILL instead of SIGTERM
  --help            Show this message

Examples:
  stop_all.sh                      # Stop all proxies gracefully
  stop_all.sh --device nvidia      # Stop only nvidia proxy
  stop_all.sh --force              # Force kill all proxies
EOF
}

selected_devices=""
signal="TERM"

while (($#)); do
  case "$1" in
    --device)
      selected_devices=$2
      shift 2
      ;;
    --force)
      signal="KILL"
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

[[ -d "$pid_dir" ]] || {
  printf 'No PID directory found: %s\n' "$pid_dir" >&2
  printf 'No proxies appear to be running.\n' >&2
  exit 0
}

# Build device filter
if [[ -n "$selected_devices" ]]; then
  IFS=',' read -ra device_list <<< "$selected_devices"
  device_filter=$(printf "|%s" "${device_list[@]}")
  device_filter="^(${device_filter:1})\\.pid$"
else
  device_filter=".*\\.pid$"
fi

printf 'Stopping proxies (signal: SIG%s)...\n' "$signal"

stopped=0
for pid_file in "$pid_dir"/*.pid; do
  [[ -f "$pid_file" ]] || continue

  basename_pid=$(basename "$pid_file")
  device_name="${basename_pid%.pid}"

  # Apply device filter
  [[ "$basename_pid" =~ $device_filter ]] || continue

  pid=$(cat "$pid_file")
  printf '  %s (pid %s) ... ' "$device_name" "$pid"

  if ! kill -0 "$pid" 2>/dev/null; then
    printf 'not running\n'
    rm -f "$pid_file"
    continue
  fi

  kill -s "$signal" "$pid" 2>/dev/null || {
    printf 'failed to signal\n'
    continue
  }

  # Wait up to 3 seconds for graceful shutdown
  if [[ "$signal" == "TERM" ]]; then
    for i in {1..30}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  fi

  if kill -0 "$pid" 2>/dev/null; then
    printf 'still running (may need --force)\n'
  else
    printf 'stopped\n'
    rm -f "$pid_file"
    ((++stopped))
  fi
done

# Clean up log files for stopped processes
for log_file in "$pid_dir"/*.log; do
  [[ -f "$log_file" ]] || continue
  basename_log=$(basename "$log_file" .log)
  [[ ! -f "$pid_dir/$basename_log.pid" ]] && rm -f "$log_file"
done

if ((stopped == 0)); then
  printf 'No proxies stopped (not running or no matching devices)\n'
else
  printf '\nStopped %d proxy(ies)\n' "$stopped"
fi

# Clean up empty pid_dir
if [[ -d "$pid_dir" ]] && [[ -z "$(ls -A "$pid_dir")" ]]; then
  rmdir "$pid_dir"
  printf 'Cleaned up PID directory\n'
fi
