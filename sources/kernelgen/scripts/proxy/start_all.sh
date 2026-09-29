#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
repo_root=$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)
source "$repo_root/tests/multi_device_batch_lib.sh"
inventory="$MULTI_DEVICE_INVENTORY"
proxy_script="$repo_root/scripts/remote_server/run_persistent_remote_http_proxy.sh"
pid_dir="$repo_root/.proxy_pids"

usage() {
  cat <<'EOF'
Usage:
  start_all.sh [--local-port-base BASE] [--device DEVICE1,DEVICE2,...] [--jump-user USER]

Start HTTP proxies for all devices in the inventory. Each device's remote
KernelGen Server becomes accessible on localhost at a predictable port.

Options:
  --local-port-base BASE  Local port base (default: 18000)
                          Device port 18308 → local 18008 when base=18000
  --device DEVICES        Comma-separated device names (default: all)
  --jump-user USER        Override legacy inventory login (default: KG_JUMP_USER)
  --help                  Show this message

Examples:
  start_all.sh                           # Start all devices
  start_all.sh --device nvidia,ascend    # Start only nvidia and ascend
  start_all.sh --jump-user gupan         # Use gupan@secure@TARGET for JumpServer
  start_all.sh --local-port-base 19000   # Map to 190xx ports

The script writes PIDs to .proxy_pids/ for stop_all.sh to use.
EOF
}

local_port_base=18000
selected_devices=""
jump_user=${KG_JUMP_USER:-}

while (($#)); do
  case "$1" in
    --local-port-base)
      local_port_base=$2
      shift 2
      ;;
    --device)
      selected_devices=$2
      shift 2
      ;;
    --jump-user)
      jump_user=$2
      shift 2
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

[[ -r "$inventory" ]] || {
  printf 'inventory not readable: %s\n' "$inventory" >&2
  exit 2
}
[[ -x "$proxy_script" ]] || {
  printf 'proxy script not executable: %s\n' "$proxy_script" >&2
  exit 2
}
if [[ -n "$jump_user" ]] && [[ ! "$jump_user" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'invalid jump user: %s\n' "$jump_user" >&2
  exit 2
fi

mkdir -p "$pid_dir"

# Build device filter
if [[ -n "$selected_devices" ]]; then
  IFS=',' read -ra device_list <<< "$selected_devices"
  device_filter=$(printf "|%s" "${device_list[@]}")
  device_filter="^(${device_filter:1})$"
else
  device_filter=".*"
fi

printf 'Starting proxies (local port base: %d)...\n' "$local_port_base"

started=0
records=$(multi_device_inventory_records) || {
  printf 'invalid inventory section: %s\n' "$inventory" >&2
  exit 2
}
while IFS='|' read -r name mode addr ssh_port container deploy port devices jump; do
  # Skip comments and empty lines
  [[ "$name" =~ ^[a-z] ]] || continue
  # Apply device filter
  [[ "$name" =~ $device_filter ]] || continue

  # Map remote port to local: 18308 → 18008 (last 2 digits + base)
  # 10#$port_suffix forces base-10 so "08"/"09" don't trigger bash's octal parsing.
  port_suffix=${port: -2}
  local_port=$((local_port_base + 10#$port_suffix))

  printf '  %s: remote :%s → local :%s ... ' "$name" "$port" "$local_port"

  # Check if already running
  pid_file="$pid_dir/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf 'already running (pid %s)\n' "$(cat "$pid_file")"
    continue
  fi

  # Start proxy in background
  proxy_args=(
    --device "$name"
    --listen-port "$local_port"
    --remote-port "$port"
  )
  if [[ -n "$jump_user" && "$mode" == jump ]]; then
    proxy_args+=(--jump-user "$jump_user")
  fi
  "$proxy_script" "${proxy_args[@]}" \
    > "$pid_dir/$name.log" 2>&1 &

  proxy_pid=$!
  echo "$proxy_pid" > "$pid_file"
  printf 'started (pid %s)\n' "$proxy_pid"
  ((++started))

  # Stagger launches to avoid thundering herd
  sleep 0.3
done <<< "$records"

if ((started == 0)); then
  printf 'No proxies started (already running or no matching devices)\n'
else
  printf '\nStarted %d proxy(ies). Logs in: %s/\n' "$started" "$pid_dir"
  printf 'Stop with: %s/stop_all.sh\n' "$script_dir"
fi
