#!/usr/bin/env bash
set -euo pipefail

script_dir=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)
repo_root=$(
  cd -- "$script_dir/../.." >/dev/null 2>&1
  pwd
)
# shellcheck source=tests/multi_device_batch_lib.sh
source "$repo_root/tests/multi_device_batch_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  run_remote_http_proxy.sh \
    --device NAME \
    --listen-port PORT \
    [--remote-port PORT] \
    [--max-ssh-sessions N] \
    [--gateway-retry-attempts N] \
    [--inventory FILE] \
    [--identity FILE]

Expose one target's loopback KernelGen Server on local loopback. Connection
details are read from tests/hosts.md by default.
EOF
}

device=""
listen_port=""
remote_port=""
max_ssh_sessions=6
gateway_retry_attempts=6

while (($#)); do
  case "$1" in
    --device)
      device=$2
      shift 2
      ;;
    --listen-port)
      listen_port=$2
      shift 2
      ;;
    --remote-port)
      remote_port=$2
      shift 2
      ;;
    --max-ssh-sessions)
      max_ssh_sessions=$2
      shift 2
      ;;
    --gateway-retry-attempts)
      gateway_retry_attempts=$2
      shift 2
      ;;
    --inventory)
      MULTI_DEVICE_INVENTORY=$2
      shift 2
      ;;
    --identity)
      MULTI_DEVICE_IDENTITY=$2
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

[[ "$device" =~ ^[a-z0-9_-]+$ ]] || {
  printf 'invalid or missing device: %s\n' "$device" >&2
  exit 2
}
[[ "$listen_port" =~ ^[0-9]+$ ]] || {
  printf 'invalid or missing listen port: %s\n' "$listen_port" >&2
  exit 2
}
[[ "$max_ssh_sessions" =~ ^[1-9][0-9]*$ ]] || {
  printf 'max-ssh-sessions must be a positive integer\n' >&2
  exit 2
}
[[ "$gateway_retry_attempts" =~ ^[1-9][0-9]*$ ]] || {
  printf 'gateway-retry-attempts must be a positive integer\n' >&2
  exit 2
}
[[ -r "$MULTI_DEVICE_INVENTORY" ]] || {
  printf 'inventory is not readable: %s\n' "$MULTI_DEVICE_INVENTORY" >&2
  exit 2
}
[[ -r "$MULTI_DEVICE_IDENTITY" ]] || {
  printf 'SSH identity is not readable: %s\n' "$MULTI_DEVICE_IDENTITY" >&2
  exit 2
}

record=$(
  multi_device_inventory_records | awk -F'|' -v selected="$device" \
    '$1 == selected { record = $0; found++ } END { if (found == 1) print record; else exit 1 }'
) || {
  printf 'device is not present in inventory: %s\n' "$device" >&2
  exit 2
}

IFS='|' read -r -a record_fields <<<"$record"
case "${#record_fields[@]}" in
  8)
    name=${record_fields[0]}
    mode=${record_fields[1]}
    address=${record_fields[2]}
    ssh_port=${record_fields[3]}
    container=${record_fields[4]}
    deploy_base=${record_fields[5]}
    configured_port=${record_fields[6]}
    jump_login=${record_fields[7]}
    ;;
  9)
    name=${record_fields[0]}
    mode=${record_fields[1]}
    address=${record_fields[2]}
    ssh_port=${record_fields[3]}
    container=${record_fields[4]}
    deploy_base=${record_fields[5]}
    configured_port=${record_fields[6]}
    jump_login=${record_fields[8]}
    ;;
  *)
    printf 'invalid inventory field count for %s: expected 8 or 9\n' "$device" >&2
    exit 2
    ;;
esac
if [[ "$mode" == jump && "$jump_login" =~ ^ssh[[:space:]]+([^[:space:]]+)[[:space:]]+-p[[:space:]]+([0-9]+)$ ]]; then
  jump_destination=${BASH_REMATCH[1]}
  jump_command_port=${BASH_REMATCH[2]}
  jump_bastion_suffix="@$MULTI_DEVICE_BASTION"
  if [[ "$jump_command_port" != "$ssh_port" || "$jump_destination" != *"$jump_bastion_suffix" ]]; then
    printf 'inventory jump command does not match ssh_port/bastion for %s\n' "$device" >&2
    exit 2
  fi
  jump_login=${jump_destination%"$jump_bastion_suffix"}
fi
multi_device_validate_record \
  "$name" "$mode" "$address" "$ssh_port" "$container" \
  "$deploy_base" "$configured_port" "$jump_login" || {
    printf 'invalid inventory record for %s\n' "$device" >&2
    exit 2
  }

if [[ -z "$remote_port" ]]; then
  remote_port=$configured_port
fi
[[ "$remote_port" =~ ^[0-9]+$ ]] || {
  printf 'invalid remote port: %s\n' "$remote_port" >&2
  exit 2
}

args=(
  "$script_dir/ssh_stdio_http_proxy.py"
  --listen-port "$listen_port"
  --mode "$mode"
  --address "$address"
  --ssh-port "$ssh_port"
  --identity "$MULTI_DEVICE_IDENTITY"
  --remote-port "$remote_port"
  --container "$container"
  --bastion "$MULTI_DEVICE_BASTION"
  --max-ssh-sessions "$max_ssh_sessions"
  --gateway-retry-attempts "$gateway_retry_attempts"
)
if [[ "$mode" == jump ]]; then
  args+=(--jump-login "$jump_login")
fi

exec python3 "${args[@]}"
