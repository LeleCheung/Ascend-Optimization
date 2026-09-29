#!/usr/bin/env bash
set -euo pipefail

script_dir=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)
# shellcheck source=tests/multi_device_batch_lib.sh
source "$script_dir/multi_device_batch_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  check_multi_device_batch_results.sh \
    [--run-name NAME] \
    [--inventory FILE] \
    [--identity FILE] \
    [--device NAME]...

Read each remote BatchSimpleOpt workspace without changing it. The default run
name is batch_simple_opt_v5_overlap15_20260730.
EOF
}

run_name="batch_simple_opt_v5_overlap15_20260730"
devices=()

while (($#)); do
  case "$1" in
    --run-name)
      run_name=$2
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
    --device)
      devices+=("$2")
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

[[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'invalid run name: %s\n' "$run_name" >&2
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

records=$(multi_device_inventory_records) || {
  printf 'invalid inventory section: %s\n' "$MULTI_DEVICE_INVENTORY" >&2
  exit 2
}
overall_exit=0
while IFS='|' read -r \
  name mode address ssh_port container deploy_base server_port cards \
  jump_login; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  multi_device_validate_record \
    "$name" "$mode" "$address" "$ssh_port" "$container" \
    "$deploy_base" "$server_port" "$jump_login" || {
      printf 'invalid inventory record for %s\n' "$name" >&2
      exit 2
    }
  multi_device_selected "$name" "${devices[@]}" || continue

  printf '\n[%s] address=%s container=%s cards=%s\n' \
    "$name" "$address" "$container" "$cards"
  if ! {
    printf 'name=%q\n' "$name"
    printf 'deploy_base=%q\n' "$deploy_base"
    printf 'run_name=%q\n' "$run_name"
    printf 'server_port=%q\n' "$server_port"
    cat <<'REMOTE'
set -u
run_root="$deploy_base/runs/${run_name}_${name}"
if [[ ! -d "$run_root" ]]; then
  run_root="$deploy_base/runs/$run_name"
fi
workspace="$run_root/workspace"
python_bin="$deploy_base/venv/bin/python"

printf 'checked_at=%s\n' "$(date -Iseconds)"
printf 'run_root=%s\n' "$run_root"
if [[ ! -d "$workspace" ]]; then
  printf 'workspace=missing\n'
  exit 3
fi

unset PYTHONPATH
"$python_bin" -m kernelgen.tools.monitor_batch_simple_opt \
  --workspace "$workspace" --once
monitor_exit=$?
printf 'monitor_exit=%s\n' "$monitor_exit"

runner_count=$(
  pgrep -af 'examples/batch_simple_opt_definition/run_example.py|kernelgen.cli.legacy_batch_simple_opt' |
    grep -F -- "$workspace" |
    grep -vc 'grep -F' || true
)
printf 'runner_count=%s\n' "$runner_count"
printf 'summary_file='
[[ -f "$workspace/batch_simple_opt_definition_output.json" ]] &&
  echo present || echo missing
printf 'server_status='
curl -fsS --max-time 10 "http://127.0.0.1:$server_port/status" 2>/dev/null ||
  printf 'unreachable'
printf '\n'
REMOTE
  } | multi_device_remote_bash \
    "$mode" "$address" "$ssh_port" "$container" "$jump_login"; then
    printf '[%s] remote check failed\n' "$name" >&2
    overall_exit=1
  fi
done <<< "$records"

exit "$overall_exit"
