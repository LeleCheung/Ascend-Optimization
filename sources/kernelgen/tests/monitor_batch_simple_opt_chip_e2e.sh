#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  monitor_batch_simple_opt_chip_e2e.sh \
    --server URL \
    --run-root DIR

Print the eval-server slot state and the current BatchSimpleOpt phase/results.
EOF
}

server=""
run_root=""

while (($#)); do
  case "$1" in
    --server)
      server=$2
      shift 2
      ;;
    --run-root)
      run_root=$2
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

if [[ -z "$server" || -z "$run_root" ]]; then
  usage >&2
  exit 2
fi

printf 'checked_at=%s\n' "$(date -Iseconds)"
printf 'server_status='
curl -sS --max-time 10 "$server/status"
printf '\n'

runner_count=$(
  pgrep -af 'batch_simple_opt_definition/run_example.py' |
    grep -F -- "$run_root/" |
    grep -vc 'pgrep -af' || true
)
agent_count=$(
  pgrep -af 'claude -p' |
    grep -F -- "$run_root/" |
    grep -vc 'pgrep -af' || true
)
printf 'runner_count=%s agent_count=%s\n' "$runner_count" "$agent_count"

for phase in legacy v4; do
  marker="$run_root/$phase.exit"
  output="$run_root/$phase/batch_simple_opt_definition_output.json"
  if [[ -f "$marker" ]]; then
    printf '%s_exit=%s\n' "$phase" "$(<"$marker")"
  else
    printf '%s_exit=running_or_pending\n' "$phase"
  fi
  if [[ -f "$output" ]]; then
    summary=$(
      sed -n \
        's/^[[:space:]]*"summary":[[:space:]]*"\(.*\)",[[:space:]]*$/\1/p' \
        "$output" |
        head -n 1
    )
    printf '%s_summary=%s\n' "$phase" "$summary"
  fi
done

if [[ -f "$run_root/exit" ]]; then
  printf 'final_exit=%s\n' "$(<"$run_root/exit")"
else
  printf 'final_exit=running\n'
fi
