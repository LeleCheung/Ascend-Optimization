#!/usr/bin/env bash
set -euo pipefail

operator=${1:?usage: run-kg-operator-910b.sh OPERATOR no-profile|profile WORKSPACE_SUFFIX [MAX_ROUND] [SEED_CODE_PATH]}
arm=${2:?usage: run-kg-operator-910b.sh OPERATOR no-profile|profile WORKSPACE_SUFFIX [MAX_ROUND] [SEED_CODE_PATH]}
suffix=${3:?usage: run-kg-operator-910b.sh OPERATOR no-profile|profile WORKSPACE_SUFFIX [MAX_ROUND] [SEED_CODE_PATH]}
max_round=${4:-2}
seed=${5:-}
catalog_path=${KG_CATALOG_PATH:-}

base=/data/hanle/ascend-optimization/runtime/kg-controller
case "$arm" in
  no-profile) profile_flag=--no-profile; port=${KG_WALLTIME_PORT:-19652} ;;
  profile) profile_flag=--profile; port=${KG_PROFILE_PORT:-19652} ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

workspace="$base/runs/${operator}-${suffix}"
if test -e "$workspace"; then
  echo "workspace already exists: $workspace" >&2
  exit 2
fi

curl -fsS "http://127.0.0.1:$port/status" >/dev/null
source /data/yy/kgen/kernelgen/env.sh
export PATH="$base/venv/bin:/usr/local/python3.11.15/bin:$PATH"

args=(
  run --mode simple_opt
  --definition "$operator"
  --eval-server "http://127.0.0.1:$port"
  --runtime claude --model 'deepseek-v4-flash[1m]'
  --skip-review "$profile_flag"
  --min-rounds 2 --max-round "$max_round" --early-stop-rounds 0
  --max-coder-sessions 1 --eval-timeout-seconds 1500
  --warmup-ms 1000 --benchmark-ms 100 --num-trials 3
  --workspace "$workspace"
)
if test -n "$catalog_path"; then
  test -f "$catalog_path/manifest.json" || { echo "catalog manifest not found: $catalog_path/manifest.json" >&2; exit 2; }
  args+=(--catalog-path "$catalog_path")
else
  args+=(--catalog-name "${KG_CATALOG_NAME:-flaggems-adapter-definitions}")
fi
if test -n "$seed"; then
  test -f "$seed" || { echo "seed file not found: $seed" >&2; exit 2; }
  args+=(--seed-code-path "$seed")
fi
"$base/venv/bin/kg" "${args[@]}"

process_file="$workspace/.kernelgen/run-process.json"
log_file="$workspace/.kernelgen/runner.log"
worker_pid=$(sed -n 's/^[[:space:]]*"pid":[[:space:]]*\([0-9][0-9]*\),/\1/p' "$process_file")
test -n "$worker_pid" || { echo "worker process metadata missing: $process_file" >&2; exit 1; }
test -f "$log_file" || { echo "worker log missing: $log_file" >&2; exit 1; }
exec tail --pid="$worker_pid" -F "$log_file"
