#!/usr/bin/env bash
set -euo pipefail

operator=${1:?usage: run-kg-operator-910b.sh OPERATOR no-profile|profile WORKSPACE_SUFFIX [MAX_ROUND] [SEED_CODE_PATH]}
arm=${2:?usage: run-kg-operator-910b.sh OPERATOR no-profile|profile WORKSPACE_SUFFIX [MAX_ROUND] [SEED_CODE_PATH]}
suffix=${3:?usage: run-kg-operator-910b.sh OPERATOR no-profile|profile WORKSPACE_SUFFIX [MAX_ROUND] [SEED_CODE_PATH]}
max_round=${4:-2}
seed=${5:-}
catalog_path=${KG_CATALOG_PATH:-}

base=${KG_CONTROLLER_ROOT:-/data/hanle/ascend-optimization/runtime/kg-controller}
run_root=${KG_RUN_ROOT:-$base/runs}
case "$arm" in
  no-profile) profile_flag=--no-profile; port=${KG_WALLTIME_PORT:-19652} ;;
  profile) profile_flag=--profile; port=${KG_PROFILE_PORT:-19652} ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

workspace="$run_root/${operator}-${suffix}"
if test -e "$workspace"; then
  echo "workspace already exists: $workspace" >&2
  exit 2
fi

curl --noproxy '*' -fsS "http://127.0.0.1:$port/status" >/dev/null
source /data/yy/kgen/kernelgen/env.sh
export PATH="$base/bin:$base/venv/bin:/usr/local/python3.11.15/bin:$PATH"
# 每次实验显式选择源码，避免 editable install 指向旧仓库。
if test -n "${KG_SOURCE_ROOT:-}"; then
  test -f "$KG_SOURCE_ROOT/kernelgen/__init__.py"
  export PYTHONPATH="$KG_SOURCE_ROOT:$KG_SOURCE_ROOT/kernelgen_server/client:$KG_SOURCE_ROOT/kernelgen_server:${PYTHONPATH:-}"
fi
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

args=(
  run --mode simple_opt
  --definition "$operator"
  --eval-server "http://127.0.0.1:$port"
  --runtime claude --model "${KG_MODEL:-deepseek-v4-flash[1m]}"
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
# KG 6.7.0 的 foreground 路径会在提交锁内再次取得同一把锁。
# 使用原生后台 worker，由当前脚本等待其结果；tmux 仍能托管整个任务。
"$base/venv/bin/kg" "${args[@]}"

process_file="$workspace/.kernelgen/run-process.json"
log_file="$workspace/.kernelgen/runner.log"
test -f "$log_file" || { echo "worker log missing: $log_file" >&2; exit 1; }
"$base/venv/bin/python3" "$(dirname "$0")/wait-kg-worker.py" "$workspace"
