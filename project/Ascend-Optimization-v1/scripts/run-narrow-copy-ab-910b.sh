#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: run-narrow-copy-ab-910b.sh no-profile|profile}
case "$arm" in
  no-profile) profile_flag=--no-profile ;;
  profile) profile_flag=--profile ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

base=/data/hanle/ascend-optimization/runtime/kg-controller
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
seed="$project_dir/reports/ascend910b/narrow-copy-20261005/ab-shared-seed.py"
prompt="$project_dir/experiments/ascend910b/narrow_copy/ab-prompt.md"
workspace="$base/runs/narrow-copy-ab-20261005-$arm"
if test -e "$workspace"; then
  echo "workspace already exists: $workspace" >&2
  exit 2
fi

source /data/yy/kgen/kernelgen/env.sh
export PATH="$base/bin:$PATH"
exec "$base/venv/bin/kg" run \
  --mode simple_opt \
  --definition narrow_copy \
  --catalog-name flaggems-adapter-definitions \
  --eval-server http://127.0.0.1:19652 \
  --runtime claude \
  --model 'deepseek-v4-flash[1m]' \
  "$profile_flag" \
  --min-rounds 2 --max-round 2 --early-stop-rounds 0 \
  --max-coder-sessions 1 --eval-timeout-seconds 1500 \
  --warmup-ms 1000 --benchmark-ms 100 --num-trials 3 \
  --seed-code-path "$seed" \
  --reference-code-path "$seed" \
  --reference-code-prompt-path "$prompt" \
  --workspace "$workspace"
