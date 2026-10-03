#!/usr/bin/env bash
set -euo pipefail

root=/data/hanle/ascend-optimization/runtime/kg-controller
repo=/data/hanle/ascend-optimization/Ascend-Optimization
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ANTHROPIC_MODEL=deepseek-flash
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
export PATH="$root:$root/venv/bin:$PATH"
workspace="${1:-$root/runs/square-$(date +%Y%m%d-%H%M%S)}"

exec "$root/venv/bin/kg" run --mode simple_opt \
  --definition kernelgenbench_square --catalog-path "$repo/runtime/catalogs/catalog-square" \
  --eval-server http://127.0.0.1:19650 --workspace "$workspace" \
  --runtime claude --model deepseek-flash --no-profile \
  --num-trials 3 \
  --min-rounds 1 --max-round 1 --early-stop-rounds 1 --max-coder-sessions 1
