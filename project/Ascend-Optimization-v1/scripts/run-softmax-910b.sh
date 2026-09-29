#!/usr/bin/env bash
set -euo pipefail

root=/data/hanle/ascend-optimization/runtime/kg-controller
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ANTHROPIC_MODEL=deepseek-flash
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
export PATH="$root:$root/venv/bin:$PATH"

exec "$root/venv/bin/kg" run --mode simple_opt \
  --definition softmax \
  --catalog-path "$root/catalog-softmax" \
  --eval-server http://127.0.0.1:19650 \
  --workspace "$root/runs/softmax-native-claude-1" \
  --runtime claude --model deepseek-flash --no-profile \
  --min-rounds 1 --max-round 1 --early-stop-rounds 1 \
  --max-coder-sessions 1
