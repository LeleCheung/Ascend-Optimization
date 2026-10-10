#!/usr/bin/env bash
# 同卡依次等待已有实验，完整复验最终合并候选。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
for session in kg-narrow-grid-device-master-20261010 kg-narrow-task-scopes-20261010 kg-narrow-dma-campaign-20261010 kg-narrow-hybrid-trace-20261010 kg-matmul-dual-dispatch-v8-20261010; do
 while tmux has-session -t "$session" 2>/dev/null; do sleep 15; done
done
"$py" "$p/operators/narrow_copy/scripts/build-profile-dma-hybrid.py"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/profile-small-dma-hybrid-20261010.py" \
 --server http://127.0.0.1:19657 \
 --output "$p/operators/narrow_copy/reports/profile-small-dma-grid-device-20261010" \
 --label profile-small-dma-hybrid --repeats 2 --timeout 1800 --timing-scope device_task
"$py" "$p/operators/narrow_copy/scripts/finalize-grid-delivery.py" "$p"
"$py" "$p/tools/evaluation/debug-operator.py" \
 --server http://127.0.0.1:19657 --script "$p/operators/narrow_copy/scripts/probe-profile-dma-trace.py" \
 --file "$p/operators/narrow_copy/scripts/probe-hybrid-copy-trace.py" \
 --file "$p/operators/narrow_copy/candidates/profile-small-dma-hybrid-20261010.py" \
 --output "$r/diagnostics/narrow-profile-dma-trace-20261010" --timeout 900
