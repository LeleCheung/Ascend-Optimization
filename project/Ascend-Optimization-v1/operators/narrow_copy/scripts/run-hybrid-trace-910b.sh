#!/usr/bin/env bash
# 排在本轮独立评测及开销分解之后，同卡保持串行。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
for session in kg-narrow-grid-device-master-20261010 kg-narrow-task-scopes-20261010 kg-narrow-dma-campaign-20261010; do
 while tmux has-session -t "$session" 2>/dev/null; do sleep 15; done
done
/usr/local/python3.11.15/bin/python3.11 "$p/tools/evaluation/debug-operator.py" \
 --server http://127.0.0.1:19657 --script "$p/operators/narrow_copy/scripts/probe-hybrid-copy-trace.py" \
 --file "$p/operators/narrow_copy/candidates/small-dma-hybrid-20261010.py" \
 --file "$p/operators/narrow_copy/candidates/contiguous-dma-hybrid-20261010.py" \
 --output "$r/diagnostics/narrow-hybrid-copy-trace-20261010" --timeout 900
