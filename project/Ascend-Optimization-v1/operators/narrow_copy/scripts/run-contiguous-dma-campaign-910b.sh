#!/usr/bin/env bash
# 接在四版本和独立开销分解之后，保留原先评测结果。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
for session in kg-narrow-grid-device-master-20261010 kg-narrow-task-scopes-20261010; do
 while tmux has-session -t "$session" 2>/dev/null; do sleep 15; done
done
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/contiguous-dma-hybrid-20261010.py" \
 --server http://127.0.0.1:19657 \
 --output "$p/operators/narrow_copy/reports/contiguous-dma-grid-device-20261010" \
 --label contiguous-dma-hybrid --repeats 2 --timeout 1800 --timing-scope device_task
"$py" "$p/operators/narrow_copy/scripts/finalize-grid-delivery.py" "$p"
