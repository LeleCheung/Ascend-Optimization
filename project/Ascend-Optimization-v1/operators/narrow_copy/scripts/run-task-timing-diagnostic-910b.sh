#!/usr/bin/env bash
# 等待本轮四版本任务结束，再通过同一个 KGS 队列采集独立开销分解。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
while tmux has-session -t kg-narrow-grid-device-master-20261010 2>/dev/null; do sleep 15; done
/usr/local/python3.11.15/bin/python3.11 "$p/tools/evaluation/debug-operator.py" \
 --server http://127.0.0.1:19657 \
 --script "$p/operators/narrow_copy/scripts/probe-current-task-timing.py" \
 --file "$p/operators/narrow_copy/scripts/probe-current-timing.py" \
 --file "$p/operators/narrow_copy/scripts/ascend-copy-timer.py" \
 --file "$p/operators/narrow_copy/candidates/contiguous-copy-20261010.py" \
 --output "$r/diagnostics/narrow-current-device-task-scopes-20261010" --timeout 900
