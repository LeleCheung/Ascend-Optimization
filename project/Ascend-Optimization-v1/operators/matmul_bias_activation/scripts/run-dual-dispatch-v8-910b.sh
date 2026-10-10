#!/usr/bin/env bash
# 等本轮 narrow 队列全部结束，同物理卡串行完整复验合并候选。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
for session in kg-narrow-grid-device-master-20261010 kg-narrow-task-scopes-20261010 kg-narrow-dma-campaign-20261010 kg-narrow-hybrid-trace-20261010; do
 while tmux has-session -t "$session" 2>/dev/null; do sleep 15; done
done
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator matmul_bias_activation \
 --source "$p/operators/matmul_bias_activation/candidates/profiling-dual-dispatch-v8-20261010.py" \
 --server http://127.0.0.1:19655 \
 --output "$p/operators/matmul_bias_activation/reports/profiling-dual-dispatch-v8-20261010" \
 --label profiling-dual-dispatch-v8 --repeats 2 --timeout 1800 --timing-scope device_kernel
