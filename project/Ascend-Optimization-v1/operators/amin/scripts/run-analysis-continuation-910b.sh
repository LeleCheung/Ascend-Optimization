#!/usr/bin/env bash
# 待原生 matmul profile 完成，再串行评测本轮分析候选。
set -euo pipefail
r=${1:?usage: run-analysis-continuation-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
while tmux has-session -t kg-master-continue-v2-20261010 2>/dev/null; do sleep 15; done
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator amin \
 --source "$p/operators/amin/candidates/direct-axis-tiles-v4-20261010.py" \
 --output "$p/operators/amin/reports/direct-axis-tiles-v4-20261010" \
 --label direct-axis-tiles-v4 --timeout 1800
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/amin/scripts/probe-native-reduction.py" \
 --output "$r/diagnostics/amin-native-reduction-20261010" --timeout 1800
