#!/usr/bin/env bash
# 与原生 Coder 和 amin 诊断串行。
set -euo pipefail
r=${1:?usage: run-wide-tiles-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
while tmux has-session -t kg-master-continue-v2-20261010 2>/dev/null || \
      tmux has-session -t kg-amin-direct-tiles-20261010 2>/dev/null; do sleep 15; done
/usr/local/python3.11.15/bin/python3.11 "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/matmul_bias_activation/scripts/probe-wide-tiles.py" \
 --file "$p/operators/matmul_bias_activation/scripts/probe-pipeline-tiles.py" \
 --file "$p/operators/matmul_bias_activation/candidates/profiling-k256-dotacc-v4-20261010.py" \
 --output "$r/diagnostics/matmul-wide-tiles-20261010" --timeout 1800
