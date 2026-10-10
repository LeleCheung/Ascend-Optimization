#!/usr/bin/env bash
# 先完成宽 tile 筛选，再完整评测；仅使用本次实验队列。
set -euo pipefail
r=${1:?usage: run-wide-v5-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
bash "$p/operators/matmul_bias_activation/scripts/run-wide-tiles-910b.sh" "$r"
"$py" "$p/operators/matmul_bias_activation/scripts/build-wide-v5.py" \
 "$r/diagnostics/matmul-wide-tiles-20261010/artifacts/wide-tiles.json"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator matmul_bias_activation \
 --source "$p/operators/matmul_bias_activation/candidates/profiling-kwide-v5-20261010.py" \
 --output "$p/operators/matmul_bias_activation/reports/profiling-kwide-v5-20261010" \
 --label profiling-kwide-v5 --timeout 1800
