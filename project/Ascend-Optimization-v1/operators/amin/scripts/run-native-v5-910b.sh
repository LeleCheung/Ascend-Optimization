#!/usr/bin/env bash
# 使用同一被后续任务等待的 tmux 会话，避免评测交错。
set -euo pipefail
r=${1:?usage: run-native-v5-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
bash "$p/operators/amin/scripts/run-analysis-continuation-910b.sh" "$r"
"$py" "$p/operators/amin/scripts/build-native-v5.py" \
 "$r/diagnostics/amin-native-reduction-20261010/artifacts/native-reduction.json"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator amin \
 --source "$p/operators/amin/candidates/direct-axis-native-v5-20261010.py" \
 --output "$p/operators/amin/reports/direct-axis-native-v5-20261010" \
 --label direct-axis-native-v5 --timeout 1800
