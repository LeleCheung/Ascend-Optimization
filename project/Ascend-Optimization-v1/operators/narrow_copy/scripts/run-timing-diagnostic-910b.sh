#!/usr/bin/env bash
# 排在矩阵实验后，不与同卡实验同时运行。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
while tmux has-session -t kg-matmul-compiler-v7-20261010 2>/dev/null; do sleep 15; done
/usr/local/python3.11.15/bin/python3.11 "$p/tools/evaluation/debug-operator.py" \
 --server http://127.0.0.1:19656 --script "$p/operators/narrow_copy/scripts/probe-pytorch-copy-timing.py" \
 --output "$r/diagnostics/narrow-pytorch-copy-timing-20261010" --timeout 900
