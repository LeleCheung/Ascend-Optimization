#!/usr/bin/env bash
set -euo pipefail
root=/data/hanle/ascend-optimization
p="$root/Ascend-Optimization/project/Ascend-Optimization-v1"
bash "$p/tools/profiling/run-four-operator-workflow.sh" 20261014-four-refined method 2 20261014-four
