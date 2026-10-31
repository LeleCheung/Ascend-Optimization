#!/bin/bash
set -euo pipefail
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/matmul_pipeline/review.py --batch /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/matmul-pipeline-20261027
