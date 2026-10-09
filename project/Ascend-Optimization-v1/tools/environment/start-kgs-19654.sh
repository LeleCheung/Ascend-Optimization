#!/usr/bin/env bash
set -euo pipefail
export KGS_FLAGGEMS_ROOT=/data/hanle/ascend-optimization/FlagGems
export PYTHONPATH=/usr/local/python3.11.15/lib/python3.11/site-packages:/data/hanle/ascend-optimization/Ascend-Optimization/sources:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/client:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server:${PYTHONPATH:-}
exec /usr/local/python3.11.15/bin/python3.11 -c 'from kernelgen_server.server import main; main()' \
  --backend npu --timing walltime --max-workers 1 --port 19654 \
  --profile-artifact-root /data/hanle/ascend-optimization/runtime/kg-controller/runs/profiles-phase1-4772d816
