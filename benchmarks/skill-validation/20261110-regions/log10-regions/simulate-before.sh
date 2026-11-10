set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
export PYTHONDONTWRITEBYTECODE=1
for attempt in $(seq 1 900); do
  if test -f /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/paired-45c7380a8062-log10_-timing-3/context.json; then
    /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u -m tools.skill_validation simulate --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json --paired-directory /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/paired-45c7380a8062-log10_-timing-3 --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/simulator-before --soc Ascend910B4 --simulator-library /usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib --trace-unit us > /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/simulate-before.log 2>&1
    exit $?
  fi
  sleep 1
done
exit 2
