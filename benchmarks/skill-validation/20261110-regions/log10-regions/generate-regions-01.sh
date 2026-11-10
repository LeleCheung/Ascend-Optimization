set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
export PYTHONDONTWRITEBYTECODE=1
for attempt in $(seq 1 1200); do
  if test -f /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/simulator-before-complete/summary.json; then
    /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u -m tools.skill_validation generate --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json --invocation regions-01 --max-turns 16 > /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/generate-regions-01.log 2>&1
    exit $?
  fi
  sleep 1
done
exit 2
