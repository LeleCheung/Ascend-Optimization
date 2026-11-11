set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
export PYTHONDONTWRITEBYTECODE=1
for attempt in $(seq 1 900); do
 if test -f /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/attempts/25e8b11f2c2f8e40e2637b0fdd981c1f18b813d04b9d9f1f7d8cddec63cc05fe/preflight.json; then
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["status"]=="passed" else 2)' /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/attempts/25e8b11f2c2f8e40e2637b0fdd981c1f18b813d04b9d9f1f7d8cddec63cc05fe/preflight.json
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u -m tools.skill_validation evaluate --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json --port 19698 > /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-01.log 2>&1
  exit $?
 fi
 sleep 1
done
exit 2
