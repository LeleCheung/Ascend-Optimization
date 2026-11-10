set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/method-package-v3/src
for attempt in $(seq 1 1200); do
 if test -f /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-cbb329b76639/performance-feedback.json; then
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u tools/skill_validation/collect_pair.py --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json --source /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/workspace/candidate.py --case log10_-timing-3 --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/paired-after-cbb3-timing3
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u tools/skill_validation/collect_simulator.py --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json --paired-directory /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/paired-after-cbb3-timing3 --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/simulator-after-cbb3 --soc Ascend910B4 --simulator-library /usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib --trace-unit us --timeout-minutes 15 --allow-postcapture-exit
  exit 0
 fi
 sleep 1
done
exit 2
