set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
export PYTHONDONTWRITEBYTECODE=1
for attempt in $(seq 1 600); do
 if test -f /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/log10-edges.json; then
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u tools/regression/run.py --plan /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/plan.json --container tle_yy --device 7 --port 19698 --lock /data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock --device-python /usr/local/python3.11.15/bin/python3.11 --cann-environment /usr/local/Ascend/cann-9.0.0/set_env.sh
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python tools/regression/audit.py --run-directory /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/audit-final.json
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python tools/regression/summarize.py --run-directory /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/summarize-final.json
  /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -m hgkda feedback --result /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/operators/log10_/evaluation/result.json --audit /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/audit-final.json --source /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/operators/log10_/candidate.py --operator log10_ --timing-cases /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/reference-catalog/timing.jsonl --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/performance-feedback.json
  cp /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/performance-feedback.json /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/workspace/performance-feedback.json
  exit 0
 fi
 sleep 1
done
exit 2
