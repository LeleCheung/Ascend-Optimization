#!/bin/bash
set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
exec /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u /data/hanle/ascend-optimization/Ascend-Optimization/tools/regression/run.py --plan /data/hanle/ascend-optimization/runs/independent-regression-20261109-v3/plan.json --container tle_yy --device 7 --port 19663 --lock /data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock --device-python /usr/local/python3.11.15/bin/python3.11 --cann-environment /usr/local/Ascend/cann-9.0.0/set_env.sh >> /data/hanle/ascend-optimization/runs/independent-regression-20261109-v3/controller.log 2>&1
