set -eu
cd /data/hanle/ascend-optimization/Ascend-Optimization
export PYTHONDONTWRITEBYTECODE=1
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u -m tools.skill_validation preflight --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -u -m tools.skill_validation evaluate --config /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/log10-regions/config.json --port 19698
