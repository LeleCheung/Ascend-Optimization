set -eu
exec 9>/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock
flock -w 600 9
docker exec tle_yy bash -c 'set -eu
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/special-check-cache
cd /data/hanle/ascend-optimization/Ascend-Optimization
/usr/local/python3.11.15/bin/python3.11 -m tools.skill_validation check-selu-offset --source /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/operators/selu_/candidate.py --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/selu-offset.json
/usr/local/python3.11.15/bin/python3.11 -m tools.skill_validation check-selu-runtime --source /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/operators/selu_/candidate.py --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/selu-runtime.json
/usr/local/python3.11.15/bin/python3.11 -m tools.skill_validation check-log10-edges --source /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/operators/log10_/candidate.py --output /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/log10-edges.json --leaf /data/hanle/ascend-optimization/runs/independent-regression-20261109-v3/operators/log10_/cann_supplementary-catalog
' > /data/hanle/ascend-optimization/runs/overnight-regions-20261110-v1/entry-regression/special-checks.log 2>&1
