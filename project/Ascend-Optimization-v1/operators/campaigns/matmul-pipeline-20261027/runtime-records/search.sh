#!/bin/bash
set -euo pipefail
exec 9>/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock
flock -w 600 9
docker exec -i tle_yy bash -s <<'INNER'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/client:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server:/data/hanle/ascend-optimization/Ascend-Optimization/sources
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
/usr/local/python3.11.15/bin/python3.11 -u tools/profiling/hgkda/matmul_pipeline/search.py --parent operators/campaigns/roofline-20261027/validation/delivered.py --output operators/campaigns/matmul-pipeline-20261027/search --configs dual-n-constexpr dual-m-reuse-b quad-64x128 quad-128x64 four-n-64 dual-n-saving dual-n-interleave

INNER
