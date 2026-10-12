#!/usr/bin/env bash
# 由宿主机 tmux 启动，KGS debug 队列负责物理卡7的实验。
set -euo pipefail
r=/data/hanle/ascend-optimization
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
batch=$r/runs/20261012
mkdir -p "$batch/campaign-logs"
exec 8>"$batch/matmul-optimization.lock"
flock -n 8 || exit 2
docker exec -i tle_yy bash -s <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
"$py" - <<'PY'
import json,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in (19655,19657):
    d=json.load(opener.open(f'http://127.0.0.1:{port}/status',timeout=10))['scheduler']
    assert d['active']==0 and d['waiting']==0 and d['healthy']==1, d
PY
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/matmul_bias_activation/scripts/probe-dual-acc-20261012.py" \
 --file "$p/operators/matmul_bias_activation/candidates/profiling-dual-dispatch-v8-20261010.py" \
 --output "$p/operators/matmul_bias_activation/reports/dual-acc-probe-20261012" \
 --timeout 1800
BASH
