#!/usr/bin/env bash
# 宿主机 tmux 中执行两次完整合同评测。
set -euo pipefail
r=/data/hanle/ascend-optimization
mkdir -p "$r/runs/20261012/campaign-logs"
exec 8>"$r/runs/20261012/matmul-optimization.lock"
flock -n 8 || exit 2
docker exec -i tle_yy bash -s <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
py=/usr/local/python3.11.15/bin/python3.11
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
"$py" - <<'PY'
import json,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in (19655,19657):
    d=json.load(opener.open(f'http://127.0.0.1:{port}/status',timeout=10))['scheduler']
    assert d['active']==0 and d['waiting']==0 and d['healthy']==1,d
PY
"$py" "$p/tools/evaluation/evaluate-operator.py" \
 --operator matmul_bias_activation \
 --source "$p/operators/matmul_bias_activation/candidates/profiling-dual-acc-v9-20261012.py" \
 --output "$p/operators/matmul_bias_activation/reports/profiling-dual-acc-v9-20261012" \
 --label optimized-v9 --repeats 2 --timeout 1800 --timing-scope device_kernel
BASH
