#!/usr/bin/env bash
# 在相同仓库 HEAD 下复验三组已交付源码；不重跑 KG 优化、不改变其候选。
set -euo pipefail
r=/data/hanle/ascend-optimization
exec 8>"$r/runs/20261012/matmul-optimization.lock"
flock -n 8 || exit 2
docker exec -i tle_yy bash -s <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
py=/usr/local/python3.11.15/bin/python3.11
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
op=$p/operators/matmul_bias_activation
"$py" - <<'PY'
import json,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in (19655,19657):
    d=json.load(opener.open(f'http://127.0.0.1:{port}/status',timeout=10))['scheduler']
    assert d['active']==0 and d['waiting']==0 and d['healthy']==1,d
PY
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator matmul_bias_activation \
 --source "$op/reports/master-20261010/flaggems-master.py" \
 --output "$op/reports/master-closure-20261012/baselines/master" \
 --label master --timeout 1800 --timing-scope device_kernel
for arm in no-profile profile; do
 "$py" "$p/tools/evaluation/evaluate-operator.py" --operator matmul_bias_activation \
  --source "$op/reports/kg-$arm-master-20261010/independent/kg-$arm.py" \
  --output "$op/reports/master-closure-20261012/baselines/kg-$arm" \
  --label "kg-$arm" --timeout 1800 --timing-scope device_kernel
done
BASH
