#!/usr/bin/env bash
# 宿主机 tmux 持续运行矩阵乘、amin 两个独立闭环；同卡全程串行。
set -euo pipefail
r=/data/hanle/ascend-optimization
batch=$r/runs/20261012-agent
mkdir -p "$batch/campaign-logs"
exec 8>"$r/runs/physical-card-7-campaign.lock"
flock -n 8 || exit 2
docker exec -i tle_yy bash -s <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
"$py" - <<'PY'
import json,urllib.request
for port in (19655,19657):
 s=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/status'))['scheduler']
 assert s['active']==s['waiting']==s['broken']==0 and s['healthy']==1,s
PY
for operator in matmul_bias_activation amin; do
 if test "$operator" = matmul_bias_activation; then
  seed=profiling-dual-acc-v9-20261012.py
 else
  seed=direct-axis-native-v5-20261010.py
 fi
 "$py" "$p/tools/profiling/optimization_campaign.py" \
  --operator "$operator" --seed "$p/operators/$operator/candidates/$seed" \
  --output "$p/operators/$operator/reports/profiling-agent-20261012-agent" \
  --knowledge "$p/tools/profiling/910B优化知识.md" --rounds 6 --profile-cases 2
done
"$py" "$p/tools/profiling/screen-next-operator.py" --batch 20261012-agent
BASH
