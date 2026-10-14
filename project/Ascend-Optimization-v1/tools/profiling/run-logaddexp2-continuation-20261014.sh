#!/usr/bin/env bash
# 宿主机入口，仅使用物理卡7，全局锁保护持续候选优化。
set -euo pipefail
r=/data/hanle/ascend-optimization
p="$r/Ascend-Optimization/project/Ascend-Optimization-v1"
exec 8>"$r/runs/physical-card-7-campaign.lock"
flock -n 8 || exit 2
docker exec -i tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
"$py" "$p/tools/profiling/optimization_campaign.py" --operator logaddexp2 \
 --seed "$p/operators/logaddexp2/candidates/agent-bounded-grid-20261014.py" \
 --output "$p/operators/logaddexp2/reports/profiling-agent-20261014d-agent" \
 --rounds 6 --profile-cases 2 --knowledge "$p/tools/profiling/910B优化知识.md" --plain-json
RUN
