#!/usr/bin/env bash
# 两组自动任务结束后复验历史对照和整理报告；不与任何同卡任务并行。
set -euo pipefail
r=/data/hanle/ascend-optimization
for session in profiling-agent-20261012 profiling-new-operator-20261012; do
 while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
done
exec 8>"$r/runs/physical-card-7-campaign.lock"
flock -n 8 || exit 2
docker exec -i tle_yy bash -s <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
"$py" "$p/tools/profiling/finalize-agent-campaign.py" --operator matmul_bias_activation --prior-closure master-closure-20261012
"$py" "$p/tools/profiling/finalize-agent-campaign.py" --operator amin --prior-closure master-closure-20261010
BASH
