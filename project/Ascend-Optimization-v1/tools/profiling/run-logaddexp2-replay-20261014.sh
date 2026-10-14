#!/usr/bin/env bash
# 固化历史有效方法并复现；仅使用物理卡7，同卡串行。
set -euo pipefail
replay_batch=${1:-20261014e-replay}
[[ "$replay_batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
r=/data/hanle/ascend-optimization
p="$r/Ascend-Optimization/project/Ascend-Optimization-v1"
exec 8>"$r/runs/physical-card-7-campaign.lock"
flock -n 8 || exit 2
docker exec -i -e ASCEND_REPLAY_BATCH="$replay_batch" tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
"$py" "$p/tools/profiling/optimization_campaign.py" --operator logaddexp2 \
 --seed "$p/operators/logaddexp2/reports/kg-profile-20261012-agent/export/main.py" \
 --output "$p/operators/logaddexp2/reports/profiling-agent-$ASCEND_REPLAY_BATCH" \
 --rounds 1 --profile-cases 2 \
 --knowledge "$p/tools/profiling/历史复现知识-20261014.md" \
 --plain-json --thinking-budget 8192 --max-output-tokens 16384 --replay
"$py" "$p/tools/profiling/verify-agent-campaign.py" \
 "$p/operators/logaddexp2/reports/profiling-agent-$ASCEND_REPLAY_BATCH"
"$py" "$p/tools/profiling/finalize-agent-campaign.py" --operator logaddexp2 \
 --batch 20261012-agent --campaign-batch "$ASCEND_REPLAY_BATCH"
RUN
