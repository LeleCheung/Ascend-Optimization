#!/usr/bin/env bash
# 宿主机入口：全程持有物理卡7锁，tmux可托管；不改服务、镜像或他人进程。
set -euo pipefail
workflow_batch=${1:?请指定新批次名}
workflow_mode=${2:-all}
workflow_rounds=${3:-3}
[[ "$workflow_batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$workflow_mode" =~ ^(all|replay|method)$ ]] || exit 2
[[ "$workflow_rounds" =~ ^[1-6]$ ]] || exit 2
workflow_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
workflow_parent=$(dirname "$workflow_root")
mkdir -p "$workflow_parent/runs/$workflow_batch/campaign-logs"
exec 8>"$workflow_parent/runs/physical-card-7-campaign.lock"
flock -n 8 || { echo '物理卡7已有本项目任务，停止重复启动'; exit 2; }
docker exec -i -e ASCEND_WORKFLOW_BATCH="$workflow_batch" \
 -e ASCEND_WORKFLOW_MODE="$workflow_mode" -e ASCEND_WORKFLOW_ROUNDS="$workflow_rounds" \
 tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
"$py" "$p/tools/profiling/four_operator_workflow.py" \
 --batch "$ASCEND_WORKFLOW_BATCH" --mode "$ASCEND_WORKFLOW_MODE" --rounds "$ASCEND_WORKFLOW_ROUNDS"
RUN
