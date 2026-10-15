#!/usr/bin/env bash
# 在宿主 tmux 中执行；串行使用现有物理卡7与KGS，不修改共享服务。
set -euo pipefail
hgkda_batch=${1:?请指定新的批次名}
[[ "$hgkda_batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
hgkda_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
hgkda_root=$(dirname "$hgkda_repo")
mkdir -p "$hgkda_root/runs/$hgkda_batch"
exec 8>"$hgkda_root/runs/physical-card-7-campaign.lock"
flock -n 8 || { echo '物理卡7已有本项目任务，停止重复启动'; exit 2; }
docker exec -i -e HGKDA_BATCH="$hgkda_batch" tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
out="$p/operators/rad2deg_/reports/hgkda-$HGKDA_BATCH"
mkdir -p "$out"
"$py" "$p/tools/profiling/hgkda/verify-baseline.py" \
  --server http://127.0.0.1:19655 --output "$out/baseline-verified.json"
"$py" "$p/tools/profiling/optimization_campaign.py" \
  --operator rad2deg_ --seed "$p/operators/rad2deg_/candidates/flaggems-fixed.py" \
  --output "$out" --knowledge "$p/tools/profiling/hgkda/阶段通用知识.md" \
  --server http://127.0.0.1:19655 --rounds 2 --profile-cases 2 \
  --plain-json --provider-thinking disabled --hgkda
RUN
