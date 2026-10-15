#!/usr/bin/env bash
# 910B 宿主执行；不修改服务、镜像或原始实验批次。
set -euo pipefail
repro_batch=${1:?请指定未使用的新批次名}
[[ "$repro_batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
repro_project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
repro_repo=$(cd "$repro_project/../.." && pwd)
repro_root=$(dirname "$repro_repo")
exec 9>"$repro_root/runs/physical-card-7-campaign.lock"
flock -n 9 || { echo '物理卡7已有本项目任务'; exit 2; }
docker exec -i -e REPRO_BATCH="$repro_batch" tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
op="$p/operators/rad2deg_"
out="$op/reports/reproduce-$REPRO_BATCH"
[[ ! -e "$out" ]] || { echo '输出已存在，请换新批次名'; exit 2; }
mkdir -p "$out"
"$py" "$p/tools/profiling/hgkda/verify-baseline.py" \
  --server http://127.0.0.1:19655 --output "$out/baseline-verified.json"
"$py" "$p/tools/profiling/hgkda/skills/hgkda/scripts/check_rad2deg.py" \
  --source "$op/candidates/hgkda-final-20261015.py" --output "$out/supplementary.json"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator rad2deg_ \
  --source "$op/candidates/flaggems-fixed.py" --label FlagGems --output "$out/baseline"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator rad2deg_ \
  --source "$op/candidates/hgkda-final-20261015.py" --label HG-KDA --repeats 2 --output "$out/final"
"$py" "$p/tools/evaluation/compare-operator-versions.py" \
  --version FlagGems "$out/baseline/FlagGems-1.result.json" \
  --version HG-KDA-1 "$out/final/HG-KDA-1.result.json" \
  --version HG-KDA-2 "$out/final/HG-KDA-2.result.json" --output "$out/版本对比.md"
RUN
