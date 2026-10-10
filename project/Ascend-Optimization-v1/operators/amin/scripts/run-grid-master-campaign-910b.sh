#!/usr/bin/env bash
# 超限启动兼容修复后的基线与两组原生 KG；失败证据保留，组间不传优化意见。
set -euo pipefail
r=${1:?usage: run-grid-master-campaign-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
seed=$p/operators/amin/candidates/flaggems-master-grid-adapter-20261010.py
exec 9>"$r/amin-grid-campaign.lock"
flock -n 9 || exit 2
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator amin --source "$seed" \
 --output "$p/operators/amin/reports/master-grid-adapter-20261010" \
 --label flaggems-master-grid-adapted --timeout 1800
export KG_RUN_ROOT="$r/kg-runs" KG_SOURCE_ROOT="$r/Ascend-Optimization/sources"
export KG_WALLTIME_PORT=19655 KG_PROFILE_PORT=19655
for arm in no-profile profile; do
 bash "$p/tools/evaluation/run-kg-operator-910b.sh" amin "$arm" \
  "$arm-grid-master-20261010" 2 "$seed" \
  >"$r/campaign-logs/amin-grid-master-$arm.log" 2>&1 \
  || echo "AMIN_KG_FAILED_$arm"
done
