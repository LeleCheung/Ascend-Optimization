#!/usr/bin/env bash
# 本轮生成任务结束后串行独立复验；不会向原生 KG 组注入建议。
set -euo pipefail
r=${1:?usage: finalize-master-campaign-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
exec 8>"$r/finalize-master.lock"
flock -n 8 || exit 2
sessions=(kg-master-continue-v2-20261010 kg-master-no-profile-recovery-20261010
          kg-v2-and-amin-v4-20261010 kg-amin-v4-continuation-20261010
          kg-matmul-v3-eval-20261010 kg-roofline-and-global-20261010
          kg-matmul-tuning-20261010 kg-amin-grid-master-20261010
          kg-matmul-v4-eval-20261010 kg-amin-direct-tiles-20261010)
while true; do
 running=0
 for session in "${sessions[@]}"; do
  tmux has-session -t "$session" 2>/dev/null && running=1
 done
 test "$running" -eq 0 && break
 sleep 15
done
for operator in matmul_bias_activation amin; do
 if test "$operator" = amin; then suffix=grid-master-20261010; else suffix=master-20261010; fi
 for arm in no-profile profile; do
  workspace=$r/kg-runs/$operator-$arm-$suffix
  if ! test -d "$workspace"; then echo "MISSING_KG_WORKSPACE $operator $arm"; continue; fi
  "$py" "$p/tools/evaluation/finalize-kg-arm.py" "$workspace" "$operator" \
   "$p/operators/$operator/reports/kg-$arm-$suffix" --label "kg-$arm" \
   || echo "KG_INDEPENDENT_FAILED $operator $arm"
 done
 if test "$operator" = amin; then
  baseline=$p/operators/amin/reports/master-grid-adapter-20261010/flaggems-master-grid-adapted-1.result.json
  baseline_label=master-compatible
 else
  baseline=$p/operators/matmul_bias_activation/reports/master-20261010/flaggems-master-1.result.json
  baseline_label=master
 fi
 # 从已完成的分析候选中选择完整通过且整体最快的一份，原始失败始终保留。
 # 四版本工具还会核验源码、用例、设备和 benchmark 指纹。
 optimized=$("$py" - "$p/operators/$operator/reports" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); candidates=[]
for directory in root.iterdir():
    if '20261010' not in directory.name or not directory.name.startswith(('direct-axis-', 'profiling-k')):
        continue
    for path in directory.glob('*.result.json'):
        result=json.loads(path.read_text())
        if (result.get('status')=='PASSED' and result.get('num_passed')==result.get('num_workloads')
                and isinstance(result.get('geo_mean'),(float,int)) and result['geo_mean']>0):
            candidates.append((result['geo_mean'],str(path)))
if not candidates:raise SystemExit('没有完整通过的分析候选')
print(max(candidates)[1])
PY
 ) || { echo "MISSING_VALID_OPTIMIZED $operator"; continue; }
 "$py" "$p/tools/evaluation/compare-operator-versions.py" \
  --version "$baseline_label" "$baseline" \
  --version kg-no-profile "$p/operators/$operator/reports/kg-no-profile-$suffix/independent/kg-no-profile-1.result.json" \
  --version kg-profile "$p/operators/$operator/reports/kg-profile-$suffix/independent/kg-profile-1.result.json" \
  --version optimized "$optimized" \
  --output "$p/operators/$operator/reports/master-closure-20261010/版本对比.md" \
  || echo "FOUR_VERSION_CLOSURE_INCOMPLETE $operator"
done
