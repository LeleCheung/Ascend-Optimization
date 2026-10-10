#!/usr/bin/env bash
# 原生 KG 与前序指标采集结束后验证 bf16 编译类型修复。
set -euo pipefail
r=${1:?usage: run-bf16-v6-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
exec 8>"$r/amin-bf16-v6.lock"
flock -n 8 || exit 2
sessions=(kg-master-no-profile-recovery-20261010 kg-amin-grid-master-20261010
          kg-finalize-master-20261010 kg-matmul-grouped-v6-20261010)
while true; do
 running=0
 for s in "${sessions[@]}"; do tmux has-session -t "$s" 2>/dev/null && running=1; done
 test "$running" = 0 && break
 sleep 15
done
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/amin/scripts/probe-bf16-cast.py" \
 --file "$p/operators/amin/candidates/direct-axis-native-v5-20261010.py" \
 --output "$r/diagnostics/amin-bf16-cast-20261010" --timeout 1800
"$py" "$p/operators/amin/scripts/build-bf16-v6.py" \
 "$r/diagnostics/amin-bf16-cast-20261010/artifacts/bf16-cast.json"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator amin \
 --source "$p/operators/amin/candidates/direct-axis-bf16-v6-20261010.py" \
 --output "$p/operators/amin/reports/direct-axis-bf16-v6-20261010" \
 --label direct-axis-bf16-v6 --timeout 1800
"$py" - "$p" <<'PY'
import json,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);reports=p/'operators/amin/reports'
valid=[]
for directory in reports.iterdir():
    if '20261010' not in directory.name or not directory.name.startswith('direct-axis-'):continue
    for f in directory.glob('*.result.json'):
        d=json.loads(f.read_text())
        if d.get('status')=='PASSED' and d.get('num_passed')==d.get('num_workloads'):
            valid.append((d['geo_mean'],f))
best=max(valid)[1]
args=[sys.executable,str(p/'tools/evaluation/compare-operator-versions.py'),
 '--version','master-compatible',str(reports/'master-grid-adapter-20261010/flaggems-master-grid-adapted-1.result.json')]
for arm in ('no-profile','profile'):
    args+=['--version','kg-'+arm,str(reports/('kg-'+arm+'-grid-master-20261010')/'independent'/('kg-'+arm+'-1.result.json'))]
args+=['--version','optimized',str(best),'--output',str(reports/'master-closure-20261010/版本对比.md')]
subprocess.run(args,check=True)
print('AMIN_BF16_FULL_VALIDATION_PASSED',str(best),flush=True)
PY
