#!/usr/bin/env bash
# 大网格分批启动后，在已修复 N/A 解析的合同中完成四版本。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
server=http://127.0.0.1:19657
while tmux has-session -t kg-narrow-large-launch-20261010 2>/dev/null; do sleep 15; done
exec 8>"$r/narrow-grid-campaign.lock"
flock -n 8 || exit 2
seed=$p/operators/narrow_copy/candidates/flaggems-master-grid-compatible-20261010.py
baseline=$p/operators/narrow_copy/reports/master-grid-device-20261010
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy --source "$seed" \
 --server "$server" --output "$baseline" --label flaggems-master-grid --timeout 1800 --timing-scope device_task
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/contiguous-copy-20261010.py" \
 --server "$server" --output "$p/operators/narrow_copy/reports/contiguous-copy-grid-device-20261010" \
 --label contiguous-copy --timeout 1800 --timing-scope device_task
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/small-dma-hybrid-20261010.py" \
 --server "$server" --output "$p/operators/narrow_copy/reports/small-dma-grid-device-20261010" \
 --label small-dma-hybrid --timeout 1800 --timing-scope device_task || echo 'DMA_HYBRID_FAILED'
export KG_RUN_ROOT="$r/kg-runs" KG_SOURCE_ROOT="$r/Ascend-Optimization/sources"
export KG_WALLTIME_PORT=19657 KG_PROFILE_PORT=19657
for arm in no-profile profile; do
 bash "$p/tools/evaluation/run-kg-operator-910b.sh" narrow_copy "$arm" "$arm-grid-device-master-20261010" 2 "$seed" \
  >"$r/campaign-logs/narrow-grid-device-$arm.log" 2>&1 || echo "NARROW_KG_FAILED_$arm"
 "$py" "$p/tools/evaluation/finalize-kg-arm.py" \
  "$r/kg-runs/narrow_copy-$arm-grid-device-master-20261010" narrow_copy \
  "$p/operators/narrow_copy/reports/kg-$arm-grid-device-master-20261010" \
  --label "kg-$arm" --server "$server" --timing-scope device_task \
  || echo "NARROW_KG_INDEPENDENT_FAILED_$arm"
done
"$py" "$p/tools/evaluation/collect-optimized-profiles.py" narrow_copy \
 "$p/operators/narrow_copy/reports/contiguous-copy-grid-device-20261010/contiguous-copy-1.result.json" \
 "$p/operators/narrow_copy/reports/optimized-grid-device-profile-20261010" \
 --server "$server" --kernel-prefix narrow_copy_contiguous_kernel --timing-scope device_task \
 --case-id 'benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::0' \
 --case-id 'benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::4' \
 || echo 'NARROW_OPTIMIZED_PROFILE_FAILED'
"$py" - "$p" <<'PY'
import json,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);reports=p/'operators/narrow_copy/reports'
valid=[]
for directory in ('contiguous-copy-grid-device-20261010','small-dma-grid-device-20261010'):
    for f in (reports/directory).glob('*.result.json'):
        d=json.loads(f.read_text())
        if d.get('status')=='PASSED' and d.get('num_passed')==33 and d.get('num_workloads')==33:
            valid.append((d['geo_mean'],f))
best=max(valid)[1]
args=[sys.executable,str(p/'tools/evaluation/compare-operator-versions.py'),
      '--version','master-compatible',str(reports/'master-grid-device-20261010/flaggems-master-grid-1.result.json')]
for arm in ('no-profile','profile'):
    args+=['--version','kg-'+arm,str(reports/('kg-'+arm+'-grid-device-master-20261010')/'independent'/('kg-'+arm+'-1.result.json'))]
args+=['--version','optimized',str(best),'--output',str(reports/'master-closure-20261010/版本对比.md')]
subprocess.run(args,check=True)
print('NARROW_FINAL_BEST',str(best),flush=True)
PY
"$py" "$p/tools/evaluation/summarize-master-goal.py" "$p" \
 "$p/operators/低于0.8算子四版本闭环-20261010.md"
