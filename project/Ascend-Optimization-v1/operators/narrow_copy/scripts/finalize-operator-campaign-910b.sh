#!/usr/bin/env bash
# 原生两组结束后验证分析候选并采集设备指标；不与同卡任务并发。
set -euo pipefail
r=${1:?EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
server=http://127.0.0.1:19658
while tmux has-session -t kg-narrow-operator-master-20261010 2>/dev/null; do sleep 15; done
exec 8>"$r/narrow-operator-finalize.lock"
flock -n 8 || exit 2
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator narrow_copy \
 --source "$p/operators/narrow_copy/candidates/small-dma-hybrid-20261010.py" \
 --server "$server" --output "$p/operators/narrow_copy/reports/small-dma-hybrid-operator-20261010" \
 --label small-dma-hybrid --timeout 1800 --timing-scope walltime || echo 'DMA_HYBRID_FAILED'
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/narrow_copy/scripts/probe-current-task-timing.py" \
 --file "$p/operators/narrow_copy/scripts/probe-current-timing.py" \
 --file "$p/operators/narrow_copy/scripts/ascend-copy-timer.py" \
 --file "$p/operators/narrow_copy/candidates/contiguous-copy-20261010.py" \
 --server "$server" --output "$r/diagnostics/narrow-current-task-timing-20261010" --timeout 900 \
 || echo 'NARROW_TIMING_SCOPES_FAILED'
"$py" "$p/tools/evaluation/collect-optimized-profiles.py" narrow_copy \
 "$p/operators/narrow_copy/reports/contiguous-copy-operator-20261010/contiguous-copy-1.result.json" \
 "$p/operators/narrow_copy/reports/optimized-operator-profile-20261010" \
 --server "$server" --kernel-prefix narrow_copy_contiguous_kernel --timing-scope walltime \
 --case-id 'benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::0' \
 --case-id 'benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::4' \
 || echo 'NARROW_OPTIMIZED_PROFILE_FAILED'
"$py" - "$p" <<'PY'
import json,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);reports=p/'operators/narrow_copy/reports'
valid=[]
for directory in ('contiguous-copy-operator-20261010','small-dma-hybrid-operator-20261010'):
    for f in (reports/directory).glob('*.result.json'):
        d=json.loads(f.read_text())
        if d.get('status')=='PASSED' and d.get('num_passed')==33 and d.get('num_workloads')==33:
            valid.append((d['geo_mean'],f))
best=max(valid)[1]
args=[sys.executable,str(p/'tools/evaluation/compare-operator-versions.py'),
      '--version','master',str(reports/'master-operator-adapter-20261010/flaggems-master-1.result.json')]
for arm in ('no-profile','profile'):
    args+=['--version','kg-'+arm,str(reports/('kg-'+arm+'-operator-master-20261010')/'independent'/('kg-'+arm+'-1.result.json'))]
args+=['--version','optimized',str(best),'--output',str(reports/'master-closure-20261010/版本对比.md')]
subprocess.run(args,check=True)
print('NARROW_FINAL_BEST',str(best),flush=True)
PY
"$py" "$p/tools/evaluation/summarize-master-goal.py" "$p" \
 "$p/operators/低于0.8算子四版本闭环-20261010.md"
