#!/usr/bin/env bash
# 排在本轮 narrow 之后；串行验证规则矩阵的 mask 与后端搬运选项。
set -euo pipefail
r=${1:?usage: run-compiler-v7-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
exec 8>"$r/matmul-compiler-v7.lock"
flock -n 8 || exit 2
sessions=(kg-amin-grid-master-20261010 kg-finalize-master-20261010
          kg-matmul-grouped-v6-20261010 kg-amin-bf16-v6-20261010 kg-narrow-master-20261010)
while true; do
 running=0
 for session in "${sessions[@]}"; do tmux has-session -t "$session" 2>/dev/null && running=1; done
 test "$running" = 0 && break
 sleep 15
done
"$py" - <<'PY'
import json,urllib.error,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in (19655,19656):
    try:
        response=opener.open(f'http://127.0.0.1:{port}/status',timeout=10)
    except urllib.error.URLError:
        if port==19655:raise
        continue
    with response:data=json.load(response)
    assert data['scheduler']['active']==0 and data['scheduler']['waiting']==0, '同卡服务仍有任务'
PY
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/matmul_bias_activation/scripts/probe-compiler-pipeline.py" \
 --file "$p/operators/matmul_bias_activation/candidates/profiling-kwide-v5-20261010.py" \
 --output "$r/diagnostics/matmul-compiler-pipeline-20261010" --timeout 1800
"$py" "$p/operators/matmul_bias_activation/scripts/build-compiler-v7.py" \
 "$r/diagnostics/matmul-compiler-pipeline-20261010/artifacts/compiler-pipeline.json"
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator matmul_bias_activation \
 --source "$p/operators/matmul_bias_activation/candidates/profiling-kcompiler-v7-20261010.py" \
 --output "$p/operators/matmul_bias_activation/reports/profiling-kcompiler-v7-20261010" \
 --label profiling-kcompiler-v7 --timeout 1800
echo 'MATMUL_COMPILER_V7_FULL_VALIDATION_PASSED'
"$py" - "$p" <<'PY'
import json,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);reports=p/'operators/matmul_bias_activation/reports'
valid=[]
for directory in reports.iterdir():
    if '20261010' not in directory.name or not directory.name.startswith('profiling-k'):continue
    for f in directory.glob('*.result.json'):
        d=json.loads(f.read_text())
        if d.get('status')=='PASSED' and d.get('num_passed')==d.get('num_workloads'):
            valid.append((d['geo_mean'],f))
best=max(valid)[1]
args=[sys.executable,str(p/'tools/evaluation/compare-operator-versions.py'),
      '--version','master',str(reports/'master-20261010/flaggems-master-1.result.json')]
for arm in ('no-profile','profile'):
    args+=['--version','kg-'+arm,str(reports/('kg-'+arm+'-master-20261010')/'independent'/('kg-'+arm+'-1.result.json'))]
args+=['--version','optimized',str(best),'--output',str(reports/'master-closure-20261010/版本对比.md')]
subprocess.run(args,check=True)
print('MATMUL_FINAL_BEST',str(best),flush=True)
PY
"$py" "$p/tools/evaluation/summarize-master-goal.py" "$p" \
 "$p/operators/低于0.8算子四版本闭环-20261010.md"
