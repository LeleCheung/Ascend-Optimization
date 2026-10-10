#!/usr/bin/env bash
# 原生 KG 两组结束后测试分组，最终独立复验等待本会话结束。
set -euo pipefail
r=${1:?usage: run-grouped-v6-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
sessions=(kg-master-no-profile-recovery-20261010 kg-amin-grid-master-20261010 kg-finalize-master-20261010)
while true; do
 running=0
 for s in "${sessions[@]}"; do tmux has-session -t "$s" 2>/dev/null && running=1; done
 test "$running" = 0 && break
 sleep 15
done
"$py" "$p/tools/evaluation/collect-optimized-profiles.py" amin \
 "$p/operators/amin/reports/direct-axis-native-v5-20261010/direct-axis-native-v5-1.result.json" \
 "$p/operators/amin/reports/optimized-profile-20261010" \
 --kernel-prefix amin_native_middle_kernel \
 --case-id 'benchmark/test_amin.py::test_amin[dtype0]::core::float16::4' \
 --case-id 'benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::4' \
 || echo 'AMIN_OPTIMIZED_PROFILE_FAILED'
"$py" "$p/tools/evaluation/debug-operator.py" \
 --script "$p/operators/matmul_bias_activation/scripts/probe-grouped-tiles.py" \
 --file "$p/operators/matmul_bias_activation/scripts/probe-pipeline-tiles.py" \
 --file "$p/operators/matmul_bias_activation/candidates/profiling-kwide-v5-20261010.py" \
 --output "$r/diagnostics/matmul-grouped-tiles-20261010" --timeout 1800
"$py" "$p/operators/matmul_bias_activation/scripts/build-wide-v5.py" \
 "$r/diagnostics/matmul-grouped-tiles-20261010/artifacts/grouped-tiles.json" \
 --parent profiling-kwide-v5-20261010 --name profiling-kgroup-v6-20261010 --expected-groups 6
"$py" "$p/tools/evaluation/evaluate-operator.py" --operator matmul_bias_activation \
 --source "$p/operators/matmul_bias_activation/candidates/profiling-kgroup-v6-20261010.py" \
 --output "$p/operators/matmul_bias_activation/reports/profiling-kgroup-v6-20261010" \
 --label profiling-kgroup-v6 --timeout 1800
# 保留四版本结构并重新择优；较慢的新候选留作证据。
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
PY
# 采集固定 v5，用于与 master 的 metrics 对照；若 v6 更快，它仍单独完整复验。
"$py" "$p/tools/evaluation/collect-optimized-profiles.py" matmul_bias_activation \
 "$p/operators/matmul_bias_activation/reports/profiling-kwide-v5-20261010/profiling-kwide-v5-1.result.json" \
 "$p/operators/matmul_bias_activation/reports/optimized-v5-profile-20261010" \
 --kernel-prefix mba_wide_pipeline_kernel \
 --case-id 'benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::3' \
 --case-id 'benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::3' \
 || echo 'MATMUL_OPTIMIZED_PROFILE_FAILED'
