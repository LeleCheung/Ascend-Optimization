#!/usr/bin/env bash
# KGS 单线程执行器也处理合同读取；避免与其他长任务交错而触发 60 秒超时。
set -euo pipefail
r=${1:?usage: resume-grid-campaign-910b.sh EXPERIMENT_ROOT}
p=$r/Ascend-Optimization/project/Ascend-Optimization-v1
c=/data/hanle/ascend-optimization/runtime/kg-controller
sessions=(kg-master-continue-v2-20261010 kg-master-no-profile-recovery-20261010
          kg-matmul-v4-eval-20261010 kg-amin-direct-tiles-20261010)
while true; do
 running=0
 for s in "${sessions[@]}"; do tmux has-session -t "$s" 2>/dev/null && running=1; done
 test "$running" = 0 && break
 sleep 15
done
source /data/yy/kgen/kernelgen/env.sh
export PATH="$c/bin:$c/venv/bin:/usr/local/python3.11.15/bin:$PATH"
export PYTHONPATH="$r/Ascend-Optimization/sources:$r/Ascend-Optimization/sources/kernelgen_server/client:$r/Ascend-Optimization/sources/kernelgen_server"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
for arm in no-profile profile; do
 w=$r/kg-runs/amin-$arm-grid-master-20261010
 evidence=$p/operators/amin/reports/kg-$arm-grid-master-20261010/startup-failure
 mkdir -p "$evidence"
 for f in runner.log run-process.json run-progress.json; do
  test ! -e "$evidence/$f" || { echo '已有故障快照，请检查后再恢复'; exit 2; }
  cp "$w/.kernelgen/$f" "$evidence/$f"
 done
 "$c/venv/bin/kg" resume "$w"
 "$c/venv/bin/python3" "$p/tools/evaluation/wait-kg-worker.py" "$w" \
  || echo "AMIN_RESUME_FAILED_$arm"
done
